from common import *
import argparse,ast,copy,contextlib,statistics,types,gc
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,streaming,distributed as parallel,losses,batching
from motion_training.io import identity,sha,copy_state
from model_io import load
import pipeline_accel as fast,lazy_plan
import serial_geometry_graph as geometry

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();a.mode='gate'
parallel.setup();assert parallel.world()==4
OUT=Path(a.output);OUT.mkdir(exist_ok=True,parents=True)
m,c,opt,ck=load(a.checkpoint);raw=m.denoiser.model;cfg=read(ROOT/'configs/batch512.json')
late=lazy_plan.Plan(read(LATE/'MANIFEST_TEMPLATE.json'))
tree=ast.parse((SUITE/'scripts/worker.py').read_text())
funcs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['build','update','delta']]
env=dict(globals());exec(compile(ast.Module(body=funcs,type_ignores=[]),'frozen_worker_helpers','exec'),env)

class LossModule(torch.nn.Module):
    def __init__(self,raw):super().__init__();self.raw=raw;self.cache=geometry.GeometryGraphs(raw,m.hybrid)
    def forward(self,f,reference=False):
        pred=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
        with torch.autocast('cuda',enabled=False):
            if reference:
                values=[fast.components_group(self.raw,batching.split_field(f,i),pred[i:i+1])[0] for i in range(len(pred))]
                return torch.stack(values).mean()
            h=int(f['history_len'][0]);lo=h//4
            assert torch.equal(f['history_len'],f['history_len'][:1].expand_as(f['history_len']))
            expected=torch.zeros_like(f['generation_mask']);expected[:,h:h+52]=True
            assert torch.equal(f['generation_mask'],expected),'Graph requires original fully valid G'
            expected_tokens=expected.reshape(len(pred),-1,4).all(-1)
            assert torch.equal(f['generation_token_mask'],expected_tokens)
            fixed=dict(target=f['target'][:,lo:lo+13],supervised=f['supervised_motion'],
                observed=f['observed'][:,h:h+52],mask=f['motion_mask'][:,h:h+52].bool(),
                prefix=f.get('decoder_history_hybrid'))
            return geometry._Geometry.apply(pred[:,lo:lo+13],self.cache,fixed)[0]

module=DDP(LossModule(raw),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,
    find_unused_parameters=False,gradient_as_bucket_view=True,bucket_cap_mb=25)
env['module']=module;update=env['update'];build=env['build'];delta=env['delta']
# Clone CPU Adam step counters too; load_state_dict can alias them.
ck['optimizer']=copy.deepcopy(ck['optimizer'])
thresholds=dict(loss=1e-5,gradient_relative_l2=1e-4,weight_max=1e-6,Adam_relative_l2=1e-5,clip=1e-4)
checks=[]
for j,hf in enumerate([(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]):
    records=late.records_at(ck['step']-75000+j,512,layout=hf);local=records[parallel.rank()*128:parallel.rank()*128+64]
    f=build(local,cfg,True);runs=[]
    for reference in [True,False]:
        raw.train();m.autoencoder.eval();raw.zero_grad(set_to_none=True)
        torch.manual_seed(local[0]['seed']);torch.cuda.manual_seed_all(local[0]['seed'])
        with torch.autocast('cuda',dtype=torch.bfloat16):v=module(f,reference=reference)
        (v/2).backward();torch.cuda.synchronize()
        grad={n:p.grad.detach().float().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None}
        runs.append((float(v),grad));del v
    gm,ge,gn=delta(runs[0][1],runs[1][1]);row=dict(layout=hf,loss_delta=abs(runs[0][0]-runs[1][0]),gradient_relative_l2=(ge/max(gn,1e-30))**.5,gradient_max=gm)
    row['passed']=row['loss_delta']<=thresholds['loss'] and row['gradient_relative_l2']<=thresholds['gradient_relative_l2']
    checks.append(row);print('LAYOUT',parallel.rank(),json.dumps(row),flush=True)
    del runs,f;opt.zero_grad(set_to_none=True)
allcases=parallel.gather(checks)
if parallel.leader():write(OUT/'LAYOUTS.json',dict(passed=all(r['passed'] for rank in allcases for r in rank),ranks=allcases,thresholds=thresholds,checkpoint=a.checkpoint,production_updates=0))

def restore():
    raw.load_state_dict(ck['model']);opt.load_state_dict(copy.deepcopy(ck['optimizer']));opt.training_schedule=copy.deepcopy(ck['scheduler'])
    runtime.restore_rng(ck['rng_by_rank'][parallel.rank()]);opt.zero_grad(set_to_none=True)

records=late.records_at(ck['step']-75000,512)
restore();print('FULL_REFERENCE',parallel.rank(),flush=True);ref=update(records,cfg,True,True)
state=copy_state(raw.state_dict());ostate=copy_state(opt.state_dict());rrng=runtime.rng()
restore();print('FULL_CANDIDATE',parallel.rank(),flush=True);cand=update(records,cfg,False,True)
gm,ge,gn=delta(ref['gradient'],cand['gradient']);wm,we,wn=delta(state,raw.state_dict());am,ae,an=delta(ostate,opt.state_dict())
row=dict(rank=parallel.rank(),fields_exact=ref['fields']==cand['fields'],loss_delta=abs(ref['loss']-cand['loss']),gradient_relative_l2=(ge/max(gn,1e-30))**.5,gradient_max=gm,
    weight_max=wm,Adam_relative_l2=(ae/max(an,1e-30))**.5,RNG_exact=identity(rrng)==identity(runtime.rng()),clip_delta=abs(ref['clip']-cand['clip']),
    reference_seconds=ref['seconds'],candidate_seconds=cand['seconds'])
moment_checks={}
for key in ['exp_avg','exp_avg_sq']:
    x={str(k):v[key] for k,v in ostate['state'].items() if key in v}
    y={str(k):v[key] for k,v in opt.state_dict()['state'].items() if key in v}
    _,e,n=delta(x,y);moment_checks[key]=(e/max(n,1e-30))**.5
row['moments_independently']=moment_checks
row['passed']=row['fields_exact'] and row['loss_delta']<=1e-5 and row['gradient_relative_l2']<=1e-4 and wm<=1e-6 and row['Adam_relative_l2']<=1e-5 and row['RNG_exact'] and row['clip_delta']<=1e-4 and all(x<=1e-5 for x in moment_checks.values())
parts=parallel.gather(row)
if parallel.leader():write(OUT/'FULL_UPDATE.json',dict(passed=all(x['passed'] for x in parts),ranks=parts,thresholds=thresholds,checkpoint=a.checkpoint,production_updates=0))
passed=all(x['passed'] for x in parts) and all(r['passed'] for rank in allcases for r in rank)
del ref,cand,state,ostate;gc.collect();opt.zero_grad(set_to_none=True)
if passed:
    # Same complete512 updates, no tensor hashing or gradient CPU copies in timing.
    restore();times=[]
    for j in range(12):
        a.mode='probe'
        rec=update(late.records_at(ck['step']-75000+j,512),cfg,False)
        times.append(rec['seconds']);print('TIMING',parallel.rank(),j,rec['seconds'],flush=True)
    hashes=parallel.gather(identity([raw.state_dict(),opt.state_dict()]));assert len(set(hashes))==1
    gathered=parallel.gather(times)
    if parallel.leader():write(OUT/'TIMING.json',dict(passed=True,seconds_by_rank=gathered,warm_median=max(statistics.median(x[6:]) for x in gathered),warm_mean=max(statistics.mean(x[6:]) for x in gathered),discarded_updates=14,rank_model_Adam_equal=True))
if parallel.leader():write(OUT/'RESULT.json',dict(passed=passed,backend='serial_B1_graph_chunk4',checkpoint=a.checkpoint,time=time.time(),no_recipe_change=True,no_production_updates=True))
parallel.barrier();torch.distributed.destroy_process_group()
