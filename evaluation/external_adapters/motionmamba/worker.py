from common import *
import argparse,contextlib,statistics,copy,fcntl,shutil,hashlib
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,streaming,distributed as parallel,losses,batching
from motion_training.io import identity,sha,append,copy_state
from model_io import load,save
import pipeline_accel as fast,lazy_plan
import serial_geometry_graph as geometry

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['gate','train'],required=True);p.add_argument('--resume');a=p.parse_args()
parallel.setup();assert parallel.world()==4
m,c,opt,ck=load(a.resume);raw=m.denoiser.model
base=read(ROOT/'configs/batch32.json');latecfg=read(ROOT/'configs/batch512.json')
early=streaming.load(OLD/'train/data/MANIFEST.json')['plan'];late=lazy_plan.Plan(read(LATE/'MANIFEST_TEMPLATE.json'))

class ReferenceLossModule(torch.nn.Module):
    def __init__(self,raw):super().__init__();self.raw=raw
    def forward(self,f,reference=False):
        pred=self.raw(*(f[k] for k in losses.FORWARD_KEYS))
        with torch.autocast('cuda',enabled=False):
            # Batched geometry failed the frozen gradient/Adam gate. Keep the
            # original individual geometry for both paths; only frozen codec
            # graph and CPU preparation are accelerated. No tolerance change.
            vals=[]
            for i in range(len(pred)):
                value,_,_=fast.components_group(self.raw,batching.split_field(f,i),pred[i:i+1]);vals.append(value)
            value=torch.stack(vals).mean()
        return value

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

if a.mode=='gate':
    objective_class=LossModule
else:
    ga=read(ROOT/'gates/GEOMETRY_GRAPH_ADMISSION.json')
    assert ga['passed'] and ga['method']==read(ROOT/'PLAN.json')['method']
    for name,digest in ga['source_hashes'].items():assert sha(SUITE/name)==digest
    objective_class=LossModule

module=DDP(objective_class(raw),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,
    find_unused_parameters=False,gradient_as_bucket_view=True,bucket_cap_mb=25)
def build(records,cfg,graph=True):
    pending=fast.submit_reads(c,records,cfg,2)
    if graph:return fast.build_group(m,c,fast.materialize(pending),cfg,False,serial_codec=True)[0]
    # Reference keeps each individual codec GEMM, exactly as original Native.
    return batching.collate([batching.corrected_build(m,c,r,cfg,False)[0] for r in records])

def update(records,cfg,reference=False,retain_grad=False):
    t0=time.perf_counter();raw.train();m.autoencoder.eval();opt.zero_grad(set_to_none=True)
    lr=runtime.set_update_lr(opt);width=len(records)//4;micro=cfg['per_rank_microbatch'];n=width//micro
    assert len(records)==cfg['batch_size'] and width%micro==0
    local=records[parallel.rank()*width:(parallel.rank()+1)*width]
    pending=fast.submit_reads(c,local,cfg,2);values=[];field_hashes=[]
    for j in range(n):
        subset=local[j*micro:(j+1)*micro]
        if reference:f=build(subset,cfg,False)
        else:f=fast.build_group(m,c,fast.materialize(pending[j*micro:(j+1)*micro]),cfg,False,serial_codec=True)[0]
        if a.mode=='gate':field_hashes.append(identity([f[k] for k in losses.FORWARD_KEYS]+[f['target'],f['supervised_motion'],f.get('decoder_history_hybrid')]))
        torch.manual_seed(subset[0]['seed']);torch.cuda.manual_seed_all(subset[0]['seed'])
        with module.no_sync() if j<n-1 else contextlib.nullcontext():
            with torch.autocast('cuda',dtype=torch.bfloat16):value=module(f,reference=reference)
            assert torch.isfinite(value),'nonfinite loss'
            (value/n).backward()
        values.append(float(value.detach()));del f,value
    clip=torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True)
    grad={k:v.grad.detach().float().cpu().clone() for k,v in raw.named_parameters() if v.grad is not None} if retain_grad else None
    opt.step();opt.training_schedule['completed']+=1;torch.cuda.synchronize()
    return dict(loss=float(np.mean(values)),clip=float(clip),seconds=time.perf_counter()-t0,lr=lr,
                fields=field_hashes,gradient=grad)

def delta(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();pairs=[delta(a[k],b[k]) for k in a];return max([x[0] for x in pairs],default=0.),sum(x[1] for x in pairs),sum(x[2] for x in pairs)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b);pairs=[delta(x,y) for x,y in zip(a,b)];return max([x[0] for x in pairs],default=0.),sum(x[1] for x in pairs),sum(x[2] for x in pairs)
    if torch.is_tensor(a):
        x=a.detach().double().cpu();y=b.detach().double().cpu();e=x-y
        return float(e.abs().max()) if e.numel() else 0.,float((e*e).sum()),float((x*x).sum())
    assert a==b,(a,b);return 0.,0.,0.

if a.mode=='gate':
    write(ROOT/f'gates/rank{parallel.rank()}_INIT.json',dict(method=raw.method,parameters=sum(p.numel() for p in raw.parameters()),source=raw.source,source_sha=raw.source_sha256,source_edits=raw.source_edits))
    warm_checks=8
    for j in range(warm_checks):
        print('DISCARDED_MATURE_GATE_WARMUP',j,flush=True)
        update(early[j*32:(j+1)*32],base,reference=True)
    gates=[]
    for batch,cfg in [(32,base),(512,latecfg)]:
        records=early[0:32] if batch==32 else late.records_at(0,512)
        state=copy_state(raw.state_dict());ostate=copy_state(opt.state_dict());schedule=copy.deepcopy(opt.training_schedule);rng=runtime.rng()
        print('GATE_REFERENCE',batch,flush=True);ref=update(records,cfg,True,True)
        refstate=copy_state(raw.state_dict());refopt=copy_state(opt.state_dict());refrng=runtime.rng()
        raw.load_state_dict(state);opt.load_state_dict(ostate);opt.training_schedule=copy.deepcopy(schedule);runtime.restore_rng(rng)
        print('GATE_CANDIDATE',batch,flush=True);cand=update(records,cfg,False,True)
        gm,ge,gn=delta(ref['gradient'],cand['gradient']);wm,we,wn=delta(refstate,raw.state_dict());am,ae,an=delta(refopt,opt.state_dict())
        row=dict(batch=batch,fields_exact=ref['fields']==cand['fields'],loss_delta=abs(ref['loss']-cand['loss']),
            gradient_relative_l2=(ge/max(gn,1e-30))**.5,gradient_max=gm,weight_max=wm,Adam_relative_l2=(ae/max(an,1e-30))**.5,
            RNG_exact=identity(refrng)==identity(runtime.rng()),clip_delta=abs(ref['clip']-cand['clip']),
            reference_seconds=ref['seconds'],candidate_seconds=cand['seconds'],
            fixed_tolerances=dict(loss=1e-5,gradient_relative_l2=1e-4,weight_max=1e-6,Adam_relative_l2=1e-5,clip=1e-4))
        row['passed']=row['fields_exact'] and row['loss_delta']<=1e-5 and row['gradient_relative_l2']<=1e-4 and wm<=1e-6 and row['Adam_relative_l2']<=1e-5 and row['RNG_exact'] and row['clip_delta']<=1e-4
        gates.append(row);print('EQUIVALENCE',json.dumps(row),flush=True)
        raw.load_state_dict(state);opt.load_state_dict(ostate);opt.training_schedule=copy.deepcopy(schedule);runtime.restore_rng(rng)
        del ref,cand,state,ostate,refstate,refopt;opt.zero_grad(set_to_none=True)
    results=parallel.gather(gates)
    if parallel.leader():write(ROOT/'gates/NUMERICS.json',dict(passed=all(r['passed'] for g in results for r in g),ranks=results,discarded_updates=12,reference='individual codec and geometry; same batched official core, AMP/dropout/DDP/noise'))
    assert all(r['passed'] for g in results for r in g),'Acceleration equivalence failed; do not launch training'
    from interface_gate import run as check_interface
    check_interface(m,raw,opt)
    capacity=[]
    for batch,cfg in [(32,base),(512,latecfg)]:
        for hf in [(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]:
            print('CAPACITY',batch,hf,flush=True)
            width=batch//4;recs=late.records_at(11,batch,layout=hf)[parallel.rank()*width:(parallel.rank()+1)*width][:cfg['per_rank_microbatch']]
            f=build(recs,cfg);raw.train();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):val=module(f)
            val.backward();assert torch.isfinite(val) and all(p.grad is None or torch.isfinite(p.grad).all() for p in raw.parameters())
            row=dict(batch=batch,layout=hf,peak_GiB=torch.cuda.max_memory_reserved()/2**30)
            if batch in [32,512]:
                comparisons=[]
                for is_reference in [True,False]:
                    opt.zero_grad(set_to_none=True)
                    torch.manual_seed(recs[0]['seed']);torch.cuda.manual_seed_all(recs[0]['seed'])
                    with torch.autocast('cuda',dtype=torch.bfloat16):vv=module(f,reference=is_reference)
                    vv.backward()
                    comparisons.append((float(vv),copy_state({n:p.grad for n,p in raw.named_parameters() if p.grad is not None})))
                _,ge,gn=delta(comparisons[0][1],comparisons[1][1])
                row.update(loss_delta=abs(comparisons[0][0]-comparisons[1][0]),gradient_relative_l2=(ge/max(gn,1e-30))**.5)
                assert row['loss_delta']<=1e-5 and row['gradient_relative_l2']<=1e-4,row
                del comparisons
            capacity.append(row);del f,val;opt.zero_grad(set_to_none=True)
    a.mode='timing' # Do not include CPU input hashing in throughput timing.
    times=[]
    for batch,cfg in [(32,base),(512,latecfg)]:
        durations=[]
        for i in range(5):
            print('TIMING',batch,i,flush=True)
            r=update(early[i*32:(i+1)*32] if batch==32 else late.records_at(i,512),cfg)
            durations.append(r['seconds'])
        times.append(dict(batch=batch,seconds=durations,warm_median_seconds=statistics.median(durations[1:]),worst_seconds=max(durations),peak_GiB=torch.cuda.max_memory_reserved()/2**30))
    allcap=parallel.gather(capacity);alltimes=parallel.gather(times)
    # Explicit negative guard: a changed G validity mask must not reach cached geometry.
    rbad=late.records_at(88,512,layout=(0,0));width=128
    fbad=build(rbad[parallel.rank()*width:parallel.rank()*width+64],latecfg,True)
    fbad['generation_mask']=fbad['generation_mask'].clone();fbad['generation_mask'][:,0]=False
    rejected=False
    try:
        with torch.autocast('cuda',dtype=torch.bfloat16):module(fbad)
    except AssertionError:rejected=True
    assert rejected,'Invalid generation mask was not rejected'
    del fbad;opt.zero_grad(set_to_none=True)
    # Reload the exact initial constructor state; no check update is production.
    if parallel.leader():write(ROOT/'gates/ADMISSION.json',dict(passed=True,world_size=4,global_batches=[32,512],capacity=allcap,timings=alltimes,discarded_updates_per_rank=22,full_100k_recipe=True,production_started=False))
    if parallel.leader():
        paths=['scripts/worker.py','scripts/external_model.py','scripts/serial_geometry_graph.py','scripts/graph_constants.py','scripts/pipeline_accel.py','scripts/codec_graph.py','scripts/model_io.py','scripts/interface_gate.py']
        write(ROOT/'gates/GEOMETRY_GRAPH_ADMISSION.json',dict(passed=True,method=raw.method,full_update_gate=str(ROOT/'gates/NUMERICS.json'),all_six_layouts_verified=True,invalid_G_mask_rejected=True,first_order_gradient_only=True,source_hashes={n:sha(SUITE/n) for n in paths},thresholds_unchanged=True,time=time.time()))
    parallel.barrier();torch.distributed.destroy_process_group();raise SystemExit

admission=read(ROOT/'gates/ADMISSION.json');assert admission['passed']
assert read(ROOT/'gates/PROTOCOL.json')['passed']
step=ck['step'];started=read(ROOT/'STARTED_TRAINING.json')['time'] if (ROOT/'STARTED_TRAINING.json').exists() else time.time();wall_limit=read(ROOT/'PLAN.json')['wall_budget_hours']*3600
if not a.resume:
    save(m,opt,0,ROOT/'initial/checkpoint0.pt')
    write(ROOT/f'gates/rank{parallel.rank()}_FRESH.json',dict(step=0,optimizer_states=len(opt.state),parameters=sum(p.numel() for p in raw.parameters())))
history={32:[],512:[]}
if parallel.leader():write(ROOT/'STATUS.json',dict(stage='REAL_TRAINING',step=step,time=time.time(),method=raw.method))
try:
    for step in range(step+1,100001):
        batch=32 if step<=75000 else 512;cfg=base if batch==32 else latecfg
        recs=early[(step-1)*32:step*32] if batch==32 else late.records_at(step-75001,512)
        r=update(recs,cfg);history[batch].append(r['seconds'])
        summaries=parallel.gather({k:v for k,v in r.items() if k not in ['fields','gradient']})
        if parallel.leader():
            loss=float(np.mean([v['loss'] for v in summaries]));sec=max(v['seconds'] for v in summaries)
            mean=float(np.mean(history[batch][-100:]));probe=float(np.median([x[1]['warm_median_seconds'] for x in admission['timings']]))
            remaining=(max(75000-step,0)*mean+25000*probe) if batch==32 else (100000-step)*mean
            live=dict(step=step,batch=batch,world_size=4,microbatch=cfg['per_rank_microbatch'],accumulation=batch//4//cfg['per_rank_microbatch'],
                loss=loss,seconds=sec,recent_mean_seconds=mean,remaining_training_hours=remaining/3600,time=time.time(),lr=r['lr'],
                peak_GiB=torch.cuda.max_memory_reserved()/2**30)
            write(ROOT/'LIVE.json',live);append(ROOT/'train/train.jsonl',live)
            if step<=10 or step%100==0:print('UPDATE',step,loss,sec,flush=True)
        if step==ck['step']+100 or step in [10,100,1000,75000,100000] or step%10000==0:save(m,opt,step,ROOT/f'train/checkpoint{step}.pt')
        if step==10 and parallel.leader():write(ROOT/'STARTED_TRAINING.json',dict(step=10,checkpoint=str(ROOT/'train/checkpoint10.pt'),actual_updates=10,global_batch=32,world_size=4,Adam_states=len(opt.state),checks_not_counted_as_training=True,time=time.time()))
        if step%100==0:
            stop=(ROOT/'BOUNDARY_REQUEST.json').exists() or time.time()-started>wall_limit or shutil.disk_usage(ROOT).free<25*2**30
            if any(parallel.gather(stop)):
                save(m,opt,step,ROOT/f'train/checkpoint{step}_boundary.pt')
                if parallel.leader():write(ROOT/'STATUS.json',dict(stage='SAVED_RESOURCE_BOUNDARY',step=step,time=time.time()))
                break
    else:
        if parallel.leader():write(ROOT/'STATUS.json',dict(stage='COMPLETE_TRAINING100K_PENDING_EVALUATION',step=100000,time=time.time()))
except BaseException as e:
    if parallel.leader():write(ROOT/'FAILURE.json',dict(type=type(e).__name__,error=str(e),step=step,time=time.time(),no_automatic_recipe_change=True))
    raise
finally:
    if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
