from common import *
import ast
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,history,distributed as parallel
from motion_training.acceleration import configure
import history_v3 as candidate
for path in (ROOT/'train.py',ROOT/'acceleration_v3/v3_full_gate.py'):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())
parallel.setup();assert parallel.world()==8
cfg=read(ROOT/'config.json');pool=read(ROOT/'pools/g1/POOL.json');dest=ROOT/'acceleration_v3'
single=read(dest/'SINGLE_GATE.json');assert single['complete'] and single['passed'] and single['source_sha256']==sha(dest/'history_v3.py')
checkpoint=Path(read(ROOT/'LATEST.json')['checkpoint']);probe_step=read(ROOT/'LATEST.json')['stage_step']+2
m,c,ck=runtime.load(checkpoint,True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
opt=runtime.optimizer(m,ck,cfg,resume=True);before=copy_state(raw.state_dict());before_opt=copy_state(opt.state_dict());before_schedule=copy_state(opt.training_schedule);rng=runtime.rng()
module=DDP(MixedLoss(raw,c),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
plan=lazy_plan.load(ROOT/'train/data/MANIFEST.json')['plan']
for mode in ('dense','canonical'):restore();update(mode,probe_step,False)
rows=[];pair={}
for mode in ('canonical','dense'):
 restore();r=update(mode,probe_step,True);pair[mode]=r;allrows=parallel.gather(r)
 if parallel.leader():rows.extend(allrows);print('V3_FINAL_UPDATE',mode,max(x['seconds'] for x in allrows),flush=True)
check=dict(rank=parallel.rank(),gradients_exact=pair['dense']['grads']==pair['canonical']['grads'],weights_exact=pair['dense']['weights']==pair['canonical']['weights'],Adam_exact=pair['dense']['Adam']==pair['canonical']['Adam'],rng_exact=pair['dense']['rng']==pair['canonical']['rng'],loss_exact=pair['dense']['loss']==pair['canonical']['loss'],clip_exact=pair['dense']['clip']==pair['canonical']['clip'])
check['passed']=all(v for k,v in check.items() if k!='rank');checks=parallel.gather(check)
if parallel.leader():
 a=max(r['seconds'] for r in rows if r['mode']=='dense');b=max(r['seconds'] for r in rows if r['mode']=='canonical')
 result=dict(passed=all(r['passed'] for r in checks),complete=True,checks=checks,rows=rows,source_sha256=sha(dest/'history_v3.py'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),discarded_updates=4,scope=f'Eight-rank exact gradients, full weights and Adam; GT448 + teacher64 actual step{probe_step}. Both paths separately warmed on same batch. One matched timing pair after warm-up; production validation required.')
 write(dest/'EIGHT_RANK_GATE.json',result)
 write(dest/'WARM_TIMING.json',dict(rows=rows,dense_median_seconds=a,canonical_median_seconds=b,paired_median_speedup=a/b,pairs=[a/b],scope=result['scope'],discarded_updates=4))
 budget=__import__('motion_training.io',fromlist=['Budget']).Budget(cfg['budget_root'])
 for i in range(4):batching.fast_ticket(budget,'checks',discarded_optimizer_probe='v3_final_without_CPU_fingerprint_'+str(i),checkpoint_written=False)
 print('V3_FINAL_RESULT',result['passed'],a,b,a/b,flush=True)
assert all(r['passed'] for r in checks)
parallel.barrier()
