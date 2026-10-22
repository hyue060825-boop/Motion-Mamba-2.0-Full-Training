from common import *
import ast,statistics
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from motion_training import runtime,history,distributed as parallel
from motion_training.acceleration import configure
import history_v3 as candidate
tree=ast.parse((ROOT/'train.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.ClassDef))]
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(ROOT/'train.py'),'exec'),globals())
parallel.setup();assert parallel.world()==8
cfg=read(ROOT/'config.json');pool=read(ROOT/'pools/g1/POOL.json')
checkpoint=ROOT/'train/checkpoint3490.pt';dest=ROOT/'acceleration_v3'
assert read(dest/'SINGLE_GATE.json')['passed'] and read(dest/'SINGLE_GATE.json')['complete']
m,c,ck=runtime.load(checkpoint,True);raw=m.denoiser.model;raw.train();m.autoencoder.eval()
opt=runtime.optimizer(m,ck,cfg,resume=True)
before=copy_state(raw.state_dict());before_opt=copy_state(opt.state_dict());before_schedule=copy_state(opt.training_schedule);rng=runtime.rng()
module=DDP(MixedLoss(raw,c),device_ids=[torch.cuda.current_device()],broadcast_buffers=False,find_unused_parameters=True,gradient_as_bucket_view=True)
plan=lazy_plan.load(ROOT/'train/data/MANIFEST.json')['plan']
def update(mode,step,record_hash):
 candidate.install(mode,audit=False);configure(c,'reference');opt.zero_grad(set_to_none=True)
 runtime.set_update_lr(opt);torch.cuda.reset_peak_memory_stats();parallel.barrier();torch.cuda.synchronize();tick=time.perf_counter()
 records=plan.records_at(step-1,448);width=56;lane=parallel.rank();subset=records[lane*width:(lane+1)*width]
 prepared=fast.materialize(fast.submit_reads(c,subset,cfg,4));f,_=fast.build_group(m,c,prepared,cfg,audit=False,serial_codec=True)
 torch.manual_seed(subset[0]['seed']);torch.cuda.manual_seed_all(subset[0]['seed'])
 with module.no_sync():
  with torch.autocast('cuda',dtype=torch.bfloat16):v,_=module(f)
  (v*(8*width/512)).backward()
 values=[float(v)*width];del v,f,prepared
 torch.cuda.synchronize();gt=time.perf_counter()-tick;prime_time=0.;current_time=0.
 rs=choose(pool,step,lane,8,cfg)
 for j,r in enumerate(rs):
  start=time.perf_counter();f,_=history.teacher_batch(m,c,r,cfg,ck['config']);torch.cuda.synchronize();prime_time+=time.perf_counter()-start
  torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed']);start=time.perf_counter()
  with module.no_sync() if j<7 else contextlib.nullcontext():
   with torch.autocast('cuda',dtype=torch.bfloat16):v,_=module(f)
   (v*(8/512)).backward()
  values.append(float(v));del v,f;torch.cuda.synchronize();current_time+=time.perf_counter()-start
 # Capture reduced gradients before clipping; keep diagnostic copies outside timing.
 elapsed_compute=time.perf_counter()-tick
 if record_hash:grad_hash=identity({n:p.grad for n,p in raw.named_parameters() if p.grad is not None})
 else:grad_hash=None
 start=time.perf_counter();clip=torch.nn.utils.clip_grad_norm_(raw.parameters(),cfg['clip'],error_if_nonfinite=True)
 opt.step();opt.training_schedule['completed']+=1;torch.cuda.synchronize();optimizer=time.perf_counter()-start
 seconds=elapsed_compute+optimizer
 row=dict(rank=lane,mode=mode,step=step,seconds=seconds,GT_seconds=gt,history_seconds=prime_time,teacher_gradient_and_DDP_seconds=current_time,clip_and_Adam_seconds=optimizer,loss=sum(values)/64,clip=float(clip),peak_GiB=torch.cuda.max_memory_allocated()/2**30,depths=[r['prime_chunks'] for r in rs],grads=grad_hash)
 if record_hash:row.update(weights=identity(raw.state_dict()),Adam=identity(opt.state_dict()),rng=identity(runtime.rng()))
 return row
def restore():
 raw.load_state_dict(before)
 # PyTorch keeps CPU step-counter tensors by reference. A fresh deep tensor
 # copy prevents a discarded update from mutating the baseline snapshot.
 opt.load_state_dict(copy_state(before_opt));opt.training_schedule=copy_state(before_schedule);runtime.restore_rng(rng)
 assert identity(opt.state_dict())==identity(before_opt)
rows=[];checks=[]
# Warm caches, libraries and DDP buckets before paired production-sized updates.
update('dense',3491,False);restore();update('canonical',3491,False);restore()
for rep,step in enumerate((3491,3492,3493)):
 pair={}
 for mode in (('dense','canonical') if rep%2==0 else ('canonical','dense')):
  restore();row=update(mode,step,True);pair[mode]=row
  allrows=parallel.gather(row)
  if parallel.leader():
   rows.extend(allrows);print('V3_FULL_UPDATE',step,mode,max(r['seconds'] for r in allrows),flush=True)
 check=dict(rank=parallel.rank(),step=step,gradients_exact=pair['dense']['grads']==pair['canonical']['grads'],weights_exact=pair['dense']['weights']==pair['canonical']['weights'],Adam_exact=pair['dense']['Adam']==pair['canonical']['Adam'],rng_exact=pair['dense']['rng']==pair['canonical']['rng'],loss_exact=pair['dense']['loss']==pair['canonical']['loss'],clip_exact=pair['dense']['clip']==pair['canonical']['clip'])
 check['passed']=all(v for k,v in check.items() if k not in ('rank','step'))
 checks.extend(parallel.gather(check))
 assert all(x['passed'] for x in parallel.gather(check)),check
 if parallel.leader():write(dest/'EIGHT_RANK_GATE.json',dict(passed=all(x['passed'] for x in checks),complete=False,checks=checks,rows=rows))
if parallel.leader():
 pairs=[max(r['seconds'] for r in rows if r['step']==s and r['mode']=='dense')/max(r['seconds'] for r in rows if r['step']==s and r['mode']=='canonical') for s in (3491,3492,3493)]
 dense=[max(r['seconds'] for r in rows if r['step']==s and r['mode']=='dense') for s in (3491,3492,3493)]
 canonical=[max(r['seconds'] for r in rows if r['step']==s and r['mode']=='canonical') for s in (3491,3492,3493)]
 write(dest/'EIGHT_RANK_GATE.json',dict(passed=True,complete=True,checks=checks,rows=rows,dense_median_seconds=statistics.median(dense),canonical_median_seconds=statistics.median(canonical),paired_median_speedup=statistics.median(pairs),all_timing_pairs_speedup=pairs,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_sha256=sha(dest/'history_v3.py'),discarded_updates=8,scope='Same eight-rank DDP global512, GT448+teacher64, exact weights/Adam restored before each candidate; synchronized timing excludes diagnostic hashing, includes data/history/gradient/sync/clip/Adam; no training checkpoint written'))
 budget=__import__('motion_training.io',fromlist=['Budget']).Budget(cfg['budget_root'])
 for i in range(8):batching.fast_ticket(budget,'checks',discarded_optimizer_probe='v3_eight_rank_'+str(i),checkpoint_written=False)
 print('V3_FULL_PASS',statistics.median(dense),statistics.median(canonical),statistics.median(pairs),flush=True)
parallel.barrier();parallel.cleanup() if hasattr(parallel,'cleanup') else None
