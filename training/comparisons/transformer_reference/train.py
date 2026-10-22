from common import *
import argparse,time,statistics,shutil,sys
import torch
import adapter,pipeline_accel as fast,lazy_plan,accelerated_loss
from motion_training import runtime,engine,streaming,distributed as parallel
from motion_training.io import Budget,append,identity,sha,rows
adapter.install();parallel.setup()
p=argparse.ArgumentParser();p.add_argument('--resume');p.add_argument('--until',type=int,default=100000);p.add_argument('--capacity-only',action='store_true');p.add_argument('--topology-probe',action='store_true');a=p.parse_args()
assert parallel.world() in ([2,4] if a.topology_probe else [4])
cfg=read(ROOT/'configs/batch32.json');cp=a.resume or str(ROOT/'initial/checkpoint0.pt')
m,c,ck=runtime.load(cp,True);start=ck['step'];opt=runtime.optimizer(m,ck,cfg,resume=True)
runtime.restore_rng(ck['rng_by_rank'][parallel.rank()%len(ck['rng_by_rank'])])
assert opt.training_schedule['completed']==start
assert identity(opt.state_dict())==identity(ck['optimizer'])
expected=ck['framework'].get('rank_state_hashes')
if expected:assert identity([m.denoiser.model.state_dict(),opt.state_dict()])==expected[parallel.rank()%len(expected)]
budget=Budget(ROOT/'budget');engine.update=__import__("native_geometry_accel").install()
assert read(ROOT/'gates/ACCELERATION.json')['passed']
index=streaming.load(OLD/'train/data/MANIFEST.json')['plan'];late=lazy_plan.Plan(read(LATE/'MANIFEST_TEMPLATE.json'))
architecture=read(ROOT/'PLAN.json')['architecture']
if a.topology_probe:
 cfg=dict(cfg,audit_every=1)
 out=ROOT/'gates'/('two' if parallel.world()==2 else 'four');out.mkdir(exist_ok=True)
 if parallel.leader():write(out/'RESTORED.json',dict(model=identity(m.denoiser.model.state_dict()),optimizer=identity(opt.state_dict()),scheduler=opt.training_schedule,sampler_cursor=start*32,parent_sha256=sha(cp)))
 rec=engine.update(m,c,ck,opt,cfg,index[start*32:(start+1)*32],start+1,'checks',budget)
 if parallel.leader():
  write(out/'UPDATE.json',rec);torch.save({n:p.grad.detach().float().cpu().clone() for n,p in m.denoiser.model.named_parameters() if p.grad is not None},out/'gradients.pt')
 ck['_framework_global_start']=0;engine.save(out/'checkpoint.pt',m,ck,opt,cfg,start+1,OLD/'train/data/MANIFEST.json',None)
 parallel.barrier();torch.distributed.destroy_process_group();sys.exit(0)
if not (ROOT/'gates/CAPACITY64.json').exists():
 cap=[];capacity_cfg=read(ROOT/'configs/batch512.json')
 for layout in [(0,0),(52,0),(196,0),(0,196),(52,144),(144,52)]:
  rs=late.records_at(111+parallel.rank(),64,layout=layout);cpu=fast.materialize(fast.submit_reads(c,rs,capacity_cfg,4));f,_=fast.build_group(m,c,cpu,capacity_cfg,False,serial_codec=True)
  opt.zero_grad(set_to_none=True);m.denoiser.model.train()
  with torch.autocast('cuda',dtype=torch.bfloat16):value,_=accelerated_loss.LossModule(m.denoiser.model,c)(f)
  value.backward();torch.cuda.synchronize()
  assert torch.isfinite(value) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.denoiser.model.parameters())
  peak=torch.cuda.max_memory_reserved()/2**30;limit=torch.cuda.get_device_properties(0).total_memory/2**30*.9
  assert peak<limit,(layout,peak,limit)
  cap.append(dict(layout=layout,reserved_GiB=peak,limit_GiB=limit));opt.zero_grad(set_to_none=True);del value,f,cpu,rs
 cap=parallel.gather(cap)
 if parallel.leader():write(ROOT/'gates/CAPACITY64.json',dict(passed=True,world_size=4,per_rank_microbatch=64,accumulation=2,global_batch=512,ranks=cap,optimizer_updates=0))
 if hasattr(c,'_frozen_codec_graphs'):del c._frozen_codec_graphs
 torch.cuda.empty_cache();parallel.barrier()
if a.capacity_only:
 measured=[]
 with parallel.timed(budget,'discarded_four_rank_full_updates'):
  for b in [32,512]:
   conf=read(ROOT/f'configs/batch{b}.json');timings=[]
   for i in range(4):
    records=index[(start+i)*32:(start+i+1)*32] if b==32 else late.records_at(10+i,512)
    rec=engine.update(m,c,ck,opt,conf,records,11+i,'checks',budget);timings.append(rec['seconds'])
   hashes=parallel.gather(identity([m.denoiser.model.state_dict(),opt.state_dict()]))
   assert len(set(hashes))==1,'Ranks diverged'
   measured.append(dict(batch=b,warm_seconds=timings[1:],median=statistics.median(timings[1:]),rank_states_exact=True,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
 if parallel.leader():write(ROOT/'gates/FULL_UPDATE.json',dict(passed=True,measurements=measured,discarded_optimizer_updates=8,source_checkpoint=cp,actual_training_checkpoint_unchanged=True))
 parallel.barrier();torch.distributed.destroy_process_group();sys.exit(0)
assert read(ROOT/'gates/FULL_UPDATE.json')['passed'] and read(ROOT/'gates/TOPOLOGY.json')['passed']
if parallel.leader():write(ROOT/'START_VERIFICATION.json',dict(step=start,resume=cp,parent_sha256=sha(cp),model_and_Adam_restored_exact=True,scheduler=opt.training_schedule,world_size=4,global_batch=32 if start<75000 else 512,parent_rng_mapping='saved rank modulo saved world; per-record noise unchanged',sampler_cursor=start*32 if start<75000 else (start-75000)*512,architecture=architecture))
history={32:[],512:[]};last_saved=start;started=time.time()
def save(step):
 global last_saved
 batch=32 if step<=75000 else 512;conf=read(ROOT/f'configs/batch{batch}.json');offset=0 if batch==32 else 75000;ck['_framework_global_start']=offset
 manifest=OLD/'train/data/MANIFEST.json' if batch==32 else LATE/'MANIFEST_TEMPLATE.json'
 path=ROOT/'train'/f'checkpoint{step}.pt';storage=read(ROOT/'STORAGE_PLAN.json')
 actual=Path(storage['overflow_root'])/path.name if storage.get('overflow_root') and step>=75000 else path
 engine.save(actual,m,ck,opt,conf,step-offset,manifest,None);last_saved=step
 if parallel.leader() and actual!=path:path.symlink_to(actual)
 if parallel.leader():write(ROOT/'LATEST.json',dict(step=step,checkpoint=str(ROOT/'train'/f'checkpoint{step}.pt'),exposures=min(step,75000)*32+max(step-75000,0)*512,architecture=architecture))
 parallel.barrier()
with parallel.timed(budget,'matched_four_rank_training'):
 for step in range(start+1,a.until+1):
  phase=32 if step<=75000 else 512;cfg=read(ROOT/f'configs/batch{phase}.json');relative=step if phase==32 else step-75000
  records=index[(step-1)*32:step*32] if phase==32 else late.records_at(relative-1,512)
  rec=engine.update(m,c,ck,opt,cfg,records,relative,'B',budget);rec.update(global_step=step,architecture=architecture);history[phase].append(rec['seconds'])
  if parallel.leader():
   append(ROOT/'train/train.jsonl',rec);avg=statistics.mean(history[phase][-100:])
   future=read(ROOT/'gates/FULL_UPDATE.json')['measurements'][1]['median'];remaining=max(75000-step,0)*avg+25000*future if phase==32 else (a.until-step)*avg
   write(ROOT/'LIVE.json',dict(step=step,batch=phase,world_size=4,per_rank_microbatch=cfg['per_rank_microbatch'],accumulation=phase//4//cfg['per_rank_microbatch'],seconds=rec['seconds'],recent_mean_seconds=avg,remaining_training_hours=remaining/3600,batch512_estimate_is_discarded_probe=True,time=time.time(),loss=rec['loss'],lr=rec['lr'],scheduler_step=rec['scheduler_step'],peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
   print('UPDATE',step,rec['loss'],rec['seconds'],flush=True)
  if step in cfg.get('savepoints',[75000,100000]) or step%cfg.get('save_every',10000)==0 or step==a.until:save(step)
  stop=False
  if step%100==0:
   requests=parallel.gather((ROOT/'BOUNDARY_REQUEST.json').exists())
   if parallel.leader():
    used=sum(x['seconds']*x.get('devices',1) for x in rows(ROOT/'budget/GPU_TIME.jsonl'));active=read(ROOT/'budget/ACTIVE.json') if (ROOT/'budget/ACTIVE.json').exists() else {};used+=sum((time.time()-x['start'])*x.get('devices',1) for x in active.values())
    storage=read(ROOT/'STORAGE_PLAN.json');disk_low=shutil.disk_usage(ROOT).free<25*2**30 or bool(storage.get('overflow_root') and shutil.disk_usage(storage['overflow_root']).free<25*2**30)
    stop=any(requests) or used>480*3600*.95 or time.time()-started>96*3600 or disk_low
   stop=any(parallel.gather(stop))
  if stop:
   if last_saved!=step:save(step)
   if parallel.leader():write(ROOT/'STATUS.json',dict(stage='SAVED_AT_COMPLETE_BOUNDARY',step=step,time=time.time(),quality_stop=False))
   break
 else:
  if parallel.leader():write(ROOT/'STATUS.json',dict(stage='TRAINING_COMPLETE',step=a.until,time=time.time(),quality_not_yet_evaluated=True))
parallel.barrier();torch.distributed.destroy_process_group()
