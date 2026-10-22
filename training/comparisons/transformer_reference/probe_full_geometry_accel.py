from common import *
import argparse, copy, statistics, gc
import torch
import adapter, accelerated_loss, pipeline_accel as fast, lazy_plan
from motion_training import runtime, engine, distributed as parallel
from motion_training.io import identity, Budget
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
adapter.install();parallel.setup();assert parallel.world()==4
m,c,ck=runtime.load(a.checkpoint,True);raw=m.denoiser.model
cfg=read(ROOT/'configs/batch512.json');opt=runtime.optimizer(m,ck,cfg,resume=True)
start=ck['step'];idx=parallel.rank()%len(ck['rng_by_rank']);base_state={n:v.detach().cpu().clone() for n,v in raw.state_dict().items()}
base_opt=copy.deepcopy(ck['optimizer']);base_sched=copy.deepcopy(opt.training_schedule)
budgetroot=ROOT/'geometry_accel_probe_budget';budgetroot.mkdir(exist_ok=True)
if parallel.leader():write(budgetroot/'LIMITS.json',dict(caps=dict(checks=30),total_updates=30))
parallel.barrier();budget=Budget(budgetroot);late=lazy_plan.Plan(read(LATE/'MANIFEST_TEMPLATE.json'))
records=late.records_at(max(start-75000,0),512)
def setmode(kind):
 if hasattr(c,'_batch_loss'):
  del c._batch_loss;gc.collect()
 engine.update=accelerated_loss.install() if kind=='serial' else fast.install('serial_codec',4)
 raw.load_state_dict(base_state);opt.load_state_dict(copy.deepcopy(base_opt));opt.training_schedule=copy.deepcopy(base_sched)
 runtime.restore_rng(ck['rng_by_rank'][idx]);c.reset_memory()
 assert identity(opt.state_dict())==identity(base_opt)
def cpu_state():
 return {n:v.detach().float().cpu().clone() for n,v in raw.state_dict().items()}
def relative(x,y):
 n=sum(float(v.double().square().sum()) for v in x.values());e=sum(float((x[k].double()-y[k].double()).square().sum()) for k in x)
 dot=sum(float((x[k].double()*y[k].double()).sum()) for k in x);n2=sum(float(v.double().square().sum()) for v in y.values())
 return dict(relative_l2=(e/max(n,1e-30))**.5,cosine=dot/max((n*n2)**.5,1e-30),max_absolute=max(float((x[k]-y[k]).abs().max()) for k in x))
setmode('serial');baseline=engine.update(m,c,ck,opt,cfg,records,777,'checks',budget)
grad0={n:p.grad.detach().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None};state0=cpu_state();opt0=copy.deepcopy(opt.state_dict());schedule0=copy.deepcopy(opt.training_schedule)
print('BASELINE_COMPLETE',parallel.rank(),baseline['seconds'],flush=True)
setmode('batch');candidate=engine.update(m,c,ck,opt,cfg,records,777,'checks',budget)
grad1={n:p.grad.detach().cpu().clone() for n,p in raw.named_parameters() if p.grad is not None};state1=cpu_state();opt1=opt.state_dict()
g=relative(grad0,grad1);updates=relative({n:state0[n]-base_state[n].float() for n in state0},{n:state1[n]-base_state[n].float() for n in state1})
lossrel=abs(candidate['loss']-baseline['loss'])/max(abs(baseline['loss']),1e-9)
mom0={};mom1={}
for k,v in opt0['state'].items():
 for name,t in v.items():
  if torch.is_tensor(t):mom0[str(k)+name]=t.detach().cpu().float();mom1[str(k)+name]=opt1['state'][k][name].detach().cpu().float()
mo=relative(mom0,mom1)
local=dict(rank=parallel.rank(),gradient=g,parameter_update=updates,optimizer_moments=mo,loss_relative=lossrel,scheduler_equal=schedule0==opt.training_schedule,baseline=baseline['seconds'],candidate=candidate['seconds'])
local['passed']=g['relative_l2']<=1e-3 and lossrel<=1e-5 and updates['relative_l2']<=.05 and updates['cosine']>=.995 and updates['max_absolute']<=5e-6 and mo['relative_l2']<=1e-3 and local['scheduler_equal']
parts=parallel.gather(local)
if parallel.leader():write(ROOT/'gates/GEOMETRY_FULL_UPDATE_PARTIAL.json',dict(ranks=parts,checkpoint=a.checkpoint))
assert all(x['passed'] for x in parts),parts
# Same batch512, four ranks, actual optimizer updates; discard all probe weights.
del grad0,grad1,state0,state1,opt0,opt1,base_opt,base_state,mom0,mom1;gc.collect()
measure=[]
for i in range(6):
 rec=engine.update(m,c,ck,opt,cfg,late.records_at(max(start-75000,0)+i+1,512),778+i,'checks',budget)
 measure.append(dict(seconds=rec['seconds'],phases=rec['phases_by_rank'],loss=rec['loss']))
 print('TIMING',i,rec['seconds'],flush=True)
hashes=parallel.gather(identity([raw.state_dict(),opt.state_dict()]))
assert len(set(hashes))==1
reserved=parallel.gather(torch.cuda.max_memory_reserved()/2**30)
assert max(reserved)<torch.cuda.get_device_properties(0).total_memory/2**30*.90
if parallel.leader():write(ROOT/'gates/GEOMETRY_FULL_UPDATE.json',dict(passed=True,ranks=parts,checkpoint=a.checkpoint,checkpoint_sha256=__import__('motion_training.io',fromlist=['sha']).sha(a.checkpoint),four_rank_states_exact=True,measurements=measure,median_seconds=statistics.median([x['seconds'] for x in measure]),peak_reserved_GiB=reserved,discarded_updates=8,production_checkpoint_unchanged=True))
parallel.barrier();torch.distributed.destroy_process_group()
