import argparse, gc, statistics
from common import *
import torch
from motion_training import runtime, batching, distributed as parallel
from motion_training.io import Budget, append, identity
from lazy_plan import Plan
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['capacity','equivalence','distributed'],required=True)
p.add_argument('--micro',type=int,required=True);p.add_argument('--label',required=True);p.add_argument('--parent')
p.add_argument('--save-update',action='store_true')
a=p.parse_args();parallel.setup();batching.install()
cfg=read(ROOT/'config.json');cfg.update(name='discarded_batch_admission_probe',budget_root=str(ROOT/'probe_budget'),per_rank_microbatch=a.micro,audit_every=1000000)
plan=Plan(read(ROOT/'MANIFEST_TEMPLATE.json'))
folder=ROOT/'probes'/a.label;folder.mkdir(parents=True,exist_ok=True)
cp=a.parent or cfg['parent'];m,c,ck=runtime.load(cp,True);opt=runtime.optimizer(m,ck,cfg,resume=True)
runtime.restore_rng(ck['rng_by_rank'][parallel.rank()%len(ck['rng_by_rank'])])
assert opt.training_schedule['completed']>=75000
if parallel.leader():write(folder/'RESTORED.json',dict(parent=cp,model_hash=identity(m.denoiser.model.state_dict()),optimizer_hash=identity(opt.state_dict()),scheduler=opt.training_schedule,weights_never_promoted=True))
if parallel.leader():
    stochastic={n:float(q.p) for n,q in m.denoiser.model.named_modules() if isinstance(q,torch.nn.Dropout) and q.p>0}
    write(folder/'STOCHASTIC_MODULES.json',stochastic)
budget=Budget(cfg['budget_root']);measure=[];failed=None
try:
    if a.mode=='capacity':
        assert parallel.world()==1
        cfg['batch_size']=a.micro;layouts=[(196,0),(144,52),(52,144),(0,196),(52,0),(0,0),(52,0),(52,0)]
    elif a.mode=='equivalence':
        cfg['batch_size']=64;layouts=[(196,0)]
    else:
        cfg['batch_size']=512;layouts=[(196,0),(52,0),(144,52),(0,196),(52,144),(52,0)]
    for i,hf in enumerate(layouts):
        cfg['audit_every']=1 if i==0 and a.mode!='capacity' else 1000000
        records=plan.records_at(60000+i,cfg['batch_size'],layout=hf)
        if a.save_update and i==0 and parallel.leader():
            before={n:q.detach().float().cpu().clone() for n,q in runtime.master_named(m.denoiser.model,opt)}
        torch.cuda.reset_peak_memory_stats()
        rec=batching.update(m,c,ck,opt,cfg,records,2+i,'checks',budget)
        total=torch.cuda.get_device_properties(0).total_memory
        memory=dict(allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved(),total=total)
        peaks=parallel.gather(memory)
        out=dict(H=hf[0],F=hf[1],loss=rec['loss'],seconds=rec['seconds'],micro=a.micro,global_batch=cfg['batch_size'],world_size=parallel.world(),peaks=peaks,
                 gradient_norms=[x['clip'] for x in rec['phases_by_rank']],finite_loss_and_gradient=True,
                 receipt_hash=identity(sorted([v for part in rec['receipts'] for v in part],key=lambda v:v['id'])) if rec['receipts'] else None)
        if parallel.leader():append(folder/'updates.jsonl',out);print('PROBE',a.label,hf,rec['seconds'],peaks,flush=True)
        measure.append(out)
        if a.mode in ['equivalence','distributed'] and i==0:
            if parallel.leader():torch.save({n:q.grad.detach().float().cpu() for n,q in m.denoiser.model.named_parameters() if q.grad is not None},folder/'gradients.pt')
            if a.save_update and parallel.leader():
                deltas={n:q.detach().float().cpu()-before[n] for n,q in runtime.master_named(m.denoiser.model,opt)}
                assert all(torch.isfinite(v).all() for v in deltas.values())
                torch.save(deltas,folder/'master_update.pt');del before,deltas
        if any(v['reserved_peak']>.90*v['total'] for v in peaks):
            failed='less_than_10_percent_memory_headroom';break
    if a.mode=='distributed':
        hashes=parallel.gather(identity([m.denoiser.model.state_dict(),opt.state_dict()]))
        assert len(set(hashes))==1,'distributed ranks diverged'
except torch.cuda.OutOfMemoryError as error:
    failed='CUDA_OOM: '+str(error)
    if parallel.world()>1: raise
if parallel.leader():
    write(folder/'RESULT.json',dict(passed=failed is None,failure=failed,mode=a.mode,micro=a.micro,world_size=parallel.world(),global_batch=cfg['batch_size'],rows=measure,
        steady_seconds=statistics.median([r['seconds'] for r in measure[1:]]) if len(measure)>1 else None,
        reserved_peak=max([r['peaks'][0]['reserved_peak'] for r in measure] or [0]),weights_discarded=True))
if parallel.world()>1:parallel.barrier();torch.distributed.destroy_process_group()
