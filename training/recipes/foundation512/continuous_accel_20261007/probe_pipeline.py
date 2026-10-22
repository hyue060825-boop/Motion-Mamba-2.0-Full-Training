from motion_mamba2.paths import artifact_path
import argparse,sys,time,gc,statistics
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'continuous_accel_20261007'
sys.path.insert(0,str(R));from common import *
import torch
from motion_training import runtime,batching,distributed as parallel
from motion_training.io import Budget,append,identity
from lazy_plan import Plan
sys.path.insert(0,str(R/'codec_graph_accel_20261007'))
import pipeline_accel as acc
sys.path.insert(0,str(S))
from optimizer_copy import install_optimizer_copy
from profile_parts import install_profile
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reference','loss_only','batch_all','serial_codec'],required=True)
p.add_argument('--workers',type=int,default=4);p.add_argument('--label',required=True);p.add_argument('--save-gate',action='store_true')
p.add_argument('--resume-gate-save',action='store_true');p.add_argument('--resume-gate-load');p.add_argument('--profile',action='store_true');p.add_argument('--count',type=int,default=18);a=p.parse_args();parallel.setup();batching.install()
cfg=read(R/'config.json');cfg['budget_root']=str(S/'budget');cfg['audit_every']=10000000
folder=S/'probes'/a.label;folder.mkdir(parents=True,exist_ok=True)
cp=a.resume_gate_load or read(S/'BOUNDARY.json')['checkpoint'];m,c,ck=runtime.load(cp,True);raw=m.denoiser.model
opt=runtime.optimizer(m,ck,cfg,resume=True);runtime.restore_rng(ck['rng_by_rank'][parallel.rank()])
fn=acc.install('serial_codec',0)
if a.mode!='reference':install_optimizer_copy(opt)
if a.profile:install_profile(acc,opt,c)
plan=Plan(read(R/'MANIFEST_TEMPLATE.json'));budget=Budget(cfg['budget_root'])
if parallel.leader():
    write(folder/'RESTORED.json',dict(checkpoint=cp,model_hash=identity(raw.state_dict()),optimizer_hash=identity(opt.state_dict()),scheduler=opt.training_schedule))
layouts=[(196,0),(52,0),(0,0),(0,196),(52,144),(144,52)]
rows=[]
for i in (range(1,2) if a.resume_gate_load else range(a.count)):
    hf=layouts[i%len(layouts)];records=plan.records_at(67000+i,cfg['batch_size'],layout=hf)
    if a.save_gate and i==0 and parallel.leader():before={n:q.detach().float().cpu().clone() for n,q in runtime.master_named(raw,opt)}
    if a.profile:c._profile_parts=dict(denoiser=0.,geometry_loss=0.)
    torch.cuda.reset_peak_memory_stats();rec=fn(m,c,ck,opt,cfg,records,2+i,'checks',budget)
    peaks=parallel.gather(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),total=torch.cuda.get_device_properties(0).total_memory))
    row=dict(index=i,H=hf[0],F=hf[1],seconds=rec['seconds'],loss=rec['loss'],phase=rec['phases_by_rank'],peaks=peaks,
        mode=a.mode,workers=a.workers,global_batch=512,micro=64,ranks=parallel.world(),sampler_identity=identity(records),detail=parallel.gather(dict(getattr(c,'_profile_parts',{}),optimizer=getattr(opt,'_profile_optimizer_seconds',0.))))
    rows.append(row)
    if parallel.leader():append(folder/'updates.jsonl',row);print('PROBE',a.label,i,hf,round(rec['seconds'],4),peaks,flush=True)
    if a.save_gate and i==0 and parallel.leader():
        torch.save({n:q.grad.detach().float().cpu() for n,q in raw.named_parameters() if q.grad is not None},folder/'gradients.pt')
        delta={n:q.detach().float().cpu()-before[n] for n,q in runtime.master_named(raw,opt)}
        assert all(torch.isfinite(v).all() for v in delta.values())
        torch.save(delta,folder/'master_delta.pt');del delta,before
        state=opt.state_dict();torch.save({str(i):{k:v.detach().float().cpu() for k,v in d.items() if k in ['exp_avg','exp_avg_sq']} for i,d in state['state'].items()},folder/'moments.pt')
    if a.resume_gate_save and i==0:
        from motion_training import engine
        ck['_framework_global_start']=75000
        engine.save(folder/'resume_checkpoint.pt',m,ck,opt,cfg,read(S/'BOUNDARY.json')['step']+1,R/'train/data/MANIFEST.json',None)
        parallel.barrier()
    if (a.resume_gate_save or a.resume_gate_load) and i==1:
        h=identity([raw.state_dict(),opt.state_dict(),opt.training_schedule])
        assert len(set(parallel.gather(h)))==1
        if parallel.leader():write(folder/'AFTER_NEXT_UPDATE.json',dict(state_hash=h,loss=rec['loss'],sampler_identity=identity(records)))
    if any(p['reserved']>.9*p['total'] for p in peaks):raise RuntimeError('Less than10% GPU memory headroom')
if parallel.world()>1:
    hashes=parallel.gather(identity([raw.state_dict(),opt.state_dict()]));assert len(set(hashes))==1,'Rank state diverged'
if parallel.leader():
    steady=rows[6:] or rows;write(folder/'RESULT.json',dict(passed=True,mode=a.mode,workers=a.workers,rows=rows,
        mean_seconds=statistics.mean(x['seconds'] for x in steady),median_seconds=statistics.median(x['seconds'] for x in steady),
        peak_reserved=max(p['reserved'] for x in rows for p in x['peaks']),global_batch=512,micro=64,ranks=parallel.world(),
        discarded_probe_updates=True,checkpoint=cp))
if hasattr(c,'_cpu_prefetch_pool'):c._cpu_prefetch_pool.shutdown()
if parallel.world()>1:parallel.barrier();torch.distributed.destroy_process_group()
