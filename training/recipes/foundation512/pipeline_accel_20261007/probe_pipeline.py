from motion_mamba2.paths import artifact_path
import argparse,sys,time,gc,statistics
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'pipeline_accel_20261007'
sys.path.insert(0,str(R));from common import *
import torch
from motion_training import runtime,batching,distributed as parallel
from motion_training.io import Budget,append,identity
from lazy_plan import Plan
import pipeline_accel as acc
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reference','loss_only','batch_all','serial_codec'],required=True)
p.add_argument('--workers',type=int,default=4);p.add_argument('--label',required=True);p.add_argument('--save-gate',action='store_true')
p.add_argument('--count',type=int,default=18);a=p.parse_args();parallel.setup();batching.install()
cfg=read(R/'config.json');cfg['budget_root']=str(S/'budget');cfg['audit_every']=10000000
folder=S/'probes'/a.label;folder.mkdir(parents=True,exist_ok=True)
cp=read(S/'BOUNDARY.json')['checkpoint'];m,c,ck=runtime.load(cp,True);raw=m.denoiser.model
opt=runtime.optimizer(m,ck,cfg,resume=True);runtime.restore_rng(ck['rng_by_rank'][parallel.rank()])
fn=batching.update if a.mode=='reference' else acc.install(a.mode,a.workers)
plan=Plan(read(R/'MANIFEST_TEMPLATE.json'));budget=Budget(cfg['budget_root'])
if parallel.leader():
    write(folder/'RESTORED.json',dict(checkpoint=cp,model_hash=identity(raw.state_dict()),optimizer_hash=identity(opt.state_dict()),scheduler=opt.training_schedule))
layouts=[(196,0),(52,0),(0,0),(0,196),(52,144),(144,52)]
rows=[]
for i in range(a.count):
    hf=layouts[i%len(layouts)];records=plan.records_at(67000+i,cfg['batch_size'],layout=hf)
    if a.save_gate and i==0 and parallel.leader():before={n:q.detach().float().cpu().clone() for n,q in runtime.master_named(raw,opt)}
    torch.cuda.reset_peak_memory_stats();rec=fn(m,c,ck,opt,cfg,records,2+i,'checks',budget)
    peaks=parallel.gather(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),total=torch.cuda.get_device_properties(0).total_memory))
    row=dict(index=i,H=hf[0],F=hf[1],seconds=rec['seconds'],loss=rec['loss'],phase=rec['phases_by_rank'],peaks=peaks,
        mode=a.mode,workers=a.workers,global_batch=512,micro=64,ranks=parallel.world(),sampler_identity=identity(records))
    rows.append(row)
    if parallel.leader():append(folder/'updates.jsonl',row);print('PROBE',a.label,i,hf,round(rec['seconds'],4),peaks,flush=True)
    if a.save_gate and i==0 and parallel.leader():
        torch.save({n:q.grad.detach().float().cpu() for n,q in raw.named_parameters() if q.grad is not None},folder/'gradients.pt')
        delta={n:q.detach().float().cpu()-before[n] for n,q in runtime.master_named(raw,opt)}
        assert all(torch.isfinite(v).all() for v in delta.values())
        torch.save(delta,folder/'master_delta.pt');del delta,before
        state=opt.state_dict();torch.save({str(i):{k:v.detach().float().cpu() for k,v in d.items() if k in ['exp_avg','exp_avg_sq']} for i,d in state['state'].items()},folder/'moments.pt')
    if any(p['reserved']>.9*p['total'] for p in peaks):raise RuntimeError('Less than10% GPU memory headroom')
if parallel.world()>1:
    hashes=parallel.gather(identity([raw.state_dict(),opt.state_dict()]));assert len(set(hashes))==1,'Rank state diverged'
if parallel.leader():
    steady=rows[6:];write(folder/'RESULT.json',dict(passed=True,mode=a.mode,workers=a.workers,rows=rows,
        mean_seconds=statistics.mean(x['seconds'] for x in steady),median_seconds=statistics.median(x['seconds'] for x in steady),
        peak_reserved=max(p['reserved'] for x in rows for p in x['peaks']),global_batch=512,micro=64,ranks=parallel.world(),
        discarded_probe_updates=True,checkpoint=cp))
if hasattr(c,'_cpu_prefetch_pool'):c._cpu_prefetch_pool.shutdown()
if parallel.world()>1:parallel.barrier();torch.distributed.destroy_process_group()
