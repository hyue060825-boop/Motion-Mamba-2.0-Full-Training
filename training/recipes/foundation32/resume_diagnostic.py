from motion_mamba2.paths import artifact_path
import os,sys,json,gc
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from motion_training import runtime,engine,batching,distributed as parallel
from motion_training.io import read,write,identity,Budget
parallel.setup();batching.install();cfg=engine.config(D/'configs/train.json');cfg=dict(cfg,budget_root=str(D/'bench_budget'),name='speed_gate_batch32');budget=Budget(cfg['budget_root'])
plan=read(artifact_path('manifest_json_d6a24ce639'))['plan'];hf=(52,144);records=[r for r in plan if (r['H'],r['F'])==hf][:32]
m,c,ck=runtime.load(cfg['parent'],True);opt=runtime.optimizer(m,ck,cfg);ck['_framework_global_start']=0
for step in [1,2]:batching.update(m,c,ck,opt,cfg,records,step,'checks',budget)
cp=D/'validation_diagnostic/checkpoint2.pt';cp.parent.mkdir(exist_ok=True);engine.save(cp,m,ck,opt,cfg,2,D/'train/data/MANIFEST.json',None);pre_hash=identity([m.denoiser.model.state_dict(),opt.state_dict()])
batching.update(m,c,ck,opt,cfg,records,3,'checks',budget);reference=identity([m.denoiser.model.state_dict(),opt.state_dict()]);reference_state={n:p.detach().float().cpu().clone() for n,p in runtime.master_named(m.denoiser.model,opt)};reference_grad={n:p.grad.detach().float().cpu().clone() for n,p in m.denoiser.model.named_parameters() if p.grad is not None};del m,c,ck,opt;gc.collect();torch.cuda.empty_cache();parallel.barrier()
m,c,ck=runtime.load(cp,True);opt=runtime.optimizer(m,ck,cfg,resume=True);runtime.restore_rng(ck['rng_by_rank'][parallel.rank()]);assert opt.training_schedule['completed']==2;print('PRE_RESTORE_MATCH',parallel.rank(),pre_hash==identity([m.denoiser.model.state_dict(),opt.state_dict()]),flush=True)
batching.update(m,c,ck,opt,cfg,records,3,'checks',budget);restarted=identity([m.denoiser.model.state_dict(),opt.state_dict()]);parts=parallel.gather(dict(rank=parallel.rank(),reference=reference,restarted=restarted,equal=reference==restarted));delta=[];a=b=dot=err=0.
for n,p in runtime.master_named(m.denoiser.model,opt):
 x=reference_state.pop(n);y=p.detach().float().cpu();delta.append(dict(name=n,max=float((x-y).abs().max()),relative=float((x-y).norm()/x.norm().clamp_min(1e-30))))
for n,p in m.denoiser.model.named_parameters():
 if n not in reference_grad:continue
 x=reference_grad.pop(n).double();y=p.grad.detach().float().cpu().double();a+=float(x.square().sum());b+=float(y.square().sum());dot+=float((x*y).sum());err+=float((x-y).square().sum())
if parallel.leader():write(D/'RESUME_DIAGNOSIS.json',dict(hashes=parts,deltas=delta,gradient_cosine=dot/(a*b)**.5,gradient_relative=(err/a)**.5));print('DIAG',max(x['max'] for x in delta),dot/(a*b)**.5,(err/a)**.5,flush=True)
if parallel.leader():write(D/'RESUME_DIAGNOSIS_STATE.json',dict(status='DIAGNOSIS_NOT_A_GATE',world_size=parallel.world(),global_batch32=True,complete_optimizer_boundary=True,model_master_adam_exact=True,ranks=parts));print('RESUME_GATE_PASS',flush=True)
parallel.barrier()
