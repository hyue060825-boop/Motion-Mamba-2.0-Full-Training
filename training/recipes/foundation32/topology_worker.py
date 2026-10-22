from motion_mamba2.paths import artifact_path
import os,sys,json,time,argparse,gc
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'));T=D/'topology4'
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import torch
from motion_training import runtime,engine,batching,streaming,distributed as parallel
from motion_training.io import read,write,append,identity,Budget,sha
parallel.setup();batching.install();cfg=engine.config(D/'configs/train.json');cp=T/'PARENT.json';parent=read(cp);start=parent['step'];checkpoint=parent['checkpoint'];out=T/('two' if parallel.world()==2 else 'four');out.mkdir(exist_ok=True)
probe_cfg=dict(cfg,name='discarded_resource_probe',budget_root=str(T/'budget'));budget=Budget(probe_cfg['budget_root']);plan=streaming.load(D/'train/data/MANIFEST.json')['plan']
m,c,ck=runtime.load(checkpoint,True);opt=runtime.optimizer(m,ck,probe_cfg,resume=True);runtime.restore_rng(ck['rng_by_rank'][parallel.rank()%2]);assert opt.training_schedule['completed']==start;ck['_framework_global_start']=start
# Hash restored state before any discarded probe update.
if parallel.leader():write(out/'RESTORED.json',dict(model=identity(m.denoiser.model.state_dict()),optimizer=identity(opt.state_dict()),scheduler=opt.training_schedule,sampler_cursor=start*32,parent_sha256=sha(checkpoint)))
for i in range(1,22):
 step=start+i;records=plan[(step-1)*32:step*32];probe_cfg['audit_every']=1 if i==1 else 1000
 rec=batching.update(m,c,ck,opt,probe_cfg,records,step,'checks',budget)
 if parallel.leader():append(out/'updates.jsonl',rec);print('TOPOLOGY',parallel.world(),step,round(rec['seconds'],3),round(rec['loss'],6),flush=True)
 if i==1:
  if parallel.leader():torch.save({n:p.grad.detach().float().cpu() for n,p in m.denoiser.model.named_parameters() if p.grad is not None},out/'first_gradients.pt')
  engine.save(out/'checkpoint1.pt',m,ck,opt,probe_cfg,1,D/'train/data/MANIFEST.json',None)
 if i==21:engine.save(out/'checkpoint21.pt',m,ck,opt,probe_cfg,21,D/'train/data/MANIFEST.json',None)
parallel.barrier();torch.distributed.destroy_process_group()
