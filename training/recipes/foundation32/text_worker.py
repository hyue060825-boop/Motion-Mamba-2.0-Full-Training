from motion_mamba2.paths import artifact_path
import os,sys,json,time,argparse,hashlib
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from motion_training import engine,data
from motion_training.io import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();cfg=engine.config(D/'configs/train.json');texts=read(D/'train/data/CAPTIONS.json');available=set()
for root in cfg['text_roots']:
 if Path(root).exists():available.update(p.stem for p in Path(root).glob('*.npy'))
missing=[t for t in texts if data.text_key(t) not in available and int(data.text_key(t),16)%2==a.shard];started=time.time();print('MISSING_SHARD',a.shard,len(missing),flush=True)
if missing:
 from motion_mamba2.models.load_model import load_text_encoder
 torch.set_num_threads(2);encoder=load_text_encoder(mode='local',device='cuda:0');folder=Path(cfg['text_roots'][0]);folder.mkdir(exist_ok=True);tick=time.time()
 with torch.inference_mode():
  for i,text in enumerate(missing):
   p=folder/(data.text_key(text)+'.npy')
   if p.exists():continue
   v,n=encoder(text);v=v[:n].float().cpu().numpy();assert v.size==4096 and np.isfinite(v).all();tmp=p.with_suffix('.tmp.npy');np.save(tmp,v);tmp.replace(p)
   if i%100==0:
    rate=(i+1)/max(time.time()-tick,1e-9);print('TEXT_SHARD',a.shard,i+1,'/',len(missing),'rate',round(rate,2),'remaining_sec',round((len(missing)-i-1)/rate),flush=True)
write(D/f'text_shard{a.shard}_DONE.json',dict(complete=True,exact_official_encoder=True,internal_batch_size=1,encoded=len(missing),seconds=time.time()-started,caption_sha256=sha(D/'train/data/CAPTIONS.json')))
print('TEXT_SHARD_DONE',a.shard,flush=True)
