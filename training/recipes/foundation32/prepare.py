from motion_mamba2.paths import artifact_path
import os,sys,json,time
from pathlib import Path
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
sys.path[:0]=[str(D/'code/src/motion_training/_runtime'),str(D/'code/src'),artifact_path('mamba_op_triton_3220f2c818'),artifact_path('state-spaces-mamba_fd8479ca04')]
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from motion_training import streaming,data,engine
from motion_training.io import read,write,sha
cfg=engine.config(D/'configs/train.json');h=streaming.prepare(cfg,D/'train/data');texts=read(D/'train/data/CAPTIONS.json');available=set()
for folder in cfg['text_roots']:
 if Path(folder).exists():available.update(p.stem for p in Path(folder).glob('*.npy'))
missing=[t for t in texts if data.text_key(t) not in available]
write(D/'TEXT_REQUIREMENTS.json',dict(total=len(texts),missing=len(missing),captions=missing))
print('TEXT_MISSING',len(missing),'TOTAL',len(texts),flush=True)
if missing:
 import torch,numpy as np
 from motion_mamba2.models.load_model import load_text_encoder
 torch.set_num_threads(2);encoder=load_text_encoder(mode='local',device='cuda:0');folder=Path(cfg['text_roots'][0]);folder.mkdir(exist_ok=True)
 with torch.inference_mode():
  for i,text in enumerate(missing):
   p=folder/(data.text_key(text)+'.npy')
   if p.exists():continue
   value,length=encoder(text);a=value[:length].float().cpu().numpy();assert a.size==4096 and np.isfinite(a).all();tmp=p.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(p)
   if i%100==0:print('TEXT_ENCODED',i,'/',len(missing),flush=True)
write(D/'train/TEXT_READY.json',dict(total=len(texts),new=len(missing),exact_encoder=True,manifest_sha256=sha(D/'train/data/MANIFEST.json')))
print('DATA_TEXT_READY',flush=True)
