from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,hashlib
R=Path(__file__).resolve().parent
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2')
base=Path(artifact_path('src_40b98bd860'));sys.path[:0]=[str(base/'motion_training/_runtime'),str(base)]
import numpy as np,torch
from motion_mamba2.models.load_model import load_text_encoder
from execute import rd,wr,sha
assert rd(R/'PROTOCOL_FROZEN.json')['passed']
if not (R/'TEXT_INDEX.json').exists():
 torch.set_num_threads(2);captions=rd(R/'CAPTIONS.json');enc=load_text_encoder(mode='local',device='cuda:0');out=[]
 with torch.inference_mode():
  for i,c in enumerate(captions):
   f,l=enc([c]);assert f.shape==(1,1,4096) and l==[1] and torch.isfinite(f).all();out.append(f[0,0].float().cpu().numpy())
   if i%16==0:wr(R/'TEXT_PROGRESS.json',dict(completed=i+1,total=len(captions),time=time.time()))
 np.save(R/'TEXT_FEATURES.npy',np.stack(out));wr(R/'TEXT_INDEX.json',dict(prompts=captions,frozen_encoder=True,no_rewritten_text=True,time=time.time()))
