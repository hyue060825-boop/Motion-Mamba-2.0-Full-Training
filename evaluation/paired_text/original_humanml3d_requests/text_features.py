"""Encode only frozen command strings; no task-dependent answer expansion."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time
import numpy as np
R=Path(os.environ.get('UNIFIED_EVAL_ROOT',artifact_path('unified_interactive_easy_v3_20261021_ebde76396c')))
if (R/'TEXT_INDEX.json').exists():raise SystemExit(0)
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2')
base=Path(artifact_path('src_40b98bd860'))
sys.path[:0]=[str(base/'motion_training/_runtime'),str(base)]
import torch
torch.set_num_threads(2)
from motion_mamba2.models.load_model import load_text_encoder
prompts=list(dict.fromkeys(json.loads((R/'PLAN.json').read_text())['prompts'].values()))
enc=load_text_encoder(mode='local',device='cuda:0')
out=[]
with torch.inference_mode():
 for p in prompts:
  f,l=enc([p]);assert f.shape==(1,1,4096) and l==[1] and torch.isfinite(f).all()
  out.append(f[0,0].float().cpu().numpy())
np.save(R/'TEXT_FEATURES.npy',np.stack(out))
(R/'TEXT_INDEX.json').write_text(json.dumps(dict(prompts=prompts,frozen_encoder=True,answer_not_resolved=True,training_updates=0,time=time.time()),indent=2)+'\n')
