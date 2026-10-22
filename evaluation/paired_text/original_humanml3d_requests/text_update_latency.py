"""Separate online text-update encoding timing; never hidden in cached-model latency."""
from motion_mamba2.paths import artifact_path
import os,sys,time
from pathlib import Path
from execute import R,rd,wr,digest
import numpy as np,torch
os.environ.update(HF_HOME=artifact_path('huggingface_ed9ba7d805'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
base=Path(artifact_path('src_40b98bd860'))
sys.path[:0]=[str(base/'motion_training/_runtime'),str(base)]
from motion_mamba2.models.load_model import load_text_encoder
torch.set_num_threads(2)
enc=load_text_encoder(mode='local',device='cuda:0');prompts=rd(R/'TEXT_INDEX.json')['prompts'];frozen=np.load(R/'TEXT_FEATURES.npy')
rows=[]
with torch.inference_mode():
 for p in prompts:
  for _ in range(2):enc([p])
 for i in range(32):
  j=i%len(prompts);p=prompts[j];torch.cuda.synchronize();start=time.perf_counter();f,l=enc([p]);torch.cuda.synchronize();elapsed=time.perf_counter()-start
  assert np.array_equal(f[0,0].float().cpu().numpy(),frozen[j]),'Text encoding changed frozen generator input'
  rows.append(dict(prompt=p,seconds=elapsed,exact_frozen_feature=True))
times=np.asarray([v['seconds'] for v in rows])*1000
wr(R/'TEXT_UPDATE_LATENCY.json',dict(passed=True,batch=1,warmups_per_prompt=2,measured_updates=32,median_ms=float(np.median(times)),mean_ms=float(times.mean()),p95_ms=float(np.quantile(times,.95)),rows=rows,scope='uncached current-command tokenizer+LLM2Vec encoding, separate from cached generator latency, excludes loading/network/UI',time=time.time()))
