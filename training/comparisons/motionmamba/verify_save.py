from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,time,json,hashlib
S=Path(artifact_path('g1_motionmamba_matched100k_20261022_3b602cb1f7'));R=S/'MotionMamba_G1'
os.environ['BASELINE_RUN_ROOT']=str(R);sys.path.insert(0,str(S/'scripts'))
from common import *
import torch
from motion_training.io import identity,sha

def verify():
 torch.set_num_threads(2)
 latest=read(R/'LATEST.json');cp=Path(latest['checkpoint']);c=torch.load(cp,map_location='cpu',mmap=True,weights_only=False)
 assert c['method']=='motionmamba' and c['step']==c['scheduler']['completed']==latest['step']
 expected=min(c['step'],75000)*32+max(c['step']-75000,0)*512
 assert c['exposures']==expected and len(c['rng_by_rank'])==4 and len(set(c['rank_state_hashes']))==1
 assert identity([c['model'],c['optimizer']])==c['rank_state_hashes'][0]
 states=c['optimizer']['state'];assert len(states)>0
 for v in states.values():
  assert int(v['step'])==c['step']
  for k,x in v.items():
   if torch.is_tensor(x):
    assert torch.isfinite(x).all(),k
    if k in ['exp_avg','exp_avg_sq']:assert x.dtype==torch.float32
 assert all(torch.isfinite(v).all() for v in c['model'].values() if torch.is_tensor(v))
 for n,h in c['source_hashes'].items():assert sha(S/n)==h,n
 zero=torch.load(R/'initial/checkpoint0.pt',map_location='cpu',mmap=True,weights_only=False)
 assert zero['step']==zero['scheduler']['completed']==0 and not zero['optimizer']['state']
 for name in ['CPU_PREFLIGHT','PROTOCOL','NUMERICS','INTERFACE','ADMISSION','GEOMETRY_GRAPH_ADMISSION']:assert read(R/f'gates/{name}.json')['passed']
 info=dict(passed=True,model='MotionMamba_G1',checkpoint=str(cp),checkpoint_sha256=sha(cp),actual_step=c['step'],scheduler_completed=c['scheduler']['completed'],exposures=expected,
  Adam_states=len(states),all_Adam_steps=c['step'],Adam_moments_FP32=True,finite_weights_and_Adam=True,four_rank_RNG=True,all_rank_states_equal=True,persisted_state_hash_recomputed=True,
  initial_Adam_empty=True,initial_step_zero=True,source_hashes_verified=True,discarded_gate_updates_not_counted=True,warm_timings=read(R/'gates/ADMISSION.json')['timings'],time=time.time())
 write(R/('FINAL_SAVE_VERIFIED.json' if c['step']==100000 else 'FULL_SAVE_PROOF.json'),info)
 print(json.dumps({k:v for k,v in info.items() if k!='warm_timings'},indent=2))
if __name__=='__main__':verify()
