"""Read-only second implementation of endpoint and normalized-feature metrics."""
from pathlib import Path
import os,json,csv,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from execute import R,P,rd,wr,sha,digest,schedule,verify
from prepare_scoring import representation
from native_eval.metrics import fid
import numpy as np,torch

def main():
 verify();torch.set_num_threads(2);rep,_=representation();checks=[]
 cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test'];assert len(cases)==128
 assets=dict(np.load(R/'INPUT_ASSETS.npz'));lookup={v:i for i,v in enumerate(assets['case_ids'])}
 bank=dict(np.load(R/'scoring/REFERENCE_FEATURES.npz'));real=bank['distribution'][bank['quality_indices']].astype(np.float64);real/=np.linalg.norm(real,axis=-1,keepdims=True)
 for task in ['short52','nominal9','nominal20','nominal40']:
  d=R/'models/motionmamba'/task;s=R/'scoring/models/motionmamba'/task;rec=rd(s/'COMPLETE.json')
  assert rec['passed']
  for n,h in rec['files'].items():assert sha(s/n)==h,(task,n)
  base=rd(P/'scoring/models/native100'/task/'INDEPENDENT_AUDIT.json');own=rd(s/'INDEPENDENT_AUDIT.json');assert own['passed'] and own['paired_inputs']==base['paired_inputs']
  gate=rd(d/'ACCELERATION_GATE.json');assert gate['passed'] and len(gate['rows'])==4 and all(all(v['checks'].values()) for v in gate['rows'])
  latency=rd(d/'LATENCY.json');assert latency['batch']==1 and latency['warmup_sequences']==2 and len(latency['rows'])==32
  score=rd(s/'SCORES.json');W=len(schedule(task)[0]);v=np.array([r['elapsed_s'] for r in latency['rows']])*1000/W
  assert abs(np.median(v)-latency['median_ms_window'])<1e-9 and abs(v.mean()-latency['mean_ms_window'])<1e-9
  roots=[];ends=[]
  for ci,c in enumerate(cases):
   for si,seed in enumerate(c['noise_seeds']):
    trial=d/'trials'/f'{ci:03d}_s{si}';a=rd(trial/'INPUT_AND_EVENTS.json');comp=rd(trial/'COMPLETE.json')
    assert comp['seed']==seed and sha(trial/'geometry.npz')==comp['geometry_sha256'] and sha(trial/'INPUT_AND_EVENTS.json')==comp['audit_sha256']
    initial=assets['initial_physical'][lookup[c['case_id']]];vals=assets['goal_physical_values'][lookup[c['case_id']]]
    assert digest(initial)==a['initial_sha256'] and digest(vals)==a['sparse_physical_goal_sha256'] and a['no_hidden_GT'] and a['noise_seed']==seed
    assert len(a['events'])==W and all(v['observed_scalars']==8 and v['target_motion_is_None'] for v in a['events'])
    with np.load(trial/'geometry.npz') as z:
     x=z['physical'];j=z['joints'];root=z['root'];assert x.shape==(W*52,414) and digest(x)==a['physical_sha256'] and np.array_equal(root,x[:,:3])
     with torch.inference_mode():fk=rep.inverse(torch.tensor(x[-1:])[None],is_normalized=False)['posed_joints'][0,0].numpy().astype(np.float64)
     assert np.max(np.abs(fk-j[-1]))<2e-5
     roots.append(float(np.sqrt(np.square(root[-1,[0,2]].astype(np.float64)-np.asarray(c['root_goal_xz_m'])).sum())*100))
     ends.append(float(np.sqrt(np.square(fk[c['selected_joint_index']]-np.asarray(c['end_goal_xyz_m'])).sum())*100))
  assert abs(np.mean(roots)-score['root_CE_cm']['mean'])<1e-9 and abs(np.mean(ends)-score['end_CE_cm']['mean'])<.002
  with np.load(s/'FEATURES.npz') as z:features={k:z[k] for k in z.files}
  f=features['distribution'].astype(np.float64);f/=np.linalg.norm(f,axis=-1,keepdims=True)
  fids=[fid(q,real) for q in f];assert abs(np.mean(fids)-score['mean_window_L2_FID'])<1e-6
  # Independently recompute rankings against exactly each domain's own frozen candidate pools.
  catalog=rd(R/'scoring/REFERENCE_AND_POOLS.json');commands=list(rd(R/'PLAN.json')['prompts']);r3=[]
  for wi,command in enumerate(schedule(task)[0]):
   ranks=[]
   for head in [1701,2903]:
    text=bank[f'text_{head}'][commands.index(command)].astype(np.float64);text/=np.linalg.norm(text)
    motion=features[f'retrieval{head}'][wi].astype(np.float64);motion/=np.linalg.norm(motion,axis=-1,keepdims=True)
    b=bank[f'motion_{head}'].astype(np.float64);b/=np.linalg.norm(b,axis=-1,keepdims=True)
    pos=2-2*(motion@text)
    for ids in catalog['pools'][command].values():
     neg=2-2*(b[ids]@text);ranks.append(1+(neg[None,:]<=pos[:,None]).sum(1))
   r3.append(np.mean(np.asarray(ranks)<=3))
  assert abs(np.mean(r3)*100-score['R3_ref_percent']['mean'])<1e-9
  checks.append(dict(task=task,rollouts=384,root_CE=float(np.mean(roots)),end_CE=float(np.mean(ends)),mean_covariance_L2_FID=float(np.mean(fids)),R3_ref_percent=float(np.mean(r3)*100),geometry_input_FK_goal_bindings=True,paired_to_parent=True,latency_recomputed=True))
 wr(R/'SECOND_INDEPENDENT_AUDIT.json',dict(passed=True,rows=checks,assets=rd(R/'scoring/ASSET_BINDINGS.json'),source_sha256=sha(__file__),time=time.time()))
if __name__=='__main__':main()
