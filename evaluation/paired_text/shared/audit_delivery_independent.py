"""Read-only audit from raw saved geometry and a separate covariance FID formula."""
from pathlib import Path
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import json,csv,hashlib,time
import numpy as np
import torch
torch.set_num_threads(2)
R=Path(__file__).resolve().parent
def rd(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def norm(x):
 x=np.asarray(x,np.float64);n=np.linalg.norm(x,axis=-1,keepdims=True);assert np.isfinite(x).all() and (n>1e-12).all();return x/n
def covariance_fid(x,y):
 x=norm(x);y=norm(y);a=np.cov(x,rowvar=False);b=np.cov(y,rowvar=False)
 ev,u=np.linalg.eigh((a+a.T)/2);sq=(u*np.sqrt(np.maximum(ev,0)))@u.T;v=sq@b@sq;v=(v+v.T)/2
 score=np.square(x.mean(0)-y.mean(0)).sum()+np.trace(a)+np.trace(b)-2*np.sqrt(np.maximum(np.linalg.eigvalsh(v),0)).sum()
 return float(max(0.,score))
def main():
 start=time.time();D=R/'delivery';complete=rd(D/'COMPLETE.json');assert complete['passed']
 for n,h in complete['files'].items():assert sha(D/n)==h,n
 for n,h in rd(R/'FROZEN.json')['hashes'].items():assert sha(R/n)==h,n
 cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test'];assert len(cases)==128
 summary=list(csv.DictReader((D/'SUMMARY.csv').open()));assert len(summary)==52 and len({r['model'] for r in summary})==13
 with np.load(R/'scoring/REFERENCE_FEATURES.npz') as z:real=z['distribution'][z['quality_indices']].copy()
 registry=rd(R/'CHECKPOINTS.json');normalizer_bindings={}
 report=[];paired={};maximum={'CE_cm':0.,'command_percent':0.,'FID':0.,'latency_ms':0.};total_trials=0;windows=0
 schedules=rd(R/'PLAN.json')['tasks'];prompts=rd(R/'PLAN.json')['prompts']
 for row in summary:
  method,task=row['model'],row['task'];G=R/'models'/method/task;S=R/'scoring/models'/method/task;W=schedules[task]['windows'];stages=[k for k,n in schedules[task]['stages'] for _ in range(n)]
  assert rd(G/'GENERATION_COMPLETE.json')['rollouts']==384;cc=rd(S/'COMPLETE.json')
  for n,h in cc['files'].items():assert sha(S/n)==h,(method,task,n)
  gate=rd(G/'ACCELERATION_GATE.json');assert gate['passed'] and len(gate['rows'])==4 and all(all(v['checks'].values()) for v in gate['rows'])
  assert rd(G/'LOAD.json')['all_loaded_denoiser_weights_verified']
  statsdir=Path(registry[method]['stats_dir']);mean=torch.tensor(np.load(statsdir/'mean.npy'),dtype=torch.float32);std=torch.tensor(np.load(statsdir/'std.npy'),dtype=torch.float32)
  if len(mean)==418:
   ix=list(range(5))+list(range(9,418));mean=mean[ix];std=std[ix]
  assert mean.shape==std.shape==(414,);den=torch.sqrt(std**2+1e-5)
  normalizer_bindings[method]=dict(stats_dir=str(statsdir),mean_sha256=sha(statsdir/'mean.npy'),std_sha256=sha(statsdir/'std.npy'))
  trialrows=list(csv.DictReader((S/'PER_TRIAL.csv').open()));assert len(trialrows)==384;ces=[];ends=[];commands=[];seen=[]
  for ci,c in enumerate(cases):
   coordinates=c['goal_input_coordinate_indices'];sparse=torch.zeros(1,1,414);sparse[:,:,coordinates]=torch.tensor(c['goal_input_physical_values'],dtype=torch.float32);normalized=(sparse-mean)/den
   observed=torch.zeros(1,24+52*W,414);observed[:,-1,coordinates]=normalized[:,0,coordinates];expected_observed_sha=hashlib.sha256(observed.numpy().tobytes()).hexdigest()
   for si,seed in enumerate(c['noise_seeds']):
    p=G/'trials'/f'{ci:03d}_s{si}';a=rd(p/'INPUT_AND_EVENTS.json');co=rd(p/'COMPLETE.json');assert co['case_id']==c['case_id'] and co['seed']==seed
    assert sha(p/'geometry.npz')==co['geometry_sha256'] and sha(p/'INPUT_AND_EVENTS.json')==co['audit_sha256']
    assert a['normalized_observed_sha256']==expected_observed_sha,('own_checkpoint_stats_goal_normalization',method,task,ci)
    assert [e['prompt'] for e in a['events']]==[prompts[k] for k in stages] and all(e['target_motion_is_None'] for e in a['events'])
    if method not in ['dip','dart']:assert all(e['observed_scalars']==8 for e in a['events'])
    with np.load(p/'geometry.npz') as z:root=z['root'].astype(np.float64);j=z['joints'].astype(np.float64)
    assert root.shape==(52*W,3) and np.isfinite(root).all() and np.isfinite(j).all()
    ce=float(np.sqrt(sum((root[-1,k]-float(c['root_goal_xz_m'][q]))**2 for q,k in enumerate([0,2])))*100)
    end=float(np.sqrt(sum((j[-1,c['selected_joint_index'],k]-float(c['end_goal_xyz_m'][k]))**2 for k in range(3)))*100)
    tr=trialrows[ci*3+si];assert int(tr['seed'])==seed
    diff=max(abs(ce-float(tr['root_CE_cm'])),abs(end-float(tr['end_CE_cm'])));assert diff<1e-9;maximum['CE_cm']=max(maximum['CE_cm'],diff)
    # First root is centered at the last initial-history root XZ by construction.
    previous=np.array([0.,root[0,1],0.]);v=np.diff(np.concatenate([previous[None],root]),axis=0)[:,[0,2]]*25
    hip=j[:,8]-j[:,1];face=np.column_stack((hip[:,2],-hip[:,0]));face/=np.maximum(np.linalg.norm(face,axis=1,keepdims=True),1e-8);vf=(v*face).sum(1);speed=np.linalg.norm(v,axis=1)
    match=[]
    for wi,stage in enumerate(stages):
     lo=wi*52;hi=lo+52
     m=vf[lo:hi]>.1 if stage=='forward' else vf[lo:hi]<-.1 if stage=='backward' else speed[lo:hi]<.1 if stage=='stand' else speed[lo:hi]>.1
     match.append(float(m.mean()))
    command=float(np.mean(match)*100);assert abs(command-float(tr['command_matching_fraction'])*100)<1e-9
    commands.append(command);ces.append(ce);ends.append(end);seen.append((c['case_id'],seed,a['initial_sha256'],a['sparse_physical_goal_sha256'],a['mask_sha256']))
  if task in paired:assert seen==paired[task],('unpaired_inputs',method,task)
  paired[task]=seen
  for k,v in [('root_CE_cm',np.mean(ces)),('end_CE_cm',np.mean(ends)),('command_matching_percent',np.mean(commands))]:
   diff=abs(float(row[k])-v);assert diff<1e-9;maximum['command_percent' if k=='command_matching_percent' else 'CE_cm']=max(maximum['command_percent' if k=='command_matching_percent' else 'CE_cm'],diff)
  with np.load(S/'FEATURES.npz') as z:features=z['distribution'].copy()
  assert features.shape==(W,384,256)
  fid_values=[covariance_fid(f,real) for f in features];f_delta=abs(float(np.mean(fid_values))-float(row['mean_window_L2_FID']));assert f_delta<1e-6,('FID',method,task,f_delta);maximum['FID']=max(maximum['FID'],f_delta)
  L=R/'retiming'/method/task/'LATENCY.json';L=L if L.exists() else G/'LATENCY.json';lat=rd(L);assert lat['batch']==1 and lat['warmup_sequences']==2 and lat['timed_sequences']==32 and lat['windows']==W
  ms=np.array([a['elapsed_s'] for a in lat['rows']])*1000/W;diff=max(abs(float(ms.mean())-float(row['latency_mean_ms_window'])),abs(float(np.median(ms))-float(row['latency_median_ms_window'])));assert diff<1e-8;maximum['latency_ms']=max(maximum['latency_ms'],diff)
  if method in ['fd2','omg','cmdm','motionstreamer']:
   proof=rd(R/'retiming'/method/task/'COMPLETE.json');assert proof['passed'] and (proof.get('formal_geometry_and_RNG_exact') or proof.get('formal_geometry_and_RNG_unchanged'))
  total_trials+=384;windows+=384*W;report.append(dict(model=method,task=task,root_CE_cm=float(np.mean(ces)),end_CE_cm=float(np.mean(ends)),command_matching_percent=float(np.mean(commands)),L2_FID_covariance=float(np.mean(fid_values)),FID_delta=f_delta,latency_median_ms_window=float(np.median(ms))))
  (R/'INDEPENDENT_DELIVERY_AUDIT_LIVE.json').write_text(json.dumps(dict(completed=len(report),total=52,time=time.time()))+'\n');print('AUDITED',method,task,flush=True)
 x=dict(passed=True,conditions=len(report),checkpoints=13,rollouts=total_trials,accepted_windows=windows,raw_saved_geometry_hashes_checked=True,terminal_CE_independently_recomputed=True,body_relative_command_matching_independently_recomputed=True,normalized_FID_independent_covariance_eigendecomposition=True,latency_independently_recomputed_from_all32_samples=True,paired_physical_input_hashes_across_models_verified=True,each_checkpoint_own_normalized_goal_hash_independently_rebuilt_exact=True,generator_normalizer_bindings=normalizer_bindings,all_model_weight_load_and_four_end_gates_verified=True,maximum_discrepancies=maximum,rows=report,source_sha256=sha(__file__),elapsed_seconds=time.time()-start,time=time.time(),no_training_or_generation_changes=True)
 (R/'INDEPENDENT_DELIVERY_AUDIT.json').write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
