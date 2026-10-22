"""Correct-domain scoring of saved easy-v3 motion; no new generation/training."""
from pathlib import Path
import os,sys,time,csv,math,fcntl,traceback,argparse
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from execute import R,MODELS,rd,wr,sha,digest,schedule
import numpy as np,torch
from prepare_scoring import REL,HEADROOT,HEADS,representation,canonical
from native_eval.v2_score import model_for
from native_eval.encoding import encode_arrays
from native_eval.metrics import fid
def norm(a):
 a=np.asarray(a,np.float64);n=np.linalg.norm(a,axis=-1,keepdims=True)
 assert np.isfinite(a).all() and (n>1e-12).all()
 return a/n
def fid_independent(a,b):
 a=norm(a);b=norm(b);x=(a-a.mean(0))/np.sqrt(len(a)-1);y=(b-b.mean(0))/np.sqrt(len(b)-1)
 val=float(np.square(a.mean(0)-b.mean(0)).sum()+np.square(x).sum()+np.square(y).sum()-2*np.linalg.svd(x@y.T,compute_uv=False).sum())
 return max(0.,val)
def ci(x,seed=20261021,repeats=1000):
 x=np.asarray(x).reshape(128,3).mean(1);rng=np.random.default_rng(seed)
 boot=x[rng.integers(0,128,(repeats,128))].mean(1)
 return dict(mean=float(x.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),independent_families=128,noise_seeds_per_family=3,repeats=repeats)
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def matching(root,joints,stage,previous_root=None):
 prev=root[:1] if previous_root is None else previous_root[None]
 vel=np.diff(np.concatenate((prev,root),axis=0),axis=0)[:,[0,2]]*25
 hip=joints[:,8]-joints[:,1];f=np.stack((hip[:,2],-hip[:,0]),1);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
 vf=(vel*f).sum(1);speed=np.linalg.norm(vel,axis=1)
 match=(vf>.1) if stage=='forward' else (vf<-.1) if stage=='backward' else (speed<.1) if stage=='stand' else (speed>.1)
 blocks=np.convolve(match.astype(float),np.ones(10),mode='valid')>=8
 valid=np.flatnonzero(blocks)
 return dict(matching_fraction=float(match.mean()),response_s=float(valid[0]/25) if len(valid) else None,mean_forward_speed_m_s=float(vf.mean()),mean_planar_speed_m_s=float(speed.mean()))
def features_for(physical,rep,stats,heads):
 # Physical channels are recomputed from the actual accepted rotations and root.
 rows=[dict(sample_id=f'v3:{i:05d}') for i in range(len(physical))]
 arrays=[]
 for s in range(0,len(physical),32):
  a,_=canonical(rep,physical[s:s+32]);arrays.extend(list(a))
 out={}
 for label,model in heads.items():
  h,m,_=encode_arrays(model,rows,arrays,stats if label=='distribution' else dict(np.load(HEADROOT/'retrieval_stats.npz')));out[label]=h if label=='distribution' else m
 return out
def score(method,task):
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 out=R/'scoring'/'models'/method/task;out.mkdir(parents=True,exist_ok=True)
 lock=(out/'score.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'COMPLETE.json').exists():return
 d=R/'models'/method/task;assert rd(d/'GENERATION_COMPLETE.json')['rollouts']==384
 refroot=R/'scoring';assert rd(refroot/'REFERENCE_COMPLETE.json')['passed'];bindings=rd(refroot/'ASSET_BINDINGS.json')
 assert sha(REL/'distribution.pt')==bindings['distribution'] and sha(REL/'stats.npz')==bindings['stats']
 assert sha(HEADS[1701]/'retrieval.pt')==bindings['retrieval']['1701']
 assert sha(HEADS[2903]/'retrieval.pt')==bindings['retrieval']['2903']
 assert sha(HEADROOT/'retrieval_stats.npz')==bindings['retrieval_stats']
 assert sha(refroot/'REFERENCE_FEATURES.npz')==bindings['reference'] and sha(refroot/'REFERENCE_AND_POOLS.json')==bindings['pools']
 cases=[__import__('json').loads(s) for s in (R/'CASES.jsonl').open() if __import__('json').loads(s)['split']=='locked_test'];inp=dict(np.load(R/'INPUT_ASSETS.npz'));case_lookup={v:i for i,v in enumerate(inp['case_ids'])}
 stages,texts=schedule(task);W=len(stages);rep,sk=representation();stats=dict(np.load(REL/'stats.npz'))
 heads={'distribution':model_for(REL,'distribution','cpu'),'retrieval1701':model_for(HEADS[1701],'retrieval','cpu'),'retrieval2903':model_for(HEADS[2903],'retrieval','cpu')}
 trials=[];windowrows=[];geobind=[];physical=[];identity_rows=[]
 for ci_,c in enumerate(cases):
  for si,seed in enumerate(c['noise_seeds']):
   folder=d/'trials'/f'{ci_:03d}_s{si}';comp=rd(folder/'COMPLETE.json');a=rd(folder/'INPUT_AND_EVENTS.json')
   assert comp['case_id']==c['case_id'] and comp['seed']==seed and sha(folder/'geometry.npz')==comp['geometry_sha256'] and sha(folder/'INPUT_AND_EVENTS.json')==comp['audit_sha256']
   assert a['no_hidden_GT'] and a['noise_seed']==seed and len(a['events'])==W
   assert all(e['stage']==stages[i] and e['prompt']==texts[i] and e['target_motion_is_None'] for i,e in enumerate(a['events']))
   prefix=inp['initial_physical'][case_lookup[c['case_id']]];assert digest(prefix)==a['initial_sha256']
   assert digest(inp['goal_physical_values'][case_lookup[c['case_id']]])==a['sparse_physical_goal_sha256']
   with np.load(folder/'geometry.npz') as z:
    x=z['physical'].copy();j=z['joints'].astype(float);root=z['root'].astype(float)
   assert x.shape==(52*W,414) and digest(x)==a['physical_sha256'] and np.array_equal(root,x[:,:3])
   # Independent frozen FK validation against stored scored joints, allowing only FP32 arithmetic roundoff.
   with torch.inference_mode():g=rep.inverse(torch.tensor(x)[None],is_normalized=False)
   delta=float(np.max(np.abs(g['posed_joints'][0].numpy().astype(float)-j)))
   # Different frozen runtime implementations must use identical skeleton/rotation geometry.
   assert delta<2e-5,('FK_geometry_mismatch',method,delta)
   dif=root[-1,[0,2]]-c['root_goal_xz_m'];ce=float(np.linalg.norm(dif)*100)
   assert abs(ce-math.sqrt(sum(float(q*q) for q in dif))*100)<1e-9
   dif=j[-1,c['selected_joint_index']]-c['end_goal_xyz_m'];end=float(np.linalg.norm(dif)*100)
   assert abs(end-math.sqrt(sum(float(q*q) for q in dif))*100)<1e-9
   row=dict(model=method,task=task,case=ci_,source_family=c['source_id'],seed_index=si,seed=seed,end_effector=c['selected_end_effector'],root_CE_cm=ce,end_CE_cm=end,joint_success10=ce<=10 and end<=10,joint_success20=ce<=20 and end<=20)
   fractions=[]
   for wi,stage in enumerate(stages):
    lo=wi*52;hi=lo+52;prior=root[lo-1] if lo else prefix[-1,:3].astype(float)
    v=matching(root[lo:hi],j[lo:hi],stage,prior);fractions.append(v['matching_fraction'])
    ankles=[list(sk.bone_order_names).index(k+'_ankle_roll_skel') for k in ['left','right']];feet=j[lo:hi][:,ankles]
    fv=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=-1)*25;contact=(feet[:-1,:,1]<.10)&(feet[1:,:,1]<.10)
    slide=(contact&(fv>.15)).any(1)
    seam=float(np.linalg.norm((root[lo+1]-root[lo])-(root[lo-1]-root[lo-2]))*25) if wi else None
    windowrows.append(dict(model=method,task=task,case=ci_,seed_index=si,window=wi+1,command=stage,matching_fraction=v['matching_fraction'],mean_forward_speed_m_s=v['mean_forward_speed_m_s'],mean_planar_speed_m_s=v['mean_planar_speed_m_s'],ankle_contact_coverage=float(contact.any(1).mean()),ankle_slide_proxy_percent=float(slide.mean()*100),seam_velocity_jump_m_s=seam,mean_joint_acceleration_m_s2=float(np.linalg.norm(np.diff(j[lo:hi],n=2,axis=0),axis=-1).mean()*625),goal_visible=a['events'][wi]['observed_scalars']==8))
   row['command_matching_fraction']=float(np.mean(fractions))
   for stage in dict.fromkeys(stages):
    ids=[i for i,s in enumerate(stages) if s==stage];lo=min(ids)*52;hi=(max(ids)+1)*52
    v=matching(root[lo:hi],j[lo:hi],stage,root[lo-1] if lo else prefix[-1,:3].astype(float))
    row[stage+'_response_s']=v['response_s'];row[stage+'_response_observed']=v['response_s'] is not None
   row['all_windows_goal_visible']=all(e['observed_scalars']==8 for e in a['events']);trials.append(row)
   physical.extend([x[wi*52:(wi+1)*52].copy() for wi in range(W)])
   identity_rows.append(dict(case_id=c['case_id'],seed=seed,initial_sha256=a['initial_sha256'],physical_sparse_goal_sha256=a['sparse_physical_goal_sha256']))
   geobind.append(dict(case=ci_,seed_index=si,geometry=comp['geometry_sha256'],input_audit=comp['audit_sha256']))
   if len(trials)%48==0:wr(out/'LIVE.json',dict(stage='INDEPENDENT_GEOMETRY_AUDIT',completed=len(trials),total=384,time=time.time()))
 feat=features_for(physical,rep,stats,heads);del heads;physical=[]
 # Trial-major windows; restore to W x384 xD.
 feat={k:v.reshape(384,W,-1).transpose(1,0,2) for k,v in feat.items()}
 np.savez_compressed(out/'FEATURES.npz',**feat)
 bank=dict(np.load(refroot/'REFERENCE_FEATURES.npz'));catalog=rd(refroot/'REFERENCE_AND_POOLS.json');commands=list(rd(R/'PLAN.json')['prompts'])
 real=bank['distribution'][bank['quality_indices']];balanced=bank['distribution'][bank['balanced_quality_indices']]
 win=[];r3all=[];fids=[]
 for wi,command in enumerate(stages):
  rankings=[]
  for seed in [1701,2903]:
   tg=norm(bank[f'text_{seed}'][commands.index(command)][None])[0];mg=norm(feat[f'retrieval{seed}'][wi]);mr=norm(bank[f'motion_{seed}'])
   dp=2-2*mg@tg
   for ids in catalog['pools'][command].values():
    dn=2-2*mr[ids]@tg;rankings.append(1+(dn[None,:]<=dp[:,None]).sum(1))
  ranks=np.stack(rankings);r3=(ranks<=3).mean(0);r3all.append(r3)
  f=fid_independent(feat['distribution'][wi],real);check=fid(norm(feat['distribution'][wi]),norm(real));assert abs(f-check)<1e-6,(f,check)
  fids.append(f)
  # Family-cluster interval: keep all3 noise seeds together, fixed real bank.
  rng=np.random.default_rng(20261021+wi);boot=[]
  for _ in range(128):
   ids=(rng.integers(0,128,128)[:,None]*3+np.arange(3)[None,:]).reshape(-1)
   boot.append(fid_independent(feat['distribution'][wi,ids],real))
  wr(out/'LIVE.json',dict(stage='WINDOW_FEATURE_SCORING',completed=wi+1,total=W,time=time.time()))
  a=[v for v in windowrows if v['window']==wi+1]
  win.append(dict(model=method,task=task,window=wi+1,command=command,L2_FID=f,FID_family_ci95_low=float(np.quantile(boot,.025)),FID_family_ci95_high=float(np.quantile(boot,.975)),L2_FID_balanced_reference_sensitivity=fid_independent(feat['distribution'][wi],balanced),R3_ref_percent=float(r3.mean()*100),R3_ref_ci95_low=ci(r3*100)['ci95'][0],R3_ref_ci95_high=ci(r3*100)['ci95'][1],command_matching_percent=float(np.mean([r['matching_fraction'] for r in a])*100),ankle_slide_proxy_percent=float(np.mean([r['ankle_slide_proxy_percent'] for r in a])),ankle_contact_coverage=float(np.mean([r['ankle_contact_coverage'] for r in a])),seam_velocity_jump_m_s=float(np.mean([r['seam_velocity_jump_m_s'] for r in a])) if wi else None,mean_joint_acceleration_m_s2=float(np.mean([r['mean_joint_acceleration_m_s2'] for r in a])),goal_visible_fraction=float(np.mean([r['goal_visible'] for r in a]))))
  for i,v in enumerate(a):v['R3_ref']=float(r3[i])
 np.savez_compressed(out/'RETRIEVAL_PER_TRIAL.npz',R3_ref=np.stack(r3all))
 score=dict(model=method,task=task,rollouts=384,independent_families=128,root_CE_cm=ci([v['root_CE_cm'] for v in trials]),end_CE_cm=ci([v['end_CE_cm'] for v in trials]),command_matching_percent=ci([v['command_matching_fraction']*100 for v in trials]),R3_ref_percent=ci(np.stack(r3all).mean(0)*100),mean_window_L2_FID=float(np.mean(fids)),terminal52_L2_FID=fids[-1],root_CE_median_cm=float(np.median([v['root_CE_cm'] for v in trials])),end_CE_median_cm=float(np.median([v['end_CE_cm'] for v in trials])),joint_success10_percent=float(np.mean([v['joint_success10'] for v in trials])*100),joint_success20_percent=float(np.mean([v['joint_success20'] for v in trials])*100),latency=rd(d/'LATENCY.json'),all_windows_goal_visible=all(v['all_windows_goal_visible'] for v in trials),R3_ref_status='new anchored diagnostic, calibrated real references differ strongly by command; primary command evidence is independent FK geometry',FID_scope='short52 physical distribution head, one frozen real basic-motion reference bank; not a long-sequence FID',history_ablation='not yet executed; baseline carry scores do not prove long-term memory',time=time.time())
 csvwrite(out/'PER_TRIAL.csv',trials);csvwrite(out/'PER_WINDOW_TRIAL.csv',windowrows);csvwrite(out/'WINDOW_METRICS.csv',win)
 wr(out/'SCORES.json',score)
 wr(out/'INDEPENDENT_AUDIT.json',dict(passed=True,rollouts=384,windows=384*W,FK_from_rotations_independently_checked=True,FP64_endpoint_CE_independently_recomputed=True,normalized_FID_two_independent_formula_checks=True,source_family_cluster_intervals=True,paired_inputs=identity_rows,geometry_bindings=geobind,assets=bindings,scoring_source_sha256=sha(__file__),time=time.time()))
 wr(out/'COMPLETE.json',dict(passed=True,model=method,task=task,assets=bindings,files={n:sha(out/n) for n in ['SCORES.json','INDEPENDENT_AUDIT.json','FEATURES.npz','PER_TRIAL.csv','PER_WINDOW_TRIAL.csv','WINDOW_METRICS.csv','RETRIEVAL_PER_TRIAL.npz']},time=time.time()))
 print('SCORED',method,task,score['root_CE_cm']['mean'],score['end_CE_cm']['mean'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--task',required=True);a=ap.parse_args()
 try:score(a.model,a.task)
 except BaseException:wr(R/'scoring'/'models'/a.model/a.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
