"""Paired-caption pool32 scoring; GT motion is read only in this CPU scorer."""
from pathlib import Path
import os,sys,time,csv,argparse,fcntl,traceback,importlib.util,json,math
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from execute import R,P,rd,wr,sha,digest,schedule,verify
import numpy as np,torch
from prepare_scoring import REL,HEADROOT,HEADS,representation,canonical,verify_assets
from native_eval.v2_score import model_for
from native_eval.encoding import encode_arrays
from native_eval.metrics import fid,normalize,retrieval
spec=importlib.util.spec_from_file_location('paired_original_scoring_math',P/'score.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def csvwrite(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def score(method,task):
 torch.set_num_threads(2);torch.set_num_interop_threads(2);verify();verify_assets();out=R/'scoring/models'/method/task;out.mkdir(parents=True,exist_ok=True);lock=(out/'score.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'COMPLETE.json').exists():return
 assert rd(R/'PROTOCOL_FROZEN.json')['passed'];assert rd(R/'REAL_REFERENCE_CALIBRATION.json')['passed']
 d=R/'models'/method/task;assert rd(d/'GENERATION_COMPLETE.json')['rollouts']==384
 cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test'];inp=dict(np.load(R/'INPUT_ASSETS.npz'));lookup={v:i for i,v in enumerate(inp['case_ids'])};W=len(schedule(task)[0]);phases=rd(R/'PLAN.json')['paired_phases'][task]
 rep,sk=representation();stats=dict(np.load(REL/'stats.npz'));heads={'distribution':model_for(REL,'distribution','cpu'),'retrieval1701':model_for(HEADS[1701],'retrieval','cpu'),'retrieval2903':model_for(HEADS[2903],'retrieval','cpu')}
 trials=[];windowrows=[];physical=[];geobind=[];identities=[]
 for ci,c in enumerate(cases):
  stages,texts=schedule(task,c)
  for si,seed in enumerate(c['noise_seeds']):
   folder=d/'trials'/f'{ci:03d}_s{si}';complete=rd(folder/'COMPLETE.json');a=rd(folder/'INPUT_AND_EVENTS.json');assert sha(folder/'geometry.npz')==complete['geometry_sha256'] and sha(folder/'INPUT_AND_EVENTS.json')==complete['audit_sha256'] and complete['seed']==seed
   assert a['no_hidden_GT'] and a['no_paired_GT_motion_supplied_to_generator'] and a['detailed_text_pairing'] and len(a['events'])==W
   assert digest(inp['initial_physical'][lookup[c['case_id']]])==a['initial_sha256'];assert digest(inp['goal_physical_values'][lookup[c['case_id']]])==a['sparse_physical_goal_sha256']
   for wi,e in enumerate(a['events']):assert e['prompt']==texts[wi] and e['stage']==stages[wi] and e['target_motion_is_None']
   with np.load(folder/'geometry.npz') as z:x=z['physical'].copy();j=z['joints'].astype(np.float64);root=z['root'].astype(np.float64)
   assert x.shape==(W*52,414) and digest(x)==a['physical_sha256'] and np.array_equal(root,x[:,:3])
   with torch.inference_mode():g=rep.inverse(torch.tensor(x)[None],is_normalized=False)
   assert np.max(np.abs(g['posed_joints'][0].numpy().astype(np.float64)-j))<2e-5
   ce=float(np.linalg.norm(root[-1,[0,2]]-c['root_goal_xz_m'])*100);end=float(np.linalg.norm(j[-1,c['selected_joint_index']]-c['end_goal_xyz_m'])*100)
   row=dict(model=method,task=task,case=ci,source_family=c['source_id'],seed_index=si,seed=seed,end_effector=c['selected_end_effector'],root_CE_cm=ce,end_CE_cm=end,joint_success10=ce<=10 and end<=10,joint_success20=ce<=20 and end<=20);fractions=[]
   for wi,stage in enumerate(stages):
    lo=wi*52;hi=lo+52;prefix=inp['initial_physical'][lookup[c['case_id']]];prior=root[lo-1] if lo else prefix[-1,:3].astype(float);v=old.matching(root[lo:hi],j[lo:hi],stage,prior);fractions.append(v['matching_fraction'])
    ankles=[list(sk.bone_order_names).index(s+'_ankle_roll_skel') for s in ['left','right']];feet=j[lo:hi][:,ankles];fv=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=-1)*25;contact=(feet[:-1,:,1]<.10)&(feet[1:,:,1]<.10)
    windowrows.append(dict(model=method,task=task,case=ci,source_family=c['source_id'],seed_index=si,window=wi+1,phase=phases[wi],caption=texts[wi],coarse_GT_root_state=stage,coarse_state_matching_fraction=v['matching_fraction'],ankle_slide_proxy_percent=float((contact&(fv>.15)).any(1).mean()*100),ankle_contact_coverage=float(contact.any(1).mean()),seam_velocity_jump_m_s=float(np.linalg.norm((root[lo+1]-root[lo])-(root[lo-1]-root[lo-2]))*25) if wi else None,mean_joint_acceleration_m_s2=float(np.linalg.norm(np.diff(j[lo:hi],n=2,axis=0),axis=-1).mean()*625),goal_visible=a['events'][wi]['observed_scalars']==8))
   for phase in dict.fromkeys(phases):
    ids=[wi for wi,v in enumerate(phases) if v==phase];lo=min(ids)*52;hi=(max(ids)+1)*52;state=c['paired_instruction_root_states'][phase];v=old.matching(root[lo:hi],j[lo:hi],state,root[lo-1] if lo else prefix[-1,:3].astype(float));row[f'phase{phase}_root_state_response_s']=v['response_s']
   row['coarse_state_matching_fraction']=float(np.mean(fractions));row['all_windows_goal_visible']=all(v['observed_scalars']==8 for v in a['events']);trials.append(row);physical.extend([x[wi*52:(wi+1)*52] for wi in range(W)])
   identities.append(dict(case_id=c['case_id'],seed=seed,initial=a['initial_sha256'],goal=a['sparse_physical_goal_sha256'],paired_caption_ids=a['paired_caption_ids']));geobind.append(dict(case=ci,seed_index=si,geometry=complete['geometry_sha256'],input_audit=complete['audit_sha256']))
   if len(trials)%48==0:wr(out/'LIVE.json',dict(stage='SAVED_GEOMETRY_AUDIT',completed=len(trials),total=384,time=time.time()))
 features=old.features_for(physical,rep,stats,heads);del heads;physical=[]
 features={k:v.reshape(128,3,W,-1).transpose(2,0,1,3) for k,v in features.items()};np.savez_compressed(out/'FEATURES.npz',**features)
 bank=dict(np.load(R/'REAL_PAIRED_FEATURES.npz'));catalog_lookup={int(v):i for i,v in enumerate(bank['catalog_indices'])};pools=rd(R/'QUERY_POOLS.json');win=[];ranks_out={};R1=[];R2=[];R3=[];fids=[]
 for wi,phase in enumerate(phases):
  meta=pools[str(phase)];qi=meta['query_case_indices'];realids=[catalog_lookup[c['paired_reference_indices'][phase]] for c in cases];real=bank['distribution'][realids];qrealids=[realids[i] for i in qi]
  f=old.fid_independent(features['distribution'][wi].reshape(384,-1),real);ind=fid(normalize(features['distribution'][wi].reshape(384,-1)),normalize(real));assert abs(f-ind)<1e-6;fids.append(f);rr=[]
  for head in [1701,2903]:
   texts=bank['text'+str(head)][qrealids]
   for ps,pp in meta['pools'].items():
    for si in range(3):
     motion=features['retrieval'+str(head)][wi,qi,si];score,ranks=retrieval(texts,motion,pp);rr.append(ranks);ranks_out[f'w{wi}_head{head}_pool{ps}_noise{si}']=ranks
  ranks=np.stack(rr);vals=[float((ranks<=k).mean()*100) for k in [1,2,3]];R1.append(vals[0]);R2.append(vals[1]);R3.append(vals[2]);a=[v for v in windowrows if v['window']==wi+1]
  win.append(dict(model=method,task=task,window=wi+1,phase=phase,R1_percent=vals[0],R2_percent=vals[1],R3_percent=vals[2],distinct_caption_queries=len(qi),generated_noises=3,retrieval_heads=2,candidate_pools=3,candidates=32,coarse_root_state_match_percent=float(np.mean([v['coarse_state_matching_fraction'] for v in a])*100),L2_FID=f,unique_real_reference_families=len({c['paired_instruction_records'][phase]['source_family_id'] for c in cases}),ankle_slide_proxy_percent=float(np.mean([v['ankle_slide_proxy_percent'] for v in a])),ankle_contact_coverage=float(np.mean([v['ankle_contact_coverage'] for v in a])),seam_velocity_jump_m_s=float(np.mean([v['seam_velocity_jump_m_s'] for v in a])) if wi else None,mean_joint_acceleration_m_s2=float(np.mean([v['mean_joint_acceleration_m_s2'] for v in a])),goal_visible_fraction=float(np.mean([v['goal_visible'] for v in a]))))
  wr(out/'LIVE.json',dict(stage='PAIRED_POOL32_SCORING',completed=wi+1,total=W,time=time.time()))
 np.savez_compressed(out/'RETRIEVAL_RANKS.npz',**ranks_out)
 s=dict(model=method,task=task,rollouts=384,initial_source_families=128,root_CE_cm=old.ci([v['root_CE_cm'] for v in trials]),end_CE_cm=old.ci([v['end_CE_cm'] for v in trials]),coarse_root_state_match_percent=old.ci([v['coarse_state_matching_fraction']*100 for v in trials]),R1_percent=float(np.mean(R1)),R2_percent=float(np.mean(R2)),R3_percent=float(np.mean(R3)),retrieval_scope='mean over windows; each window uses its frozen distinct-caption query subset, original paired pool32, 2heads x3pools x3noises; no R3_ref',mean_window_L2_FID=float(np.mean(fids)),joint_success10_percent=float(np.mean([v['joint_success10'] for v in trials])*100),joint_success20_percent=float(np.mean([v['joint_success20'] for v in trials])*100),all_windows_goal_visible=all(v['all_windows_goal_visible'] for v in trials),latency=rd(d/'LATENCY.json'),time=time.time())
 csvwrite(out/'PER_TRIAL.csv',trials);csvwrite(out/'PER_WINDOW_TRIAL.csv',windowrows);csvwrite(out/'WINDOW_METRICS.csv',win);wr(out/'SCORES.json',s)
 wr(out/'INDEPENDENT_AUDIT.json',dict(passed=True,rollouts=384,windows=384*W,geometry_bindings=geobind,paired_inputs=identities,FK_recomputed=True,endpoint_FP64_recomputed=True,FID_two_formulas=True,strict32_paired_caption_retrieval=True,real_calibration=sha(R/'REAL_REFERENCE_CALIBRATION.json'),query_pools=sha(R/'QUERY_POOLS.json'),scoring_source=sha(__file__),time=time.time()))
 wr(out/'COMPLETE.json',dict(passed=True,model=method,task=task,files={n:sha(out/n) for n in ['SCORES.json','INDEPENDENT_AUDIT.json','FEATURES.npz','PER_TRIAL.csv','PER_WINDOW_TRIAL.csv','WINDOW_METRICS.csv','RETRIEVAL_RANKS.npz']},time=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--task',required=True);a=p.parse_args()
 try:score(a.model,a.task)
 except BaseException:
  wr(R/'scoring/models'/a.model/a.task/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
