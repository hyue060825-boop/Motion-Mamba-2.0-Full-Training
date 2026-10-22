"""Bounded read-only process audit: full replay, independent loop, actual conditioning."""
from pathlib import Path
import os,sys,time,json,inspect,traceback,csv,random,math
import numpy as np
from execute import R,rd,wr,sha,digest,setup,backend,run,schedule
O=R/'diagnosis/native122_process_20261021'
def rng_hash():
 import torch
 return __import__('hashlib').sha256(repr(random.getstate()).encode()+repr(np.random.get_state()).encode()+torch.get_rng_state().numpy().tobytes()+torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
def trajectory_row(m,c,root,joints,goal,window,stage,reference_root):
 import torch
 return dict(window=window,stage=stage,root_xz_m=root[-1,[0,2]].tolist(),goal_xz_m=goal.tolist(),distance_to_final_goal_m=float(np.linalg.norm(root[-1,[0,2]].astype(float)-goal)),min_root_height_m=float(root[:,1].min()),max_root_height_m=float(root[:,1].max()),root_direct_latent_max_error_m=float(reference_root),accepted_commits=c._accepted_commits,completed_windows=c._completed_windows,next_chunk=c._next_chunk_id)
def independent(method,model,m,client,case,task,seed,assets,features,trace=True,goal_shift_x=0.,remove_goal=False,only_first=False):
 import torch
 from motion_mamba2.models.motion_model import translate_normalized_root_motion
 from g1_motion_contract import decode_committed_window
 c=client.controller;stages,texts=schedule(task);W=len(stages);idx=assets['case_ids'].tolist().index(case['case_id'])
 initial=assets['initial_physical'][idx];hist=torch.tensor(initial,device='cuda',dtype=torch.float32)[None];prefix=m.motion_rep.normalize(hist)
 coords=assets['goal_coordinate_indices'][idx];values=assets['goal_physical_values'][idx].copy()
 values[list(coords).index(0)]+=goal_shift_x
 sparse=hist.new_zeros(1,1,414);sparse[:,:,coords]=torch.tensor(values,device='cuda');normalized=m.motion_rep.normalize(sparse)
 target=np.asarray(case['root_goal_xz_m'],float).copy();target[0]+=goal_shift_x
 tm=torch.ones(1,1,device='cuda',dtype=torch.bool);f={p:torch.tensor(v,device='cuda',dtype=torch.float32)[None,None] for p,v in features.items()}
 current={'wi':0};conditioning=[];prime=[];positions=[];decoded=[];normalized_windows=[];latents=[];noise=[]
 original_step=m.denoising_step;signature=c._signature
 def watch(*args,**kwargs):
  b=signature.bind(*args,**kwargs);b.apply_defaults();q=b.arguments;wi=current['wi'];mask=q['motion_mask'];obs=q['observed_motion']
  if trace:
   text_equal=torch.equal(q['text_feat'],f[texts[wi]]);assert text_equal,('stale_actual_denoising_text',wi)
   assert torch.isfinite(q['x']).all() and torch.isfinite(obs).all()
   nz=mask[0].nonzero();n=int(mask.count_nonzero());expected=0 if remove_goal else 8;assert n==expected,(wi,n)
   assert q.get('target_motion') is None
   if n:
    frame=52*(W-wi)-1;assert bool((nz[:,0]==frame).all()),('wrong_goal_deadline',wi,nz[:,0].tolist(),frame)
    assert sorted(nz[:,1].tolist())==sorted(coords.tolist()),('wrong_observed_coordinates',wi)
    actual=m.motion_rep.unnormalize(obs)[0,-1].detach().float().cpu().numpy()
    error=max(abs(float(actual[0])+float(translation[0,0])-target[0]),abs(float(actual[2])+float(translation[0,2])-target[1]))
    assert error<2e-5,('wrong_goal_world_rebase',wi,error)
    selected=[v for v in coords if v not in [0,2]];expected_vals={int(k):float(v) for k,v in zip(coords,values)}
    assert all(abs(float(actual[k])-expected_vals[k])<2e-5 for k in selected),('goal_height_heading_or_joint_changed',wi)
   else:error=0.
   conditioning.append(dict(window=wi+1,timestep=int(q['t'].max()),actual_text_exact=text_equal,goal_observed_scalars=n,goal_frame_local=52*(W-wi)-1,remaining_goal_seconds=(52*(W-wi))/25,goal_world_rebase_max_error_m=error,finite_latents=True))
  if int(q['t'].max())==9:noise.append(digest(q['x'][q['generation_token_mask']].detach().cpu().numpy()))
  return original_step(*args,**kwargs)
 m.denoising_step=watch
 import motion_mamba2.adapters.diffusion_memory as dm
 original_prime=dm.canonical_history_prime
 def prime_watch(raw,h,**kw):
  if trace:
   expected=f[texts[current['wi']]];assert torch.equal(kw['text_feat'],expected),'Accepted-history commit used stale text'
   prime.append(dict(window=current['wi']+1,history_tokens=h.shape[1],text_correct=True,chunk_id=kw['chunk_id'],synthetic_tail=kw['include_synthetic_generation']))
  return original_prime(raw,h,**kw)
 dm.canonical_history_prime=prime_watch
 random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 translation=torch.zeros(1,3,device='cuda');heading=torch.atan2(hist[:,-1,4],hist[:,-1,3])
 try:
  with torch.inference_mode():
   client.reset_memory();hp=torch.ones(1,24,device='cuda',dtype=torch.bool);hybrid,_=m.hybrid.get_hybrid_motion_from_explicit(prefix,hp.sum(1),hp)
   c.prime_accepted_history(hybrid,text_feat=f[texts[0]],text_pad_mask=tm,first_heading_angle=heading)
   previous=hybrid[:,-52//m.num_frames_per_token:]
   m.diffusion.calc_diffusion_vars(m.diffusion.space_timesteps(10)[0]);stepcount=torch.tensor([10],device='cuda')
   for wi in range(1 if only_first else W):
    current['wi']=wi;remaining=52*(W-wi)
    mask=hist.new_zeros(1,remaining,414);obs=mask.clone()
    if not remove_goal:mask[:,-1,coords]=1;obs[:,-1,coords]=normalized[:,0,coords]
    before=c._accepted_commits
    seq=m._generate_window(None,translation,0,0,remaining,f[texts[wi]],tm,heading,mask,obs,stepcount,(2.,2.),list(range(9,-1,-1)),progress_bar=lambda x:x,target_motion=None,cfg_type=None)
    seq,center,root=m._recenter_history(seq,torch.tensor([51],device='cuda'),requantize=True)
    heading=m.motion_rep.get_root_heading_angle(m.motion_rep.global_root_stats.unnormalize(root))[:,-1]
    m._sequence_kernel_post_window_hook(seq,center,root,heading,wi)
    assert c._accepted_commits==before+1 and c._completed_windows==wi+1,('wrong_commit_count',wi)
    translation=translation+center
    localroot,body=m.hybrid.get_root_and_latent_body_motion_from_hybrid(seq[:,-13:]);worldroot=translate_normalized_root_motion(localroot,translation,m.motion_rep)
    world=m.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(worldroot,body)
    out,previous=decode_committed_window(m,world,previous)
    physical=m.motion_rep.unnormalize(out).float().cpu().numpy()[0];g=m.motion_rep.inverse(out,is_normalized=True)
    roots=g['root_positions'].float().cpu().numpy()[0];joints=g['posed_joints'].float().cpu().numpy()[0]
    direct=m.motion_rep.global_root_stats.unnormalize(worldroot).float().cpu().numpy()[0,:,:3]
    maxdelta=float(np.max(np.abs(direct-roots)));assert maxdelta<2e-5,('world_root_decode_mismatch',wi,maxdelta)
    assert np.isfinite(physical).all() and np.isfinite(joints).all()
    decoded.append((physical,joints,roots));normalized_windows.append(out.detach().clone());latents.append(world.detach().cpu().numpy())
    positions.append(trajectory_row(m,c,roots,joints,target,wi+1,stages[wi],maxdelta))
 finally:
  m.denoising_step=original_step;dm.canonical_history_prime=original_prime
 result={key:np.concatenate([v[i] for v in decoded]) for i,key in enumerate(['physical','joints','root'])}
 # Match the formal FK batch shape exactly. Per52-window FK uses different
 # GPU matrix batch kernels than a full988-frame FK; do not relax equality.
 with torch.inference_mode():full_geometry=m.motion_rep.inverse(torch.cat(normalized_windows,dim=1),is_normalized=True)
 full_joints=full_geometry['posed_joints'].float().cpu().numpy()[0]
 fk_window_vs_full_max_delta_m=float(np.max(np.abs(full_joints-result['joints'])))
 result['joints']=full_joints
 return result,dict(conditioning=conditioning,clean_prime_calls=prime,windows=positions,initial_noise_hashes=noise,final_RNG=rng_hash(),per_window_vs_full_batch_FK_max_delta_m=fk_window_vs_full_max_delta_m,full_batch_FK_policy='same full accepted normalized sequence and shape as formal evaluation; strict bitwise comparison retained',final_root_CE_cm=float(np.linalg.norm(result['root'][-1,[0,2]].astype(float)-target)*100)),latents
def main():
 O.mkdir(parents=True,exist_ok=True)
 import fcntl,torch
 lock=(O/'audit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr(O/'PID.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),time=time.time()))
 d=R/'scoring/models/native122/nominal40';rows=list(csv.DictReader((d/'PER_TRIAL.csv').open()));ordered=sorted(rows,key=lambda v:float(v['root_CE_cm']))
 chosen=[('largest_error',ordered[-1]),('median_error',ordered[len(ordered)//2]),('smallest_error',ordered[0])]
 selections=[dict(role=role,case=int(row['case']),seed_index=int(row['seed_index']),stored_CE_cm=float(row['root_CE_cm'])) for role,row in chosen]
 wr(O/'SELECTION.json',dict(diagnostic_cases_only=True,selection='fixed extremes and middle of the existing saved122k40s errors, not a new formal cohort or model selection',rows=selections))
 assets=dict(np.load(R/'INPUT_ASSETS.npz'));features=dict(zip(rd(R/'TEXT_INDEX.json')['prompts'],np.load(R/'TEXT_FEATURES.npy')))
 cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test']
 model,m,client=setup('native122',O/'load');proofs=[]
 for item in selections:
  idx=item['case'];si=item['seed_index'];case=cases[idx];seed=case['noise_seeds'][si];dest=O/f'{idx:03d}_s{si}';dest.mkdir(exist_ok=True)
  formal=R/'models/native122/nominal40/trials'/f'{idx:03d}_s{si}';saved=dict(np.load(formal/'geometry.npz'));original=rd(formal/'INPUT_AND_EVENTS.json')
  backend('native122',m,client,False);ref,ra=run('native122',model,m,client,case,'nominal40',seed,assets,features,gate=True)
  checks={k:bool(np.array_equal(ref[k],saved[k])) for k in ref};checks['RNG']=ra['final_RNG']==original['final_RNG'];assert all(checks.values()),('reference_replay_differs_from_saved',item,checks)
  wr(dest/'REFERENCE_REPLAY.json',dict(passed=True,checks=checks,state_RNG=ra['per_window_state_RNG']))
  backend('native122',m,client,True);alt,trace,lat=independent('native122',model,m,client,case,'nominal40',seed,assets,features)
  checks2={k:bool(np.array_equal(alt[k],saved[k])) for k in alt};checks2['RNG']=trace['final_RNG']==original['final_RNG'];assert all(checks2.values()),('independent_loop_changed_saved_result',item,checks2)
  wr(dest/'ACTUAL_CONDITIONING_AND_WINDOW_TRACE.json',trace)
  wr(dest/'INDEPENDENT_LOOP.json',dict(passed=True,checks=checks2,final_CE_cm=trace['final_root_CE_cm']))
  proofs.append(dict(**item,reference_replay_exact=True,independent_step_loop_exact=True,final_CE_cm=trace['final_root_CE_cm'],conditioned_denoising_calls=len(trace['conditioning']),clean_history_primes=len(trace['clean_prime_calls'])))
  wr(O/'LIVE.json',dict(stage='ACTUAL_REPLAY_AND_CONDITIONING_AUDIT',completed=len(proofs),total=3,time=time.time()))
  print('REPLAY_PASS',item,flush=True)
 # Causal visibility check at the first window of the worst trajectory.
 item=selections[0];case=cases[item['case']];seed=case['noise_seeds'][item['seed_index']];backend('native122',m,client,False)
 base,_,_=independent('native122',model,m,client,case,'nominal40',seed,assets,features,only_first=True)
 shifted,_,_=independent('native122',model,m,client,case,'nominal40',seed,assets,features,goal_shift_x=1.,only_first=True)
 removed,_,_=independent('native122',model,m,client,case,'nominal40',seed,assets,features,remove_goal=True,only_first=True)
 sensitivity=dict(goal_shift1m_root_max_difference_m=float(np.max(np.abs(shifted['root']-base['root']))),goal_removal_root_max_difference_m=float(np.max(np.abs(removed['root']-base['root']))),same_weights_seed_history_text=True,scope='first52frames while the target is still39.52s away; positive difference shows early target influences output, not that it is planned successfully')
 assert sensitivity['goal_shift1m_root_max_difference_m']>0 and sensitivity['goal_removal_root_max_difference_m']>0,('target_has_no_early_effect',sensitivity)
 wr(O/'EARLY_GOAL_SENSITIVITY.json',sensitivity)
 wr(O/'REPLAY_COMPLETE.json',dict(passed=True,proofs=proofs,sensitivity=sensitivity,no_formal_scores_changed=True,source_sha256=sha(__file__),time=time.time()))
 print('PROCESS_REPLAY_AUDIT_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:wr(O/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
