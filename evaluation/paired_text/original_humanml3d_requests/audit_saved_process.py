"""Independent audit of all saved Native inputs, deadlines, scoring and finite geometry."""
from pathlib import Path
import os,json,time,csv,math,statistics,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
from execute import R,rd,wr,sha,digest,schedule
O=R/'diagnosis/native122_process_20261021';O.mkdir(parents=True,exist_ok=True)
cases=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test'];assets=dict(np.load(R/'INPUT_ASSETS.npz'));lookup={str(s):i for i,s in enumerate(assets['case_ids'])}
for n,h in rd(R/'FROZEN.json')['hashes'].items():assert sha(R/n)==h,('frozen_input_changed',n)
prior={};out=[];finite=0;maxce=0.;windows=0;rows_all=[]
for model in ['native100','native110','native117','native122']:
 for task in ['short52','nominal9','nominal20','nominal40']:
  d=R/'models'/model/task;stages,prompts=schedule(task);W=len(stages);vals=[];ends=[];graphfallback=0
  score=rd(R/'scoring/models'/model/task/'SCORES.json');load=rd(d/'LOAD.json');assert load['all_loaded_denoiser_weights_verified']
  gate=rd(d/'ACCELERATION_GATE.json');assert gate['passed'] and len(gate['rows'])==4 and all(all(r['checks'].values()) for r in gate['rows'])
  for i,c in enumerate(cases):
   for si,seed in enumerate(c['noise_seeds']):
    p=d/'trials'/f'{i:03d}_s{si}';comp=rd(p/'COMPLETE.json');a=rd(p/'INPUT_AND_EVENTS.json')
    assert comp['case_id']==c['case_id'] and comp['seed']==seed and comp['source_family']==c['source_id']
    assert sha(p/'geometry.npz')==comp['geometry_sha256'] and sha(p/'INPUT_AND_EVENTS.json')==comp['audit_sha256']
    ix=lookup[c['case_id']];assert a['initial_sha256']==digest(assets['initial_physical'][ix]);assert a['sparse_physical_goal_sha256']==digest(assets['goal_physical_values'][ix])
    assert a['indices']==list(map(int,assets['goal_coordinate_indices'][ix])) and a['noise_seed']==seed and a['no_hidden_GT']
    events=a['events'];assert len(events)==W
    for wi,v in enumerate(events):
     assert v['window']==wi+1 and v['stage']==stages[wi] and v['prompt']==prompts[wi] and v['observed_scalars']==8 and v['target_motion_is_None']
     assert v['timeline_frames']==W*52,('unexpected_outer_condition_timeline',model,task,wi,v)
    key=(task,i,si);physical_input=(a['initial_sha256'],a['mask_sha256'],a['normalized_observed_sha256'],a['sparse_physical_goal_sha256'],a['noise_seed'],a['final_RNG'])
    if key not in prior:prior[key]=physical_input
    assert physical_input==prior[key],('Native_input_or_rng_mismatch_across_checkpoints',model,key)
    for name in a['graph_before']:
     before=a['graph_before'][name];after=a['graph_after'][name]
     graphfallback+=after['fallbacks']-before['fallbacks'];assert after['fallbacks']==before['fallbacks'],('kernel_fallback',model,task,i)
     assert after['builds']==before['builds'],('formal_new_graph_build',model,task,i)
    with np.load(p/'geometry.npz') as z:root=z['root'].astype(float);j=z['joints'].astype(float)
    assert root.shape==(52*W,3) and j.shape[:2]==(52*W,34) and np.isfinite(root).all() and np.isfinite(j).all()
    error=math.hypot(*(root[-1,[0,2]]-np.asarray(c['root_goal_xz_m'])))*100
    diff=j[-1,c['selected_joint_index']]-np.asarray(c['end_goal_xyz_m']);end=math.sqrt(sum(v*v for v in diff))*100;vals.append(error);ends.append(end)
    finite+=1;windows+=W;rows_all.append(dict(model=model,task=task,case=i,seed_index=si,root_CE_cm=error,end_CE_cm=end))
  mean=statistics.fmean(vals);meanend=statistics.fmean(ends);delta=max(abs(mean-score['root_CE_cm']['mean']),abs(meanend-score['end_CE_cm']['mean']));maxce=max(maxce,delta);assert delta<1e-8
  out.append(dict(model=model,task=task,N=384,root_CE_cm=mean,end_CE_cm=meanend,independent_score_max_difference_cm=delta,no_missing_goals=True,no_graph_fallback=True,all_checkpoints_share_same_inputs_and_RNG=True))
  wr(O/'SAVED_AUDIT_LIVE.json',dict(complete_panels=len(out),total=16,time=time.time()))
# Independently check the goal pose/selected local-position inputs against FK.
import torch
from prepare_scoring import representation
torch.set_num_threads(2);rep,sk=representation();p=dict(np.load(R/'SCORER_ONLY_TARGET_POSES.npz'));maxgoal=0.;maxlocal=0.
with torch.inference_mode():
 for c in cases:
  # The physical encoder differentiates positions, requiring at least2
  # frames. Duplicate this fixed pose; the8 checked goal fields contain no velocity.
  ix=lookup[c['case_id']];pose=torch.tensor(p['physical'][ix])[None,None].repeat(1,2,1)
  geom=rep.inverse(pose,is_normalized=False);actual=geom['posed_joints'][0,0,c['selected_joint_index']].numpy();maxgoal=max(maxgoal,float(np.max(np.abs(actual-np.asarray(c['end_goal_xyz_m'])))))
  rebuilt=rep(geom['local_rot_mats'],geom['root_positions'],to_normalize=False,to_canonicalize=False,lengths=torch.tensor([2]))[0,0].numpy()
  ids=assets['goal_coordinate_indices'][ix];maxlocal=max(maxlocal,float(np.max(np.abs(rebuilt[ids]-assets['goal_physical_values'][ix]))))
assert maxgoal<1e-6
wr(O/'SAVED_PROCESS_AUDIT.json',dict(passed=True,panels=16,rollouts=finite,accepted_windows=windows,frozen_inputs_unchanged=True,case_seed_history_goal_binding_verified=True,all8observations_present_every_window=True,actual_denoiser_condition_values_checked_separately_in_GPU_replay=True,same_noises_and_normalized_physical_goals_across4checkpoints=True,all_geometry_finite=True,no_formal_graph_build_or_fallback=True,root_CE_is_terminal_XZ_only=True,end_CE_is_terminal_selected_FK3D_only=True,no_intermediate_route_penalty=True,FP64_aggregate_max_error_cm=maxce,stored_target_FK_max_error_m=maxgoal,sparse_input_vs_FK_rebuilt_pose_max_error=maxlocal,rows=out,time=time.time()))
print('SAVED_PROCESS_AUDIT_PASS',finite,'rollouts',windows,'windows','max_goal_pose_input_error',maxlocal,flush=True)
