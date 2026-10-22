"""Freeze easier G1 control requests using only real source data, not scores.

CPU only. Does not generate motion, change training, or publish model metrics.
"""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,hashlib,math,collections
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'))
OLD=Path(artifact_path('stationary_short_qualification_v2_444fcfd331'))
REL=Path(artifact_path('humanml3d-g1-native-v2_65063e489a'))
SOURCE=Path(artifact_path('accepted_test_jsonl_5d9e5407c8'))
TRAIN=Path(artifact_path('data_index_jsonl_9966c1bc60'))
SEED=2026102103
BASE=Path(artifact_path('_runtime_924bb4d0b9'))
sys.path.insert(0,str(BASE))
import numpy as np,torch,re
torch.set_num_threads(2)
from motion_mamba2.skeleton import G1Skeleton34
from motion_mamba2.representation.reps.motion_representation import MotionMamba2MotionRep
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(8<<20),b''):h.update(a)
 return h.hexdigest()
def wr(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(a,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def lines(p,aa):Path(p).write_text(''.join(json.dumps(a,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n' for a in aa))
def key(x):return hashlib.sha256(f'{SEED}:{x}'.encode()).hexdigest()
def main():
 R.mkdir(exist_ok=True)
 assert not (R/'FROZEN.json').exists(),'Never silently replace frozen requests'
 sk=G1Skeleton34(load=False)
 sk.register_buffer('neutral_joints',torch.tensor(json.loads((REL/'SKELETON.json').read_text())['data'],dtype=torch.float32),persistent=False)
 rep=MotionMamba2MotionRep(sk,25)
 names=list(sk.bone_order_names)
 ends={k:names.index(v) for k,v in dict(left_wrist='left_wrist_yaw_skel',right_wrist='right_wrist_yaw_skel',left_ankle='left_ankle_roll_skel',right_ankle='right_ankle_roll_skel').items()}
 train=set()
 for s in TRAIN.open():
  a=json.loads(s)
  if a['dataset']=='humanml3d':train.add(a['source_family_id'])
 selected=[];seen=set();rejected=collections.Counter();input_rows=[]
 for s in SOURCE.open():
  a=json.loads(s)
  if a.get('dataset')!='humanml3d' or a.get('mirror_flag') or a['source_family_id'] in train or a['frames']<76:continue
  captions=[c['text'] for c in a.get('caption_time_spans',[]) if c['whole_motion']]
  if not any(re.search(r'\b(?:walk\w*|stand\w*|step\w*)\b',c,re.I) for c in captions):continue
  input_rows.append(a)
 input_rows.sort(key=lambda a:key(a['sample_id']))
 with torch.inference_mode():
  for a in input_rows:
   if a['source_family_id'] in seen:continue
   x=np.asarray(np.load(a['feature_path']),dtype=np.float32)
   assert x.shape==(a['frames'],414) and np.isfinite(x).all()
   # Exactly one deterministic crop is examined per family; no model used.
   start=int(key('crop:'+a['sample_id'])[:12],16)%(len(x)-76+1)
   prefix=np.array(x[start:start+24],copy=True)
   prefix[:,0]-=prefix[-1,0];prefix[:,2]-=prefix[-1,2]
   g=rep.inverse(torch.tensor(prefix)[None],is_normalized=False)
   j=g['posed_joints'][0].numpy();root=g['root_positions'][0].numpy()
   speed=np.linalg.norm(np.diff(root[:,[0,2]],axis=0),axis=-1)*25
   up=float(g['global_rot_mats'][0,-1,sk.root_idx,1,1])
   wrist_y=[float(j[-1,ends[s+'_wrist'],1]-root[-1,1]) for s in ['left','right']]
   ankle_y=[float(j[-1,ends[s+'_ankle'],1]) for s in ['left','right']]
   if not .65<=root[-1,1]<=1.0:rejected['root_height_outside_basic_upright']+=1;continue
   if up<.90:rejected['large_root_tilt']+=1;continue
   if np.max(speed)>1.0:rejected['fast_initial_motion']+=1;continue
   if max(wrist_y)>.10:rejected['initial_raised_arm']+=1;continue
   if max(ankle_y)>.12 or min(ankle_y)<-.05:rejected['initial_feet_not_low']+=1;continue
   if sha(a['feature_path'])!=a['features_hash']:raise AssertionError(('source_hash',a['sample_id']))
   seen.add(a['source_family_id'])
   selected.append(dict(id=a['sample_id'],source_id=a['source_family_id'],actor_uid=a['actor_id_namespaced'],
    feature_path=a['feature_path'],feature_sha256=a['features_hash'],crop_start=start,
    prefix=prefix,last_joints=j[-1],actual_root_height_m=float(root[-1,1]),
    initial_root_tilt_up_dot=up,initial_max_speed_m_s=float(np.max(speed))))
 assert len(selected)>=160,('Need128 locked test +32 development families',len(selected),dict(rejected))
 selected=selected[:160]
 # Split and goals are frozen before any generator is run. DEV cannot tune on FINAL.
 records=[];prefixes=[];poses=[];indices=[];values=[]
 for i,a in enumerate(selected):
  split='development' if i<32 else 'locked_test';n=i if i<32 else i-32
  end=list(ends)[n%4];joint=ends[end]
  distance=[.5,.75,1.0][(n//4)%3]
  angle_offset=[-15.,0.,15.][(n//12)%3]
  hips=a['last_joints'][sk.hip_joint_idx[0]]-a['last_joints'][sk.hip_joint_idx[1]]
  theta=math.atan2(float(hips[2]),-float(hips[0]))+math.radians(angle_offset)
  goal=np.asarray([math.sin(theta),math.cos(theta)],dtype=np.float32)*distance
  pose=a['prefix'][-1].copy();pose[[0,2]]=goal
  geom=rep.inverse(torch.tensor(pose)[None,None],is_normalized=False)
  end_xyz=geom['posed_joints'][0,0,joint].numpy()
  coords=list(range(rep.slice_dict['root_pos'].start,rep.slice_dict['root_pos'].stop))
  coords+=list(range(rep.slice_dict['global_root_heading'].start,rep.slice_dict['global_root_heading'].stop))
  offset=rep.slice_dict['local_joints_positions'].start+(joint-1)*3;coords+=list(range(offset,offset+3))
  assert len(coords)==8 and len(set(coords))==8
  rec={k:v for k,v in a.items() if k not in ['prefix','last_joints']}
  rec.update(case_id=f'easyv3:{split}:{n:03d}:{a["id"]}',split=split,selected_end_effector=end,
   selected_joint_index=joint,goal_distance_m=distance,goal_angle_offset_degrees=angle_offset,
   root_goal_xz_m=goal.tolist(),end_goal_xyz_m=end_xyz.tolist(),
   goal_input_coordinate_indices=coords,goal_input_physical_values=pose[coords].tolist(),
   root_height_input_m=float(pose[1]),goal_pose_source='last real initial-history pose translated in XZ only',
   noise_seeds=[int(key(f'noise:{a["id"]}:{s}')[:15],16) for s in [1701,2903,20261021]],
   requested_goal_groups=['root_XZ',end],additional_representation_input_fields=['root_Y','root_heading_cos_sin'],
   input_future_GT_trajectory_present=False)
  records.append(rec);prefixes.append(a['prefix']);poses.append(pose);indices.append(coords);values.append(pose[coords])
 # Input assets intentionally contain only the real prefix and8 sparse goal values.
 np.savez_compressed(R/'INPUT_ASSETS.npz',case_ids=np.asarray([r['case_id'] for r in records]),
  initial_physical=np.stack(prefixes),goal_coordinate_indices=np.asarray(indices),goal_physical_values=np.stack(values))
 np.savez_compressed(R/'SCORER_ONLY_TARGET_POSES.npz',case_ids=np.asarray([r['case_id'] for r in records]),physical=np.stack(poses))
 lines(R/'CASES.jsonl',records)
 registry=json.loads((OLD/'CHECKPOINTS.json').read_text())
 wr(R/'CHECKPOINTS.json',registry)
 tasks={
  'short52':dict(windows=1,stages=[['walk',1]]),
  'nominal9':dict(windows=4,stages=[['forward',1],['stand',1],['backward',1],['walk',1]]),
  'nominal20':dict(windows=9,stages=[['forward',3],['stand',1],['backward',2],['walk',3]]),
  'nominal40':dict(windows=19,stages=[['forward',6],['stand',2],['backward',5],['walk',6]])}
 for t,v in tasks.items():
  v.update(generated_frames=v['windows']*52,generated_seconds=v['windows']*52/25,
   total_seconds_including_initial_history=(v['windows']*52+24)/25,
   terminal_goal_window_phase=51,goal_at_generated_frame=v['windows']*52-1)
 plan=dict(protocol='G1_Basic_Continuous_Control_Easy_v3',version=3,created=time.time(),seed=SEED,
  purpose='One basic rollout measures endpoint control, simple command response, motion quality, and utility of historical state without an untrained verbal-recall task.',
  state='DESIGN_AND_CPU_INPUT_PREFLIGHT_ONLY_NO_NEW_GENERATOR_SCORES',
  primary_task='nominal20',duration_extension='nominal40',anchors=['short52','nominal9'],tasks=tasks,
  fps=25,initial_history_frames=24,window_frames=52,denoising_steps=10,cfg=[2.,2.],
  prompts={'forward':'Walk forward.','stand':'Stand still.','backward':'Walk backward.','walk':'Walk.'},
  instruction_delivery='Only the current command; no future command list; no repeat-first or same-hand rule.',
  goals={'root_distance_m':[.5,.75,1.0],'frontal_cone_degrees':[-15.,0.,15.],
   'end_effectors_per_case':1,'balanced_pool':list(ends),'target_pose':'valid low-arm real initial pose, no stretching or high kicks',
   'root_CE':'XZ distance at deadline only','end_CE':'selected joint 3D FK distance at deadline only',
   'same_goals_across_durations':True,'same_terminal_window_phase':51,'intermediate_path_penalty':False,
   'Root_ADE':False,'input_values_per_goal':8,'supporting_input_fields':['root_Y','root_heading_cos_sin']},
  cohort={'development_sources':32,'locked_test_sources':128,'seeds_per_source':3,
   'locked_rollouts_per_model_per_duration':384,'independent_test_families':128,
   'selection':'real source captions and initial GT geometry only, never generated scores',
   'training_family_overlap':0,'natural_long_reference_claim':False,
   'scope':'new composed G1 interactive tasks initialized from retargeted HumanML3D test; not canonical HumanML3D or natural40s clips'},
  model_roles={'primary_matched100k':['native100','cmdm','motionstreamer','fd2','omg','dart'],
   'retained_failed_training_diagnostics':['ardy','dip'],'separate_official_reference':['motion_mamba2_official'],
   'optional_finetuned_references':['native110','native117','native122','native122long'],
   'Minimal':'ablation only, not a main baseline'},
  metrics={'primary':['root_CE_cm','selected_end_effector_CE_cm','command_matching_frame_fraction','command_response_seconds'],
   'supplementary':['success_10cm','success_20cm','skate_and_contact','seam_velocity_jump','second_difference'],
   'direction':'actual FK hip-facing line; backward must have negative body-forward velocity; a turn followed by forward walking is not backward',
   'speed_threshold_m_s':.1,'response_confirmation_frames':10,'response_matching_frames_min':8,
   'success_thresholds_unchanged_across_models_and_durations':True,
   'FID':'per-vector L2 normalization, FP64, identical frozen HumanML3D-G1-native-v2 distribution head and fixed real reference bank; report independent-source counts and cluster uncertainty',
   'retrieval':'R@3_ref: one generated positive plus31 fixed, semantically incompatible real-motion distractors; same3 pools and two frozen retrieval heads; real positives calibrated before generator scoring',
   'retrieval_disclosure':'This anchored retrieval protocol differs from all-generated-pool R@3. Never silently replace the old scores or call it the canonical leaderboard.',
   'retrieval_exclusions':'exclude same-action positives from negatives using GT labels only, not generator similarities',
   'latency':'batch1,2 complete warmups,32 fixed timed inputs, admitted kernels only; report model+decode and text-update time separately'},
  history_utility={'same_task_and_cases':True,'checkpoint':'native100','conditions':['normal full persistent state','rebuild from last196 real accepted frames only'],
   'rebuild_contract':'replay accepted latents and their original text/coordinates without resampling; retain same current pose, decoder prefix, future targets and next-window noise; verify full-history replay reproduces carry before using truncated replay',
   'current_KV_or_decoder_not_arbitrarily_cleared':True,
   'controls':'apply matching reconstruction at early and late boundaries; compare full-replay control and ordinary current-command behavior',
   'reported_effects':['delta_CE','delta_command_matching_fraction','delta_R3_ref','delta_FID','effect_vs_window_number'],
   'interpretation':'utility of information earlier than local history; not proof of explicit episodic recall or understanding repeat-the-first-gesture'},
  qualification={'no_generated_score_threshold_for_model_admission':True,
   'shared_short_primitive_probe':'document execution of literal walking/backward/still commands; retain failures rather than choose different task lists for each model',
   'reference_metric_audit':'calibrate actual short action/reference pools on real GT; report reference performance rather than assign85% to generators',
   'new_mask_kernel_gate':'complete output,state,RNG equivalence for root+each single-end mask on this cohort before accelerated inference',
   'fixed_capacity':'log actual future goal visibility for DiP/DART; do not claim information equivalence if the goal is truncated'},
  reporting={'one_main_table':'one row per checkpoint;52f/9/20/40 groups;rootCE,endCE,command accuracy,R@3_ref,L2FID,latency',
   'curves':'window number on X;command correctness,feature quality,slip and seams on Y; mark command switches; do not call non-goal frames constraint error',
   'history_rows':'same protocol, Native full state and recent-history reconstruction as ablation rows',
   'no_composite_score_hiding_tradeoffs':True},
  training_changes=0,simulation_changes=0,existing_main_tex_unchanged=True)
 wr(R/'PLAN.json',plan)
 audit=dict(status='PASS_CPU_SOURCE_FAMILY_PREFIX_NEUTRAL_GOALS_AND_DEADLINE_PREFLIGHT',time=time.time(),
  independent_sources=len(records),development_sources=32,locked_test_sources=128,
  eligible_before_cap=len(seen),candidate_source_rows=len(input_rows),rejected=dict(rejected),
  actors_test=len({r['actor_uid'] for r in records if r['split']=='locked_test'}),
  end_effector_test_counts=dict(collections.Counter(r['selected_end_effector'] for r in records if r['split']=='locked_test')),
  distance_test_counts=dict(collections.Counter(str(r['goal_distance_m']) for r in records if r['split']=='locked_test')),
  source_family_overlap=0,finite_inputs=True,all_single_end_targets=True,
  no_hidden_future_trajectory_in_input_asset=True,target_root_height_range_m=[min(r['root_height_input_m'] for r in records),max(r['root_height_input_m'] for r in records)],
  target_coordinate_count=8,static_root_baseline_CE_cm_mean=float(np.mean([r['goal_distance_m']*100 for r in records if r['split']=='locked_test'])),
  generation_started=False,scoring_reference_pools_and_history_replay_implementation_not_yet_qualified=True)
 wr(R/'CPU_PREFLIGHT.json',audit)
 wr(R/'FROZEN.json',dict(status='PROTOCOL_AND_REQUESTS_FROZEN_BEFORE_GENERATION',time=time.time(),
  hashes={n:sha(R/n) for n in ['PLAN.json','CASES.jsonl','INPUT_ASSETS.npz','SCORER_ONLY_TARGET_POSES.npz','CHECKPOINTS.json','CPU_PREFLIGHT.json']},
  original_inputs={'test_manifest':sha(SOURCE),'training_index':sha(TRAIN),'skeleton':sha(REL/'SKELETON.json')}))
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
