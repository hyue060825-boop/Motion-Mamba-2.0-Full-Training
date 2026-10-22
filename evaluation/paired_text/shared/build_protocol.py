"""Freeze HiPHI source histories before any generator is run. CPU only."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,hashlib,math,collections,shutil
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
R=Path(artifact_path('unified_interactive_easy_v3_hiphi_20261021_32d9db1045'))
PARENT=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'))
REL=Path(artifact_path('bones-g1-native-v2_adcc182e95'))
HEADROOT=Path(artifact_path('hiphi_window_head_de5d446e81'))
MAN=Path(artifact_path('manifests_8245329bf1'))
TRAIN=Path(artifact_path('data_index_jsonl_9966c1bc60'))
BASE=Path(artifact_path('_runtime_924bb4d0b9'))
sys.path.insert(0,str(BASE))
import numpy as np,torch
from motion_mamba2.skeleton import G1Skeleton34
from motion_mamba2.representation.reps.motion_representation import MotionMamba2MotionRep
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def wr(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def key(s):return hashlib.sha256(f'2026102103:{s}'.encode()).hexdigest()
def main():
 torch.set_num_threads(2);R.mkdir(exist_ok=True)
 assert not (R/'FROZEN.json').exists(),'Never replace frozen cases'
 sk=G1Skeleton34(load=False);sk.register_buffer('neutral_joints',torch.tensor(rd(REL/'SKELETON.json')['data'],dtype=torch.float32),persistent=False);rep=MotionMamba2MotionRep(sk,25)
 names=list(sk.bone_order_names);ends={k:names.index(v) for k,v in dict(left_wrist='left_wrist_yaw_skel',right_wrist='right_wrist_yaw_skel',left_ankle='left_ankle_roll_skel',right_ankle='right_ankle_roll_skel').items()}
 train={json.loads(s)['source_family_id'] for s in TRAIN.open()};head_train_dev=set()
 for n in ['train.jsonl','development.jsonl']:
  head_train_dev|={json.loads(s)['source_family_id'] for s in (HEADROOT/n).open()}
 chosen=[];counts={};rejects={}
 for split,number in [('val',32),('test',128)]:
  raw=[json.loads(s) for s in (MAN/f'accepted_{split}.jsonl').open()]
  raw=[a for a in raw if a.get('dataset')=='hiphi' and not a.get('mirror_flag') and a['frames']>=76 and a['source_family_id'] not in train and a['source_family_id'] not in head_train_dev]
  raw.sort(key=lambda a:key(a['sample_id']));valid=[];seen=set();bad=collections.Counter()
  with torch.inference_mode():
   for a in raw:
    if a['source_family_id'] in seen:continue
    x=np.asarray(np.load(a['feature_path']),np.float32);assert x.shape==(a['frames'],414) and np.isfinite(x).all()
    start=int(key('crop:'+a['sample_id'])[:12],16)%(len(x)-76+1);prefix=x[start:start+24].copy();prefix[:,0]-=prefix[-1,0];prefix[:,2]-=prefix[-1,2]
    g=rep.inverse(torch.tensor(prefix)[None],is_normalized=False);j=g['posed_joints'][0].numpy();root=g['root_positions'][0].numpy();speed=np.linalg.norm(np.diff(root[:,[0,2]],axis=0),axis=-1)*25
    up=float(g['global_rot_mats'][0,-1,sk.root_idx,1,1]);wy=[float(j[-1,ends[s+'_wrist'],1]-root[-1,1]) for s in ['left','right']];ay=[float(j[-1,ends[s+'_ankle'],1]) for s in ['left','right']]
    if not .65<=root[-1,1]<=1.:bad['height']+=1;continue
    if up<.90:bad['tilt']+=1;continue
    if np.max(speed)>1.:bad['speed']+=1;continue
    if max(wy)>.10:bad['arms']+=1;continue
    if max(ay)>.12 or min(ay)<-.05:bad['feet']+=1;continue
    assert sha(a['feature_path'])==a['features_hash'];seen.add(a['source_family_id'])
    valid.append(dict(id=a['sample_id'],source_id=a['source_family_id'],actor_uid=a['actor_id_namespaced'],feature_path=a['feature_path'],feature_sha256=a['features_hash'],crop_start=start,prefix=prefix,last_joints=j[-1],actual_root_height_m=float(root[-1,1]),initial_root_tilt_up_dot=up,initial_max_speed_m_s=float(np.max(speed)),source_split=split,source_captions=a.get('captions',[])))
  assert len(valid)>=number,(split,len(valid),number,dict(bad));chosen.extend(valid[:number]);counts[split]=len(valid);rejects[split]=dict(bad)
 records=[];prefixes=[];poses=[];indices=[];values=[]
 for i,a in enumerate(chosen):
  split='development' if i<32 else 'locked_test';n=i if i<32 else i-32;end=list(ends)[n%4];joint=ends[end];distance=[.5,.75,1.][(n//4)%3];angle=[-15.,0.,15.][(n//12)%3]
  hips=a['last_joints'][sk.hip_joint_idx[0]]-a['last_joints'][sk.hip_joint_idx[1]];theta=math.atan2(float(hips[2]),-float(hips[0]))+math.radians(angle)
  goal=np.asarray([math.sin(theta),math.cos(theta)],np.float32)*distance;pose=a['prefix'][-1].copy();pose[[0,2]]=goal
  geom=rep.inverse(torch.tensor(pose)[None,None],is_normalized=False);end_xyz=geom['posed_joints'][0,0,joint].numpy()
  coords=list(range(rep.slice_dict['root_pos'].start,rep.slice_dict['root_pos'].stop))+list(range(rep.slice_dict['global_root_heading'].start,rep.slice_dict['global_root_heading'].stop));offset=rep.slice_dict['local_joints_positions'].start+(joint-1)*3;coords+=list(range(offset,offset+3));assert len(set(coords))==8
  rec={k:v for k,v in a.items() if k not in ['prefix','last_joints']};rec.update(case_id=f'easyv3_hiphi:{split}:{n:03d}:{a["id"]}',split=split,selected_end_effector=end,selected_joint_index=joint,goal_distance_m=distance,goal_angle_offset_degrees=angle,root_goal_xz_m=goal.tolist(),end_goal_xyz_m=end_xyz.tolist(),goal_input_coordinate_indices=coords,goal_input_physical_values=pose[coords].tolist(),root_height_input_m=float(pose[1]),goal_pose_source='last real initial-history pose translated in XZ only',noise_seeds=[int(key(f'noise:{a["id"]}:{s}')[:15],16) for s in [1701,2903,20261021]],requested_goal_groups=['root_XZ',end],additional_representation_input_fields=['root_Y','root_heading_cos_sin'],input_future_GT_trajectory_present=False)
  records.append(rec);prefixes.append(a['prefix']);poses.append(pose);indices.append(coords);values.append(pose[coords])
 test=[r for r in records if r['split']=='locked_test'];dev=[r for r in records if r['split']=='development'];assert not ({r['source_id'] for r in test}&{r['source_id'] for r in dev})
 np.savez_compressed(R/'INPUT_ASSETS.npz',case_ids=np.asarray([r['case_id'] for r in records]),initial_physical=np.stack(prefixes),goal_coordinate_indices=np.asarray(indices),goal_physical_values=np.stack(values));np.savez_compressed(R/'SCORER_ONLY_TARGET_POSES.npz',case_ids=np.asarray([r['case_id'] for r in records]),physical=np.stack(poses))
 (R/'CASES.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
 registry=rd(PARENT/'CHECKPOINTS.json');registry={m:v for m,v in registry.items() if m!='native122long'};registry['minimal']=rd(artifact_path('checkpoints_json_798bfdca7e'))['minimal'];identities=[]
 for m,v in registry.items():
  cp=Path(v['checkpoint']);st=cp.stat();assert sha(cp)==v['sha256'],('checkpoint_identity',m);v['frozen_file_stat']=dict(size=st.st_size,mtime_ns=st.st_mtime_ns);identities.append(dict(model=m,step=v['step'],checkpoint=str(cp),sha256=v['sha256']))
 wr(R/'CHECKPOINTS.json',registry)
 ad=R/'adapters/minimal/sampler';ad.mkdir(parents=True,exist_ok=True)
 shutil.copy2(artifact_path('checkpoint_loader_py_a84a0d02ec'),ad/'checkpoint_loader.py');wr(ad/'CHECKPOINTS.json',{'minimal':registry['minimal']})
 # Same literal prompts use the exact frozen generator embeddings; this is input reuse, never result reuse.
 for n in ['TEXT_INDEX.json','TEXT_FEATURES.npy']:shutil.copy2(PARENT/n,R/n)
 (R/'adapters/minimal/util.py').write_text('from pathlib import Path\nimport json\ndef write(p,x):\n Path(p).write_text(json.dumps(x,default=str))\n')
 plan=rd(PARENT/'PLAN.json');plan.update(protocol='G1_Basic_Continuous_Control_Easy_v3_HiPHI',created=time.time(),state='FROZEN_HIPHI_INPUTS_BEFORE_GENERATION',initialization_dataset='HiPHI',model_count=13,formal_conditions=52,formal_rollouts=19968)
 plan['cohort'].update(scope='composed interaction task initialized from retargeted HiPHI project-test sources; not a canonical HiPHI benchmark or natural reference trajectory',development_split='HiPHI project val',test_split='HiPHI project test',test_actors=len({r['actor_uid'] for r in test}),initial_source_caption_policy='All captions eligible for initialization: physical prefix gates unchanged. Unlike the HML parent, walk/stand wording is not required; the generated literal prompts remain identical.',generator_training_family_overlap=0,evaluator_training_development_family_overlap=0)
 plan['model_roles']['structural_ablation']=['minimal'];plan['model_roles']['optional_finetuned_references']=['native110','native117','native122'];plan['model_roles']['Minimal']='Included by explicit user request; label as structural ablation, not an external baseline'
 plan['metrics']['FID']='Frozen BONES-v2 distribution head with existing HiPHI transfer qualification; per-vector L2 normalization, FP64, newly frozen real HiPHI reference bank; not directly interchangeable across dataset/reference tables';plan['metrics']['retrieval']='Frozen HiPHI-local52 retrieval extension, two seeds1701/2903, own retrieval stats; newly frozen31-real-negative candidate pools; diagnostic only, with real action calibration'
 plan['history_utility']['execution_status']='not executed by this evaluation; do not claim causal memory evidence'
 wr(R/'PLAN.json',plan)
 audit=dict(passed=True,scope='CPU input and identity preflight; GPU generation not yet started',independent_sources=160,development_sources=32,locked_test_sources=128,eligible=counts,rejected=rejects,actors_test=len({r['actor_uid'] for r in test}),actors_test_ids=sorted({r['actor_uid'] for r in test}),end_effector_test_counts=dict(collections.Counter(r['selected_end_effector'] for r in test)),source_family_overlap=0,evaluator_train_dev_overlap=0,checkpoint_identities=identities,source_caption_policy=plan['cohort']['initial_source_caption_policy'],source_label='project split, not an invented official split',no_hidden_future_trajectory_in_input_asset=True,training_changes=0,time=time.time());wr(R/'CPU_PREFLIGHT.json',audit)
 wr(R/'FROZEN.json',dict(status='PROTOCOL_AND_REQUESTS_FROZEN_BEFORE_GENERATION',hashes={n:sha(R/n) for n in ['PLAN.json','CASES.jsonl','INPUT_ASSETS.npz','SCORER_ONLY_TARGET_POSES.npz','CHECKPOINTS.json','CPU_PREFLIGHT.json','TEXT_INDEX.json','TEXT_FEATURES.npy']},original_inputs={'test_manifest':sha(MAN/'accepted_test.jsonl'),'val_manifest':sha(MAN/'accepted_val.jsonl'),'training_index':sha(TRAIN),'skeleton':sha(REL/'SKELETON.json'),'parent_protocol':sha(PARENT/'FROZEN.json'),'HiPHI_evaluator_qualification':sha(HEADROOT/'WINDOW_EVALUATOR_QUALIFICATION.json')},time=time.time()))
 print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
