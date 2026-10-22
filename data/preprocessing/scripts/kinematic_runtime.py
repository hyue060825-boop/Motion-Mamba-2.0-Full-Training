"""Actual mixed G1 loader, tokenizer bridge and three-model training smoke."""
from motion_mamba2.paths import artifact_path
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com')
os.environ.setdefault('HF_HOME',artifact_path('huggingface_ed9ba7d805'))
import argparse,gc,json,sys,math,collections
from pathlib import Path
import numpy as np
import torch
from g1_pipeline.common import ROOT,REPO,sha,write_json,write_jsonl,read_jsonl,setup_imports,now
from g1_pipeline.kinematic_policy import BASE,authorize,load
setup_imports();sys.path.insert(0,str(REPO/'mamba-transformer-keneral/examples'))
for path in [artifact_path('state-spaces-mamba_fd8479ca04'),artifact_path('mamba_op_triton_3220f2c818')]:sys.path.insert(0,path)

def loader_roundtrip(project,physical,batch,records,chosen):
 from motion_mamba2.exports.mujoco import MujocoQposConverter
 from scipy.spatial.transform import Rotation
 from g1_pipeline.robot import compile_target,BASIS
 import mujoco
 converter=MujocoQposConverter(project.skeleton);robot,_=compile_target();data=mujoco.MjData(robot)
 indices=[project.skeleton.bone_order_names.index(robot.joint(j).name.replace('_joint','_skel')) for j in range(1,robot.njnt)]
 result={}
 for b,(index,_) in enumerate(chosen):
  row=records[index];x=physical[b:b+1];transform=row['canonical_transform']
  restored=project.translate_2d(x,torch.tensor(transform['planar_translation'],device=x.device)[None])
  # A crop may start after a turn: undo the full-clip rotation by its fixed
  # delta, rather than resetting the crop's first heading.
  restored=project.rotate(restored,torch.tensor([transform['heading']],device=x.device))
  inv=project.inverse(restored,is_normalized=False);q=converter.dict_to_qpos(inv)[0]
  start=int(batch['crop_start'][b]);expected=np.load(ROOT/row['qpos_path'],mmap_mode='r')[start:start+int(batch['length'][b])]
  assert q.shape==expected.shape==(104,36)
  values={'translation_max_m':float(abs(q[:,:3]-expected[:,:3]).max()),'orientation_max_deg':float(np.rad2deg((Rotation.from_quat(expected[:,3:7],scalar_first=True).inv()*Rotation.from_quat(q[:,3:7],scalar_first=True)).magnitude()).max()),'hinge_max_deg':float(np.rad2deg(abs(q[:,7:]-expected[:,7:])).max())}
  error=0.
  for frame,pose in enumerate(q):
   data.qpos[:]=pose;mujoco.mj_forward(robot,data)
   error=max(error,float(np.linalg.norm(data.xpos[robot.jnt_bodyid[1:]]@BASIS.T-inv['posed_joints'][0,frame,indices].cpu().numpy(),axis=-1).max()))
  values['fk_position_max_m']=error
  limits={'translation_max_m':1e-4,'orientation_max_deg':.02,'hinge_max_deg':.02,'fk_position_max_m':.001}
  if any(not math.isfinite(values[k]) or values[k]>limit for k,limit in limits.items()):raise ValueError('LOADER_INVERSE_QPOS_FK_REGRESSION:'+repr(values))
  result[row['dataset']]=values
 return result

def run(mode,device,manifest_path=None):
 authorize();torch.manual_seed(20260917);torch.cuda.set_device(device)
 from g1_pipeline.training_data import G1Dataset,SourceFamilySampler
 manifest=Path(manifest_path).resolve() if manifest_path else BASE/'manifests/accepted_train.jsonl';records=read_jsonl(manifest)
 runtime_root=BASE/('runtime_preflight' if manifest_path else 'runtime')
 contract=load(ROOT/'contracts/motion_mamba2_414.json');stats=Path(contract['chosen_normalizer'])
 dataset=G1Dataset(manifest,stats,frames=104,horizon=52)
 # Real training sampler, fixed seed; select a mixed full-length batch from its draws.
 sampler=SourceFamilySampler(records,samples=10000,seed=20260917);selected={};exposure=collections.defaultdict(collections.Counter)
 for idx,seed in sampler:
  row=records[idx]
  exposure[row['dataset']+':'+row['category']]['draws']+=1
  exposure[row['dataset']+':'+row['category']]['valid_frames']+=min(104,row['target_frames'])
  if row['target_frames']>=104 and row['dataset'] not in selected:selected[row['dataset']]=(idx,seed)
 if len(selected)!=3:raise ValueError('MIXED_SOURCE_104_FRAME_SMOKE_COVERAGE_MISSING')
 chosen=[selected[s] for s in ['bones','humanml3d','hiphi']]
 batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=3,sampler=chosen,num_workers=0)))
 assert batch['motion'].shape==(3,104,414) and batch['motion'].dtype==torch.float32
 assert batch['valid_mask'].dtype==torch.bool and batch['valid_mask'].all()
 assert all(batch['fps']==25) and set(batch['dataset'])=={'bones','humanml3d','hiphi'}
 # Test variable-length padding separately, including shortest actual accepted clip.
 shortest=min(range(len(records)),key=lambda i:records[i]['target_frames'])
 pad_frames=max(104,52*(records[shortest]['target_frames']//52+1))
 variable=G1Dataset(manifest,stats,frames=pad_frames,horizon=52)[(shortest,20260917)]
 assert torch.equal(variable['valid_mask'],torch.arange(pad_frames)<int(variable['length'])) and not variable['valid_mask'].all()
 if not torch.isfinite(variable['motion'][variable['valid_mask']]).all():raise ValueError('NONFINITE_LOADER_VALID_FRAMES')
 out=runtime_root/mode;out.mkdir(parents=True,exist_ok=True)
 write_json(out/'sampler_exposure.json',{'seed':20260917,'draws':10000,'method':'existing equal-source then equal-family sampler; valid crop frames only','groups':dict(exposure)})
 np.savez_compressed(out/'mixed_batch.npz',features=batch['motion'].numpy(),lengths=batch['length'].numpy(),mask=batch['valid_mask'].numpy(),sources=np.asarray(batch['dataset']))
 # Use the actual released text encoder, never hash vectors as text features.
 assets=load(ROOT/'audit/training_assets_v1.json');textpath=runtime_root/'text_features.npy';textmeta=runtime_root/'text_inputs.json'
 if textpath.exists():
  textrecord=load(textmeta)
  if textrecord['texts']!=batch['text'] or textrecord['sha256']!=sha(textpath):raise ValueError('SMOKE_TEXT_CACHE_CHANGED')
  embedding=np.load(textpath)
 else:
  from motion_mamba2.adapters.text_encoder import ExactMotionMamba2TextEncoder
  encoder=ExactMotionMamba2TextEncoder(base_mirror=str(BASE/'model_frontend/llama_local'),mntp_adapter=assets['models']['mntp']['snapshot'],supervised_adapter=assets['models']['supervised']['snapshot'],device=device)
  values,_=encoder(batch['text']);embedding=values[:,0].float().cpu().numpy();np.save(textpath,embedding)
  write_json(textmeta,{'texts':batch['text'],'sha256':sha(textpath),'encoder_assets':{k:assets['models'][k] for k in ['llama','mntp','supervised']}})
  del encoder,values;gc.collect();torch.cuda.empty_cache()
 batch['text_feature']=torch.from_numpy(embedding);batch['text_dropped']=torch.zeros(3,dtype=torch.bool)
 from motion_mamba2.models import load_model
 from motion_mamba2.representation import MotionMamba2MotionRep
 from g1_pipeline.tokenizer_normalizer_bridge import FrozenTokenizerBridge
 from g1_pipeline.kinematic_tokenizer_bridge import KinematicHybridConverter
 from motion_mamba2.models.latent_utils import HybridMotionConverter
 from train_humanml3d_motion_mamba2_tiny import _build_batch,_loss
 model=load_model('g1',device=device,text_encoder=False);raw=model.denoiser.model
 if model.skeleton.nbjoints!=34 or model.motion_rep.motion_rep_dim!=414 or model.motion_rep.fps!=25:raise ValueError('WRONG_G1_MODEL')
 released=model.motion_rep;project=MotionMamba2MotionRep(model.skeleton,25,stats_path=str(stats))
 original_auto=model.autoencoder;bridge=FrozenTokenizerBridge(original_auto,project).to(device)
 # Check the bridge against the unchanged released tokenizer with true masks.
 physical=project.unnormalize(batch['motion'].to(device));lengths=batch['length'].to(device);mask=batch['valid_mask'].to(device)
 with torch.no_grad():
  released_input=released.normalize(physical);reference_tokens,_=model.hybrid.get_hybrid_motion_from_explicit(released_input,lengths,mask)
  released_recon=released.unnormalize(model.hybrid.get_explicit_motion_from_hybrid(reference_tokens,mask,lengths))
 model.motion_rep=project;raw.motion_rep=project
 for owner in (model,raw):
  official=HybridMotionConverter(denoiser=raw,autoencoder=original_auto,motion_rep=released,gen_horizon_len=52)
  owner.hybrid=KinematicHybridConverter(project_rep=project,released_converter=official,bridge=bridge)
 model.autoencoder=bridge
 with torch.no_grad():
  tokens,_=model.hybrid.get_hybrid_motion_from_explicit(batch['motion'].to(device),lengths,mask)
  reconstructed=project.unnormalize(model.hybrid.get_explicit_motion_from_hybrid(tokens,mask,lengths))
  bridge_error=float(abs(reconstructed-released_recon)[mask].max())
  write_json(out/'bridge_diagnostics.json',{'max_abs':bridge_error,'root_max_abs':float(abs(reconstructed[...,:5]-released_recon[...,:5])[mask].max()),'body_max_abs':float(abs(reconstructed[...,5:]-released_recon[...,5:])[mask].max()),'latent_max_abs':float(abs(tokens[...,raw.nframe_root_dim:]-reference_tokens[...,raw.nframe_root_dim:]).max()),'released_rep_type':str(type(released)),'project_rep_type':str(type(project))})
  assert torch.allclose(reconstructed[mask],released_recon[mask],atol=1e-5,rtol=1e-5),'TOKENIZER_BRIDGE_MISMATCH'
  target_joints=project.inverse(physical,is_normalized=False)['posed_joints']
  reconstructed_joints=project.inverse(reconstructed,is_normalized=False)['posed_joints']
  tokenizer_mpjpe=float(torch.linalg.vector_norm(reconstructed_joints-target_joints,dim=-1)[mask].mean())
  from motion_mamba2.exports.mujoco import MujocoQposConverter
  inverse=project.inverse(physical,is_normalized=False)
  recovered=MujocoQposConverter(model.skeleton).dict_to_qpos(inverse)
  if not torch.isfinite(torch.as_tensor(recovered)).all():raise ValueError('LOADER_INVERSE_QPOS_NONFINITE')
  regression=loader_roundtrip(project,physical,batch,records,chosen)
 for parameter in model.parameters():parameter.requires_grad_(False)
 if mode=='ardy':client=model
 else:
  from motion_mamba2.adapters import MotionMamba2SequenceKernelConfig,install_motion_mamba2_sequence_kernel
  client=install_motion_mamba2_sequence_kernel(model,MotionMamba2SequenceKernelConfig(mode=mode,attention_backend='sdpa',window_size_chunks=4,num_sink_tokens=4))
 raw=model.denoiser.model
 for parameter in raw.parameters():parameter.requires_grad_(True)
 params=[p for p in raw.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=1e-5)
 fixed=_build_batch(model,batch,torch.device(device),fixed_noise=True)
 steps=[];raw.eval()
 for step in range(1):
  optimizer.zero_grad(set_to_none=True);loss=_loss(raw,client,fixed);loss.backward()
  if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise ValueError('NONFINITE_SMOKE_LOSS_OR_GRADIENT')
  norm=float(torch.nn.utils.clip_grad_norm_(params,1.0));assert math.isfinite(norm) and norm>0
  optimizer.step();steps.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':norm})
 with torch.no_grad():expected=float(_loss(raw,client,fixed))
 cp=out/'temporary_smoke_checkpoint.pt';torch.save({'mode':mode,'model':'g1','state_dict':raw.state_dict(),'input_manifest_sha256':sha(manifest),'stats':str(stats),'scope':'one_step_smoke_not_formal_training'},cp)
 saved=torch.load(cp,map_location=device,weights_only=False);raw.load_state_dict(saved['state_dict'],strict=True)
 with torch.no_grad():actual=float(_loss(raw,client,fixed))
 assert abs(expected-actual)<=1e-6,'CHECKPOINT_RELOAD_MISMATCH'
 report={'time':now(),'status':'PASS','model':{'minimal':'Minimal-G1','ardy':'ARDY-G1','hybrid':'NativeHybrid-G1'}[mode],'mode':mode,'steps':steps,'checkpoint_reloaded':True,'checkpoint_reload_loss_max_abs':abs(expected-actual),'actual_project_dataloader':True,'actual_project_model':True,'actual_text_encoder':True,'normalizer_checkpoint_bridge_equivalence':True,'bridge_max_abs':bridge_error,'tokenizer_reconstruction_mpjpe_m':tokenizer_mpjpe,'tokenizer_reconstruction_note':'Measured learned reconstruction; deterministic full-motion encode/inverse/FK limits are separately enforced per accepted clip. Bridge matches released frontend within existing float tolerance.','model_contract':{'skeleton':'G1Skeleton34','dof':29,'representation':414,'fps':25},'selected_samples':[records[i]['sample_id'] for i,_ in chosen],'shortest_loader_sample':records[shortest]['sample_id'],'input_bindings':{str(manifest.relative_to(ROOT)):sha(manifest),str((out/'mixed_batch.npz').relative_to(ROOT)):sha(out/'mixed_batch.npz'),str(cp.relative_to(ROOT)):sha(cp),str(textmeta.relative_to(ROOT)):sha(textmeta),str(stats/'mean.npy'):sha(stats/'mean.npy'),str(stats/'std.npy'):sha(stats/'std.npy')},'formal_training_started':False,'final_manifest_smoke':manifest_path is None}
 report.update(per_source_loader_roundtrip=regression,padding_test={'sample_id':records[shortest]['sample_id'],'valid_frames':int(variable['length']),'padded_frames':pad_frames-int(variable['length'])},loader_class='torch.utils.data.DataLoader')
 for path in [Path(__file__),ROOT/'g1_pipeline/kinematic_tokenizer_bridge.py',ROOT/'g1_pipeline/tokenizer_normalizer_bridge.py',out/'sampler_exposure.json']:
  report['input_bindings'][str(path.relative_to(ROOT))]=sha(path)
 write_json(out/'acceptance.json',report);print({'mode':mode,'status':'PASS','steps':steps,'bridge_error':bridge_error})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['minimal','ardy','hybrid'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--manifest');a=p.parse_args();run(a.mode,a.device,a.manifest)
