"""Freeze GT-only real banks and anchored candidate pools before any scoring."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,re,json,time,collections,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from execute import R,rd,wr,sha,digest
import numpy as np,torch
REL=Path(artifact_path('bones-g1-native-v2_adcc182e95'))
HEADROOT=Path(artifact_path('hiphi_window_head_de5d446e81'))
HEADS={seed:HEADROOT/f'seed{seed}' for seed in [1701,2903]}
sys.path.insert(0,str(REL/'code'))
sys.path.insert(0,artifact_path('_runtime_924bb4d0b9'))
from native_eval.v2_score import model_for,verify_release
from native_eval.encoding import encode_arrays
from motion_mamba2.skeleton import G1Skeleton34
from motion_mamba2.representation.reps.motion_representation import MotionMamba2MotionRep
def representation():
 sk=G1Skeleton34(load=False);sk.register_buffer('neutral_joints',torch.tensor(rd(REL/'SKELETON.json')['data'],dtype=torch.float32),persistent=False)
 return MotionMamba2MotionRep(sk,25),sk
def canonical(rep,arr):
 x=torch.tensor(np.stack(arr));lens=torch.full((len(x),),x.shape[1],dtype=torch.long)
 with torch.inference_mode():
  g=rep.inverse(x,is_normalized=False)
  y=rep(g['local_rot_mats'],g['root_positions'],to_normalize=False,to_canonicalize=False,lengths=lens)
 assert torch.equal(y[:,:,:3],x[:,:,:3])
 return y.numpy(),g
def labels(c):
 c=c.lower()
 if re.search(r'\b(then|while|before|after|turn\w*|circle|circl\w*|runs?|running|spins?|stops?)\b',c):return None
 walk=re.search(r'\b(walk\w*|step\w*)\b',c)
 if walk:
  if 'backward' in c:return 'backward'
  if 'forward' in c:return 'forward'
  return None
 if re.search(r'\bstand\w*\b.*\bstill\b|\bstandstill\b',c) and not re.search(r'hand|arm|bend|pick|lift|kick|push|strum',c):return 'stand'
 if re.search(r'\b(clap\w*|jump\w*|kick\w*|wav\w*|squat\w*|sit\w*|punch\w*|bow\w*)\b',c) and not re.search(r'walk|step|jog|run|mov\w* forward|mov\w* backward',c):return 'other'
 return None
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 out=R/'scoring';out.mkdir(exist_ok=True)
 if (out/'REFERENCE_COMPLETE.json').exists():return
 release=verify_release(REL);rep,sk=representation()
 qualification=rd(HEADROOT/'WINDOW_EVALUATOR_QUALIFICATION.json');assert qualification['status']=='PASS'
 for seed,folder in HEADS.items():assert sha(folder/'retrieval.pt')==qualification['binding']['retrieval_checkpoints'][str(seed)]
 assert sha(HEADROOT/'retrieval_stats.npz')==qualification['binding']['retrieval_stats_sha256']
 train={json.loads(s)['source_family_id'] for s in Path(artifact_path('data_index_jsonl_9966c1bc60')).open()}
 raw=[json.loads(s) for s in Path(artifact_path('accepted_val_jsonl_bf609ce99f')).open()]
 raw=[r for r in raw if r['dataset']=='hiphi' and not r['mirror_flag'] and r['source_family_id'] not in train and r['frames']>=52]
 raw.sort(key=lambda x:__import__('hashlib').sha256(('v3_reference:'+x['sample_id']).encode()).hexdigest())
 records=[];arrays=[];seen=set();counts=collections.Counter();caps={'forward':96,'backward':64,'stand':32,'other':192}
 for r in raw:
  if r['source_family_id'] in seen:continue
  caption=r['captions'][0];text_label=labels(caption);a=np.asarray(np.load(r['feature_path']),np.float32)
  selected=None
  for start in dict.fromkeys([(len(a)-52)//2,0,len(a)-52]):
   x=a[start:start+52].copy();y,g=canonical(rep,[x]);root=g['root_positions'][0].numpy();j=g['posed_joints'][0].numpy()
   vel=np.diff(root[:,[0,2]],axis=0)*25;hip=j[:-1,8]-j[:-1,1];f=np.stack((hip[:,2],-hip[:,0]),1);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
   vf=(vel*f).sum(1);speed=np.linalg.norm(vel,axis=1);joint_speed=np.linalg.norm(np.diff(j,axis=0),axis=-1)*25
   label=None;basis=None
   if text_label=='forward' and np.mean(vf>.1)>=.8:label='forward';basis='source caption and physical body-forward speed'
   elif text_label=='backward' and np.mean(vf<-.1)>=.8:label='backward';basis='source caption and physical body-backward speed'
   elif np.mean(speed<.1)>=.8 and np.mean(joint_speed)<.15:label='stand';basis='GT stationary root and low joint-speed crop; not a claim about original full-clip caption'
   elif not re.search(r'\b(walk\w*|step\w*|run\w*|jog\w*|standstill)\b',caption,re.I):label='other';basis='non-locomotion source caption'
   if label and counts[label]<caps[label]:selected=(start,y,root,j,vf,speed,label,basis);break
  if selected is None:continue
  start,y,root,j,vf,speed,label,basis=selected
  if sha(r['feature_path'])!=r['features_hash']:raise ValueError('source identity changed')
  records.append(dict(sample_id=r['sample_id'],source_family_id=r['source_family_id'],caption=caption,label=label,feature_path=r['feature_path'],sha256=r['features_hash'],start=start,frames=52,label_basis=basis,actual_forward_speed_m_s=float(vf.mean()),actual_planar_speed_m_s=float(speed.mean())))
  arrays.append(y[0]);seen.add(r['source_family_id']);counts[label]+=1
 assert counts['forward']>=2 and counts['backward']>=2 and counts['stand']>=2 and counts['other']>=31,dict(counts)
 pools={};command_positive={}
 for command in ['forward','backward','stand','walk']:
  positive_labels=['forward','backward'] if command=='walk' else [command]
  command_positive[command]=[i for i,r in enumerate(records) if r['label'] in positive_labels]
  # Generic walk excludes all locomotion from distractors; direction commands may use the opposite direction.
  negatives=[i for i,r in enumerate(records) if r['label'] not in positive_labels]
  assert len(negatives)>=31
  pools[command]={str(seed):np.random.default_rng(seed).choice(negatives,31,replace=False).tolist() for seed in [1701,2903,20261021]}
 data=dict(records=records,counts=dict(counts),positive_indices=command_positive,pools=pools,selection='HiPHI project-val, one source family once, fixed middle/start/end crop order; GT caption and geometry only; stationary crops identified by root and joint speed; no generator result is read',retrieval='fixed real31 negatives plus generated positive, cosine T2M, strict tie counts before positive',quality_reference='all qualified forward/backward/still real families; one frozen distribution for every model and window; class-balanced sensitivity also retained',domain='HiPHI-local52 retrieval + qualified BONES-v2 distribution transfer',natural_long_reference=False)
 # Freeze full pool identities before extracting/scoring any generated motion.
 wr(out/'REFERENCE_AND_POOLS.json',data)
 stats=dict(np.load(REL/'stats.npz'));dist=model_for(REL,'distribution','cpu');h=encode_arrays(dist,records,arrays,stats)[0];del dist
 balanced=min(counts[k] for k in ['forward','backward','stand'])
 balanced_ids=sum([[i for i,r in enumerate(records) if r['label']==lab][:balanced] for lab in ['forward','backward','stand']],[])
 qids=[i for i,r in enumerate(records) if r['label']!='other']
 # Also retain unbalanced full bank and per-label subsets for sensitivity.
 pack={'distribution':h,'quality_indices':np.asarray(qids),'balanced_quality_indices':np.asarray(balanced_ids),'physical':np.stack(arrays)}
 from native_eval.common import TEXT
 from native_eval.tokenizer import FrozenTokenizer
 from transformers import T5EncoderModel
 prompts=rd(R/'PLAN.json')['prompts'];cmd=list(prompts)
 tok=FrozenTokenizer(TEXT/'spiece.model');enc=T5EncoderModel.from_pretrained(str(TEXT),local_files_only=True,dtype=torch.bfloat16).eval()
 with torch.inference_mode():
  t=tok([prompts[k] for k in cmd],padding=True,truncation=False,return_tensors='pt');v=enc(**t).last_hidden_state.float();mask=t['attention_mask'][:,:,None];tf=((v*mask).sum(1)/mask.sum(1)).numpy()
 del enc
 np.savez_compressed(out/'T5_COMMANDS.npz',commands=np.asarray(cmd),features=tf)
 heads=HEADS;cal=[]
 for seed,folder in heads.items():
  model=model_for(folder,'retrieval','cpu');_,mm,_=encode_arrays(model,records,arrays,dict(np.load(HEADROOT/'retrieval_stats.npz')))
  with torch.inference_mode():tt=model.text(torch.tensor(tf)).numpy()
  pack[f'motion_{seed}']=mm;pack[f'text_{seed}']=tt
  m=mm.astype(float);m/=np.linalg.norm(m,axis=1,keepdims=True);t=tt.astype(float);t/=np.linalg.norm(t,axis=1,keepdims=True)
  for ci,c in enumerate(cmd):
   pos=command_positive[c];ranks=[]
   for pi in pools[c].values():
    dneg=2-2*t[ci]@m[pi].T;dp=2-2*m[pos]@t[ci]
    ranks.append(1+(dneg[None,:]<=dp[:,None]).sum(1))
   cal.append(dict(head=seed,command=c,positive_real_families=len(pos),R3_ref_percent=float((np.asarray(ranks)<=3).mean()*100)))
  del model
 np.savez_compressed(out/'REFERENCE_FEATURES.npz',**pack)
 wr(out/'REAL_REFERENCE_CALIBRATION.json',dict(rows=cal,strict_ties='distractor distance <= positive counts ahead',scope='new simple-command reference-anchored pool; not the historical1024 full-semantic85% calibration',counts=dict(counts),quality_reference_families=len(qids),balanced_quality_reference_families=len(balanced_ids),finite_sample_warning='Small stand-still bank limits class-balanced FID; unbalanced full-bank primary and balanced sensitivity both retained',not_generator_scores=True,time=time.time()))
 wr(out/'ASSET_BINDINGS.json',dict(release_sha256=release['release_sha256'],distribution=sha(REL/'distribution.pt'),retrieval={str(k):sha(p/'retrieval.pt') for k,p in heads.items()},stats=sha(REL/'stats.npz'),retrieval_stats=sha(HEADROOT/'retrieval_stats.npz'),HiPHI_qualification=sha(HEADROOT/'WINDOW_EVALUATOR_QUALIFICATION.json'),text=sha(out/'T5_COMMANDS.npz'),reference=sha(out/'REFERENCE_FEATURES.npz'),pools=sha(out/'REFERENCE_AND_POOLS.json'),source=sha(__file__)))
 wr(out/'REFERENCE_COMPLETE.json',dict(passed=True,reference_families=len(records),quality_families=len(qids),counts=dict(counts),time=time.time()))
 print(json.dumps(dict(counts=dict(counts),calibration=cal)),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:wr(R/'scoring'/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
