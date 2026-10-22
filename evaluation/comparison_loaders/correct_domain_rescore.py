"""Domain-correct frozen evaluator audit and rescoring of saved geometry only.

Does not generate/train motion, edit old scores, or publish unqualified retrieval.
"""
from motion_mamba2.paths import artifact_path
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import sys,json,time,hashlib,fcntl,traceback
from pathlib import Path
import numpy as np
import torch

R=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
O=R/'correct_domain_v2_20261021'
REL=Path(artifact_path('humanml3d-g1-native-v2_65063e489a'))
EV=Path(artifact_path('evaluator_native_v2_cc741fa946'))
sys.path.insert(0,str(REL/'code'))
from native_eval.v2_score import verify_release,model_for
from native_eval.encoding import encode_arrays
from native_eval.metrics import candidate_pools,retrieval,fid
sys.path.insert(0,artifact_path('_runtime_8df4cd05cf'))
from motion_mamba2.skeleton import G1Skeleton34
from motion_mamba2.representation.reps.motion_representation import MotionMamba2MotionRep

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def wr(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def r3(text,motion,pools):return float(np.mean([100*np.mean(retrieval(text,motion,p)[1]<=3) for p in pools.values()]))
def norm(a):
 a=np.asarray(a,dtype=np.float64);n=np.linalg.norm(a,axis=-1,keepdims=True)
 assert np.isfinite(a).all() and (n>1e-12).all()
 return a/n
def independent_fid(a,b):
 a=norm(a);b=norm(b)
 # Independent low-rank nuclear-norm expression of covariance Bures distance.
 x=(a-a.mean(0))/np.sqrt(len(a)-1);y=(b-b.mean(0))/np.sqrt(len(b)-1)
 val=float(np.square(a.mean(0)-b.mean(0)).sum()+np.square(x).sum()+np.square(y).sum()-2*np.linalg.svd(x@y.T,compute_uv=False).sum())
 check=float(fid(a,b));assert abs(val-check)<1e-6,(val,check)
 return max(0.,val)

def main():
 O.mkdir(exist_ok=True)
 lock=(O/'rescore.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 wr(O/'CONTROLLER_PID.json',dict(pid=os.getpid(),cwd=str(Path.cwd()),script=str(Path(__file__).resolve()),source_sha256=sha(__file__),CPU_only=True,time=time.time()))
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 release=verify_release(REL)
 ref=np.load(REL/'reference_embeddings.npz');panel=rd(REL/'retrieval_panel.json')
 assert list(ref['retrieval_ids'])==[r['sample_id'] for r in panel['records']]
 replay={s:retrieval(ref['text'],ref['motion'],p)[0] for s,p in panel['candidate_pools'].items()}
 expected=float(np.mean([d['R@3'] for d in replay.values()]))
 wr(O/'RELEASE_REFERENCE_REPLAY.json',dict(release=release['release_sha256'],N=len(ref['motion']),scores=replay,mean_R3_percent=100*expected,scope='stored frozen full-semantic evaluator calibration, not generator score',time=time.time()))
 stats=dict(np.load(REL/'stats.npz'))
 heads={1701:REL,2903:EV/'runs/R0/humanml3d/seed2903'}
 assert sha(heads[1701]/'retrieval.pt')==release['selection']['retrieval']['checkpoint_sha256']
 cohort=[json.loads(s) for s in (R/'COHORT.jsonl').open()]
 records=[dict(sample_id=c['id'],source_family_id=c['source_id'],caption=c['caption']) for c in cohort]
 pools={str(s):candidate_pools(records,s) for s in [1701,2903,20261020]}
 textfile=R/'evaluator/T5_FEATURES.npz'
 with np.load(textfile) as z:textmap={str(c):v.copy() for c,v in zip(z['captions'],z['features'])}
 texts=[textmap[r['caption']] for r in records]
 bindings=dict(release=release['release_sha256'],distribution=sha(REL/'distribution.pt'),retrieval={str(s):sha(p/'retrieval.pt') for s,p in heads.items()},stats=sha(REL/'stats.npz'),text=sha(textfile),cohort=sha(R/'COHORT.jsonl'),source=sha(__file__))
 wr(O/'ASSET_BINDINGS.json',bindings);wr(O/'QUERY_POOLS.json',pools)
 # Same held-out 181 captions: compare complete real sources and generated-only interval lengths.
 if not (O/'CURRENT_COHORT_REFERENCE_AUDIT.json').exists():
  arrays=[]
  for c in cohort:
   assert sha(c['feature_path'])==c['feature_sha256']
   arrays.append(np.asarray(np.load(c['feature_path']),dtype=np.float32))
  modes={'full_source':arrays,'short52_reference':[a[c['crop_start']+24:c['crop_start']+76] for a,c in zip(arrays,cohort)],'real9s_generated_interval':[a[c['crop_start']+24:c['crop_start']+232] for a,c in zip(arrays,cohort)]}
  result=[]
  for seed,folder in heads.items():
   model=model_for(folder,'retrieval','cpu')
   for name,arr in modes.items():
    _,mm,tt=encode_arrays(model,records,arr,stats,texts)
    result.append(dict(seed=seed,scope=name,N=len(arr),R3_percent=r3(tt,mm,pools),frames_min=min(map(len,arr)),frames_max=max(map(len,arr))))
   del model
  wr(O/'CURRENT_COHORT_REFERENCE_AUDIT.json',dict(rows=result,full_source_caption_vs_crop_distinction=True,not_a_generator_score=True,time=time.time()))
  print('REFERENCE_AUDIT',json.dumps(result),flush=True)
 skdata=rd(REL/'SKELETON.json');sk=G1Skeleton34(load=False)
 sk.register_buffer('neutral_joints',torch.tensor(skdata['data'],dtype=torch.float32),persistent=False)
 rep=MotionMamba2MotionRep(sk,25)
 methods=rd(R/'PLAN.json')['models'];tasks=['short52','aligned_w04']
 started=time.time()
 while True:
  completed=[];waiting=[]
  for method in methods:
   d=R/'models'/method
   if not (d/'COMPLETE.json').exists():waiting.append(method);continue
   for task in tasks:
    dest=O/'models'/method/task
    if (dest/'COMPLETE.json').exists():
     assert rd(dest/'COMPLETE.json')['assets']==bindings
     continue
    ga=[];ta=[];ids=[];geo_bind=[]
    for folder in sorted((d/'evaluation'/method/task).glob('shard*')):
     comp=rd(folder/'COMPLETE.json');gp=folder/'geometry.npz'
     assert sha(gp)==comp['files']['geometry.npz']
     geo_bind.append(dict(path=str(gp),sha256=sha(gp)))
     with np.load(gp) as z:
      ids.extend(map(str,z['sample_ids']))
      for role,out in [('generated',ga),('target',ta)]:
       x=torch.tensor(z[role+'_physical']);length=torch.full((len(x),),x.shape[1],dtype=torch.long)
       with torch.inference_mode():
        g=rep.inverse(x,is_normalized=False)
        y=rep(g['local_rot_mats'],g['root_positions'],to_normalize=False,to_canonicalize=False,lengths=length)
       assert torch.equal(x[:,:,:3],y[:,:,:3])
       out.extend(list(y.numpy()))
    assert len(ga)==181 and len(set(ids))==181
    wanted=[f'hml3d:{c["id"]}:{c["crop_start"]}:{task}' for c in cohort]
    assert ids==wanted,(ids[0],wanted[0])
    T=len(ga[0]);assert T in [52,208]
    dist=model_for(REL,'distribution','cpu')
    starts=list(range(0,T-51,52));gh=[];rh=[]
    for start in starts:
     gh.append(encode_arrays(dist,records,[a[start:start+52] for a in ga],stats)[0])
     rh.append(encode_arrays(dist,records,[a[start:start+52] for a in ta],stats)[0])
    del dist
    pack=dict(distribution_generated=np.stack(gh),distribution_real=np.stack(rh),sample_ids=np.asarray(ids),starts=np.asarray(starts))
    rs=[]
    for seed,folder in heads.items():
     model=model_for(folder,'retrieval','cpu')
     for scope,aa,bb in [('generated_interval',ga,ta),('terminal52',[a[-52:] for a in ga],[a[-52:] for a in ta])]:
      _,mg,tt=encode_arrays(model,records,aa,stats,texts);_,mr,_=encode_arrays(model,records,bb,stats)
      pack[f'{scope}_generated_seed{seed}']=mg;pack[f'{scope}_real_seed{seed}']=mr;pack[f'text_seed{seed}']=tt
      rs.append(dict(seed=seed,scope=scope,generated_R3_percent=r3(tt,mg,pools),real_R3_percent=r3(tt,mr,pools)))
     del model
    dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'FEATURES.npz',**pack)
    score=dict(method=method,task=task,N=181,frames=T,terminal52_L2_FID=independent_fid(rh[-1],gh[-1]),retrieval=rs,
     scope='domain-correct finite-cohort scores; retrieval crop/semantic qualification pending; not canonical leaderboard',assets=bindings,geometry=geo_bind,features_sha256=sha(dest/'FEATURES.npz'),time=time.time())
    wr(dest/'COMPLETE.json',score);print('RESCORED',method,task,flush=True)
   completed.append(method)
  wr(O/'STATUS.json',dict(stage='COMPLETE_FEATURE_RESCORING_PENDING_SCOPE_AND_MANUSCRIPT_AUDIT' if not waiting else 'WAITING_FULL100K_MODEL_EVALUATION',completed=completed,waiting=waiting,not_ready_for_automatic_publication=True,elapsed_s=time.time()-started,time=time.time()))
  if not waiting:break
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except BaseException:
  wr(O/'FAILURE.json',dict(error=traceback.format_exc(),time=time.time()));raise
