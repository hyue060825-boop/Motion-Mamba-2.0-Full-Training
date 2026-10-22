"""Complete domain-correct window metrics from immutable saved motion, CPU only."""
from motion_mamba2.paths import artifact_path
import sys,json,csv,time
from pathlib import Path
import numpy as np
import torch
R=Path(artifact_path('humanml3d_section1_20261020_c1c76adb30'))
sys.path.insert(0,str(R))
from correct_domain_rescore import O,REL,EV,rd,wr,sha,r3,independent_fid,model_for,encode_arrays,G1Skeleton34,MotionMamba2MotionRep

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(2)
 cohort=[json.loads(s) for s in (R/'COHORT.jsonl').open()]
 records=[dict(sample_id=c['id'],source_family_id=c['source_id'],caption=c['caption']) for c in cohort]
 assets=rd(O/'ASSET_BINDINGS.json');stats=dict(np.load(REL/'stats.npz'));pools=rd(O/'QUERY_POOLS.json')
 with np.load(R/'evaluator/T5_FEATURES.npz') as z:txt={str(c):v.copy() for c,v in zip(z['captions'],z['features'])}
 texts=[txt[c['caption']] for c in cohort]
 skdata=rd(REL/'SKELETON.json');sk=G1Skeleton34(load=False)
 sk.register_buffer('neutral_joints',torch.tensor(skdata['data'],dtype=torch.float32),persistent=False)
 rep=MotionMamba2MotionRep(sk,25)
 rows=[];proof=[];ref={}
 for method in rd(R/'PLAN.json')['models']:
  for task in ['short52','aligned_w04']:
   d=O/'models'/method/task;receipt=rd(d/'COMPLETE.json')
   assert receipt['assets']==assets and sha(d/'FEATURES.npz')==receipt['features_sha256']
   with np.load(d/'FEATURES.npz') as z:bank={k:z[k].copy() for k in z.files}
   if (d/'WINDOW_FEATURES_COMPLETE.json').exists():
    done=rd(d/'WINDOW_FEATURES_COMPLETE.json');assert done['assets']==assets
    assert sha(d/'WINDOW_RETRIEVAL_EMBEDDINGS.npz')==done['sha256']
    with np.load(d/'WINDOW_RETRIEVAL_EMBEDDINGS.npz') as z:wb={k:z[k].copy() for k in z.files}
   else:
    ga=[];ta=[]
    for saved in receipt['geometry']:
     p=Path(saved['path']);assert sha(p)==saved['sha256']
     with np.load(p) as z:
      for role,target in [('generated',ga),('target',ta)]:
       x=torch.tensor(z[role+'_physical']);length=torch.full((len(x),),x.shape[1],dtype=torch.long)
       with torch.inference_mode():
        g=rep.inverse(x,is_normalized=False)
        y=rep(g['local_rot_mats'],g['root_positions'],to_normalize=False,to_canonicalize=False,lengths=length)
       assert torch.equal(x[:,:,:3],y[:,:,:3]);target.extend(list(y.numpy()))
    wb={}
    for seed,folder in [(1701,REL),(2903,EV/'runs/R0/humanml3d/seed2903')]:
     model=model_for(folder,'retrieval','cpu');mg=[];mr=[]
     for start in bank['starts']:
      _,g,t=encode_arrays(model,records,[a[start:start+52] for a in ga],stats,texts)
      _,a,_=encode_arrays(model,records,[a[start:start+52] for a in ta],stats)
      mg.append(g);mr.append(a)
     del model
     wb[f'generated_seed{seed}']=np.stack(mg);wb[f'real_seed{seed}']=np.stack(mr);wb[f'text_seed{seed}']=t
     assert np.allclose(mg[-1],bank[f'terminal52_generated_seed{seed}'],atol=1e-5,rtol=1e-4)
     assert np.allclose(mr[-1],bank[f'terminal52_real_seed{seed}'],atol=1e-5,rtol=1e-4)
    np.savez_compressed(d/'WINDOW_RETRIEVAL_EMBEDDINGS.npz',**wb)
    done=dict(assets=assets,sha256=sha(d/'WINDOW_RETRIEVAL_EMBEDDINGS.npz'),geometry=receipt['geometry'],scope='52-frame local source-caption diagnostic',time=time.time())
    wr(d/'WINDOW_FEATURES_COMPLETE.json',done)
   if task not in ref:ref[task]=wb
   for seed in [1701,2903]:
    assert np.allclose(wb[f'real_seed{seed}'],ref[task][f'real_seed{seed}'],atol=1e-5,rtol=1e-4)
    assert np.allclose(wb[f'text_seed{seed}'],ref[task][f'text_seed{seed}'],atol=1e-6)
   for w,start in enumerate(bank['starts']):
    rows.append(dict(model=method,task=task,window=w+1,start_frame=int(start),N=181,
     FID_L2=independent_fid(bank['distribution_real'][w],bank['distribution_generated'][w]),
     R3_percent=float(np.mean([r3(wb[f'text_seed{s}'],wb[f'generated_seed{s}'][w],pools) for s in [1701,2903]])),
     real_R3_percent=float(np.mean([r3(wb[f'text_seed{s}'],wb[f'real_seed{s}'][w],pools) for s in [1701,2903]])),
     retrieval_scope='local52 source-caption diagnostic; not a qualified event-stage metric'))
   proof.append(dict(method=method,task=task,receipt_sha256=sha(d/'WINDOW_FEATURES_COMPLETE.json')))
   print('WINDOWS_VERIFIED',method,task,flush=True)
 with (O/'WINDOW_METRICS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 assert len(rows)==65
 wr(O/'WINDOW_AUDIT.json',dict(passed=True,models=13,conditions=26,window_groups=65,N=181,
  correct_domain_assets=True,all_window_real_embeddings_paired=True,terminal_feature_consistency=True,
  window_FID_independently_recomputed=True,local_retrieval_diagnostic=True,proof=proof,
  summary_sha256=sha(O/'WINDOW_METRICS.csv'),time=time.time()))
if __name__=='__main__':main()
