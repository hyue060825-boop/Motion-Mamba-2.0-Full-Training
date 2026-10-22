from motion_mamba2.paths import artifact_path
import json,hashlib,time
from pathlib import Path
import numpy as np
D=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'));cfg=json.loads((D/'configs/train.json').read_text());R=Path(artifact_path('bones_seed_text_cache_llm2vec_b33795dc93'));ix=json.loads((R/'index.json').read_text());features=np.load(R/'embeddings.npy',mmap_mode='r');by={r['caption']:r['row'] for r in ix['samples']};requirements=json.loads((D/'TEXT_REQUIREMENTS.json').read_text());verified=[]
for text,row in by.items():
 key=hashlib.sha256(text.encode()).hexdigest()
 for root in cfg['text_roots'][1:]:
  p=Path(root)/(key+'.npy')
  if p.exists():
   a=np.load(p).reshape(-1);b=np.array(features[row],dtype=np.float32).reshape(-1);verified.append(dict(caption=text,equal=bool(np.array_equal(a,b)),max_difference=float(np.max(np.abs(a-b)))))
   break
 if len(verified)>=128:break
compatible=bool(verified) and all(r['equal'] for r in verified)
count=0
if compatible:
 for text in requirements['captions']:
  if text not in by:continue
  key=hashlib.sha256(text.encode()).hexdigest();p=D/'text'/(key+'.npy')
  if not p.exists():
   a=np.array(features[by[text]],dtype=np.float32);assert a.size==4096 and np.isfinite(a).all();tmp=p.with_suffix('.tmp.npy');np.save(tmp,a);tmp.replace(p);count+=1
(D/'TEXT_CACHE_REUSE.json').write_text(json.dumps(dict(source=str(R),compatible=compatible,verified=verified,copied=count),ensure_ascii=False,indent=2))
print('TEXT_REUSE',compatible,len(verified),count,flush=True)
