from pathlib import Path
import os,json,hashlib,time
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import numpy as np
R=Path(__file__).resolve().parent
rd=lambda p:json.loads(Path(p).read_text())
def norm(x):x=np.asarray(x,np.float64);return x/np.linalg.norm(x,axis=-1,keepdims=True)
def main():
 frozen=rd(R/'PROTOCOL_FROZEN.json');assert frozen['passed']
 for n,h in frozen['hashes'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 tests=[json.loads(s) for s in (R/'CASES.jsonl').open() if json.loads(s)['split']=='locked_test'];bank=dict(np.load(R/'REAL_PAIRED_FEATURES.npz'));ix={int(v):i for i,v in enumerate(bank['catalog_indices'])};pools=rd(R/'QUERY_POOLS.json');cal=rd(R/'REAL_REFERENCE_CALIBRATION.json');checks=[]
 for old in cal['checks']:
  p=old['phase'];meta=pools[str(p)];q=meta['query_case_indices'];ids=[ix[tests[i]['paired_reference_indices'][p]] for i in q];seed=old['head'];t=norm(bank['text'+str(seed)][ids]);m=norm(bank['retrieval'+str(seed)][ids]);dist=np.maximum(0,2-2*t@m.T);ranks=[]
  assert len(q)>=32 and len(set(meta['captions']))==len(q)
  for i,choices in enumerate(meta['pools'][old['pool_seed']]):
   assert len(choices)==len(set(choices))==32 and i in choices
   assert len({meta['captions'][j] for j in choices})==32
   order=np.argsort(dist[i,choices],kind='stable');ranks.append(1+int(np.flatnonzero(np.asarray(choices)[order]==i)[0]))
  scores={f'R@{k}':float((np.asarray(ranks)<=k).mean()) for k in [1,2,3]}
  for k,v in scores.items():assert abs(v-old['scores'][k])<1e-12
  assert old['passed'];checks.append(dict(phase=p,head=seed,pool=old['pool_seed'],queries=len(q),scores=scores))
 assert len(checks)==24
 (R/'REAL_REFERENCE_INDEPENDENT_AUDIT.json').write_text(json.dumps(dict(passed=True,all24_phase_head_pool_checks=checks,strict32_unique_caption_candidates=True,no_real_negative_anchor=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time()),indent=2)+'\n')
 print('24 real paired-caption retrieval checks independently reproduced')
if __name__=='__main__':main()
