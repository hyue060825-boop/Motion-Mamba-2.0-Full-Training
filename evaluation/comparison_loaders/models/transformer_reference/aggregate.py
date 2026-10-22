"""Same frozen evaluator and retrieval protocol; FID point estimates, no misleading bootstrap intervals."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,json,time,csv,hashlib,concurrent.futures
import numpy as np
from util import ROOT,TASKS,MODELS,read,write,sha
import aggregate_reference as reference
sys.path.insert(0,artifact_path('evaluator_native_v2_a933d7fb8b'))
from native_eval.metrics import candidate_pools,retrieval
from native_eval.v2_statistics import fid_to_many
OUT=ROOT/'aggregate';OUT.mkdir(exist_ok=True)
OLD=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
AUDIT=Path(artifact_path('transformer_reference100k_inference_audit_20261017_8a0402e361'))
def quality(task):
 b=reference.bank('ardy',task);dest=OUT/task
 with np.load(OLD/'aggregate'/task/'motion_mamba2_bank.npz') as prior:
  for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(b[k],prior[k]),(task,k)
  for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:
   assert np.allclose(b[k],prior[k],atol=1e-4,rtol=1e-5),(task,k)
 selected=[];seen=set()
 for i in sorted(range(506),key=lambda i:str(b['source_ids'][i])):
  caption=str(b['captions'][i])
  if caption not in seen:seen.add(caption);selected.append(i)
 q=np.array(selected);records=[dict(sample_id=str(b['source_ids'][i]),source_family_id=str(b['source_ids'][i]),caption=str(b['captions'][i]),frames=52) for i in q]
 pools={str(s):candidate_pools(records,s,length_matched=True) for s in [4101,4102,4103]}
 oldp=read(OLD/'aggregate'/task/'QUERY_POOLS.json')
 assert oldp['query_source_ids']==[x['sample_id'] for x in records] and oldp['pools']==pools
 write(dest/'QUERY_POOLS.json',dict(query_indices=selected,query_source_ids=oldp['query_source_ids'],pools=pools,scope=oldp['scope']))
 frows=[];rrows=[]
 for w,start in enumerate(b['starts']):
  cache=dest/f'quality_w{w+1:02d}.json'
  if cache.exists():
   c=read(cache);frows+=c['FID'];rrows+=c['retrieval'];continue
  point=float(fid_to_many(b['distribution_real'][w],[b['distribution_generated'][w]])[0])
  fr=[dict(task=task,model='ardy',window=w+1,start_generated_frame=int(start),end_generated_frame_exclusive=int(start+52),N_sources=506,LocalFID52=point,
    uncertainty='point estimate only; no calibrated FID CI claimed',scope='conditional natural-reference local52 distribution; not full-sequence FID')]
  actors=b['actor_ids'][q];rng=np.random.default_rng(3101701+w);draws=[reference.cluster_draw(actors,rng) for _ in range(2000)]
  rr=[]
  for seed in [1701,2903]:
   tt=b[f'text_seed{seed}'][w,q]
   real_ranks=np.array([retrieval(tt,b[f'retrieval_real_seed{seed}'][w,q],p)[1] for p in pools.values()])
   ranks=np.array([retrieval(tt,b[f'retrieval_generated_seed{seed}'][w,q],p)[1] for p in pools.values()])
   r=dict(task=task,model='ardy',window=w+1,encoder_seed=seed,N_queries=len(q),candidates=32,source_pool_seeds='4101/4102/4103',scope='local active-instruction compatibility, not event-order accuracy')
   for k in [1,2,3]:
    success=(ranks<=k).mean(axis=0);bs=np.array([success[ix].mean() for ix in draws]);lo,hi=np.quantile(bs,[.025,.975])
    r.update({f'R{k}':float(success.mean()),f'R{k}_ci95_low':float(lo),f'R{k}_ci95_high':float(hi),f'real_R{k}':float((real_ranks<=k).mean())})
   rr.append(r)
  write(cache,dict(FID=fr,retrieval=rr,window=w+1,time=time.time()));frows+=fr;rrows+=rr
  print('QUALITY_WINDOW',task,w+1,len(b['starts']),flush=True)
 reference.csvwrite(dest/'WINDOW_FID.csv',frows);reference.csvwrite(dest/'WINDOW_RETRIEVAL.csv',rrows)
 write(dest/'COMPLETE.json',dict(status='COMPLETE_CORRECTED_PAIRED_LOCAL52_QUALITY',N=506,windows=len(b['starts']),models=MODELS,N_queries=len(q),time=time.time()))
def main():
 plan=read(ROOT/'PLAN.json');checks=[];audit_by_source={}
 for g in sorted((AUDIT/'corrected_h196').glob('group*')):
  with np.load(g/'roots.npz') as z:
   audit_by_source.update({str(s):r[24:].copy() for s,r in zip(z['source_ids'],z['root'])})
 assert len(audit_by_source)==506
 for c in plan['catalog']:
  d=ROOT/'evaluation/ardy'/c['task']/f'shard{c["shard"]:02d}';trace=read(d/'INFERENCE_TRACE.json')
  assert trace['status']=='ACTUAL_COMMITTED_V3_EXPLICIT_HISTORY_196_AND_DECODER_52'
  for batch in trace['calls']:
   assert len(batch['windows'])==len(range(0,plan['tasks'][c['task']]['generated_frames'],52))
   assert all(x['history']<=196 and x['decoder_context_frames']<=52 and x['goal_visible'] for x in batch['windows'])
  if c['task']=='aligned_w19':
   source=reference.rows(d/'per_sample.jsonl')
   with np.load(d/'geometry.npz') as z:
    delta=max(float(np.max(abs(root-audit_by_source[r['source_id']]))) for root,r in zip(z['generated_root'],source))
   assert delta<1e-3,delta
   checks.append(dict(shard=c['shard'],root_max_abs_difference_to_previous_corrected_audit_m=delta))
 write(ROOT/'INFERENCE_EXECUTION_AUDIT.json',dict(status='PASS_COMMITTED_DECODER_AND_EXPLICIT_CROPPED_HISTORY_ROOT_COMPATIBLE_WITHIN_1MM',checks=checks,all_future_goals_always_visible=True))
 reference.endpoints()
 (OUT/'PAIRED_MODEL_CONTRASTS.csv').write_text('task,stratum,model,reference,N_sources,mean_CE_difference_cm,paired_ci95_low,paired_ci95_high\n')
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(quality,TASKS))
 f=[];r=[]
 for task in TASKS:
  f+=list(csv.DictReader((OUT/task/'WINDOW_FID.csv').open()));r+=list(csv.DictReader((OUT/task/'WINDOW_RETRIEVAL.csv').open()))
 reference.csvwrite(OUT/'ALL_WINDOW_LOCALFID52.csv',f);reference.csvwrite(OUT/'ALL_WINDOW_RETRIEVAL.csv',r)
 write(ROOT/'NUMERICS_COMPLETE.json',dict(status='COMPLETE_CORRECTED_RETRAINED_ARDY100K_AND_EVALUATOR_V2',new_trajectories=1518,N_sources=506,actors=22,models=MODELS,tasks=TASKS,
  result_hashes={n:sha(OUT/n) for n in ['ENDPOINT_SUMMARY.csv','PAIRED_MODEL_CONTRASTS.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']},
  endpoint_audit=sha(OUT/'ENDPOINT_AUDIT.json'),no_root_ADE=True,source_exposure='Known generator training sources, post-hoc diagnosis',time=time.time()))
 print('NUMERICS_COMPLETE',flush=True)
if __name__=='__main__':main()
