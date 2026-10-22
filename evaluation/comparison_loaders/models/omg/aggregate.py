"""Frozen evaluator_v2, paired endpoints; descriptive FID without bootstrap CI."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import os,sys,csv,time,concurrent.futures
import numpy as np
from util import ROOT,TASKS,MODELS,read,write,sha
import aggregate_reference as ref
sys.path.insert(0,artifact_path('evaluator_native_v2_a933d7fb8b'))
from native_eval.metrics import candidate_pools,retrieval
from native_eval.v2_statistics import fid_to_many
OUT=ROOT/'aggregate';OUT.mkdir(exist_ok=True)
def quality(task):
 banks={m:ref.bank(m,task) for m in MODELS};b=banks[MODELS[0]];dest=OUT/task
 for m,v in banks.items():
  for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(b[k],v[k]),(task,m,k)
  for k in ['distribution_real','text_seed1701','text_seed2903','retrieval_real_seed1701','retrieval_real_seed2903']:assert np.allclose(b[k],v[k],atol=1e-4,rtol=1e-5)
 selected=[];seen=set()
 for i in sorted(range(506),key=lambda i:str(b['source_ids'][i])):
  c=str(b['captions'][i])
  if c not in seen:seen.add(c);selected.append(i)
 q=np.array(selected);records=[dict(sample_id=str(b['source_ids'][i]),source_family_id=str(b['source_ids'][i]),caption=str(b['captions'][i]),frames=52) for i in q]
 pools={str(s):candidate_pools(records,s,length_matched=True) for s in [4101,4102,4103]}
 old=read(Path(artifact_path('aggregate_1aad8b047a'))/task/'QUERY_POOLS.json')
 assert old['query_source_ids']==[r['sample_id'] for r in records] and old['pools']==pools
 write(dest/'QUERY_POOLS.json',dict(query_indices=selected,query_source_ids=old['query_source_ids'],pools=pools))
 frows=[];rrows=[]
 for w,start in enumerate(b['starts']):
  cache=dest/f'quality_w{w+1:02d}.json'
  if cache.exists():
   v=read(cache);frows+=v['FID'];rrows+=v['retrieval'];continue
  points=fid_to_many(b['distribution_real'][w],[banks[m]['distribution_generated'][w] for m in MODELS])
  fr=[dict(task=task,model=m,window=w+1,start_generated_frame=int(start),end_generated_frame_exclusive=int(start+52),N_sources=506,LocalFID52=float(points[i]),uncertainty='point estimate; no calibrated FID CI claimed',scope='pooled local52 distribution; not sequence FID') for i,m in enumerate(MODELS)]
  rr=[];rng=np.random.default_rng(3101701+w);draws=[ref.cluster_draw(b['actor_ids'][q],rng) for _ in range(2000)]
  for seed in [1701,2903]:
   tt=b[f'text_seed{seed}'][w,q];real=np.array([retrieval(tt,b[f'retrieval_real_seed{seed}'][w,q],p)[1] for p in pools.values()])
   for m in MODELS:
    ranks=np.array([retrieval(tt,banks[m][f'retrieval_generated_seed{seed}'][w,q],p)[1] for p in pools.values()])
    r=dict(task=task,model=m,window=w+1,encoder_seed=seed,N_queries=len(q),candidates=32,source_pool_seeds='4101/4102/4103')
    for k in [1,2,3]:
     success=(ranks<=k).mean(axis=0);lo,hi=np.quantile([success[ix].mean() for ix in draws],[.025,.975])
     r.update({f'R{k}':float(success.mean()),f'R{k}_ci95_low':float(lo),f'R{k}_ci95_high':float(hi),f'real_R{k}':float((real<=k).mean())})
    rr.append(r)
  write(cache,dict(FID=fr,retrieval=rr));frows+=fr;rrows+=rr
 ref.csvwrite(dest/'WINDOW_FID.csv',frows);ref.csvwrite(dest/'WINDOW_RETRIEVAL.csv',rrows)
def main():
 plan=read(ROOT/'PLAN.json');visibility=[]
 for m in MODELS:
  for c in plan['catalog']:
   p=ROOT/'evaluation'/m/c['task']/f'shard{c["shard"]:02d}'
   for call in read(p/'INFERENCE_TRACE.json')['calls']:
    ws=call['windows'];assert all(x['actual_input_frames']<=248 and x['history']<=196 for x in ws)
    visibility.append(dict(model=m,task=c['task'],shard=c['shard'],first_visible_window=next((x['window'] for x in ws if x['goal_visible']),None),goal_visible_windows=[x['window'] for x in ws if x['goal_visible']]))
 write(ROOT/'INFERENCE_EXECUTION_AUDIT.json',dict(status='VERIFIED_NATIVE_FIXED_INPUT_CAPACITY',records=visibility,full_future_conditioning_equivalent=False,no_time_compression_or_synthetic_waypoints=True))
 ref.endpoints()
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(quality,TASKS))
 for fn,output in [('WINDOW_FID.csv','ALL_WINDOW_LOCALFID52.csv'),('WINDOW_RETRIEVAL.csv','ALL_WINDOW_RETRIEVAL.csv')]:
  combined=[]
  for t in TASKS:combined+=list(csv.DictReader((OUT/t/fn).open()))
  ref.csvwrite(OUT/output,combined)
 comparison=[]
 roots=[ROOT,Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d')),Path(artifact_path('transformer_reference100k_corrected_evaluation_20261017_7a91e02b7e')),Path(artifact_path('native100k_controlled_endpoint_20261017_4c3b70479f')),Path(artifact_path('horizon_controlled_motion_mamba2_official_corrected_20261017_633fa66262'))]
 for r in roots:
  p=r/'aggregate/ENDPOINT_SUMMARY.csv'
  if p.exists():
   for row in csv.DictReader(p.open()):
    if row['model'] in ['minimal','minimal_w4']:continue
    if r.name=='horizon_controlled_four_models_20261017' and row['model']=='ardy':continue
    comparison.append(dict(row,result_root=str(r),full_future_conditioning_equivalent=r!=ROOT))
 ref.csvwrite(OUT/'COMPARISON_ENDPOINTS.csv',comparison)
 report='''# Frozen100k external architecture controls\n\nDiP-G1 and DART-MLP-G1 are adapted architecture controls, not full official systems. Both checkpoints finished100k updates (75k batch32 +25k batch512).\n\nAll seven506-source paired conditions reuse the frozen source IDs, fixed spatial goals, prefix, text and generator seeds. Only terminal root-XZ constraint error is primary; no Root ADE or intermediate-route penalty. These are generator-training-exposed post-hoc diagnostics, not unseen held-out test claims.\n\nImportant input-capacity disclosure: the trained adapters accept at most248 total frames. Rolling inference retains up to196 history frames and52 generated frames; only remaining native capacity receives future observations. Globally provided goals are not compressed into earlier tokens. INFERENCE_EXECUTION_AUDIT.json records actual per-window visibility. Thus these results describe the complete fixed-context systems on the same requested tasks, and are not an equal-full-future-conditioning architecture ablation. Do not infer inferior long-term planning from a hidden input interface limitation. No new planner, synthetic path or retraining was introduced.\n\nThe qualified evaluator_v2 scores pooled local52-window FID and32-way R@3 using the same two encoders, deduplicated captions and three fixed candidate pools as existing results. Per-window CSVs retain all points. Minimal is excluded from main comparison.\n'''
 (ROOT/'REPORT.md').write_text(report)
 write(ROOT/'NUMERICS_COMPLETE.json',dict(status='COMPLETE_TWO_FROZEN100K_FIXED_CONTEXT_BASELINES',N_sources=506,new_trajectories=7084,models=MODELS,tasks=TASKS,no_root_ADE=True,full_future_conditioning_equivalent=False,time=time.time(),hashes={n:sha(OUT/n) for n in ['ENDPOINT_SUMMARY.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv','COMPARISON_ENDPOINTS.csv']}))
if __name__=='__main__':main()
