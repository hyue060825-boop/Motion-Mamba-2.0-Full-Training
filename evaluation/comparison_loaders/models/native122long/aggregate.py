"""Same frozen evaluator; compare to identical existing real features/pools."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import csv,time,sys,concurrent.futures
import numpy as np
from util import ROOT,TASKS,read,write,sha
import aggregate_reference as ref
sys.path.insert(0,artifact_path('evaluator_native_v2_a933d7fb8b'))
from native_eval.metrics import candidate_pools,retrieval
from native_eval.v2_statistics import fid_to_many

OLD=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
OUT=ROOT/'aggregate';OUT.mkdir(exist_ok=True)

def quality(task):
 b=ref.bank('native122long',task);dest=OUT/task
 with np.load(OLD/'aggregate'/task/'native110_bank.npz') as prior:
  for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(b[k],prior[k]),(task,k)
  for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:
   assert np.allclose(b[k],prior[k],atol=1e-4,rtol=1e-5),(task,k)
 selected=[];seen=set()
 for i in sorted(range(506),key=lambda i:str(b['source_ids'][i])):
  caption=str(b['captions'][i])
  if caption not in seen:seen.add(caption);selected.append(i)
 q=np.array(selected)
 records=[dict(sample_id=str(b['source_ids'][i]),source_family_id=str(b['source_ids'][i]),caption=str(b['captions'][i]),frames=52) for i in q]
 pools={str(s):candidate_pools(records,s,length_matched=True) for s in [4101,4102,4103]}
 previous=read(OLD/'aggregate'/task/'QUERY_POOLS.json')
 assert previous['query_source_ids']==[x['sample_id'] for x in records] and previous['pools']==pools
 write(dest/'QUERY_POOLS.json',dict(query_indices=selected,query_source_ids=previous['query_source_ids'],pools=pools,scope=previous['scope']))
 frows=[];rrows=[]
 for w,start in enumerate(b['starts']):
  cache=dest/f'quality_w{w+1:02d}.json'
  if cache.exists():
   c=read(cache);frows+=c['FID'];rrows+=c['retrieval'];continue
  fid=float(fid_to_many(b['distribution_real'][w],[b['distribution_generated'][w]])[0])
  fr=[dict(task=task,model='native122long',window=w+1,start_generated_frame=int(start),end_generated_frame_exclusive=int(start+52),
           N_sources=506,LocalFID52=fid,uncertainty='point estimate only; no calibrated FID CI claimed',
           scope='same natural-reference local52 distribution; not whole-motion FID')]
  actors=b['actor_ids'][q];rng=np.random.default_rng(3101701+w)
  draws=[ref.cluster_draw(actors,rng) for _ in range(2000)];rr=[]
  for seed in [1701,2903]:
   tt=b[f'text_seed{seed}'][w,q]
   real=np.array([retrieval(tt,b[f'retrieval_real_seed{seed}'][w,q],p)[1] for p in pools.values()])
   ranks=np.array([retrieval(tt,b[f'retrieval_generated_seed{seed}'][w,q],p)[1] for p in pools.values()])
   r=dict(task=task,model='native122long',window=w+1,encoder_seed=seed,N_queries=len(q),candidates=32,
          source_pool_seeds='4101/4102/4103',scope='local active-instruction compatibility, not event-order accuracy')
   for k in [1,2,3]:
    success=(ranks<=k).mean(axis=0);bs=np.array([success[ix].mean() for ix in draws]);lo,hi=np.quantile(bs,[.025,.975])
    r.update({f'R{k}':float(success.mean()),f'R{k}_ci95_low':float(lo),f'R{k}_ci95_high':float(hi),f'real_R{k}':float((real<=k).mean())})
   rr.append(r)
  write(cache,dict(FID=fr,retrieval=rr,window=w+1,time=time.time()));frows+=fr;rrows+=rr
  print('QUALITY_WINDOW',task,w+1,len(b['starts']),flush=True)
 ref.csvwrite(dest/'WINDOW_FID.csv',frows);ref.csvwrite(dest/'WINDOW_RETRIEVAL.csv',rrows)
 write(dest/'COMPLETE.json',dict(status='COMPLETE_NATIVE122K-LONG_LOCAL52_QUALITY',N=506,windows=len(b['starts']),queries=len(q),time=time.time()))

def main():
 plan=read(ROOT/'PLAN.json');checks=[]
 for c in plan['catalog']:
  d=ROOT/'evaluation/native122long'/c['task']/f'shard{c["shard"]:02d}'
  load=read(d/'LOAD.json');command=read(d/'COMMAND.json')
  assert load['global_step']==122000 and load['all_loaded_denoiser_weights_verified']
  assert load['generation_contract']=='g1_committed_v3' and load['decoder_history_frames']==52
  assert command['checkpoint_sha256']==read(ROOT/'CHECKPOINTS.json')['native122long']['sha256']
  checks.append(dict(task=c['task'],shard=c['shard'],N=c['N'],step=load['global_step'],
                     architecture='Native Hybrid',weights_exact=True,committed_decoder_frames=52))
 write(ROOT/'INFERENCE_EXECUTION_AUDIT.json',dict(status='PASS_NATIVE122K-LONG_CHECKPOINT_ARCHITECTURE_AND_COMMITTED_ROLLOUT',checks=checks,
  windows_commits_checked_inside_unchanged_sampler=True,training_updates=0,time=time.time()))
 ref.endpoints()
 (OUT/'PAIRED_MODEL_CONTRASTS.csv').write_text('task,stratum,model,reference,N_sources,mean_CE_difference_cm,paired_ci95_low,paired_ci95_high\n')
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:list(pool.map(quality,TASKS))
 f=[];r=[]
 for task in TASKS:
  f+=list(csv.DictReader((OUT/task/'WINDOW_FID.csv').open()));r+=list(csv.DictReader((OUT/task/'WINDOW_RETRIEVAL.csv').open()))
 ref.csvwrite(OUT/'ALL_WINDOW_LOCALFID52.csv',f);ref.csvwrite(OUT/'ALL_WINDOW_RETRIEVAL.csv',r)
 write(ROOT/'NUMERICS_COMPLETE.json',dict(status='COMPLETE_NATIVE122K-LONG_ENDPOINTS_AND_FROZEN_EVALUATOR_V2',
  new_trajectories=1518,N_sources=506,actors=22,models=['native122long'],tasks=TASKS,training_updates=0,no_root_ADE=True,
  source_exposure='known generator training sources; posthoc controlled diagnosis, not unseen-source evaluation',
  endpoint_audit=sha(OUT/'ENDPOINT_AUDIT.json'),
  result_hashes={n:sha(OUT/n) for n in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','PAIRED_MODEL_CONTRASTS.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']},time=time.time()))
 print('NUMERICS_COMPLETE',flush=True)

if __name__=='__main__':main()
