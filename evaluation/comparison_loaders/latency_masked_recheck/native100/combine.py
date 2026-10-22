"""Five requested models only; use the corrected self-trainedARDY results."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import csv,json,time,sys
import numpy as np
import pandas as pd
from util import ROOT,TASKS,read,write,sha
import aggregate_reference as ref

PREVIOUS=Path(artifact_path('transformer_reference100k_corrected_evaluation_20261017_7a91e02b7e'))
OLD=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
MODELS=['ardy','minimal','native100','native110','native117']

def main():
 assert read(ROOT/'NUMERICS_COMPLETE.json')['new_trajectories']==1518
 assert read(PREVIOUS/'FINAL_VERIFICATION.json')['training_updates']==0
 dest=ROOT/'comparison';out=dest/'aggregate';out.mkdir(parents=True,exist_ok=True)
 checks=[]
 for task in TASKS:
  with np.load(ROOT/'aggregate'/task/'native100_bank.npz') as new,np.load(PREVIOUS/'aggregate'/task/'motion_mamba2_bank.npz') as prev:
   for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(new[k],prev[k]),(task,k)
   deltas={}
   for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:
    deltas[k]=float(np.max(abs(new[k]-prev[k])));assert np.allclose(new[k],prev[k],atol=1e-4,rtol=1e-5),(task,k)
  qp=read(ROOT/'aggregate'/task/'QUERY_POOLS.json');pp=read(PREVIOUS/'aggregate'/task/'QUERY_POOLS.json')
  assert qp['query_source_ids']==pp['query_source_ids'] and qp['pools']==pp['pools']
  checks.append(dict(task=task,identical_query_pools=True,real_feature_deltas=deltas))
 verified={}
 for name in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']:
  priorpath=PREVIOUS/'comparison/aggregate'/name
  # Rebuild the prior table from its independently hashed sources, not a mutable unchecked CSV.
  fixed=pd.read_csv(PREVIOUS/'aggregate'/name)
  raw=pd.read_csv(OLD/'aggregate'/name)
  oldproof=read(OLD/'NUMERICS_COMPLETE.json')
  if name in oldproof['result_hashes']:assert sha(OLD/'aggregate'/name)==oldproof['result_hashes'][name]
  newproof=read(PREVIOUS/'NUMERICS_COMPLETE.json')
  if name in newproof['result_hashes']:assert sha(PREVIOUS/'aggregate'/name)==newproof['result_hashes'][name]
  prior=pd.concat([fixed,raw[raw.model.isin(['minimal','native110','native117'])]],ignore_index=True)
  prior=prior[prior.task.isin(TASKS)&prior.model.isin(MODELS)]
  mine=pd.read_csv(ROOT/'aggregate'/name)
  pd.concat([prior,mine],ignore_index=True).to_csv(out/name,index=False)
  verified[name]=dict(corrected_motion_mamba2_source=sha(PREVIOUS/'aggregate'/name),other_three_source=sha(OLD/'aggregate'/name),new_native100_source=sha(ROOT/'aggregate'/name))
 write(dest/'COMPARABILITY_AUDIT.json',dict(status='PASS_MATCHED_INPUTS_REFERENCES_POOLS_AND_CORRECTED_ARDY',
  checks=checks,provenance=verified,official_motion_mamba2_not_mixed_into_requested_five_models=True,time=time.time()))
 ce=pd.read_csv(out/'ENDPOINT_SUMMARY.csv');fid=pd.read_csv(out/'ALL_WINDOW_LOCALFID52.csv');ret=pd.read_csv(out/'ALL_WINDOW_RETRIEVAL.csv')
 final=[]
 for task in TASKS:
  w=int(read(ROOT/'PLAN.json')['tasks'][task]['generated_frames']//52)
  for model in MODELS:
   c=ce[(ce.task==task)&(ce.model==model)&(ce.stratum=='ALL')].iloc[0]
   f=fid[(fid.task==task)&(fid.model==model)&(fid.window==w)].iloc[0]
   r=ret[(ret.task==task)&(ret.model==model)&(ret.window==w)&(ret.encoder_seed==1701)].iloc[0]
   final.append(dict(task=task,model=model,windows=w,time_sec=float(c.goal_time_including_prefix_sec),
    endpoint_CE_cm=c.constraint_error_cm,CE_ci95_low=c.CE_ci95_low,CE_ci95_high=c.CE_ci95_high,
    success20cm_percent=c.success_20cm*100,final_window_LocalFID52=f.LocalFID52,final_window_R3_percent=r.R3*100,
    N_sources=506,N_queries=int(r.N_queries)))
 pd.DataFrame(final).to_csv(dest/'FINAL_WINDOW_COMPARISON.csv',index=False)
 # Paired differences relative toNative100k, with identical actor/source bootstrap draws.
 per=pd.read_csv(out/'ENDPOINT_PER_SOURCE.csv');contrasts=[]
 for task in TASKS:
  d=per[per.task==task];base=d[d.model=='native100'].sort_values('source_id');ids=base.source_id.tolist()
  actors=base.actor_uid.to_numpy();rng=np.random.default_rng(2026101700+int(task[-2:]));draws=[ref.cluster_draw(actors,rng) for _ in range(2000)]
  for model in [m for m in MODELS if m!='native100']:
   v=d[d.model==model].sort_values('source_id');assert v.source_id.tolist()==ids
   delta=v.constraint_error_cm.to_numpy()-base.constraint_error_cm.to_numpy()
   bs=np.array([delta[ix].mean() for ix in draws]);lo,hi=np.quantile(bs,[.025,.975])
   contrasts.append(dict(task=task,model=model,reference='native100',N_sources=506,mean_CE_difference_cm=float(delta.mean()),
                         paired_ci95_low=float(lo),paired_ci95_high=float(hi),definition='model minusNative100;positive favorsNative100'))
 pd.DataFrame(contrasts).to_csv(out/'PAIRED_NATIVE100_CONTRASTS.csv',index=False)
 write(ROOT/'COMPARISON_COMPLETE.json',dict(status='COMPLETE_REQUESTED_FIVE_MODEL_FIRST_EXPERIMENT',
   models=MODELS,new_native100_trajectories=1518,sources_per_condition=506,tasks=TASKS,
   comparison=str(dest),result_hashes={n:sha(out/n) for n in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv','PAIRED_NATIVE100_CONTRASTS.csv']},time=time.time()))
 print(pd.DataFrame(final).to_string(index=False),flush=True)

if __name__=='__main__':main()
