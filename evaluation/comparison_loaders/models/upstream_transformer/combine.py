"""Preserve existing four-model results; replace only official aligned9/20/40s rows."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,time
import numpy as np
import pandas as pd
from util import ROOT,TASKS,read,write,sha
OLD=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
PRIOR_OFFICIAL=Path(artifact_path('horizon_controlled_motion_mamba2_official_streaming_20261017_5c22e1a818'))
def main():
 out=ROOT/'comparison';a=out/'aggregate';a.mkdir(parents=True,exist_ok=True)
 assert read(ROOT/'NUMERICS_COMPLETE.json')['new_trajectories']==1518
 oldproof=read(OLD/'NUMERICS_COMPLETE.json')
 for n,h in oldproof['result_hashes'].items():assert sha(OLD/'aggregate'/n)==h
 checks=[]
 for t in TASKS:
  with np.load(ROOT/'aggregate'/t/'motion_mamba2_official_bank.npz') as x,np.load(OLD/'aggregate'/t/'motion_mamba2_bank.npz') as y:
   for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(x[k],y[k]),(t,k)
   deltas={}
   for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:
    deltas[k]=float(np.max(abs(x[k]-y[k])));assert np.allclose(x[k],y[k],atol=1e-4,rtol=1e-5),(t,k)
  p=read(ROOT/'aggregate'/t/'QUERY_POOLS.json');q=read(OLD/'aggregate'/t/'QUERY_POOLS.json')
  assert p['pools']==q['pools'] and p['query_source_ids']==q['query_source_ids']
  checks.append(dict(task=t,reference_feature_max_deltas=deltas,identical_query_ids_and_pools=True))
 write(out/'COMPARABILITY_AUDIT.json',dict(status='PASS_SAME_SOURCES_REFERENCES_EVALUATOR_AND_RETRIEVAL_POOLS',checks=checks,time=time.time()))
 for n in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']:
  old=pd.read_csv(OLD/'aggregate'/n);old=old[old.task.isin(TASKS)]
  pd.concat([pd.read_csv(ROOT/'aggregate'/n),old],ignore_index=True).to_csv(a/n,index=False)
 ce=pd.read_csv(a/'ENDPOINT_SUMMARY.csv');fid=pd.read_csv(a/'ALL_WINDOW_LOCALFID52.csv');ret=pd.read_csv(a/'ALL_WINDOW_RETRIEVAL.csv')
 rows=[]
 for t in TASKS:
  final=int(read(ROOT/'PLAN.json')['tasks'][t]['generated_frames']/52)
  for m in ['motion_mamba2_official','ardy','minimal','native110','native117']:
   c=ce[(ce.task==t)&(ce.model==m)&(ce.stratum=='ALL')].iloc[0]
   f=fid[(fid.task==t)&(fid.model==m)&(fid.window==final)].iloc[0]
   r=ret[(ret.task==t)&(ret.model==m)&(ret.window==final)&(ret.encoder_seed==1701)].iloc[0]
   rows.append(dict(task=t,model=m,windows=final,goal_time_including_prefix_sec=c.goal_time_including_prefix_sec,endpoint_CE_cm=c.constraint_error_cm,
    final_window_LocalFID52=f.LocalFID52,final_window_R3_percent=r.R3*100,N_sources=506,N_queries=int(r.N_queries)))
 pd.DataFrame(rows).to_csv(out/'FINAL_WINDOW_COMPARISON.csv',index=False)
 oldfid=pd.read_csv(PRIOR_OFFICIAL/'aggregate/ALL_WINDOW_LOCALFID52.csv');oldr=pd.read_csv(PRIOR_OFFICIAL/'aggregate/ALL_WINDOW_RETRIEVAL.csv')
 delta=[]
 for x in rows:
  if x['model']!='motion_mamba2_official':continue
  t=x['task'];w=x['windows'];f=oldfid[(oldfid.task==t)&(oldfid.window==w)].iloc[0];r=oldr[(oldr.task==t)&(oldr.window==w)&(oldr.encoder_seed==1701)].iloc[0]
  delta.append(dict(task=t,old_final_LocalFID52=f.LocalFID52,corrected_final_LocalFID52=x['final_window_LocalFID52'],old_final_R3_percent=r.R3*100,corrected_final_R3_percent=x['final_window_R3_percent']))
 pd.DataFrame(delta).to_csv(out/'OFFICIAL_BEFORE_AFTER.csv',index=False)
 write(ROOT/'COMPARISON_COMPLETE.json',dict(status='COMPLETE_CORRECTED_FIVE_MODEL_ALIGNED_HORIZON_METRICS',comparison_root=str(out),sources=506,actors=22,tasks=TASKS,time=time.time()))
 print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
