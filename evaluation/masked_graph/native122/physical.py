"""Summarize verified decoded-motion window diagnostics for the requested models."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import hashlib,json,time
import pandas as pd
from util import ROOT,TASKS,read,write,sha

OLD=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
FIXED=Path(artifact_path('transformer_reference100k_corrected_evaluation_20261017_7a91e02b7e'))
SOURCES={'ardy':FIXED,'minimal':OLD,'native122':ROOT,'native110':OLD,'native117':OLD}

def main():
 assert (ROOT/'COMPARISON_COMPLETE.json').exists()
 physical=[];seams=[];bindings=[]
 for model,source in SOURCES.items():
  for task in TASKS:
   for c in [x for x in read(ROOT/'PLAN.json')['catalog'] if x['task']==task]:
    d=source/'evaluation'/model/task/f'shard{c["shard"]:02d}'
    proof=read(d/'WINDOW_SCORE_COMPLETE.json');assert proof['N']==c['N']
    assert sha(d/'WINDOW_PHYSICAL_METRICS.json')==proof['metrics_sha256']
    assert sha(d/'WINDOW_EMBEDDINGS.npz')==proof['features_sha256']
    z=read(d/'WINDOW_PHYSICAL_METRICS.json')
    physical += [dict(model=model,task=task,**x) for x in z['window_metrics'] if x['role']=='generated']
    seams += [dict(model=model,task=task,**x) for x in z['seams'] if x['role']=='generated']
    bindings.append(dict(model=model,task=task,shard=c['shard'],score_receipt_sha256=sha(d/'WINDOW_SCORE_COMPLETE.json')))
 ph=pd.DataFrame(physical);ss=pd.DataFrame(seams)
 out=ROOT/'comparison/aggregate'
 windows=ph.groupby(['task','model','window'],as_index=False).agg(
  sliding_proxy_percent=('foot_sliding_proxy_percent','mean'),
  contact_support_percent=('predicted_support_fraction_percent','mean'),
  joint_second_difference_m_per_frame2=('joint_second_difference_m_per_frame2','mean'),
  sliding_slots=('sliding_foot_frames','sum'),contact_slots=('predicted_support_foot_frames','sum'),
  N=('source_id','count'))
 assert (windows.N==506).all()
 windows['support_conditional_sliding_percent']=100*windows.sliding_slots/windows.contact_slots.replace(0,float('nan'))
 windows.to_csv(out/'WINDOW_PHYSICAL_MEANS.csv',index=False)
 sm=ss.groupby(['task','model','entering_window'],as_index=False).agg(
  root_velocity_discontinuity_m_s=('root_velocity_discontinuity_m_s','mean'),
  root_interframe_step_m=('root_interframe_step_m','mean'),
  joint_rotation_step_rms_deg=('local_joint_rotation_step_rms_deg','mean'),N=('source_id','count'))
 assert (sm.N==506).all();sm.to_csv(out/'WINDOW_SEAM_MEANS.csv',index=False)
 write(ROOT/'PHYSICAL_COMPLETE.json',dict(status='COMPLETE_VERIFIED_FIVE_MODEL_WINDOW_PHYSICAL_DIAGNOSTICS',
  bindings=bindings,slip_definition='horizontal foot speed>0.15m/s AND raw predicted contact>0.5; denominator all foot-frame slots',
  contact_support_is_predicted=True,not_physics_or_viewer_seam_acceptance=True,
  result_hashes={n:sha(out/n) for n in ['WINDOW_PHYSICAL_MEANS.csv','WINDOW_SEAM_MEANS.csv']},time=time.time()))
 print('PHYSICAL_COMPLETE')

if __name__=='__main__':main()
