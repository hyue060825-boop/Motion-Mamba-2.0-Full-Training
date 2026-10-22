"""Paired117k/122k: exact same cases, targets, phases, evaluator and retrieval pools."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import time,sys,json
import numpy as np
import pandas as pd
from util import ROOT,TASKS,read,write,sha
import aggregate_reference as ref
OLD=Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d'))
def main():
 assert read(ROOT/'NUMERICS_COMPLETE.json')['new_trajectories']==1518
 out=ROOT/'comparison/aggregate';out.mkdir(parents=True,exist_ok=True);checks=[]
 for task in TASKS:
  with np.load(ROOT/'aggregate'/task/'native122_bank.npz') as new,np.load(OLD/'aggregate'/task/'native117_bank.npz') as prev:
   for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(new[k],prev[k]),(task,k)
   for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:assert np.allclose(new[k],prev[k],atol=1e-4,rtol=1e-5),(task,k)
  qp=read(ROOT/'aggregate'/task/'QUERY_POOLS.json');pp=read(OLD/'aggregate'/task/'QUERY_POOLS.json')
  assert qp['query_source_ids']==pp['query_source_ids'] and qp['pools']==pp['pools']
  checks.append(dict(task=task,identical_samples_goals_real_features_and_query_pools=True))
 for name in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']:
  proof=read(OLD/'NUMERICS_COMPLETE.json')
  assert sha(OLD/'aggregate'/name)==read(ROOT/'REFERENCE_RESULTS.json')['files'][name]
  if name in proof['result_hashes']:assert sha(OLD/'aggregate'/name)==proof['result_hashes'][name]
  old=pd.read_csv(OLD/'aggregate'/name);new=pd.read_csv(ROOT/'aggregate'/name)
  pd.concat([old[old.model=='native117'],new],ignore_index=True).to_csv(out/name,index=False)
 ce=pd.read_csv(out/'ENDPOINT_SUMMARY.csv');fid=pd.read_csv(out/'ALL_WINDOW_LOCALFID52.csv');ret=pd.read_csv(out/'ALL_WINDOW_RETRIEVAL.csv');per=pd.read_csv(out/'ENDPOINT_PER_SOURCE.csv')
 rows=[];paired=[]
 for task in TASKS:
  w=read(ROOT/'PLAN.json')['tasks'][task]['goal_window']
  for model in ['native117','native122']:
   c=ce[(ce.task==task)&(ce.model==model)&(ce.stratum=='ALL')].iloc[0]
   f=fid[(fid.task==task)&(fid.model==model)&(fid.window==w)].iloc[0]
   r=ret[(ret.task==task)&(ret.model==model)&(ret.window==w)&(ret.encoder_seed==1701)].iloc[0]
   rows.append(dict(model=model,task=task,time_sec=c.goal_time_including_prefix_sec,endpoint_CE_cm=c.constraint_error_cm,success20cm_percent=100*c.success_20cm,final_window_LocalFID52=f.LocalFID52,final_window_R3_percent=100*r.R3,N=506))
  a=per[(per.task==task)&(per.model=='native117')].sort_values('source_id');b=per[(per.task==task)&(per.model=='native122')].sort_values('source_id');assert a.source_id.tolist()==b.source_id.tolist()
  assert len(a)==len(b)==506
  for label,values in [('native117',a),('native122',b)]:
   expected=ce[(ce.task==task)&(ce.model==label)&(ce.stratum=='ALL')].iloc[0].constraint_error_cm
   assert abs(values.constraint_error_cm.mean()-expected)<1e-5
  delta=b.constraint_error_cm.to_numpy()-a.constraint_error_cm.to_numpy();actors=a.actor_uid.to_numpy();rng=np.random.default_rng(2026101800+w)
  bs=np.array([delta[ref.cluster_draw(actors,rng)].mean() for _ in range(2000)]);lo,hi=np.quantile(bs,[.025,.975])
  paired.append(dict(task=task,reference='native117',model='native122',mean_CE_difference_cm=float(delta.mean()),paired_ci95_low=float(lo),paired_ci95_high=float(hi),N=506))
 pd.DataFrame(rows).to_csv(ROOT/'comparison/FINAL_WINDOW_COMPARISON.csv',index=False)
 pd.DataFrame(paired).to_csv(out/'PAIRED117K_TO122K.csv',index=False)
 write(ROOT/'comparison/COMPARABILITY_AUDIT.json',dict(passed=True,checks=checks,old117k_results_reused=True,not_new_unseen_sources=True))
 write(ROOT/'COMPARISON_COMPLETE.json',dict(status='COMPLETE_PAIRED117K122K_ENDPOINT_AND_EVALUATOR_V2',models=['native117','native122'],new_trajectories=1518,checks=checks,time=time.time()))
 table=pd.DataFrame(rows).to_string(index=False);contrast=pd.DataFrame(paired).to_string(index=False)
 report='# Native117k再微调5000步：固定测评结果\n\n实际训练117000→122000，八卡global batch512，继承Adam/FP32master/RNG及原117k配方。\n\n下表仅使用原506条固定HiPHI诊断材料；相同目标坐标、末窗位置、噪声和evaluator_v2。原117k成绩复用已核验结果。样本有训练来源暴露，不能写作未见样本的独立泛化结果。\n\n```text\n'+table+'\n```\n\n逐演员/来源配对2000次bootstrap，负值表示122k终点约束误差更低：\n\n```text\n'+contrast+'\n```\n\n质量指标为52帧逐窗LocalFID52和R@3，不是整个40秒序列FID，也不是独立长记忆证明。Root ADE不作为主指标。基础功能与完整交互结果另见训练目录evaluation/5000与evaluation/10000，同候选所有指标如实报告，不以loss下降宣布成功。\n'
 (ROOT/'REPORT.md').write_text(report)
 print(table,flush=True);print(contrast,flush=True)
if __name__=='__main__':main()
