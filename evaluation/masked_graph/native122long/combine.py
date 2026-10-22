"""Frozen same506 endpoint cases; short-distill control and official teacher."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import numpy as np
import pandas as pd
import time
from util import ROOT,TASKS,read,write,sha
import aggregate_reference as ref
BASES={
 'native117':Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d')),
 'native122':Path(artifact_path('native122k_controlled_endpoint_20261018_9d1c42d4fe')),
 'motion_mamba2_official':Path(artifact_path('horizon_controlled_motion_mamba2_official_corrected_20261017_633fa66262'))}
NEW='native122long'
def main():
 assert read(ROOT/'NUMERICS_COMPLETE.json')['new_trajectories']==1518
 out=ROOT/'comparison/aggregate';out.mkdir(parents=True,exist_ok=True);checks=[]
 frozen=read(ROOT/'REFERENCE_CONTROL_RESULTS.json')
 for model,base in BASES.items():
  for task in TASKS:
   with np.load(ROOT/'aggregate'/task/(NEW+'_bank.npz')) as a,np.load(base/'aggregate'/task/(model+'_bank.npz')) as b:
    for k in ['sample_ids','source_ids','actor_ids','captions','starts']:assert np.array_equal(a[k],b[k]),(model,task,k)
    for k in ['distribution_real','retrieval_real_seed1701','retrieval_real_seed2903','text_seed1701','text_seed2903']:assert np.allclose(a[k],b[k],atol=1e-4,rtol=1e-5),(model,task,k)
   qa=read(ROOT/'aggregate'/task/'QUERY_POOLS.json');qb=read(base/'aggregate'/task/'QUERY_POOLS.json')
   assert qa['query_source_ids']==qb['query_source_ids'] and qa['pools']==qb['pools']
   checks.append(dict(model=model,task=task,same_samples_targets_phases_seeds_and_evaluator_pools=True))
 for name in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']:
  frames=[pd.read_csv(ROOT/'aggregate'/name)]
  for model,base in BASES.items():
   assert sha(base/'aggregate'/name)==frozen[model]['files'][name]
   df=pd.read_csv(base/'aggregate'/name);frames.append(df[df.model==model])
  pd.concat(frames,ignore_index=True).to_csv(out/name,index=False)
 ce=pd.read_csv(out/'ENDPOINT_SUMMARY.csv');fid=pd.read_csv(out/'ALL_WINDOW_LOCALFID52.csv');ret=pd.read_csv(out/'ALL_WINDOW_RETRIEVAL.csv');per=pd.read_csv(out/'ENDPOINT_PER_SOURCE.csv')
 rows=[];paired=[]
 for task in TASKS:
  w=read(ROOT/'PLAN.json')['tasks'][task]['goal_window']
  for model in [*BASES,NEW]:
   c=ce[(ce.task==task)&(ce.model==model)&(ce.stratum=='ALL')].iloc[0]
   f=fid[(fid.task==task)&(fid.model==model)&(fid.window==w)].iloc[0]
   r=ret[(ret.task==task)&(ret.model==model)&(ret.window==w)&(ret.encoder_seed==1701)].iloc[0]
   rows.append(dict(model=model,task=task,time_sec=c.goal_time_including_prefix_sec,endpoint_CE_cm=c.constraint_error_cm,success20cm_percent=100*c.success_20cm,
     final_window_LocalFID52=f.LocalFID52,final_window_R3_percent=100*r.R3,N=506))
  b=per[(per.task==task)&(per.model==NEW)].sort_values('source_id');assert len(b)==506
  for model in BASES:
   a=per[(per.task==task)&(per.model==model)].sort_values('source_id');assert a.source_id.tolist()==b.source_id.tolist()
   delta=b.constraint_error_cm.to_numpy()-a.constraint_error_cm.to_numpy();actors=a.actor_uid.to_numpy();rng=np.random.default_rng(2026101800+w)
   boot=np.array([delta[ref.cluster_draw(actors,rng)].mean() for _ in range(2000)]);lo,hi=np.quantile(boot,[.025,.975])
   paired.append(dict(task=task,reference=model,model=NEW,mean_CE_difference_cm=float(delta.mean()),paired_ci95_low=float(lo),paired_ci95_high=float(hi),N=506))
 pd.DataFrame(rows).to_csv(ROOT/'comparison/FINAL_WINDOW_COMPARISON.csv',index=False)
 pd.DataFrame(paired).to_csv(out/'PAIRED_LONG_DISTILL.csv',index=False)
 write(ROOT/'comparison/COMPARABILITY_AUDIT.json',dict(passed=True,checks=checks,old_results_reused=True,not_unseen_sources=True))
 write(ROOT/'COMPARISON_COMPLETE.json',dict(status='COMPLETE_LONG_DISTILL117K_PAIRED_WITH_SHORT122K_AND_TEACHER',models=[*BASES,NEW],new_trajectories=1518,time=time.time()))
 report='# Native117k：长轨迹教师蒸馏5000步验收\n\n同117k起点、同5000更新、同batch512与原四项损失，改变同状态教师材料的剩余时间覆盖。对照为原117k、原短时协议122k及官方ARDY，旧结果只读复用。\n\n```text\n'+pd.DataFrame(rows).to_string(index=False)+'\n```\n\n配对演员/来源bootstrap2000次，差值=新长时蒸馏−对应参考：\n\n```text\n'+pd.DataFrame(paired).to_string(index=False)+'\n```\n\n上述506条固定HiPHI材料有生成器训练来源暴露，仅用于配对诊断，不能当作未见来源的泛化结果。所有方法使用相同空间目标、窗口位置、文本、噪声与evaluator_v2。质量为逐窗口LocalFID52/R@3，不是整段40秒FID，也不是长记忆证明。终点约束误差不惩罚合理中途路径，不报告Root ADE主指标。必须结合训练目录的基础功能、完整交互、接触与接缝结果判断，不自动替换线上仿真。\n'
 (ROOT/'REPORT.md').write_text(report)
 print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
