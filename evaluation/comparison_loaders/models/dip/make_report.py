"""Publish completed frozen results; do not tune or select checkpoints."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import csv,json,collections,statistics,zipfile
R=Path(artifact_path('g1_external_matched100k_evaluation_20261019_aa0c66c5b2'))
OUT=R/'publication';OUT.mkdir(exist_ok=True)
assert (R/'NUMERICS_COMPLETE.json').exists()
def rows(p):return list(csv.DictReader(p.open()))
def writecsv(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def read(p):return json.loads(p.read_text())
for n in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','ENDPOINT_AUDIT.json','PAIRED_MODEL_CONTRASTS.csv']:
 (OUT/n).write_bytes((R/'aggregate'/n).read_bytes())
writecsv(OUT/'PHASE_ENDPOINTS.csv',[r for r in rows(R/'aggregate/ENDPOINT_SUMMARY.csv') if r['task'].startswith('phase_')])
sources=[(R,{'dip':'DiP-G1 (100k)','dart':'DART-MLP-G1 (100k)'}),
 (Path(artifact_path('transformer_reference100k_corrected_evaluation_20261017_7a91e02b7e')),{'ardy':'ARDY-G1 (100k, matched)'}),
 (Path(artifact_path('native100k_controlled_endpoint_20261017_4c3b70479f')),{'native100':'Native (100k)'}),
 (Path(artifact_path('horizon_controlled_four_models_20261017_132d28341d')),{'native110':'Native (110k)','native117':'Native (117k)'}),
 (Path(artifact_path('horizon_controlled_motion_mamba2_official_corrected_20261017_633fa66262')),{'motion_mamba2_official':'ARDY (official release)'})]
tables=[];allfid=[];allr=[];paired={}
for root,mapping in sources:
 endpoint=root/'aggregate/ENDPOINT_SUMMARY.csv'
 if not endpoint.exists():continue
 fs=rows(root/'aggregate/ALL_WINDOW_LOCALFID52.csv');rs=rows(root/'aggregate/ALL_WINDOW_RETRIEVAL.csv')
 for rr,dest in [(fs,allfid),(rs,allr)]:
  for r in rr:
   if r['model'] in mapping:dest.append(dict(r,method=mapping[r['model']],result_root=str(root)))
 for r in rows(endpoint):
  m=r['model'];t=r['task']
  if m not in mapping or r['stratum']!='ALL' or t not in ['aligned_w04','aligned_w09','aligned_w19']:continue
  W={'aligned_w04':4,'aligned_w09':9,'aligned_w19':19}[t]
  f=[v for v in fs if v['model']==m and v['task']==t and int(v['window'])==W]
  rr=[v for v in rs if v['model']==m and v['task']==t and int(v['window'])==W]
  assert len(f)==1 and len(rr)==2,(root,m,t,len(f),len(rr))
  tables.append(dict(method=mapping[m],model=m,task=t,duration_sec=r['goal_time_including_prefix_sec'],constraint_error_cm=r['constraint_error_cm'],CE_ci95_low=r['CE_ci95_low'],CE_ci95_high=r['CE_ci95_high'],
   success20_percent=100*float(r['success_20cm']),final_window=W,final_LocalFID52=f[0]['LocalFID52'],final_R3_percent=100*statistics.mean(float(v['R3']) for v in rr),
   encoder1701_R3_percent=100*float(next(v['R3'] for v in rr if v['encoder_seed']=='1701')),encoder2903_R3_percent=100*float(next(v['R3'] for v in rr if v['encoder_seed']=='2903')),
   conditioning='native capacity248 frames' if root==R else 'existing frozen inference contract',result_root=str(root)))
 # Check paired identities/goals/noise against the new results, not mere N equality.
 p=root/'aggregate/ENDPOINT_PER_SOURCE.csv'
 if p.exists():
  for r in rows(p):
   if r['model'] not in mapping:continue
   key=(r['task'],r['source_id']);v=(r['noise_seed'],json.loads(r['goal_root_xz_m']))
   if key in paired:assert paired[key]==v,(root,key)
   else:paired[key]=v
writecsv(OUT/'MAIN_ALIGNED_ENDPOINT_AND_FINAL_WINDOW_QUALITY.csv',tables)
writecsv(OUT/'ALL_WINDOW_LOCALFID52.csv',allfid);writecsv(OUT/'ALL_WINDOW_RETRIEVAL.csv',allr)
# New-model contact and seam summaries, preserving zero-support cases.
physical=collections.defaultdict(list);seams=collections.defaultdict(list)
for p in (R/'evaluation').glob('*/*/shard*/WINDOW_PHYSICAL_METRICS.json'):
 model,task=p.parts[-4:-2];d=read(p)
 for r in d['window_metrics']:
  if r['role']=='generated':physical[(model,task,r['window'])].append(r)
 for r in d['seams']:
  if r['role']=='generated':seams[(model,task,r['entering_window'])].append(r)
pr=[]
for (m,t,w),rr in sorted(physical.items()):
 assert len(rr)==506
 bad=sum(x['sliding_foot_frames'] for x in rr);support=sum(x['predicted_support_foot_frames'] for x in rr);slots=sum(x['total_foot_frames'] for x in rr)
 ss=seams.get((m,t,w),[])
 pr.append(dict(model=m,task=t,window=w,N=506,slip_proxy_percent=100*bad/slots,predicted_support_percent=100*support/slots,
  support_conditional_slip_percent=100*bad/support if support else '',no_predicted_support_cases=sum(x['predicted_support_foot_frames']==0 for x in rr),
  joint_second_difference_m_per_frame2=statistics.mean(x['joint_second_difference_m_per_frame2'] for x in rr),
  seam_velocity_difference_m_s=statistics.mean(x['root_velocity_discontinuity_m_s'] for x in ss) if ss else ''))
writecsv(OUT/'NEW_MODELS_WINDOW_PHYSICAL_QUALITY.csv',pr)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
fig,axs=plt.subplots(1,3,figsize=(13,3.4))
methods=list(dict.fromkeys(r['method'] for r in tables))
for method in methods:
 rr=sorted([r for r in tables if r['method']==method],key=lambda r:float(r['duration_sec']))
 axs[0].plot([float(r['duration_sec']) for r in rr],[float(r['constraint_error_cm']) for r in rr],marker='o',markersize=3,label=method)
 fr=sorted([r for r in allfid if r['method']==method and r['task']=='aligned_w19'],key=lambda r:int(r['window']))
 if fr:axs[1].plot([int(r['window']) for r in fr],[float(r['LocalFID52']) for r in fr],label=method)
 rrs=[r for r in allr if r['method']==method and r['task']=='aligned_w19'];wins=sorted(set(int(r['window']) for r in rrs))
 if wins:axs[2].plot(wins,[100*statistics.mean(float(r['R3']) for r in rrs if int(r['window'])==w) for w in wins],label=method)
axs[0].set(xlabel='Target time including prefix (s)',ylabel='Terminal constraint error (cm)',xticks=[9.28,19.68,40.48])
axs[1].set(xlabel='Generated window index (52 frames/window)',ylabel='Local FID52',xticks=[1,4,8,12,16,19])
axs[2].set(xlabel='Generated window index (52 frames/window)',ylabel='R@3 (%)',xticks=[1,4,8,12,16,19])
for ax in axs:ax.grid(alpha=.2)
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.13))
fig.suptitle('Fixed-target diagnostic: endpoint control and window-wise generation quality',fontsize=11)
fig.tight_layout();fig.savefig(OUT/'comparison.png',bbox_inches='tight');plt.close(fig)
lines=['# 最新两个100k模型的评测结果','','全部7组×506条×2模型，共7084条新生成轨迹与逐窗evaluator_v2评分已完成。Minimal不进入主对比。',
 '', '表中的FID/R@3对应终点所在的最后一个52帧窗口，FID使用该窗口506条动作池化计算，R@3为两套检索编码器的均值；完整逐窗数据见CSV。终点误差仅计算root-XZ指定目标，不含Root ADE或中间路线惩罚。',
 '', '| 模型 | 实际终点时间(s) | 终点误差(cm)↓ | 20cm成功率(%)↑ | 末窗FID↓ | 末窗R@3(%)↑ |','|---|---:|---:|---:|---:|---:|']
for r in tables:lines.append(f"|{r['method']}|{float(r['duration_sec']):.2f}|{float(r['constraint_error_cm']):.2f}|{r['success20_percent']:.2f}|{float(r['final_LocalFID52']):.2f}|{r['final_R3_percent']:.2f}|")
lines+=['','![结果曲线](comparison.png)','','## 比较范围','',
 '这两个结果是DiP-G1和DART-MLP-G1的共同G1适配版本，不是完整官方DiP/DartControl系统。100k训练预算匹配；Native110k、117k的额外训练及官方ARDY预训练预算不匹配，须分开解释。',
 '', '重要限制：两适配模型的原生输入最多248帧，保持最多196帧历史、52帧生成，其余容量提供未来；没有把远期目标前移、压缩时间或添加规划器。相同外部任务不等于完全相同的长未来输入机制，因此不能用该表声称原方法的长时架构在完全同输入下劣于Native。目标实际可见性已逐窗留档。',
 '', '来源材料为生成器训练接触过的HiPHI来源上的事后受控诊断，不是独立未见测试集；同目标、文本、前缀、源ID和噪声配对已检查。逐窗R@3反映局部文本兼容性，不能单独证明长记忆。',
 '', '质量诊断还保留预测接触比例、以预测接触为条件的滑步、接缝及关节抖动。预测接触为零时，滑步proxy为零不能解释为动作质量很好。',
 '', '补充root位移核查：DiP-G1在约9/20/40秒末相对前缀原点的平均位移仅3.11/6.69/14.07厘米，因此其终点误差近似不变主要伴随移动不足，不能解释为长时控制稳定；DART-MLP-G1对应位移为111.11/183.58/481.80厘米，存在随时长增加的偏移。该项只是动作诊断，不对合理中间路线加罚。DiP单次可见目标扰动的forward差异接近FP32数值精度，不能把“非严格零”当成已学会空间控制的证据。',
 '', '## 结果位置','',f'服务器：`{R}`。原始geometry、逐样本误差、每窗特征、完整门禁和全部失败修复证据均保留；本次没有修改权重或替换线上仿真。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
for n in ['EVALUATION_SCOPE.md','GATES.json','CONDITIONING_FORWARD_PROBE.json','ROOT_MOTION_DIAGNOSTIC.json','INFERENCE_EXECUTION_AUDIT.json','CHECKPOINTS.json','NUMERICS_COMPLETE.json']:
 (OUT/n).write_bytes((R/n).read_bytes())
with zipfile.ZipFile(R/'two_models_evaluation_20261019.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.is_file():z.write(p,arcname=p.name)
(R/'DELIVERY_COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',publication=str(OUT),zip=str(R/'two_models_evaluation_20261019.zip'),compared_sources=len(paired),models=list(dict.fromkeys(x['model'] for x in tables))),indent=2))
print(json.dumps([r for r in tables if r['model'] in ['dip','dart','native100']],indent=2))
