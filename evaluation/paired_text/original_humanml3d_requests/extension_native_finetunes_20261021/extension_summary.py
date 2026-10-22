"""Summarize fresh additional checkpoints, without changing previous measurements."""
from motion_mamba2.paths import artifact_path
from pathlib import Path
import json,csv,zipfile,time,hashlib
R=Path(artifact_path('unified_interactive_easy_v3_20261021_ebde76396c'));E=R/'extension_native_finetunes_20261021';D=E/'delivery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def finish():
 rows=list(csv.DictReader((D/'SUMMARY.csv').open()));by={(x['model'],x['task']):x for x in rows}
 assert len(rows)==48 and len({x['model'] for x in rows})==12
 native=[x for x in rows if x['model'] in ['native100','native110','native117','native122']]
 with (D/'NATIVE_COMPARISON.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(native[0]));w.writeheader();w.writerows(native)
 text=['# Native110k、117k、122k追加测评','','新增三个权重各4条件、每条件384次新生成，共4608次；与之前9个checkpoint的36条件合并后为12个checkpoint、48条件、18432次生成。','', '原36条件几何和评分不重跑、不改动；新条件共享原冻结的128来源/3noise、初始24帧、终点相位51、0.5–1m root+一个末端、当前指令切换、评测器/参考库/候选池。122k选择117k续训5k的普通分支，没有混入长窗蒸馏分支。','','每格是终点root-XZ / 选定手腕或脚踝三维FK误差（cm），不惩罚中间路径：','','| Native checkpoint | 52帧 | 208帧（8.32s） | 468帧（18.72s） | 988帧（39.52s） |','|---|---:|---:|---:|---:|']
 for m in ['native100','native110','native117','native122']:
  a=[f"{float(by[m,t]['root_CE_cm']):.2f} / {float(by[m,t]['end_CE_cm']):.2f}" for t in ['short52','nominal9','nominal20','nominal40']]
  text.append('| '+m+' | '+' | '.join(a)+' |')
 text += ['','长时的动作执行与短窗特征质量，独立于终点误差报告：','','| checkpoint/时长 | 指令匹配帧 | L2-FID（逐窗均值） | R@3_ref（仅诊断） | root与末端均≤20cm | 每窗中位延迟ms |','|---|---:|---:|---:|---:|---:|']
 for m in ['native100','native110','native117','native122']:
  for t in ['nominal20','nominal40']:
   x=by[m,t];text.append(f"| {m}/{t} | {float(x['command_matching_percent']):.2f}% | {float(x['mean_window_L2_FID']):.4f} | {float(x['R3_ref_percent']):.2f}% | {float(x['joint_success20_percent']):.2f}% | {float(x['latency_median_ms_window']):.1f} |")
 text += ['','新增权重均通过该队列四种末端mask下完整输出、FK几何、逐窗记忆状态、RNG的参考/加速精确一致性门禁；32条B1计时有两次完整预热，实际mask-aware Graph不得在计时中建图。','', 'R@3_ref保留原有真实动作校准和明确诊断标签；静态ARDY100k短窗可得85.42%但实际运动匹配0%的反例仍适用，不能用这项指标独立证明方向指令跟随或长记忆。FID是正确HumanML3D域短窗分布特征逐向量L2归一化后的FP64值，不是整段40秒FID。','', 'Native110k/117k/122k属于额外微调对照，不冒充训练预算相同的100k架构对照。完整历史重放/近期196帧因果记忆消融未执行，不声称本次完成了记忆因果证明。','', '本目录含全部12模型的SUMMARY.csv、逐窗曲线、物理指标、响应时间和PDF/PNG/TeX总表。原始9模型交付包与原始结果仍保留。']
 (D/'RESULTS.md').write_text('\n'.join(text)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':10})
 native_models=['native100','native110','native117','native122'];tasks=['short52','nominal9','nominal20','nominal40']
 fig,axes=plt.subplots(1,2,figsize=(10,3.8))
 for ax,metric,title in zip(axes,['root_CE_cm','end_CE_cm'],['Terminal root-XZ constraint error','Terminal selected-joint FK constraint error']):
  for m in native_models:
   yy=[float(by[m,t][metric]) for t in tasks];prefix='root_CE_' if metric=='root_CE_cm' else 'end_CE_'
   low=[float(by[m,t][prefix+'low']) for t in tasks];high=[float(by[m,t][prefix+'high']) for t in tasks]
   ax.errorbar([1,4,9,19],yy,yerr=[[y-lo for y,lo in zip(yy,low)],[hi-y for y,hi in zip(yy,high)]],marker='o',capsize=3,label=m.replace('native','Native '))
  ax.set(title=title,xlabel='Terminal generation window',ylabel='Constraint error (cm)',xticks=[1,4,9,19]);ax.grid(alpha=.2)
 axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(D/'native_endpoint_comparison.pdf');fig.savefig(D/'native_endpoint_comparison.png',dpi=240);plt.close(fig)
 win=list(csv.DictReader((D/'WINDOW_METRICS.csv').open()));fig,axes=plt.subplots(2,3,figsize=(13,7))
 metrics=['command_matching_percent','R3_ref_percent','L2_FID','ankle_slide_proxy_percent','ankle_contact_coverage','seam_velocity_jump_m_s'];labels=['Command-matching frames (%)','R@3_ref (diagnostic, %)','Short-window L2-FID','Ankle-slip proxy (%)','Ankle contact coverage','Seam velocity jump (m/s)']
 for ax,metric,label in zip(axes.flat,metrics,labels):
  for m in native_models:
   a=sorted([x for x in win if x['model']==m and x['task']=='nominal40'],key=lambda v:int(v['window']))
   ax.plot([int(v['window']) for v in a],[float(v[metric]) if v[metric] else float('nan') for v in a],label=m.replace('native','Native '))
  for boundary in [6.5,8.5,13.5]:ax.axvline(boundary,color='.7',ls=':',lw=.8)
  ax.set(xlabel='Accepted generation window',ylabel=label);ax.grid(alpha=.2)
 axes[0,0].legend(frameon=False);fig.suptitle('Same frozen39.52-s task; vertical lines mark command changes');fig.tight_layout();fig.savefig(D/'native_long_window_comparison.pdf');fig.savefig(D/'native_long_window_comparison.png',dpi=240);plt.close(fig)
 c=json.loads((D/'COMPLETE.json').read_text());c.update(models=12,baseline_checkpoints=8,native_checkpoints=4,new_checkpoints=3,conditions=48,new_conditions=12,new_rollouts=4608,combined_rollouts=18432,local_manuscript_compile_pending=True)
 c['files']={p.name:sha(p) for p in D.iterdir() if p.is_file() and p.name!='COMPLETE.json'};(D/'COMPLETE.json').write_text(json.dumps(c,indent=2)+'\n')
 with zipfile.ZipFile(E/'Unified_Easy_V3_All12_With_Native_Finetunes.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in D.iterdir():
   if p.is_file():z.write(p,'Unified_Easy_V3_All12/'+p.name)
 (E/'FINAL_DELIVERY.json').write_text(json.dumps(dict(remote_evaluation_complete=True,models=12,conditions=48,new_rollouts=4608,zip_sha256=sha(E/'Unified_Easy_V3_All12_With_Native_Finetunes.zip'),complete_sha256=sha(D/'COMPLETE.json'),local_manuscript_compilation_pending=True,time=time.time()),indent=2)+'\n')
