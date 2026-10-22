"""Publish only independently scored complete easy-v3 results and all tradeoffs."""
from pathlib import Path
import os,json,time,csv,zipfile
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from execute import R,MODELS,rd,wr,sha,schedule,verify
import numpy as np
def csvwrite(p,rows):
 with Path(p).open('w') as f:
  fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 verify()
 # Do not publish latency until the separate admitted external-core replay finishes.
 while not (R/'retiming/COMPLETE.json').exists():
  if (R/'retiming/FAILURE.json').exists():raise RuntimeError('Admitted external retiming requires repair before final publication')
  wr(R/'scoring/FINALIZATION_STATUS.json',dict(stage='WAITING_VERIFIED_EXTERNAL_CORE_LATENCY',time=time.time()));time.sleep(20)
 rows=[];windows=[];audits=[];physical=[];responses=[];reference={};tasks=['short52','nominal9','nominal20','nominal40'];prev={}
 for m in MODELS:
  for t in tasks:
   d=R/'scoring/models'/m/t;c=rd(d/'COMPLETE.json')
   for n,h in c['files'].items():assert sha(d/n)==h,('changed_scored_file',m,t,n)
   s=rd(d/'SCORES.json');a=rd(d/'INDEPENDENT_AUDIT.json');assert a['passed']
   paired=[(r['case_id'],r['seed'],r['initial_sha256'],r['physical_sparse_goal_sha256']) for r in a['paired_inputs']]
   if t not in reference:reference[t]=paired
   assert paired==reference[t],('unpaired_inputs_across_models',m,t)
   if m not in prev:prev[m]=paired
   assert prev[m]==paired,('goals_differ_across_durations',m,t)
   la=R/'retiming'/m/t/'LATENCY.json'
   latency=rd(la) if la.exists() else s['latency']
   row=dict(model=m,task=t,generated_seconds=rd(R/'PLAN.json')['tasks'][t]['generated_seconds'],windows=len(schedule(t)[0]),independent_sources=128,rollouts=384,root_CE_cm=s['root_CE_cm']['mean'],root_CE_low=s['root_CE_cm']['ci95'][0],root_CE_high=s['root_CE_cm']['ci95'][1],end_CE_cm=s['end_CE_cm']['mean'],end_CE_low=s['end_CE_cm']['ci95'][0],end_CE_high=s['end_CE_cm']['ci95'][1],command_matching_percent=s['command_matching_percent']['mean'],R3_ref_percent=s['R3_ref_percent']['mean'],R3_ref_low=s['R3_ref_percent']['ci95'][0],R3_ref_high=s['R3_ref_percent']['ci95'][1],mean_window_L2_FID=s['mean_window_L2_FID'],terminal52_L2_FID=s['terminal52_L2_FID'],joint_success10_percent=s['joint_success10_percent'],joint_success20_percent=s['joint_success20_percent'],latency_median_ms_window=latency['median_ms_window'],latency_mean_ms_window=latency['mean_ms_window'],backend=latency['backend'],all_windows_goal_visible=s['all_windows_goal_visible'])
   rows.append(row);windows.extend(list(csv.DictReader((d/'WINDOW_METRICS.csv').open())))
   trials=list(csv.DictReader((d/'PER_TRIAL.csv').open()));physical.extend(trials)
   for command in dict.fromkeys(schedule(t)[0]):
    values=[float(v[command+'_response_s']) for v in trials if v[command+'_response_s']]
    responses.append(dict(model=m,task=t,command=command,N=384,response_observed_percent=len(values)/384*100,median_response_among_observed_s=float(np.median(values)) if values else None,command_interval_s=schedule(t)[0].count(command)*52/25,nonresponses='right-censored at end of command interval, never assigned zero latency'))
   audits.append(dict(model=m,task=t,score_complete=sha(d/'COMPLETE.json'),audit=sha(d/'INDEPENDENT_AUDIT.json'),latency_source=str(la) if la.exists() else str(d/'SCORES.json')))
 out=R/'delivery';out.mkdir(exist_ok=True)
 csvwrite(out/'SUMMARY.csv',rows);csvwrite(out/'WINDOW_METRICS.csv',windows);csvwrite(out/'PHYSICAL_PER_TRIAL.csv',physical);csvwrite(out/'COMMAND_RESPONSE.csv',responses)
 wr(out/'TEXT_UPDATE_LATENCY.json',rd(R/'TEXT_UPDATE_LATENCY.json'))
 names={'native100':'Native (ours, 100k)','ardy':'ARDY-G1 (100k; diagnostic)','dip':'DiP-G1 (100k; diagnostic)','dart':'DART-MLP-G1 (100k)','fd2':'FD2-Wan-G1 (100k)','omg':'OMG300M-G1 (100k)','cmdm':'CMDM-G1 (100k)','motionstreamer':'MotionStreamer-G1 (100k)','motion_mamba2_official':'ARDY (official reference)'}
 ordered=[m for m in MODELS if m not in ['native100','motion_mamba2_official']]+['native100','motion_mamba2_official']
 by={(r['model'],r['task']):r for r in rows};cal=rd(R/'scoring/REAL_REFERENCE_CALIBRATION.json')
 def refR(t):
  stages=schedule(t)[0]
  return float(np.mean([q['R3_ref_percent'] for c in stages for q in cal['rows'] if q['command']==c]))
 tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Unified basic continuous control on held-out retargeted HumanML3D initializations. Root and selected end-effector errors are measured only at the common deadline; no intermediate reference path is penalized.}',r'\label{tab:unified-easy-v3}',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{l'+'r'*24+'}',r'\toprule',r'Method & \multicolumn{6}{c}{52 frames (1 window)} & \multicolumn{6}{c}{8.32 s (4 windows)} & \multicolumn{6}{c}{18.72 s (9 windows)} & \multicolumn{6}{c}{39.52 s (19 windows)} \\',r'\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-19}\cmidrule(lr){20-25}']
 labels=[r'Root CE$\downarrow$',r'End CE$\downarrow$',r'Cmd.$\uparrow$',r'$R@3_{\rm ref}\uparrow$',r'L2-FID$\downarrow$',r'Lat.$\downarrow$']
 tex.append('Method & '+' & '.join(labels*4)+r' \\ \midrule')
 cells=[]
 for t in tasks:cells+=['---','---','---',f'{refR(t):.2f}','0.000','---']
 tex.append(r'\emph{Real basic-motion bank (reference)} & '+' & '.join(cells)+r' \\ \midrule')
 plot_values=[];plot_rows=[]
 for m in ordered:
  cells=[]
  for t in tasks:
   v=by[m,t];cells += [f'{v["root_CE_cm"]:.2f}',f'{v["end_CE_cm"]:.2f}',f'{v["command_matching_percent"]:.2f}',f'{v["R3_ref_percent"]:.2f}',f'{v["mean_window_L2_FID"]:.3f}',f'{v["latency_median_ms_window"]:.1f}']
  name=names[m]
  if m=='native100':name=r'\textbf{'+name+'}'
  tex.append(name+' & '+' & '.join(cells)+r' \\');plot_values.append(cells);plot_rows.append(names[m])
 tex += [r'\bottomrule',r'\end{tabular}}',r'\vspace{2pt}',r'\parbox{\linewidth}{\footnotesize CE is in cm; command matching and $R@3_{\rm ref}$ are percentages; latency is ms per accepted 52-frame window (B1, two complete warmups and 32 paired timed rollouts). Each method uses 128 independent source families and three noises (384 rollouts per duration). The goal is available from the start and occupies the last frame of the last window; only the current text instruction is supplied. One wrist or ankle is constrained per trial. DiP/DART fixed input capacity can truncate distant goals; their visibility is reported separately. The official ARDY release has a different pretraining corpus/budget; retrained ARDY/DiP retain failed-training diagnostic labels. L2-FID uses individually normalized 256-D short-window features and the fixed correct-domain real bank. The new reference-anchored $R@3_{\rm ref}$ uses one generated positive and 31 fixed incompatible real distractors, three pools and two frozen heads; it is a calibrated diagnostic, not the historical full-semantic retrieval benchmark or a memory score. The real-reference row reports the same command-weighted retrieval calibration, while its FID is a self-reference zero.}',r'\end{table*}']
 (out/'table_unified_easy_v3.tex').write_text('\n'.join(tex)+'\n')
 intro=r'''\section*{Unified basic continuous control}
We evaluate endpoint control, command responsiveness, and generated-motion quality within the same continuous rollout. Initial histories are drawn from 128 held-out retargeted HumanML3D source families; every checkpoint receives the same 24-frame initialization, sparse physical goals, command timing, and three noise seeds. Terminal goals lie 0.5--1.0 m away and constrain the root and one balanced wrist/ankle. Only the current instruction is exposed at a window boundary. The four-window and longer schedules include forward walking, standing, backward walking, and unconstrained walking. Durations refer to newly generated frames, excluding the 0.96-s initialization. These are composed interaction tasks, rather than natural long HumanML3D clips or the canonical HumanML3D leaderboard.

Root-XZ and selected-joint 3-D FK errors are measured at the deadline only; the intermediate route is free. Command execution is assessed using root velocity relative to the actual decoded hip-facing direction. In particular, turning and then walking forward is distinguished from walking backward. We report joint 10/20-cm success, responses after each command enters the generation window, contact coverage, a clearly labeled geometric ankle-slip proxy, seam velocity changes, and joint acceleration. Source-cluster confidence intervals retain the three noise realizations together; windows are never treated as independent subjects.

All features use frozen HumanML3D-G1-native-v2 assets and FK-consistent physical motion. L2-FID compares short-window feature distributions after per-feature-vector normalization and is independently recomputed in FP64. It does not evaluate a full long sequence. Retrieval uses fixed real distractors to avoid treating identical basic commands as incorrect competing captions. Real-action calibration is reported for each command and both heads. Its directional discrimination is unequal: therefore reference-anchored retrieval is supplementary, and physical command execution supplies the primary directional evidence. This evaluation alone does not establish episodic recall or the causal benefit of persistent memory; that requires the separately gated history-replay ablation.
'''
 main=r'\documentclass[10pt]{article}'+'\n'+r'\usepackage[a3paper,landscape,margin=14mm]{geometry}'+'\n'+r'\usepackage{booktabs,graphicx,amsmath}'+'\n'+r'\begin{document}'+'\n'+intro+'\n'+'\n'.join(tex)+'\n'+r'\end{document}'+'\n'
 (out/'main.tex').write_text(main)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
 fig,axes=plt.subplots(6,4,figsize=(16,15),sharex='col')
 metrics=['command_matching_percent','R3_ref_percent','L2_FID','ankle_slide_proxy_percent','ankle_contact_coverage','seam_velocity_jump_m_s'];ylabs=['Command matching (%)',r'$R@3_{ref}$ (%)','L2-FID','Ankle-slip proxy (%)','Ankle contact coverage','Seam velocity jump (m/s)']
 for j,t in enumerate(tasks):
  for i,(metric,label) in enumerate(zip(metrics,ylabs)):
   ax=axes[i,j]
   for m in ordered:
    data=sorted([x for x in windows if x['model']==m and x['task']==t],key=lambda x:int(x['window']))
    ax.plot([int(x['window']) for x in data],[float(x[metric]) if x[metric] else np.nan for x in data],marker='.' if len(data)<5 else None,lw=2 if m=='native100' else 1,ls='--' if m in ['ardy','dip'] else '-',label=names[m])
   stages=schedule(t)[0]
   for wi in range(1,len(stages)):
    if stages[wi]!=stages[wi-1]:ax.axvline(wi+.5,color='.75',lw=.7)
   ax.grid(alpha=.18)
   if i==0:ax.set_title(f'{len(stages)} windows / {len(stages)*2.08:.2f} s')
   if j==0:ax.set_ylabel(label)
   if i==5:ax.set_xlabel('Accepted generation window')
 handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=3,fontsize=8)
 fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'window_curves.pdf',bbox_inches='tight');fig.savefig(out/'window_curves.png',dpi=240,bbox_inches='tight');plt.close(fig)
 # Fully populated vector table preview; TeX above remains the editable paper artifact.
 fig,ax=plt.subplots(figsize=(24,5));ax.axis('off')
 table=ax.table(cellText=plot_values,rowLabels=plot_rows,colLabels=[f'{title}\n{metric}' for title in ['52f','8.32s','18.72s','39.52s'] for metric in ['Root CE','End CE','Cmd %','R3ref %','L2-FID','ms/w']],loc='center',cellLoc='right')
 table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.65)
 for (ri,ci_),cell in table.get_celld().items():
  cell.set_edgecolor('.85');cell.set_linewidth(.3)
  if ri==0:cell.set_facecolor('#e9eef4');cell.set_text_props(weight='bold')
 ax.set_title('Unified easy-v3: all outputs generated and independently audited; diagnostic retrieval and failed-training rows labeled',fontsize=12,pad=20)
 fig.savefig(out/'table_overview.pdf',bbox_inches='tight');fig.savefig(out/'table_overview.png',dpi=240,bbox_inches='tight');plt.close(fig)
 report=['# Unified easy-v3 evaluation','',f'Completed: {len(MODELS)} checkpoints (eight baselines plus Native100k), 36 panels, 13,824 fresh rollouts; 128 independent source families per panel.','', 'Each rollout jointly measures one terminal root goal, one selected wrist/ankle goal, changing basic commands, and motion quality. No Root ADE or intermediate-path penalty is used.','', 'R@3_ref is a new reference-anchored diagnostic. Real forward-motion calibration is only 48.61% / 38.19% for the two heads, despite geometric forward qualification; it cannot independently establish reliable forward/backward semantic discrimination or memory. Real backward calibration is 100%, generic-walk calibration93.44% /89.34%, and stand66.67% /71.43%. These differences are disclosed, not used to select a different scorer.','', 'Native carry-state generation alone does not establish the causal value of long memory. The accepted-latent full-replay equivalence control and recent-history reconstruction ablation remain separate required evidence. No unverified ablation score is reported.','', 'Files: SUMMARY.csv, WINDOW_METRICS.csv, table_unified_easy_v3.tex, main.tex, table_overview.pdf/png, window_curves.pdf/png. Baseline adaptation, future-visibility and official-pretraining distinctions are retained. Full per-trial geometry and hashes remain on the server.','']
 (out/'REPORT.md').write_text('\n'.join(report))
 wr(out/'FINAL_AUDIT.json',dict(passed=True,models=9,baseline_checkpoints=8,panels=36,rollouts=13824,paired_physical_inputs_across_models_and_durations=True,original_protocol_hashes_verified=True,assets=rd(R/'scoring/ASSET_BINDINGS.json'),receipts=audits,history_ablation_complete=False,episodic_memory_claim=False,time=time.time()))
 wr(out/'COMPLETE.json',dict(passed=True,scope='eight baselines plus Native100k carry, all four fixed durations, physical metrics+calibrated diagnostic retrieval+normalizedFID+exact warmed latency',history_ablation_complete=False,files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['COMPLETE.json']},time=time.time()))
 with zipfile.ZipFile(R/'Unified_Easy_V3_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for p in out.iterdir():
   if p.is_file():z.write(p,'Unified_Easy_V3/'+p.name)
 wr(R/'STATUS.json',dict(stage='EIGHT_BASELINES_PLUS_NATIVE_CARRY_EVALUATION_COMPLETE',history_ablation_complete=False,delivery=str(out),time=time.time()))
if __name__=='__main__':main()
