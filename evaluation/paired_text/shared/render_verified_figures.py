"""Presentation only: render verified CSVs without rescoring or generation."""
from pathlib import Path
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,csv,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;D=R/'delivery'
tasks=['short52','nominal9','nominal20','nominal40']
ordered=['ardy','dip','dart','fd2','omg','cmdm','motionstreamer','native100','native110','native117','native122','minimal','motion_mamba2_official']
names={'native100':'Native (ours, 100k)','ardy':'ARDY-G1 (100k; diagnostic)','dip':'DiP-G1 (100k; diagnostic)','dart':'DART-MLP-G1 (100k)','fd2':'FD2-Wan-G1 (100k)','omg':'OMG300M-G1 (100k)','cmdm':'CMDM-G1 (100k)','motionstreamer':'MotionStreamer-G1 (100k)','motion_mamba2_official':'ARDY (official reference)','native110':'Native (ours, 110k)','native117':'Native (ours, 117k)','native122':'Native (ours, 122k)','minimal':'Minimal (100k; ablation)'}
colors=dict(zip(ordered,['#b8a99a','#d3a6ad','#a7b2be','#846db0','#b29b38','#799d7d','#419caa','#192e5b','#345eaa','#698ad6','#1690b6','#797979','#c74e39']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
rows=list(csv.DictReader((D/'SUMMARY.csv').open()));by={(r['model'],r['task']):r for r in rows};plan=json.loads((R/'PLAN.json').read_text());cal=json.loads((D/'REAL_REFERENCE_CALIBRATION.json').read_text())
def ref(t):
 stages=[s for s,n in plan['tasks'][t]['stages'] for _ in range(n)]
 return float(np.mean([r['R3_ref_percent'] for c in stages for r in cal['rows'] if r['command']==c]))
data=[];line=['Real basic-motion bank (reference)']
for t in tasks:line+=['—','—','—',f'{ref(t):.2f}','0.000','—']
data.append(line)
for m in ordered:
 line=[names[m]]
 for t in tasks:
  r=by[m,t];line += [f"{float(r[k]):.{digits}f}" for k,digits in [('root_CE_cm',2),('end_CE_cm',2),('command_matching_percent',2),('R3_ref_percent',2),('mean_window_L2_FID',3),('latency_median_ms_window',1)]]
 data.append(line)
fig,ax=plt.subplots(figsize=(25.5,7.4));ax.axis('off')
labels=['Method / checkpoint']+['Root CE ↓','End CE ↓','Cmd. ↑','R@3ref ↑','L2-FID ↓','Lat. ↓']*4
widths=[.19]+[.81/24]*24
table=ax.table(cellText=data,colLabels=labels,colWidths=widths,cellLoc='right',colLoc='center',bbox=[0,.27,1,.60])
table.auto_set_font_size(False);table.set_fontsize(8.5)
for (ri,ci),cell in table.get_celld().items():
 cell.set_linewidth(0);cell.set_edgecolor('white');cell.set_facecolor('white')
 if ci==0:cell.set_text_props(ha='left');cell.PAD=.025
 if ri==0:cell.set_text_props(weight='bold')
 if ri in [9,10,11,12]:cell.set_text_props(weight='bold')
 if ri in [1,14]:cell.set_text_props(style='italic')
centers=[.19+(.81/4)*(i+.5) for i in range(4)]
for i,(center,label) in enumerate(zip(centers,['52 frames (1 window)','8.32 s (4 windows)','18.72 s (9 windows)','39.52 s (19 windows)'])):
 ax.text(center,.92,label,transform=ax.transAxes,ha='center',va='center',fontsize=11)
 ax.plot([.19+i*.81/4+.006,.19+(i+1)*.81/4-.006],[.895,.895],transform=ax.transAxes,color='black',lw=.65)
ax.plot([0,1],[.947,.947],transform=ax.transAxes,color='black',lw=1.2)
ax.plot([0,1],[.87-.60/15,.87-.60/15],transform=ax.transAxes,color='black',lw=.65)
ax.plot([0,1],[.27,.27],transform=ax.transAxes,color='black',lw=1.2)
ax.set_title('Unified continuous control with held-out retargeted HiPHI initializations',fontsize=14,pad=18)
notes=[
 'CE: cm at the terminal deadline only. Cmd.: body-relative command-matching frames (%). Lat.: ms per accepted 52-frame window, B1, two complete warmups and 32 paired timed rollouts. All methods use the same 128 source families and three seeds per family; intermediate paths are free.',
 'R@3ref is diagnostic only: real forward and standing calibration is weak. L2-FID uses individually L2-normalized 256-D short-window features; the table reports the mean over windows, not full-sequence FID. The real-bank FID of zero is a self-comparison.',
 'Retrieval uses the frozen HiPHI-local52 heads and their own statistics. Distribution features use frozen BONES-v2 with prior HiPHI transfer qualification. The real quality bank contains 112 families; 63 balanced families are retained for sensitivity analysis.',
 'Minimal is a structural ablation. Native110k/117k/regular122k are finetuned references; official ARDY has different pretraining. Retrained ARDY/DiP retain failed-training diagnostic labels. DiP/DART fixed input capacity may truncate distant targets.',
 'These are composed tasks initialized from the HiPHI project test split (three actors), not a canonical HiPHI leaderboard or natural 40-s reference reconstruction. This table alone does not establish a causal long-memory advantage.'
]
for i,line in enumerate(notes):ax.text(0,.23-i*.046,line,transform=ax.transAxes,ha='left',va='top',fontsize=8.0)
fig.tight_layout();fig.savefig(D/'table_overview.pdf',bbox_inches='tight');fig.savefig(D/'table_overview.png',dpi=220,bbox_inches='tight');plt.close(fig)

win=list(csv.DictReader((D/'WINDOW_METRICS.csv').open()));metrics=['command_matching_percent','R3_ref_percent','L2_FID','ankle_slide_proxy_percent','ankle_contact_coverage','seam_velocity_jump_m_s'];ylabs=['Command matching (%)',r'$R@3_{ref}$ (%) [diagnostic]','Mean-window L2-FID','Ankle-slip proxy (%)','Ankle contact coverage','Seam velocity jump (m/s)']
fig,axes=plt.subplots(6,4,figsize=(17,15),sharex='col')
for j,t in enumerate(tasks):
 stages=[s for s,n in plan['tasks'][t]['stages'] for _ in range(n)];W=len(stages)
 for i,(metric,label) in enumerate(zip(metrics,ylabs)):
  ax=axes[i,j]
  for m in ordered:
   a=sorted([x for x in win if x['model']==m and x['task']==t],key=lambda x:int(x['window']))
   ax.plot([int(x['window']) for x in a],[float(x[metric]) if x[metric] else np.nan for x in a],color=colors[m],marker='.' if W<5 else None,lw=1.8 if m.startswith('native') else 1.05,ls=':' if m in ['ardy','dip'] else '--' if m=='minimal' else '-',label=names[m],alpha=.95)
  for k in range(1,W):
   if stages[k]!=stages[k-1]:ax.axvline(k+.5,color='.65',lw=.65,ls='--')
  ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.14);ax.set_xlim(.65,W+.35);ax.set_xticks([1] if W==1 else range(1,W+1,1 if W<=9 else 2))
  if i==0:ax.set_title(f'{W} window'+('' if W==1 else 's')+f' / {W*2.08:.2f} s')
  if j==0:ax.set_ylabel(label)
  if i==5:ax.set_xlabel('Accepted generation window')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8,frameon=False)
fig.tight_layout(rect=(0,.075,1,1));fig.savefig(D/'window_curves.pdf',bbox_inches='tight');fig.savefig(D/'window_curves.png',dpi=220,bbox_inches='tight');plt.close(fig)
print('Presentation-only figures rendered from unchanged CSVs')
