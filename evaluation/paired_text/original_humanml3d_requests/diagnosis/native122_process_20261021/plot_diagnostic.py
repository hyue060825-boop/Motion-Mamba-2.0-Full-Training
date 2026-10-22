"""Plot saved diagnostic traces without rerunning or rescoring any model."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = Path(__file__).resolve().parent
base = json.loads((p/'001_s0/ACTUAL_CONDITIONING_AND_WINDOW_TRACE.json').read_text())['windows']
alt = json.loads((p/'001_s0/DIAGNOSTIC_EXPLICIT_FORWARD_TRACE.json').read_text())['windows']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(8.4, 4.3), layout='constrained')
for lo, hi, color, label in [(0.5,6.5,'#e5efff','Forward'),(6.5,8.5,'#e8ecef','Stand'),(8.5,13.5,'#fff0dc','Backward'),(13.5,19.5,'#e7f3e8','Walk')]:
    ax.axvspan(lo,hi,color=color,zorder=0)
    ax.text((lo+hi)/2,17.5,label,ha='center',va='center',fontsize=9)
ax.plot([w['window'] for w in base],[w['distance_to_final_goal_m'] for w in base],'-o',c='#204e80',ms=4,label='Frozen protocol: final text = Walk.')
ax.plot([w['window'] for w in alt],[w['distance_to_final_goal_m'] for w in alt],'--s',c='#a24320',ms=3,label='Diagnostic: final text = Walk forward.')
ax.annotate('Final error: 12.30 m',xy=(19,base[-1]['distance_to_final_goal_m']),xytext=(14.3,9.0),arrowprops={'arrowstyle':'->','color':'#204e80'},color='#204e80')
ax.set(xlabel='Accepted window number (52 frames per window)',ylabel='Distance to fixed final root target (m)',xlim=(.5,19.5),ylim=(0,18.3),xticks=range(1,20,2))
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.23)
ax.legend(loc='upper left',bbox_to_anchor=(0,0.91),frameon=False,fontsize=9)
fig.suptitle('Native122k: worst-error diagnostic case (case 1, seed 0)',fontsize=12)
fig.supxlabel('Diagnostic trajectory only; intermediate distances are not included in the terminal score.',fontsize=9)
fig.savefig(p/'worst_case_window_trace.png',dpi=220)
fig.savefig(p/'worst_case_window_trace.pdf')
