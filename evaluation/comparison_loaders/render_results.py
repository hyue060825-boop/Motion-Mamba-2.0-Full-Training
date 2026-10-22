"""Render only measured HumanML3D results into the preserved main.tex template."""
from pathlib import Path
import csv,json,argparse

p=argparse.ArgumentParser();p.add_argument('--results-root',type=Path);args=p.parse_args()
R=args.results_root or Path(__file__).resolve().parent
audit=json.loads((R/'FINAL_AUDIT.json').read_text());assert audit['passed']
missing=audit.get('missing_models',[])
assert not missing or audit.get('partial') is True
rows=list(csv.DictReader((R/'SUMMARY.csv').open()))
lookup={(r['model'],r['task']):r for r in rows}
models=['retarget','ardy','dip','dart','fd2','omg','cmdm','motionstreamer','native100','motion_mamba2_official','native110','native117','native122','native122long']
labels=[r'\textit{Dataset (Our retarget)}',r'ARDY-G1 (100k)$^{\rm d}$',r'DiP-G1 (100k)$^{\rm a,d}$',r'DART-MLP-G1 (100k)$^{\rm a}$',r'FD2-Wan-G1 (100k)$^{\rm a}$',r'OMG300M-G1 (100k)$^{\rm a}$',r'CMDM-G1 (100k)$^{\rm a}$',r'MotionStreamer-G1 (100k)$^{\rm a}$',r'\textbf{Native (ours, 100k)}',r'ARDY (official)$^{\rm b}$',r'Native (ours, 110k)',r'Native (ours, 117k)',r'Native (ours, 122k)',r'Native (122k, long-distill)']
text=(R/'main_template.tex').read_text()
start=text.index('% BEGIN HUMANML3D_SECTION1_20261020');stop=text.index('% END HUMANML3D_SECTION1_20261020',start)
section=text[start:stop]
lines=section.splitlines()
for i,(model,label) in enumerate(zip(models,labels)):
    matches=[n for n,s in enumerate(lines) if s.startswith(label.split('$^{')[0]+' &') or s.startswith(label+' &')]
    # Pending CMDM/MotionStreamer rows have a superscript marking unfinished training.
    if not matches:matches=[n for n,s in enumerate(lines) if s.startswith(label.split('$^{')[0]) and 'Pending' in s]
    assert len(matches)==1,(model,matches)
    if model in missing:
        lines[matches[0]]=label+r' & \multicolumn{4}{c}{\textit{Pending}} & \multicolumn{4}{c}{\textit{Pending}} & \multicolumn{4}{c}{N/A} & \multicolumn{4}{c}{N/A} \\'
        continue
    values=[]
    for task in ['short52','aligned_w04']:
        row=lookup[model,task]
        for key,digits in [('CE_cm',2),('FID_L2',3),('R3_percent',2),('latency_ms_per_window',1)]:
            values.append(f'{float(row[key]):.{digits}f}' if row[key] else r'\textemdash')
    lines[matches[0]]=label+' & '+' & '.join(values)+r' & \multicolumn{4}{c}{N/A} & \multicolumn{4}{c}{N/A} \\'
section='\n'.join(lines)+'\n'
status=(r'\textbf{Evaluation status: 12 of 13 generator checkpoints have completed the real 52-frame and nine-second evaluations; MotionStreamer-G1 remains pending its full 100k training checkpoint.}'
    if missing else r'\textbf{All listed frozen checkpoints have completed the real 52-frame and nine-second evaluations.}')
section=section.replace(r'\textbf{Evaluation status: queued; numerical results are not yet available.}',status)
section=section.replace('paired terminal-control evaluation (pending)','paired terminal-control evaluation')
if not missing:section=section.replace('Pending denotes unmeasured entries; N/A indicates','N/A indicates')
section=section.replace('The measured retargeted-data reference and embedding diagnostics will accompany the transfer scores.',
    'The measured retargeted-data reference and the released embedding diagnostics accompany the transfer scores.')
section=section.replace(r'$^{\rm c}$Awaiting completion of the currently authorized 100k training run before evaluation. ','')
section=section.replace('All cells are presently unmeasured; no HiPHI result is reused as a HumanML3D result.',
    'Every populated HumanML3D cell is bound to newly generated geometry and extracted features; no HiPHI score is reused. Quantitative comparisons here use point estimates on this fixed cohort.')
real52=float(lookup['retarget','short52']['R3_percent']);real9=float(lookup['retarget','aligned_w04']['R3_percent'])
native=[float(lookup['native100',t]['CE_cm']) for t in ['short52','aligned_w04']]
cmdm=[float(lookup['cmdm',t]['CE_cm']) for t in ['short52','aligned_w04']]
official=[float(lookup['motion_mamba2_official',t]['CE_cm']) for t in ['short52','aligned_w04']]
qualification=(f'The real-motion retrieval reference obtains {real52:.2f}\\% and {real9:.2f}\\% R@3 in the two conditions, '
    'against a 9.375\\% random-retrieval reference. The source captions describe whole clips, whereas these scores use local terminal windows; '
    'the frozen encoders have not been requalified as a calibrated HumanML3D semantic benchmark. We therefore interpret R@3 as a cross-dataset transfer diagnostic, '
    'rather than a canonical HumanML3D retrieval score or a direct test of memory-dependent recall. '
    f'Native100k obtains CE of {native[0]:.2f}/{native[1]:.2f}\\,cm, compared with {cmdm[0]:.2f}/{cmdm[1]:.2f}\\,cm for CMDM-G1 at matched training exposure. '
    f'The separately pretrained official ARDY release remains more accurate at {official[0]:.2f}/{official[1]:.2f}\\,cm. '
    'The terminal-control comparison does not imply dominance in distributional quality, retrieval or latency.\n\n')
anchor=r'\begin{table*}[t]'
section=section.replace(anchor,qualification+anchor,1)
physical={(r['model'],r['task']):r for r in csv.DictReader((R/'PHYSICAL.csv').open())}
section+=r'''\begin{table*}[t]
\centering
\caption{\textbf{Endpoint success and physical-quality diagnostics on retargeted HumanML3D.} Definitions match the HiPHI supplementary table. The 20- and 40-second conditions are unavailable owing to the absence of natural long reference clips.}
\label{tab:humanml3d_physical}
\begingroup\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.2}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}lrrrrr@{\hspace{8pt}}rrrrr@{}}
\toprule
Method / checkpoint & \multicolumn{5}{c}{52 frames} & \multicolumn{5}{c}{9\,s} \\
\cmidrule(lr){2-6}\cmidrule(l){7-11}
 & SR$_{20}\uparrow$ & Slip$\downarrow$ & Contact & Seam$\downarrow$ & Jitter$\downarrow$ & SR$_{20}\uparrow$ & Slip$\downarrow$ & Contact & Seam$\downarrow$ & Jitter$\downarrow$ \\
\midrule
'''
for model,label in zip(models,labels):
    if model in missing:
        section+=label+r' & \multicolumn{5}{c}{\textit{Pending}} & \multicolumn{5}{c}{\textit{Pending}} \\'+'\n'
        continue
    values=[]
    for task in ['short52','aligned_w04']:
        row=physical[model,task];sr=lookup[model,task]['SR20_percent']
        values.append(f'{float(sr):.2f}' if sr else r'\textemdash')
        for key,multiplier,digits in [('support_conditional_slip_percent',1,2),('predicted_support_percent',1,2),('seam_velocity_m_s',1,3),('joint_second_difference_m_per_frame2',1000,2)]:
            values.append(f'{float(row[key])*multiplier:.{digits}f}' if row[key] else r'\textemdash')
    section+=label+' & '+' & '.join(values)+r' \\'+'\n'
section+=r'''\bottomrule
\end{tabular}%
}\endgroup
\begin{minipage}{\textwidth}\footnotesize
SR$_{20}$, Slip and Contact are in percent; Seam is in m/s; Jitter is in $10^{-3}$m/frame$^2$. Sliding uses horizontal foot speed above 0.15\,m/s during predicted support; absent support yields undefined conditional sliding. A single window has no internal generation seam. The measured natural reference supplies quality and retrieval anchors; its success against a separate control request is not assigned. The checkpoint qualifications of Table~\ref{tab:humanml3d_terminal_control} also apply here.
\end{minipage}
\end{table*}
'''
(R/'humanml3d_results.tex').write_text(section)
(R/'main.tex').write_text(text[:start]+section+text[stop:])
