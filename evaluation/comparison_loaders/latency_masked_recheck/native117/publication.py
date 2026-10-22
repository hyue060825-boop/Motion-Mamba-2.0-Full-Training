"""Build scientific assets from complete, audited numbers only; never invent scores."""
from pathlib import Path
import argparse,collections,csv,fcntl,hashlib,json,os,shutil,subprocess,time,zipfile
import numpy as np
from util import ROOT,MODELS,TASKS,read,write,rows,sha

LABELS={'ardy':'ARDY-100k (ours)','minimal':'Minimal-100k','native110':'Native-110k','native117':'Native-117k'}
COLORS={'ardy':'#676767','minimal':'#3172A8','native110':'#CB702D','native117':'#258276'}
ALIGNED=TASKS[:3]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');a=parser.parse_args()
    lock=(ROOT/'publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'NUMERICS_COMPLETE.json').exists():
        if not a.wait:raise RuntimeError('All numerical audits must complete first')
        write(ROOT/'PUBLICATION_LIVE.json',dict(status='WAITING_COMPLETE_NUMERICAL_AUDITS',pid=os.getpid(),time=time.time()));time.sleep(30)
    assert read(ROOT/'NUMERICS_COMPLETE.json')['status']=='COMPLETE_ALL_FOUR_MODELS_CONTROLLED_ENDPOINTS_AND_FROZEN_LOCAL52_METRICS'
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.55})
    out=ROOT/'publication';figdir=out/'figures';tables=out/'tables';data=out/'data'
    for p in [out,figdir,tables,data]:p.mkdir(exist_ok=True,parents=True)
    e=pd.read_csv(ROOT/'aggregate/ENDPOINT_SUMMARY.csv');f=pd.read_csv(ROOT/'aggregate/ALL_WINDOW_LOCALFID52.csv');r=pd.read_csv(ROOT/'aggregate/ALL_WINDOW_RETRIEVAL.csv');r=r[r.encoder_seed==1701]
    plan=read(ROOT/'PLAN.json');legend=[plt.Line2D([],[],color=COLORS[m],label=LABELS[m]) for m in MODELS]
    def save(fig,name):
        fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
        fig.savefig(figdir/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.04)
        plt.close(fig)
    def series(ax,d,x,y,lo,hi,model):
        z=d[d.model==model].sort_values(x);ax.plot(z[x],z[y],'-o',color=COLORS[model],ms=3)
        ax.fill_between(z[x].to_numpy(float),z[lo].to_numpy(float),z[hi].to_numpy(float),color=COLORS[model],alpha=.12,lw=0)
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.5))
    strata=[('ALL','All sources (n=506)'),('near_0.5_1.5m','Near: 0.5-1.5 m (n=281)'),('medium_1.5_3m','Medium: 1.5-3 m (n=198)'),('far_3_6m','Far: 3-6 m (n=27)')]
    for ax,(group,title) in zip(axes.flat,strata):
        dd=e[(e.stratum==group)&e.task.isin(ALIGNED)]
        for m in MODELS:series(ax,dd,'goal_time_including_prefix_sec','constraint_error_cm','CE_ci95_low','CE_ci95_high',m)
        ax.set(title=title,xlabel='Total duration including prefix (s)',ylabel='Endpoint constraint error (cm)',xticks=[9.28,19.68,40.48]);ax.grid(alpha=.17)
    fig.legend(handles=legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.015));fig.tight_layout(rect=[0,0,1,.94]);save(fig,'aligned_endpoint_ce')
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.5))
    for ax,(group,title) in zip(axes.flat,strata):
        dd=e[(e.stratum==group)&e.task.str.startswith('phase_')]
        for m in MODELS:series(ax,dd,'goal_time_including_prefix_sec','constraint_error_cm','CE_ci95_low','CE_ci95_high',m)
        ax.set(title=title,xlabel='Goal deadline (s)',ylabel='Endpoint constraint error (cm)',xticks=[19.2,20.,21.28,22.08]);ax.grid(alpha=.17)
    fig.legend(handles=legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.015));fig.tight_layout(rect=[0,0,1,.94]);save(fig,'phase_endpoint_ce')
    for tasks,name in [(ALIGNED,'aligned_local_quality'),(TASKS[3:],'phase_local_quality')]:
        fig,axes=plt.subplots(2,len(tasks),figsize=(7.16,4.2),squeeze=False)
        for j,t in enumerate(tasks):
            dd=f[f.task==t];rr=r[r.task==t]
            for m in MODELS:
                series(axes[0,j],dd,'window','LocalFID52','FID_ci95_low','FID_ci95_high',m)
                series(axes[1,j],rr,'window','R3','R3_ci95_low','R3_ci95_high',m)
            spec=plan['tasks'][t];title=(f'{spec["total_duration_sec"]:.2f} s' if t in ALIGNED else f'Goal {spec["goal_time_including_prefix_sec"]:.2f} s; phase {spec["within_window_goal_frame"]+1}/52')
            axes[0,j].set_title(title)
            for i in [0,1]:axes[i,j].set_xlabel('Generated window index');axes[i,j].grid(alpha=.17)
            axes[0,j].set_ylabel('LocalFID52' if j==0 else '');axes[1,j].set_ylabel('Local R@3 (pool 32)' if j==0 else '');axes[1,j].set_ylim(0,1)
        fig.legend(handles=legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.015));fig.tight_layout(rect=[0,0,1,.94]);save(fig,name)
    physical=[];seams=[]
    for task in TASKS:
        for model in MODELS:
            for c in [x for x in plan['catalog'] if x['task']==task]:
                p=ROOT/'evaluation'/model/task/f'shard{c["shard"]:02d}';z=read(p/'WINDOW_PHYSICAL_METRICS.json')
                physical += [dict(model=model,task=task,**v) for v in z['window_metrics'] if v['role']=='generated']
                seams += [dict(model=model,task=task,**v) for v in z['seams'] if v['role']=='generated']
    ph=pd.DataFrame(physical);ss=pd.DataFrame(seams)
    ph.groupby(['task','model','window'],as_index=False).agg(sliding_proxy_percent=('foot_sliding_proxy_percent','mean'),contact_support_percent=('predicted_support_fraction_percent','mean'),joint_second_difference=('joint_second_difference_m_per_frame2','mean'),N=('source_id','count')).to_csv(data/'WINDOW_PHYSICAL_MEANS.csv',index=False)
    ss.groupby(['task','model','entering_window'],as_index=False).agg(root_velocity_discontinuity_m_s=('root_velocity_discontinuity_m_s','mean'),joint_rotation_step_rms_deg=('local_joint_rotation_step_rms_deg','mean'),N=('source_id','count')).to_csv(data/'WINDOW_SEAM_MEANS.csv',index=False)
    ph=pd.read_csv(data/'WINDOW_PHYSICAL_MEANS.csv');ss=pd.read_csv(data/'WINDOW_SEAM_MEANS.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.35))
    for m in MODELS:
        z=ph[(ph.task=='aligned_w19')&(ph.model==m)];s=ss[(ss.task=='aligned_w19')&(ss.model==m)]
        axes[0].plot(z.window,z.sliding_proxy_percent,color=COLORS[m]);axes[1].plot(z.window,z.contact_support_percent,color=COLORS[m]);axes[2].plot(s.entering_window,s.root_velocity_discontinuity_m_s,color=COLORS[m])
    for ax,title,ylabel in zip(axes,['Sliding with predicted contact','Predicted contact support','Seam velocity change'],['Foot-frame slots (%)','Foot-frame slots (%)','Root velocity change (m/s)']):
        ax.set(title=title,ylabel=ylabel,xlabel='Generated window index');ax.grid(alpha=.17)
    fig.legend(handles=legend,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.05));fig.tight_layout(rect=[0,0,1,.91]);save(fig,'long_window_physical_quality')
    for n in ['ENDPOINT_SUMMARY.csv','ENDPOINT_PER_SOURCE.csv','PAIRED_MODEL_CONTRASTS.csv','ALL_WINDOW_LOCALFID52.csv','ALL_WINDOW_RETRIEVAL.csv']:
        shutil.copy2(ROOT/'aggregate'/n,data/n)
    # LaTeX fragments can be inserted into a paper; the separate preview is
    # self-contained, so the native single-file compiler can validate tables.
    def cell(v,l,h):return f'{v:.2f} [{l:.2f}, {h:.2f}]'
    erows=[];qrows=[]
    for m in MODELS:
        cells=[];quality=[]
        for t in ALIGNED:
            z=e[(e.model==m)&(e.task==t)&(e.stratum=='ALL')].iloc[0]
            cells.append(cell(z.constraint_error_cm,z.CE_ci95_low,z.CE_ci95_high))
            ff=f[(f.model==m)&(f.task==t)].sort_values('window').iloc[-1];rr=r[(r.model==m)&(r.task==t)].sort_values('window').iloc[-1]
            quality.extend([f'{ff.LocalFID52:.3f}',f'{rr.R3:.3f}'])
        label=LABELS[m].replace(' (ours)','')
        erows.append(label+' & '+' & '.join(cells)+r' \\')
        qrows.append(label+' & '+' & '.join(quality)+r' \\')
    et=r'''\begin{table*}[t]
\centering\small
\caption{Fixed-goal endpoint control on 506 training-exposed HiPHI sources. Each source retains the same goal coordinate, initial history, caption and noise seed across durations. Only the prescribed terminal root XZ error is scored (cm; lower is better). Brackets denote pointwise actor-then-source bootstrap 95\% intervals. Durations include the 24-frame prefix. This is a post-hoc protocol diagnosis, not an unseen-source benchmark.}
\label{tab:controlled_endpoint}
\begin{tabular}{lrrr}\toprule
Method & 9.28 s & 19.68 s & 40.48 s \\\midrule
'''+ '\n'.join(erows)+r'''
\bottomrule\end{tabular}
\end{table*}
'''
    qt=r'''\begin{table*}[t]
\centering\small
\caption{Final-window local motion quality under the same controlled task. LocalFID52 uses 506 source windows; local R@3 uses deduplicated captions, identical 32-candidate pools and the frozen qualified HiPHI-local52 retrieval head (seed 1701). These are local-window metrics, not whole-sequence FID or temporal-order accuracy.}
\label{tab:controlled_local_quality}
\begin{tabular}{lrrrrrr}\toprule
 & \multicolumn{2}{c}{9.28 s} & \multicolumn{2}{c}{19.68 s} & \multicolumn{2}{c}{40.48 s} \\
Method & FID$\downarrow$ & R@3$\uparrow$ & FID$\downarrow$ & R@3$\uparrow$ & FID$\downarrow$ & R@3$\uparrow$ \\\midrule
'''+ '\n'.join(qrows)+r'''
\bottomrule\end{tabular}
\end{table*}
'''
    (tables/'endpoint_table.tex').write_text(et);(tables/'local_quality_table.tex').write_text(qt)
    section=r'''\subsection{Main Experiment III: Controlled Long-Horizon Endpoint Following}
\paragraph{Protocol diagnosis.}
The original duration comparison conflated endpoint displacement with the position of the target inside the final generation window. We therefore conduct a paired diagnostic on 506 locomotion sources spanning 22 actors. Sources are retained using caption and ground-truth displacement criteria only: 281, 198 and 27 sources fall in the 0.5--1.5, 1.5--3 and 3--6\,m strata. For each source, we freeze the initial 24-frame history, caption, spatial root target and random seed. We vary the number of complete 52-frame generation windows (4, 9 and 19), yielding prefix-inclusive durations of 9.28, 19.68 and 40.48\,s at 25\,fps. The target is placed at the final frame of a complete window in every condition. Endpoint error measures the planar root distance at the specified target frame; intermediate routes are unrestricted.
\paragraph{Window-position control.}
A complementary test fixes the total generated rollout to 572 frames and places the same target at generated-frame indices 455, 475, 507 and 527. These deadlines alternate between within-window indices 39 and 7, with matched-position controls separated by 52 frames. This test probes sensitivity to the available current-window correction interval while retaining a duration control; it does not by itself identify window position as the sole cause of a performance difference.
\paragraph{Quality and uncertainty.}
We report terminal constraint error, threshold success rates and paired actor-then-source bootstrap intervals. We additionally score every complete generated window using the frozen evaluator: LocalFID52 and local R@3 with source-matched 32-candidate pools. Retrieval tests compatibility with the active source instruction rather than event-order correctness. Contact support, sliding proxies and seam velocity changes are reported separately. The 27-source far stratum is not used for a standalone FID ranking.
\paragraph{Scope.}
This cohort is drawn from previously observed, generator-training-exposed HiPHI material. The experiment is a post-hoc controlled diagnosis, not evidence of unseen-source generalization. Existing checkpoints have different training histories; their comparison does not isolate an architectural effect.
\input{tables/endpoint_table.tex}
\input{tables/local_quality_table.tex}
'''
    (out/'paper_section.tex').write_text(section)
    preview=r'\documentclass[10pt]{article}'+'\n'+r'\usepackage[a4paper,landscape,margin=18mm]{geometry}'+'\n'+r'\usepackage{booktabs}'+'\n'+r'\begin{document}'+'\n'+r'\section*{Controlled long-horizon endpoint diagnosis}'+'\n'+et.replace('table*','table')+qt.replace('table*','table')+'\n'+r'\end{document}'+'\n'
    (out/'tables_preview.tex').write_text(preview)
    report=['# 四模型同目标与窗口位置控制：全量结果','',
        '506条源动作、22名演员，原GT-only规则固定；近281/中198/远27。四个模型各7条件，共14,168条新生成。采样器、权重、训练、仿真均未改变。','',
        '以下仅统计指定终点root XZ约束，单位厘米；不对中途路线扣分，不报告Root ADE。总时长按25fps帧计数并包含24帧初态。','',
        '|模型|9.28秒CE|19.68秒CE|40.48秒CE|','|---|---:|---:|---:|']
    for m in MODELS:
        vv=[e[(e.model==m)&(e.task==t)&(e.stratum=='ALL')].iloc[0].constraint_error_cm for t in ALIGNED]
        report.append('|'+LABELS[m]+'|'+'|'.join(f'{v:.2f}' for v in vv)+'|')
    report += ['', '完整分距离层均值、中位数、P90、10/20/50厘米成功率及配对区间见data/ENDPOINT_SUMMARY.csv和PAIRED_MODEL_CONTRASTS.csv。相位对照有四个固定截止帧，不应从混合均值直接宣称某因素为唯一根因。','',
        'LocalFID52基于506个自然参考同窗分布；R@3以去重文本和共同32候选池计算，seed1701为主、2903为敏感性。它们不是长序列整体FID或事件顺序正确率。远距离27条不做独立FID排名。','',
        '这些材料暴露于生成器训练且属于事后协议诊断，不能包装为新独立未见benchmark；四权重训练历史不同，不单独用于架构因果结论。原1024条评测保持原样。','',
        '服务器已生成PDF/600dpiPNG及论文TeX片段，但仍需本地实际视觉检查与内置单文件TeX编译，不能据文件存在宣称最终出版QA完成。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    for name in ['PLAN.json','CHECKPOINTS.json','PREPARED.json','GATES.json','NUMERICS_COMPLETE.json']:
        shutil.copy2(ROOT/name,out/name)
    render=out/'qa_render';render.mkdir(exist_ok=True)
    for pdf in sorted(figdir.glob('*.pdf')):
        subprocess.run(['pdftoppm','-scale-to','1600','-singlefile','-png',str(pdf),str(render/pdf.stem)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='FILE_MANIFEST.json']
    write(out/'FILE_MANIFEST.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]))
    zipped=ROOT/'controlled_four_model_assets_PENDING_VISUAL_QA.zip'
    with zipfile.ZipFile(zipped,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,str(Path('controlled_four_models')/p.relative_to(out)))
    with zipfile.ZipFile(zipped) as z:assert z.testzip() is None
    write(ROOT/'PUBLICATION_READY.json',dict(status='ASSETS_AND_ZIP_READY_PENDING_LOCAL_VISUAL_QA_AND_NATIVE_TEX_COMPILATION',publication=str(out),zip=str(zipped),zip_sha256=sha(zipped),source_sha256=sha(Path(__file__)),time=time.time()))
    print('PUBLICATION_ASSETS_READY_NOT_FINAL_VISUAL_ACCEPTANCE',out,flush=True)

if __name__=='__main__':main()
