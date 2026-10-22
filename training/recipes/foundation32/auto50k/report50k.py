"""CPU-only aggregation of the frozen, paired Old/New50k evaluations."""
from motion_mamba2.paths import artifact_path
import argparse, hashlib, json, math, statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path(artifact_path('g1_fusion_batched_pretrain_20261006_4bab227953'))
OLD=Path(artifact_path('old_c1509953a8'))
ASSETS=['parameters','codec_hash','root_stats_hash','decoder_history_frames','generation_frames','inference_conditions','sequence_config']
def read(p): return json.loads(Path(p).read_text())
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def finite(v): return isinstance(v,(int,float)) and math.isfinite(v)
def avg(v):
    v=[x for x in v if finite(x)];return statistics.mean(v) if v else None
def foot_summary(whole):
    # All attempted windows are included; undefined contact rates remain missing.
    good=[w for w in whole if finite(w.get('contact_slip_percent'))]
    denom=sum((w['frames']-1)*w['contact_fraction'] for w in whole if finite(w.get('contact_fraction')))
    numerator=sum((w['frames']-1)*w['legacy_skate_percent']/100 for w in whole if finite(w.get('legacy_skate_percent')))
    return dict(windows=len(whole),valid_slip_windows=len(good),undefined_slip_windows=len(whole)-len(good),
        mean_contact_slip_percent=avg([w.get('contact_slip_percent') for w in whole]),
        pooled_contact_slip_percent=100*numerator/denom if denom>0 else None,
        mean_legacy_skate_percent=avg([w.get('legacy_skate_percent') for w in whole]),
        mean_contact_fraction=avg([w.get('contact_fraction') for w in whole]),
        mean_contact_speed_m_s=avg([w.get('contact_speed_m_s') for w in whole]))
def long_summary(res):
    case=[dict(id=r['id'],**foot_summary([w['metrics']['whole'] for w in r['windows']])) for r in res['rows']]
    expected=dict(T=3,W=24,S=18,X=16)
    return dict(n=len(res['rows']),function=sum(r['function_pass'] for r in res['rows']),joint=sum(r['all_pass'] for r in res['rows']),
        completed_cases=sum(bool(r.get('rollout_complete',False)) for r in res['rows']),
        planned_scored_windows=sum(expected[r['source']['group']] for r in res['rows']),
        generated_scored_windows=sum(len(r['windows']) for r in res['rows']),
        mean_case_contact_slip_percent=avg([c['mean_contact_slip_percent'] for c in case]),
        all_windows=foot_summary([w['metrics']['whole'] for r in res['rows'] for w in r['windows']]),by_case=case)
def summary(dev,final):
    b=dev['foundation'];rr=b['rows'];by_action={};by_cell={}
    for key,dest in [('action',by_action),('cell',by_cell)]:
        for val in sorted({r[key] for r in rr}):
            subset=[r for r in rr if r[key]==val]
            dest[val]=dict(n=len(subset),function=sum(r['function_pass'] for r in subset),joint=sum(r['joint_pass'] for r in subset),foot=foot_summary([r['metrics']['whole'] for r in subset]))
    return dict(function=b['function'],joint=b['joint'],n=b['windows'],by_action=by_action,by_cell=by_cell,
        foundation_foot=foot_summary([r['metrics']['whole'] for r in rr]),DEV=long_summary(dev),FINAL=long_summary(final))
def evaluate_report(new_root,new_load,output,self_test=False):
    nr=Path(new_root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    data={a:{s:read(p/s/'RESULT.json') for s in ['DEV','FINAL']} for a,p in [('Old',OLD),('New',nr)]}
    loads={'Old':read(OLD/'LOAD_evaluate.json'),'New':read(new_load)}
    assert all(l['stored_step']==50000 for l in loads.values()),'Both stored steps must be50000'
    assets={k:loads['Old'][k]==loads['New'][k] for k in ASSETS};assert all(assets.values()),assets
    hashes={}
    for arm in ['Old','New']:
        cp=loads[arm]['checkpoint'];hashes[arm]=sha(cp)
        assert hashes[arm]==loads[arm]['checkpoint_sha256']
        for split in ['DEV','FINAL']:
            res=data[arm][split];assert res['complete'] and res['split']==split
            assert res['checkpoint']==cp and res['checkpoint_sha256']==hashes[arm]
            assert len(res['rows'])==16
            assert res['foundation']['windows']==(96 if split=='DEV' else 0)
    old_source=read(OLD.parent/'SOURCE_HASHES.json')
    baseline=[h for p,h in old_source.items() if p.endswith('/motion_training/_compat/metrics.py')]
    metric_sha=sha(ROOT/'code/src/motion_training/_compat/metrics.py')
    assert baseline==[metric_sha], 'Slip metric implementation differs from frozen old evaluation'
    checks=[]
    for split in ['DEV','FINAL']:
        a,b=data['Old'][split],data['New'][split]
        base=lambda res:[(r['id'],r['source_id'],r['family'],r['prompt'],r['seed'],r['noise'],r['control']) for r in res['foundation']['rows']]
        seq=lambda res:[(r['id'],r['source'],r['seed']) for r in res['rows']]
        check=dict(split=split,metric_contract=a['metric_contract_sha256']==b['metric_contract_sha256'],foundation_inputs=base(a)==base(b),sequence_inputs=seq(a)==seq(b),noise_prefixes=all([x['noise_hash'] for x in ra['receipts'][:min(len(ra['receipts']),len(rb['receipts']))]]==[x['noise_hash'] for x in rb['receipts'][:min(len(ra['receipts']),len(rb['receipts']))]] for ra,rb in zip(a['rows'],b['rows'])))
        assert all(v for k,v in check.items() if k!='split'),check;checks.append(check)
    arms={arm:summary(data[arm]['DEV'],data[arm]['FINAL']) for arm in ['Old','New']}
    if self_test:assert arms['Old']==arms['New']
    result=dict(complete=True,self_test=self_test,time_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),stored_steps=dict(Old=50000,New=50000),global_batch=dict(Old=32,New=32),nominal_exposures_each=1600000,
        checkpoints={arm:loads[arm]['checkpoint'] for arm in loads},sha256=hashes,runtime_assets_equal=assets,evaluation_identity_checks=checks,slip_metric_sha256=metric_sha,
        averaging='Foundation: arithmetic mean per52frame test, including failed tests; pooled contact rate uses same skeleton and (frames-1)*contact_fraction weights. Long tests: both percase mean and perwindow mean. Undefined rates are counted, never assigned0. Only generated scored windows enter averages; truncated rollouts remain failed.',
        contact_mask='predicted contacts from original frozen evaluator; not GT contacts or physics tracking',arms=arms,
        same_training_recipe=False,training_difference='Old BONES-only; New70%BONES/15%HumanML3D/15%HiPHI; initialization, losses, precision and historical input recipes differ. Equal step/exposure compares complete recipes, not an isolated loss effect.',actual_ui=False,physics=False)
    result['delta_new_minus_old']=dict(function=arms['New']['function']-arms['Old']['function'],joint=arms['New']['joint']-arms['Old']['joint'],foundation_mean_slip_pp=None if any(arms[x]['foundation_foot']['mean_contact_slip_percent'] is None for x in arms) else arms['New']['foundation_foot']['mean_contact_slip_percent']-arms['Old']['foundation_foot']['mean_contact_slip_percent'],DEV_joint=arms['New']['DEV']['joint']-arms['Old']['DEV']['joint'],FINAL_joint=arms['New']['FINAL']['joint']-arms['Old']['FINAL']['joint'])
    def pair(fn):return ' | '.join(str(fn(arms[a])) for a in ['Old','New'])
    def rate(v):return '无有效接触' if v is None else f'{v:.2f}%'
    def row(label,fn):return f'| {label} | {pair(fn)} |'
    lines=['# 原旧Native50k与本轮融合新50k：同条件测评','',('**这是旧50k对自身的报告程序自检，不是新50k训练结果。**' if self_test else '**完整评测和对照核验已完成。结果不等于真实G1仿真验收。**'),'',f"生成时间（北京时间）：{result['time_beijing']}",'','| 指标 | 原旧Native50k | 本轮融合新50k |','|---|---:|---:|',
        row('基础动作功能通过',lambda s:f"{s['function']}/96"),row('功能与动作质量同时通过',lambda s:f"{s['joint']}/96"),
        row('纯文本冷启动功能',lambda s:f"{s['by_cell']['cold_text']['function']}/32"),
        *[row(label,lambda s,a=action:f"{s['by_action'][a]['function']}/12") for label,action in [('短窗走路功能','walk'),('后退功能','backwalk'),('转向功能','turn')]],
        row('完整连续任务，开发组／保留组',lambda s:f"{s['DEV']['joint']}/16、{s['FINAL']['joint']}/16"),
        row('平均滑步率：96项短窗等权',lambda s:rate(s['foundation_foot']['mean_contact_slip_percent'])),
        row('合并所有接触脚帧后的滑步率',lambda s:rate(s['foundation_foot']['pooled_contact_slip_percent'])),
        row('旧口径平均滑步占比：全部脚帧作分母',lambda s:rate(s['foundation_foot']['mean_legacy_skate_percent'])),
        row('平均脚接触速度',lambda s:'无有效接触' if s['foundation_foot']['mean_contact_speed_m_s'] is None else f"{s['foundation_foot']['mean_contact_speed_m_s']:.3f}m/s"),
        row('基础平均接触覆盖',lambda s:rate(100*s['foundation_foot']['mean_contact_fraction']) if s['foundation_foot']['mean_contact_fraction'] is not None else '缺失'),
        row('基础滑步率有效／未定义测试数',lambda s:f"{s['foundation_foot']['valid_slip_windows']}／{s['foundation_foot']['undefined_slip_windows']}"),
        '', '## 按动作看功能、质量与滑步', '', '| 动作 | 旧功能／质量 | 新功能／质量 | 旧平均滑步率 | 新平均滑步率 |','|---|---:|---:|---:|---:|']
    for action in sorted(arms['Old']['by_action']):
        x,y=(arms[a]['by_action'][action] for a in ['Old','New'])
        lines.append(f"| {action} | {x['function']}/{x['n']}、{x['joint']}/{x['n']} | {y['function']}/{y['n']}、{y['joint']}/{y['n']} | {rate(x['foot']['mean_contact_slip_percent'])} | {rate(y['foot']['mean_contact_slip_percent'])} |")
    lines+=['','## 连续任务滑步及生成完整性','','| 指标 | 原旧Native50k | 本轮融合新50k |','|---|---:|---:|']
    for split,label in [('DEV','开发组'),('FINAL','保留组')]:
        lines += [row(label+'平均滑步率，各任务等权',lambda s,k=split:rate(s[k]['mean_case_contact_slip_percent'])),row(label+'平均滑步率，各生成窗口等权',lambda s,k=split:rate(s[k]['all_windows']['mean_contact_slip_percent'])),row(label+'完整生成的任务数',lambda s,k=split:f"{s[k]['completed_cases']}/16"),row(label+'已评分窗口／计划窗口',lambda s,k=split:f"{s[k]['generated_scored_windows']}／{s[k]['planned_scored_windows']}"),row(label+'无有效接触窗口数',lambda s,k=split:s[k]['all_windows']['undefined_slip_windows'])]
    lines+=['','## 如何理解结果','','主滑步率：原评测认为脚处于接触状态时，水平脚速超过0.15m/s的比例；各短窗取算术平均。旧口径的分母是全部脚帧，两者不可混称。同一接触判定、帧率、阈值和归约用于新旧两组；所有失败测试保留，无有效接触不按0%计。接触覆盖和脚速一起报告，避免把缺少接触误当成滑步改善。', '', '连续任务若提前失败，已生成窗口仍参与滑步统计，并明确报告缺失窗口和失败任务；不把尚未生成的窗口补成0%。基础动作、质量和完整连续通过率必须一起看，不能只凭滑步低或位置更准判定成功。', '', '新旧均为50k×整体32样本=160万次总曝光；旧训练为BONES，新训练为70%BONES＋15%HumanML3D＋15%HiPHI。初始化、目标及历史训练处理也不同，因此这是整套配方的比较，不能单独证明某个损失函数更优。这里没有真实界面、文本实时响应或SONIC动力学仿真验收。', '', '旧50k复用此前已核验的完整逐样本结果，本次只计算其汇总滑步率；没有重新训练或替换旧权重。新50k使用同一基础96项及开发／保留各16条连续任务、相同源动作／文本／控制／噪声、25FPS、52帧窗口、10步去噪与CFG2/2。模型、codec、统计量和推理合同核验一致。', '', '## 原始证据','',f"旧结果：{OLD}",f"新结果：{nr}",f"本报告汇总：{out/'SUMMARY.json'}",f"新权重：{loads['New']['checkpoint']}",'']
    tmp=out/'REPORT.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(out/'REPORT.md')
    atomic(out/'SUMMARY.json',result)
    atomic(out/'DONE.json',dict(complete=True,self_test=self_test,report=str(out/'REPORT.md'),summary=str(out/'SUMMARY.json'),time_beijing=result['time_beijing']))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--new-root',default=str(ROOT/'evaluation/50000'));p.add_argument('--new-load',default=str(ROOT/'evaluation/50000/LOAD_evaluate.json'));p.add_argument('--output',default=str(ROOT/'evaluation/50000/comparison'));p.add_argument('--self-test',action='store_true');a=p.parse_args()
    r=evaluate_report(a.new_root,a.new_load,a.output,a.self_test);print(json.dumps({'complete':r['complete'],'self_test':r['self_test'],'delta':r['delta_new_minus_old']},ensure_ascii=False))
