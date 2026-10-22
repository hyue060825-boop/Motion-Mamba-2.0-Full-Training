"""Aggregate the unchanged full75k panel and reuse the audited50k references."""
from motion_mamba2.paths import artifact_path
import importlib.util
from common import *
from motion_training.io import sha
spec=importlib.util.spec_from_file_location('audited_report',OLD/'auto50k/report50k.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
folder=ROOT/'evaluation/75000';out=folder/'comparison';out.mkdir(exist_ok=True)
load=read(folder/'LOAD_evaluate.json');assert load['stored_step']==75000
data={s:read(folder/s/'RESULT.json') for s in ['DEV','FINAL']}
assert read(folder/'COMPLETE.json')['complete']
assert load['checkpoint_sha256']==sha(load['checkpoint'])
for split,r in data.items():
    assert r['complete'] and r['evaluation_scope']=='full' and r['split']==split
    assert r['checkpoint']==load['checkpoint'] and r['checkpoint_sha256']==load['checkpoint_sha256']
    assert len(r['rows'])==16 and r['foundation']['windows']==(96 if split=='DEV' else 0)
summary=audit.summary(data['DEV'],data['FINAL'])
refs={};checks={}
for label,path in [('旧Native50k',Path(artifact_path('old_c1509953a8'))),('本轮融合50k',OLD/'evaluation/50000')]:
    loaded=read(path/'LOAD_evaluate.json');d={s:read(path/s/'RESULT.json') for s in ['DEV','FINAL']}
    assets={k:load[k]==loaded[k] for k in audit.ASSETS};assert all(assets.values()),assets
    foundation=lambda r:[(v['id'],v['source_id'],v['family'],v['prompt'],v['seed'],v['noise'],v['control']) for v in r['foundation']['rows']]
    sequences=lambda r:[(v['id'],v['source'],v['seed']) for v in r['rows']]
    paired={s:dict(metric_contract=data[s]['metric_contract_sha256']==d[s]['metric_contract_sha256'],foundation_inputs=foundation(data[s])==foundation(d[s]),long_inputs=sequences(data[s])==sequences(d[s])) for s in data}
    assert all(all(x.values()) for x in paired.values()),paired
    refs[label]=audit.summary(d['DEV'],d['FINAL']);checks[label]=dict(assets=assets,panels=paired,prior_results_reused=True)
result=dict(complete=True,checkpoint=load['checkpoint'],sha256=load['checkpoint_sha256'],stored_step=75000,
    summary=summary,references=refs,reference_checks=checks,physical_simulation_verified=False,
    averaging='96 foundation windows equal weight, including failures; undefined contact rates are missing, not zero; continuous generated windows equal weight, with full completion counts reported',
    comparison_limit='Same evaluation, different training steps and total sample exposure; not an isolated loss or batch comparison.')
write(out/'SUMMARY.json',result)
columns=[refs['旧Native50k'],refs['本轮融合50k'],summary]
def row(label,fn):return '| '+label+' | '+' | '.join(str(fn(s)) for s in columns)+' |'
def percent(v):return '未定义' if v is None else f'{v:.2f}%'
lines=['# 最新融合75k同条件测评','',f"权重：{load['checkpoint']}",'',
    '| 指标 | 旧Native50k | 本轮融合50k | 本轮融合75k |','|---|---:|---:|---:|',
    row('基础动作功能通过',lambda s:f"{s['function']}/96"),row('功能与动作质量同时通过',lambda s:f"{s['joint']}/96"),
    row('纯文本冷启动功能',lambda s:f"{s['by_cell']['cold_text']['function']}/32"),
    *[row(label,lambda s,a=a:f"{s['by_action'][a]['function']}/12") for a,label in [('walk','走路'),('backwalk','后退'),('turn','转向')]],
    row('短窗平均接触滑步率',lambda s:percent(s['foundation_foot']['mean_contact_slip_percent'])),
    row('短窗平均接触脚速m/s',lambda s:round(s['foundation_foot']['mean_contact_speed_m_s'],4) if s['foundation_foot']['mean_contact_speed_m_s'] is not None else '未定义'),
    row('短窗接触覆盖',lambda s:percent(100*s['foundation_foot']['mean_contact_fraction']) if s['foundation_foot']['mean_contact_fraction'] is not None else '未定义'),
    row('短窗滑步率未定义数',lambda s:s['foundation_foot']['undefined_slip_windows']),
    *[row(label+'完整连续任务通过',lambda s,k=k:f"{s[k]['joint']}/16") for k,label in [('DEV','开发组'),('FINAL','保留组')]],
    *[row(label+'连续窗口平均滑步率',lambda s,k=k:percent(s[k]['all_windows']['mean_contact_slip_percent'])) for k,label in [('DEV','开发组'),('FINAL','保留组')]],
    *[row(label+'完整生成任务数',lambda s,k=k:f"{s[k]['completed_cases']}/16") for k,label in [('DEV','开发组'),('FINAL','保留组')]],
    '', '同一文本、动作来源、空间条件、噪声、25FPS、52帧窗口、10步去噪和CFG2/2。旧50k及本轮融合50k直接复用已核验的逐例结果。',
    '', '75k之前整体batch32，总曝光240万次。75k比50k多了训练步数和曝光；这个表比较实际能力，不能单独归因于某项损失。',
    '', '滑步率包含失败样本。无有效接触不计为0%，同时报告脚速、接触覆盖和生成完整性。这里是生成结果测评，尚未进行真实Viewer和SONIC动力学仿真验收。',
    '', f"原始数据：{folder/'DEV/RESULT.json'}；{folder/'FINAL/RESULT.json'}。"]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=True,function=summary['function'],joint=summary['joint'],slip=summary['foundation_foot']['mean_contact_slip_percent'],DEV=summary['DEV']['joint'],FINAL=summary['FINAL']['joint']),ensure_ascii=False),flush=True)
