"""Reuse audited metric aggregation; compare75k start against100k continuation."""
import importlib.util
from common import *
from motion_training.io import sha
spec=importlib.util.spec_from_file_location('audited50k',OLD/'auto50k/report50k.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
loads={step:read(ROOT/f'evaluation/{step}/LOAD_evaluate.json') for step in [75000,100000]}
data={step:{split:read(ROOT/f'evaluation/{step}/{split}/RESULT.json') for split in ['DEV','FINAL']} for step in loads}
assert all(loads[s]['stored_step']==s for s in loads)
assets={key:loads[75000][key]==loads[100000][key] for key in module.ASSETS};assert all(assets.values()),assets
checks=[]
for split in ['DEV','FINAL']:
    a=data[75000][split];b=data[100000][split]
    key=lambda r:[(v['id'],v['source_id'],v['family'],v['prompt'],v['seed'],v['noise'],v['control']) for v in r['foundation']['rows']]
    seq=lambda r:[(v['id'],v['source'],v['seed']) for v in r['rows']]
    check=dict(split=split,metric_same=a['metric_contract_sha256']==b['metric_contract_sha256'],foundation_same=key(a)==key(b),sequences_same=seq(a)==seq(b))
    assert all(v for k,v in check.items() if k!='split')
    assert all(r['complete'] and r['split']==split for r in [a,b])
    checks.append(check)
summaries={s:module.summary(data[s]['DEV'],data[s]['FINAL']) for s in data}
result=dict(complete=True,checkpoints={s:loads[s]['checkpoint'] for s in loads},summaries=summaries,
    shared_inference_assets=assets,evaluation_checks=checks,previous_exposures=2400000,
    new_exposures=12800000,total_exposures=15200000,physics_verified=False,
    causal_limit='Before/after comparison of continuation with increased batch and exposure. No unchanged-batch25k parallel control; cannot attribute differences uniquely to batch.')
out=ROOT/'comparison';out.mkdir(exist_ok=True)
write(out/'SUMMARY.json',result)
def row(label,fn):return f'| {label} | {fn(summaries[75000])} | {fn(summaries[100000])} |'
def rate(x):return '未定义' if x is None else f'{x:.2f}%'
lines=['#75k起点与总batch512继续训练至100k的结果','',
    '同一测评入口、数据、文本、空间条件和噪声；本报告不代表实际G1动力学仿真通过。','',
    '| 指标 |75k起点 |100k大批量继续训练 |','|---|---:|---:|',
    row('基础动作功能通过',lambda s:f"{s['function']}/96"),row('功能与动作质量同时通过',lambda s:f"{s['joint']}/96"),
    row('纯文本冷启动功能',lambda s:f"{s['by_cell']['cold_text']['function']}/32"),
    *[row(label,lambda s,a=action:f"{s['by_action'][a]['function']}/12") for label,action in [('走路','walk'),('后退','backwalk'),('转向','turn'),('蹲下','squat'),('弯腰','bend')]],
    row('平均接触滑步率',lambda s:rate(s['foundation_foot']['mean_contact_slip_percent'])),
    row('平均接触脚速',lambda s:s['foundation_foot']['mean_contact_speed_m_s']),
    row('接触覆盖比例',lambda s:s['foundation_foot']['mean_contact_fraction']),
    row('滑步率未定义窗口数',lambda s:s['foundation_foot']['undefined_slip_windows']),
    row('开发组完整连续任务',lambda s:f"{s['DEV']['joint']}/16"),row('保留组完整连续任务',lambda s:f"{s['FINAL']['joint']}/16"),
    row('开发组每窗平均滑步率',lambda s:rate(s['DEV']['all_windows']['mean_contact_slip_percent'])),
    row('保留组每窗平均滑步率',lambda s:rate(s['FINAL']['all_windows']['mean_contact_slip_percent'])),
    row('开发组完整生成任务数',lambda s:f"{s['DEV']['completed_cases']}/16"),
    row('保留组完整生成任务数',lambda s:f"{s['FINAL']['completed_cases']}/16"),
    '', '75k之前整体batch32，曝光240万次；之后整体batch512，新增1280万次，合计1520万次。曝光包含重复取样，不能当作独立动作文件数量。',
    '', '基础目标、数据来源比例、官方输入输出及文本映射保持原合同；取样计划改为按更新编号生成新的合法片段与噪声。Adam状态与学习率进度继承75k，未按batch倍数放大学习率。',
    '', '这是继续训练前后的比较，没有同时训练一条batch32到100k的控制线，因此不能单独证明大batch本身更优。动作正确性、质量和完整连续通过率需要共同改善；只看位置或滑步不能判为成功。',
    '', '## 原始证据','',str(ROOT/'evaluation/75000'),str(ROOT/'evaluation/100000'),str(out/'SUMMARY.json')]
(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
write(out/'COMPLETE.json',dict(complete=True,report=str(out/'REPORT.md'),physics_verified=False))
print('REPORT_COMPLETE',result['total_exposures'],flush=True)
