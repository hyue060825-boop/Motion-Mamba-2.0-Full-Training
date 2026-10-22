"""Write a readable three-source report from complete bound pilot QA."""
import json
from g1_pipeline.common import ROOT,sha,atomic_bytes,write_json,now


def run():
 p=ROOT/'qa/aggregate/pilot_three_source_compact_qa.json';qa=json.loads(p.read_text())
 audit=json.loads((ROOT/'audit/pilot_final_evidence_audit.json').read_text())
 assert qa['status']=='PILOT_COMPARISON_ONLY' and audit['status']=='EVIDENCE_CONSISTENT'
 assert not qa['training_data_ready'] and not audit['training_data_ready']
 frozen=qa['bones_frozen_candidate_acceptance'];official=qa['sources']['bones_official_reference']
 assert frozen['status']=='FAIL' and frozen['core_input']==118 and frozen['core_kinematic_retained']==112
 lines=['# 三源 compact QA（仅隔离 pilot）','',
  'BONES v25 frozen acceptance 保持原 FAIL、118条core分母及全部绑定证据。HML/HiPHI均使用不变的v25；pilot结果不是生产放行或训练可用声明。','',
  '| 验收对象 | attempted | retained | rejected/error/pending | 状态 |','|---|---:|---:|---|---|',
  f"| BONES原冻结candidate | {frozen['attempted']} | core {frozen['core_kinematic_retained']}/{frozen['core_input']} | 原冻结记录保留 | FAIL |"]
 for key in ['humanml3d','hiphi']:
  r=qa['sources'][key]
  assert r['attempted']==100 and r['review_pending']==0 and r['accepted']+r['rejected']+r['error']==100
  lines.append(f"| {key} pilot | {r['attempted']} | {r['accepted']}/100 | {r['rejected']}/{r['error']}/{r['review_pending']} | {r['pilot_status']} |")
 lines.extend(['',f"BONES官方reference另做只读诊断：{official['measured']}/{official['expected']}条已测，contract PASS {official['contract_pass']}；flat-core硬QA {official['flat_core_hard_qa_pass']}/{official['flat_core_input']}。这些是既有配对时间支撑上的参考诊断，不是新的BONES训练池验收，不能用于替换原candidate的FAIL。",'',
  '共同目标为25 FPS、36维qpos、29物理DOF、34骨架节点、414-D表示。既有G1 normalizer保持不变。保留项必须通过数值、表示、CSV/official loader/project loader及inverse round-trip和对应scope硬QA；单个outlier直接拒绝。','',
  '| pilot | 处理帧数 | retained帧数 | contract PASS/已测 | retained skating p95最大(m/s) | flat-core retained/输入 |',
  '|---|---:|---:|---:|---:|---:|'])
 for key in ['humanml3d','hiphi']:
  r=qa['sources'][key]
  skate=r['accepted_skating_p95_max_m_s']
  lines.append(f"| {key} | {r['frames']} | {r['accepted_frames']} | {r['contract_pass']}/{r['contract_measured']} | {skate:.6f} | {r['flat_core_retained']}/{r['flat_core_input']} |")
 lines.extend(['','## 保留的问题与分组',''])
 for key in ['humanml3d','hiphi']:
  r=qa['sources'][key]
  lines.extend([f"### {key}",'',r['systematic_check_rationale'],'',
                '系统性检查：'+', '.join(k+'='+v for k,v in r['systematic_checks'].items())+'。','',
                '硬拒绝原因计数（一个clip可有多个原因）：'+json.dumps(r['hard_rejection_counts'],ensure_ascii=False,sort_keys=True)+'。','',
                '| 分组 | 输入 | retained |','|---|---:|---:|'])
  for name,g in sorted(r['groups'].items()):lines.append(f"| {name} | {g['input']} | {g['retained']} |")
  lines.extend(['','主要数值统计（已处理全部clip，包含拒绝项；不冒充retained-only统计）：','',
                '| 指标 | clip均值 | clip p95 | clip最大 |','|---|---:|---:|---:|'])
  for section in ['geometry','contract_errors']:
   for name,value in r[section].items():
    if value['measured_clips']:
     lines.append(f"| {name} | {value['mean']:.6g} | {value['p95']:.6g} | {value['max']:.6g} |")
  lines.extend([''])
 lines.extend(['HML DFaust native floor origin、HiPHI vault/scramble手撑地类别覆盖缺口都没有被重新标注、删除分母或通过调共享solver掩盖。图片观察不能覆盖correction budget或物理硬QA。','',
  'BONES五条RELATIVE_CATEGORY_UNSUPPORTED的只读溯源见 `reports/pilot_bones_unsupported_readonly.md`：两条transition与三条acrobatics未命中冻结relative标签；三条倒立又因原scene规则缺少handstand/对应描述覆盖而落入core。原分类和FAIL均保留。','',
  '视觉方法为有hash绑定的均匀抽帧及必要极值帧观察，加完整序列数值QA；完整25FPS视频作为证据。不是人类签名、逐帧完整观看或动态可跟踪性验收。','',
  '## 生产仍缺的证据',''])
 for item in qa['production_missing']:lines.append('- '+item)
 lines.extend(['','若后续修复source adapter或建立新的有效生产scope，需单独授权和验收；不得改写本轮固定分母、原BONES FAIL或冒用calibration binding。本轮未启动30条dynamic sanity、1000 canary、full batch、训练或真机控制。','',
  f"最终证据一致性审计：原BONES {audit['bones_original_files_verified']}项hash不变，7个production gates与原release checkpoint保持；production accepted={json.dumps(audit['production_accepted'],ensure_ascii=False)}，training_data_ready=false。",'',
  '机器可读详情：`qa/aggregate/pilot_three_source_compact_qa.json`；最终hash复核：`audit/pilot_final_evidence_audit.json`；并行吞吐、成本和条件ETA另见 `reports/parallel_execution_final.md`。'])
 out=ROOT/'reports/pilot_three_source_compact.md';atomic_bytes(out,('\n'.join(lines)+'\n').encode())
 write_json(ROOT/'audit/pilot_compact_report_binding.json',{'time':now(),'source_qa_sha256':sha(p),'final_audit_sha256':sha(ROOT/'audit/pilot_final_evidence_audit.json'),'report_sha256':sha(out),'production_release':False})
 print('Wrote',str(out.relative_to(ROOT)))
if __name__=='__main__':run()
