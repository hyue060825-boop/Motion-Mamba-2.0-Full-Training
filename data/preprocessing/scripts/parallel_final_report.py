"""Require complete measured evidence and write the compact parallel report."""
import json
import numpy as np
from collections import Counter
from pathlib import Path
from g1_pipeline.common import ROOT,sha,write_json,atomic_bytes,now,read_jsonl


def load(path):return json.loads((ROOT/path).read_text())

def checked(path,status='PASS'):
 value=load(path)
 if value['status']!=status:raise ValueError('REQUIRED_EVIDENCE_NOT_PASSED:'+path)
 return value

def run():
 protected=checked('audit/pilot_final_evidence_audit.json','EVIDENCE_CONSISTENT')
 exact=checked('audit/parallel_one_vs_four_exact.json')
 recovery=checked('audit/parallel_recovery_test.json');guards=checked('audit/parallel_guard_tests.json')
 reuse=checked('audit/parallel_boundary_reuse_test.json')
 io_retry=checked('audit/parallel_io_retry_test.json')
 schema=checked('audit/parallel_pilot_schema_test.json')
 hiphi_reuse=checked('audit/parallel_hiphi_reuse.json')
 upgrade=checked('audit/parallel_retry_executor_upgrade.json','VERIFIED')
 current_executor_sha=sha(ROOT/'g1_pipeline/parallel_runner.py')
 assert io_retry['executor_sha256']==schema['executor_sha256']==hiphi_reuse['executor_sha256']==upgrade['to_sha256']==current_executor_sha
 assert upgrade['schema_test_sha256']==sha(ROOT/'audit/parallel_pilot_schema_test.json')
 assert not upgrade['old_run_fingerprints_rewritten'] and not upgrade['frozen_algorithm_changed']
 assert sha(ROOT/upgrade['from_source_path'])==upgrade['from_sha256']
 assert sha(ROOT/upgrade['to_source_path'])==upgrade['to_sha256']
 assert upgrade['focused_test_sha256']==sha(ROOT/'audit/parallel_io_retry_test.json')
 comparisons={n:checked(f'audit/{n}_comparison.json') for n in ['benchmark_equivalence_w1_v1','benchmark_equivalence_w4_v1','benchmark_throughput_w16_v1','benchmark_throughput_w32_v1']}
 resources={w:checked(f'audit/benchmark_throughput_w{w}_v1_resources.json') for w in [16,32]}
 render=checked('audit/parallel_render_benchmark.json','MEASURED')
 select=checked('audit/parallel_worker_selection.json','MEASURED')
 inventory=load('audit/parallel_eta_inventory.json')
 if sha(ROOT/inventory['manifest_path'])!=inventory['manifest_sha256']:raise ValueError('ETA_INVENTORY_CHANGED')
 compact=checked('qa/aggregate/pilot_three_source_compact_qa.json','PILOT_COMPARISON_ONLY')
 for w,r in resources.items():
  base=ROOT/'work'/r['namespace'];design=load(str(base.relative_to(ROOT)/'run.json'))
  assert len(read_jsonl(base/'results.jsonl'))==64 and r['complete_clips']==64
  for path,bound in design['bindings'].items():
   if sha(ROOT/path)==bound:continue
   if path=='g1_pipeline/parallel_runner.py' and bound==upgrade['from_sha256']:
    # The measured v2 executor is preserved by hash. v4 keeps v3 retry/schema and corrects reused-only throughput validity;
    # record the version distinction rather than rewriting its old fingerprint.
    assert sha(ROOT/upgrade['from_source_path'])==bound
    assert r['counts'].get('ERROR',0)==0
    continue
   raise ValueError('FINAL_EXECUTOR_BOUND_INPUT_CHANGED:'+path)
  assert sha(base/'run.json')==r['bindings']['run'] and sha(base/'results.jsonl')==r['bindings']['ledger']
 assert resources[16]['panel_sha256']==resources[32]['panel_sha256']
 assert resources[16]['frames']==resources[32]['frames']==65296
 assert recovery['namespace']=='benchmark_recovery_v2'
 assert not protected['training_data_ready'] and protected['original_bones_status']=='FAIL'
 assert protected['original_core_input']==118 and protected['original_core_retained']==112
 roundoffs=[row['historical_roundoff'] for report in comparisons.values() for row in report['results'] if row.get('historical_roundoff')]
 max_qpos_roundoff=max((r['qpos_max_abs_rad'] for r in roundoffs),default=0.)
 max_qa_roundoff=max((d['max_abs'] for r in roundoffs for d in r['different_qa_scalars']),default=0.)
 max_root_roundoff=max((r.get('root_translation_max_abs_m',0.) for r in roundoffs),default=0.)
 execution_files=['g1_pipeline/parallel_clip.py','g1_pipeline/parallel_runner.py']
 validation_files=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('parallel_*.py'))
 changed_files={p:sha(ROOT/p) for p in execution_files+validation_files}
 tails={}
 for w,r in resources.items():
  measured=load('work/'+r['namespace']+'/summary.json')
  low_tail=[]
  for sample in reversed(measured['samples']):
   if sample['active']>4:break
   low_tail.append(sample)
  tails[w]={'clip_service_p50_sec':float(np.percentile(r['clip_wall_seconds'],50)),
            'clip_service_p95_sec':float(np.percentile(r['clip_wall_seconds'],95)),
            'clip_service_max_sec':max(r['clip_wall_seconds']),
            'final_at_most_four_active_workers_sec':max(0.,measured['wall_sec']-low_tail[-1]['elapsed_sec']) if low_tail else 0.,
            'tail_sampling_resolution_sec':5,
            'average_worker_cpu_cores':r['cpu_seconds_sum_worker_rusage']/r['actual_process_wall_sec_including_preflight']}
 selected=resources[select['selected_workers']]
 candidate_frames=sum(inventory['sources'][s]['estimated_candidate_25hz_frames'] for s in ['humanml3d','hiphi'])
 processing_hours=candidate_frames/selected['frames_per_sec']/3600
 video_all_hours=candidate_frames/render['render_frames_per_sec']/3600
 evidence={p:sha(ROOT/p) for p in ['audit/pilot_final_evidence_audit.json','audit/parallel_recovery_test.json','audit/parallel_guard_tests.json',
          'audit/parallel_hiphi_reuse.json','audit/parallel_boundary_reuse_test.json','audit/parallel_pilot_schema_test.json','audit/parallel_io_retry_test.json','audit/parallel_retry_executor_upgrade.json','audit/parallel_one_vs_four_exact.json','audit/parallel_projection_roundoff_diagnostic.json','audit/parallel_ground_roundoff_diagnostic.json','audit/parallel_worker_selection.json','audit/parallel_eta_inventory.json','audit/parallel_render_benchmark.json',
          'qa/aggregate/pilot_three_source_compact_qa.json']}
 for n in comparisons:evidence[f'audit/{n}_comparison.json']=sha(ROOT/f'audit/{n}_comparison.json')
 for r in resources.values():evidence[f'audit/{r["namespace"]}_resources.json']=sha(ROOT/f'audit/{r["namespace"]}_resources.json')
 report={'time':now(),'status':'PARALLEL_PILOT_WORK_COMPLETE_NO_PRODUCTION_RELEASE',
         'changed_execution_and_validation_files':changed_files,'measured_service_times_and_tail':tails,
         'selected_workers':select['selected_workers'],'speedup_32_over_16':select['speedup_32_over_16'],
         'equivalence':{n:{'status':r['status'],'compared':r['compared']} for n,r in comparisons.items()},
         'historical_root_translation_max_abs_m':max_root_roundoff,'historical_qpos_max_abs_rad':max_qpos_roundoff,'historical_qa_scalar_max_abs':max_qa_roundoff,
         'one_vs_four_full_clip_bit_exact':True,'recovery':'PASS','stop_resume':'PASS','bounded_io_retry':'PASS','original_pilot_evaluator_schema':'PASS',
         'current_executor_sha256':current_executor_sha,'measured_executor_sha256':upgrade['from_sha256'],
         'retry_extension_fresh_throughput_remeasured':False,'hiphi_old_outputs_reused':hiphi_reuse['reused_clips'],'hiphi_solver_reruns':hiphi_reuse['new_solver_attempts'],'hml_old_outputs_reused':100,'solver_reruns_for_hml_reuse':0,
         'frozen_bones_files_unchanged':protected['bones_original_files_verified'],
         'production_accepted':protected['production_accepted'],'pilot_sources':protected['sources'],
         'future_metadata_candidate_scenario':{'hml_plus_hiphi_estimated_25hz_frames':candidate_frames,
             'solve_export_hardqa_hours_using_measured_panel_rate':processing_hours,
             'optional_25_percent_scheduling_io_contingency_hours':processing_hours*.25,
             'render_every_frame_single_renderer_hours':video_all_hours,
             'render_policy_formula':'rendered source fraction * all-frame rendering hours; approved production selection unknown',
             'archive_staging_hours':None,'exception_visual_review_hours':None,'bones_official_export_hours':None,
             'remaining_authorized_pilot_clips':0,
             'limitations':'Metadata duration estimates include mirrors and scenes awaiting admission; mixed 64-clip full-run rate includes its observed long tail, initialization, export and QA. Archive extraction/full-library mapping staging, exception review, dynamic QA and BONES export costs are unmeasured. No ETA for an approved full three-source production release can be asserted.'},
         'evidence':evidence,'training_data_ready':False,'production_release':False}
 write_json(ROOT/'audit/parallel_final_acceptance.json',report)
 lines=['# 多 worker 与隔离 pilot 实测报告','',
  '本次完成范围为并行执行器验证和已授权两源 pilot。BONES frozen acceptance 永久保持 FAIL（core 112/118），生产 accepted 仍为0；未启动 dynamic sanity、1000 canary、full batch 或训练。','',
  '## 调度及正确性','',
  '`g1_pipeline/parallel_clip.py` 从原 hash-bound 串行模块编译单条完整 clip 处理逻辑；`parallel_runner.py` 使用动态领取、每条新进程、原 SQLite WAL/per-item flock 和单 coordinator 有序 ledger。每个 worker 在数值库导入前固定三个线程环境变量为1。solver、参数、精度、时间轴、后处理、阈值及原 production 入口不变。资产诊断只写新 namespace。','',
  f'初版1 worker及乱序4 worker四条完整输出逐字节一致，覆盖两源普通动作、已知硬拒绝和5220帧长片。实测v2调度版本在16/32各64条完整动作上再验证。相对历史串行，最终hinge最大差值{max_qpos_roundoff:.3e}rad、root translation最大差值{max_root_roundoff:.3e}m，派生QA标量最大差值{max_qa_roundoff:.3e}；原有ProxQP速度投影及地面lift QP在单进程相同输入重放时可复现末位浮点波动。CSV、414-D、时间轴、human-target数组、contract及接受/拒绝结论保持完全一致。比较沿用冻结numeric合同，独立重算差异派生指标；未增加solver或硬QA容差。初始严格字节比较的FAIL报告保留在executor_history。路径、运行时间、PID、attempt等执行记录单列。','',
  '真实受控中断、重复 coordinator/worker、clip边界 stop/resume、错误 native hash、production guard均通过。已提交项不重算，中断项仅恢复一次，无残缺成功项/重复sample/占用中的孤儿lock。HML100在新隔离namespace直接引用原产物，额外solver尝试为0；原HML验收不变。HiPHI原runner自然完成后，使用实测选定worker数在v4隔离pilot入口引用其100条产物，新增求解0条、原gate/accepted/ledger不变；没有把复用耗时计作吞吐。','',
  '初版executor代码按hash存档，原测试fingerprint未重签；最终版只增加resume统计、STOP消费和已完成pilot产物验证引用。原HiPHI无安全停止API，保留其自然完成，无同panel第二个完整runner。随后v3增加显式瞬态I/O有限重试（总attempt最多2）及原pilot验收器要求的scope/fingerprint字段桥接，真实注入测试确认恢复、耗尽以及schema错误不重试；原失败attempt产物保留。v4仅继续修正纯复用运行的吞吐有效性统计，纯复用不报frames/s。吞吐为存档v2执行器实测，最终v4在全新namespace重新通过重试路径和原验收器真实执行验证；后者引用100条HML旧产物且在写gate前截获，并断言纯复用吞吐为空，未改原HML结果。v3源码/测试/升级证据按hash保留。没有冒称重测最终版整套吞吐。','',
  '## 吞吐与资源','',
  '| workers | 完整clips/frames | 总耗时(s) | frames/s | 源动作小时/小时 | 采样峰值RSS(GiB) | Chrome平均占核 |',
  '|---:|---:|---:|---:|---:|---:|---:|']
 for w,r in resources.items():
  lines.append(f"| {w} | {r['complete_clips']}/{r['frames']} | {r['actual_process_wall_sec_including_preflight']:.2f} | {r['frames_per_sec']:.3f} | {r['source_motion_hours_per_hour']:.3f} | {r['peak_sampled_process_tree_rss_bytes']/2**30:.3f} | {r['chrome_mean_cpu_cores']:.2f} |")
 lines.extend(['',f"32/16实测速率比为{select['speedup_32_over_16']:.3f}，选择{select['selected_workers']} workers；未测48。两次使用同一64条完整panel及缓存政策（不清除其他任务的OS缓存，16先于32）；拒绝与待复核都计入成本。数据点包含长尾，不能外推线性加速。Chrome等其他任务未被修改。",'',
  '| 阶段（worker累计秒，不能直接相加当并发墙钟） | 16 | 32 |','|---|---:|---:|'])
 phases=sorted(set(resources[16]['phase_seconds_sum_across_workers'])|set(resources[32]['phase_seconds_sum_across_workers']))
 for phase in phases:lines.append(f"| {phase} | {resources[16]['phase_seconds_sum_across_workers'].get(phase,0):.2f} | {resources[32]['phase_seconds_sum_across_workers'].get(phase,0):.2f} |")
 lines.extend(['',f"渲染独立测试{render['full_clips']}条/{render['frames']}帧，单renderer、ffmpeg 2线程：{render['wall_sec']:.2f}s，{render['render_frames_per_sec']:.3f} frames/s，采样峰值{render['peak_sampled_process_tree_rss_bytes']/2**30:.3f} GiB。同四条求解/导出/QA与渲染计时之和为{render['four_clip_compute_qa_plus_render_seconds_sum']:.2f}s；这是四条的分阶段实测，不冒充64条全渲染端到端吞吐。EGL使用现有默认设备，没有声称分散到8卡。",'',
  'CPU、I/O、RSS时间序列、每clip服务时间、线程记录与原始GPU采样保存在对应resources及render报告中。求解表包括解析/FK、初始化、IK、后处理、完整硬QA/导出；不包括视频和额外人工/agent复核。','',
  *[f"{w} workers：clip服务耗时p50={v['clip_service_p50_sec']:.2f}s、p95={v['clip_service_p95_sec']:.2f}s、max={v['clip_service_max_sec']:.2f}s；最后不超过4个worker活跃的长尾约{v['final_at_most_four_active_workers_sec']:.2f}s（5秒采样）。累计worker CPU时间/真实墙钟约{v['average_worker_cpu_cores']:.2f}核。" for w,v in tails.items()], '',
  '## 最终pilot与放行','',
  '| 来源 | attempted | kinematic retained | rejected | pending | 状态 |','|---|---:|---:|---:|---:|---|'])
 for short,r in protected['sources'].items():lines.append(f"| {short} | {r['attempted']} | {r['accepted']} | {r['rejected']} | {r['review_pending']} | {r['status']} |")
 lines.extend(['','HML DFaust native floor-origin问题、HiPHI vault/scramble手支撑分类覆盖缺口均保留；未改标签、分母或阈值。所有保留项仍受数值、表示、round-trip及scope硬QA约束。视觉结论限于有hash绑定的抽帧观察和完整序列数值指标，不冒充人类签字或逐帧观看。三源compact QA与具体production缺口见 `reports/pilot_three_source_compact.md`。','',
  f"最终核验原BONES {protected['bones_original_files_verified']}条证据及7个生产gate/原release checkpoint，均未改变；生产accepted=0，training_data_ready=false。五条BONES unsupported原因见 `reports/pilot_bones_unsupported_readonly.md`，未重签BONES PASS。",'',
  '## 条件ETA','',
  f"已授权pilot剩余0条。按当前metadata inventory，HML+HiPHI候选约{candidate_frames:,.0f}个25Hz帧；以选择的实测速率估算，求解+导出+硬QA约{processing_hours:.1f}小时，可另列25%排队/I/O规划余量{processing_hours*.25:.1f}小时（余量是假设，不是实测）。",'',
  f"若每帧都渲染，单renderer另约{video_all_hours:.1f}小时；实际新增渲染时间须乘获批抽检比例。异常视觉复核、解包/全库mapping准备、dynamic QA和BONES官方导出未实测，不能给出完整三源生产ETA。当前候选包含镜像和未获准场景，不代表production可接收规模。371.8小时已核验为HiPHI候选metadata时长，并非当前生产训练集时长。",'',
  '## 证据与命令','',
  '实际执行代码新增/修改为 `g1_pipeline/parallel_clip.py` 和 `g1_pipeline/parallel_runner.py`；验证、测量和报告脚本的完整路径/hash列在最终JSON的changed_execution_and_validation_files。原串行模块未修改。','',
  '当前可复现入口、工作目录、环境变量、stop/resume/status及验证记录见 `reports/commands.md`。完整机器可读验收索引为 `audit/parallel_final_acceptance.json`，其中每项证据均有SHA256绑定。'])
 atomic_bytes(ROOT/'reports/parallel_execution_final.md', ('\n'.join(lines)+'\n').encode())
 print({'status':report['status'],'selected_workers':report['selected_workers'],'processing_hours_scenario':processing_hours})

if __name__=='__main__':run()
