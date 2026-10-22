from motion_mamba2.paths import artifact_path
import json, pathlib, statistics, time
R=pathlib.Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'codec_graph_accel_20261007'
def read(p):return json.loads(p.read_text())
b=read(S/'BENCHMARK.json');g=read(S/'UPDATE_GATE.json');i=read(S/'INPUT_GATE.json')
ref=b['confirm_reference'];cand=b['confirm_candidate']
summary=dict(time=time.time(),previous_seconds=ref['mean_seconds'],new_seconds=cand['mean_seconds'],
    speedup=b['mean_speedup'],time_reduction_fraction=b['time_reduction_fraction'],
    samples_per_second=512/cand['mean_seconds'],input_exact=i['passed'],actual_update_gate=g,
    previous_peak_reserved_gib=ref['peak_reserved']/1024**3,new_peak_reserved_gib=cand['peak_reserved']/1024**3,
    cold_start_and_checkpoint_not_in_benchmark=True,full_training_quality_not_claimed=True)
live=read(S/'LIVE_RESULT.json') if (S/'LIVE_RESULT.json').exists() else None
if (S/'ATTRIBUTION.json').exists():summary['attribution']=read(S/'ATTRIBUTION.json')
if live:summary['live']={k:v for k,v in live.items() if k!='rows'}
lines=[json.loads(x) for x in (R/'train/train.jsonl').read_text().splitlines()]
current=[x for x in lines if x.get('execution_backend')=='frozen_serial_codec_graph_v2']
if current:
    subset=current[-40:] if len(current)>=60 else current[-20:]
    summary['recent_live']=dict(count=len(subset),global_step=75000+subset[-1]['step'],
        seconds=statistics.mean(x['seconds'] for x in subset),
        remaining_training_hours=(25000-subset[-1]['step'])*statistics.mean(x['seconds'] for x in subset)/3600)
(S/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
body=f'''# 第二轮训练流程提速（2026-10-07）

比较对象是已经完成第一轮优化的四卡训练，不是最初逐条解码的慢实现。

| 同一父权重、同一512样本的完整更新 | 第一轮优化后 | 本轮 |
|---|---:|---:|
| 稳态平均秒/步 | {ref['mean_seconds']:.4f} | {cand['mean_seconds']:.4f} |
| 样本曝光/秒 | {512/ref['mean_seconds']:.1f} | {512/cand['mean_seconds']:.1f} |
| 每卡峰值reserved显存GiB | {summary['previous_peak_reserved_gib']:.2f} | {summary['new_peak_reserved_gib']:.2f} |

本轮额外提速{b['mean_speedup']:.3f}倍，完整步骤耗时下降{b['time_reduction_fraction']:.2%}。

## 改动与范围

冻结FP32动作编码器仍逐条执行B1矩阵计算，保留原量化、数值和52帧历史上下文；只把固定形状的多个B1编码调用捕获成CUDA Graph，批量复制输入后一次提交。

ARDY编码器根据encoder.is_causal生成固定的标准因果掩码。原Transformer入口每次通过GPU计算并回读来重新识别此掩码，本原型仅在冻结编码调用内部显式提供同一is_causal设置，并恢复原forward。掩码本身不变。

限定全真实帧、统一长度且按4帧对齐的当前基础训练输入；历史按原52帧片段和52帧上下文切分。Graph输出在重放后复制，避免后续重放覆盖之前的历史或目标。

没有捕获生成器的梯度、解码器损失、Adam或通信，没有引入训练A/Δ kernel，不修改仿真控制运行时。训练配方、数据、随机噪声、损失、学习率、FP32精度边界与总batch512不变。

## 正确性与计时

两轮不同样本、每轮六种历史/未来长度、每组64样本，共768个样本，全部原张量字段严格数值相等；第二轮复用图并换入新输入，也验证了输出不会被重放覆盖。图形状：{i['graph_shapes']}，共重放{i['replays']}次。

四卡从相同完整保存点开始，使用同一采样记录。首个完整更新的梯度相对L2差{g['gradient']['total']['relative_l2']:.6%}，实际FP32主参数更新相对L2差{g['actual_adam_master_delta']['total']['relative_l2']:.6%}；方向余弦分别为{g['gradient']['total']['cosine']:.10f}和{g['actual_adam_master_delta']['total']['cosine']:.10f}。Adam状态也通过核验，各卡状态一致且有限，显存至少保留10%余量。

两次测试交换参考与候选顺序。每进程先执行六种长度预热，再测12个完整更新；含CPU准备、编码、模型、损失、反向、梯度通信、Adam和预算记账。启动加载、Graph首次捕获、保存和独立评测不在稳态计时内。主对照72次速度探针更新全部丢弃，不进入正式权重。

第一次捕获遇到原Transformer对GPU掩码的CPU判断，失败记录保存在attempt1_capture_rejection；当时自动恢复上一轮已验证的训练实现。显式提供相同因果设置后重新执行全部门禁，没有放宽输入一致性要求。

## 正式训练

'''
if 'attribution' in summary:
    a=summary['attribution']
    body+=f"额外做了18步隔离对照，只提供显式因果设置及已知全真实历史切分，不使用Graph：平均{a['eager_causal_seconds']:.4f}秒/步。再加入Graph为{a['graph_seconds']:.4f}秒/步。这18次更新也全部丢弃。对照用于区分反复GPU回读和小算子提交的成本，不宣称已把每项开销精确拆开。\n\n"
if live:
    body+=f"已从{75000+read(S/'BOUNDARY.json')['step']}步的完整保存点继承权重、Adam、学习率日程和采样位置续训；首12次实际更新完成并保存于{live['global_step']}步。首段平均{live['mean_seconds']:.4f}秒/步，可能含部分首次捕获开销。\n\n"
if 'recent_live' in summary:
    a=summary['recent_live'];body+=f"最近{a['count']}次实际更新平均{a['seconds']:.4f}秒/步，已推进至{a['global_step']}步。按此短期速度，到100k剩余纯训练约{a['remaining_training_hours']:.2f}小时；保存、重新加载、捕获、评测及系统负载另计。\n\n"
body+='本轮确认执行速度和数值门禁，不宣称长期参数逐位一致或最终动作质量已经改善。原定100k完整生成与交互验收保持。执行源码SHA、门禁和恢复点记录在ENABLED.json及checkpoint的execution_acceleration字段。\n'
(S/'REPORT.md').write_text(body)
print(json.dumps({k:v for k,v in summary.items() if k!='actual_update_gate'},indent=2))
