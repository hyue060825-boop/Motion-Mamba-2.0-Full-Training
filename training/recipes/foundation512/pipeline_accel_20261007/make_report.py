from motion_mamba2.paths import artifact_path
import sys,json,statistics,time
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'pipeline_accel_20261007'
sys.path.insert(0,str(R));from common import *
b=read(S/'BENCHMARK.json');live=read(S/'LIVE_RESULT.json');eq=read(S/'UPDATE_GATE.json')
ref=b['confirm_reference'];cand=b['confirm_candidate']
def phase(result,k):return statistics.mean(statistics.mean(p[k] for p in x['phase']) for x in result['rows'][6:])
summary=dict(benchmark=dict(reference_seconds=ref['mean_seconds'],candidate_seconds=cand['mean_seconds'],speedup=b['mean_speedup'],
    time_reduction_fraction=b['time_reduction_fraction'],reference_samples_per_second=512/ref['mean_seconds'],candidate_samples_per_second=512/cand['mean_seconds'],
    reference_build=phase(ref,'build'),candidate_build=phase(cand,'build'),reference_forward_backward=phase(ref,'forward_backward'),candidate_forward_backward=phase(cand,'forward_backward'),
    reference_peak_reserved_gib=ref['peak_reserved']/2**30,candidate_peak_reserved_gib=cand['peak_reserved']/2**30),
    selected=b['selected'],update_gate=eq,live=live,resumed=read(S/'RESUMED.json'),global_batch=512,microbatch=64,ranks=4,
    old_parent_immutable=True,probe_updates_discarded=True,quality_success_not_claimed=True)
if (S/'SERIAL_INPUT_GATE.json').exists():summary['serial_input_gate']=read(S/'SERIAL_INPUT_GATE.json')
write(S/'SUMMARY.json',summary)
v=summary['benchmark'];d=b['selected'];hours=live['remaining_training_hours']
text=f'''# 四卡训练流程提速实测（2026-10-07）

已接入通过核验的执行优化，正式训练已从完整保存点继承权重、FP32主参数、Adam状态、学习率日程和采样位置继续。训练配方、输入输出映射、损失系数、精度边界和总batch512保持不变。

| 完整四卡optimizer update | 原实现 | 优化后 |
|---|---:|---:|
| 稳态平均秒/步 | {v['reference_seconds']:.4f} | {v['candidate_seconds']:.4f} |
| 样本曝光/秒 | {v['reference_samples_per_second']:.1f} | {v['candidate_samples_per_second']:.1f} |
| 每卡前向/损失/反向平均秒 | {v['reference_forward_backward']:.4f} | {v['candidate_forward_backward']:.4f} |
| 每卡峰值reserved显存GiB | {v['reference_peak_reserved_gib']:.2f} | {v['candidate_peak_reserved_gib']:.2f} |

完整步提速{v['speedup']:.3f}倍，时间下降{v['time_reduction_fraction']*100:.2f}%。当前仍为4–7号卡，每卡64样本、累积2次，总batch512。选择的CPU读取线程数为{d['workers']}，执行模式为{d['mode']}。

## 核验与测试范围

六种H/F长度组合，每种64样本；覆盖纯文本、无条件、不同空间条件、条件丢弃和不可见未来。保留逐条编码后，优化输入、目标、噪声、文本、约束、掩码及解码历史通过完全数值相等检查。

四卡完整512样本的实际更新对照，两组来自同一checkpoint、同一模型和Adam状态，使用相同采样记录。损失相对差{eq['loss_relative_difference']:.8g}；梯度相对L2差{eq['gradient']['total']['relative_l2']*100:.5f}%，方向余弦{eq['gradient']['total']['cosine']:.10f}；FP32主参数实际更新相对L2差{eq['actual_adam_master_delta']['total']['relative_l2']*100:.5f}%，方向余弦{eq['actual_adam_master_delta']['total']['cosine']:.10f}。各卡模型和Adam状态一致，有限数检查及显存门禁通过。

时间对照包含CPU准备、冻结编码、生成器、可微解码、几何损失、反向、梯度通信、优化器和预算记账。每个进程先预热六种形状，再计时；交换参考/候选执行顺序复测。计时不包含启动加载、保存checkpoint、评测及附加梯度序列化。准备阶段也包含冻结编码GPU工作；新方案CPU提前准备包含在完整步时间内，但不全部落在原build阶段的计时范围。前向/反向阶段包含损失和同步等待，不能当作纯GPU时间。

## 实施内容与保留行为

- 批量解码与FK/几何损失，保持先按样本归约、再求平均，保留有效G52帧和解码历史。
- 非审计更新不再逐样本取回损失标量；仍保留必要有限数、梯度范数和预算记录。
- CPU侧读取和控制mask准备与GPU工作分离；测试0/2/4/8个读取线程，复用预计算文本向量缓存，批量传输输入。
- 批量编码原型改变了少量量化latent，未通过目标完全一致检查，因此拒绝接入。正式方案保留原单样本编码形状与52帧历史边界。
- 未接入训练A/Δ原型、训练Graph或新精度，不覆盖控制运行时；未停止0–3号卡的其他作业。

## 正式续训

真实续训前12步完成并保存：global step {live['global_step']}，checkpoint `{live['checkpoint']}`。排除前两步后实际平均{live['mean_seconds']:.4f}秒/步，约{live['samples_per_second']:.1f}样本曝光/秒。

按这段真实速度，到100k剩余纯训练约{hours:.2f}小时；保存、进程重新加载、后续评测和服务器负载另计。12步包含的长度比例不一定等于长期比例，因此这个预计不是固定承诺。

数值通过容差核验，不宣称参数更新逐位相等或长期收敛完全一致。本轮验证的是训练速度，模型最终动作质量仍由原定100k完整生成和交互测评验收。

## 原始证据

`INPUT_LOSS_GATE.json`保留被拒绝的批量编码结果；`SERIAL_INPUT_GATE.json`保留通过的输入核验；`UPDATE_GATE.json`为实际更新核验；`BENCHMARK.json`及`probes/*/updates.jsonl`保留全部配置和原始时间；`LIVE_RESULT.json`保留真实续训日志；`ENABLED.json`及checkpoint中的execution_acceleration记录执行源码SHA256和门禁。所有速度探针更新均丢弃，不进入正式权重。
'''
(S/'REPORT.md').write_text(text)
print(json.dumps(dict(benchmark=v,selected=d,live_seconds=live['mean_seconds'],global_step=live['global_step'],remaining_hours=hours),indent=2))
