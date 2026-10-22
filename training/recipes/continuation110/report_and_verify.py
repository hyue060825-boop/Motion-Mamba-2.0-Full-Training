from common import *
import statistics,collections,math
from motion_training.io import rows
v=ROOT/'acceleration_v5';final=read(v/'FINAL_TIMING.json');live_weights=read(v/'LIVE_WEIGHT_GATE.json')
assert live_weights['passed'] and read(v/'ADMISSION.json')['passed']
old=final['medians']['current'];new=final['medians']['optimized']
records=[r for r in rows(ROOT/'train/train.jsonl') if r['step']>1180 and ':v5:graph' in r.get('speed_backend','')]
assert len(records)>=20,'Wait for at least20 real resumed updates'
assert time.time()-read(ROOT/'LIVE.json')['time']<90
assert sha(ROOT/'config.json')==read(v/'ADMISSION.json')['recipe_sha256']
assert not (ROOT/'BOUNDARY_REQUEST.json').exists()
assert all(math.isfinite(r['loss']) and math.isfinite(r['clip']) and r['world_size']==8 and r['global_batch']==512 and r['GT_slots']==448 and r['teacher_slots']==64 for r in records)
first=records[0];norms=[r['future_projection_update_norm'] for r in first['ranks']]
assert all(d==norms[0] for d in norms) and norms[0] and all(x>0 for x in norms[0].values())
assert all(r['history_stats'].get('graph_replays',0)>0 and r['history_stats']['extra'].get('replays',0)>0 and r['history_stats']['encoder'].get('replays',0)>0 for r in records[-1]['ranks'])
gate=read(ROOT/'gates/ACCELERATION_GATE.json');assert gate['passed'] and len(gate['checks'])==24
assert gate['checkpoint_sha256']==read(v/'ADMISSION.json')['checkpoint_sha256']
steady=records[2:];med=statistics.median(r['seconds'] for r in steady)
updates=rows(ROOT/'budget/UPDATES.jsonl');counter=collections.Counter(r['category'] for r in updates)
verification=dict(passed=True,first_step=records[0]['step'],latest_step=records[-1]['step'],resumed_updates=len(records),global_step=records[-1]['global_step'],steady_median_seconds=med,steady_min_seconds=min(r['seconds'] for r in steady),steady_max_seconds=max(r['seconds'] for r in steady),finite_losses_and_grad_norms=True,future_projection_first_update_norms=norms[0],all8rank_first_update_norms_exact=True,global_batch=512,world_size=8,GT_slots=448,teacher_slots=64,peak_GiB=max(x['peak_GiB'] for r in records for x in r['ranks']),config_sha256=sha(ROOT/'config.json'),budget_updates=dict(counter),time=time.time())
write(v/'PRODUCTION_VERIFICATION.json',verification)
text=f"""# Native112k 远期交互微调：v5 加速实测与恢复

本轮在第1180步完整保存后做隔离实验，再从同一保存点恢复。基础累计113180步，继续原定1180→2500→当前学生刷新→5000和独立验收。主工程与线上仿真未修改。

## 已接入与实际收益

接入三项仅用于eval且no-grad历史重建的数值计算图：Mamba完整chunk前向、原MLP/条件调制、输入与文本编码的纯数值部分。每次重放复制当前输入及全部历史状态，输出拥有独立存储；参数更新后读取同一存储中的最新参数。位置索引原验证仍逐次执行，没有删除检查。启用梯度或training时走原实现。

最终直接配对使用checkpoint1180、同一个g1教师池、实际第1182步、八卡448GT+64教师。每次恢复完整权重、FP32master、Adam、调度和随机状态，排除编译/捕获预热与诊断哈希，但完整计时包含读取、历史、当前前后向、同步和optimizer。

| 配对 | 原canonical v4完整步骤 | 最终v5完整步骤 |
|---|---:|---:|
| 1 | {final['timing']['current'][0]:.4f}秒 | {final['timing']['optimized'][0]:.4f}秒 |
| 2（反向顺序） | {final['timing']['current'][1]:.4f}秒 | {final['timing']['optimized'][1]:.4f}秒 |
| 中位数 | {old:.4f}秒 | {new:.4f}秒 |

完整步骤耗时减少{(1-new/old)*100:.2f}%，加速{old/new:.3f}倍。仅代表上述两组同样本配对和当前负载，不能按此承诺所有后续步骤，不能把不同进程的局部收益相乘。图的初始化时间和显存计入预算，首次捕获不计为稳定训练速度。

## 正确性

逐项八卡完整真实optimizer update检查：输入/全部有效历史数值与元数据/损失/梯度/裁剪/更新权重/Adam/随机状态一致，已采用项梯度相对差均为0。比较时启用原先已核验的确定性检查设置，生产训练不改精度/确定性设置，未放宽阈值。

另外在保持参数存储地址不变时，分别扰动Mamba投影、MLP、条件调制及文本映射参数：历史状态确实随新权重变化，参考与加速状态仍相同，第二轮无需重捕获。证据LIVE_WEIGHT_GATE.json。正式入口24项检查全部通过，覆盖legacy/arrival以及7种F外窗长度。历史数值不跨optimizer更新复用。

## 已试但没有接入

- 历史前缀复用：单独约2%收益，但与Mamba图结合未稳定叠加；不采用。
- 同一步文本/调制复用：等价通过，最终5.9265→5.9906秒，略慢；不采用。
- 重新平衡八卡历史：全球记录和输入一致，但梯度相对差约1.64e-4，权重/Adam未通过严格门禁；不采用，未声称其数学上不能实现。
- 真实等长历史合成B2：两段29窗口历史均出现数值状态差异，包括相位/SSM/K/V；不采用，也未用补零冒充等价。
- 输入编码首次捕获因GPU断言需要CPU判断而失败：保留原位置编码及检查在图外，仅捕获原函数的数值前半段，重新通过完整检查。
- 文本复用初次探针未兼容body的Identity入口，修正后测试通过但性能无收益；没有把失败探针用于训练。

## 实际恢复验证

八卡真实训练已从1180继续到{verification['latest_step']}，新增{verification['resumed_updates']}个真实更新。排除前两步后的中位耗时{med:.4f}秒，范围{verification['steady_min_seconds']:.4f}–{verification['steady_max_seconds']:.4f}秒，峰值allocated显存{verification['peak_GiB']:.2f}GiB。loss及梯度范数有限，首次远期投影参数更新八卡完全相同且非零。普通生产步骤的历史长短不同，这些速度不能代替配对数据计算加速百分比。

配置哈希不变，每卡56GT+8教师，全局batch512、GT448+教师64，其中旧交互32、远期32。样本/目标/损失/学习率/噪声/历史长度/BPTT/Adam和八卡RNG均保持原协议。本轮只改善耗时，不宣称动作质量改善。

## 预算、剩余瓶颈与边界

隔离探针累计checks类别{counter['checks']}个被丢弃的更新，全部入账，不写生产checkpoint、不当成有效训练。为响应本轮进一步优化，诊断上限由40分阶段记录为80；正式训练仍5000步、总GPU预算仍144小时，未改config.json。预算扩展与失败探针费用保留在*_BUDGET.json和budget/中。

余下仍有逐窗口历史递推、当前带梯度前后向/几何损失与卡间等待；没有删除历史或减少教师占比来制造提速。已测无收益或未通过一致性门禁的方案不上线。更大范围的整栈图、批量历史、重排归约或新backward内核属于下一轮研究，尚未证明等价或生产收益。此结论是本轮已验证的可用组合，不能宣称达到绝对硬件极限。

关键证据：CANDIDATE_GATES/FULL_TIMING、EXTRA_GATES/EXTRA_TIMING、ENCODE_GATES/ENCODE_TIMING、CONDITION_GATES/CONDITION_TIMING、BATCH_HISTORY_FEASIBILITY、FINAL_TIMING、LIVE_WEIGHT_GATE、ADMISSION、RESUME_ADMISSION、RESUMPTION、PRODUCTION_VERIFICATION，均位于acceleration_v5/。
"""
(v/'REPORT.md').write_text(text)
print(json.dumps(verification,ensure_ascii=False,indent=2))
