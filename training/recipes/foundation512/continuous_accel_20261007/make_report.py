from motion_mamba2.paths import artifact_path
import sys,json,statistics,time
from pathlib import Path
R=Path(artifact_path('batch512_after75k_20261007_c710d5b650'));S=R/'continuous_accel_20261007'
sys.path.insert(0,str(R));from common import *
bench=read(S/'BENCHMARK.json');gate=read(S/'UPDATE_GATE.json');cont=read(S/'CONTINUITY_GATE.json');live=read(S/'LIVE_RESULT.json')
recent=[json.loads(s) for s in (R/'train/train.jsonl').read_text().splitlines()[-40:]]
mean=statistics.mean(x['seconds'] for x in recent);stage=recent[-1]['step'];status=read(R/'STATUS.json')
r=bench['reference']['rows'][6:]
profile={k:statistics.mean(max(v[k] for v in x['detail']) for x in r) for k in ['denoiser','geometry_loss','optimizer']}
profile['input_and_codec']=statistics.mean(max(v['build'] for v in x['phase']) for x in r)
profile['forward_backward']=statistics.mean(max(v['forward_backward'] for v in x['phase']) for x in r)
profile['backward_and_ddp_residual']=profile['forward_backward']-statistics.mean(max(v['denoiser']+v['geometry_loss'] for v in x['detail']) for x in r)
summary=dict(time=time.time(),status=status,actual_stage_step=stage,actual_global_step=75000+stage,recent_seconds_per_step=mean,
 remaining_pure_training_hours=(25000-stage)*mean/3600,optimizer_candidate_enabled=cont['optimizer_copy_enabled'],
 reference_seconds=bench['confirm_reference']['mean_seconds'],candidate_seconds=bench['confirm_candidate']['mean_seconds'],
 candidate_speedup=bench['mean_speedup'],profile_seconds=profile,update_gate=gate,continuity_gate=cont,
 no_recipe_model_precision_or_data_change=True,quality_success_not_claimed=True)
write(S/'SUMMARY.json',summary)
text=f"""# 第三轮训练执行优化（2026-10-07）

已启用连续训练：每1000步仍保存完整checkpoint，但保存后继续使用同一进程中的模型、Adam和冻结编码器Graph缓存。原控制器、预算、停止请求以及空闲后四卡转八卡门禁保留。模型源码快照、训练配置哈希、采样器、global batch512、每卡micro64、精度和损失均不改。

## 实验结果

另一项候选把FP32 master梯度拷贝与回写改成批量提交，保留原AdamW。首个完整512样本更新的梯度、FP32主参数变化及Adam状态差异均为0，18步loss轨迹也完全一致。

但第一次有计时细分的对照为{bench['reference']['mean_seconds']:.4f}→{bench['candidate']['mean_seconds']:.4f}秒/步；交换顺序、不插入细分同步的复测为{bench['confirm_reference']['mean_seconds']:.4f}→{bench['confirm_candidate']['mean_seconds']:.4f}秒/步，仅下降{bench['time_reduction_fraction']*100:.2f}%。收益不稳定，未达到3%门禁，未部署参数拷贝候选。

四组各18次、共72次速度探针，加保存/重新加载连续性验证3次，所有75次隔离更新均丢弃；预算100次更新、4 GPU小时。没有使用探针训练后的权重替换正式权重。

## 连续性门禁

从同一完整权重、Adam、学习率日程和各卡随机状态开始，比较第一个更新后直接进行下一更新，与保存完整状态、独立新进程加载后进行同样下一更新。模型/优化器/日程哈希、loss和采样记录完全一致：{cont['passed']}。

训练函数AST仅增加原保存条件的`step%1000==0`，逆变换严格还原原函数AST；并穷举本次剩余更新检查每1000步保存。原保存、断点、有限数值、预算和输入合同检查保留。外部执行源码SHA和各门禁保存在ENABLED.json及checkpoint的continuous_execution字段。

正式路径另完成12次实际更新，并成功保存至{live['global_step']}步，随后恢复原控制器运行。12步是真实后续数据与优化器续训，不是探针数据。

## 耗时定位与实际进度

诊断均值：输入与冻结编码约{profile['input_and_codec']:.3f}秒；生成器前向约{profile['denoiser']:.3f}秒；几何loss前向约{profile['geometry_loss']:.3f}秒；反向及DDP等剩余约{profile['backward_and_ddp_residual']:.3f}秒；Adam与参数拷贝约{profile['optimizer']:.3f}秒。诊断插入同步，不能代替纯GPU时间，反向与DDP尚未拆开。

截至报告，实际到{75000+stage}步，最近40步平均{mean:.4f}秒/步；到100k剩余纯训练约{(25000-stage)*mean/3600:.2f}小时，保存和评测另计。

此前每1000步完整段的墙钟时间比更新日志累加多约68–81秒（6%–7%），包含保存、退出、控制器等待、重新加载约4.57GB完整权重和Graph预热。本轮只消除反复退出、重新加载与预热；完整保存仍保留，不能把6%–7%全部宣称为本轮提速。连续段的实际整段墙钟收益待首个完整1000步保存后核算，不能把计算步1.04秒说成又一次大幅提速。

训练仍使用4–7号卡；0–3号卡的他人任务未干预。100k原定完整生成、交互和滑步评测保留；执行提速不代表动作质量已改善。
"""
(S/'REPORT.md').write_text(text)
print(json.dumps({k:summary[k] for k in ['actual_global_step','recent_seconds_per_step','remaining_pure_training_hours','optimizer_candidate_enabled']},ensure_ascii=False))
