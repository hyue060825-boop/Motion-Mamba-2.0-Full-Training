from motion_mamba2.paths import artifact_path
import json,pathlib,statistics,time
roots=[pathlib.Path(artifact_path('g1_motion_mamba2_four_gpu_matched100k_20261009_a4a8fc8513')),pathlib.Path(artifact_path('g1_minimal_four_gpu_matched100k_20261009_9abe8ef732'))]
aggregate=[]
for root in roots:
 read=lambda p:json.loads((root/p).read_text())
 admission=read('gates/NATIVE_GEOMETRY_ADMISSION.json');live=read('LIVE.json');start=admission['resume_checkpoint']['step'];assert live['step']>start
 records=[]
 with (root/'train/train.jsonl').open() as fp:
  for line in fp:
   x=json.loads(line)
   if x['global_step']>start+20:records.append(x)
 sample=records[-50:];assert sample
 seconds=[x['seconds'] for x in sample];full=read('gates/GEOMETRY_FULL_UPDATE.json');old=read('gates/FULL_UPDATE_BEFORE_NATIVE_GEOMETRY.json')
 warmmean=statistics.mean(seconds);remaining=((100000-live['step'])*warmmean if live['batch']==512 else (75000-live['step'])*warmmean+25000*full['median_seconds'])/3600
 info=dict(architecture=admission['architecture'],current=live,resumed_from=start,production_sample_count=len(seconds),production_interval=[sample[0]['global_step'],sample[-1]['global_step']],production_mean_seconds=warmmean,production_median_seconds=statistics.median(seconds),production_mean_excludes_first_20_updates=True,batch512_probe_median_seconds=full['median_seconds'],old_batch512_probe_median_seconds=old['measurements'][1]['median'],remaining_training_hours_estimate=remaining,source=root.as_posix(),time=time.time())
 (root/'gates/GEOMETRY_ACCELERATION_RESULT.json').write_text(json.dumps(info,indent=2)+'\n');aggregate.append(info)
 b64=read('gates/GEOMETRY_BATCH64_ADMISSION.json');b8=read('gates/GEOMETRY_BATCH8_ADMISSION.json')
 report=f'''# Native计算加速接入结果（2026-10-09）

{info['architecture']}从{start}步的完整checkpoint继续，当前{live['step']}步。权重、Adam FP32主权重、学习率日程、样本游标与随机状态按原记录恢复。未改变训练数据、模型、损失系数或batch日程。

## 改动与原因

已有加速保留CPU四线程预取、冻结编码器B1 CUDA Graph、批量denoiser与连续训练。原来解码和几何损失按每个样本B1分别执行；本轮在每卡microbatch64阶段改为Native相同的批量解码和几何损失。每卡microbatch8仍用原B1方式。

B64六类H/F布局全部通过；B8批量候选{'通过' if b8['passed'] else '未全部通过，拒绝使用'}。门限始终为loss相对误差≤1e-5、参数梯度相对L2≤1e-3。B64最大梯度相对误差{max(x['batch_gradient_relative'] for x in b64['cases']):.8f}。这是容差等价，不宣称逐位等价。此前随机初始化附近的批量几何失败记录仍保留；本轮测试当前成熟权重，并未放宽门限。

四卡完整batch512更新从同一checkpoint和Adam出发，样本与噪声一致。loss、参数梯度、参数更新和scheduler检查通过；四个rank最后模型与optimizer状态哈希相同。参数更新相对L2 {full['ranks'][0]['parameter_update']['relative_l2']:.8f}，最大绝对差{full['ranks'][0]['parameter_update']['max_absolute']:.8g}。原optimizer_moments汇总包含step计数，仅作附带记录，不能当成每个Adam矩张量的独立精度证明。所有8次探针更新均丢弃，生产从未更新过的保存点恢复。

## 耗时

旧batch512四卡完整更新探针中位数{old['measurements'][1]['median']:.3f}秒；本轮完整更新探针中位数{full['median_seconds']:.3f}秒，六个计时包含遇到新形状的首次缓存开销。两轮不是交替配对的长期性能统计。

恢复后，排除前20步建立缓存，最新{len(seconds)}个实际生产更新（{sample[0]['global_step']}–{sample[-1]['global_step']}）在总batch{live['batch']}下，平均{warmmean:.3f}秒，中位数{statistics.median(seconds):.3f}秒。按实测及已验证batch512阶段估算，剩余纯训练约{remaining:.2f}小时；保存、系统负载和最终评测另计。

## 已保存证据

- gates/GEOMETRY_BATCH64_ADMISSION.json、GEOMETRY_BATCH8_ADMISSION.json：六类输入布局与梯度差。
- gates/GEOMETRY_FULL_UPDATE.json：四卡实际更新、参数差、峰值显存与速度。
- gates/NATIVE_GEOMETRY_ADMISSION.json：代码哈希、checkpoint哈希与接入范围。
- START_VERIFICATION.json：完整checkpoint和Adam恢复确认。
- gates/GEOMETRY_ACCELERATION_RESULT.json：恢复后的实际训练速度。
- 原train.py备份：train.py.before_native_geometry_accel。

隔离的serial_geometry_graph.py原型因不可捕获的CPU到GPU索引操作失败，日志保留。该原型没有接入生产，生产使用既有Native批量实现。完整动作质量仍由原benchmark验收；本轮完成的是计算加速。
'''
 (root/'GEOMETRY_ACCELERATION_REPORT.md').write_text(report)
 print(json.dumps(info,ensure_ascii=False),flush=True)
pathlib.Path(artifact_path('two_model_acceleration_json_4d4706321d')).write_text(json.dumps(aggregate,indent=2)+'\n')
