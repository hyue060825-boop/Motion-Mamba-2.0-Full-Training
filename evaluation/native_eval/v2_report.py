"""Chinese delivery reports derived from real completed artifacts."""
from motion_mamba2.paths import artifact_path
from .common import *
from .v2_prepare import DOMAINS

def textfile(name,lines):output_path(ROOT/name).write_text('\n'.join(lines)+'\n')

def supporting_reports():
    counts=read(ROOT/'panels/SEMANTIC_PANELS.json');cal=read(ROOT/'benchmarks/STATISTICAL_CALIBRATION.json')
    coverage={}
    for d in DOMAINS:
        for split in ('development','holdout'):
            rr=list(rows(ROOT/'data'/f'{d}_{split}.jsonl'))
            coverage[d+':'+split]={str(n):dict(families=len({r['source_family_id'] for r in rr if r['frames']>=n}),actors=len({r['actor_id_namespaced'] for r in rr if r['frames']>=n}),status='NOT_TESTED',note='source duration coverage only, not validated complete-sequence FID') for n in (225,500,1000)}
    write(ROOT/'LONG_SEQUENCE_COVERAGE_V2.json',coverage)
    contract=read(artifact_path('eval_contract_json_be08a2ae73'))
    contract.update(version='native-v2',feature='256D pre-projection distribution h; separately named retrieval checkpoint normalized projections',inference_precision='FP32 deterministic length-sorted batch32 with ID restoration; BF16 training only',normalization='v2 remaining train families only, family-balanced, std floor0.05',metric_qualification='per-capability CAPABILITIES.json; explicit diagnostic flag for non-PASS; no unsupported order/full-sequence calculation')
    contract['fid']['panel']='FID_native52_center_v2';contract['fid']['uncertainty']='finite-panel paired 1000 bootstrap; descriptive not population-calibrated; five fixed real-real splits'
    write(ROOT/'EVAL_CONTRACT_V2.json',contract)
    b=read(ROOT/'BASELINE_REPLAY.json');base=['# v1 只读回放','', 'v1 结论保持原样：HumanML3D FAILED_VALIDATION，BONES READY_FOR_GENERATOR_INTEGRATION。旧数据面板现在只作已暴露开发证据。','', '| 域 | R@1 | 顺序 family | FID 每侧 N | time_shuffle:52 增量 | v1 区间 |','|---|---:|---:|---:|---:|---|']
    for d in DOMAINS:
        r=b[d]['1701']['confirmation'];v=r['distribution']['variants']['time_shuffle:52']
        base.append(f'| {d} | {r["retrieval"]["4101"]["R@1"]:.4f} | {r["order_hard_negatives"]["N"]} | {r["distribution"]["split_half_N"]} | {v["delta_over_independent_real_baseline"]:.5f} | {v["family_bootstrap_delta_ci95"]} |')
    base += ['','handoff 的 BONES 225-family 匹配 N 诊断，5 个子集的区间全部跨零。不是据此将旧 FAIL 改为 PASS；该结果说明统计功效影响结论。独立 CPU 回放见 benchmarks/V1_STATISTICAL_REPLAY.json，保持 v1 architecture 与权重契约匹配。','四帧均值抹除块内完整特征行的顺序已定位；其诊断没有重算 velocity，不能等同所有物理编辑。片头/中心差异是待 R0/R1 检验的假设。顺序 panel 覆盖不足与基础 R 的能力绑定过宽，二者在 v2 独立报告。']
    textfile('BASELINE_REPLAY.md',base)
    textfile('CAPTION_AND_ORDER_PANEL_REPORT.md',['# Caption 与顺序 panel 审计','',
        '全部 captions 按单词边界扫描 then、before、after、followed by、subsequently、finally；按 family 去重。原始候选、来源、动作词、方向词、排除理由与早/晚高度和速度证据均在 panels/*_order_review.jsonl。',
        '这些运动学摘要不能验证自然的 A→B / B→A 反事实，不编造人工或时间标注核验。两域事件顺序能力保留 INSUFFICIENT_EVIDENCE；反放仅是辅助方向/自然性变化。','',
        '| 域/用途 | 顺序 caption 候选 | family | 已核验方向镜像对 | 已独立核验顺序对 |','|---|---:|---:|---:|---:|']+
        [f'| {k} | {v["order_caption_candidates"]} | {v["order_families"]} | {v["direction_pairs"]} | 0 |' for k,v in counts.items()]+['','后续需补充事件起止标注或人工复核的正序/错序文本及同长度自然动作；不能用当前模型 margin 挑例。'])
    textfile('UNCERTAINTY_METHOD_REPORT.md',['# 不确定性方法与限制','',
        '保留标准 256D、float64、ddof=1 FID，无 shrinkage/PCA 替代。固定 5 个 real-real 划分；每次 paired bootstrap 的真实候选和编辑候选共享索引、共享 reference。最终 1,000 次重采样，同时报告 percentile 和 centered basic 区间。HumanML3D 按 family，BONES 另按 actor 整组划分与重采样；保存每侧 N、协方差秩、有效秩。',
        '高斯模拟每个条件 100 次 Monte Carlo，每次 500 bootstrap。位置平移下使用与完整 FID 等价的协方差消去公式，并用完整公式核对。零效应 identity 编辑应为零；中心化平移零假设也检查假阳性。','',
        '| 维度 | 每侧 N | 平移效应 basic 覆盖率 | 中心化零效应假阳性 |','|---:|---:|---:|---:|']+
        [f'| {x["dimension"]} | {x["per_side_N"]} | {x["location_shift_coverage"]:.2f} | {x["centered_null_false_positive"]:.2f} |' for x in cal['results']]+['',
        '模拟确认 raw bootstrap FID 存在明显上偏，不能将其 percentile 区间当作已校准的总体 FID 区间。平移效应覆盖率不能推广到改变协方差的运动编辑。正式报告的编辑差值区间只描述固定有限 panel 的重采样稳定性；短窗验收还要求双种子、5 划分与 actor 敏感性稳定。未声称名义 95% 总体覆盖已解决。',
        '两个域均保存每侧 N=112/256/512 的 real-real 曲线（可用时），同时保留全量。相同 N 不令不同 encoder 的 FID 成为同一尺度。增加 bootstrap 次数不能增加独立动作数量。'])
    covariance=ROOT/'benchmarks/COVARIANCE_CALIBRATION.json'
    if covariance.exists():
        c=read(covariance);p=ROOT/'UNCERTAINTY_METHOD_REPORT.md'
        p.write_text(p.read_text()+'\n## 协方差变化额外校准\n\n高斯协方差缩放0.8，对每个条件24次独立试验、每次500 bootstrap；样本量有限导致 paired FID delta 也可能偏离总体效应，不能仅用平移测试来认证。\n\n| 维度 | 每侧 N | basic 覆盖 | percentile 覆盖 | 点估计有限样本偏差 |\n|---:|---:|---:|---:|---:|\n'+ '\n'.join(f'| {r["dimension"]} | {r["per_side_N"]} | {r["basic_coverage"]:.3f} | {r["percentile_coverage"]:.3f} | {r["finite_sample_delta_bias"]:.3f} |' for r in c['results'])+'\n')
    write(ROOT/'PERTURBATION_CONTRACT.json',dict(version='native-v2',motion_edits=['static','repeat4','repeat13','shuffle4','shuffle52','reverse','seam'],pose_consistency='copy complete pose rows including positions, rotations and heading; preserves per-frame FK residual; recompute forward-difference world velocity at25fps, duplicate last velocity',limitations='contacts copied from source pose; not re-estimated under new dynamics; seams may be physically unnatural; no dynamics/ground-contact validity claim',bad_data=['NaN','channel_shuffle','velocity_inconsistent','invalid_rotation6d','wrong_fps'],bad_data_scope='strict input rejection diagnostics, not quality semantics',held_out_from_training='none of these edits is used as supervised corruption labels',reverse='auxiliary only, changes direction/naturalness, never pure event-order counterexample'))
    textfile('RUNBOOK.md',['# v2 运行与恢复','',
        '代码目录：${MM2_WORKSPACE}/evaluator_native_v2。结果目录：${MM2_WORKSPACE}/evaluator_native_v2。依赖只读复用 v1 专用 Python 环境，不向生成器环境安装包。',
        '```bash','cd ${MM2_WORKSPACE}/evaluator_native_v2',artifact_path('python_-m_native_eval_prepare_ee68e2ba2d'),'tmux new-session -d -s evaluator_v2_1004 "bash ${MM2_WORKSPACE}/evaluator_native_v2/run_v2.sh > ${MM2_WORKSPACE}/evaluator_native_v2/pipeline.log 2>&1"','tmux attach -t evaluator_v2_1004','```','',
        'train-all 使用独立 gpu.lock，启动时重新查 GPU4–7 的活动进程，只选择无人占用的卡。无空闲卡时每15分钟检查一次。训练有 optimizer、scheduler、RNG、epoch/batch/step 和数据/代码/文本/统计量契约；相同入口可恢复，契约变化拒绝续训。',
        '独立入口：`audit` / `prepare` / `train --domain humanml3d --variant R2 --seed 1701` / `validate` / `freeze` / `report` / `score`。validate 需要已冻结选择与策略，不能重新选择已暴露 holdout。',
        '前期错误开发 panel 的训练保存在 rejected_setup_01，完全不参与 v2 选择，其 GPU 时间仍计费。v1 原状态和分数保留；主生成器不被改写或停止。新增 GPU 预算24小时、磁盘50GiB，data3/data4分别保留150/100GiB空闲。'])

def report(results,caps,sel,releases,smokes):
    supporting_reports();lines=['# G1 native evaluator v2 测评与交付','',
        '这里评估的是测评器，不是 ARDY/Minimal/Hybrid 的生成成绩。两域均为 G1Skeleton34、physical414、25fps 的 native/custom 指标，不与官方 HumanML3D 排行榜直接比较。','',
        '| 域 | 种子 | 检索配置/头 | 分布配置/头 | 检索 N | R@1 / R@2 / R@3 | 错配 R@1 | 同长度 R@1 | 分布 family |','|---|---:|---|---|---:|---|---:|---:|---:|']
    for d in DOMAINS:
        s=sel['selected'][d]
        for seed,r in results[d].items():
            rr=r['retrieval']['4101'];triple=' / '.join(f'{rr[f"R@{k}"]:.4f}' for k in (1,2,3))
            lines.append(f'| {d} | {seed} | {s["retrieval"]["variant"]}/{s["retrieval"]["head"]} | {s["distribution"]["variant"]}/{s["distribution"]["head"]} | {r["N"]} | {triple} | {r["shuffled"]["R@1"]:.4f} | {r["length_matched"]["R1"]:.4f} | {r["distribution"]["N"]} |')
    lines += ['','| 能力 | HumanML3D | BONES |','|---|---|---|']
    for k in caps['humanml3d']:lines.append(f'| {k} | {caps["humanml3d"][k]["status"]} | {caps["bones"][k]["status"]} |')
    lines += ['','## 控制消融','', '| 域 | 配置 | 检索头开发 R@1 | 分布头开发重建误差 |','|---|---|---:|---:|']
    for d,variants in sel['development'].items():
        for v,heads in variants.items():
            r=heads['retrieval']['development'];dist=heads.get('distribution',heads['retrieval'])['development'];lines.append(f'| {d} | {v} | {r["retrieval"]["R@1"]:.4f} | {dist["reconstruction"]:.5f} |')
    lines += ['','R0=旧四帧均值+片头短窗；R1=相同结构+原坐标开头/中间/结尾均衡短窗；R2=R1+四帧有序拼接投影。各域内部三者相同 train/development/holdout、优化器和更新上限；两域 family 数不同，相同50 epoch上限不等于相同优化更新数，实际 updates/曲线与早停在 ABLATION_RESULTS.json。完整语义文本监督不绑定到任意短窗。分布头由开发重建而非最低 FID 或损坏测试选择；该代理选择不保证分布能力通过。','',
        '## 分布与动态验证','', '| 域/种子 | 编辑 | 相对独立 real-real 增量 | basic 区间 | percentile 区间 | 正向划分/5 |','|---|---|---:|---|---|---:|']
    for d in DOMAINS:
        for seed,r in results[d].items():
            for k,v in r['distribution']['variants'].items():lines.append(f'| {d}/{seed} | {k} | {v["point"]:.5f} | {v["basic_ci95"]} | {v["percentile_ci95"]} | {v["positive_splits"]} |')
    lines += ['','完整逐样本检索、方向 margin、两种子缓存和统计分割在 validation/；BONES actor-cluster 结果保存在各 RESULT.json，未以行数冒充独立 actor。未通过的 shuffle/reverse/seam 结果均保留。','',
        '## 六点结论','',
        '1. v1 HumanML3D 已定位的问题是小样本高维 FID 稳定性、单 caption 顺序 panel 覆盖和结构上的块内均值信息损失；普通检索并非完全失效。已修正 tokenizer 不是本次仍在发生的故障。G1 重定向语义损失仍是未证实假设。',
        '2. v2 实际重划 evaluator holdout、重算 train-only 统计量、从随机初始化做 R0/R1/R2、修正短窗原坐标覆盖及保序模块、独立能力门禁、共享索引重采样与 actor 诊断。修改是否有效以以上受控开发结果和一次确认逐项判断，不能笼统称全部变好。',
        '3. 仅 CAPABILITIES.json 标 PASS 的具体指标可以用于限定协议的 native 主实验。basic R 不代表事件顺序，短窗 FID 不代表长时动作质量；未验证指标只允许显式诊断，缺少协议的事件顺序/完整长序列评分会拒绝。',
        '4. 新 holdout 对本轮随机初始化的 evaluator 编码器训练未见，但来自上游原 train，历史 v1 和生成器可能/已见。HumanML3D 按 family 留出但非 actor-disjoint，BONES 按 actor 留出。不能称生成器未见 test。',
        '5. v1 与 v2 的 encoder、统计量、reference 和 panel 不同，原始 FID 数值不可直接横比。旧生成结果若满足 v2 样本 panel/条件和完整元数据，可复用动作数组，但必须用对应 v2 编码器重算 embedding、FID 和 R；不能复用 v1 reference/cache。',
        '6. 训练、验证、release 与完整真实样本离线评分链路不依赖生成器新 checkpoint。接入仍需新生成器按冻结 panel 提供 generated-only physical414 与真实 checkpoint/code/sampling/time metadata；既有输出只用于导入和帧角色检查，缺失身份未编造。','',
        '## 审计与恢复','',
        '初次实现误将 train 作为 development，已停止并隔离，未用于正式选择；holdout 当时尚未执行模型推理。重跑记录、费用和修正原因保留。',
        'v1 只读内容核验见 V1_READONLY_VERIFICATION.json。离线真实样本评分冒烟见 OFFLINE_SCORE_SMOKE.json；生成输出导入检查见 EXISTING_GENERATOR_IMPORT_CHECK.json。训练曲线、完整 last checkpoint、契约、GPU/磁盘预算和恢复入口见 RUNBOOK.md。']
    textfile('VALIDATION_REPORT_V2.md',lines)
    textfile('INTERFACE_FOR_EXPERIMENT_V2.md',['# v2 实验接入','',
        'release 位于 releases/HumanML3D-G1-native-v2 与 releases/BONES-G1-native-v2。每项能力独立门禁，缺少能力时必须使用 --diagnostic；事件顺序和完整长序列缺少参考协议，诊断模式也不会凭空计算。',
        '```bash','cd ${MM2_WORKSPACE}/evaluator_native_v2',artifact_path('python_-m_native_eval_score__f94396cb6d'),'  --release ${MM2_WORKSPACE}/evaluator_native_v2/releases/BONES-G1-native-v2 \\','  --samples /path/to/completed_samples.jsonl \\','  --output ${MM2_WORKSPACE}/evaluator_native_v2/scores/my_run.json \\','  --metric retrieval','```','',
        '`--metric retrieval` 对应 retrieval_panel.json；`--metric fid` 对应 fid52_panel.json（52帧）；`--metric both` 对应 combined_panel.json（全语义段，FID取中心52帧，R取预注册子集）。每个 panel 的 sample ID 与 panel_sha256 必须完全匹配；不得按生成模型各自成功样本筛选。',
        '保留 motion_eval_sample_v1 字段。迁移到v2需要真实指定新 track/panel_id，核验原数组长度和条件匹配；绝不改写缺失的生成历史身份。raw 与投影/后处理结果通过 postprocess 分组分别评分。',
        '输入必须 physical414/G1Skeleton34/25fps/meters、未归一化，hash与数据一致；说明 array_layout、GT/history/preroll/padding角色、连续性、caption时间和采样条件。generated_only 不会再裁掉 H=104。不同域/release/normalizer/reference 禁止混用。',
        'smoke/*/real_reference_replay.jsonl 展示完整字段，其明确标为真实样本接口回放，不是生成器输出。现有生成输出缺失元数据见 EXISTING_GENERATOR_IMPORT_CHECK.json。'])
