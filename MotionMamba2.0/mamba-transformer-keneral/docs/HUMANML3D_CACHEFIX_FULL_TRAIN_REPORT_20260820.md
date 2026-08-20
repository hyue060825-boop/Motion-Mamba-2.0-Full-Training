# HumanML3D Cache-Fix 50k 全量训练报告

## 一、结论先行

Minimal 与 Root-Mamba/Body-Attention Hybrid 已完成 HumanML3D 4 卡、50k step 全量训练。两次训练均通过数值稳定性门控：梯度有限、loss 明显下降、验证 loss 下降，且约一半训练 batch 使用了 sequential-cache 路径。修正 cache 提交时机和 recenter 坐标生命周期后，persistent-memory 的长时序边界行为明显比修复前稳定。

但本轮不能宣称“复现了 ARDY 论文 Table 4/5”，也不能宣称 Hybrid 已经优于 ARDY。当前还没有在同一 HumanML3D 训练集、测试集、训练预算和生成协议下训练的 HumanML3D ARDY baseline；Table 4/5 中的几何、skate 和延迟是统一协议下的 proxy，R-Precision/FID 仍因 CoreSkeleton27/330-D 到 HumanML3D 22-joint/263-D 的生成结果转换未完成而暂缺。

## 二、规则审计

| 规则 | 本报告状态 |
|---|---|
| 官方 checkpoint 只能标为 `Released ARDY native checkpoint` | 遵守。本报告不把官方 checkpoint 写成 HumanML3D-trained ARDY baseline。 |
| 正式架构对比必须使用同一 HumanML3D 管线 | Minimal 与 Hybrid 的内部 proxy 对比使用同一测试 manifest、生成协议和指标实现；正式 ARDY 架构对比尚未完成，因为 HumanML3D-trained ARDY baseline 尚未训练。 |
| 50k 只是训练预算 | 遵守。50k 被记录为当前预算，不等于最终收敛；是否延长到 100k/150k 要根据 loss、Table 4/5 和长时序稳定性决定。 |
| 主表采用 ARDY 论文风格 Table 4/5 | 本报告主表只放模型结果；Dataset calibration 不放入主表，仅作为评测器附录证据。 |
| 区分正式、统一协议和 smoke/proxy | 训练 PASS 是正式训练工程结果；Table 4/5 是统一协议下的 proxy 结果；不把它们写成官方论文指标。 |

## 三、全量训练结果（正式训练工程结果）

| 方法 | GPU / 全局 batch | steps | 用时 | 最终 eval loss | sequential-cache steps | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| Minimal | 4 / 32 | 50,000 | 6.36 h | 0.3135 | 24,846 | 6,192 MiB |
| Hybrid | 4 / 32 | 50,000 | 8.45 h | 0.2997 | 24,846 | 7,484 MiB |

共同配置：`sequential_cache_probability=0.5`、10-step diffusion evaluation、`llm2vec_cache` 文本后端、SDPA attention。训练在 tmux 会话 `hml3d_full_cachefix_20260820` 中完成，两个 `metrics.json` 的 `status` 均为 `PASS`，`all_gradients_finite` 均为 `true`。

Checkpoint：

- Minimal：`/data4/zts/result/humanml3d_full_cachefix_20260820_minimal/minimal_humanml3d_ddp.pt`
- Hybrid：`/data4/zts/result/humanml3d_full_cachefix_20260820_hybrid/hybrid_humanml3d_ddp.pt`

SHA256：Minimal `40e4cc53d85edbdb8126cd6df6b1aa75227006e114c4558063375d369e8ef72f`；Hybrid `03f42714cd549e2a398658aa15feb40d0876694dbc8aa1a8be8e08b8eea55344`。

## 四、Table 4：40 帧统一协议 proxy

| Method | R-Precision ↑ | FID ↓ | Skate proxy ↓ | Error cm ↓ | Latency s/sample ↓ |
|---|---:|---:|---:|---:|---:|
| Minimal | N/A* | N/A* | 34.44% | 44.02 | 0.1652 |
| Hybrid | N/A* | N/A* | 36.73% | 62.19 | 0.3710 |

## 五、Table 5：80 帧滑窗统一协议 proxy

| Method | R-Precision ↑ | FID ↓ | Skate proxy ↓ | Error cm ↓ | Latency s/sample ↓ |
|---|---:|---:|---:|---:|---:|
| Minimal (in) | N/A* | N/A* | 27.69% | 41.84 | 0.2513 |
| Hybrid (in) | N/A* | N/A* | 26.97% | 50.09 | 0.6584 |
| Minimal (out) | N/A* | N/A* | 28.59% | 69.35 | 0.2697 |
| Hybrid (out) | N/A* | N/A* | 28.55% | 56.10 | 0.6506 |

窗口边界最大 root speed：Table 5 in 中 Minimal/Hybrid 为 `4.380/4.217 m/s`；Table 5 out 中为 `4.047/2.772 m/s`。

## 六、对架构预期的判断

当前结果只支持“局部长时序互补”，不支持“Hybrid 已显著优于 ARDY”：

- Table 4 中 Hybrid error `62.19 cm`，高于 Minimal 的 `44.02 cm`，且延迟约为 `2.25x`。
- Table 5 in 中 Hybrid error `50.09 cm`，高于 Minimal 的 `41.84 cm`，但 skate proxy 略低。
- Table 5 out 中 Hybrid error `56.10 cm`，低于 Minimal 的 `69.35 cm`，下降约 `19%`；边界最大 root speed 也由 `4.047` 降至 `2.772 m/s`。

因此，Mamba recurrent memory 与 attention sliding-window/sink 对 out-of-horizon 的误差累积和窗口边界有初步帮助，但尚未形成全面优势。DiT 作为去噪 mixer 也不必然仅凭架构名称优于 ARDY 的两阶段 Transformer：参数规模、训练目标、训练预算、表示转换和 rollout 训练方式都会影响结果。当前 50k 还只是预算，不能据此判定模型已经收敛。

此外，当前 sink 是 attention sink，不能直接等同于已经验证有效的“沉降压缩记忆”；必须通过消融和更长训练验证 memory 容量、状态更新和信息保留是否真正带来收益。

## 七、严格边界

`N/A*` 不是零值，而是生成结果尚未完成 HumanML3D evaluator 所需的统一表示转换。Dataset calibration 只用于校准 evaluator，不能作为最终 Table 4/5 的模型行。`Released ARDY native checkpoint` 也不能作为 HumanML3D-trained ARDY baseline；因此本报告没有用它和当前模型做正式架构结论。

## 八、下一步

1. 使用相同 HumanML3D 训练集、测试集、训练配置、步数和生成协议训练 HumanML3D ARDY baseline。
2. 完成并验证 CoreSkeleton27/330-D -> HumanML3D 22-joint/263-D 转换，在同一测试样本上补齐 R-Precision/FID。
3. 根据 50k、100k 候选 checkpoint 的 loss、长时序边界和 Table 4/5 结果决定是否延长到 150k，而不是预先假设 50k 已收敛。
4. 做 Minimal、Mamba-only、Hybrid-no-sink、Hybrid-no-recurrent-memory 和完整 Hybrid 消融；baseline 冻结后再单独报告 kernel 前后质量、延迟、显存和 active parameters。

