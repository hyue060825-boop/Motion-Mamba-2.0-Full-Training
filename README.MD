# Motion Mamba 2.0

G1 motion modeling with hybrid explicit/latent motion features, a Transformer–Mamba sequence model, full-state training checkpoints, and detailed paired-text continuous-control evaluation.

This is the organized source release of the **executed experiment code**. Public packages, training commands and model files use the Motion Mamba 2.0 name. Upstream attribution and legacy checkpoint-format compatibility are retained in the notices and compatibility module; external models are not presented as our original work.

## Repository layout

```text
src/motion_mamba2/
  models/                 model architecture, diffusion and frozen codec interfaces
  adapters/               attention/Mamba integration and accepted-history state
  kernels/                sequence, attention and recurrent-state implementations
  representation/         physical/hybrid motion representation and normalization
  skeleton/               G1 skeleton and forward kinematics
  runtime/                precision, decoding, masking and geometry contracts
training/
  motion_training/        losses, data sampling, optimizer, checkpointing, DDP
  recipes/                executed foundation and continuation recipe sources
  revisions/              separate actual32 and512-stage engine implementations
data/
  motion_data/            dataset readers and conversion adapters
  preprocessing/          retargeting/QA implementation; historical gates remain enforced
  manifest.json           data contract; raw motion is not committed
evaluation/
  native_eval/            independently trained evaluator v2 implementation
  paired_text/            HiPHI and HumanML3D detailed-text protocol/scoring/audits
configs/                  actual training recipes and portable artifact templates
checkpoints/manifest.json selected self-trained checkpoint identities and SHA-256
third_party/              licenses and exact Mamba source snapshot used by the runtime
tools/                    environment inspection and checkpoint verification
tests/                    release import, path and numerical checks
```

Raw datasets, model weights, generated motion, local logs, credentials, installed environments and temporary caches are **not Git source files**. Server-side organization keeps these separately under `data/`, `checkpoints/`, `private_archive/` and `audit/` in the external workspace. No private absolute server path is needed in the public source.

## Environment

Use Python 3.11 (the recorded runtime) and a separate environment. The recorded experiment versions are in `requirements.lock.json`; use that record rather than substituting the newest library versions. In particular, the executed model used PyTorch 2.6.0+cu124 and an isolated Triton **3.5.1** import path. The environment's default Triton distribution metadata can report 3.2.0; these are different and must not be confused.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the recorded PyTorch/CUDA and Python dependencies first.
python -m pip install --no-deps -e .
export MM2_WORKSPACE="$PWD/workspace"
export MM2_ARTIFACT_MAP="$PWD/artifacts.local.json"
python tools/doctor.py
```

The vendored Mamba source includes Mamba-3 and is individually SHA-bound in `third_party/mamba/SNAPSHOT_MANIFEST.json`. The source archive had no Git metadata; an upstream commit is deliberately not invented. CUDA extension builds require a compatible CUDA compiler/toolkit. See `third_party/README.md` for isolated installation. Do not replace the live server's global PyTorch, Mamba or Triton environment.

## Data and artifact preparation

The training mixture is BONES-SEED/HiPHI/HumanML3D at **0.70/0.15/0.15**, 25 fps, physical414 and hybrid148. Dataset access and redistribution rights remain with their respective providers. Obtain the authorized original datasets and the frozen shared codec/text-model assets separately; no dataset, third-party weight, or non-existent download link is bundled here.

Copy `configs/artifacts.example.json` to the ignored `artifacts.local.json`, and bind each required artifact to its local path. The keys distinguish data indices, exclusion lists, normalization, cached text, teacher pools and checkpoint parents. Bind only the artifacts required for your chosen recipe. `artifact://...` values in configurations resolve through this explicit map. Missing prerequisites are errors; the code does not silently borrow files from an author's machine.

The required training index is JSONL with `dataset`, `source_family_id`, `project_split`, `feature_path`, frame counts, original captions and their time spans. Preserve original family/split identity and all frozen exclusions. Use `data/motion_data` and `data/preprocessing` to construct G1 representations. Historical rejected retargeting/calibration stages are not automatically declared valid: the executed training used its explicitly accepted index and exclusions. Do not bypass a failed data gate to regenerate a different training set.

The frozen shared statistics must match the checkpoint's representation fingerprint. They are not the retrieval-head statistics. Detailed online text uses the original LLM2Vec4096 encoding; cached embeddings and newly encoded text must use the same encoder and tokenizer.

```bash
python tools/doctor.py --strict-assets --config configs/training/foundation32.json
python training/prepare.py --phase foundation32 --output "$MM2_WORKSPACE/foundation32/train" --encode
```

Use `training/create_initial.py` to reconstruct the initial state from the exported initialization metadata, the frozen statistics, and the pinned upstream codec/frontends. Native sequence modules are initialized with seed123. The original foundation initialization reused compatible upstream frontend/projection tensors; it was **not** an all-parameters-from-scratch initialization. The codec is frozen. The original initialization receipt is included in the experiment metadata.

## Foundation training: 100k

The recipe has **75k updates at global32 plus 25k at global512**, totaling **15.2 million sample exposures**. Preserve the four original losses, H/G/F sampling, text/constraint dropout, FP32-master Adam, clipping and warmup-cosine schedule. A filename such as `checkpoint25000.pt` may be a stage counter: inspect the saved actual step, scheduler and parent lineage.

First inspect the resolved recipe without starting training:

```bash
python training/launch.py --phase foundation32 --output "$MM2_WORKSPACE/foundation32/train" --until 75000 --dry-run
```

Then launch on explicitly allocated GPUs. The examples use four ranks; preserve the recorded per-rank microbatch/accumulation and do not change global batch to fit memory without a new numerical gate.

```bash
torchrun --standalone --nproc_per_node=4 training/launch.py \
  --phase foundation32 --output "$MM2_WORKSPACE/foundation32/train" --until 75000
# Bind foundation512's parent to the complete75k checkpoint first.
torchrun --standalone --nproc_per_node=4 training/launch.py \
  --phase foundation512 --output "$MM2_WORKSPACE/foundation512/train" --until 25000
```

A resumed run must retain the model, FP32 Adam masters/moments, scheduler, sampler cursor and all rank RNG states. Metadata/path migration is explicit and never rewrites tensor values. The legacy recipe sources include their original equivalence gates; a renamed source tree does not make an old source-hash admission valid for new acceleration. Reference execution is the default. Re-admit any acceleration before using it.

## Continuation checkpoints and lineage

The four model rows are separate verified checkpoints. **Do not assume the displayed110k,117k and122k rows form one linear chain.** Their original parent registrations/configurations are preserved under `docs/experiment_metadata/`.

- 100k: the 75k+25k foundation schedule above.
- Displayed110k: the separately registered100k continuation branch.
- 117k: the historical112k foundation parent plus its5k interaction stage.
- Regular122k: the ordinary continuation of that117k branch; not the later long-distillation branch.

The interaction/continuation recipes use recorded GT/teacher mixtures, teacher pools, refresh points and current-history construction. They are not equivalent to adding arbitrary extra GT updates. Rebuild the recorded pools with the supplied collectors and retain their admission receipts before invoking the continuation worker. Disabled experimental content-memory, delta/state-summary modules in the source are **not** evidence that these were trained or active in a released checkpoint.

## Checkpoints

`checkpoints/manifest.json` lists **self-trained experimental** saves used in the comparisons, with actual step and SHA-256. Failed-training diagnostic baselines remain labeled diagnostic; official pretrained comparison weights are not mixed into this collection. Full retained experiment saves are organized in the external server workspace, not duplicated into Git.

```bash
CUDA_VISIBLE_DEVICES='' python tools/verify_checkpoint.py /path/to/checkpoint.pt \
  --expected-step 100000 --expected-sha256 YOUR_RECORDED_SHA256
```

The verifier checks saved state integrity; it does not certify motion quality. Large checkpoint binaries are not hosted by this Git commit. Access to those weights, or re-training from the supplied recipe and authorized assets, is required for checkpoint-based reproduction.

## Detailed paired-text continuous evaluation

Use the **v4 paired-text** implementation in `evaluation/paired_text/`. Build the protocol from the authorized domain-specific test data, run real-reference calibration first, freeze inputs/checkpoints, then generate and score. Use each domain's own evaluator heads and normalizers. Do not substitute HiPHI heads/statistics for HumanML3D or vice versa.

The fixed test has128 initial families ×3 noises per duration, initial24 frames,1/4/9/19 accepted52-frame windows (2.08/8.32/18.72/39.52s),25fps,10 denoising steps and CFG2/2. Only current detailed text is supplied. Root/End CE are terminal errors, not intermediate-path ADE. No paired future GT motion is given to the generator. R1/R2/R3 use paired detailed text with32 generated same-window candidates, two heads, three fixed pools and three noises. FID uses individually L2-normalized256D short-window features in FP64; its window mean is not whole40s FID.

Actual distinct-caption query counts by phase are **HiPHI61/110/61/42** and **HumanML3D89/128/89/128**. Initialization actors are three for HiPHI and61 for HumanML3D. Coarse root-state agreement does not measure every detail of an instruction. Fixed-capacity baselines can lose distant targets. The old short-command R3ref protocol is not the main semantic metric here. A changed text/reference task is not a single-factor improvement over the old task, and these evaluations do not establish episodic-memory causality.

Keep independent geometry/FK, CE, retrieval-pool, covariance-FID, input/asset identity and real B1 latency audits. Latency is measured separately from training throughput and text encoding. See `docs/REPRODUCIBILITY.md` for verified release checks and remaining external prerequisites.

## License and attribution

Project contributions are released under Apache-2.0. Individual upstream files retain their original copyright/SPDX headers and licenses. See `THIRD_PARTY_NOTICES.md`. Public renaming does not remove upstream attribution or alter third-party weight/dataset licenses.

## 中文说明

项目统一名称为 **Motion Mamba 2.0**。模型架构在 `src/motion_mamba2`，训练在 `training`，数据处理在 `data`，评测在 `evaluation`。原始数据和实验权重在外部工作区集中管理，不上传到源码Git历史。先按上面的依赖、数据/统计量/文本模型绑定与训练阶段顺序准备环境；详细步骤和验证范围见 `docs/REPRODUCIBILITY.md`。

已验证整理后代码能严格载入原100k模型，并在CUDA上用原seed逐项重建422个初始化张量；这不等于重新跑完100k训练。继续训练必须保留完整Adam、调度器、采样位置和各rank RNG。第三方来源、官方模型标识和旧权重格式兼容信息保留；失败诊断模型和未完成修复不会包装为成功结果。
