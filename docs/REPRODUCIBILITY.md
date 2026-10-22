# Reproduction contract

This source release separates the executed model, data pipeline, phase-specific training implementations, evaluator implementation and paired-text comparison harness. It does not claim that a clean-machine full100k training run has already been repeated from the renamed tree. CPU import/syntax checks, strict full-model loading, and numerical rename checks have narrower scopes than full training or motion quality.

## Required external assets

1. Authorized BONES-SEED, HiPHI and HumanML3D motion and original captions, retargeted to the recorded G1 skeleton,25fps and physical414 representation. Preserve source-family identity, project split, accepted samples and exclusion lists. `data/preprocessing` retains the original QA implementation and rejected-stage guards; a historical failed gate is not an accepted release.
2. The accepted JSONL training index and its frozen exclusion JSON. Each feature path must resolve to the original accepted representation, with matching normalizer and codec fingerprints. Moving data requires an explicit `legacy_path` to local `path` binding in the private artifact map, not a change to sample identity.
3. Pinned upstream codec/frontends and LLM2Vec text assets referenced by `motion_mamba2.models.registry`. Pretrained frontend/projection reuse is part of foundation initialization; do not substitute an all-random frontend. Text caches contain4096 values per caption and must use the same encoder/tokenizer.
4. Complete parent checkpoints, teacher pools and pool receipts for continuation. The displayed110k/117k/122k checkpoint registrations are separate branches; inspect their metadata. Do not infer parents from a label or stage filename.
5. For paired evaluation: original initial-request bundle; accepted retargeted manifests; domain-specific released distribution/retrieval heads, two correct head normalizers, and calibration/asset receipts. No domain may borrow the other domain's statistics. External comparison architectures additionally require their own upstream repositories and licensed assets; they are comparison prerequisites, not Motion Mamba 2.0 modules.

`configs/artifacts.example.json` lists named bindings. Repository source bindings are automatic; external bindings are deliberately unresolved until assets are supplied. Create a private map:

```bash
python tools/configure_workspace.py --workspace "$PWD/workspace" \
  --bind data_index_jsonl_9966c1bc60=/authorized/data_index.jsonl \
  --bind data_exclusions_json_cea79e86ab=/authorized/DATA_EXCLUSIONS.json \
  --bind stats_a5227494f3=/authorized/frozen_stats
export MM2_WORKSPACE="$PWD/workspace"
export MM2_ARTIFACT_MAP="$PWD/artifacts.local.json"
```

Further bindings are recipe-specific. The private server organization includes the complete original binding map and source-to-release audit; that map is excluded from Git because it exposes internal directories. The public map never silently accesses an author's server. Run `tools/doctor.py` to inspect missing prerequisites. Installed package metadata and the actually imported isolated Triton version must be distinguished.

## Foundation execution

- `training/create_initial.py --output .../checkpoint0.pt` reconstructs compatible upstream frontend reuse and new native sequence modules with seed123, saving empty Adam. It performs zero training updates. Use CUDA for original-device initialization; CPU and CUDA random constructors are not interchangeable for exact tensor identity.
- Bind `checkpoint0_pt_17f1a13e6b` to that complete zero-step save.
- `training/prepare.py --phase foundation32 --output .../train --encode` prepares the indexed shared-layout sampler and missing text embeddings. It retains the original max100k sampler plan; stop the32 phase at75k.
- `training/launch.py --phase foundation32 ... --until 75000` uses the actual early32 implementation. Four-rank effective batch32 is rank8 without accumulation; two-rank execution would require its original admitted accumulation and a separate transition gate.
- Bind `checkpoint75000_pt_48db5cf600` to the complete75k save. `training/prepare.py --phase foundation512 ...` uses the actual lazy512 plan, not the32 sampler. The512 entry uses its separate executed engine/batching revision. Preserve saved rank RNG, full FP32-master Adam and100k scheduler across the boundary.
- The512 stage runs25k additional updates. Total exposures:75000×32+25000×512=15200000.

All training is explicit. No publication script queues or restarts a live experiment. The default budget helper intentionally has small protective caps when no `LIMITS.json` is supplied; install the recorded limits from `docs/experiment_metadata` into your new run budget directory before a full run. Never expand a live run's budget silently.

Use `tools/verify_checkpoint.py` with expected actual step and SHA. A checkpoint filename alone is insufficient. For legacy full-state resume, `tools/migrate_checkpoint.py` requires exact recipe equality after explicit path/namespace migration and checks tensor/Adam/RNG hashes before and after serialization. It never transfers an old GPU-source admission to renamed code, never changes tensor keys or values, and rejects any other recipe change. Changing batch size, precision, kernels or losses requires new equivalence admission.

## Evaluation order

Each domain has its own `build_catalog.py`, `prepare_protocol.py`, `audit_real.py`, `text_features.py`, `execute.py`, `score.py` and `audit_outputs.py` under `evaluation/paired_text`. These operate on a **new writable protocol directory** containing copies of the scripts, the corresponding uppercase `PLAN.json`, checkpoint registry, authorized parent request assets and evaluator bindings. Never run a builder over the source directory or a previously frozen experiment.

1. Supply the original request bundle and frozen dataset/evaluator assets. Build the GT-only catalog and paired detailed-text protocol. Preserve the original family split and DEV/TEST separation.
2. Run real-reference calibration and independent `audit_real.py`. All24 phase/head/pool checks must pass the fixed thresholds before generation. A CPU calibration is not generated-model performance.
3. Materialize current detailed-text LLM2Vec features. Freeze all protocol inputs, selected complete checkpoints and actual current source hashes. The archived experiment's source hashes do not admit reorganized code: generate a new binding and rerun all affected gates. Do not edit a hash merely to make a changed experiment pass.
4. Run `execute.py --model MODEL --task TASK` for each of the14×4 conditions, on explicitly allocated GPUs. Each worker retains the four DEV wrist/ankle checks, full-output/state/RNG equivalence, two full warmups and32 paired B1 timings, then384 formal rollouts. `shared/execute.py`, comparison loaders and masked-graph candidate source are included; external model dependencies and immutable input bundles must be bound separately.
5. Score using the matching domain's own heads/statistics and same-model/same-window/same-noise32-generated candidate pools. At most three2-thread CPU workers were used. Keep R1/R2/R3, per-window physics, terminal CE and two independent L2-FID calculations.
6. Run `audit_outputs.py`; inspect every geometry/input/text/noise/checkpoint binding, terminal FK/CE, all paired ranks and covariance-FID. A report or ZIP is complete only after the independent audit and local numerical/visual review.

No published score is replaced by a real-reference calibration score. R3ref from the old short-command task remains a historical diagnostic, not the new detailed paired-text metric. A detailed-text rerun changes the task and its FID reference; differences from the old task do not prove a scoring error or a causal memory improvement.

## Verification scope

The release audit records the exact checks run and their outcomes in `docs/release_validation.json`. Syntax/import checks and strict checkpoint loading do not establish a fresh100k reproduction, full GPU output equivalence or comparison-model quality. The original experimental checkpoints/receipts remain preserved externally; failed diagnostic baselines are not relabeled successful. Any unfinished baseline repair remains separate from these frozen results.
