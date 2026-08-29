# Motion Mamba 2.0 Full Training Snapshot

This repository contains the implementation and experiment-report snapshot
for the HumanML3D and BONES-SEED full-training work through 2026-08-29.

## Contents

- `MotionMamba2.0/`: ARDY integration, Minimal Motion DiT, Attention/Mamba
  hybrid implementation, training/evaluation code, tests, and reports.
- `runtime/`: reusable launch scripts and runtime configuration. Transient
  files under `runtime/tmp/` are intentionally excluded.

The source working tree was based on `BaselineAndC1`, including its local
implementation and report changes. This snapshot is published only on the
independent `full-training-20260829` branch in this repository. It does not
update or overwrite the source repository's `BaselineAndC1` branch.

## Artifact Policy

Source code, configuration, documentation, tests, and small required skeleton
assets are included. Model checkpoints, datasets, generated results, logs,
Python caches, evaluator caches, and temporary runtime files are excluded.
Training checkpoints and metrics remain in the external result storage paths
recorded by the experiment reports.

The current training and evaluation audits are:

- `MotionMamba2.0/mamba-transformer-keneral/docs/BONES_SEED_EXPOSURE_GAP_TRAINING_AUDIT_20260829.md`
- `MotionMamba2.0/mamba-transformer-keneral/docs/BONES_SEED_V9_V2_EXACT_TEST_PROTOCOL_COMPARISON_20260829.md`
- `MotionMamba2.0/mamba-transformer-keneral/docs/DATA4_CHECKPOINT_RETENTION_AUDIT_20260829.md`
