# Motion Mamba 2.0 Full Training Snapshot

This repository contains the implementation and experiment-report snapshot
for the HumanML3D full-training work completed on 2026-08-20.

## Contents

- `MotionMamba2.0/`: ARDY integration, Minimal Motion DiT, Attention/Mamba
  hybrid implementation, training/evaluation code, tests, and reports.
- `runtime/`: reusable launch scripts and runtime configuration. Transient
  files under `runtime/tmp/` are intentionally excluded.

The source working tree was based on `BaselineAndC1`, including its local
implementation and report changes. This snapshot is published only on the
independent `full-training-20260820` branch in this repository. It does not
update or overwrite the source repository's `BaselineAndC1` branch.

## Artifact Policy

Source code, configuration, documentation, tests, and small required skeleton
assets are included. Model checkpoints, datasets, generated results, logs,
Python caches, evaluator caches, and temporary runtime files are excluded.
Training checkpoints and metrics remain in the external result storage paths
recorded by the experiment reports.

The main completed-training report is:

`MotionMamba2.0/mamba-transformer-keneral/docs/HUMANML3D_CACHEFIX_FULL_TRAIN_REPORT_20260820.md`

