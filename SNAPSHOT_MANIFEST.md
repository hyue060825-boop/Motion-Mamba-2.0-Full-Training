# Snapshot Manifest

- Snapshot date: 2026-08-29
- Target branch: `full-training-20260829`
- Included top-level directories: `MotionMamba2.0/`, `runtime/`
- Source-code branch at snapshot time: `BaselineAndC1`
- Source working-tree changes: included
- Source `.git` metadata: excluded
- Regression verification: `102 passed, 6 skipped`
- Shell launcher syntax verification: passed

## Excluded Generated Content

- `runtime/tmp/`
- Python, pytest, mypy, ruff, and generic cache directories
- logs and local editor files
- checkpoint and result directories
- model/checkpoint formats (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.onnx`)
- datasets and external result storage, which were not located inside the two
  uploaded source directories

No file in this snapshot exceeds GitHub's 100 MB per-file limit.
