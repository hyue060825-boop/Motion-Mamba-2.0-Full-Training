#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_CACHE=true

ROOT=/data3/zts
REPO="$ROOT/MotionMamba2.0/mamba-transformer-keneral"
PYTHON="$ROOT/conda_envs/ardy_env/bin/torchrun"
OUTPUT="$ROOT/results/humanml3d_ardy_full_v5r_minimal_20260819"
LOG="$ROOT/results/mm2_full_v5r_min.log"

"$PYTHON" --standalone --nproc-per-node=4 \
  "$REPO/examples/train_humanml3d_ardy_ddp.py" \
  --manifest "$ROOT/results/humanml3d_ardy_train/manifest.jsonl" \
  --dev-manifest "$ROOT/results/humanml3d_ardy_val/manifest.jsonl" \
  --stats-dir "$ROOT/results/humanml3d_ardy_train" \
  --text-root "$ROOT/datasets/HumanML3D/HumanML3D/texts" \
  --text-cache "$ROOT/results/humanml3d_text_cache_llm2vec" \
  --output-dir "$OUTPUT" \
  --mode minimal \
  --train-scope denoiser \
  --steps 50000 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --seed 123 \
  --log-every 50 \
  --eval-every 1000 \
  --save-every 5000 \
  --keep-last-checkpoints 1 \
  --frames 80 \
  --history-probability 0.5 \
  --text-dropout 0.1 \
  --constraint-dropout 0.1 \
  --sequential-cache-probability 0 \
  --eval-batches 4 \
  --boundary-loss-weight 10 \
  --root-velocity-loss-weight 50 \
  --constraint-loss-weight 1 \
  --fk-constraint-loss-weight 1 \
  --foot-skate-loss-weight 50 \
  2>&1 | tee "$LOG"
