#!/usr/bin/env bash
set -euo pipefail

export LOCAL_CACHE=true

ROOT=/data3/zts
REPO="$ROOT/MotionMamba2.0/mamba-transformer-keneral"
PYTHON="$ROOT/conda_envs/ardy_env/bin/torchrun"
OUTPUT="$ROOT/results/humanml3d_ardy_full_v5r_hybrid_20260819"
LOG="$ROOT/results/mm2_full_v5r_hyb.log"
LAUNCH_LOG="$ROOT/results/mm2_full_v5r_hyb_launcher.log"
MINIMAL_SESSION=mm2_full_v5r_minimal

gpus_free() {
  local gpu
  for gpu in "$@"; do
    if nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+$'; then
      return 1
    fi
  done
}

while true; do
  if gpus_free 4 5 6 7; then
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    NPROC=4
    BATCH_SIZE=8
    break
  fi
  if gpus_free 4 6; then
    export CUDA_VISIBLE_DEVICES=4,6
    NPROC=2
    BATCH_SIZE=16
    break
  fi
  if ! tmux has-session -t "$MINIMAL_SESSION" 2>/dev/null && gpus_free 0 1 2 3; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    NPROC=4
    BATCH_SIZE=8
    break
  fi
  printf '%s waiting for four free GPUs\n' "$(date --iso-8601=seconds)" >> "$LAUNCH_LOG"
  sleep 30
done

printf '%s launching Hybrid on physical GPUs %s (nproc=%s, per-rank-batch=%s)\n' \
  "$(date --iso-8601=seconds)" "$CUDA_VISIBLE_DEVICES" "$NPROC" "$BATCH_SIZE" >> "$LAUNCH_LOG"

"$PYTHON" --standalone --nproc-per-node="$NPROC" \
  "$REPO/examples/train_humanml3d_ardy_ddp.py" \
  --manifest "$ROOT/results/humanml3d_ardy_train/manifest.jsonl" \
  --dev-manifest "$ROOT/results/humanml3d_ardy_val/manifest.jsonl" \
  --stats-dir "$ROOT/results/humanml3d_ardy_train" \
  --text-root "$ROOT/datasets/HumanML3D/HumanML3D/texts" \
  --text-cache "$ROOT/results/humanml3d_text_cache_llm2vec" \
  --output-dir "$OUTPUT" \
  --mode hybrid \
  --train-scope denoiser \
  --steps 50000 \
  --batch-size "$BATCH_SIZE" \
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
