#!/bin/bash
# v4 — class-weighted BCE + macro-F1 early stopping + 2-layer MLP probe in Phase 1.
# Properly handles class imbalance so we measure real discrimination, not collapse.
#   8 seeds × {depthwise, linear} × {pqc, mlp} × {mrpc, paws} = 64 runs
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v4

LOG=logs/hybrid_v4_multiseed.log
echo "[$(date +%H:%M:%S)] v4 sweep starting (class-weighted BCE)" | tee "$LOG"

for task in mrpc paws; do
  if [ "$task" = "mrpc" ]; then
    N_TRAIN="--n-train 2000"
  else
    N_TRAIN="--n-train 2000"
  fi
  for reducer in depthwise linear; do
    for head in pqc mlp; do
      for s in 0 1 2 3 4 5 6 7; do
        TAG="${task}_${reducer}_${head}_s${s}"
        echo "[$(date +%H:%M:%S)] start $TAG" | tee -a "$LOG"
        .venv/bin/python src/train_hybrid_v4.py \
          --task "$task" \
          --reducer "$reducer" \
          --head "$head" \
          --n-qubits 8 --n-layers 2 \
          --seed "$s" \
          --lr 0.002 \
          --pretrain-epochs 20 \
          --epochs 25 \
          --patience 6 \
          --n-val 200 \
          $N_TRAIN \
          > "logs/hybrid_v4_${TAG}.log" 2>&1
        rc=$?
        echo "[$(date +%H:%M:%S)] finish $TAG rc=$rc" | tee -a "$LOG"
      done
    done
  done
done

echo "[$(date +%H:%M:%S)] DONE v4 sweep" | tee -a "$LOG"
