#!/bin/bash
# v5 — calibration-lever experiment (see PREREGISTRATION_v5.md)
# v4 + 2-param affine calibration on BOTH heads.
# DEPTHWISE FIRST (headline test where v4 gap was significant),
# then LINEAR (control where v4 gap was already absent).
#   8 seeds x {depthwise,linear} x {pqc,mlp} x {mrpc,paws} = 64 runs
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v5

LOG=logs/hybrid_v5_multiseed.log
echo "[$(date +%H:%M:%S)] v5 calibration-lever sweep starting" | tee "$LOG"
echo "[$(date +%H:%M:%S)] depthwise (headline) first, then linear (control)" | tee -a "$LOG"

for reducer in depthwise linear; do
  for task in mrpc paws; do
    if [ "$task" = "mrpc" ]; then N_TRAIN="--n-train 2000"; else N_TRAIN="--n-train 2000"; fi
    for head in pqc mlp; do
      for s in 0 1 2 3 4 5 6 7; do
        TAG="${reducer}_${task}_${head}_s${s}"
        echo "[$(date +%H:%M:%S)] start $TAG" | tee -a "$LOG"
        .venv/bin/python src/train_hybrid_v5.py \
          --task "$task" --reducer "$reducer" --head "$head" \
          --n-qubits 8 --n-layers 2 --seed "$s" \
          --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
          --n-val 200 $N_TRAIN \
          > "logs/hybrid_v5_${TAG}.log" 2>&1
        echo "[$(date +%H:%M:%S)] finish $TAG rc=$?" | tee -a "$LOG"
      done
    done
  done
done
echo "[$(date +%H:%M:%S)] DONE v5 sweep" | tee -a "$LOG"
