#!/bin/bash
# Reviewer-requested effective-parameter control.
# Calibrated MLP h=2: 21 base + 2 affine = 23 trainable parameters.
# Calibrated PQC: 22 output-relevant circuit + 2 affine = 24.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v9

LOG=logs/v9_effective_match.log
: > "$LOG"
echo "[$(date +%H:%M:%S)] v9 effective-match control starting" >> "$LOG"

run_one() {
  local task=$1 seed=$2
  local out="results/v9/hybrid_v9_${task}_depthwise_mlp_effmatch_h2_q8_l2_s${seed}.json"
  if [ -f "$out" ]; then
    echo "[$(date +%H:%M:%S)] skip ${task}_effmatch_h2_s${seed} (already complete)" >> "$LOG"
    return 0
  fi
  .venv/bin/python src/train_hybrid_v5.py \
    --task "$task" --reducer depthwise --head mlp --mlp-hidden 2 \
    --n-qubits 8 --n-layers 2 --seed "$seed" --lr 0.002 \
    --pretrain-epochs 20 --epochs 25 --patience 6 --n-val 200 --n-train 2000 \
    --out-subdir v9 --run-tag effmatch_h2 \
    > "logs/v9_${task}_effmatch_h2_s${seed}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done ${task}_effmatch_h2_s${seed} rc=$?" >> "$LOG"
}

for task in mrpc paws qqp; do
  for seed in 0 1 2 3 4 5 6 7; do
    while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge 6 ]; do sleep 1; done
    run_one "$task" "$seed" &
  done
done
wait
echo "[$(date +%H:%M:%S)] V9 EFFECTIVE-MATCH COMPLETE" >> "$LOG"
