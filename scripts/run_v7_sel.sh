#!/bin/bash
# v7 E2 -- StronglyEntanglingLayers head (second ansatz family), generality
# of the bias-role asymmetry. MRPC depthwise, q=8, L=1, seeds 0-7.
#   uncal: --head sel --freeze-calib --run-tag uncal   (v4-protocol, no calibration)
#   cal:   --head sel                --run-tag cal     (v5-protocol, learnable affine)
# 2 parallel blocks (uncal | cal) -> concurrency == 2.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v7

run_sel() { # run-tag [extra...]
  local tag=$1; shift
  for s in 0 1 2 3 4 5 6 7; do
    .venv/bin/python src/train_hybrid_v5.py --task mrpc --reducer depthwise --head sel \
      --n-qubits 8 --n-layers 1 --seed "$s" \
      --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 --n-val 200 --n-train 2000 \
      --out-subdir v7 --run-tag "$tag" "$@" > "logs/v7_sel_${tag}_s${s}.log" 2>&1
    echo "[$(date +%H:%M:%S)] done sel_${tag}_s${s} rc=$?" >> logs/v7_sel.log
  done
  echo "[$(date +%H:%M:%S)] BLOCK ${tag} COMPLETE" >> logs/v7_sel.log
}

echo "[$(date +%H:%M:%S)] v7 E2 SEL sweep starting (uncal|cal parallel)" > logs/v7_sel.log
run_sel uncal --freeze-calib &
run_sel cal &
wait
echo "[$(date +%H:%M:%S)] ALL BLOCKS COMPLETE" >> logs/v7_sel.log
