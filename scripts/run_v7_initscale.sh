#!/bin/bash
# v7 E4 -- zero-parameter remedy (see PREREGISTRATION_v7_mechanism.md E4).
#
# Uncalibrated PQC (--freeze-calib pins the affine at identity => EXACTLY 32
# params, no parameters added vs v4) but with --init-scale 1.0, which centres
# the <Z_0> readout (+0.92 -> +0.005 at init; scripts/readout_offset_v7.py).
#
# Baselines already on disk -- do NOT re-run:
#   v4 uncalibrated PQC (init 0.1): results/v4/hybrid_v4_{task}_depthwise_pqc_q8_l2_s*.json
#   v4 MLP:                          results/v4/hybrid_v4_{task}_depthwise_mlp_q8_l2_s*.json
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v7

run() { # task seed
  local task=$1 s=$2
  local TAG="v7_${task}_pqc_init1p0_uncal_s${s}"
  .venv/bin/python src/train_hybrid_v5.py \
    --task "$task" --reducer depthwise --head pqc \
    --n-qubits 8 --n-layers 2 --seed "$s" \
    --init-scale 1.0 --freeze-calib \
    --out-subdir v7 --run-tag init1p0uncal \
    --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
    --n-val 200 --n-train 2000 > "logs/gaps_${TAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $TAG rc=$?" >> logs/v7_initscale.log
}

echo "[$(date +%H:%M:%S)] v7 E4 init-scale sweep starting (16 runs, 2 blocks)" >> logs/v7_initscale.log
# MRPC first (fast, and it is the task P4 names); PAWS second, as its own block.
( for s in 0 1 2 3 4 5 6 7; do run mrpc "$s"; done; echo "[$(date +%H:%M:%S)] MRPC BLOCK COMPLETE" >> logs/v7_initscale.log ) &
( for s in 0 1 2 3 4 5 6 7; do run paws "$s"; done; echo "[$(date +%H:%M:%S)] PAWS BLOCK COMPLETE" >> logs/v7_initscale.log ) &
wait
echo "[$(date +%H:%M:%S)] E4 ALL BLOCKS COMPLETE" >> logs/v7_initscale.log
