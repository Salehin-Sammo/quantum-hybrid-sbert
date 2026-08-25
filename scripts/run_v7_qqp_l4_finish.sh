#!/bin/bash
# Finish E1's QQP L=4 cell: seeds 3-7 (seeds 0-2 already on disk).
#
# Seeds 0-2 were produced by the pre-patch trainer. That patch (evaluate the
# test set only on new-best-val epochs) was verified bit-identical -- all 9
# metrics to 9 dp, predictions identical, max |logit diff| = 0.00e+00, full
# train-loss trajectory unchanged -- so those three runs need no redo.
#
# All 5 run concurrently: 14 cores, nothing else scheduled. OMP_NUM_THREADS is
# capped so 5 processes don't oversubscribe and re-create the starvation that
# stretched the first pass to 162 min/run.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v7
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run() {
  local s=$1
  local TAG="v7_qqp_depthwise_pqc_fourierL4_s${s}"
  .venv/bin/python src/train_hybrid_v5.py \
    --task qqp --reducer depthwise --head pqc \
    --n-qubits 8 --n-layers 4 --seed "$s" \
    --out-subdir v7 --run-tag fourierL4 \
    --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
    --n-val 200 --n-train 2000 > "logs/${TAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $TAG rc=$?" >> logs/v7_fourier.log
}

echo "[$(date +%H:%M:%S)] QQP L4 finish: seeds 3-7, 5-way parallel, patched trainer" >> logs/v7_fourier.log
for s in 3 4 5 6 7; do run "$s" & done
wait
echo "[$(date +%H:%M:%S)] BLOCK B (PQC L4) COMPLETE" >> logs/v7_fourier.log
