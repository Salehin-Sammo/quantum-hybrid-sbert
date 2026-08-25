#!/bin/bash
# v7 E1 -- reuploading-depth (Fourier) sweep. See PREREGISTRATION_v7_mechanism.md §E1.
# v5 calibrated protocol, depthwise reducer, q=8, seeds 0-7, tasks qqp+paws.
# 3 parallel blocks, one per run-tag config, PAWS run before QQP within each
# block so partial (fast) results land early and the slow QQP tail is spread
# across all 3 blocks concurrently instead of piling up in one:
#   A: PQC L=1  (18 params incl. calib)  --run-tag fourierL1     16 runs
#   B: PQC L=4  (66 params incl. calib)  --run-tag fourierL4     16 runs
#   C: MLP h=6  (63 params incl. calib)  --run-tag fourierMLPh6  16 runs
# L=2 PQC / hidden=3 MLP baselines already exist in results/v5/ -- not rerun.
# All output: results/v7/, filenames hybrid_v7_<task>_depthwise_<head>_<tag>_q8_l<L>_s<seed>.json
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v7

run7() { # task head tag seed [extra...]
  local task=$1 head=$2 tag=$3 s=$4; shift 4
  local LOGTAG="v7_${task}_depthwise_${head}_${tag}_s${s}"
  .venv/bin/python src/train_hybrid_v5.py --task "$task" --reducer depthwise --head "$head" \
    --seed "$s" --n-qubits 8 --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
    --n-val 200 --n-train 2000 --out-subdir v7 --run-tag "$tag" "$@" \
    > "logs/${LOGTAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $LOGTAG rc=$?" >> logs/v7_fourier.log
}

blockA() { # PQC L=1
  for s in 0 1 2 3 4 5 6 7; do run7 paws pqc fourierL1 "$s" --n-layers 1; done
  for s in 0 1 2 3 4 5 6 7; do run7 qqp  pqc fourierL1 "$s" --n-layers 1; done
  echo "[$(date +%H:%M:%S)] BLOCK A (PQC L1) COMPLETE" >> logs/v7_fourier.log
}
blockB() { # PQC L=4
  for s in 0 1 2 3 4 5 6 7; do run7 paws pqc fourierL4 "$s" --n-layers 4; done
  for s in 0 1 2 3 4 5 6 7; do run7 qqp  pqc fourierL4 "$s" --n-layers 4; done
  echo "[$(date +%H:%M:%S)] BLOCK B (PQC L4) COMPLETE" >> logs/v7_fourier.log
}
blockC() { # MLP hidden=6 (~63 params, matches L4 PQC's 66)
  for s in 0 1 2 3 4 5 6 7; do run7 paws mlp fourierMLPh6 "$s" --n-layers 2 --mlp-hidden 6; done
  for s in 0 1 2 3 4 5 6 7; do run7 qqp  mlp fourierMLPh6 "$s" --n-layers 2 --mlp-hidden 6; done
  echo "[$(date +%H:%M:%S)] BLOCK C (MLP h6) COMPLETE" >> logs/v7_fourier.log
}

echo "[$(date +%H:%M:%S)] v7 E1 Fourier sweep starting (A|B|C parallel, 48 runs total)" > logs/v7_fourier.log
blockA & blockB & blockC &
wait
echo "[$(date +%H:%M:%S)] ALL BLOCKS COMPLETE" >> logs/v7_fourier.log
