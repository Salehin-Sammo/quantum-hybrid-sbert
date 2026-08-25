#!/bin/bash
# v6 gap-closing sweep — 4 parallel blocks, priority-ordered within each.
#   A: v5 PAWS depthwise seeds 8-15          (resolve p=0.061 marginality; 16 runs)
#   B: v4 PAWS linear seeds 0-7              (missing baseline for v5 linear-PAWS p=0.028; 16 runs)
#   C: QQP depthwise v5 then v4, seeds 0-7   (close the QQP claim; 32 runs)
#   D: q-sweep v5 MRPC depthwise q=4,12      (single-qubit-count criticism; 32 runs)
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v4 results/v5

run5() { # task reducer head seed [extra...]
  local task=$1 reducer=$2 head=$3 s=$4; shift 4
  local TAG="v5_${reducer}_${task}_${head}_s${s}_$*"
  TAG=${TAG// /}; TAG=${TAG//--/}
  .venv/bin/python src/train_hybrid_v5.py --task "$task" --reducer "$reducer" --head "$head" \
    --seed "$s" --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
    --n-val 200 --n-train 2000 "$@" > "logs/gaps_${TAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $TAG rc=$?" >> logs/v6_gaps.log
}
run4() {
  local task=$1 reducer=$2 head=$3 s=$4; shift 4
  local TAG="v4_${reducer}_${task}_${head}_s${s}_$*"
  TAG=${TAG// /}; TAG=${TAG//--/}
  .venv/bin/python src/train_hybrid_v4.py --task "$task" --reducer "$reducer" --head "$head" \
    --seed "$s" --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
    --n-val 200 --n-train 2000 "$@" > "logs/gaps_${TAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $TAG rc=$?" >> logs/v6_gaps.log
}

blockA() { for s in 8 9 10 11 12 13 14 15; do for h in pqc mlp; do run5 paws depthwise $h $s; done; done
  echo "[$(date +%H:%M:%S)] BLOCK A COMPLETE" >> logs/v6_gaps.log; }
blockB() { for s in 0 1 2 3 4 5 6 7; do for h in pqc mlp; do run4 paws linear $h $s; done; done
  echo "[$(date +%H:%M:%S)] BLOCK B COMPLETE" >> logs/v6_gaps.log; }
blockC() { for s in 0 1 2 3 4 5 6 7; do for h in pqc mlp; do run5 qqp depthwise $h $s; done; done
  for s in 0 1 2 3 4 5 6 7; do for h in pqc mlp; do run4 qqp depthwise $h $s; done; done
  echo "[$(date +%H:%M:%S)] BLOCK C COMPLETE" >> logs/v6_gaps.log; }
blockD() { for q in 4 12; do for s in 0 1 2 3 4 5 6 7; do for h in pqc mlp; do
    run5 mrpc depthwise $h $s --n-qubits $q; done; done; done
  echo "[$(date +%H:%M:%S)] BLOCK D COMPLETE" >> logs/v6_gaps.log; }

echo "[$(date +%H:%M:%S)] v6 gaps sweep starting (A|B|C|D parallel)" > logs/v6_gaps.log
blockA & blockB & blockC & blockD &
wait
echo "[$(date +%H:%M:%S)] ALL BLOCKS COMPLETE" >> logs/v6_gaps.log
