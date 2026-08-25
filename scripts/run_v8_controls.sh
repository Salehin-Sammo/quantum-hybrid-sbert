#!/bin/bash
# v8 controls, both experiments Rakib scoped. Results -> results/v8/.
#
# A: MID-CAPACITY CONTROL (q=16, calibrated). Answers the referee objection
#    that everything is measured at or near chance. Reports AUC/MCC so we can
#    state whether both heads clear chance, and whether the QQP residual
#    survives when they do. QQP test is subsampled to 4,000 pairs because at
#    q=16 a 40,230-pair eval dominates wall-clock; MRPC keeps its full 208.
#
# B: BIAS-ONLY ABLATION (q=8, PQC). Separates threshold shift from rescaling.
#    Baselines already exist: mode=none is v4, mode=both is v5. This adds the
#    two single-parameter arms to complete the 2x2.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/v8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
LOG=logs/v8_controls.log

run() { # task q calib_mode head seed extra...
  local task=$1 q=$2 mode=$3 head=$4 s=$5; shift 5
  local TAG="v8_${task}_q${q}_${mode}_${head}_s${s}"
  .venv/bin/python src/train_hybrid_v5.py \
    --task "$task" --reducer depthwise --head "$head" \
    --n-qubits "$q" --n-layers 2 --seed "$s" \
    --calib-mode "$mode" --out-subdir v8 --run-tag "q${q}${mode}" \
    --lr 0.002 --pretrain-epochs 20 --epochs 25 --patience 6 \
    --n-val 200 --n-train 2000 "$@" > "logs/${TAG}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done $TAG rc=$?" >> "$LOG"
}

sem() { while [ "$(jobs -rp | wc -l)" -ge 6 ]; do wait -n; done; }

echo "[$(date +%H:%M:%S)] v8 controls starting (A: q16 mid-capacity, B: bias ablation)" >> "$LOG"

# ---- B first: cheap, gives early signal ----
for s in 0 1 2 3 4 5 6 7; do
  for mode in bias scale; do
    sem; run mrpc 8 "$mode" pqc "$s" &
  done
done
wait
echo "[$(date +%H:%M:%S)] B/mrpc COMPLETE" >> "$LOG"

for s in 0 1 2 3 4 5 6 7; do
  for mode in bias scale; do
    sem; run qqp 8 "$mode" pqc "$s" &
  done
done
wait
echo "[$(date +%H:%M:%S)] B/qqp COMPLETE -- EXPERIMENT B DONE" >> "$LOG"

# ---- A: mid-capacity q=16 ----
for s in 0 1 2 3 4 5 6 7; do
  for head in pqc mlp; do
    sem; run mrpc 16 both "$head" "$s" &
  done
done
wait
echo "[$(date +%H:%M:%S)] A/mrpc COMPLETE" >> "$LOG"

for s in 0 1 2 3 4 5 6 7; do
  for head in pqc mlp; do
    sem; run qqp 16 both "$head" "$s" --n-test 4000 &
  done
done
wait
echo "[$(date +%H:%M:%S)] A/qqp COMPLETE -- ALL v8 CONTROLS DONE" >> "$LOG"
