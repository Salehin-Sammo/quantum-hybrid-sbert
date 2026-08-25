# Pre-Registration — v5 Calibration-Lever Experiment

**Registered:** 2026-05-15, BEFORE running any v5 experiment.
**Author:** Nazmus Salehin Sammo

## Motivating observation (from v4, already collected)

Under class-weighted BCE + macro-F1 early stopping, on the **depthwise** reducer:

- MRPC: MLP macro F1 = 0.540 ± 0.032 vs PQC 0.500 ± 0.027, Welch p = 0.017
- PAWS: MLP macro F1 = 0.505 ± 0.005 vs PQC 0.466 ± 0.022, Welch p = 0.0012
- PQC systematically over-predicts the positive class: pos_pred_rate 0.7–0.9,
  while MLP balances at ~0.5.
- The gap **vanishes** with the rich linear reducer (MRPC Welch p = 0.285).

## Mechanistic hypothesis

The `ClassicalMLPHead` uses `nn.Linear` layers, which **include bias terms**,
so it can freely shift its decision threshold. The `PQCHead` outputs the raw
expectation value ⟨Z₀⟩ ∈ [−1, 1] with **no bias and no scale term** — it is
structurally unable to recenter or rescale its decision boundary. The
"matched parameter" comparison in v4 (and in much of the QNLP literature)
therefore contains an uncontrolled architectural asymmetry: the classical
head gets free threshold-shifting parameters, the quantum head does not.

**H1 (calibration):** The PQC–MLP gap on depthwise features is largely a
calibration artefact caused by the PQC's missing bias/scale. Adding a
learnable affine calibration (scale `a`, bias `b`; 2 parameters) on the head
logit will substantially close the gap.

**H0 (representational):** The gap is a genuine representational deficit of
the quantum head. Calibration will not close it (or will help both heads
equally, leaving the gap and its significance intact).

## Falsifiable predictions (registered before the run)

The SAME 2-parameter affine calibration is added to **both** heads, keeping
the comparison matched (PQC: 32→34 params; MLP: 31→33 params).

- **If H1 is true:** calibrated-PQC macro F1 rises toward calibrated-MLP, and
  the Welch p for PQC-vs-MLP on **depthwise** becomes non-significant
  (p > 0.05) on BOTH MRPC and PAWS. The MLP barely changes (it already had
  biases). PQC pos_pred_rate moves toward ~0.5.
- **If H0 is true:** the gap persists, Welch p stays < 0.05 on depthwise,
  and PQC pos_pred_rate stays skewed (> 0.65) even after calibration.

## Decision rule

- H1 supported if, after calibration, depthwise PQC-vs-MLP Welch p > 0.05 on
  both tasks AND |Δ macro F1| < 0.015.
- H0 supported if depthwise Welch p < 0.05 on at least one task after
  calibration.
- Partial / mixed outcomes will be reported honestly as such.

## Why both outcomes are publishable

- H1 → "Apparent matched-parameter PQC deficits in QNLP may be a missing
  classical-bias artefact; controlling for it restores parity." This
  re-frames a swath of QNLP negative results.
- H0 → "The PQC deficit is genuinely representational, not calibration,"
  sharpening the dequantization mechanism with a control the literature
  lacks.

## Scope

- Primary: **depthwise** reducer (where the v4 gap was significant),
  8 seeds, MRPC + PAWS, both heads. = 32 runs.
- Control: **linear** reducer (where v4 gap was already absent) — run if
  time permits; prediction is "no change either way."
- All other hyperparameters identical to v4 (class-weighted BCE, macro-F1
  early stopping patience 6, lr 0.002, 20 Phase-1 + 25 Phase-2 epochs,
  n_train 2000, n_val 200).
