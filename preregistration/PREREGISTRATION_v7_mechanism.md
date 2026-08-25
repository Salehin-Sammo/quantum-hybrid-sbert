# Pre-Registration — v7 Mechanism Experiments

**Registered:** 2026-07-13, BEFORE running any v7 experiment.
**Author:** Nazmus Salehin Sammo

## Motivating state (v6, already collected)

The calibration lever (v5) decomposed the matched-parameter PQC deficit into a
dominant bias-role artefact plus a residual that is task-dependent: MRPC none,
PAWS +0.0073 (16 seeds, p=0.033), QQP +0.037 (d=2.7, Holm-significant).
Limitations §3 names three candidate mechanisms for the residual: (a) limited
Fourier expressivity of 2-layer data reuploading, (b) optimization difficulty,
(c) a further unidentified role asymmetry. These experiments test (a) and
partially (c); all are post-hoc relative to PREREGISTRATION_v5.md and will be
reported as such.

## Experiment E1 — Reuploading-depth (Fourier) sweep on the residual cells

Data-reuploading circuits realize truncated Fourier series whose accessible
spectrum grows with the number of reuploading layers L (Schuld, Sweke & Meyer
2021). If the residual is Fourier-expressivity, deepening the circuit should
shrink it; if it is optimization or a hidden role asymmetry, depth alone
should not.

**Runs:** v5 (calibrated) protocol, depthwise reducer, q=8, seeds 0–7, on QQP
and PAWS: PQC at L=1 (18 params incl. calibration) and L=4 (66 params), added
to the existing L=2 (34 params). Classical comparator at the L=4 scale: MLP
with hidden width 6 (63 params incl. calibration, vs the L=4 PQC's 66),
same seeds/tasks.

**P1 (expressivity):** the calibrated PQC–MLP gap decreases monotonically in
L (L1 > L2 > L4) on both residual tasks, and at L=4 the QQP gap vs the
matched ~63-param MLP is less than half the L=2 gap (< 0.0185).
**P1-null:** the L=4 QQP gap stays ≥ 0.0185 → depth/spectrum is not the
binding constraint; optimization difficulty or a further role asymmetry
becomes the leading candidate.

## Experiment E2 — Second ansatz family (generality of bias-role asymmetry)

The bias-role asymmetry claim should not be specific to data reuploading.
We add a StronglyEntanglingLayers head (single angle encoding, no
reuploading, raw ⟨Z₀⟩ readout, no bias — 24 circuit params at q=8, L=1),
run uncalibrated (v4 protocol) and calibrated (v5 protocol) on MRPC
depthwise, seeds 0–7, against the existing MLP results.

**P2 (generality):** uncalibrated SEL shows the same pathology as
uncalibrated reuploading (pos_pred_rate > 0.65; MLP−SEL gap positive), and
calibration normalizes pos_pred_rate toward 0.5 and shrinks the gap to
non-significance (p > 0.05) — i.e., the artefact and its remedy transfer
across ansatz families.
**P2-null:** SEL shows no uncalibrated pathology (the asymmetry would then be
reuploading-specific), or calibration fails to move it.

## Experiment E3 — Noise proxy at multi-seed

The v6 depolarizing-noise proxy used a single seed. We repeat it for seeds
0–7 (MRPC depthwise, v5, q=8, L=2) at p ∈ {0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1}.

**P3 (stability):** for every seed, macro F1 degrades by < 0.02 up to
p = 1e-2 (the NISQ-realistic range), with collapse only at p ≥ 5e-2.
**P3-null:** ≥ 2 seeds degrade > 0.02 within the NISQ range → the v6
single-seed stability claim must be withdrawn or weakened.

## Experiment E4 — Zero-parameter remedy (added 2026-07-13, before any E4 run)

**Provenance, stated honestly:** E2's first two seeds showed uncalibrated SEL
with pos_pred_rate 0.49/0.58 — no positive-bias pathology, unlike
data-reuploading's 0.79. That prompted an *initialisation-time* probe
(`scripts/readout_offset_v7.py`, no training): on real frozen MRPC features,
mean <Z_0> at init is **+0.922** for reuploading (init pos_pred_rate 1.000)
versus **+0.010** for SEL (0.542), across 8 seeds with across-seed std ~0.02.
Rescaling the reuploading init 0.1 -> 0.5 -> 1.0 -> 2.0 drives the offset
+0.922 -> +0.176 -> +0.005 -> +0.001, causally confirming the cause: the
encoding angle RY(theta_i * x_i * pi) is scaled by a small-initialised
parameter, so at init all rotations ~ 0, the state stays near |0...0>, and
<Z_0> is pinned near +1.

These are initialisation-time facts. **Whether they change TRAINED task
performance is untested and is what E4 tests.**

**Refined mechanism claim (to be tested):** the bias-role asymmetry bites only
when a bias-free head's readout is *systematically offset from the decision
threshold*. If so, the offset — not the missing bias per se — is the operative
condition, and removing the offset by any means should reproduce the
calibration benefit.

**Runs:** v5 trainer with `--freeze-calib` (affine pinned at identity, so the
PQC has **exactly 32 parameters — no parameters added vs v4**), depthwise, q=8,
L=2, seeds 0-7, MRPC and PAWS, at `--init-scale 1.0`. Compared against the
existing v4 uncalibrated PQC (init 0.1, 32 params) and v4 MLP (31 params).

**P4 (offset, not bias-count):** at init-scale 1.0 the uncalibrated PQC's
pos_pred_rate falls below 0.65 on both tasks, and the MLP-PQC gap shrinks to
non-significance (Welch p > 0.05) on MRPC — replicating the calibration
benefit with **zero added parameters**. This would refute the residual
objection that v5's benefit came from the 2 extra parameters buying capacity.
**P4-null:** the gap persists at p < 0.05 despite the centred readout => the
init-time offset is not what drove the v4 deficit, the refined mechanism claim
is wrong, and the E4 result will be reported as a failed prediction.

## Reporting rule

All three outcomes are reported regardless of direction, in the
Extended-Evidence section, labelled post-hoc-registered (this document
predates the runs; PREREGISTRATION_v5.md governs the primary claims).
Parameter counts are no longer matched in E1 by design — the question is
mechanism, not fairness — and the text will say so explicitly.
