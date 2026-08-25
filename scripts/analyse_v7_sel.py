#!/usr/bin/env python3
"""v7 E2 analysis -- generality of the bias-role asymmetry on a second ansatz
family (StronglyEntanglingHead: single AngleEmbedding + StronglyEntanglingLayers,
raw <Z_0> readout, no bias -- 24 circuit params at q=8, L=1).

Pre-registered in PREREGISTRATION_v7_mechanism.md SS E2 (read-only; not altered
by this script). Post-hoc-registered relative to PREREGISTRATION_v5.md.

P2 (generality): uncalibrated SEL shows the same pathology as uncalibrated
  reuploading (pos_pred_rate > 0.65; MLP-SEL gap positive), and calibration
  normalizes pos_pred_rate toward 0.5 and shrinks the gap to non-significance
  (p > 0.05).
P2-null: SEL shows no uncalibrated pathology, or calibration fails to move it.

SEL runs are grouped by config['freeze_calib']/config['run_tag'] (cross-checked
against each other), not by directory or filename -- both cal and uncal SEL
runs land in the same results/v7/ dir. MLP comparators are the fixed,
already-collected v4 (uncalibrated protocol) and v5 (calibrated protocol)
depthwise q8 baselines -- not re-run here.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 10_000


def load_sel():
    """SEL runs from results/v7/*sel*.json, grouped by calibration status."""
    groups = defaultdict(list)
    for f in sorted((ROOT / "results/v7").glob("*sel*.json")):
        d = json.load(open(f))
        c, test = d["config"], d["phase2"]["test"]
        if c.get("head") != "sel":
            continue
        uncal = bool(c.get("freeze_calib", False))
        tag = c.get("run_tag", "")
        expected_tag = "uncal" if uncal else "cal"
        if tag != expected_tag:
            print(f"  [warn] {f.name}: freeze_calib={uncal} but run_tag={tag!r} "
                  f"(expected {expected_tag!r}) -- trusting freeze_calib")
        groups["uncal" if uncal else "cal"].append({
            "seed": c["seed"],
            "macro_f1": test.get("macro_f1", float("nan")),
            "pos_pred_rate": test.get("pos_pred_rate", float("nan")),
            "file": f.name,
        })
    return groups


def load_mlp(version):
    """Fixed comparator baselines (depthwise, q8, l2, seeds 0-7) -- not re-run."""
    rows = []
    for s in range(8):
        f = ROOT / f"results/{version}/hybrid_{version}_mrpc_depthwise_mlp_q8_l2_s{s}.json"
        d = json.load(open(f))
        test = d["phase2"]["test"]
        rows.append({
            "seed": s,
            "macro_f1": test.get("macro_f1", float("nan")),
            "pos_pred_rate": test.get("pos_pred_rate", float("nan")),
        })
    return rows


def welch(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def bootstrap_ci(a, b, n_resamples=N_BOOT, seed=0):
    """95% CI of mean(b) - mean(a) via independent resampling of each arm."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(a), size=(n_resamples, len(a)))
    idx_b = rng.integers(0, len(b), size=(n_resamples, len(b)))
    diffs = b[idx_b].mean(axis=1) - a[idx_a].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def fmt(vals):
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean():.4f}±{vals.std(ddof=1):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}(n=1)"


def main():
    sel = load_sel()
    mlp_uncal = load_mlp("v4")
    mlp_cal = load_mlp("v5")

    print("=" * 100)
    print("v7 E2 ANALYSIS -- second ansatz family (StronglyEntanglingHead), generality of bias-role asymmetry")
    print("PREREGISTRATION_v7_mechanism.md SS E2 (post-hoc-registered relative to PREREGISTRATION_v5.md)")
    print("=" * 100)

    for tag, n_expected in (("uncal", 8), ("cal", 8)):
        n = len(sel[tag])
        if n != n_expected:
            print(f"  [warn] SEL {tag}: {n}/{n_expected} seeds present in results/v7/")

    sel_uncal_f1 = [r["macro_f1"] for r in sel["uncal"]]
    sel_cal_f1 = [r["macro_f1"] for r in sel["cal"]]
    sel_uncal_pp = [r["pos_pred_rate"] for r in sel["uncal"]]
    sel_cal_pp = [r["pos_pred_rate"] for r in sel["cal"]]
    mlp_uncal_f1 = [r["macro_f1"] for r in mlp_uncal]
    mlp_cal_f1 = [r["macro_f1"] for r in mlp_cal]
    mlp_uncal_pp = [r["pos_pred_rate"] for r in mlp_uncal]
    mlp_cal_pp = [r["pos_pred_rate"] for r in mlp_cal]

    print(f"\n{'Group':<30} {'n':<4} {'macro F1':<18} {'pos_pred_rate':<16}")
    print("-" * 72)
    print(f"{'SEL uncalibrated':<30} {len(sel_uncal_f1):<4} {fmt(sel_uncal_f1):<18} {fmt(sel_uncal_pp)}")
    print(f"{'SEL calibrated':<30} {len(sel_cal_f1):<4} {fmt(sel_cal_f1):<18} {fmt(sel_cal_pp)}")
    print(f"{'MLP uncalibrated (v4, fixed)':<30} {len(mlp_uncal_f1):<4} {fmt(mlp_uncal_f1):<18} {fmt(mlp_uncal_pp)}")
    print(f"{'MLP calibrated (v5, fixed)':<30} {len(mlp_cal_f1):<4} {fmt(mlp_cal_f1):<18} {fmt(mlp_cal_pp)}")

    # Gaps: MLP - SEL, uncalibrated and calibrated
    t_uncal, p_uncal = welch(sel_uncal_f1, mlp_uncal_f1)
    ci_uncal = bootstrap_ci(sel_uncal_f1, mlp_uncal_f1, seed=1)
    gap_uncal = float(np.mean(mlp_uncal_f1) - np.mean(sel_uncal_f1))

    t_cal, p_cal = welch(sel_cal_f1, mlp_cal_f1)
    ci_cal = bootstrap_ci(sel_cal_f1, mlp_cal_f1, seed=2)
    gap_cal = float(np.mean(mlp_cal_f1) - np.mean(sel_cal_f1))

    print(f"\n{'Gap (MLP - SEL)':<16} {'delta':<10} {'Welch p':<10} {'95% CI (10k boot)':<22}")
    print("-" * 60)
    print(f"{'uncalibrated':<16} {gap_uncal:+.4f}    {p_uncal:.4f}    [{ci_uncal[0]:+.4f}, {ci_uncal[1]:+.4f}]")
    print(f"{'calibrated':<16} {gap_cal:+.4f}    {p_cal:.4f}    [{ci_cal[0]:+.4f}, {ci_cal[1]:+.4f}]")

    # SEL's own uncal -> cal shift
    t_self_f1, p_self_f1 = welch(sel_uncal_f1, sel_cal_f1)
    t_self_pp, p_self_pp = welch(sel_uncal_pp, sel_cal_pp)
    delta_self_f1 = float(np.mean(sel_cal_f1) - np.mean(sel_uncal_f1))
    delta_self_pp = float(np.mean(sel_cal_pp) - np.mean(sel_uncal_pp))

    print("\nSEL uncal -> cal change:")
    print(f"  macro F1:      {delta_self_f1:+.4f}  Welch p={p_self_f1:.4f}")
    print(f"  pos_pred_rate: {delta_self_pp:+.4f}  Welch p={p_self_pp:.4f}")

    # P2 check (PREREGISTRATION_v7_mechanism.md SS E2, verbatim thresholds)
    pp_uncal_mean = float(np.mean(sel_uncal_pp))
    pp_cal_mean = float(np.mean(sel_cal_pp))
    pathology_present = pp_uncal_mean > 0.65 and gap_uncal > 0
    moved_toward_half = abs(pp_cal_mean - 0.5) < abs(pp_uncal_mean - 0.5)
    calibration_normalizes = moved_toward_half and p_cal > 0.05
    p2_holds = pathology_present and calibration_normalizes

    print("\n" + "=" * 100)
    print("P2 CHECK (PREREGISTRATION_v7_mechanism.md SS E2)")
    print("=" * 100)
    print(f"  (a) Uncalibrated SEL pathology: pos_pred_rate={pp_uncal_mean:.4f} (>0.65? {pp_uncal_mean > 0.65}), "
          f"MLP-SEL gap={gap_uncal:+.4f} (positive? {gap_uncal > 0}) "
          f"-> {'PATHOLOGY PRESENT' if pathology_present else 'NO PATHOLOGY'}")
    print(f"  (b) Calibration normalizes: pos_pred_rate {pp_uncal_mean:.4f} -> {pp_cal_mean:.4f} "
          f"(moved toward 0.5? {moved_toward_half}), calibrated gap p={p_cal:.4f} (non-sig? {p_cal > 0.05}) "
          f"-> {'NORMALIZED' if calibration_normalizes else 'NOT NORMALIZED'}")
    print(f"\n  VERDICT: {'P2 (generality holds -- artefact + remedy transfer across ansatz families)' if p2_holds else 'P2-null (asymmetry may be reuploading-specific, or calibration failed to move it)'}")

    summary = {
        "prereg": "PREREGISTRATION_v7_mechanism.md SS E2",
        "sel_uncalibrated": {
            "n": len(sel_uncal_f1),
            "macro_f1_mean": float(np.mean(sel_uncal_f1)), "macro_f1_std": float(np.std(sel_uncal_f1, ddof=1)),
            "pos_pred_rate_mean": pp_uncal_mean, "pos_pred_rate_std": float(np.std(sel_uncal_pp, ddof=1)),
        },
        "sel_calibrated": {
            "n": len(sel_cal_f1),
            "macro_f1_mean": float(np.mean(sel_cal_f1)), "macro_f1_std": float(np.std(sel_cal_f1, ddof=1)),
            "pos_pred_rate_mean": pp_cal_mean, "pos_pred_rate_std": float(np.std(sel_cal_pp, ddof=1)),
        },
        "gap_mlp_minus_sel_uncalibrated": {
            "delta": gap_uncal, "welch_t": t_uncal, "welch_p": p_uncal, "bootstrap_ci95": list(ci_uncal),
        },
        "gap_mlp_minus_sel_calibrated": {
            "delta": gap_cal, "welch_t": t_cal, "welch_p": p_cal, "bootstrap_ci95": list(ci_cal),
        },
        "sel_uncal_to_cal_shift": {
            "delta_macro_f1": delta_self_f1, "welch_p_macro_f1": p_self_f1,
            "delta_pos_pred_rate": delta_self_pp, "welch_p_pos_pred_rate": p_self_pp,
        },
        "p2_check": {
            "pathology_present_uncalibrated": pathology_present,
            "calibration_normalizes": calibration_normalizes,
            "verdict": "P2" if p2_holds else "P2-null",
        },
    }
    out_path = ROOT / "results/v7_sel_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
