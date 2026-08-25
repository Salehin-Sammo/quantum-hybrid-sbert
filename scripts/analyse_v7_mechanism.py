#!/usr/bin/env python3
"""v7 E1 analysis -- reuploading-depth (Fourier) sweep. See PREREGISTRATION_v7_mechanism.md §E1.

Reads results/v7/*.json (new L=1/L=4 PQC + MLP-h6 runs) and the L=2 PQC /
MLP-h3 baselines from results/v5/. Groups by (task, head, n_layers, run_tag)
-- run_tag has to be in the key because MLP-h3 (baseline, run_tag="") and
MLP-h6 (this sweep, run_tag="fourierMLPh6") share the same n_layers=2 and
would otherwise collide into one cell.

Per task, reports:
  - macro F1 mean+-std and pos_pred_rate for PQC at L=1,2,4 and MLP at h=3,6
  - the calibrated gap (MLP_h6 - PQC) at each L, Welch p, 10k bootstrap 95% CI
  - P1: is the gap monotone decreasing in L, and is the L=4 QQP gap < 0.0185
    (half of the pre-registered L2 QQP gap, 0.037)?

Safe to run mid-sweep: missing/short cells print as pending rather than crash.
Outputs: results/v7_fourier_summary.json.
"""
from __future__ import annotations
import sys, json
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_rigor_v6 import bootstrap_gap_ci  # noqa: E402  (reuse: 10k-resample 95% CI of mean(mlp)-mean(pqc))

TASKS = ("qqp", "paws")
# (display label, head, n_layers, run_tag) -- the 5 reference cells this experiment needs.
ROWS = [
    ("PQC L1",  "pqc", 1, "fourierL1"),
    ("PQC L2",  "pqc", 2, ""),
    ("PQC L4",  "pqc", 4, "fourierL4"),
    ("MLP h3",  "mlp", 2, ""),
    ("MLP h6",  "mlp", 2, "fourierMLPh6"),
]
GAP_LS = [1, 2, 4]
PQC_TAG_FOR_L = {1: "fourierL1", 2: "", 4: "fourierL4"}
L2_QQP_GAP_PREREG = 0.037          # PREREGISTRATION_v7_mechanism.md "Motivating state"
P1_THRESHOLD = L2_QQP_GAP_PREREG / 2  # 0.0185, the pre-registered L=4 QQP pass bar


def load_runs():
    """(task, head, n_layers, run_tag) -> list of {seed, macro_f1, pos_pred_rate}.

    Scoped to reducer=depthwise and seeds 0-7 only, matching the prereg's E1
    scope exactly: results/v5/ also holds a linear-reducer variant (same
    task/head/n_layers/run_tag -- would silently collide without this filter)
    and, for paws specifically, an extra seeds-8-15 batch from an unrelated
    v6 gap-closing sweep that would unbalance the L2 row's seed count against
    the freshly-run L1/L4/h6 cells (seeds 0-7 only) if left in.
    """
    groups = defaultdict(list)
    seen = {}
    files = sorted((ROOT / "results/v7").glob("hybrid_v7_*.json")) + \
        sorted((ROOT / "results/v5").glob("hybrid_v5_*.json"))
    for f in files:
        d = json.load(open(f))
        c, test = d["config"], d["phase2"]["test"]
        if c["task"] not in TASKS or c["head"] not in ("pqc", "mlp"):
            continue
        if c.get("reducer") != "depthwise" or c["seed"] not in range(8):
            continue
        key = (c["task"], c["head"], c["n_layers"], c.get("run_tag", ""))
        dedup_key = key + (c["seed"],)
        if dedup_key in seen:
            print(f"  [warn] duplicate seed for {dedup_key}: keeping {seen[dedup_key]}, ignoring {f.name}")
            continue
        seen[dedup_key] = f.name
        groups[key].append({
            "seed": c["seed"],
            "macro_f1": test.get("macro_f1", float("nan")),
            "pos_pred_rate": test.get("pos_pred_rate", float("nan")),
        })
    return groups


def cell_arr(groups, task, head, n_layers, run_tag, field):
    rows = groups.get((task, head, n_layers, run_tag), [])
    return np.array([r[field] for r in rows])


def fmt_cell(macro_f1, pp):
    if len(macro_f1) == 0:
        return "  --- pending (0 seeds)"
    if len(macro_f1) < 2:
        return f"  {macro_f1[0]:.4f} (n=1, no std)          pp={pp[0]:.3f}"
    return f"  {macro_f1.mean():.4f}+-{macro_f1.std(ddof=1):.3f} (n={len(macro_f1)})   pp={pp.mean():.3f}"


def main():
    groups = load_runs()
    summary = {"tasks": {}, "prereg_l2_qqp_gap": L2_QQP_GAP_PREREG, "p1_threshold_l4_qqp": P1_THRESHOLD}

    print("=" * 100)
    print("v7 E1 -- REUPLOADING-DEPTH (FOURIER) SWEEP  (pre-registered: PREREGISTRATION_v7_mechanism.md §E1)")
    print("=" * 100)

    for task in TASKS:
        print(f"\n### Task: {task}")
        print(f"{'Row':<10} {'macro F1':<30} {'pos_pred_rate':<10}")
        print("-" * 55)
        task_rows = {}
        for label, head, nl, tag in ROWS:
            mf1 = cell_arr(groups, task, head, nl, tag, "macro_f1")
            pp = cell_arr(groups, task, head, nl, tag, "pos_pred_rate")
            task_rows[label] = {
                "n_seeds": int(len(mf1)),
                "macro_f1_mean": float(mf1.mean()) if len(mf1) else None,
                "macro_f1_std": float(mf1.std(ddof=1)) if len(mf1) > 1 else None,
                "pos_pred_rate_mean": float(pp.mean()) if len(pp) else None,
            }
            print(f"{label:<10}{fmt_cell(mf1, pp)}")

        # Gap (MLP_h6 - PQC_L) at L=1,2,4 -- param counts are intentionally NOT
        # matched at every L in E1 (prereg's own framing); MLP h6 is the fixed
        # ~63-param classical anchor throughout, matched only at L=4 (66 params).
        mlp6 = cell_arr(groups, task, "mlp", 2, "fourierMLPh6", "macro_f1")
        print(f"\n  Gap = MLP_h6 - PQC_L  (MLP_h6: n={len(mlp6)})")
        print(f"  {'L':<4}{'gap (MLP_h6-PQC)':<20}{'Welch p':<12}{'95% CI (10k boot)':<24}{'n(pqc)':<8}")
        gap_by_l = {}
        for L in GAP_LS:
            pqc = cell_arr(groups, task, "pqc", L, PQC_TAG_FOR_L[L], "macro_f1")
            if len(pqc) < 2 or len(mlp6) < 2:
                print(f"  L={L:<2} pending ({len(pqc)} pqc seeds, {len(mlp6)} mlp_h6 seeds)")
                gap_by_l[L] = None
                continue
            gap = float(mlp6.mean() - pqc.mean())
            _, p = stats.ttest_ind(mlp6, pqc, equal_var=False)
            ci_lo, ci_hi = bootstrap_gap_ci(pqc, mlp6)
            gap_by_l[L] = {"gap": gap, "welch_p": float(p), "ci95": [ci_lo, ci_hi], "n_pqc": int(len(pqc))}
            print(f"  L={L:<2} {gap:+.4f}             {p:.4f}      [{ci_lo:+.4f}, {ci_hi:+.4f}]      {len(pqc)}")

        # P1 checks
        vals = [gap_by_l[L]["gap"] for L in GAP_LS if gap_by_l[L] is not None]
        complete = len(vals) == 3
        monotone = complete and (vals[0] > vals[1] > vals[2])
        threshold_pass = None
        if task == "qqp" and gap_by_l[4] is not None:
            threshold_pass = gap_by_l[4]["gap"] < P1_THRESHOLD

        task_rows_summary = {
            "gap_by_L": gap_by_l,
            "monotone_decreasing": monotone if complete else None,
            "l4_qqp_below_threshold": threshold_pass,
        }
        summary["tasks"][task] = {"rows": task_rows, **task_rows_summary}

        print(f"\n  Monotone decreasing (L1>L2>L4)?  "
              f"{'--- pending' if not complete else ('YES' if monotone else 'NO')}")
        if task == "qqp":
            if gap_by_l[4] is None:
                print(f"  L=4 QQP gap < {P1_THRESHOLD:.4f} (half of prereg L2 gap {L2_QQP_GAP_PREREG})?  --- pending")
            else:
                g4 = gap_by_l[4]["gap"]
                print(f"  L=4 QQP gap < {P1_THRESHOLD:.4f} (half of prereg L2 gap {L2_QQP_GAP_PREREG})?  "
                      f"{'YES' if threshold_pass else 'NO'}  (measured {g4:+.4f})")

    # ---- Overall verdict ----------------------------------------------------
    print("\n" + "=" * 100)
    print("P1 VERDICT  (PREREGISTRATION_v7_mechanism.md §E1)")
    print("=" * 100)
    print("  P1 (expressivity): gap decreases monotonically in L (L1>L2>L4) on BOTH residual tasks,")
    print(f"                     AND L=4 QQP gap < {P1_THRESHOLD:.4f} (half the L2 gap).")
    print("  P1-null: L=4 QQP gap stays >= 0.0185 -> depth/spectrum is not the binding constraint.")

    mono_all = {t: summary["tasks"][t]["monotone_decreasing"] for t in TASKS}
    qqp_thresh = summary["tasks"]["qqp"]["l4_qqp_below_threshold"]
    ready = all(v is not None for v in mono_all.values()) and qqp_thresh is not None
    verdict = None
    if ready:
        verdict = "P1" if (all(mono_all.values()) and qqp_thresh) else "P1-null"
        print(f"\n  monotone: {mono_all}   l4_qqp_below_{P1_THRESHOLD:.4f}: {qqp_thresh}")
        print(f"  -> {verdict}")
    else:
        print(f"\n  monotone so far: {mono_all}   l4_qqp_below_{P1_THRESHOLD:.4f}: {qqp_thresh}")
        print("  [verdict pending -- sweep still running]")
    summary["verdict"] = verdict

    out_path = ROOT / "results/v7_fourier_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
