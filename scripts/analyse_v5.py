#!/usr/bin/env python3
"""v5 calibration-lever analysis — directly tests the pre-registered hypothesis.

Compares v4 (no calibration) vs v5 (2-param affine calibration on both heads)
for the depthwise reducer, where the v4 PQC-MLP gap was significant.

Decision rule (from PREREGISTRATION_v5.md):
  H1 (calibration artefact) supported if, after calibration, depthwise
    PQC-vs-MLP Welch p > 0.05 on BOTH tasks AND |Δ macro F1| < 0.015.
  H0 (representational deficit) supported if Welch p < 0.05 on >=1 task.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
import json, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

def load(version):
    out = defaultdict(list)
    paths = sorted((ROOT / f"results/{version}").glob(f"hybrid_{version}_*.json"))
    if version == "v5":  # robustness: catch any stray files from the path-bug window
        paths += sorted((ROOT / "results/v4").glob("hybrid_v5_*.json"))
    for f in paths:
        d = json.load(open(f))
        c = d["config"]; p2 = d["phase2"]["test"]
        out[(c["task"], c["reducer"], c["head"])].append({
            "seed": c["seed"],
            "macro_f1": p2.get("macro_f1", float("nan")),
            "f1": p2.get("f1", float("nan")),
            "pp_rate": p2.get("pos_pred_rate", float("nan")),
        })
    return out

def stat(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 2 or len(b) < 2:
        return None
    _, pw = stats.ttest_ind(a, b, equal_var=False)
    _, pmw = stats.mannwhitneyu(a, b, alternative="two-sided")
    return pw, pmw, b.mean() - a.mean()

v4, v5 = load("v4"), load("v5")

print("="*86)
print("v5 CALIBRATION-LEVER ANALYSIS  (pre-registered: PREREGISTRATION_v5.md)")
print("="*86)

for reducer in ("depthwise", "linear"):
    print(f"\n### Reducer: {reducer}  {'(HEADLINE TEST)' if reducer=='depthwise' else '(CONTROL)'}")
    print(f"{'Task':<6} {'Ver':<4} {'PQC macroF1':<16} {'MLP macroF1':<16} "
          f"{'Δ(MLP-PQC)':<11} {'Welch p':<9} {'PQC pp_rate':<12}")
    print("-"*86)
    for task in ("mrpc", "paws"):
        for ver, data in (("v4", v4), ("v5", v5)):
            pk = (task, reducer, "pqc"); mk = (task, reducer, "mlp")
            if pk not in data or mk not in data:
                continue
            pqc = [r["macro_f1"] for r in data[pk]]
            mlp = [r["macro_f1"] for r in data[mk]]
            pqc_pp = np.mean([r["pp_rate"] for r in data[pk]])
            s = stat(pqc, mlp)
            pqc_s = f"{np.mean(pqc):.4f}±{np.std(pqc,ddof=1):.3f}" if len(pqc)>1 else f"{pqc[0]:.4f}(n=1)" if pqc else "—"
            mlp_s = f"{np.mean(mlp):.4f}±{np.std(mlp,ddof=1):.3f}" if len(mlp)>1 else f"{mlp[0]:.4f}(n=1)" if mlp else "—"
            if s:
                pw, pmw, delta = s
                sig = "*" if pw < 0.05 else " "
                print(f"{task:<6} {ver:<4} {pqc_s:<16} {mlp_s:<16} "
                      f"{delta:+.4f}    {pw:.4f}{sig}  {pqc_pp:.3f}")
            else:
                print(f"{task:<6} {ver:<4} {pqc_s:<16} {mlp_s:<16} "
                      f"(n<2 for sig test)  pp={pqc_pp:.3f}")

print("\n" + "="*86)
print("PRE-REGISTERED DECISION (depthwise only):")
print("="*86)
verdict_ready = True
for task in ("mrpc", "paws"):
    pk = (task, "depthwise", "pqc"); mk = (task, "depthwise", "mlp")
    if pk in v5 and mk in v5:
        pqc = [r["macro_f1"] for r in v5[pk]]
        mlp = [r["macro_f1"] for r in v5[mk]]
        if len(pqc) >= 2 and len(mlp) >= 2:
            pw = stats.ttest_ind(pqc, mlp, equal_var=False)[1]
            delta = abs(np.mean(mlp) - np.mean(pqc))
            h1 = (pw > 0.05 and delta < 0.015)
            print(f"  {task}: v5 Welch p={pw:.4f}, |Δ|={delta:.4f}  "
                  f"→ {'H1 (calibration closes gap)' if h1 else 'H0 (gap persists)'}")
        else:
            print(f"  {task}: insufficient v5 seeds ({len(pqc)} PQC, {len(mlp)} MLP)")
            verdict_ready = False
    else:
        print(f"  {task}: v5 depthwise data not yet available")
        verdict_ready = False
if not verdict_ready:
    print("\n  [verdict pending — sweep still running]")
