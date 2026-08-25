#!/usr/bin/env python3
"""Mid-capacity control (q=16) vs the q=8 main results.

Answers the two questions the control exists for:
  Q1  Does the extra capacity lift BOTH heads decisively clear of chance?
      (tested as AUC vs 0.5, since macro F1 near chance is threshold-driven)
  Q2  Does the PQC-MLP gap persist once they do?

Safe to run on partial data: every cell prints its own n and any cell with
fewer than 3 seeds per arm is marked PRELIMINARY rather than tested.
"""
from __future__ import annotations
import glob, json
import numpy as np
from scipy import stats

FIELDS = ("macro_f1", "auc", "mcc", "pos_pred_rate")


def cell(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f)); t = d["phase2"]["test"]
        out[d["config"]["seed"]] = tuple(t.get(k, float("nan")) for k in FIELDS)
    return out


def arr(d, i, seeds=None):
    ss = sorted(d) if seeds is None else [s for s in sorted(d) if s in seeds]
    return np.array([d[s][i] for s in ss])


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = b[rng.integers(0, len(b), (n, len(b)))].mean(1) - a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return np.percentile(g, [2.5, 97.5])


PATS = {
    ("mrpc", 8):  "results/v5/hybrid_v5_mrpc_depthwise_{h}_q8_l2_s*.json",
    ("qqp", 8):   "results/v5/hybrid_v5_qqp_depthwise_{h}_q8_l2_s*.json",
    ("mrpc", 16): "results/v8/hybrid_v8_mrpc_depthwise_{h}_q16both_q16_l2_s*.json",
    ("qqp", 16):  "results/v8/hybrid_v8_qqp_depthwise_{h}_q16both_q16_l2_s*.json",
}

print("=" * 86)
print("MID-CAPACITY CONTROL: q=16 vs q=8 (depthwise, calibrated)")
print("=" * 86)
print(f"{'task':<6}{'q':>3}{'head':>5}{'n':>4}{'macroF1':>17}{'AUC':>8}{'AUC>0.5 p':>11}{'MCC':>8}{'pp':>7}")
print("-" * 86)

store = {}
for task in ("mrpc", "qqp"):
    for q in (8, 16):
        for h in ("pqc", "mlp"):
            d = cell(PATS[(task, q)].format(h=h))
            if not d:
                continue
            # q=8 PAWS/MRPC v5 cells may carry extra seeds; restrict to 0-7 for comparability
            seeds = set(range(8))
            f1, auc, mcc, pp = (arr(d, i, seeds) for i in range(4))
            store[(task, q, h)] = (f1, auc, mcc, pp)
            pa = stats.ttest_1samp(auc, 0.5).pvalue if len(auc) > 1 else float("nan")
            print(f"{task:<6}{q:>3}{h:>5}{len(f1):>4}{f1.mean():>11.4f}±{f1.std(ddof=1) if len(f1)>1 else 0:.4f}"
                  f"{auc.mean():>8.4f}{pa:>11.4f}{mcc.mean():>8.4f}{pp.mean():>7.3f}")

print("\n" + "=" * 86)
print("Q2: does the PQC-MLP gap persist?")
print("=" * 86)
for task in ("mrpc", "qqp"):
    for q in (8, 16):
        k_p, k_m = (task, q, "pqc"), (task, q, "mlp")
        if k_p not in store or k_m not in store:
            continue
        p, m = store[k_p][0], store[k_m][0]
        if len(p) < 3 or len(m) < 3:
            print(f"  {task} q={q}: PRELIMINARY, n={len(p)}/{len(m)}, "
                  f"gap={m.mean()-p.mean():+.4f} (not tested)")
            continue
        lo, hi = boot(p, m)
        print(f"  {task} q={q}: gap={m.mean()-p.mean():+.4f}  Welch p={stats.ttest_ind(p,m,equal_var=False).pvalue:.4f}"
              f"  95% CI [{lo:+.4f},{hi:+.4f}]  n={len(p)}/{len(m)}")

print("\n" + "=" * 86)
print("Q1 VERDICT: is the capacity-starved objection answered?")
print("=" * 86)
for task in ("mrpc", "qqp"):
    k_p, k_m = (task, 16, "pqc"), (task, 16, "mlp")
    if k_p not in store or k_m not in store:
        print(f"  {task} q=16: incomplete"); continue
    ap, am = store[k_p][1], store[k_m][1]
    if len(ap) < 3 or len(am) < 3:
        print(f"  {task} q=16: PRELIMINARY (n={len(ap)}/{len(am)}) "
              f"AUC pqc={ap.mean():.4f} mlp={am.mean():.4f}"); continue
    pp = stats.ttest_1samp(ap, 0.5).pvalue
    pm = stats.ttest_1samp(am, 0.5).pvalue
    both = pp < 0.05 and pm < 0.05 and ap.mean() > 0.55 and am.mean() > 0.55
    print(f"  {task} q=16: AUC pqc={ap.mean():.4f} (p={pp:.4f}), mlp={am.mean():.4f} (p={pm:.4f})"
          f"  -> both decisively clear of chance: {both}")
