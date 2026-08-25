#!/usr/bin/env python3
"""Re-score the q=8 QQP cell on the SAME 4,000-pair test subsample used at q=16.

Raised in co-author review: the q=8 -> q=16 QQP comparison put a full-split
(40,230-pair) number beside a subsampled (4,000-pair) one. The per-run JSONs
store every test prediction, so the q=8 cell can be re-scored on the identical
subsample without retraining. The index is reproduced exactly as the trainer
drew it (train_hybrid_v5.py):

    idx = np.random.default_rng(seed).choice(len(test_pairs), n_test, replace=False)
    test_pairs = [test_pairs[i] for i in sorted(idx)]

so seed s selects the same 4,000 of 40,230 pairs it selected at q=16.
"""
from __future__ import annotations
import glob, json
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

N_TEST, N_FULL = 4000, 40230


def load(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f))
        out[d["config"]["seed"]] = d
    return out


def score(d, seed, subsample: bool):
    y = np.array(d["test_labels"]); p = np.array(d["test_predictions"])
    lg = np.array(d["test_logits"])
    if subsample:
        assert len(y) == N_FULL, f"expected full split, got {len(y)}"
        idx = np.sort(np.random.default_rng(seed).choice(N_FULL, N_TEST, replace=False))
        y, p, lg = y[idx], p[idx], lg[idx]
    return (f1_score(y, p, average="macro", zero_division=0),
            roc_auc_score(y, lg) if len(set(y)) > 1 else float("nan"))


q8p, q8m = load("results/v5/hybrid_v5_qqp_depthwise_pqc_q8_l2_s*.json"), \
           load("results/v5/hybrid_v5_qqp_depthwise_mlp_q8_l2_s*.json")
q16p, q16m = load("results/v8/hybrid_v8_qqp_depthwise_pqc_q16both_q16_l2_s*.json"), \
             load("results/v8/hybrid_v8_qqp_depthwise_mlp_q16both_q16_l2_s*.json")

seeds = sorted(set(q8p) & set(q8m) & set(q16p) & set(q16m) & set(range(8)))
print(f"seeds: {seeds}\n")

rows = {}
for lab, (P, M, sub) in {
    "q=8  full split (40,230)": (q8p, q8m, False),
    "q=8  matched subsample (4,000)": (q8p, q8m, True),
    "q=16 subsample (4,000)": (q16p, q16m, False),
}.items():
    pf = np.array([score(P[s], s, sub)[0] for s in seeds])
    mf = np.array([score(M[s], s, sub)[0] for s in seeds])
    gap = mf.mean() - pf.mean()
    p = stats.ttest_ind(pf, mf, equal_var=False).pvalue
    rows[lab] = (pf.mean(), mf.mean(), gap, p)
    print(f"{lab:<32} PQC {pf.mean():.4f}  MLP {mf.mean():.4f}  gap {gap:+.4f}  p={p:.4f}")

full = rows["q=8  full split (40,230)"][2]
matched = rows["q=8  matched subsample (4,000)"][2]
q16 = rows["q=16 subsample (4,000)"][2]
print(f"\nsubsampling shifts the q=8 gap by {matched - full:+.4f} "
      f"({abs(matched-full)/abs(full)*100:.1f}% of it)")
print(f"like-for-like reduction q=8 -> q=16: {(1 - q16/matched)*100:.1f}% "
      f"(previously quoted against the full split: {(1 - q16/full)*100:.1f}%)")
