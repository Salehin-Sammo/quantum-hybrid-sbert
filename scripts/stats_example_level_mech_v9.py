#!/usr/bin/env python3
"""Item-level versions of the calibration-mechanism contrasts.

The eight secondary tests of the calibration experiment (per-head macro F1
change, threshold normalisation and head x protocol interaction on MRPC and
PAWS, depthwise, q = 8, seeds 0-7) compare runs that share a test partition
within a seed. As in stats_example_level_v9.py the development item is taken
as the sampling unit: two-arm contrasts get a cluster permutation test that
swaps the two arms' predictions for an item in every seed, and every contrast
gets a hierarchical (seeds x items) bootstrap interval.

Run from the repository root: python scripts/stats_example_level_mech_v9.py
Writes results/stats_example_level_mech_v9.json.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "scripts" in os.path.abspath(__file__) else os.getcwd()
if len(sys.argv) > 1:
    ROOT = sys.argv[1]
N_BOOT, CHUNK = 10000, 500
DEV_SIZE = {"mrpc": 408, "paws": 8000, "qqp": 40430}
SEEDS = list(range(8))


def test_ids(task, seed, n_val):
    idx = list(range(DEV_SIZE[task]))
    random.Random(seed).shuffle(idx)
    return np.asarray(idx[n_val:])


def load(paths):
    runs = defaultdict(dict)
    for f in paths:
        m = re.match(r"^hybrid_(v\d+)_", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        c = d["config"]
        if c.get("n_layers", 2) != 2 or c["head"] not in ("pqc", "mlp") or c["reducer"] != "depthwise" or c["n_qubits"] != 8:
            continue
        runs[(m.group(1), c["task"], c["head"])][c["seed"]] = d
    return runs


def macro_f1(c):
    c = np.asarray(c, np.float64)
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        fpos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        fneg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return 0.5 * (fpos + fneg)


def pos_rate(c):
    c = np.asarray(c, np.float64)
    return (c[..., 0] + c[..., 1]) / c.sum(-1)


def indicators(task, arms):
    """arms: dict label -> {seed: run}. Returns pool size and A[label] of shape (N, S*4)."""
    ids, labels = {}, {}
    for s in SEEDS:
        c = next(iter(arms.values()))[s]["config"]
        ids[s] = test_ids(task, s, c["n_val"])
        labels[s] = np.asarray(next(iter(arms.values()))[s]["test_labels"], int)
        for r in arms.values():
            assert r[s]["test_labels"] == labels[s].tolist(), "labels differ between arms"
            assert len(ids[s]) == len(labels[s])
    seen = {}
    for s in SEEDS:
        for i, y in zip(ids[s], labels[s]):
            assert seen.setdefault(int(i), int(y)) == int(y)
    pool = np.array(sorted(seen))
    row = {int(i): k for k, i in enumerate(pool)}
    N, S = len(pool), len(SEEDS)
    A = {}
    for lab, r in arms.items():
        M = np.zeros((N, S, 4), np.float32)
        for k, s in enumerate(SEEDS):
            rr = np.array([row[int(i)] for i in ids[s]])
            y = labels[s]
            p = np.asarray(r[s]["test_predictions"], int)
            M[rr, k, 0] = (p == 1) & (y == 1)
            M[rr, k, 1] = (p == 1) & (y == 0)
            M[rr, k, 2] = (p == 0) & (y == 1)
            M[rr, k, 3] = (p == 0) & (y == 0)
        A[lab] = M.reshape(N, S * 4)
    return N, S, A


def analyse(task, arms, stat, perm_pair=None, rng_seed=0):
    """stat(counts: dict label -> array (..., S, 4)) -> array (..., S) per-seed statistic."""
    N, S, A = indicators(task, arms)
    base = {lab: A[lab].sum(0).reshape(S, 4) for lab in A}
    per_seed = stat(base)
    obs = float(per_seed.mean())
    rng = np.random.default_rng(rng_seed)
    hier, perm = [], []
    done = 0
    while done < N_BOOT:
        b = min(CHUNK, N_BOOT - done)
        draws = rng.integers(0, N, size=(b, N))
        W = np.stack([np.bincount(dr, minlength=N) for dr in draws]).astype(np.float32)
        counts = {lab: (W @ A[lab]).reshape(b, S, 4) for lab in A}
        d_ws = stat(counts)  # (b, S)
        sd = rng.integers(0, S, size=(b, S))
        M = np.stack([np.bincount(x, minlength=S) for x in sd]).astype(np.float64)
        hier.extend(((d_ws * M).sum(1) / S).tolist())
        if perm_pair is not None:
            a, c = perm_pair
            F = rng.integers(0, 2, size=(b, N)).astype(np.float32)
            FD = (F @ (A[c] - A[a])).reshape(b, S, 4)
            swapped = {a: base[a][None] + FD, c: base[c][None] - FD}
            perm.extend(stat(swapped).mean(1).tolist())
        done += b
    out = {"n_items_pool": int(N), "value": obs, "per_seed": per_seed.tolist(),
           "seed_paired_t_p": float(stats.ttest_1samp(per_seed, 0.0).pvalue),
           "hier_boot_ci95": [float(x) for x in np.percentile(hier, [2.5, 97.5])]}
    if perm_pair is not None:
        perm = np.asarray(perm)
        out["perm_p"] = float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (N_BOOT + 1))
    return out


def main():
    runs = load(sorted(glob.glob(os.path.join(ROOT, "results/v4/*.json"))) + sorted(glob.glob(os.path.join(ROOT, "results/v5/*.json"))))
    results = {}
    for task in ("mrpc", "paws"):
        arms = {"pqc_u": runs[("v4", task, "pqc")], "pqc_c": runs[("v5", task, "pqc")],
                "mlp_u": runs[("v4", task, "mlp")], "mlp_c": runs[("v5", task, "mlp")]}
        arms = {k: {s: v[s] for s in SEEDS} for k, v in arms.items()}
        results[task] = {
            "pqc_gain": analyse(task, {k: arms[k] for k in ("pqc_u", "pqc_c")}, lambda c: macro_f1(c["pqc_c"]) - macro_f1(c["pqc_u"]), ("pqc_u", "pqc_c")),
            "mlp_shift": analyse(task, {k: arms[k] for k in ("mlp_u", "mlp_c")}, lambda c: macro_f1(c["mlp_c"]) - macro_f1(c["mlp_u"]), ("mlp_u", "mlp_c")),
            "threshold_norm": analyse(task, {k: arms[k] for k in ("pqc_u", "pqc_c")}, lambda c: pos_rate(c["pqc_u"]) - pos_rate(c["pqc_c"]), ("pqc_u", "pqc_c")),
            "interaction": analyse(task, arms, lambda c: (macro_f1(c["mlp_c"]) - macro_f1(c["pqc_c"])) - (macro_f1(c["mlp_u"]) - macro_f1(c["pqc_u"])), None),
        }
        for k, v in results[task].items():
            print(f'{task} {k:<15} value {v["value"]:+.4f} seed-level p {v["seed_paired_t_p"]:.4f} hier CI [{v["hier_boot_ci95"][0]:+.4f},{v["hier_boot_ci95"][1]:+.4f}]'
                  + (f' perm p {v["perm_p"]:.4f}' if "perm_p" in v else ""))
    json.dump(results, open(os.path.join(ROOT, "results/stats_example_level_mech_v9.json"), "w"), indent=1)
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    names = {"pqc_gain": r"\PQC{} macro F1 gain (\prot{Calibrated} $-$ \prot{Corrected})",
             "mlp_shift": r"\MLP{} macro F1 shift (\prot{Calibrated} $-$ \prot{Corrected})",
             "threshold_norm": r"Threshold normalisation (\PQC{} pos\_pred\_rate, \prot{Corrected} $-$ \prot{Calibrated})",
             "interaction": r"Head $\times$ protocol interaction (calibrated gap $-$ uncalibrated gap)"}
    lines = [
        "%% Auto-generated by scripts/stats_example_level_mech_v9.py -- do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{The eight calibration-mechanism contrasts of \S\ref{sec:pitfall2} (depthwise reducer, $q = 8$, seeds 0--7) with the development item as the sampling unit, as in Table~\ref{tab:example_level_v9}. The seed-level paired $p$ is the value reported in \S\ref{sec:pitfall2}; the hierarchical 95\% CI resamples seeds and then items; $p_{\mathrm{perm}}$ swaps the two arms' predictions for an item in every seed and is defined for the two-arm contrasts only.}",
        r"\label{tab:example_level_mech_v9}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Contrast & value & seed-level $p$ & 95\% CI (seeds $\times$ items) & $p_{\mathrm{perm}}$ \\",
        r"\midrule",
    ]
    task = {"mrpc": r"\MRPC{}", "paws": r"\PAWS{}"}
    for t in ("mrpc", "paws"):
        for k in ("pqc_gain", "mlp_shift", "threshold_norm", "interaction"):
            v = results[t][k]
            lines.append(f'{task[t]} & {names[k]} & {v["value"]:+.4f} & {v["seed_paired_t_p"]:.4f} & [{v["hier_boot_ci95"][0]:+.4f}, {v["hier_boot_ci95"][1]:+.4f}] & '
                         + (f'{v["perm_p"]:.4f}' if "perm_p" in v else "---") + r" \\")
        if t == "mrpc":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(ROOT, "paper/table_example_level_mech_v9.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("wrote results/stats_example_level_mech_v9.json and paper/table_example_level_mech_v9.tex")


if __name__ == "__main__":
    main()
