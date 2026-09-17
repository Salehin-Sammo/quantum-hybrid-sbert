#!/usr/bin/env python3
"""Example-level re-analysis of every PQC-versus-MLP head comparison.

Across seeds the validation/test partition is re-drawn from the same
development split, so per-seed scores share test items (about half of them on
MRPC, nearly all of them on PAWS and QQP) and are not independent
observations. This script rebuilds each seed's test-item identities from the
seeded shuffle in src/datasets_qec.py (checked by cross-seed label agreement,
which must be exact) and then treats the development item as the sampling
unit:

  * cluster bootstrap over items: resample development items with
    replacement; every seed's test set is re-weighted by the same draw;
    statistic = mean over seeds of macro F1(MLP) - macro F1(PQC)
  * hierarchical bootstrap: resample seeds with replacement, then items
  * cluster permutation test: for each item, swap the two heads' predictions
    in every seed in which the item is tested; two-sided p on the same
    statistic

Holm correction over the same twelve primary cells as stats_paired_v9.py is
applied to the permutation p-values. Run from the repository root:
    python scripts/stats_example_level_v9.py
Writes results/stats_example_level_v9.json and paper/table_example_level_v9.tex.
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
N_BOOT = 10000
CHUNK = 500
DEV_SIZE = {"mrpc": 408, "paws": 8000, "qqp": 40430}  # GLUE mrpc, PAWS labeled_final, GLUE qqp validation splits


def test_ids(task, seed, n_val, n_test=None):
    """Replays src/train_hybrid_v*.py: seeded shuffle of the development split,
    first n_val items validation, remainder test, optional seeded subsample."""
    idx = list(range(DEV_SIZE[task]))
    random.Random(seed).shuffle(idx)
    te = idx[n_val:]
    if n_test is not None and len(te) > n_test:
        sel = np.random.default_rng(seed).choice(len(te), n_test, replace=False)
        te = [te[i] for i in sorted(sel)]
    return np.asarray(te)


def load(paths):
    runs = defaultdict(lambda: defaultdict(dict))
    for f in paths:
        m = re.match(r"^hybrid_(v\d+)_", os.path.basename(f))
        if not m:
            continue
        d = json.load(open(f))
        c = d["config"]
        if c.get("n_layers", 2) != 2 or c["head"] not in ("pqc", "mlp"):
            continue
        runs[(m.group(1), c["task"], c["reducer"], c["n_qubits"])][c["head"]][c["seed"]] = d
    return runs


def holm(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def macro_f1(c):
    """c[..., 4] = (tp, fp, fn, tn) -> macro F1 with F1 := 0 when its
    denominator is 0 (sklearn zero_division=0)."""
    c = np.asarray(c, np.float64)  # counts are exact integers in float32; divide in float64
    tp, fp, fn, tn = (c[..., i] for i in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        fpos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        fneg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return 0.5 * (fpos + fneg)


def cell(pqc, mlp, seeds, task, rng_seed=0):
    # rebuild ids, check labels agree across every run that tests an item
    ids, labels, preds = {}, {}, {"pqc": {}, "mlp": {}}
    for s in seeds:
        c = pqc[s]["config"]
        assert c["n_val"] == mlp[s]["config"]["n_val"]
        t = test_ids(task, s, c["n_val"], c.get("n_test"))
        assert len(t) == len(pqc[s]["test_labels"]) == len(mlp[s]["test_labels"]), "split length mismatch"
        assert pqc[s]["test_labels"] == mlp[s]["test_labels"], f"seed {s}: labels differ between heads"
        ids[s] = t
        labels[s] = np.asarray(pqc[s]["test_labels"], int)
        for h, r in (("pqc", pqc), ("mlp", mlp)):
            p = np.asarray(r[s]["test_predictions"], int)
            assert np.array_equal(p, (np.asarray(r[s]["test_logits"]) > 0).astype(int)), "stored predictions are not logit > 0"
            preds[h][s] = p
    seen = {}
    for s in seeds:
        for i, y in zip(ids[s], labels[s]):
            assert seen.setdefault(int(i), int(y)) == int(y), f"item {i}: label disagrees across seeds (id reconstruction failed)"
    pool = np.array(sorted(seen))
    row = {int(i): k for k, i in enumerate(pool)}
    S, N = len(seeds), len(pool)
    # indicator blocks: A[h] has shape (N, S, 4) with columns (tp, fp, fn, tn)
    A = {h: np.zeros((N, S, 4), np.float32) for h in ("pqc", "mlp")}
    for k, s in enumerate(seeds):
        r = np.array([row[int(i)] for i in ids[s]])
        y = labels[s]
        for h in ("pqc", "mlp"):
            p = preds[h][s]
            A[h][r, k, 0] = (p == 1) & (y == 1)
            A[h][r, k, 1] = (p == 1) & (y == 0)
            A[h][r, k, 2] = (p == 0) & (y == 1)
            A[h][r, k, 3] = (p == 0) & (y == 0)
    Ap, Am = A["pqc"].reshape(N, S * 4), A["mlp"].reshape(N, S * 4)
    base_p, base_m = Ap.sum(0).reshape(S, 4), Am.sum(0).reshape(S, 4)
    d_seed = macro_f1(base_m) - macro_f1(base_p)
    stored = np.array([mlp[s]["phase2"]["test"]["macro_f1"] - pqc[s]["phase2"]["test"]["macro_f1"] for s in seeds])
    assert np.allclose(d_seed, stored, atol=1e-9), "recomputed macro F1 does not match the stored values"
    d_obs = float(d_seed.mean())
    D = Ap - Am  # (N, S*4)

    rng = np.random.default_rng(rng_seed)
    boot_item, boot_hier, perm = [], [], []
    done = 0
    while done < N_BOOT:
        b = min(CHUNK, N_BOOT - done)
        # cluster bootstrap over items
        draws = rng.integers(0, N, size=(b, N))
        W = np.stack([np.bincount(dr, minlength=N) for dr in draws]).astype(np.float32)
        cp, cm = (W @ Ap).reshape(b, S, 4), (W @ Am).reshape(b, S, 4)
        d_ws = macro_f1(cm) - macro_f1(cp)  # (b, S)
        boot_item.extend(d_ws.mean(1).tolist())
        # hierarchical: seeds with replacement, then the same item draw
        sd = rng.integers(0, S, size=(b, S))
        M = np.stack([np.bincount(x, minlength=S) for x in sd]).astype(np.float32)
        boot_hier.extend(((d_ws * M).sum(1) / S).tolist())
        # cluster permutation: swap heads per item, consistently across seeds
        F = rng.integers(0, 2, size=(b, N)).astype(np.float32)
        FD = (F @ D).reshape(b, S, 4)
        d_perm = macro_f1(base_m[None] + FD) - macro_f1(base_p[None] - FD)
        perm.extend(d_perm.mean(1).tolist())
        done += b
    boot_item, boot_hier, perm = map(np.asarray, (boot_item, boot_hier, perm))
    return {
        "n": S,
        "n_items_pool": int(N),
        "mean_test_size": float(np.mean([len(ids[s]) for s in seeds])),
        "mean_pairwise_overlap": float(np.mean([len(set(ids[a].tolist()) & set(ids[b].tolist())) / len(ids[a]) for ai, a in enumerate(seeds) for b in seeds[ai + 1:]])) if S > 1 else 1.0,
        "delta_macro_f1": d_obs,
        "paired_t_p": float(stats.ttest_rel(macro_f1(base_m), macro_f1(base_p)).pvalue),
        "item_boot_ci95": [float(x) for x in np.percentile(boot_item, [2.5, 97.5])],
        "hier_boot_ci95": [float(x) for x in np.percentile(boot_hier, [2.5, 97.5])],
        "perm_p": float((1 + np.sum(np.abs(perm) >= abs(d_obs) - 1e-12)) / (N_BOOT + 1)),
    }


def main():
    primary = load(sorted(glob.glob(os.path.join(ROOT, "results/v4/*.json"))) + sorted(glob.glob(os.path.join(ROOT, "results/v5/*.json"))))
    posthoc = load(sorted(glob.glob(os.path.join(ROOT, "results/v8/*q16both*.json"))))
    rows = []
    for key in sorted(primary):
        h = primary[key]
        if "pqc" in h and "mlp" in h:
            seeds = sorted(set(h["pqc"]) & set(h["mlp"]))
            rows.append({"version": key[0], "task": key[1], "reducer": key[2], "n_qubits": key[3], "family": "primary", **cell(h["pqc"], h["mlp"], seeds, key[1])})
            print("done", key, flush=True)
    adj = holm([r["perm_p"] for r in rows])
    for r, a in zip(rows, adj):
        r["perm_p_holm"] = float(a)
    h = primary[("v5", "paws", "depthwise", 8)]
    rows.append({"version": "v5", "task": "paws", "reducer": "depthwise", "n_qubits": 8, "family": "pre-specified-8-seed", **cell(h["pqc"], h["mlp"], list(range(8)), "paws")})
    for key in sorted(posthoc):
        h = posthoc[key]
        seeds = sorted(set(h["pqc"]) & set(h["mlp"]))
        rows.append({"version": "v8", "task": key[1], "reducer": key[2], "n_qubits": key[3], "family": "post hoc q=16", **cell(h["pqc"], h["mlp"], seeds, key[1])})

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results/stats_example_level_v9.json"), "w"), indent=1)

    task = {"mrpc": r"\MRPC{}", "paws": r"\PAWS{}", "qqp": r"\QQP{}"}
    lines = [
        "%% Auto-generated by scripts/stats_example_level_v9.py -- do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Example-level re-analysis of the head comparisons of Table~\ref{tab:paired_v9}, treating the development item rather than the seed as the sampling unit. Each seed's test-item identities are rebuilt from the seeded shuffle of \S\ref{sec:results} and checked by exact label agreement across seeds; $\bar{o}$ is the mean fraction of test items two seeds share. $\Delta$ is the mean over seeds of macro F1(\MLP{}) $-$ macro F1(\PQC{}). The item CI resamples development items with replacement, re-weighting every seed's test set by the same draw (10{,}000 resamples); the hierarchical CI resamples seeds and then items. $p_{\mathrm{perm}}$ is a two-sided permutation test that swaps the two heads' predictions for an item in every seed in which it is tested, and $p_{\mathrm{Holm}}$ its adjustment over the twelve primary cells. The paired $t$ of Table~\ref{tab:paired_v9} is repeated for comparison. $*$ marks $p_{\mathrm{Holm}} < 0.05$.}",
        r"\label{tab:example_level_v9}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccccccccc}",
        r"\toprule",
        r"Task & Reducer & Ver & $q$ & $n$ & test items & $\bar{o}$ & $\Delta$ & 95\% CI (items) & 95\% CI (seeds $\times$ items) & $p_{\mathrm{perm}}$ & $p_{\mathrm{Holm}}$ / paired $t$ \\",
        r"\midrule",
    ]
    for r in rows:
        if r["family"] == "pre-specified-8-seed":
            lines.append(r"\midrule")
        ver = "v5 (8 s.)" if r["family"] == "pre-specified-8-seed" else r["version"]
        holm_txt = (f'{r["perm_p_holm"]:.4f}' + ("*" if r["perm_p_holm"] < 0.05 else "")) if "perm_p_holm" in r else "outside family"
        lines.append(
            f'{task[r["task"]]} & {r["reducer"]} & {ver} & {r["n_qubits"]} & {r["n"]} & {int(round(r["mean_test_size"])):,} & {r["mean_pairwise_overlap"]:.2f} & '
            f'{r["delta_macro_f1"]:+.4f} & [{r["item_boot_ci95"][0]:+.4f}, {r["item_boot_ci95"][1]:+.4f}] & '
            f'[{r["hier_boot_ci95"][0]:+.4f}, {r["hier_boot_ci95"][1]:+.4f}] & {r["perm_p"]:.4f} & {holm_txt} / {r["paired_t_p"]:.4f} \\\\'.replace(",", "{,}", 0)
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    txt = "\n".join(lines) + "\n"
    # thousands separator inside math-free cells: 7,800 -> 7{,}800
    txt = re.sub(r"(?<=\d),(?=\d{3})", "{,}", txt)
    open(os.path.join(ROOT, "paper/table_example_level_v9.tex"), "w", encoding="utf-8", newline="\n").write(txt)
    for r in rows:
        print(f'{r["version"]:<3} {r["task"]:<5} {r["reducer"]:<10} q{r["n_qubits"]:<3} n{r["n"]:<3} {r["family"]:<20} items {r["mean_test_size"]:7.0f} overlap {r["mean_pairwise_overlap"]:.3f} '
              f'd {r["delta_macro_f1"]:+.4f} item CI [{r["item_boot_ci95"][0]:+.4f},{r["item_boot_ci95"][1]:+.4f}] hier CI [{r["hier_boot_ci95"][0]:+.4f},{r["hier_boot_ci95"][1]:+.4f}] '
              f'perm p {r["perm_p"]:.4f} holm {r.get("perm_p_holm", float("nan")):.4f} paired t {r["paired_t_p"]:.4f}')
    print("wrote results/stats_example_level_v9.json and paper/table_example_level_v9.tex")


if __name__ == "__main__":
    main()
