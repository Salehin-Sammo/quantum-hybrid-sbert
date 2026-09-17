#!/usr/bin/env python3
"""Paired re-analysis of every PQC-versus-MLP head comparison.

Within a seed both heads train on the same frozen features and are scored on
the same test partition (test_labels are asserted identical), so the natural
unit is the per-seed difference MLP - PQC. For each cell this script reports
the mean paired difference, a 10,000-resample bootstrap CI of the paired
differences, the paired t-test, the Wilcoxon signed-rank test, the paired
effect size d_z, and the number of seeds with MLP > PQC. Holm correction is
applied over the same twelve primary cells as table_stats_v7; the two q=16
cells are post hoc and reported outside that family. Welch p is repeated for
comparison with the submitted analysis.

Run from the repository root:
    python scripts/stats_paired_v9.py
Writes results/stats_paired_v9.json and paper/table_paired_v9.tex.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "scripts" in os.path.abspath(__file__) else os.getcwd()
if len(sys.argv) > 1:
    ROOT = sys.argv[1]
N_BOOT = 10000


def load(paths):
    runs = defaultdict(lambda: defaultdict(dict))
    for f in paths:
        name = os.path.basename(f)
        m = re.match(r"^hybrid_(v\d+)_", name)
        if not m:
            continue
        d = json.load(open(f))
        c = d["config"]
        if c.get("n_layers", 2) != 2 or c["head"] not in ("pqc", "mlp"):
            continue
        ver = m.group(1)
        key = (ver, c["task"], c["reducer"], c["n_qubits"])
        runs[key][c["head"]][c["seed"]] = d
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


def cell_stats(pqc, mlp, seeds, rng_seed=0):
    for s in seeds:
        assert pqc[s]["test_labels"] == mlp[s]["test_labels"], f"seed {s}: test labels differ"
    f_p = np.array([pqc[s]["phase2"]["test"]["macro_f1"] for s in seeds])
    f_m = np.array([mlp[s]["phase2"]["test"]["macro_f1"] for s in seeds])
    a_p = np.array([roc_auc_score(pqc[s]["test_labels"], pqc[s]["test_logits"]) for s in seeds])
    a_m = np.array([roc_auc_score(mlp[s]["test_labels"], mlp[s]["test_logits"]) for s in seeds])
    d = f_m - f_p
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return {
        "n": len(seeds),
        "delta_macro_f1": float(d.mean()),
        "paired_ci95": [float(lo), float(hi)],
        "paired_t_p": float(stats.ttest_rel(f_m, f_p).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(d).pvalue),
        "welch_p": float(stats.ttest_ind(f_m, f_p, equal_var=False).pvalue),
        "d_z": float(d.mean() / d.std(ddof=1)),
        "seeds_mlp_gt_pqc": int((d > 0).sum()),
        "delta_auc": float((a_m - a_p).mean()),
        "paired_t_p_auc": float(stats.ttest_rel(a_m, a_p).pvalue),
        "welch_p_auc": float(stats.ttest_ind(a_m, a_p, equal_var=False).pvalue),
    }


def main():
    primary = load(sorted(glob.glob(os.path.join(ROOT, "results/v4/*.json"))) + sorted(glob.glob(os.path.join(ROOT, "results/v5/*.json"))))
    posthoc = load(sorted(glob.glob(os.path.join(ROOT, "results/v8/*q16both*.json"))))
    rows = []
    for key in sorted(primary):
        h = primary[key]
        if "pqc" in h and "mlp" in h:
            seeds = sorted(set(h["pqc"]) & set(h["mlp"]))
            rows.append({"version": key[0], "task": key[1], "reducer": key[2], "n_qubits": key[3], "family": "primary", **cell_stats(h["pqc"], h["mlp"], seeds)})
    adj = holm([r["paired_t_p"] for r in rows])
    for r, a in zip(rows, adj):
        r["paired_t_p_holm"] = float(a)
    # registered 8-seed PAWS calibrated cell, reported separately (the 16-seed cell is the family member)
    h = primary[("v5", "paws", "depthwise", 8)]
    rows.append({"version": "v5", "task": "paws", "reducer": "depthwise", "n_qubits": 8, "family": "pre-specified-8-seed", **cell_stats(h["pqc"], h["mlp"], list(range(8)))})
    for key in sorted(posthoc):
        h = posthoc[key]
        seeds = sorted(set(h["pqc"]) & set(h["mlp"]))
        rows.append({"version": "v8", "task": key[1], "reducer": key[2], "n_qubits": key[3], "family": "post hoc q=16", **cell_stats(h["pqc"], h["mlp"], seeds)})

    # one family over every head comparison in the table (fifteen cells)
    for r, a in zip(rows, holm([r["paired_t_p"] for r in rows])):
        r["paired_t_p_holm_all"] = float(a)
    for r, a in zip(rows, holm([r["welch_p"] for r in rows])):
        r["welch_p_holm_all"] = float(a)

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "paper"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "results/stats_paired_v9.json"), "w"), indent=1)

    task = {"mrpc": r"\MRPC{}", "paws": r"\PAWS{}", "qqp": r"\QQP{}"}
    lines = [
        "%% Auto-generated by scripts/stats_paired_v9.py -- do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired re-analysis of every \PQC{}-versus-\MLP{} head comparison. Within a seed both heads consume identical frozen features and are scored on an identical test partition, so each cell is analysed on the per-seed differences $\Delta_s = \text{macro F1}(\MLP{}) - \text{macro F1}(\PQC{})$. The 95\% CI is a 10{,}000-resample bootstrap of the mean paired difference; $p_{\mathrm{paired}}$ is a paired $t$-test, $p_{\mathrm{Holm}}$ its Holm--Bonferroni adjustment over the same twelve primary cells as Table~\ref{tab:stats_rigor_v7}, and $p_{\mathrm{Holm}}^{15}$ its adjustment over all fifteen cells of this table as one family; $W$ is the Wilcoxon signed-rank $p$; $d_z$ is the paired effect size; $+$ counts the seeds in which the \MLP{} scores higher. The Welch $p$ of Table~\ref{tab:stats_rigor_v7} is repeated for comparison. The pre-specified 8-seed \PAWS{} cell and the two post hoc $q = 16$ cells sit outside the twelve-cell family; v8 is the repository label of the $q = 16$ runs, which follow the \prot{Calibrated} protocol. $*$ marks an adjusted $p < 0.05$. The \MRPC{} caveat of \S\ref{sec:results} applies unchanged. Its seeds share about half their test items, so the paired differences there are not independent and the tests are anti-conservative.}",
        r"\label{tab:paired_v9}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccccccccccc}",
        r"\toprule",
        r"Task & Reducer & Ver & $q$ & $n$ & $\Delta$ & 95\% CI (paired) & $p_{\mathrm{paired}}$ & $p_{\mathrm{Holm}}$ & $p_{\mathrm{Holm}}^{15}$ & $W$ & $d_z$ & $+$ & Welch $p$ \\",
        r"\midrule",
    ]
    for r in rows:
        if r["family"] == "pre-specified-8-seed":
            lines.append(r"\midrule")
        ver = {"v4": "v4", "v5": "v5", "v8": "v8"}[r["version"]]
        if r["family"] == "pre-specified-8-seed":
            ver = "v5 (8 s.)"
        holm_txt = f'{r["paired_t_p_holm"]:.4f}' + ("*" if r["paired_t_p_holm"] < 0.05 else "") if "paired_t_p_holm" in r else "outside family"
        lines.append(
            f'{task[r["task"]]} & {r["reducer"]} & {ver} & {r["n_qubits"]} & {r["n"]} & {r["delta_macro_f1"]:+.4f} & '
            f'[{r["paired_ci95"][0]:+.4f}, {r["paired_ci95"][1]:+.4f}] & {r["paired_t_p"]:.4f} & {holm_txt} & '
            f'{r["paired_t_p_holm_all"]:.4f}{"*" if r["paired_t_p_holm_all"] < 0.05 else ""} & '
            f'{r["wilcoxon_p"]:.4f} & {r["d_z"]:.2f} & {r["seeds_mlp_gt_pqc"]}/{r["n"]} & {r["welch_p"]:.4f} \\\\'
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(ROOT, "paper/table_paired_v9.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    for r in rows:
        print(f'{r["version"]:<3} {r["task"]:<5} {r["reducer"]:<10} q{r["n_qubits"]:<3} n{r["n"]:<3} {r["family"]:<18} '
              f'dF1 {r["delta_macro_f1"]:+.4f} CI [{r["paired_ci95"][0]:+.4f},{r["paired_ci95"][1]:+.4f}] '
              f'p_paired {r["paired_t_p"]:.4f} holm {r.get("paired_t_p_holm", float("nan")):.4f} W {r["wilcoxon_p"]:.4f} '
              f'dz {r["d_z"]:.2f} +{r["seeds_mlp_gt_pqc"]} welch {r["welch_p"]:.4f} | dAUC {r["delta_auc"]:+.3f} p_paired {r["paired_t_p_auc"]:.4f} welch {r["welch_p_auc"]:.4f}')
    print("wrote results/stats_paired_v9.json and paper/table_paired_v9.tex")


if __name__ == "__main__":
    main()
