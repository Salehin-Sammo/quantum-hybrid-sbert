#!/usr/bin/env python3
"""Stats rigor pass over v4+v5 hybrid results: Welch t / Mann-Whitney U /
Cohen's d / bootstrap CI for PQC vs MLP at matched (version, task, reducer,
n_qubits), Holm-Bonferroni correction across all cells, and a post-hoc power
estimate.

Dynamic by design: groups by whatever (task, reducer, n_qubits) combos and
seed counts exist on disk right now -- no hardcoded task/seed lists, so it's
safe to re-run as the background sweep drops more files into results/v5/.

Outputs: results/stats_rigor_v6.json, paper/table_stats_v6.tex.
"""
from __future__ import annotations
import json, re, math
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FNAME_VERSION = re.compile(r"^hybrid_(v\d+)_")


def load_runs():
    """(version, task, reducer, head, n_qubits) -> list of {macro_f1, pos_pred_rate}.

    Version is parsed from the filename, not the containing folder: a known
    path bug occasionally drops a v5 run into results/v4/ (see
    scripts/analyse_v5.py's comment on this), so trusting the directory would
    silently mis-bucket it into the wrong cell -- or into v4 when it's
    actually a missing v5 seed.
    """
    groups = defaultdict(list)
    seen = {}
    files = sorted(ROOT.glob("results/v4/*.json")) + sorted(ROOT.glob("results/v5/*.json"))
    for f in files:
        m = FNAME_VERSION.match(f.name)
        if not m:
            continue
        version = m.group(1)
        d = json.load(open(f))
        c, test = d["config"], d["phase2"]["test"]
        # Main table covers only the primary l=2 pqc/mlp cells. Ignore any
        # v7 mechanism-sweep run (other layer counts, the sel ansatz) that may
        # land in results/v5/ so it can never collide with a primary cell.
        if c.get("n_layers", 2) != 2 or c["head"] not in ("pqc", "mlp"):
            continue
        key = (version, c["task"], c["reducer"], c["head"], c["n_qubits"])
        dedup_key = key + (c["seed"],)
        if dedup_key in seen:
            print(f"  [warn] duplicate seed for {dedup_key}: keeping {seen[dedup_key]}, ignoring {f.name}")
            continue
        seen[dedup_key] = f.name
        groups[key].append({
            "macro_f1": test.get("macro_f1", float("nan")),
            "pos_pred_rate": test.get("pos_pred_rate", float("nan")),
        })
    return groups


def cohens_d(pqc, mlp):
    n1, n2 = len(pqc), len(mlp)
    pooled_sd = math.sqrt(((n1 - 1) * pqc.var(ddof=1) + (n2 - 1) * mlp.var(ddof=1)) / (n1 + n2 - 2))
    return (mlp.mean() - pqc.mean()) / pooled_sd if pooled_sd > 0 else float("nan")


def bootstrap_gap_ci(pqc, mlp, n_resamples=10000, seed=0):
    """95% CI of mean(mlp) - mean(pqc) via independent resampling of each arm.

    Fresh RNG per call rather than one shared stream: a cell's CI must stay
    reproducible regardless of how many other cells get processed before it,
    since new cells (tasks/q values) keep landing from the live sweep.
    """
    rng = np.random.default_rng(seed)
    idx_pqc = rng.integers(0, len(pqc), size=(n_resamples, len(pqc)))
    idx_mlp = rng.integers(0, len(mlp), size=(n_resamples, len(mlp)))
    gaps = mlp[idx_mlp].mean(axis=1) - pqc[idx_pqc].mean(axis=1)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)


def holm_bonferroni(pvals):
    """Standard Holm step-down. Returns adjusted p-values in the input order."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min((n - rank) * pvals[i], 1.0))
        adj[i] = running_max
    return adj


def power_n(d, power=0.80, alpha_z=0.95):
    """n_seeds/arm for `power` at observed |d|, per the spec's own formula:
    n ~ ((z_alpha + z_power) / d)^2. This is the approximate heuristic given,
    not the textbook two-sample n = 2*(...)^2 formula -- kept as specified.
    """
    if not np.isfinite(d) or d == 0:
        return float("inf")
    z = stats.norm.ppf(alpha_z) + stats.norm.ppf(power)
    return math.ceil((z / abs(d)) ** 2)


def analyze(groups):
    cells = defaultdict(dict)
    for (version, task, reducer, head, nq), rows in groups.items():
        cells[(version, task, reducer, nq)][head] = rows

    results = []
    for key in sorted(cells):
        version, task, reducer, nq = key
        heads = cells[key]
        if "pqc" not in heads or "mlp" not in heads:
            continue
        pqc = np.array([r["macro_f1"] for r in heads["pqc"]])
        mlp = np.array([r["macro_f1"] for r in heads["mlp"]])
        if len(pqc) < 2 or len(mlp) < 2:
            print(f"  [skip] {version}/{task}/{reducer}/q{nq}: n_pqc={len(pqc)} n_mlp={len(mlp)} (<2 seeds, no test)")
            continue
        pqc_pp = np.array([r["pos_pred_rate"] for r in heads["pqc"]])
        mlp_pp = np.array([r["pos_pred_rate"] for r in heads["mlp"]])

        t_stat, p_welch = stats.ttest_ind(mlp, pqc, equal_var=False)
        u_stat, p_mwu = stats.mannwhitneyu(mlp, pqc, alternative="two-sided")
        d = cohens_d(pqc, mlp)
        ci_lo, ci_hi = bootstrap_gap_ci(pqc, mlp)

        results.append({
            "version": version, "task": task, "reducer": reducer, "n_qubits": int(nq),
            "n_seeds_pqc": len(pqc), "n_seeds_mlp": len(mlp),
            "pqc_macro_f1_mean": float(pqc.mean()), "pqc_macro_f1_std": float(pqc.std(ddof=1)),
            "mlp_macro_f1_mean": float(mlp.mean()), "mlp_macro_f1_std": float(mlp.std(ddof=1)),
            "delta_mlp_minus_pqc": float(mlp.mean() - pqc.mean()),
            "bootstrap_ci95": [ci_lo, ci_hi],
            "welch_t": float(t_stat), "welch_p_raw": float(p_welch),
            "mannwhitney_u": float(u_stat), "mannwhitney_p": float(p_mwu),
            "cohens_d": float(d),
            "n_seeds_for_80pct_power": power_n(d),
            "pqc_pos_pred_rate_mean": float(pqc_pp.mean()),
            "mlp_pos_pred_rate_mean": float(mlp_pp.mean()),
        })

    p_raw = np.array([r["welch_p_raw"] for r in results])
    p_holm = holm_bonferroni(p_raw) if len(p_raw) else p_raw
    for r, p in zip(results, p_holm):
        r["welch_p_holm"] = float(p)
    return results


def write_latex_table(results, out_path):
    lines = [
        r"% Auto-generated by scripts/stats_rigor_v6.py -- do not hand-edit.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{PQC vs.\ MLP head, test macro F1, matched reducer and qubit count. "
        r"$\Delta$ = MLP$-$PQC mean; 95\% CI is a 10{,}000-resample bootstrap of $\Delta$; "
        r"$p_\mathrm{Holm}$ is Holm--Bonferroni adjusted across all cells in this table. "
        r"$*$ marks $p_\mathrm{Holm}<0.05$.}",
        r"\label{tab:stats_rigor_v6}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllccccccc}",
        r"\toprule",
        r"Task & Reducer & Ver & $q$ & $n$ & $\Delta$(MLP$-$PQC) & 95\% CI & $p_\mathrm{raw}$ & $p_\mathrm{Holm}$ & $d$ \\",
        r"\midrule",
    ]
    for r in results:
        n = str(r["n_seeds_pqc"]) if r["n_seeds_pqc"] == r["n_seeds_mlp"] else f'{r["n_seeds_pqc"]}/{r["n_seeds_mlp"]}'
        sig = "*" if r["welch_p_holm"] < 0.05 else ""
        lines.append(
            f'\\textsc{{{r["task"]}}} & {r["reducer"]} & {r["version"]} & {r["n_qubits"]} & {n} & '
            f'{r["delta_mlp_minus_pqc"]:+.4f} & '
            f'[{r["bootstrap_ci95"][0]:+.4f}, {r["bootstrap_ci95"][1]:+.4f}] & '
            f'{r["welch_p_raw"]:.4f} & {r["welch_p_holm"]:.4f}{sig} & {r["cohens_d"]:.3f} \\\\'
        )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def print_summary(results):
    print("=" * 112)
    print("STATS RIGOR v6 -- PQC vs MLP, test macro F1, dynamic grouping over results/v4 + results/v5")
    print("=" * 112)
    print(f'{"task":<6} {"reducer":<10} {"ver":<4} {"q":<3} {"n":<6} {"delta(MLP-PQC)":<16} '
          f'{"95% CI":<22} {"p_raw":<9} {"p_Holm":<10} {"d":<7}')
    print("-" * 112)
    for r in results:
        n = str(r["n_seeds_pqc"]) if r["n_seeds_pqc"] == r["n_seeds_mlp"] else f'{r["n_seeds_pqc"]}/{r["n_seeds_mlp"]}'
        ci = f'[{r["bootstrap_ci95"][0]:+.4f},{r["bootstrap_ci95"][1]:+.4f}]'
        sig = "*" if r["welch_p_holm"] < 0.05 else " "
        print(f'{r["task"]:<6} {r["reducer"]:<10} {r["version"]:<4} {r["n_qubits"]:<3} {n:<6} '
              f'{r["delta_mlp_minus_pqc"]:+.4f}           {ci:<22} {r["welch_p_raw"]:.4f}    '
              f'{r["welch_p_holm"]:.4f}{sig}  {r["cohens_d"]:+.3f}')
    n_sig = sum(1 for r in results if r["welch_p_holm"] < 0.05)
    print("-" * 112)
    print(f"{n_sig}/{len(results)} cells remain significant after Holm-Bonferroni (p_Holm < 0.05)")
    print("\nPower (n_seeds/arm needed for 80% power at observed |d|):")
    for r in results:
        print(f'  {r["task"]:<6} {r["reducer"]:<10} {r["version"]:<4} q{r["n_qubits"]:<3} '
              f'd={r["cohens_d"]:+.3f}  n_needed={r["n_seeds_for_80pct_power"]}  '
              f'(have pqc={r["n_seeds_pqc"]}, mlp={r["n_seeds_mlp"]})')


def main():
    groups = load_runs()
    results = analyze(groups)

    out_json = ROOT / "results/stats_rigor_v6.json"
    out_json.write_text(json.dumps(results, indent=2))
    out_tex = ROOT / "paper/table_stats_v6.tex"
    write_latex_table(results, out_tex)

    print_summary(results)
    print(f"\nWrote {out_json.relative_to(ROOT)} and {out_tex.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
