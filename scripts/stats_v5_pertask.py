#!/usr/bin/env python3
"""Per-head v4-vs-v5 tests, interaction (diff-in-diff), pp_rate shift, TOST.

Supplies the exact numbers for the per-head response paragraph in main_v5.tex.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load(version):
    out = {}
    paths = sorted((ROOT / f"results/{version}").glob(f"hybrid_{version}_*.json"))
    if version == "v5":
        paths += sorted((ROOT / "results/v4").glob("hybrid_v5_*.json"))
    for f in paths:
        d = json.load(open(f))
        c = d["config"]
        if c["reducer"] != "depthwise":
            continue
        key = (c["task"], c["head"])
        out.setdefault(key, {})[c["seed"]] = {
            "macro_f1": d["phase2"]["test"]["macro_f1"],
            "pp": d["phase2"]["test"]["pos_pred_rate"],
            "p1_val": d["phase1"].get("val", {}).get("macro_f1"),
        }
    return out


def arr(d, key, field):
    seeds = sorted(d[key])
    return np.array([d[key][s][field] for s in seeds]), seeds


v4, v5 = load("v4"), load("v5")

for task in ("mrpc", "paws"):
    print(f"\n=== {task.upper()} (depthwise) ===")
    p4, s4 = arr(v4, (task, "pqc"), "macro_f1")
    p5, s5 = arr(v5, (task, "pqc"), "macro_f1")
    m4, _ = arr(v4, (task, "mlp"), "macro_f1")
    m5, _ = arr(v5, (task, "mlp"), "macro_f1")
    pp4, _ = arr(v4, (task, "pqc"), "pp")
    pp5, _ = arr(v5, (task, "pqc"), "pp")

    # pairing check: identical Phase-1 features per seed?
    pv4, _ = arr(v4, (task, "pqc"), "p1_val")
    pv5, _ = arr(v5, (task, "pqc"), "p1_val")
    paired_ok = s4 == s5 and np.allclose(pv4.astype(float), pv5.astype(float), atol=1e-9)
    print(f"seeds v4={s4} v5={s5}; phase1 identical per seed: {paired_ok}")

    _, p_pqc = stats.ttest_ind(p4, p5, equal_var=False)
    _, p_mlp = stats.ttest_ind(m4, m5, equal_var=False)
    _, p_pp = stats.ttest_ind(pp4, pp5, equal_var=False)
    print(f"PQC v4->v5: {p4.mean():.4f}->{p5.mean():.4f}  Welch p={p_pqc:.4f}")
    print(f"MLP v4->v5: {m4.mean():.4f}->{m5.mean():.4f}  Welch p={p_mlp:.4f}")
    print(f"PQC pp_rate v4->v5: {pp4.mean():.3f}->{pp5.mean():.3f}  Welch p={p_pp:.5f}")

    if paired_ok:
        _, p_pqc_pair = stats.ttest_rel(p5, p4)
        d_pqc, d_mlp = p5 - p4, m5 - m4
        _, p_inter = stats.ttest_ind(d_pqc, d_mlp, equal_var=False)
        _, p_pp_pair = stats.ttest_rel(pp5, pp4)
        print(f"paired PQC improvement: {d_pqc.mean():+.4f} p={p_pqc_pair:.4f}")
        print(f"paired PQC pp_rate shift p={p_pp_pair:.5f}")
        print(f"interaction (PQC delta vs MLP delta): "
              f"{d_pqc.mean():+.4f} vs {d_mlp.mean():+.4f}  Welch p={p_inter:.4f}")

    # TOST-style 90% CI on the v5 MLP-PQC gap (Welch-Satterthwaite df)
    diff = m5.mean() - p5.mean()
    se = np.sqrt(m5.var(ddof=1) / 8 + p5.var(ddof=1) / 8)
    df = (m5.var(ddof=1) / 8 + p5.var(ddof=1) / 8) ** 2 / (
        (m5.var(ddof=1) / 8) ** 2 / 7 + (p5.var(ddof=1) / 8) ** 2 / 7)
    tcrit = stats.t.ppf(0.95, df)
    lo, hi = diff - tcrit * se, diff + tcrit * se
    within = lo > -0.015 and hi < 0.015
    print(f"v5 gap 90% CI: [{lo:+.4f}, {hi:+.4f}] (df={df:.1f}); "
          f"within pre-reg margin +/-0.015: {within}")
