#!/usr/bin/env python3
"""Single source of truth for every headline number quoted in paper/main_v6.tex.

Recomputes each from raw per-run JSONs so the paper can be diffed against it.
Cells are keyed by (task, reducer, head, n_qubits, n_layers, run_tag,
freeze_calib) -- merging across n_layers or run_tag would silently pool the
mechanism-sweep runs into the primary cells.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load(*dirs):
    cells = defaultdict(dict)
    for sub in dirs:
        for f in sorted((ROOT / "results" / sub).glob("*.json")):
            if not f.name.startswith("hybrid_"):
                continue
            d = json.load(open(f))
            c = d["config"]
            ver = f.name.split("_")[1]
            key = (ver, c["task"], c["reducer"], c["head"], c.get("n_qubits", 8),
                   c.get("n_layers", 2), c.get("run_tag", ""), bool(c.get("freeze_calib", False)))
            cells[key][c["seed"]] = {
                "f1": d["phase2"]["test"]["macro_f1"],
                "pp": d["phase2"]["test"]["pos_pred_rate"],
                "val": d["phase2"]["best_val_macro_f1"],
                "hp": d.get("head_params"),
            }
    return cells


def arr(cells, key, field="f1", seeds=None):
    if key not in cells:
        return None
    d = cells[key]
    ss = sorted(d) if seeds is None else [s for s in seeds if s in d]
    return np.array([d[s][field] for s in ss])


def welch(a, b):
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = b[rng.integers(0, len(b), (n, len(b)))].mean(1) - a[rng.integers(0, len(a), (n, len(a)))].mean(1)
    return np.percentile(g, [2.5, 97.5])


C = load("v4", "v5", "v7")
P = lambda *a: print(*a)

P("=" * 78)
P("PRIMARY CELLS (depthwise, q=8, L=2) -- v4 uncalibrated vs v5 calibrated")
P("=" * 78)
P(f"{'task':<6}{'ver':<5}{'PQC f1':>16}{'MLP f1':>16}{'gap':>9}{'welch p':>10}{'PQC pp':>8}")
dec = {}
for task in ("mrpc", "paws", "qqp"):
    for ver in ("v4", "v5"):
        kp = (ver, task, "depthwise", "pqc", 8, 2, "", False)
        km = (ver, task, "depthwise", "mlp", 8, 2, "", False)
        p, m = arr(C, kp), arr(C, km)
        if p is None or m is None:
            continue
        pp = arr(C, kp, "pp")
        gap = m.mean() - p.mean()
        dec.setdefault(task, {})[ver] = gap
        P(f"{task:<6}{ver:<5}{p.mean():>9.4f}±{p.std():.3f}{m.mean():>9.4f}±{m.std():.3f}"
          f"{gap:>+9.4f}{welch(p, m):>10.4f}{pp.mean():>8.3f}   n={len(p)}/{len(m)}")

P("\n--- decomposition: fraction of the v4 gap removed by calibration ---")
for t, d in dec.items():
    if "v4" in d and "v5" in d:
        P(f"  {t:<5} {d['v4']:+.4f} -> {d['v5']:+.4f}   removed {100*(1-d['v5']/d['v4']):.0f}%")

P("\n--- PAWS 16-seed (post-hoc expansion) ---")
kp = ("v5", "paws", "depthwise", "pqc", 8, 2, "", False)
km = ("v5", "paws", "depthwise", "mlp", 8, 2, "", False)
p, m = arr(C, kp), arr(C, km)
lo, hi = boot(p, m)
P(f"  n={len(p)}/{len(m)}  gap={m.mean()-p.mean():+.4f}  p={welch(p,m):.4f}  CI=[{lo:+.4f},{hi:+.4f}]")
p8, m8 = arr(C, kp, seeds=range(8)), arr(C, km, seeds=range(8))
P(f"  registered n=8 subset: gap={m8.mean()-p8.mean():+.4f}  p={welch(p8,m8):.4f}")

P("\n--- per-head v4->v5 change + interaction (pre-registered seeds 0-7) ---")
for task in ("mrpc", "paws"):
    a4 = arr(C, ("v4", task, "depthwise", "pqc", 8, 2, "", False), seeds=range(8))
    a5 = arr(C, ("v5", task, "depthwise", "pqc", 8, 2, "", False), seeds=range(8))
    b4 = arr(C, ("v4", task, "depthwise", "mlp", 8, 2, "", False), seeds=range(8))
    b5 = arr(C, ("v5", task, "depthwise", "mlp", 8, 2, "", False), seeds=range(8))
    p4 = arr(C, ("v4", task, "depthwise", "pqc", 8, 2, "", False), "pp", range(8))
    p5 = arr(C, ("v5", task, "depthwise", "pqc", 8, 2, "", False), "pp", range(8))
    P(f"  {task}: PQC {a4.mean():.4f}->{a5.mean():.4f} p={welch(a4,a5):.4f} | "
      f"MLP {b4.mean():.4f}->{b5.mean():.4f} p={welch(b4,b5):.4f} | "
      f"interaction p={welch(a5-a4, b5-b4):.4f} | pp {p4.mean():.3f}->{p5.mean():.3f} p={welch(p4,p5):.5f}")

P("\n" + "=" * 78)
P("E1 FOURIER DEPTH (v7, calibrated) -- gap vs MLP h6")
P("=" * 78)
for task in ("qqp", "paws"):
    mh6 = arr(C, ("v7", task, "depthwise", "mlp", 8, 2, "fourierMLPh6", False))
    for L, tag in ((1, "fourierL1"), (2, ""), (4, "fourierL4")):
        ver = "v5" if L == 2 else "v7"
        pq = arr(C, (ver, task, "depthwise", "pqc", 8, L, tag, False))
        if pq is None or mh6 is None:
            continue
        lo, hi = boot(pq, mh6)
        P(f"  {task:<5} L={L}  PQC={pq.mean():.4f}±{pq.std():.3f} (n={len(pq)})  "
          f"gap={mh6.mean()-pq.mean():+.4f}  p={welch(pq,mh6):.4f}  CI=[{lo:+.4f},{hi:+.4f}]")
    P(f"  {task:<5} MLP_h6={mh6.mean():.4f}±{mh6.std():.3f} (n={len(mh6)})")

P("\n" + "=" * 78)
P("E2 SEL ANSATZ (v7, MRPC depthwise L=1)")
P("=" * 78)
for lab, frz, tag in (("uncal", True, "uncal"), ("cal", False, "cal")):
    s = arr(C, ("v7", "mrpc", "depthwise", "sel", 8, 1, tag, frz))
    spp = arr(C, ("v7", "mrpc", "depthwise", "sel", 8, 1, tag, frz), "pp")
    mver = "v4" if frz else "v5"
    m = arr(C, (mver, "mrpc", "depthwise", "mlp", 8, 2, "", False))
    if s is None:
        continue
    P(f"  SEL {lab:<6} f1={s.mean():.4f}±{s.std():.4f}  pp={spp.mean():.4f}  "
      f"gap vs {mver} MLP={m.mean()-s.mean():+.4f}  p={welch(s,m):.4f}  n={len(s)}")
su = arr(C, ("v7", "mrpc", "depthwise", "sel", 8, 1, "uncal", True))
sc = arr(C, ("v7", "mrpc", "depthwise", "sel", 8, 1, "cal", False))
if su is not None and sc is not None:
    pu = arr(C, ("v7", "mrpc", "depthwise", "sel", 8, 1, "uncal", True), "pp")
    pc = arr(C, ("v7", "mrpc", "depthwise", "sel", 8, 1, "cal", False), "pp")
    P(f"  SEL uncal->cal: f1 {sc.mean()-su.mean():+.4f} p={welch(su,sc):.4f} | "
      f"pp {pc.mean()-pu.mean():+.4f} p={welch(pu,pc):.4f}")

P("\n" + "=" * 78)
P("E4 INIT-SCALE (v7, uncalibrated init 1.0, 32 params)")
P("=" * 78)
for task in ("mrpc", "paws"):
    e4 = arr(C, ("v7", task, "depthwise", "pqc", 8, 2, "init1p0uncal", True))
    e4p = arr(C, ("v7", task, "depthwise", "pqc", 8, 2, "init1p0uncal", True), "pp")
    e4v = arr(C, ("v7", task, "depthwise", "pqc", 8, 2, "init1p0uncal", True), "val")
    m4 = arr(C, ("v4", task, "depthwise", "mlp", 8, 2, "", False))
    v4v = arr(C, ("v4", task, "depthwise", "pqc", 8, 2, "", False), "val")
    v5v = arr(C, ("v5", task, "depthwise", "pqc", 8, 2, "", False), "val")
    if e4 is None:
        continue
    P(f"  {task}: f1={e4.mean():.4f}±{e4.std():.4f}  pp={e4p.mean():.3f}  "
      f"gap vs MLP={m4.mean()-e4.mean():+.4f}  p={welch(e4,m4):.4f}  n={len(e4)}")
    P(f"        best_val: v4(init0.1)={v4v.mean():.4f}  v5(cal)={v5v.mean():.4f}  E4(init1.0)={e4v.mean():.4f}")

P("\n" + "=" * 78)
P("E3 NOISE (8 seeds) + READOUT PROBE")
P("=" * 78)
nz = json.load(open(ROOT / "results/noise_proxy_v7_multiseed.json"))
per = {s: v["results"] for s, v in nz["per_seed"].items()}
for pv in ("0.0", "0.001", "0.005", "0.01", "0.05", "0.1"):
    v = [per[s][pv]["macro_f1"] for s in per if pv in per[s]]
    pp = [per[s][pv]["pos_pred_rate"] for s in per if pv in per[s]]
    P(f"  p={pv:<6} f1={np.mean(v):.4f}±{np.std(v):.4f}  min={min(v):.4f} max={max(v):.4f}  pp={np.mean(pp):.3f}")
viol = [s for s in per if per[s]["0.0"]["macro_f1"] - per[s]["0.01"]["macro_f1"] >= 0.02]
P(f"  seeds violating 0.02 bound at p=1e-2: {sorted(viol)} ({len(viol)}/{len(per)})")
ro = json.load(open(ROOT / "results/readout_offset_v7.json"))
for h, r in ro["heads"].items():
    P(f"  {h:<22} mean<Z0>={r['mean_Z0']:+.4f}  std={r['std_Z0_across_seeds']:.4f}  init_pp={r['init_pos_pred_rate']:.3f}")
