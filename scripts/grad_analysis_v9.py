#!/usr/bin/env python3
"""Gradient analysis of the data-reuploading head at initialisation.

Complements the circuit-definition argument of the paper (at the zero feature
vector the readout equals +1 for every parameter value, so no parameter
direction shifts every input's readout by the same amount) with a numerical
first-order analysis on the same 200 frozen MRPC features and the same eight
initialisation seeds as scripts/readout_offset_v7.py:

  * per-parameter Jacobian of <Z_0> over the 200 inputs, which identifies the
    inert parameters (identically zero column) and, for the live ones, the mean,
    spread and sign consistency of their first-order effect across inputs;
  * the least-squares fit of a uniform unit shift of the readout by the
    Jacobian columns, i.e. how much of a threshold shift is achievable to first
    order by any parameter direction (0 residual = a bias-like direction exists);
  * the gradient of the mean readout, i.e. how fast training can move the
    offset at initialisation, against the MLP output bias whose effect on the
    pre-activation is exactly 1 for every input.

Run from the repository root: python scripts/grad_analysis_v9.py
Writes results/grad_analysis_v9.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_heads import PQCHead  # noqa: E402
from hybrid_reducer import DepthwiseSBERTReducer  # noqa: E402

DT = torch.float64
N_SEEDS, N_SAMPLES, N_QUBITS = 8, 200, 8


def real_features():
    """Load the released feature snapshot used for the reported analysis.

    Failing loudly is deliberate.  A synthetic fallback would silently change
    the experiment while still producing plausible-looking numbers.
    """
    path = ROOT / "results/mrpc_depthwise_q8_s0_features.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"missing released feature snapshot: {path.relative_to(ROOT)}"
        )
    z = np.load(path)
    if z.shape != (N_SAMPLES, N_QUBITS):
        raise ValueError(
            f"unexpected feature shape {z.shape}, expected {(N_SAMPLES, N_QUBITS)}"
        )
    return torch.from_numpy(z).to(DT)


def jacobian(head, x):
    J, f = [], []
    for i in range(x.shape[0]):
        head.zero_grad()
        out = head(x[i:i + 1])[0]
        out.backward()
        J.append(head.params.grad.detach().clone().numpy())
        f.append(float(out))
    return np.array(J), np.array(f)


def main():
    x = real_features()
    source = "released real frozen depthwise reducer features (MRPC seed 0)"
    print("features:", source, tuple(x.shape))
    ref = json.load(open(ROOT / "results/readout_offset_v7.json"))["heads"]["PQC_reuploading_L2"]["mean_Z0_per_seed"]

    out = {"feature_source": source, "n_samples": N_SAMPLES, "n_seeds": N_SEEDS, "L2": {"per_seed": []}}
    zero = torch.zeros(1, N_QUBITS, dtype=DT)
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        head = PQCHead(n_qubits=N_QUBITS, n_layers=2).to(DT)
        J, f = jacobian(head, x)
        J0, f0 = jacobian(head, zero)
        inert = [int(k) for k in range(J.shape[1]) if np.abs(J[:, k]).max() < 1e-10]
        live = [k for k in range(J.shape[1]) if k not in inert]
        Jl = J[:, live]
        mean_k, sd_k = Jl.mean(0), Jl.std(0)
        sign_cons = np.maximum((Jl > 0).mean(0), (Jl < 0).mean(0))
        v, *_ = np.linalg.lstsq(Jl, np.ones(J.shape[0]), rcond=None)
        resid = np.linalg.norm(np.ones(J.shape[0]) - Jl @ v) / np.sqrt(J.shape[0])
        g = J.mean(0)
        fit = Jl @ v
        znorm2 = (x.numpy() ** 2).sum(1)
        near_inert = [int(k) for k in live if np.abs(J[:, k]).max() < 1e-3]
        # first 16 features: the subset the Qiskit Aer port of the probe uses by default
        rec = {
            "seed": seed, "mean_Z0": float(f.mean()), "mean_Z0_ref": float(ref[seed]), "mean_Z0_first16": float(f[:16].mean()),
            "f_at_zero": float(f0[0]), "max_abs_grad_at_zero": float(np.abs(J0).max()),
            "n_inert": len(inert), "inert_idx": inert,
            "live_mean_grad_max_abs": float(np.abs(mean_k).max()), "live_sd_grad_min": float(sd_k.min()),
            "live_sign_consistency_max": float(sign_cons.max()), "live_sign_consistency_mean": float(sign_cons.mean()),
            "uniform_shift_lstsq_rms_residual": float(resid),
            "mean_readout_grad_norm": float(np.linalg.norm(g)), "mean_readout_grad_max_abs": float(np.abs(g).max()),
            "max_abs_grad_any": float(np.abs(J).max()),
            "n_live_sign_consistent_098": int((sign_cons >= 0.98).sum()), "n_live": len(live),
            "n_near_inert_1e-3": len(near_inert), "near_inert_idx": near_inert,
            "fitted_shift_min": float(fit.min()), "fitted_shift_max": float(fit.max()),
            "corr_fit_znorm2": float(np.corrcoef(fit, znorm2)[0, 1]),
        }
        out["L2"]["per_seed"].append(rec)
        print(f"seed {seed}: <Z0> {rec['mean_Z0']:+.4f} (ref {rec['mean_Z0_ref']:+.4f}; first16 {rec['mean_Z0_first16']:+.4f}) "
              f"f(0)={rec['f_at_zero']:+.6f} |grad(0)|max={rec['max_abs_grad_at_zero']:.1e} inert={rec['n_inert']} {inert} "
              f"| live: max|mean grad| {rec['live_mean_grad_max_abs']:.4f}, min sd {rec['live_sd_grad_min']:.4f}, "
              f"sign-consistency max {rec['live_sign_consistency_max']:.3f} mean {rec['live_sign_consistency_mean']:.3f} "
              f"| uniform-shift residual {rec['uniform_shift_lstsq_rms_residual']:.3f} | ||grad mean|| {rec['mean_readout_grad_norm']:.4f} max {rec['mean_readout_grad_max_abs']:.4f} | max|J| {rec['max_abs_grad_any']:.3f}")

    # inert-count formula check at L = 1 and L = 4 on 20 inputs
    out["inert_check"] = {}
    for L in (1, 4):
        torch.manual_seed(0)
        head = PQCHead(n_qubits=N_QUBITS, n_layers=L).to(DT)
        J, _ = jacobian(head, x[:20])
        n_inert = int(sum(np.abs(J[:, k]).max() < 1e-10 for k in range(J.shape[1])))
        out["inert_check"][f"L{L}"] = {"n_params": int(J.shape[1]), "n_inert": n_inert}
        print(f"L={L}: {n_inert} of {J.shape[1]} inert")

    s = out["L2"]["per_seed"]
    out["L2"]["summary"] = {k: [float(np.mean([r[k] for r in s])), float(np.min([r[k] for r in s])), float(np.max([r[k] for r in s]))]
                            for k in ("mean_Z0", "mean_Z0_first16", "live_sign_consistency_max", "uniform_shift_lstsq_rms_residual",
                                      "mean_readout_grad_norm", "mean_readout_grad_max_abs", "max_abs_grad_any", "live_mean_grad_max_abs")}
    for k in ("n_live_sign_consistent_098", "n_near_inert_1e-3", "fitted_shift_min", "fitted_shift_max", "corr_fit_znorm2"):
        out["L2"]["summary"][k] = [float(np.mean([r[k] for r in s])), float(np.min([r[k] for r in s])), float(np.max([r[k] for r in s]))]
    out["L2"]["summary"]["max_abs_grad_at_zero"] = float(max(r["max_abs_grad_at_zero"] for r in s))
    out["L2"]["summary"]["f_at_zero_min"] = float(min(r["f_at_zero"] for r in s))
    out["L2"]["summary"]["n_inert"] = sorted(set(r["n_inert"] for r in s))
    out["L2"]["summary"]["max_abs_dev_from_ref"] = float(max(abs(r["mean_Z0"] - r["mean_Z0_ref"]) for r in s))
    print(json.dumps(out["L2"]["summary"], indent=1))
    (ROOT / "results/grad_analysis_v9.json").write_text(json.dumps(out, indent=1))
    print("wrote results/grad_analysis_v9.json")


if __name__ == "__main__":
    main()
