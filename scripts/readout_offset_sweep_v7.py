#!/usr/bin/env python3
"""Initialisation-scale sweep of the data-reuploading readout offset.

Companion to readout_offset_v7.py, which probes the three heads at their default
initialisation. This script sweeps the encoding-parameter initialisation scale
alone and records the resulting mean <Z_0> and the initial positive-prediction
rate. That sweep is the causal evidence behind the mechanism claim in the paper
(Section "The Readout-Offset Mechanism", initialisation-scale sweep table and
Figure "mechanism", panel b).

No training is performed: every number is one forward pass per example at
initialisation. Uses the same frozen-feature path as readout_offset_v7.py so the
two are directly comparable.

Writes results/readout_offset_sweep_v7.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_heads import PQCHead  # noqa: E402
from hybrid_reducer import DepthwiseSBERTReducer  # noqa: E402

DT = torch.float64
N_SEEDS = 8
N_SAMPLES = 200
SCALES = [0.1, 0.5, 1.0, 2.0]


def real_features() -> torch.Tensor | None:
    """Frozen depthwise-reducer features from the saved MRPC checkpoint, if present.

    Identical to readout_offset_v7.real_features() so the sweep and the
    three-head probe are measured on exactly the same inputs.
    """
    ckpt = ROOT / "results/v4/checkpoints/reducer_mrpc_depthwise_q8_s0.pt"
    if not ckpt.exists():
        return None
    try:
        from hybrid_sbert import SBERTFeatures
        from datasets_qec import load_dataset_by_name
    except Exception:
        return None
    reducer = DepthwiseSBERTReducer(sbert_dim=384, kernel_size=32, stride=32,
                                    n_qubits=8).to(DT)
    reducer.load_state_dict(torch.load(ckpt, weights_only=True))
    reducer.eval()
    pairs = load_dataset_by_name("mrpc", n_pairs=N_SAMPLES, seed=0, split="validation")
    sb = SBERTFeatures(model_name="all-MiniLM-L6-v2")
    a, b, _ = sb.encode_pairs(pairs)
    with torch.no_grad():
        return reducer(torch.from_numpy(a.astype(np.float64)).to(DT),
                       torch.from_numpy(b.astype(np.float64)).to(DT))


def probe_at_scale(x: torch.Tensor, init_scale: float) -> dict:
    """Mean/std of <Z_0> at init and the implied predict-positive rate, per seed."""
    means, pos_rates = [], []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        head = PQCHead(n_qubits=8, n_layers=2, init_scale=init_scale).to(DT)
        with torch.no_grad():
            z = head(x)
        z = z.numpy()
        means.append(float(z.mean()))
        # the trainer thresholds the raw logit at 0.0 (see train_hybrid_v5.py)
        pos_rates.append(float((z > 0.0).mean()))
    return {
        "mean_Z0_per_seed": means,
        "mean_Z0": float(np.mean(means)),
        "std_Z0_across_seeds": float(np.std(means)),
        "init_pos_pred_rate": float(np.mean(pos_rates)),
    }


def main() -> None:
    x = real_features()
    source = "real frozen depthwise reducer features (MRPC, seed 0 ckpt)"
    if x is None:
        rng = np.random.default_rng(0)
        x = torch.from_numpy(rng.uniform(-1, 1, size=(N_SAMPLES, 8))).to(DT)
        source = "synthetic uniform[-1,1]^8 (tanh output range; no ckpt found)"
    print(f"features: {source}  shape={tuple(x.shape)}\n")

    out = {
        "description": "Initialisation-scale sweep of the data-reuploading head. "
                       "No training; one forward pass per example at initialisation.",
        "feature_source": source,
        "n_samples": int(x.shape[0]),
        "n_seeds": N_SEEDS,
        "n_qubits": 8,
        "n_layers": 2,
        "init_scales": SCALES,
        "results": {},
    }

    print(f"{'init_scale':<12} {'mean<Z0>':>10} {'std(seeds)':>11} {'init pos_rate':>14}")
    print("-" * 52)
    for s in SCALES:
        r = probe_at_scale(x, s)
        out["results"][str(s)] = r
        print(f"{s:<12} {r['mean_Z0']:>+10.4f} {r['std_Z0_across_seeds']:>11.4f} "
              f"{r['init_pos_pred_rate']:>14.3f}")

    dest = ROOT / "results/readout_offset_sweep_v7.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
