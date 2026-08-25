#!/usr/bin/env python3
"""Why does the bias-role asymmetry bite some ansaetze and not others?

Mechanistic probe, no training required: measure the distribution of the raw
readout <Z_0> AT INITIALISATION for each bias-free quantum head, on real frozen
reducer features.

Hypothesis (formed after seeing E2's uncalibrated SEL show no positive-bias
pathology, unlike data-reuploading):

  PQCHead encodes as RY(theta_i * x_i * pi) -- the encoding angle is SCALED BY A
  SMALL-INITIALISED TRAINABLE PARAMETER (0.1*randn). At init the rotations are
  therefore ~0, the state stays near |0...0>, and <Z_0> is pinned near +1: a
  systematic offset from the decision threshold at 0, which is exactly what a
  bias term would absorb.

  StronglyEntanglingHead encodes as AngleEmbedding(x * pi) -- full-scale, NOT
  parameter-scaled. The state is spread at init and <Z_0> is centred near 0, so
  there is no offset for a bias to absorb.

If true, the operative condition is not "the head lacks a bias" alone, but
"the head lacks a bias AND its readout is systematically offset from the
decision threshold". That yields a cheap pre-flight diagnostic for QNLP
practitioners: measure mean <Z> at init; if |mean| is large, a calibration term
is mandatory before any matched-parameter claim.

Writes results/readout_offset_v7.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_heads import PQCHead, StronglyEntanglingHead  # noqa: E402
from hybrid_reducer import DepthwiseSBERTReducer  # noqa: E402

DT = torch.float64
N_SEEDS = 8
N_SAMPLES = 200


def real_features() -> torch.Tensor | None:
    """Frozen depthwise-reducer features from a saved MRPC checkpoint, if present."""
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


def probe(make_head, x: torch.Tensor) -> dict:
    """Mean/std of <Z_0> at init, and the implied predict-positive rate."""
    means, pos_rates = [], []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        head = make_head().to(DT)
        with torch.no_grad():
            z = head(x)
        if z.dim() > 1:
            z = z.mean(1)
        z = z.numpy()
        means.append(float(z.mean()))
        # trainer thresholds the raw logit at 0.0 (see train_hybrid_v5.py)
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

    heads = {
        "PQC_reuploading_L2": lambda: PQCHead(n_qubits=8, n_layers=2),
        "PQC_reuploading_L4": lambda: PQCHead(n_qubits=8, n_layers=4),
        "SEL_L1": lambda: StronglyEntanglingHead(n_qubits=8, n_layers=1),
    }
    out = {"feature_source": source, "n_samples": int(x.shape[0]),
           "n_seeds": N_SEEDS, "heads": {}}

    print(f"{'head':<22} {'mean<Z0>':>10} {'std(seeds)':>11} {'init pos_rate':>14}")
    print("-" * 60)
    for name, mk in heads.items():
        r = probe(mk, x)
        out["heads"][name] = r
        print(f"{name:<22} {r['mean_Z0']:>+10.4f} {r['std_Z0_across_seeds']:>11.4f} "
              f"{r['init_pos_pred_rate']:>14.3f}")

    pqc = out["heads"]["PQC_reuploading_L2"]["mean_Z0"]
    sel = out["heads"]["SEL_L1"]["mean_Z0"]
    print(f"\nPQC(L2) |mean<Z0>| = {abs(pqc):.4f}   SEL |mean<Z0>| = {abs(sel):.4f}")
    print("Offset hypothesis supported" if abs(pqc) > 0.5 and abs(sel) < 0.2
          else "Offset hypothesis NOT clearly supported -- report as such")

    (ROOT / "results/readout_offset_v7.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results/readout_offset_v7.json")


if __name__ == "__main__":
    main()
