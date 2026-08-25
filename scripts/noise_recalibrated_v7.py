#!/usr/bin/env python3
"""Depolarizing sweep with the affine calibration RE-FITTED at each noise level.

The multi-seed sweep in results/noise_proxy_v7_multiseed.json evaluates trained
heads under noise while holding the affine (a, b) frozen at its noiseless values.
A depolarizing channel contracts <Z_0> multiplicatively toward zero, so that
design confounds information loss with threshold miscalibration: at large p the
frozen negative bias drags every logit below threshold.

This script separates the two. At each noise level it re-fits ONLY the two affine
parameters (a, b), on the validation split, holding every circuit parameter fixed
at its trained value, then reports test macro F1 under both the frozen and the
re-fitted decision rule. No circuit parameter is retrained.

Writes results/noise_recalibrated_v7.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pennylane as qml  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from hybrid_heads import pqc_data_reuploading_ops  # noqa: E402
from hybrid_reducer import DepthwiseSBERTReducer  # noqa: E402
from hybrid_sbert import SBERTFeatures  # noqa: E402
from datasets_qec import load_dataset_by_name  # noqa: E402

DT = torch.float64
P_VALUES = [0.0, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
N_QUBITS, N_LAYERS, N_VAL = 8, 2, 200


def features_for_seed(seed: int, sb: SBERTFeatures):
    """Reproduce the run's frozen val/test features from the archived checkpoint."""
    ckpt = ROOT / f"results/v4/checkpoints/reducer_mrpc_depthwise_q8_s{seed}.pt"
    reducer = DepthwiseSBERTReducer(sbert_dim=384, kernel_size=32, stride=32,
                                    n_qubits=N_QUBITS).to(DT)
    reducer.load_state_dict(torch.load(ckpt, weights_only=True))
    reducer.eval()
    val_all = load_dataset_by_name("mrpc", n_pairs=None, seed=seed, split="validation")
    val_pairs, test_pairs = val_all[:N_VAL], val_all[N_VAL:]
    outs = []
    for pairs in (val_pairs, test_pairs):
        a, b, y = sb.encode_pairs(pairs)
        with torch.no_grad():
            f = reducer(torch.from_numpy(a.astype(np.float64)).to(DT),
                        torch.from_numpy(b.astype(np.float64)).to(DT))
        outs.append((f.numpy(), np.asarray(y)))
    return outs[0], outs[1]


def noisy_logits(feats: np.ndarray, params: np.ndarray, p: float) -> np.ndarray:
    dev = qml.device("default.mixed", wires=N_QUBITS)

    def noise_fn():
        for w in range(N_QUBITS):
            qml.DepolarizingChannel(p, wires=w)

    @qml.qnode(dev)
    def circuit(x):
        pqc_data_reuploading_ops(x, params, N_QUBITS, N_LAYERS,
                                 noise_fn=None if p == 0 else noise_fn)
        return qml.expval(qml.PauliZ(0))

    return np.array([float(circuit(x)) for x in feats])


def mf1(logits, y, a, b):
    return float(f1_score(y, (a * logits + b) > 0, average="macro", zero_division=0))


def refit_affine(logits, y):
    best = (1.0, 0.0, mf1(logits, y, 1.0, 0.0))
    for a in np.linspace(0.05, 2.0, 40):
        for b in np.linspace(-1.0, 1.0, 81):
            s = mf1(logits, y, a, b)
            if s > best[2]:
                best = (a, b, s)
    a0, b0, _ = best
    for a in np.linspace(max(0.01, a0 - 0.06), a0 + 0.06, 13):
        for b in np.linspace(b0 - 0.06, b0 + 0.06, 25):
            s = mf1(logits, y, a, b)
            if s > best[2]:
                best = (a, b, s)
    return float(best[0]), float(best[1])


def main() -> None:
    only = set(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else None
    src = json.loads((ROOT / "results/noise_proxy_v7_multiseed.json").read_text())
    sb = SBERTFeatures(model_name="all-MiniLM-L6-v2")
    out = {"description": "Depolarizing sweep with (a,b) re-fitted on validation at each "
                          "noise level; circuit parameters held at trained values.",
           "p_values": P_VALUES, "per_seed": {}}

    dest = ROOT / "results/noise_recalibrated_v7_seeds.json"
    acc = json.loads(dest.read_text()) if dest.exists() else {}
    for key, entry in sorted(src["per_seed"].items(), key=lambda kv: int(kv[0])):
        seed = int(key)
        if only is not None and seed not in only:
            continue
        params = np.array(entry["trained_pqc_params"], dtype=float)
        a_fr, b_fr = entry["calibration_scale"], entry["calibration_bias"]
        (Xv, yv), (Xt, yt) = features_for_seed(seed, sb)
        rec = {"seed": seed, "frozen": {}, "refit": {}}
        for p in P_VALUES:
            lv, lt = noisy_logits(Xv, params, p), noisy_logits(Xt, params, p)
            rec["frozen"][str(p)] = {"macro_f1": mf1(lt, yt, a_fr, b_fr),
                                     "pos_pred_rate": float(((a_fr * lt + b_fr) > 0).mean())}
            a, b = refit_affine(lv, yv)
            rec["refit"][str(p)] = {"macro_f1": mf1(lt, yt, a, b), "a": a, "b": b,
                                    "pos_pred_rate": float(((a * lt + b) > 0).mean())}
            print("seed %d p=%-7g frozen %.4f -> refit %.4f  (a=%.2f b=%+.2f)"
                  % (seed, p, rec["frozen"][str(p)]["macro_f1"],
                     rec["refit"][str(p)]["macro_f1"], a, b), flush=True)
        out["per_seed"][key] = rec
        acc[key] = rec
        dest.write_text(json.dumps(acc, indent=1))
        print("  saved seed", seed, flush=True)

    dest = ROOT / "results/noise_recalibrated_v7.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()
