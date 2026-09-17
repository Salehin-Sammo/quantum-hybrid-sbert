#!/usr/bin/env python3
"""Readout-offset probe, Qiskit port, runnable on Aer or on IBM hardware.

Measures <Z_0> at INITIALISATION (no training) for the data-reuploading head
and the StronglyEntanglingLayers head on real frozen depthwise features. The
PennyLane reference values this must reproduce, from
results/readout_offset_v7.json:

    reuploading L=2 : +0.9221     SEL L=1 : +0.0096

The point of running this on hardware is that device noise pulls <Z> toward
zero, which is exactly the centring the calibration term supplies. Whether a
real device partially self-corrects the offset is not answerable in a
noiseless simulator.

Usage:
    python scripts/hw_readout_probe.py --backend aer            # exact, validates the port
    python scripts/hw_readout_probe.py --backend aer-noisy      # rough hardware forecast
    python scripts/hw_readout_probe.py --backend ibm            # real QPU (needs a valid token)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

N_QUBITS, N_LAYERS_RU, N_LAYERS_SEL = 8, 2, 1
N_SEEDS, N_FEATURES, SHOTS = 8, 16, 4096
SIMULATOR_SEED = 0


def features(n: int) -> np.ndarray:
    """Released real frozen depthwise features used by the paper's probe.

    The previous fallback to synthetic uniform features made an apparently
    successful run analyse a different experiment.  Fail loudly instead.
    """
    path = ROOT / "results/mrpc_depthwise_q8_s0_features.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"missing released feature snapshot: {path.relative_to(ROOT)}"
        )
    z = np.load(path)
    if z.shape != (200, N_QUBITS):
        raise ValueError(f"unexpected feature shape {z.shape}, expected (200, {N_QUBITS})")
    if n > len(z):
        raise ValueError(f"requested {n} features; released snapshot contains {len(z)}")
    return z[:n]


def reuploading(x: np.ndarray, p: np.ndarray) -> QuantumCircuit:
    """Mirrors pqc_data_reuploading_ops: per layer RY(theta*x*pi), CNOT ring, RZ(theta)."""
    qc, k = QuantumCircuit(N_QUBITS), 0
    for _ in range(N_LAYERS_RU):
        for i in range(N_QUBITS):
            qc.ry(p[k] * x[i] * np.pi, i); k += 1
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)
        for i in range(N_QUBITS):
            qc.rz(p[k], i); k += 1
    return qc


def sel(x: np.ndarray, p: np.ndarray) -> QuantumCircuit:
    """AngleEmbedding(x*pi, rotation=Y) then StronglyEntanglingLayers.
    PennyLane Rot(a,b,c) == RZ(a) RY(b) RZ(c)."""
    qc = QuantumCircuit(N_QUBITS)
    for i in range(N_QUBITS):
        qc.ry(x[i] * np.pi, i)
    for L in range(N_LAYERS_SEL):
        for i in range(N_QUBITS):
            a, b, c = p[L, i]
            qc.rz(a, i); qc.ry(b, i); qc.rz(c, i)
        for i in range(N_QUBITS):
            qc.cx(i, (i + 1) % N_QUBITS)
    return qc


def init_params(kind: str, seed: int):
    """Same init as the torch heads: 0.1*randn."""
    torch.manual_seed(seed)
    if kind == "ru":
        return (torch.randn(2 * N_LAYERS_RU * N_QUBITS) * 0.1).numpy()
    return (torch.randn(N_LAYERS_SEL, N_QUBITS, 3) * 0.1).numpy()


def exact_z0(qc: QuantumCircuit) -> float:
    """<Z_0> from the statevector. Qiskit qubit 0 is the last tensor factor."""
    probs = Statevector.from_instruction(qc).probabilities([0])
    return float(probs[0] - probs[1])


def sampled_z0(counts: dict, n_shots: int) -> float:
    """<Z_0> from measurement counts on qubit 0 (rightmost bit of the string)."""
    p0 = sum(v for b, v in counts.items() if b[-1] == "0") / n_shots
    return float(2 * p0 - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["aer", "aer-noisy", "ibm"], default="aer")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--n-features", type=int, default=N_FEATURES)
    args = ap.parse_args()

    X = features(args.n_features)
    print(f"features: {X.shape}  backend={args.backend}  shots={args.shots}\n")

    circuits, index = [], []
    for kind, builder, nl in (("ru", reuploading, N_LAYERS_RU), ("sel", sel, N_LAYERS_SEL)):
        for s in range(N_SEEDS):
            p = init_params(kind, s)
            for j, x in enumerate(X):
                circuits.append(builder(x, p)); index.append((kind, s, j))

    # exact reference first, always
    exact = {}
    for (kind, s, j), qc in zip(index, circuits):
        exact.setdefault(kind, []).append(exact_z0(qc))
    print("EXACT (statevector, no shots):")
    for kind in ("ru", "sel"):
        v = np.array(exact[kind])
        print(f"  {kind:<4} mean<Z0> = {v.mean():+.4f}   std = {v.std():.4f}   n = {len(v)}")
    print("  PennyLane reference:  ru +0.9221   sel +0.0096\n")

    if args.backend == "aer":
        print("Aer exact requested; statevector result above is the answer.")
        out = {"mode": "exact", "feature_source": "results/mrpc_depthwise_q8_s0_features.npy",
               "n_features": args.n_features,
               "exact": {k: float(np.mean(v)) for k, v in exact.items()}}
    else:
        meas = [qc.copy() for qc in circuits]
        for qc in meas:
            qc.measure_all()
        if args.backend == "aer-noisy":
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
            nm = NoiseModel()
            # representative superconducting-device figures; a forecast, not a device
            nm.add_all_qubit_quantum_error(depolarizing_error(3e-4, 1), ["ry", "rz", "u"])
            nm.add_all_qubit_quantum_error(depolarizing_error(8e-3, 2), ["cx"])
            nm.add_all_qubit_readout_error(ReadoutError([[0.985, 0.015], [0.03, 0.97]]))
            be = AerSimulator(noise_model=nm)
            tq = transpile(meas, be, optimization_level=1)
            res = be.run(tq, shots=args.shots, seed_simulator=SIMULATOR_SEED).result()
            counts = [res.get_counts(i) for i in range(len(meas))]
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            svc = QiskitRuntimeService()
            be = svc.least_busy(operational=True, simulator=False)
            print(f"  QPU: {be.name} ({be.num_qubits} qubits, queue {be.status().pending_jobs})")
            tq = transpile(meas, be, optimization_level=3)
            job = SamplerV2(mode=be).run(tq, shots=args.shots)
            print(f"  job id: {job.job_id()}  (this may queue)")
            r = job.result()
            counts = [r[i].data.meas.get_counts() for i in range(len(meas))]

        got = {}
        for (kind, s, j), c in zip(index, counts):
            got.setdefault(kind, []).append(sampled_z0(c, args.shots))
        print(f"\nMEASURED ({args.backend}):")
        for kind in ("ru", "sel"):
            v, e = np.array(got[kind]), np.array(exact[kind])
            print(f"  {kind:<4} mean<Z0> = {v.mean():+.4f}   std = {v.std():.4f}"
                  f"   shift vs exact = {v.mean()-e.mean():+.4f}")
        out = {"mode": args.backend, "shots": args.shots,
               "feature_source": "results/mrpc_depthwise_q8_s0_features.npy",
               "n_features": args.n_features,
               "simulator_seed": SIMULATOR_SEED if args.backend == "aer-noisy" else None,
               "exact": {k: float(np.mean(v)) for k, v in exact.items()},
               "measured": {k: float(np.mean(v)) for k, v in got.items()},
               "backend": getattr(be, "name", str(be))}

    p = ROOT / f"results/hw_readout_probe_{args.backend}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
