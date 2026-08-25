"""
noise_proxy_v7_multiseed.py -- multi-seed NISQ depolarizing-noise robustness
proxy for the v5 PQC head (PREREGISTRATION_v7_mechanism.md, Experiment E3).

Same protocol as scripts/noise_proxy_v6.py (seed 0 only), repeated for seeds
0-7: retrains one v5 MRPC depthwise+PQC run per seed by importing the actual
Phase-1/Phase-2 training code from train_hybrid_v5.py (no copy-pasted
training loop), then re-evaluates the trained circuit params + affine
calibration -- with NO further training -- on a qml.device("default.mixed")
rebuild of the identical circuit (shared via hybrid_heads.pqc_data_reuploading_ops,
also not copy-pasted), with a DepolarizingChannel(p) on every qubit after each
of the 2 layers, sweeping p in {0.0, 0.001, 0.005, 0.01, 0.05, 0.1}. See
noise_proxy_v6.py's module docstring for the full rationale (why heads are
retrained rather than loaded from a checkpoint, timing probe, etc.) -- this
script only adds the seed loop plus incremental save/resume.

Writes results/noise_proxy_v7_multiseed.json (does NOT touch or overwrite
noise_proxy_v6.json -- that single-seed result stays as-is). Saves after
EVERY seed so a kill/interruption loses at most one seed's work; rerunning
the script skips seeds already present with a passed sanity check.

Run in the background:
  nohup .venv/bin/python scripts/noise_proxy_v7_multiseed.py \
      > logs/noise_proxy_v7_multiseed.log 2>&1 &

Selftest of the resume/save logic only (no SBERT/PQC training exercised):
  .venv/bin/python scripts/noise_proxy_v7_multiseed.py --selftest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pennylane as qml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_heads import PQCHead, pqc_data_reuploading_ops
from hybrid_reducer import DepthwiseSBERTReducer
from hybrid_sbert import SBERTFeatures
from datasets_qec import load_dataset_by_name
from train_hybrid_v5 import (
    CalibratedHead, run_phase1, run_phase2, compute_metrics, set_seed, set_train_mode, DT,
)

# ---- fixed v5 MRPC depthwise+PQC config (matches run_hybrid_v5.sh / noise_proxy_v6.py) ----
TASK = "mrpc"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
N_QUBITS = 8
N_LAYERS = 2
N_TRAIN = 2000
N_VAL = 200
PRETRAIN_EPOCHS = 20
EPOCHS = 25
PATIENCE = 6
LR = 0.002
BATCH_SIZE = 32
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_DIM = 384
KERNEL_SIZE = 32
STRIDE = 32

P_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
SANITY_TOL = 0.01

OUT_PATH = ROOT / "results" / "noise_proxy_v7_multiseed.json"


def to_t(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float64)).to(DT)


def train_v5_run(seed: int) -> dict:
    """Phase 1 + Phase 2 for one seed, identical to train_hybrid_v5.main(),
    minus disk writes (no checkpoint save, no results/v5 json)."""
    set_seed(seed)
    t0 = time.time()

    train_pairs = load_dataset_by_name(TASK, n_pairs=N_TRAIN, seed=seed, split="train")
    val_all = load_dataset_by_name(TASK, n_pairs=None, seed=seed, split="validation")
    val_pairs = val_all[:N_VAL]
    test_pairs = val_all[N_VAL:]
    print(f"  data -- train: {len(train_pairs)}  val: {len(val_pairs)}  test: {len(test_pairs)}")

    sb = SBERTFeatures(model_name=SBERT_MODEL)
    emb_a_tr, emb_b_tr, y_tr = sb.encode_pairs(train_pairs)
    emb_a_val, emb_b_val, y_val = sb.encode_pairs(val_pairs)
    emb_a_te, emb_b_te, y_te = sb.encode_pairs(test_pairs)
    print(f"  SBERT encode done ({time.time()-t0:.1f}s)")

    emb_a_tr_t, emb_b_tr_t = to_t(emb_a_tr), to_t(emb_b_tr)
    emb_a_val_t, emb_b_val_t = to_t(emb_a_val), to_t(emb_b_val)
    emb_a_te_t, emb_b_te_t = to_t(emb_a_te), to_t(emb_b_te)

    reducer = DepthwiseSBERTReducer(
        sbert_dim=SBERT_DIM, kernel_size=KERNEL_SIZE, stride=STRIDE, n_qubits=N_QUBITS,
    ).to(DT)
    reducer_n_params = reducer.n_params  # capture before Phase-2 freeze zeroes requires_grad
    print(f"  reducer params={reducer_n_params}")

    print("  --- Phase 1: reducer pretraining with linear probe (BCE) ---")
    p1_history, p1_best_val_f1 = run_phase1(
        reducer, n_qubits=N_QUBITS,
        emb_a_tr=emb_a_tr_t, emb_b_tr=emb_b_tr_t, y_tr=y_tr,
        emb_a_val=emb_a_val_t, emb_b_val=emb_b_val_t, y_val=y_val,
        epochs=PRETRAIN_EPOCHS, lr=LR, batch_size=BATCH_SIZE, patience=PATIENCE,
    )
    print(f"  Phase 1 done: best_val_f1={p1_best_val_f1:.4f}  ({time.time()-t0:.1f}s)")

    set_train_mode(reducer, False)
    for param in reducer.parameters():
        param.requires_grad = False
    with torch.no_grad():
        feat_tr = reducer(emb_a_tr_t, emb_b_tr_t)
        feat_val = reducer(emb_a_val_t, emb_b_val_t)
        feat_te = reducer(emb_a_te_t, emb_b_te_t)
    print(f"  frozen features: tr={tuple(feat_tr.shape)} val={tuple(feat_val.shape)} te={tuple(feat_te.shape)}")

    base_head = PQCHead(n_qubits=N_QUBITS, n_layers=N_LAYERS).to(DT)
    head = CalibratedHead(base_head).to(DT)
    print(f"  head params={head.n_params} (base {base_head.n_params} + 2 calib)")

    print("  --- Phase 2: frozen reducer + PQC head (BCE) ---")
    p2_history, p2_best_val_f1, test_metrics = run_phase2(
        feat_tr, feat_val, feat_te, y_tr, y_val, y_te, head,
        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, patience=PATIENCE,
    )
    print(f"  Phase 2 done: best_val_macro_f1={p2_best_val_f1:.4f}"
          f"  test_macro_f1={test_metrics['macro_f1']:.4f}  ({time.time()-t0:.1f}s)")

    return {
        "head": head, "base_head": base_head, "reducer": reducer,
        "reducer_n_params": reducer_n_params,
        "feat_te": feat_te, "y_te": y_te, "n_test": len(test_pairs),
        "test_metrics": test_metrics,
        "p1_best_val_f1": p1_best_val_f1, "p2_best_val_f1": p2_best_val_f1,
    }


def build_noisy_qnode(noise_p: float, dev):
    """Same circuit as PQCHead (via the shared pqc_data_reuploading_ops), on a
    caller-supplied (noisy) device, with a DepolarizingChannel(noise_p) on
    every qubit after each layer."""
    def noise_fn():
        for i in range(N_QUBITS):
            qml.DepolarizingChannel(noise_p, wires=i)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def circuit(x, params):
        pqc_data_reuploading_ops(x, params, N_QUBITS, N_LAYERS, noise_fn=noise_fn)
        return qml.expval(qml.PauliZ(0))

    return circuit


def eval_at_noise(noise_p: float, dev, pqc_params, scale: float, bias: float,
                   feat_te: torch.Tensor, y_te: np.ndarray) -> dict:
    circuit = build_noisy_qnode(noise_p, dev)
    with torch.no_grad():
        raw = torch.stack([circuit(feat_te[i], pqc_params) for i in range(feat_te.shape[0])])
        logits = (scale * raw + bias).numpy()
    m = compute_metrics(logits, y_te)
    return {"macro_f1": m["macro_f1"], "pos_pred_rate": m["pos_pred_rate"]}


def run_one_seed(seed: int) -> dict:
    t0 = time.time()
    trained = train_v5_run(seed)
    head, base_head = trained["head"], trained["base_head"]
    feat_te, y_te = trained["feat_te"], trained["y_te"]
    test_metrics = trained["test_metrics"]

    set_train_mode(head, False)
    pqc_params = base_head.params.detach().clone()
    scale = float(head.scale.detach())
    bias = float(head.bias.detach())

    n_test = feat_te.shape[0]
    print(f"  --- noise sweep on default.mixed (n_test={n_test}) ---")

    dev_mixed = qml.device("default.mixed", wires=N_QUBITS)
    results = {}
    t_sweep = time.time()
    for p in P_VALUES:
        r = eval_at_noise(p, dev_mixed, pqc_params, scale, bias, feat_te, y_te)
        results[str(p)] = r
        print(f"    seed={seed}  p={p:<6}  macro_f1={r['macro_f1']:.4f}"
              f"  pos_pred_rate={r['pos_pred_rate']:.4f}  ({time.time()-t_sweep:.0f}s elapsed)")

    # ---- Sanity check: p=0.0 on default.mixed vs the real default.qubit training-time eval ----
    clean_macro_f1 = test_metrics["macro_f1"]
    clean_pos_pred_rate = test_metrics["pos_pred_rate"]
    p0_macro_f1 = results["0.0"]["macro_f1"]
    diff = abs(p0_macro_f1 - clean_macro_f1)
    sanity_ok = diff <= SANITY_TOL
    print(f"  sanity: default.mixed p=0 macro_f1={p0_macro_f1:.4f} vs "
          f"default.qubit={clean_macro_f1:.4f}  diff={diff:.4f}  -> {'PASS' if sanity_ok else 'FAIL'}")
    # ponytail: noise_proxy_v6.py hard-crashes (raise RuntimeError) on sanity
    # failure -- correct for a single interactive run you'd stop and debug.
    # This script runs 8 seeds unattended in the background; one bad seed
    # shouldn't torch the other 7, so we record passed=False and continue.
    # The reporting step must surface any failure rather than hide it.

    below_half = [p for p in P_VALUES if results[str(p)]["macro_f1"] < 0.5]
    first_below_half = below_half[0] if below_half else None

    return {
        "seed": seed,
        "phase1_best_val_f1": trained["p1_best_val_f1"],
        "phase2_best_val_macro_f1": trained["p2_best_val_f1"],
        "reducer_params": int(trained["reducer_n_params"]),
        "head_params": int(head.n_params),
        "trained_pqc_params": pqc_params.numpy().tolist(),
        "calibration_scale": scale,
        "calibration_bias": bias,
        "n_test_actual": n_test,
        "clean_reference": {
            "device": "default.qubit (actual Phase-2 training-time eval)",
            "test_macro_f1": clean_macro_f1,
            "test_pos_pred_rate": clean_pos_pred_rate,
            "test_f1": test_metrics["f1"],
            "test_acc": test_metrics["acc"],
        },
        "sanity_check": {
            "default_mixed_p0_macro_f1": p0_macro_f1,
            "default_qubit_macro_f1": clean_macro_f1,
            "abs_diff": diff,
            "tolerance": SANITY_TOL,
            "passed": sanity_ok,
        },
        "first_p_macro_f1_below_0.5": first_below_half,
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Incremental save / resume
# ---------------------------------------------------------------------------

def shared_config() -> dict:
    return {
        "task": TASK, "reducer": "depthwise", "head": "pqc",
        "n_qubits": N_QUBITS, "n_layers": N_LAYERS,
        "n_train": N_TRAIN, "n_val": N_VAL,
        "pretrain_epochs": PRETRAIN_EPOCHS, "epochs": EPOCHS,
        "patience": PATIENCE, "lr": LR, "batch_size": BATCH_SIZE,
        "noise_model": "qml.DepolarizingChannel(p, wires=i) on all n_qubits, "
                       "inserted after each of the n_layers layers, on default.mixed",
        "p_values": P_VALUES,
        "seeds": SEEDS,
        "prereg": "PREREGISTRATION_v7_mechanism.md Experiment E3",
    }


def load_existing() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"shared_config": shared_config(), "per_seed": {}}


def save(state: dict) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(state, indent=2))


def seed_done(state: dict, seed: int) -> bool:
    rec = state.get("per_seed", {}).get(str(seed))
    return bool(rec) and bool(rec.get("sanity_check", {}).get("passed"))


def main() -> None:
    t_start = time.time()
    state = load_existing()
    state["shared_config"] = shared_config()  # keep config fresh even on resume
    save(state)

    for seed in SEEDS:
        if seed_done(state, seed):
            print(f"[seed {seed}] already done (sanity passed) -- skipping")
            continue
        print(f"\n=== seed {seed} ({time.time()-t_start:.0f}s elapsed) ===")
        state["per_seed"][str(seed)] = run_one_seed(seed)
        save(state)  # incremental: survives a kill between seeds
        print(f"[seed {seed}] saved -> {OUT_PATH}")

    print(f"\n[done] all seeds processed ({time.time()-t_start:.0f}s total)")


def _selftest() -> None:
    """ponytail: exercises the resume/save logic (the new code beyond v6)
    without running the full SBERT/PQC training pipeline."""
    import tempfile
    global OUT_PATH
    orig = OUT_PATH
    with tempfile.TemporaryDirectory() as d:
        OUT_PATH = Path(d) / "test.json"
        state = load_existing()
        assert state["per_seed"] == {}, "fresh state must start empty"

        state["per_seed"]["0"] = {"sanity_check": {"passed": True}, "dummy": 1}
        save(state)
        reloaded = load_existing()
        assert reloaded["per_seed"]["0"]["dummy"] == 1, "save/load round-trip broken"
        assert seed_done(reloaded, 0) is True, "passed seed should be marked done"
        assert seed_done(reloaded, 1) is False, "absent seed must not be done"

        reloaded["per_seed"]["2"] = {"sanity_check": {"passed": False}, "dummy": 2}
        assert seed_done(reloaded, 2) is False, "failed-sanity seed must be re-run, not skipped"
    OUT_PATH = orig
    print("[selftest] resume/save logic OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
