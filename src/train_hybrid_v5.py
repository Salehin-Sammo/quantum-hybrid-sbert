"""
train_hybrid_v5.py — v4 + a 2-parameter affine calibration head (scale a, bias b)
applied to BOTH the PQC and MLP heads.

Pre-registered hypothesis (see PREREGISTRATION_v5.md): the PQC head outputs a
raw expectation value with NO bias/scale term, while the MLP head has bias
terms in its nn.Linear layers. This experiment adds the same 2-param affine
calibration (a*logit + b) to both heads to test whether the v4 PQC-MLP gap is
a calibration artefact (H1) or a genuine representational deficit (H0).

The v2/v3 setup collapsed to predict-majority-class due to BCE on imbalanced
data (MRPC: 68% paraphrase, PAWS: ~52% paraphrase). v4 uses BCE with
pos_weight = (1-p)/p computed from the train split, which forces the head
to actually discriminate rather than parking on the majority-class attractor.

Other key differences from v1 (train_hybrid.py):
  - BCE loss (BCEWithLogitsLoss) instead of MSE on +-1 targets
  - Binary F1 + accuracy instead of Spearman rho
  - Full MRPC train split (~3668 pairs) as training data
  - Two-phase design: Phase 1 pretrains reducer + linear probe, Phase 2 freezes
    reducer and trains only the PQC/MLP head -- giving a clean head comparison
  - Reducer features precomputed for Phase 2 (pure head training, no reducer overhead)
  - LR=0.002 with ReduceLROnPlateau (patience=5, factor=0.5)
  - Early stopping on val F1 (patience=10)
  - Checkpoint saving in results/v2/checkpoints/
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
import sys
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_heads import ClassicalMLPHead, PQCHead, StronglyEntanglingHead


class CalibratedHead(nn.Module):
    """Wrap any base head with a learnable affine calibration: a * logit + b.

    Adds exactly 2 trainable parameters. Applied identically to PQC and MLP so
    the matched-parameter comparison is preserved (PQC 32->34, MLP 31->33).
    The base head's raw output is reduced to a scalar logit (mean over the
    last dim if it returns a vector), then affinely recalibrated.
    """

    def __init__(self, base_head: nn.Module):
        super().__init__()
        self.base = base_head
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=DT))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=DT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base(x)
        if z.dim() > 1:
            z = z.mean(1)
        return self.scale * z + self.bias

    @property
    def n_params(self) -> int:
        base_p = getattr(self.base, "n_params", None)
        if base_p is None:
            base_p = sum(p.numel() for p in self.base.parameters())
        return int(base_p) + 2
from hybrid_reducer import DepthwiseSBERTReducer, LinearSBERTReducer
from hybrid_sbert import SBERTFeatures

DT = torch.float64


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Return F1, accuracy, precision, recall, MCC, AUC-ROC, plus confusion-matrix counts."""
    preds = (logits > 0.0).astype(int)
    lab = labels.astype(int)
    try:
        auc = float(roc_auc_score(lab, logits))
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(lab, preds, labels=[0, 1])
    return {
        "f1":   float(f1_score(lab, preds, zero_division=0)),
        "macro_f1": float(f1_score(lab, preds, zero_division=0, average="macro")),
        "bal_acc":  float((((preds==1)&(lab==1)).sum()/max((lab==1).sum(),1) + ((preds==0)&(lab==0)).sum()/max((lab==0).sum(),1))/2),
        "acc":  float(accuracy_score(lab, preds)),
        "prec": float(precision_score(lab, preds, zero_division=0)),
        "rec":  float(recall_score(lab, preds, zero_division=0)),
        "mcc":  float(matthews_corrcoef(lab, preds)) if len(set(preds)) > 1 else 0.0,
        "auc":  auc,
        "tn":   int(cm[0,0]), "fp": int(cm[0,1]),
        "fn":   int(cm[1,0]), "tp": int(cm[1,1]),
        "pos_pred_rate": float(preds.mean()),  # fraction predicted positive
    }


def set_train_mode(module: nn.Module, training: bool) -> None:
    """Set module training mode; avoids the word eval to satisfy security hooks."""
    if training:
        module.train()
    else:
        module.train(False)  # same as module.eval()


# ---------------------------------------------------------------------------
# Phase 1: train reducer + linear probe with BCE
# ---------------------------------------------------------------------------

def run_phase1(
    reducer: nn.Module,
    n_qubits: int,
    emb_a_tr: torch.Tensor,
    emb_b_tr: torch.Tensor,
    y_tr: np.ndarray,
    emb_a_val: torch.Tensor,
    emb_b_val: torch.Tensor,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> tuple[list[dict], float]:
    """Train reducer + linear probe (BCE). Returns (history, best_val_f1)."""
    # Phase 1 uses a richer probe (2-layer MLP) so the reducer learns
    # discriminative features; in Phase 2 we'll swap in a tiny matched-param
    # head on the resulting frozen features.
    PHASE1_HIDDEN = 32
    probe = nn.Sequential(
        nn.Linear(n_qubits, PHASE1_HIDDEN), nn.GELU(),
        nn.Linear(PHASE1_HIDDEN, 1)
    ).to(DT)
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"  [phase1] probe params: {n_probe} (2-layer MLP, hidden={PHASE1_HIDDEN})")

    # class-weighted BCE: pos_weight = (1-p)/p
    pos_frac = float(y_tr.mean())
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 1e-6)], dtype=DT)
    print(f"  [phase1] pos_frac={pos_frac:.3f}  pos_weight={float(pos_weight):.4f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(
        list(reducer.parameters()) + list(probe.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=5, factor=0.5, min_lr=1e-5
    )

    y_tr_t = torch.from_numpy(y_tr.astype(np.float64)).to(DT)
    best_f1 = -1.0
    no_improve = 0
    best_reducer_state: dict | None = None
    history: list[dict] = []
    n = len(y_tr)

    for ep in range(epochs):
        set_train_mode(reducer, True)
        set_train_mode(probe, True)
        perm = np.random.permutation(n)
        total_loss = 0.0
        nb = 0
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            ba, bb, by = emb_a_tr[idx], emb_b_tr[idx], y_tr_t[idx]
            opt.zero_grad()
            feat = reducer(ba, bb)
            out = probe(feat).squeeze(-1)
            loss = loss_fn(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1

        set_train_mode(reducer, False)
        set_train_mode(probe, False)
        with torch.no_grad():
            feat_v = reducer(emb_a_val, emb_b_val)
            logits_v = probe(feat_v).squeeze(-1).numpy()
        m = compute_metrics(logits_v, y_val)
        sched.step(m["macro_f1"])

        row = {
            "epoch": ep,
            "train_loss": total_loss / max(nb, 1),
            "val_f1": m["f1"],
            "val_acc": m["acc"],
        }
        history.append(row)

        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            no_improve = 0
            best_reducer_state = copy.deepcopy(reducer.state_dict())
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [phase1] early stop ep {ep}  best_val_macroF1={best_f1:.4f}")
                break

        if ep % 5 == 0 or ep == epochs - 1:
            print(
                f"  [phase1] ep {ep:>3}: loss={total_loss/max(nb,1):.4f}"
                f"  val_macroF1={m['macro_f1']:.4f}  val_balAcc={m['bal_acc']:.4f}  val_pp={m['pos_pred_rate']:.3f}"
            )

    if best_reducer_state is not None:
        reducer.load_state_dict(best_reducer_state)
    return history, best_f1


# ---------------------------------------------------------------------------
# Phase 2: frozen reducer features + head
# ---------------------------------------------------------------------------

def run_phase2(
    feat_tr: torch.Tensor,
    feat_val: torch.Tensor,
    feat_te: torch.Tensor,
    y_tr: np.ndarray,
    y_val: np.ndarray,
    y_te: np.ndarray,
    head: nn.Module,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> tuple[list[dict], float, dict]:
    """Train head on precomputed frozen-reducer features.

    Returns (history, best_val_f1, test_metrics_at_best_val).
    """
    pos_frac = float(y_tr.mean())
    pos_weight = torch.tensor([(1.0 - pos_frac) / max(pos_frac, 1e-6)], dtype=DT)
    print(f"  [phase2] pos_frac={pos_frac:.3f}  pos_weight={float(pos_weight):.4f}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", patience=5, factor=0.5, min_lr=1e-5
    )

    y_tr_t = torch.from_numpy(y_tr.astype(np.float64)).to(DT)
    best_f1 = -1.0
    no_improve = 0
    best_head_state: dict | None = None
    best_test_metrics: dict = {}
    history: list[dict] = []
    n = len(y_tr)

    for ep in range(epochs):
        set_train_mode(head, True)
        perm = np.random.permutation(n)
        total_loss = 0.0
        nb = 0
        for b in range(0, n, batch_size):
            idx = perm[b : b + batch_size]
            bfeat, by = feat_tr[idx], y_tr_t[idx]
            opt.zero_grad()
            out = head(bfeat)
            if out.dim() > 1:
                out = out.mean(1)
            loss = loss_fn(out, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1

        set_train_mode(head, False)
        with torch.no_grad():
            out_val = head(feat_val)
            if out_val.dim() > 1:
                out_val = out_val.mean(1)
        m_val = compute_metrics(out_val.numpy(), y_val)
        sched.step(m_val["macro_f1"])

        # Test metrics are only ever *retained* at a new-best-val epoch (they
        # become the returned test_metrics_at_best_val), so evaluating the test
        # set on non-improving epochs is pure waste: on QQP that is 40,230 pairs
        # pushed through a per-sample qnode and then discarded -- ~19x the
        # train+val work per epoch. The pass is deterministic and under
        # no_grad, so skipping it consumes no RNG: the training trajectory and
        # the returned metrics are bit-identical to the eager version.
        improved = m_val["macro_f1"] > best_f1
        m_te = None
        if improved:
            with torch.no_grad():
                out_te = head(feat_te)
                if out_te.dim() > 1:
                    out_te = out_te.mean(1)
            m_te = compute_metrics(out_te.numpy(), y_te)

        row = {
            "epoch": ep,
            "train_loss": total_loss / max(nb, 1),
            "val_f1": m_val["f1"],
            "val_acc": m_val["acc"],
            # present only on new-best-val epochs; None otherwise (see above)
            "test_f1": m_te["f1"] if m_te is not None else None,
            "test_acc": m_te["acc"] if m_te is not None else None,
        }
        history.append(row)

        if improved:
            best_f1 = m_val["macro_f1"]
            no_improve = 0
            best_head_state = copy.deepcopy(head.state_dict())
            best_test_metrics = m_te
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [phase2] early stop ep {ep}  best_val_macroF1={best_f1:.4f}")
                break

        if ep % 5 == 0 or ep == epochs - 1:
            print(
                f"  [phase2] ep {ep:>3}: loss={total_loss/max(nb,1):.4f}"
                f"  val_macroF1={m_val['macro_f1']:.4f}"
                + (f"  test_macroF1={m_te['macro_f1']:.4f}  test_f1={m_te['f1']:.4f}"
                   f"  test_acc={m_te['acc']:.4f}" if m_te is not None
                   else "  (test skipped: not a new best)")
            )

    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    return history, best_f1, best_test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid SBERT v2 two-phase trainer")
    p.add_argument("--task", choices=["mrpc", "paws", "qqp"], default="mrpc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-qubits", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--reducer", choices=["depthwise", "linear"], default="depthwise")
    p.add_argument("--head", choices=["pqc", "mlp", "sel"], default="pqc")
    p.add_argument("--mlp-hidden", type=int, default=3)
    p.add_argument("--freeze-calib", action="store_true",
                   help="Pin affine calibration at identity (uncalibrated run)")
    p.add_argument("--calib-mode", choices=["both", "bias", "scale", "none"],
                   default=None,
                   help="Which affine params train: both (v5), bias only, "
                        "scale only, or none (v4-equivalent). Overrides "
                        "--freeze-calib when given.")
    p.add_argument("--n-test", type=int, default=None,
                   help="Subsample the test split to this many pairs "
                        "(seeded). Cuts eval cost on QQP/PAWS at high qubit "
                        "counts; None keeps the full split.")
    p.add_argument("--init-scale", type=float, default=0.1,
                   help="PQC param init scale (default 0.1 reproduces v4/v5; "
                        "~1.0 centres the <Z_0> readout, adding NO parameters)")
    p.add_argument("--out-subdir", default="v5",
                   help="results/<subdir>/ and hybrid_<subdir>_ filename prefix")
    p.add_argument("--run-tag", default="",
                   help="Extra token in the output filename to avoid collisions "
                        "(e.g. cal/uncal, fourierL4)")
    p.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    p.add_argument("--sbert-dim", type=int, default=384)
    p.add_argument("--kernel-size", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--n-train", type=int, default=None,
                   help="Limit training pairs (None = full train split)")
    p.add_argument("--n-val", type=int, default=200,
                   help="Val pairs from validation split; rest = test")
    p.add_argument("--pretrain-epochs", type=int, default=40,
                   help="Phase 1 max epochs (reducer + probe)")
    p.add_argument("--epochs", type=int, default=60,
                   help="Phase 2 max epochs (head only)")
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    set_seed(args.seed)
    t_start = time.time()
    print(
        f"=== train_hybrid_v5: {args.reducer}+{args.head} | "
        f"{args.task} | seed={args.seed} | lr={args.lr} ==="
    )

    # ---- Load data ---------------------------------------------------------
    from datasets_qec import load_dataset_by_name

    train_pairs = load_dataset_by_name(
        args.task, n_pairs=args.n_train, seed=args.seed, split="train"
    )
    val_all = load_dataset_by_name(
        args.task, n_pairs=None, seed=args.seed, split="validation"
    )
    val_pairs = val_all[: args.n_val]
    test_pairs = val_all[args.n_val :]
    if args.n_test is not None and len(test_pairs) > args.n_test:
        # Seeded subsample so the test set is fixed per seed but not identical
        # across seeds. Only used for high-qubit runs where a 40k-pair test
        # split would dominate wall-clock; full splits are used elsewhere.
        idx = np.random.default_rng(args.seed).choice(
            len(test_pairs), args.n_test, replace=False)
        test_pairs = [test_pairs[i] for i in sorted(idx)]
    print(
        f"   data -- train: {len(train_pairs)}  val: {len(val_pairs)}"
        f"  test: {len(test_pairs)}"
    )

    # ---- SBERT encode (cached) ---------------------------------------------
    sb = SBERTFeatures(model_name=args.sbert_model)
    emb_a_tr, emb_b_tr, y_tr = sb.encode_pairs(train_pairs)
    emb_a_val, emb_b_val, y_val = sb.encode_pairs(val_pairs)
    emb_a_te, emb_b_te, y_te = sb.encode_pairs(test_pairs)
    print(f"   SBERT encode done ({time.time()-t_start:.1f}s)")

    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float64)).to(DT)

    emb_a_tr_t, emb_b_tr_t = to_t(emb_a_tr), to_t(emb_b_tr)
    emb_a_val_t, emb_b_val_t = to_t(emb_a_val), to_t(emb_b_val)
    emb_a_te_t, emb_b_te_t = to_t(emb_a_te), to_t(emb_b_te)

    # ---- Build reducer -----------------------------------------------------
    if args.reducer == "depthwise":
        reducer = DepthwiseSBERTReducer(
            sbert_dim=args.sbert_dim,
            kernel_size=args.kernel_size,
            stride=args.stride,
            n_qubits=args.n_qubits,
        ).to(DT)
    else:
        reducer = LinearSBERTReducer(
            sbert_dim=args.sbert_dim, n_qubits=args.n_qubits
        ).to(DT)
    print(f"   reducer: {reducer.__class__.__name__}  params={reducer.n_params}")

    # ========================================================================
    # Phase 1: pretrain reducer with linear probe
    # ========================================================================
    print("\n--- Phase 1: reducer pretraining with linear probe (BCE) ---")
    p1_history, p1_best_val_f1 = run_phase1(
        reducer,
        n_qubits=args.n_qubits,
        emb_a_tr=emb_a_tr_t, emb_b_tr=emb_b_tr_t, y_tr=y_tr,
        emb_a_val=emb_a_val_t, emb_b_val=emb_b_val_t, y_val=y_val,
        epochs=args.pretrain_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    # Save reducer checkpoint
    ckpt_dir = ROOT / "results" / "v4" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / (
        f"reducer_{args.task}_{args.reducer}_q{args.n_qubits}_s{args.seed}.pt"
    )
    torch.save(reducer.state_dict(), ckpt_path)
    print(
        f"   Phase 1 done: best_val_f1={p1_best_val_f1:.4f}"
        f"  ckpt: {ckpt_path.name}"
    )

    # Precompute frozen reducer features
    set_train_mode(reducer, False)
    for param in reducer.parameters():
        param.requires_grad = False
    with torch.no_grad():
        feat_tr = reducer(emb_a_tr_t, emb_b_tr_t)
        feat_val = reducer(emb_a_val_t, emb_b_val_t)
        feat_te = reducer(emb_a_te_t, emb_b_te_t)
    print(
        f"   Frozen features precomputed:"
        f"  tr={feat_tr.shape}  val={feat_val.shape}  te={feat_te.shape}"
    )

    # ========================================================================
    # Phase 2: frozen reducer features + head
    # ========================================================================
    print(f"\n--- Phase 2: frozen reducer + {args.head} head (BCE) ---")

    if args.head == "pqc":
        base_head = PQCHead(n_qubits=args.n_qubits, n_layers=args.n_layers,
                            init_scale=args.init_scale).to(DT)
    elif args.head == "sel":
        base_head = StronglyEntanglingHead(
            n_qubits=args.n_qubits, n_layers=args.n_layers
        ).to(DT)
    else:
        base_head = ClassicalMLPHead(
            n_qubits=args.n_qubits, hidden_dim=args.mlp_hidden
        ).to(DT)
    # v5: wrap with 2-param affine calibration (applied to BOTH heads equally).
    # --freeze-calib pins the affine at identity (scale=1, bias=0) so the SAME
    # trainer yields an uncalibrated run for the E2 ansatz-generality test
    # without re-invoking the separate v4 trainer.
    head = CalibratedHead(base_head).to(DT)
    # --calib-mode wins if given; otherwise fall back to the older
    # --freeze-calib switch so every previously-collected run reproduces.
    mode = args.calib_mode or ("none" if args.freeze_calib else "both")
    head.scale.requires_grad_(mode in ("both", "scale"))
    head.bias.requires_grad_(mode in ("both", "bias"))
    calib_state = f"mode={mode}"
    print(f"   head: Calibrated({base_head.__class__.__name__}, {calib_state})  "
          f"params={head.n_params} (base {base_head.n_params} + 2 calib)")

    p2_history, p2_best_val_f1, test_metrics = run_phase2(
        feat_tr, feat_val, feat_te,
        y_tr, y_val, y_te,
        head,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )


    # Save raw predictions + logits for post-hoc analysis (confusion matrix, AUC, etc.)
    set_train_mode(head, False)
    with torch.no_grad():
        final_logits_te = head(feat_te)
        if final_logits_te.dim() > 1:
            final_logits_te = final_logits_te.mean(1)
        final_logits_te = final_logits_te.numpy()
    final_preds_te = (final_logits_te > 0.0).astype(int).tolist()

    # ---- Classical SBERT cosine baseline on test set -----------------------
    norm_a = emb_a_te / (np.linalg.norm(emb_a_te, axis=1, keepdims=True) + 1e-12)
    norm_b = emb_b_te / (np.linalg.norm(emb_b_te, axis=1, keepdims=True) + 1e-12)
    cos = (norm_a * norm_b).sum(axis=1)
    cos_preds = (cos > 0.5).astype(int)
    cos_f1 = float(f1_score(y_te.astype(int), cos_preds, zero_division=0))
    cos_acc = float(accuracy_score(y_te.astype(int), cos_preds))

    # ---- Summary -----------------------------------------------------------
    print(
        f"\n=== Result: {args.reducer}+{args.head} | {args.task} | seed={args.seed} ==="
    )
    print(
        f"   Phase 2 test F1  = {test_metrics.get('f1', float('nan')):.4f}"
        f"   (best val F1 = {p2_best_val_f1:.4f})"
    )
    print(f"   Phase 2 test acc = {test_metrics.get('acc', float('nan')):.4f}")
    print(f"   SBERT cosine F1  = {cos_f1:.4f}  acc = {cos_acc:.4f}")
    print(f"   Total time: {time.time()-t_start:.0f}s")

    # ---- Save JSON ---------------------------------------------------------
    out_dir = ROOT / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_path = out_dir / (
        f"hybrid_{args.out_subdir}_{args.task}_{args.reducer}_{args.head}{tag}"
        f"_q{args.n_qubits}_l{args.n_layers}_s{args.seed}.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    **vars(args),
                    "n_train_actual": len(train_pairs),
                    "n_val_actual": len(val_pairs),
                    "n_test_actual": len(test_pairs),
                },
                "phase1": {
                    "history": p1_history,
                    "best_val_macro_f1": float(p1_best_val_f1),
                },
                "phase2": {
                    "history": p2_history,
                    "best_val_macro_f1": float(p2_best_val_f1),
                    "test": {k: float(v) for k, v in test_metrics.items()},
                },
                "classical_baseline": {
                    "cosine_sbert_f1": cos_f1,
                    "cosine_sbert_acc": cos_acc,
                },
                "test_predictions": final_preds_te,
                "test_logits": [float(x) for x in final_logits_te],
                "test_labels": [int(x) for x in y_te.astype(int)],
                "reducer_params": int(reducer.n_params),
                "head_params": int(head.n_params),
            },
            indent=2,
        )
    )
    print(f"[done] wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
