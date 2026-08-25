"""Depthwise + pointwise feature reducer for SBERT embeddings.

Design (Howard 2017 / Chollet 2017 depthwise-separable conv adapted to NLP):

For a paraphrase pair (sent_A, sent_B) → SBERT embeddings (e_A, e_B) ∈ R^{384}:

  1. Stack: shape (2, 384) — two channels (A, B), 384 dim each.
  2. Depthwise Conv1d: kernel=k, stride=s, groups=2 (one filter per channel)
     → shape (2, L_out), with L_out = (384 - k) / s + 1.
  3. Pointwise Conv1d: 1x1 conv mixing the 2 channels → shape (1, L_out).
  4. Optional non-linearity + LayerNorm.
  5. Flatten + Linear project to n_qubits dimensions.

This preserves SBERT's per-dimension structure locally (depthwise) before
mixing channels (pointwise), and learns the projection to the qubit-count
ceiling end-to-end with the downstream head.

Total trainable parameters: 2*k (depthwise) + 2*1 (pointwise) + L_out * n_qubits
(linear projection). For k=32, n_qubits=8, L_out = (384-32)/32 + 1 = 12:
   2*32 + 2*1 + 12*8 = 64 + 2 + 96 = 162 trainable parameters.

PennyLane PQC head with 8 qubits, 2 layers, data-reuploading has 2*L*n = 32
parameters. So the reducer parameter count dwarfs the PQC parameter count —
this is intentional, since the reducer is doing the heavy lifting of
adapting SBERT to the quantum-feasible dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSBERTReducer(nn.Module):
    """Depthwise + pointwise 1D conv reducer with linear projection to n_qubits."""

    def __init__(
        self,
        sbert_dim: int = 384,
        kernel_size: int = 32,
        stride: int = 32,
        n_qubits: int = 8,
        use_layernorm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        self.sbert_dim = sbert_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = n_qubits

        # Depthwise: separate filter per channel (channel = paper sentence A or B)
        self.depthwise = nn.Conv1d(
            in_channels=2, out_channels=2,
            kernel_size=kernel_size, stride=stride, groups=2, bias=True,
        )
        # Pointwise: mix the two channels into one
        self.pointwise = nn.Conv1d(
            in_channels=2, out_channels=1, kernel_size=1, bias=True,
        )

        # Compute L_out
        self.L_out = (sbert_dim - kernel_size) // stride + 1

        self.norm = nn.LayerNorm(self.L_out) if use_layernorm else nn.Identity()
        self.act = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}.get(
            activation, nn.GELU()
        )
        # Linear projection: L_out → n_qubits (the qubit-count ceiling)
        self.project = nn.Linear(self.L_out, n_qubits)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """
        emb_a, emb_b: (batch, sbert_dim) — SBERT embeddings of sentence pair
        Returns: (batch, n_qubits) — reduced feature vector ready for PQC / MLP head
        """
        # Stack as (batch, 2, sbert_dim)
        x = torch.stack([emb_a, emb_b], dim=1)
        # Depthwise → (batch, 2, L_out)
        x = self.depthwise(x)
        x = self.act(x)
        # Pointwise → (batch, 1, L_out)
        x = self.pointwise(x)
        # Squeeze channel → (batch, L_out)
        x = x.squeeze(1)
        x = self.norm(x)
        x = self.act(x)
        # Linear → (batch, n_qubits)
        x = self.project(x)
        # Bound to a reasonable range for PQC angle encoding (downstream expects ~[-1, 1])
        return torch.tanh(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearSBERTReducer(nn.Module):
    """Baseline: plain Linear projection from concat(emb_a, emb_b) to n_qubits."""

    def __init__(self, sbert_dim: int = 384, n_qubits: int = 8):
        super().__init__()
        self.project = nn.Linear(2 * sbert_dim, n_qubits)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb_a, emb_b], dim=1)
        return torch.tanh(self.project(x))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-test
    reducer = DepthwiseSBERTReducer(sbert_dim=384, kernel_size=32, stride=32, n_qubits=8)
    emb_a = torch.randn(4, 384)
    emb_b = torch.randn(4, 384)
    out = reducer(emb_a, emb_b)
    print(f"depthwise reducer: in 2x{384} → out {out.shape}, n_params = {reducer.n_params}")

    linear_red = LinearSBERTReducer(sbert_dim=384, n_qubits=8)
    out2 = linear_red(emb_a, emb_b)
    print(f"linear baseline:   in 2x{384} → out {out2.shape}, n_params = {linear_red.n_params}")
