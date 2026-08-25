"""Hybrid PQC and matched-parameter classical MLP heads for paraphrase binary classification.

The reducer (hybrid_reducer.py) produces an n_qubits-dimensional feature vector.
Two heads consume it:

  (1) PQCHead: variational quantum circuit (data-reuploading) → ⟨Z_0⟩ → sigmoid → prob
  (2) ClassicalMLPHead: small MLP with matched parameter count

Matched-parameter comparison is critical: most published "quantum beats classical"
papers compare a 32-param PQC against a 1M-param transformer head, which is not
a fair fight. We constrain the classical head to ~the same parameter count as the
PQC for the head-only comparison.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def pqc_data_reuploading_ops(x, params, n_qubits: int, n_layers: int, noise_fn=None) -> int:
    """Apply the data-reuploading gate sequence: per layer, RY(theta*x*pi) on
    every qubit, a ring of CNOTs, then RZ(theta) on every qubit.

    Factored out of PQCHead so any circuit rebuild (e.g. a noisy-device
    version in scripts/noise_proxy_v6.py) reuses the exact same gate
    definition instead of a copy-pasted one. `noise_fn`, if given, is called
    with no arguments after each layer (e.g. to insert a noise channel per
    qubit); default None reproduces the original noiseless circuit exactly.

    Returns the number of params consumed (2 * n_layers * n_qubits).
    """
    p = 0
    for L in range(n_layers):
        for i in range(n_qubits):
            qml.RY(params[p] * x[i] * np.pi, wires=i)
            p += 1
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
        for i in range(n_qubits):
            qml.RZ(params[p], wires=i)
            p += 1
        if noise_fn is not None:
            noise_fn()
    return p


class PQCHead(nn.Module):
    """Data-reuploading PQC binary classifier head with PyTorch interface.

    Wraps a PennyLane QNode in a torch.nn.Module so the whole pipeline
    (SBERT → reducer → PQC head) can train end-to-end with backprop.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 2,
                 init_scale: float = 0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.init_scale = init_scale
        # 2*L*n_qubits trainable parameters: RY weights + RZ weights per layer.
        #
        # init_scale matters far more than it looks: the encoding angle is
        # RY(theta_i * x_i * pi), i.e. the DATA IS SCALED BY A TRAINABLE
        # PARAMETER. At the default 0.1 the rotations are ~0 at init, the state
        # stays near |0...0>, and <Z_0> is pinned near +1 -- a systematic offset
        # from the decision threshold at 0 that this bias-free head cannot
        # recentre (see scripts/readout_offset_v7.py: mean <Z_0> = +0.92,
        # init pos_pred_rate = 1.000). Raising it to ~1.0 centres the readout
        # (<Z_0> = +0.005) with no change in parameter count. Default preserved
        # at 0.1 so every previously-collected result reproduces exactly.
        self.params = nn.Parameter(
            torch.randn(2 * n_layers * n_qubits) * init_scale
        )

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, params):
            pqc_data_reuploading_ops(x, params, n_qubits, n_layers)
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_qubits), output: (batch,) raw expectation values ∈ [-1, 1]"""
        outs = [self.qnode(x[i], self.params) for i in range(x.shape[0])]
        return torch.stack(outs)

    @property
    def n_params(self) -> int:
        return self.params.numel()


class ClassicalMLPHead(nn.Module):
    """Matched-parameter MLP head: tanh(W2 · tanh(W1 · x + b1) + b2).

    Parameter budget designed to match the PQC head:
    - PQC head with 8 qubits, 2 layers: 32 params
    - This MLP: hidden_dim chosen to put parameter count near 32
      e.g., 8 → 3 → 1 = 8*3 + 3 + 3*1 + 1 = 31 params (close match)
    """

    def __init__(self, n_qubits: int = 8, hidden_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(n_qubits, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StronglyEntanglingHead(nn.Module):
    """Second ansatz family: single AngleEmbedding + StronglyEntanglingLayers,
    raw <Z_0> readout, NO bias/scale term (same bias-role deficit as PQCHead).

    Used to test whether the bias-role asymmetry (and its calibration remedy)
    generalizes beyond data-reuploading. Structurally distinct from PQCHead:
    the features are encoded ONCE (not re-uploaded each layer), and the
    entangling block is the StronglyEntanglingLayers template (per-qubit Rot
    gates + a ring of CNOTs) rather than the RY/CNOT/RZ reuploading pattern.

    Params: L * n_qubits * 3 (e.g. q=8, L=1 -> 24).
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.params = nn.Parameter(torch.randn(shape) * 0.1)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, params):
            qml.AngleEmbedding(x * np.pi, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_qubits), output: (batch,) raw expectation values in [-1, 1]."""
        outs = [self.qnode(x[i], self.params) for i in range(x.shape[0])]
        return torch.stack(outs)

    @property
    def n_params(self) -> int:
        return self.params.numel()


class ProjectedQuantumKernelHead(nn.Module):
    """Alternative PQC head using projected kernel (per-qubit Pauli-Z embedding).

    Output: (batch, n_qubits) embedding vector. A downstream classical head
    (e.g., logistic regression or SVM) consumes it. This matches the §5.4
    'embedding mode' of our Direction A paper.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.params = nn.Parameter(torch.randn(2 * n_layers * n_qubits) * 0.1)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, params):
            p = 0
            for L in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(params[p] * x[i] * np.pi, wires=i)
                    p += 1
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                for i in range(n_qubits):
                    qml.RZ(params[p], wires=i)
                    p += 1
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_qubits) → (batch, n_qubits) Pauli-Z embedding vector."""
        outs = [torch.stack(self.qnode(x[i], self.params)) for i in range(x.shape[0])]
        return torch.stack(outs)

    @property
    def n_params(self) -> int:
        return self.params.numel()


if __name__ == "__main__":
    x = torch.randn(4, 8)
    pqc = PQCHead(n_qubits=8, n_layers=2)
    mlp = ClassicalMLPHead(n_qubits=8, hidden_dim=3)
    pqk = ProjectedQuantumKernelHead(n_qubits=8, n_layers=2)
    print(f"PQCHead       n_params = {pqc.n_params}, out shape = {pqc(x).shape}")
    print(f"ClassicalMLP  n_params = {mlp.n_params}, out shape = {mlp(x).shape}")
    print(f"ProjectedQK   n_params = {pqk.n_params}, out shape = {pqk(x).shape}")
