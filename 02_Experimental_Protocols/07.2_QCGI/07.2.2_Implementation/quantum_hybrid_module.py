import torch
import torch.nn as nn
import numpy as np

# Optional Qiskit support (fallback demo only)
try:
    from qiskit import QuantumCircuit  # type: ignore
    from qiskit.quantum_info import Statevector  # type: ignore
    _QISKIT_AVAILABLE = True
except Exception:  # ImportError or environment issues
    QuantumCircuit = None  # type: ignore
    Statevector = None  # type: ignore
    _QISKIT_AVAILABLE = False

# Simulator only used if Qiskit is available; keep None otherwise
simulator = None

class QuantumHybridModule(nn.Module):
    """
    A simulated Quantum-Hybrid FFN replacement for a Transformer block.

    Enhancements:
    - Vectorized forward pass for the classical path and the default quantum proxy.
    - Differentiable, learnable quantum proxy that avoids tensor .detach() and supports autograd.
    - Optional Qiskit fallback for demonstration (non-differentiable, slower).
    """
    def __init__(self, d_model: int, d_ffn: int, n_qubits: int = 8, use_qiskit: bool = False):
        super().__init__()
        # Standard classical linear layers of a Feed-Forward Network
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()

        # Quantum parameters
        self.n_qubits = min(n_qubits, d_model)  # Number of "qubits" can't exceed model dimension
        # Enable Qiskit fallback only if available and explicitly requested
        self.use_qiskit = bool(use_qiskit and _QISKIT_AVAILABLE)

        # A learnable classical parameter that will be influenced by the quantum measurement
        self.quantum_modulation_factor = nn.Parameter(torch.randn(1))

        # Parameters for differentiable quantum-like proxy
        # Per-qubit angle parameters and a global scaling to control influence
        self.theta = nn.Parameter(torch.randn(self.n_qubits))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    # -------- Qiskit (non-differentiable) helpers --------
    def _encode_classical_to_quantum(self, x: torch.Tensor) -> QuantumCircuit:
        """Encodes a classical vector into the initial state of a quantum circuit (non-differentiable)."""
        qc = QuantumCircuit(self.n_qubits)
        # Amplitude encoding on first n_qubits dims (uses numpy, breaks autograd intentionally)
        vec = x[:self.n_qubits].detach().cpu().numpy()
        norm = np.linalg.norm(vec) or 1.0
        initial_state = vec / norm
        qc.initialize(initial_state, range(self.n_qubits))
        return qc

    def _apply_quantum_dynamics(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Applies entanglement and superposition to simulate coherent evolution (demo only)."""
        for i in range(self.n_qubits - 1):
            qc.h(i)
            qc.cx(i, i + 1)
        qc.barrier()
        return qc

    def _measure_global_property(self, qc: QuantumCircuit) -> float:
        """Measure a simple global property from the final statevector (demo-only scalar)."""
        final_state = Statevector.from_instruction(qc)
        global_property = np.abs(final_state.data[0]) ** 2
        return float(global_property)

    # -------- Differentiable quantum-like proxy --------
    def _differentiable_quantum_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a vectorized, differentiable scalar in [0, 1] per token that acts as a proxy
        for a global "quantum" property. Uses learnable parameters and PyTorch ops only.

        Args:
            x: Tensor of shape [..., d_model]
        Returns:
            scalar: Tensor of shape [..., 1]
        """
        # Take the first n_qubits features and normalize
        v = x[..., : self.n_qubits]
        eps = 1e-8
        norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
        a = v / norm  # amplitude-like, differentiable

        # Angle encoding with learnable per-qubit parameters
        # rot_i = a_i * theta_i; expectation of Z ~ cos(rot_i)
        rot = a * self.theta  # broadcast over last dim
        exp_z = torch.cos(0.5 * rot)  # smooth, bounded

        # Aggregate across qubits; add a simple pairwise coherence term via mean product
        mean_exp = exp_z.mean(dim=-1, keepdim=True)
        coherence = (exp_z.unsqueeze(-1) * exp_z.unsqueeze(-2)).mean(dim=(-1, -2), keepdim=True)
        # Combine and squash to [0,1]
        raw = self.alpha * (0.7 * mean_exp + 0.3 * coherence)
        scalar = torch.sigmoid(raw)
        return scalar  # shape [..., 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [batch_size, seq_len, d_model]
        Returns: Tensor with same shape as x.
        """
        # Classical path (vectorized over batch and sequence)
        classical_path = self.w_2(self.activation(self.w_1(x)))

        if self.use_qiskit:
            # Non-differentiable, slower fallback using Qiskit per token
            bsz, seqlen, _ = x.shape
            quantum_scalar_list = []
            for b in range(bsz):
                row = []
                for s in range(seqlen):
                    token_vector = x[b, s, :]
                    qc = self._encode_classical_to_quantum(token_vector)
                    qc = self._apply_quantum_dynamics(qc)
                    row.append(self._measure_global_property(qc))
                quantum_scalar_list.append(row)
            quantum_scalar = torch.tensor(quantum_scalar_list, dtype=classical_path.dtype, device=classical_path.device)
            quantum_scalar = quantum_scalar.unsqueeze(-1)  # [B, S, 1]
        else:
            # Differentiable, vectorized proxy
            quantum_scalar = self._differentiable_quantum_scalar(x)  # [B, S, 1]

        # Use quantum result to modulate the classical path
        quantum_influence = self.quantum_modulation_factor.view(1, 1, 1) * quantum_scalar
        output = classical_path * (1.0 + quantum_influence)
        return output
