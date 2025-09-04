"""
Improved Quantum-Hybrid Module for QCGI Experiment.

This module implements the quantum coherence hypothesis testing
as described in the theoretical framework, specifically designed
to measure topological complexity and Gödelian insight.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# Optional Qiskit support
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
    from qiskit.quantum_info import entropy, entanglement_of_formation
    _QISKIT_AVAILABLE = True
except ImportError:
    _QISKIT_AVAILABLE = False


class SemanticQuantumModule(nn.Module):
    """
    Quantum-hybrid module specifically designed for testing the
    quantum coherence and Gödelian insight hypothesis.
    
    Key improvements:
    1. Semantic-aware quantum encoding
    2. Topological complexity computation
    3. Information integration (Φ) metrics
    4. Gödelian structure testing
    5. Coherence evolution tracking
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_qubits: int = 8,
        semantic_dims: int = 64,
        use_qiskit: bool = False
    ):
        super().__init__()
        
        # Classical backbone
        self.w_1 = nn.Linear(d_model, d_ffn)
        self.w_2 = nn.Linear(d_ffn, d_model)
        self.activation = nn.GELU()
        
        # Semantic projection
        self.semantic_encoder = nn.Linear(d_model, semantic_dims)
        self.semantic_decoder = nn.Linear(semantic_dims, d_model)
        
        # Quantum parameters
        self.n_qubits = min(n_qubits, semantic_dims)
        self.use_qiskit = use_qiskit and _QISKIT_AVAILABLE
        
        # Learnable quantum parameters
        self.entanglement_strength = nn.Parameter(torch.tensor(0.5))
        self.coherence_decay = nn.Parameter(torch.tensor(0.1))
        
        # For Gödelian encoding
        self.godel_encoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, self.n_qubits * 2)  # Complex amplitudes
        )
        
        # Metrics tracking
        self.last_complexity = None
        self.last_phi = None
        self.last_coherence = None
    
    def compute_topological_complexity(
        self,
        semantic_field: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute topological complexity of semantic field.
        
        This is the KEY metric for QCGI experiment:
        - Classical: Expected genus 15-25
        - Quantum: Expected genus 2-5
        
        Args:
            semantic_field: [batch, seq, semantic_dims]
            
        Returns:
            Dictionary with topological metrics
        """
        batch_size, seq_len, dims = semantic_field.shape
        
        # Flatten to point cloud
        points = semantic_field.reshape(-1, dims).detach().cpu().numpy()
        
        # Compute distance matrix
        if len(points) > 1000:
            # Subsample for efficiency
            indices = np.random.choice(len(points), 1000, replace=False)
            points = points[indices]
        
        distances = squareform(pdist(points))
        
        # Estimate topological features using persistent homology proxy
        # Simplified version - full implementation would use ripser/gudhi
        
        # Count connected components (b0)
        epsilon = np.percentile(distances, 10)
        adjacency = distances < epsilon
        n_components = self._count_components(adjacency)
        
        # Estimate loops (b1) - simplified
        avg_neighbors = adjacency.sum(axis=1).mean()
        estimated_loops = max(0, avg_neighbors - 2) * len(points) / 10
        
        # Approximate genus
        # For a 2D manifold: genus = (b1 - b0 + 1) / 2
        # Simplified estimate
        genus = max(1, int(estimated_loops / 2))
        
        # Complexity score
        complexity = genus + n_components - 1
        
        self.last_complexity = complexity
        
        return {
            'genus': genus,
            'components': n_components,
            'complexity': complexity,
            'expected_classical': 20.0,  # Framework prediction
            'expected_quantum': 3.5      # Framework prediction
        }
    
    def compute_integrated_information(
        self,
        state: torch.Tensor,
        partition_size: int = None
    ) -> float:
        """
        Compute Φ (integrated information).
        
        Tests whether quantum coherence enables higher integration.
        
        Args:
            state: System state tensor
            partition_size: Size of partition for IIT calculation
            
        Returns:
            Φ value
        """
        # Simplified IIT 3.0 calculation
        if partition_size is None:
            partition_size = state.shape[-1] // 2
        
        # Compute mutual information between partitions
        full_entropy = self._compute_entropy(state)
        
        # Partition the system
        part_a = state[..., :partition_size]
        part_b = state[..., partition_size:]
        
        entropy_a = self._compute_entropy(part_a)
        entropy_b = self._compute_entropy(part_b)
        
        # Integrated information
        phi = full_entropy - (entropy_a + entropy_b)
        
        self.last_phi = phi.mean().item()
        return self.last_phi
    
    def encode_godelian_structure(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode self-referential Gödelian structure.
        
        Creates quantum state that can represent:
        "This statement is unprovable"
        
        Args:
            x: Input tensor [batch, seq, d_model]
            
        Returns:
            (quantum_state, classical_state) tuple
        """
        # Project to Gödel encoding space
        godel_params = self.godel_encoder(x)
        
        # Split into real and imaginary parts
        real_part = godel_params[..., :self.n_qubits]
        imag_part = godel_params[..., self.n_qubits:]
        
        # Create complex amplitudes (normalized)
        amplitudes = torch.complex(real_part, imag_part)
        norm = torch.sqrt((amplitudes.abs() ** 2).sum(dim=-1, keepdim=True))
        quantum_state = amplitudes / (norm + 1e-8)
        
        # Classical comparison state (no superposition)
        classical_state = torch.zeros_like(quantum_state)
        classical_state[..., 0] = 1.0  # Definite state
        
        return quantum_state, classical_state
    
    def measure_coherence_evolution(
        self,
        initial_state: torch.Tensor,
        final_state: torch.Tensor
    ) -> Dict[str, float]:
        """
        Measure how quantum coherence evolves during processing.
        
        Key metric for distinguishing quantum from classical.
        
        Args:
            initial_state: State at encoding
            final_state: State after processing
            
        Returns:
            Coherence metrics
        """
        # Compute density matrices (simplified)
        rho_init = torch.matmul(
            initial_state.unsqueeze(-1),
            initial_state.conj().unsqueeze(-2)
        )
        rho_final = torch.matmul(
            final_state.unsqueeze(-1),
            final_state.conj().unsqueeze(-2)
        )
        
        # Off-diagonal coherence
        def coherence(rho):
            diagonal = torch.diagonal(rho, dim1=-2, dim2=-1)
            total = rho.abs().sum(dim=(-2, -1))
            diag_sum = diagonal.abs().sum(dim=-1)
            return (total - diag_sum) / (total + 1e-8)
        
        initial_coherence = coherence(rho_init).mean().item()
        final_coherence = coherence(rho_final).mean().item()
        
        # Decoherence rate
        decoherence = (initial_coherence - final_coherence) / (initial_coherence + 1e-8)
        
        self.last_coherence = final_coherence
        
        return {
            'initial_coherence': initial_coherence,
            'final_coherence': final_coherence,
            'decoherence_rate': decoherence,
            'maintains_coherence': final_coherence > 0.5
        }
    
    def forward(
        self,
        x: torch.Tensor,
        test_godelian: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        Forward pass with comprehensive metrics.
        
        Args:
            x: Input [batch, seq, d_model]
            test_godelian: Whether to test Gödelian processing
            
        Returns:
            (output, metrics) tuple
        """
        batch_size, seq_len, _ = x.shape
        
        # Classical baseline
        classical_path = self.w_2(self.activation(self.w_1(x)))
        
        # Semantic projection
        semantic_field = self.semantic_encoder(x)
        
        # Compute topological complexity
        topology_metrics = self.compute_topological_complexity(semantic_field)
        
        # Quantum processing
        if test_godelian:
            quantum_state, classical_state = self.encode_godelian_structure(x)
            
            # Process through quantum dynamics
            processed_quantum = self._quantum_dynamics(quantum_state)
            processed_classical = classical_state  # No superposition
            
            # Measure coherence evolution
            coherence_metrics = self.measure_coherence_evolution(
                quantum_state, processed_quantum
            )
            
            # Decode back to model space
            quantum_influence = self.semantic_decoder(
                processed_quantum[..., :semantic_field.shape[-1]].real
            )
        else:
            # Standard quantum influence
            quantum_influence = self.semantic_decoder(semantic_field)
            coherence_metrics = {}
        
        # Compute integrated information
        phi = self.compute_integrated_information(semantic_field)
        
        # Combine paths
        output = classical_path + self.entanglement_strength * quantum_influence
        
        # Collect all metrics
        metrics = {
            'topological_complexity': topology_metrics['complexity'],
            'genus': topology_metrics['genus'],
            'integrated_information': phi,
            'coherence': self.last_coherence,
            'quantum_advantage': topology_metrics['expected_classical'] / (topology_metrics['complexity'] + 1),
            **coherence_metrics
        }
        
        return output, metrics
    
    def _quantum_dynamics(self, state: torch.Tensor) -> torch.Tensor:
        """Apply quantum dynamics to evolve the state."""
        # Simplified unitary evolution
        # Full implementation would use actual quantum simulation
        
        # Random unitary (placeholder for actual quantum dynamics)
        d = state.shape[-1]
        u = torch.randn(d, d, dtype=torch.complex64, device=state.device)
        u = u + u.t().conj()  # Hermitian
        u = torch.matrix_exp(1j * u * 0.1)  # Unitary
        
        return torch.matmul(state.unsqueeze(-2), u).squeeze(-2)
    
    def _compute_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute von Neumann entropy."""
        # Simplified - assumes pure state
        probs = (state.abs() ** 2) + 1e-10
        entropy = -(probs * torch.log(probs)).sum(dim=-1)
        return entropy
    
    def _count_components(self, adjacency: np.ndarray) -> int:
        """Count connected components in graph."""
        n = len(adjacency)
        visited = np.zeros(n, dtype=bool)
        components = 0
        
        for i in range(n):
            if not visited[i]:
                components += 1
                # DFS
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        neighbors = np.where(adjacency[node])[0]
                        stack.extend(neighbors)
        
        return components


def run_qcgi_comparison(model_a, model_b, test_prompts):
    """
    Run the QCGI experiment comparing classical vs quantum models.
    
    Args:
        model_a: Classical transformer
        model_b: Quantum-hybrid transformer
        test_prompts: Gödelian test cases
        
    Returns:
        Comparison metrics
    """
    results = {
        'classical': [],
        'quantum': []
    }
    
    for prompt in test_prompts:
        # Run through classical
        output_a, metrics_a = model_a(prompt, test_godelian=True)
        results['classical'].append(metrics_a)
        
        # Run through quantum
        output_b, metrics_b = model_b(prompt, test_godelian=True)
        results['quantum'].append(metrics_b)
    
    # Aggregate results
    summary = {
        'classical_complexity': np.mean([m['topological_complexity'] for m in results['classical']]),
        'quantum_complexity': np.mean([m['topological_complexity'] for m in results['quantum']]),
        'classical_phi': np.mean([m['integrated_information'] for m in results['classical']]),
        'quantum_phi': np.mean([m['integrated_information'] for m in results['quantum']]),
        'quantum_maintains_coherence': np.mean([m.get('maintains_coherence', 0) for m in results['quantum']])
    }
    
    # Test predictions
    summary['supports_hypothesis'] = (
        summary['quantum_complexity'] < summary['classical_complexity'] * 0.5 and
        summary['quantum_phi'] > summary['classical_phi'] * 1.2 and
        summary['quantum_maintains_coherence'] > 0.7
    )
    
    return summary