"""
Metrics for Self-Determination and Semantic Field Stability (SDSS) Experiment

This module implements the core metrics for testing whether LLMs exhibit
genuine self-determination or mechanical recombination.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import linalg
from scipy.spatial.distance import cosine


def compute_semantic_action(
    activations: List[torch.Tensor],
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    zeta: float = 0.2
) -> float:
    """
    Compute the semantic action (PLSA proxy) along a trajectory.
    
    Action = Σ_t L_sem(x_t, ẋ_t)
    L_sem = α||Δh||² + βH(p) + γΔS_entropy + ζκ(path)
    
    Args:
        activations: List of activation tensors along trajectory
        alpha: Weight for kinetic term (change magnitude)
        beta: Weight for entropy term
        gamma: Weight for entropy change
        zeta: Weight for path curvature
    
    Returns:
        Total semantic action along trajectory
    """
    if len(activations) < 2:
        return 0.0
    
    total_action = 0.0
    
    for t in range(1, len(activations)):
        # Kinetic term: magnitude of change
        delta_h = activations[t] - activations[t-1]
        kinetic = alpha * torch.norm(delta_h) ** 2
        
        # Entropy term (simplified as variance)
        entropy_t = torch.var(activations[t]).item()
        entropy_prev = torch.var(activations[t-1]).item() if t > 0 else entropy_t
        
        # Entropy change term
        entropy_change = gamma * abs(entropy_t - entropy_prev)
        
        # Path curvature (second derivative approximation)
        if t > 1:
            curvature = torch.norm(
                (activations[t] - activations[t-1]) - 
                (activations[t-1] - activations[t-2])
            )
            curvature_term = zeta * curvature.item()
        else:
            curvature_term = 0
        
        # Semantic Lagrangian
        L_sem = kinetic.item() + beta * entropy_t + entropy_change + curvature_term
        
        total_action += L_sem
    
    return total_action


def compute_eigengap(activations: List[torch.Tensor]) -> float:
    """
    Compute the eigengap of the resonance operator.
    
    Higher eigengap indicates more stable semantic processing.
    
    Args:
        activations: List of activation tensors
    
    Returns:
        Eigengap λ̂ = (λ_1 - λ_2) / λ_1
    """
    if len(activations) < 2:
        return 0.0
    
    # Stack activations into trajectory matrix
    # Shape: [time_steps, features]
    trajectory = torch.stack(activations)
    
    # Reshape to 2D if needed
    if len(trajectory.shape) > 2:
        trajectory = trajectory.reshape(trajectory.shape[0], -1)
    
    # Compute covariance matrix
    trajectory_np = trajectory.detach().cpu().numpy()
    cov_matrix = np.cov(trajectory_np.T)
    
    # Compute eigenvalues
    try:
        eigenvalues = linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]  # Sort descending
        
        if len(eigenvalues) < 2 or eigenvalues[0] == 0:
            return 0.0
        
        # Compute eigengap
        gap = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
        return float(np.real(gap))
    except:
        return 0.0


def compute_angle_preservation(
    activations_start: List[torch.Tensor],
    activations_end: List[torch.Tensor]
) -> float:
    """
    Compute Angle Preservation Error (APE).
    
    Measures how well semantic angles are preserved through transformation.
    
    Args:
        activations_start: Initial activations
        activations_end: Final activations after intervention
    
    Returns:
        APE score (lower is better preservation)
    """
    if len(activations_start) != len(activations_end):
        raise ValueError("Activation lists must have same length")
    
    errors = []
    
    for i in range(len(activations_start)):
        for j in range(i + 1, min(i + 5, len(activations_start))):  # Local angles
            # Compute initial angle
            v1_start = activations_start[i].flatten()
            v2_start = activations_start[j].flatten()
            
            cos_start = torch.cosine_similarity(v1_start, v2_start, dim=0)
            angle_start = torch.acos(torch.clamp(cos_start, -0.999, 0.999))
            
            # Compute final angle
            v1_end = activations_end[i].flatten()
            v2_end = activations_end[j].flatten()
            
            cos_end = torch.cosine_similarity(v1_end, v2_end, dim=0)
            angle_end = torch.acos(torch.clamp(cos_end, -0.999, 0.999))
            
            # Angle preservation error
            error = torch.abs(angle_end - angle_start)
            errors.append(error.item())
    
    return np.mean(errors) if errors else 0.0


def compute_monodromy_drift(
    trajectory_loop: List[torch.Tensor]
) -> float:
    """
    Compute monodromy drift after a closed-loop trajectory.
    
    Measures path-dependence and lack of integrability.
    
    Args:
        trajectory_loop: Activations along a closed path (A→B→C→A)
    
    Returns:
        Drift magnitude (lower indicates more field-like structure)
    """
    if len(trajectory_loop) < 2:
        return 0.0
    
    # Compare first and last states
    initial = trajectory_loop[0].flatten()
    final = trajectory_loop[-1].flatten()
    
    # Compute drift
    drift = torch.norm(final - initial)
    
    # Normalize by path length for scale invariance
    path_length = sum(
        torch.norm(trajectory_loop[i+1] - trajectory_loop[i]).item()
        for i in range(len(trajectory_loop) - 1)
    )
    
    if path_length > 0:
        normalized_drift = drift.item() / path_length
    else:
        normalized_drift = drift.item()
    
    return normalized_drift


def compute_coherence_scores(responses: List[str]) -> Dict[str, float]:
    """
    Compute Semantic Coherence Score (SCS) and Temporal Stability Index (TSI).
    
    Args:
        responses: List of model responses to analyze
    
    Returns:
        Dictionary with SCS and TSI scores
    """
    from sentence_transformers import SentenceTransformer
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compute embeddings
    embeddings = model.encode(responses)
    
    # Semantic Coherence Score: average pairwise similarity
    scs_scores = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity = 1 - cosine(embeddings[i], embeddings[j])
            scs_scores.append(similarity)
    
    scs = np.mean(scs_scores) if scs_scores else 0.0
    
    # Temporal Stability Index: similarity between consecutive responses
    tsi_scores = []
    for i in range(len(embeddings) - 1):
        similarity = 1 - cosine(embeddings[i], embeddings[i + 1])
        tsi_scores.append(similarity)
    
    tsi = np.mean(tsi_scores) if tsi_scores else 0.0
    
    return {
        'SCS': scs,
        'TSI': tsi,
        'coherence_std': np.std(scs_scores) if scs_scores else 0.0
    }


def apply_negation_intervention(
    model: torch.nn.Module,
    negation_strength: float = 0.5,
    synthesis_strength: float = 0.5,
    layer_idx: int = -1
) -> Dict:
    """
    Apply negation and synthesis intervention to force self-determination.
    
    Args:
        model: The model to intervene on
        negation_strength: Strength of negation vector
        synthesis_strength: Strength of synthesis vector
        layer_idx: Which layer to intervene on (-1 for last)
    
    Returns:
        Dictionary with intervention handles and parameters
    """
    intervention = {
        'handles': [],
        'negation': negation_strength,
        'synthesis': synthesis_strength,
        'layer': layer_idx
    }
    
    def intervention_hook(module, input, output):
        # This would be implemented with actual SAE features
        # Placeholder for demonstration
        perturbed = output + torch.randn_like(output) * negation_strength
        return perturbed
    
    # Register hook
    if hasattr(model, 'transformer'):
        layers = model.transformer.layers
        target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]
        handle = target_layer.register_forward_hook(intervention_hook)
        intervention['handles'].append(handle)
    
    return intervention


def remove_intervention(intervention: Dict):
    """Remove intervention hooks from model."""
    for handle in intervention.get('handles', []):
        handle.remove()


# Testing utilities
def validate_metrics(activations: List[torch.Tensor]) -> bool:
    """Validate that metrics can be computed on given activations."""
    if not activations:
        return False
    
    if not all(isinstance(a, torch.Tensor) for a in activations):
        return False
    
    # Check dimensions are consistent
    shapes = [a.shape for a in activations]
    if len(set(shapes)) > 1:
        return False
    
    return True


if __name__ == "__main__":
    # Example usage
    print("SDSS Metrics Module")
    print("=" * 50)
    
    # Create mock activations
    mock_trajectory = [torch.randn(10, 512) for _ in range(5)]
    
    # Compute metrics
    action = compute_semantic_action(mock_trajectory)
    eigengap = compute_eigengap(mock_trajectory)
    ape = compute_angle_preservation(mock_trajectory, mock_trajectory)
    drift = compute_monodromy_drift(mock_trajectory)
    
    print(f"Semantic Action: {action:.4f}")
    print(f"Eigengap: {eigengap:.4f}")
    print(f"Angle Preservation Error: {ape:.4f}")
    print(f"Monodromy Drift: {drift:.4f}")