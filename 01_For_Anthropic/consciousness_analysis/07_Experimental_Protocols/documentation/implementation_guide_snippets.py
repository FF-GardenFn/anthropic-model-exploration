"""
Implementation Guide Snippets (extracted from documentation/05_IMPLEMENTATION_GUIDE.md)

Purpose: Centralize Python snippets referenced by the implementation guide so
Markdown stays clean. These are illustrative stubs consistent with the guide's
structure; production implementations belong in experiment modules.
"""
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class SDSSConfig:
    models: List[str]
    n_interventions: int
    n_replications: int
    metrics: List[str]
    intervention_strengths: List[float]


class SDSSExperiment:
    def __init__(self, config: SDSSConfig):
        self.config = config
        self.results: Dict[str, Any] = {}
        self.preregistration_locked: bool = False


def collect_baselines(model: Any, prompts: List[str]) -> Dict[str, Any]:
    """Collect baseline activations, metrics, and responses (stub)."""
    return {'activations': [], 'metrics': [], 'responses': []}


def apply_negation_intervention(model: Any, strength: float = 0.5):
    """Register an illustrative intervention hook (stub). Returns a handle-like
    object that supports .remove() in real frameworks.
    """
    class _Handle:
        def remove(self):
            return None
    return _Handle()


def run_intervention_suite(model: Any, prompts: List[str], config: SDSSConfig) -> List[Dict[str, Any]]:
    """Run the intervention protocol (stub)."""
    return []


def compute_semantic_action(activations: Any) -> float:
    """PLSA proxy: semantic action along a trajectory (stub)."""
    return 0.0


def compute_eigengap(activations: Any) -> float:
    """Spectral gap diagnostic (stub)."""
    return 0.0


def compute_angle_preservation(activations: Any) -> float:
    """Angle preservation estimate (stub)."""
    return 0.0


def compute_monodromy_drift(activations: Any) -> float:
    """Monodromy drift over loops (stub)."""
    return 0.0
