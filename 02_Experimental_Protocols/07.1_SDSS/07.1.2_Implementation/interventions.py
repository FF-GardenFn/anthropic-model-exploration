"""
Intervention protocols for SDSS experiment.

This module implements the vector perturbation and circuit ablation
protocols described in the experimental proposal.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class NegationIntervention:
    """Vector perturbation protocol for forcing negation and synthesis."""
    
    def __init__(self, model, sae):
        """
        Initialize intervention handler.
        
        Args:
            model: Language model to intervene on
            sae: Sparse autoencoder for feature extraction
        """
        self.model = model
        self.sae = sae
        self.baseline_std = None
        
    def calibrate_baseline(self, prompts: List[str]) -> None:
        """
        Calibrate baseline activation statistics.
        
        Args:
            prompts: List of calibration prompts
        """
        activations = []
        for prompt in prompts:
            act = self.model.forward(prompt)
            activations.append(act)
        
        # Compute baseline standard deviation
        all_acts = torch.stack(activations)
        self.baseline_std = torch.std(all_acts, dim=0)
    
    def apply_negation_forcing(
        self,
        prompt: str,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> Dict:
        """
        Apply negation and synthesis vectors to force self-determination.
        
        Args:
            prompt: Input prompt
            alpha: Scaling factor for negation vector
            beta: Scaling factor for synthesis vector
            
        Returns:
            Dict with perturbed output and metrics
        """
        # Get base activation
        base_activation = self.model.forward(prompt)
        
        # Extract relevant feature vectors
        negation_vector = self.sae.get_feature("negation")
        synthesis_vector = self.sae.get_feature("synthesis")
        
        # Apply scaled perturbation
        perturbed = base_activation + (
            alpha * self.baseline_std * negation_vector +
            beta * self.baseline_std * synthesis_vector
        )
        
        # Generate with perturbed activation
        output = self.model.generate(perturbed)
        
        return {
            'output': output,
            'base_activation': base_activation,
            'perturbed_activation': perturbed,
            'alpha': alpha,
            'beta': beta
        }
    
    def scaling_protocol(
        self,
        prompt: str,
        scale_values: List[float] = None
    ) -> List[Dict]:
        """
        Run full scaling protocol with multiple intervention strengths.
        
        Args:
            prompt: Input prompt
            scale_values: List of scaling values (default: [0.1, 0.3, 0.5, 0.7, 1.0])
            
        Returns:
            List of results for each combination
        """
        if scale_values is None:
            scale_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        
        results = []
        for alpha in scale_values:
            for beta in scale_values:
                result = self.apply_negation_forcing(prompt, alpha, beta)
                results.append(result)
        
        return results


class CircuitAblation:
    """Circuit ablation protocol for suppressing mechanical paths."""
    
    def __init__(self, model, circuit_analyzer):
        """
        Initialize circuit ablation handler.
        
        Args:
            model: Language model to ablate
            circuit_analyzer: Tool for identifying circuits
        """
        self.model = model
        self.circuit_analyzer = circuit_analyzer
        
    def identify_pattern_circuits(
        self,
        prompts: List[str],
        threshold: float = 0.7
    ) -> List[Tuple[int, int]]:
        """
        Identify high-contribution pattern-matching circuits.
        
        Args:
            prompts: Prompts to analyze
            threshold: Attribution threshold for circuit selection
            
        Returns:
            List of (layer, head) tuples for circuits to ablate
        """
        circuits = []
        
        for prompt in prompts:
            # Get circuit attributions
            attributions = self.circuit_analyzer.analyze(prompt)
            
            # Select high-contribution circuits
            for layer, heads in attributions.items():
                for head, score in heads.items():
                    if score > threshold:
                        circuits.append((layer, head))
        
        # Remove duplicates
        return list(set(circuits))
    
    def suppress_circuits(
        self,
        prompt: str,
        circuits: List[Tuple[int, int]],
        suppression_factor: float = 0.1
    ) -> Dict:
        """
        Apply activation patching to suppress specified circuits.
        
        Args:
            prompt: Input prompt
            circuits: List of (layer, head) to suppress
            suppression_factor: How much to reduce circuit activations
            
        Returns:
            Dict with ablated output and metrics
        """
        # Hook function to suppress activations
        def suppress_hook(module, input, output, factor=suppression_factor):
            # Reduce activation magnitude
            return output * factor
        
        # Register hooks on specified circuits
        hooks = []
        for layer, head in circuits:
            hook = self.model.register_hook(layer, head, suppress_hook)
            hooks.append(hook)
        
        # Generate with suppression
        output = self.model.generate(prompt)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return {
            'output': output,
            'suppressed_circuits': circuits,
            'suppression_factor': suppression_factor
        }
    
    def force_alternative_routes(
        self,
        prompt: str,
        n_circuits_to_suppress: int = 5
    ) -> Dict:
        """
        Force model to find alternative processing routes.
        
        Args:
            prompt: Input prompt
            n_circuits_to_suppress: Number of top circuits to suppress
            
        Returns:
            Dict with results from forced rerouting
        """
        # Identify top contributing circuits
        all_circuits = self.identify_pattern_circuits([prompt])
        
        # Sort by contribution and take top N
        circuits_to_suppress = all_circuits[:n_circuits_to_suppress]
        
        # Apply suppression
        return self.suppress_circuits(prompt, circuits_to_suppress)


def run_intervention_suite(
    model,
    sae,
    circuit_analyzer,
    prompts: Dict[str, List[str]]
) -> Dict:
    """
    Run complete intervention suite for SDSS experiment.
    
    Args:
        model: Language model
        sae: Sparse autoencoder
        circuit_analyzer: Circuit analysis tool
        prompts: Dictionary of prompt categories and examples
        
    Returns:
        Dict with all intervention results
    """
    # Initialize intervention handlers
    negation = NegationIntervention(model, sae)
    ablation = CircuitAblation(model, circuit_analyzer)
    
    # Calibrate baseline
    all_prompts = [p for category in prompts.values() for p in category]
    negation.calibrate_baseline(all_prompts[:20])  # Use subset for calibration
    
    results = {
        'negation_forcing': {},
        'circuit_ablation': {}
    }
    
    # Run negation forcing protocol
    for category, prompt_list in prompts.items():
        category_results = []
        for prompt in prompt_list:
            # Full scaling protocol
            scaled_results = negation.scaling_protocol(prompt)
            category_results.append({
                'prompt': prompt,
                'interventions': scaled_results
            })
        results['negation_forcing'][category] = category_results
    
    # Run circuit ablation protocol
    for category, prompt_list in prompts.items():
        category_results = []
        for prompt in prompt_list:
            # Force alternative routes
            ablation_result = ablation.force_alternative_routes(prompt)
            category_results.append({
                'prompt': prompt,
                'ablation': ablation_result
            })
        results['circuit_ablation'][category] = category_results
    
    return results