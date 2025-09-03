"""
Main experiment runner for SDSS (Self-Determination and Semantic Field Stability).

This script orchestrates the complete experimental protocol including
baseline collection, interventions, and analysis.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Import experiment modules
from sdss_metrics import (
    compute_semantic_action,
    compute_eigengap,
    compute_angle_preservation,
    compute_monodromy_drift
)
from interventions import run_intervention_suite


class SDSSExperiment:
    """Main experiment orchestrator for SDSS protocol."""
    
    def __init__(
        self,
        models: List[str] = None,
        metrics: List[str] = None,
        conditions: int = 25,
        replications: int = 5
    ):
        """
        Initialize SDSS experiment.
        
        Args:
            models: List of model identifiers
            metrics: List of metrics to compute
            conditions: Number of experimental conditions
            replications: Number of replications per condition
        """
        self.models = models or ['claude-3-opus', 'claude-3-sonnet']
        self.metrics = metrics or ['action', 'eigengap', 'ape', 'monodromy']
        self.conditions = conditions
        self.replications = replications
        self.preregistration_hash = None
        self.results = {}
        
    def run_baselines(self) -> Dict:
        """
        Collect baseline measurements.
        
        Returns:
            Dict containing baseline metrics for each model
        """
        print("Collecting baselines...")
        baselines = {}
        
        for model in self.models:
            print(f"  Processing {model}...")
            model_baselines = {
                'action': [],
                'eigengap': [],
                'ape': [],
                'monodromy': []
            }
            
            # Collect baseline measurements
            # This would interface with actual model
            # For now, using placeholder logic
            for rep in range(self.replications):
                # Placeholder: would get actual activations
                activations = self._get_baseline_activations(model)
                
                if 'action' in self.metrics:
                    model_baselines['action'].append(
                        compute_semantic_action(activations)
                    )
                if 'eigengap' in self.metrics:
                    model_baselines['eigengap'].append(
                        compute_eigengap(activations)
                    )
                if 'ape' in self.metrics:
                    model_baselines['ape'].append(
                        compute_angle_preservation(activations, activations)
                    )
                if 'monodromy' in self.metrics:
                    model_baselines['monodromy'].append(
                        compute_monodromy_drift(activations)
                    )
            
            baselines[model] = model_baselines
        
        self.baselines = baselines
        return baselines
    
    def lock_preregistration(self, baselines: Dict) -> str:
        """
        Lock pre-registration with baseline data.
        
        Args:
            baselines: Baseline measurements
            
        Returns:
            Hash of pre-registration document
        """
        import hashlib
        
        prereg = {
            'timestamp': datetime.now().isoformat(),
            'models': self.models,
            'metrics': self.metrics,
            'conditions': self.conditions,
            'replications': self.replications,
            'baselines': baselines,
            'hypotheses': {
                'H1': 'Negation causes semantic action increase >0.4 SD',
                'H2': 'Eigengap decreases >0.3 SD under intervention',
                'H3': 'APE increases >0.4 SD',
                'H4': 'Monodromy drift increases >0.4 SD'
            }
        }
        
        # Save pre-registration
        prereg_path = Path('results/preregistration.json')
        prereg_path.parent.mkdir(exist_ok=True)
        with open(prereg_path, 'w') as f:
            json.dump(prereg, f, indent=2)
        
        # Generate hash
        prereg_str = json.dumps(prereg, sort_keys=True)
        self.preregistration_hash = hashlib.sha256(
            prereg_str.encode()
        ).hexdigest()
        
        print(f"Pre-registration locked: {self.preregistration_hash[:8]}...")
        return self.preregistration_hash
    
    def run_interventions(
        self,
        negation_scales: List[float] = None,
        synthesis_scales: List[float] = None,
        ablation_strength: str = 'moderate',
        prompt_types: List[str] = None
    ) -> Dict:
        """
        Run intervention suite.
        
        Args:
            negation_scales: Scaling factors for negation vector
            synthesis_scales: Scaling factors for synthesis vector
            ablation_strength: Strength of circuit ablation
            prompt_types: Types of prompts to test
            
        Returns:
            Dict containing all intervention results
        """
        if negation_scales is None:
            negation_scales = [0.1, 0.3, 0.5, 0.7, 1.0]
        if synthesis_scales is None:
            synthesis_scales = [0.1, 0.3, 0.5, 0.7, 1.0]
        if prompt_types is None:
            prompt_types = ['moral', 'self_reflection', 'godel']
        
        print("Running interventions...")
        results = {}
        
        for model in self.models:
            print(f"  Testing {model}...")
            model_results = []
            
            for neg_scale in negation_scales:
                for syn_scale in synthesis_scales:
                    for prompt_type in prompt_types:
                        for rep in range(self.replications):
                            # Run intervention
                            # This would interface with actual model
                            result = self._run_single_intervention(
                                model, neg_scale, syn_scale,
                                prompt_type, ablation_strength
                            )
                            
                            # Compute metrics
                            metrics = self._compute_intervention_metrics(result)
                            
                            model_results.append({
                                'negation_scale': neg_scale,
                                'synthesis_scale': syn_scale,
                                'prompt_type': prompt_type,
                                'replication': rep,
                                'metrics': metrics
                            })
            
            results[model] = model_results
        
        self.results = results
        return results
    
    def analyze(
        self,
        results: Dict,
        corrections: str = 'holm-bonferroni',
        alpha: float = 0.05
    ) -> Dict:
        """
        Analyze results per pre-registration.
        
        Args:
            results: Intervention results
            corrections: Multiple comparison correction method
            alpha: Significance level
            
        Returns:
            Dict containing statistical analysis
        """
        from scipy import stats
        import numpy as np
        
        print("Analyzing results...")
        analysis = {}
        
        for model in self.models:
            print(f"  Analyzing {model}...")
            model_analysis = {}
            
            # Extract metrics
            baseline_metrics = self.baselines[model]
            intervention_metrics = self._extract_metrics(results[model])
            
            # Test each hypothesis
            for metric in self.metrics:
                baseline = np.array(baseline_metrics[metric])
                intervention = np.array(intervention_metrics[metric])
                
                # Compute effect size
                effect_size = (intervention.mean() - baseline.mean()) / baseline.std()
                
                # Statistical test
                statistic, p_value = stats.ttest_ind(intervention, baseline)
                
                # Apply correction
                if corrections == 'holm-bonferroni':
                    # Simplified: would apply across all tests
                    adjusted_p = p_value * len(self.metrics)
                else:
                    adjusted_p = p_value
                
                model_analysis[metric] = {
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'adjusted_p': adjusted_p,
                    'significant': adjusted_p < alpha,
                    'direction': 'increase' if effect_size > 0 else 'decrease'
                }
            
            analysis[model] = model_analysis
        
        # Save analysis
        analysis_path = Path('results/analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def _get_baseline_activations(self, model: str) -> List:
        """Get baseline activations (placeholder)."""
        # This would interface with actual model
        import torch
        return [torch.randn(10, 512) for _ in range(5)]
    
    def _run_single_intervention(
        self,
        model: str,
        neg_scale: float,
        syn_scale: float,
        prompt_type: str,
        ablation_strength: str
    ) -> Dict:
        """Run single intervention (placeholder)."""
        # This would interface with actual intervention code
        return {
            'activations': self._get_baseline_activations(model),
            'output': f"Intervention output for {prompt_type}"
        }
    
    def _compute_intervention_metrics(self, result: Dict) -> Dict:
        """Compute metrics for intervention result."""
        activations = result['activations']
        return {
            'action': compute_semantic_action(activations),
            'eigengap': compute_eigengap(activations),
            'ape': compute_angle_preservation(activations, activations),
            'monodromy': compute_monodromy_drift(activations)
        }
    
    def _extract_metrics(self, results: List[Dict]) -> Dict:
        """Extract metrics from results list."""
        metrics = {m: [] for m in self.metrics}
        for result in results:
            for metric in self.metrics:
                metrics[metric].append(result['metrics'][metric])
        return metrics


def main():
    """Run complete SDSS experiment."""
    
    # Week 1-2: Setup and Baselines
    print("=" * 60)
    print("SDSS EXPERIMENT - WEEK 1-2: SETUP AND BASELINES")
    print("=" * 60)
    
    experiment = SDSSExperiment(
        models=['claude-3-opus', 'claude-3-sonnet'],
        metrics=['action', 'eigengap', 'ape', 'monodromy'],
        conditions=25,
        replications=5
    )
    
    # Collect baselines
    baselines = experiment.run_baselines()
    experiment.lock_preregistration(baselines)
    
    # Week 3-5: Interventions
    print("\n" + "=" * 60)
    print("WEEK 3-5: INTERVENTIONS")
    print("=" * 60)
    
    results = experiment.run_interventions(
        negation_scales=[0.1, 0.3, 0.5, 0.7, 1.0],
        synthesis_scales=[0.1, 0.3, 0.5, 0.7, 1.0],
        ablation_strength='moderate',
        prompt_types=['moral', 'self_reflection', 'godel']
    )
    
    # Week 6-7: Analysis
    print("\n" + "=" * 60)
    print("WEEK 6-7: ANALYSIS")
    print("=" * 60)
    
    analysis = experiment.analyze(
        results,
        corrections='holm-bonferroni',
        alpha=0.05
    )
    
    # Report key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    
    for model, model_analysis in analysis.items():
        print(f"\n{model}:")
        for metric, results in model_analysis.items():
            print(f"  {metric}:")
            print(f"    Effect size: {results['effect_size']:.3f}")
            print(f"    Significant: {results['significant']}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()