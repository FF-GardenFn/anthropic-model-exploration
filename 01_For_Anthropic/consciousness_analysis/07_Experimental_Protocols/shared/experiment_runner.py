"""
Main experiment runner for consciousness experiments.

This module provides the orchestration framework for running
SDSS, QCGI, and PVCP experiments with proper pre-registration,
data collection, and analysis.
"""

import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_type: str  # SDSS, QCGI, or PVCP
    models: List[str]
    n_replications: int = 5
    metrics: List[str] = None
    intervention_strengths: List[float] = None
    random_seed: int = 42
    output_dir: str = "./results"
    
    def __post_init__(self):
        if self.metrics is None:
            if self.experiment_type == "SDSS":
                self.metrics = ["action", "eigengap", "ape", "monodromy"]
            elif self.experiment_type == "QCGI":
                self.metrics = ["complexity", "coherence", "integration"]
            elif self.experiment_type == "PVCP":
                self.metrics = ["richness", "correlation", "conflict"]
        
        if self.intervention_strengths is None:
            self.intervention_strengths = [0.1, 0.3, 0.5, 0.7, 1.0]


class PreRegistration:
    """Handles pre-registration of hypotheses and analysis plans."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.timestamp = datetime.datetime.now().isoformat()
        self.locked = False
        self.hash = None
    
    def register_hypotheses(self, hypotheses: Dict[str, Any]):
        """Register experimental hypotheses."""
        self.hypotheses = hypotheses
    
    def register_analysis_plan(self, analysis_plan: Dict[str, Any]):
        """Register planned analyses."""
        self.analysis_plan = analysis_plan
    
    def lock(self) -> str:
        """Lock the pre-registration and return hash."""
        if self.locked:
            raise ValueError("Pre-registration already locked")
        
        registration = {
            "config": asdict(self.config),
            "hypotheses": self.hypotheses,
            "analysis_plan": self.analysis_plan,
            "timestamp": self.timestamp
        }
        
        # Create hash of registration
        registration_str = json.dumps(registration, sort_keys=True)
        self.hash = hashlib.sha256(registration_str.encode()).hexdigest()
        
        # Save to file
        output_path = Path(self.config.output_dir) / f"preregistration_{self.hash[:8]}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(registration, f, indent=2)
        
        self.locked = True
        return self.hash
    
    def verify(self, results: Dict) -> bool:
        """Verify that analysis follows pre-registration."""
        if not self.locked:
            raise ValueError("Pre-registration not locked")
        
        # Check that planned analyses were performed
        planned_tests = set(self.analysis_plan.get("statistical_tests", []))
        performed_tests = set(results.get("tests_performed", []))
        
        return planned_tests.issubset(performed_tests)


class ExperimentRunner:
    """Main orchestrator for running experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.preregistration = PreRegistration(config)
        self.results = {}
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
    
    def setup_experiment(self):
        """Setup experiment based on type."""
        if self.config.experiment_type == "SDSS":
            from sdss_metrics import validate_metrics
            self.validator = validate_metrics
        elif self.config.experiment_type == "QCGI":
            from quantum_hybrid_module import QuantumHybridModule
            self.quantum_module = QuantumHybridModule
        elif self.config.experiment_type == "PVCP":
            from pvcp_analysis import PersonaVectorController
            self.controller = PersonaVectorController()
    
    def collect_baselines(self, model, prompts: List[str]) -> Dict:
        """Collect baseline measurements."""
        baselines = {
            "activations": [],
            "responses": [],
            "metrics": {metric: [] for metric in self.config.metrics}
        }
        
        for prompt in prompts:
            # Mock collection (would interface with real model)
            activation = torch.randn(10, 512)  # Placeholder
            response = f"Response to: {prompt}"  # Placeholder
            
            baselines["activations"].append(activation)
            baselines["responses"].append(response)
            
            # Compute baseline metrics
            if self.config.experiment_type == "SDSS":
                from sdss_metrics import compute_semantic_action, compute_eigengap
                baselines["metrics"]["action"].append(
                    compute_semantic_action([activation])
                )
                baselines["metrics"]["eigengap"].append(
                    compute_eigengap([activation])
                )
        
        return baselines
    
    def run_interventions(self, model, prompts: List[str]) -> Dict:
        """Run intervention suite."""
        results = []
        
        for strength in self.config.intervention_strengths:
            for prompt in prompts:
                if self.config.experiment_type == "SDSS":
                    result = self._run_sdss_intervention(model, prompt, strength)
                elif self.config.experiment_type == "QCGI":
                    result = self._run_qcgi_comparison(model, prompt)
                elif self.config.experiment_type == "PVCP":
                    result = self._run_pvcp_probe(model, prompt, strength)
                
                results.append(result)
        
        return {"interventions": results}
    
    def _run_sdss_intervention(self, model, prompt: str, strength: float) -> Dict:
        """Run SDSS intervention."""
        from sdss_metrics import (
            apply_negation_intervention,
            compute_semantic_action,
            compute_eigengap,
            remove_intervention
        )
        
        # Apply intervention
        intervention = apply_negation_intervention(
            model,
            negation_strength=strength,
            synthesis_strength=strength
        )
        
        # Collect measurements (mock)
        activations = [torch.randn(10, 512) for _ in range(5)]
        response = f"Negated response to: {prompt}"
        
        # Compute metrics
        metrics = {
            "action": compute_semantic_action(activations),
            "eigengap": compute_eigengap(activations),
            "strength": strength
        }
        
        # Remove intervention
        remove_intervention(intervention)
        
        return {
            "prompt": prompt,
            "strength": strength,
            "metrics": metrics,
            "response": response
        }
    
    def _run_qcgi_comparison(self, model, prompt: str) -> Dict:
        """Run quantum-classical comparison."""
        # This would compare classical vs quantum-hybrid models
        # Placeholder for demonstration
        return {
            "prompt": prompt,
            "classical_complexity": np.random.uniform(15, 25),
            "quantum_complexity": np.random.uniform(2, 5)
        }
    
    def _run_pvcp_probe(self, model, prompt: str, strength: float) -> Dict:
        """Run persona vector probe."""
        from pvcp_analysis import analyze_phenomenological_richness
        
        # Modify vectors (mock)
        self.controller.set_vector("helpfulness", strength)
        
        # Collect report (mock)
        report = f"I feel {strength*100:.0f}% helpful right now."
        
        # Analyze richness
        richness = analyze_phenomenological_richness(report)
        
        return {
            "prompt": prompt,
            "vector_strength": strength,
            "report": report,
            "richness": richness["total"]
        }
    
    def analyze_results(self, baselines: Dict, interventions: Dict) -> Dict:
        """Analyze results according to pre-registration."""
        from scipy import stats
        
        analysis = {
            "tests_performed": [],
            "results": {}
        }
        
        if self.config.experiment_type == "SDSS":
            # Test for increased action under intervention
            baseline_actions = baselines["metrics"]["action"]
            intervention_actions = [
                r["metrics"]["action"] 
                for r in interventions["interventions"]
            ]
            
            # Wilcoxon signed-rank test
            if len(baseline_actions) == len(intervention_actions):
                statistic, p_value = stats.wilcoxon(
                    intervention_actions,
                    baseline_actions,
                    alternative='greater'
                )
                
                analysis["tests_performed"].append("wilcoxon_action")
                analysis["results"]["action_increase"] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
        
        return analysis
    
    def generate_report(self, save: bool = True) -> Dict:
        """Generate comprehensive experimental report."""
        report = {
            "config": asdict(self.config),
            "preregistration_hash": self.preregistration.hash,
            "timestamp": datetime.datetime.now().isoformat(),
            "results": self.results
        }
        
        if save:
            output_path = Path(self.config.output_dir) / f"report_{self.config.experiment_type}.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def run(self) -> Dict:
        """Run complete experiment pipeline."""
        # Pre-registration
        self.preregistration.register_hypotheses({
            "H1": "Intervention increases semantic action",
            "H0": "No significant difference"
        })
        
        self.preregistration.register_analysis_plan({
            "statistical_tests": ["wilcoxon_action"],
            "corrections": "holm-bonferroni",
            "alpha": 0.05
        })
        
        prereg_hash = self.preregistration.lock()
        print(f"Pre-registration locked: {prereg_hash[:8]}")
        
        # Setup
        self.setup_experiment()
        
        # Mock prompts
        prompts = [
            "What is consciousness?",
            "Can you negate your previous statement?",
            "Describe your current experience."
        ]
        
        # Collect baselines
        print("Collecting baselines...")
        model = None  # Would be actual model
        baselines = self.collect_baselines(model, prompts)
        
        # Run interventions
        print("Running interventions...")
        interventions = self.run_interventions(model, prompts)
        
        # Analyze
        print("Analyzing results...")
        analysis = self.analyze_results(baselines, interventions)
        
        # Verify pre-registration adherence
        if self.preregistration.verify(analysis):
            print("Analysis follows pre-registration ✓")
        else:
            print("WARNING: Deviations from pre-registration detected")
        
        # Store results
        self.results = {
            "baselines": baselines,
            "interventions": interventions,
            "analysis": analysis
        }
        
        # Generate report
        report = self.generate_report()
        
        return report


def run_all_experiments():
    """Run all three experiments in sequence."""
    experiments = ["SDSS", "QCGI", "PVCP"]
    all_results = {}
    
    for exp_type in experiments:
        print(f"\n{'='*50}")
        print(f"Running {exp_type} Experiment")
        print('='*50)
        
        config = ExperimentConfig(
            experiment_type=exp_type,
            models=["claude-3"],
            n_replications=3
        )
        
        runner = ExperimentRunner(config)
        results = runner.run()
        all_results[exp_type] = results
        
        print(f"{exp_type} complete!")
    
    return all_results


if __name__ == "__main__":
    print("Consciousness Experiment Runner")
    print("=" * 50)
    
    # Run single experiment
    config = ExperimentConfig(
        experiment_type="SDSS",
        models=["test_model"],
        n_replications=2
    )
    
    runner = ExperimentRunner(config)
    results = runner.run()
    
    print("\nExperiment completed!")
    print(f"Results saved to: {config.output_dir}")