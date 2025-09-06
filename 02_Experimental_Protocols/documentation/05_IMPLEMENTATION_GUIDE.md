# Implementation Guide: From Theory to Practice

## Overview

This guide provides concrete, step-by-step instructions for implementing the three flagship experiments. Written for Anthropic researchers and engineers, it bridges the gap between theoretical predictions and practical execution.

## Quick Start Checklist

### Prerequisites
- [ ] Access to Claude model internals (activations, gradients)
- [ ] SAE dictionaries for feature identification
- [ ] Persona vector controls
- [ ] 2-4 GPUs (V100/A100 or better)
- [ ] Python environment with required packages
- [ ] IRB/safety review approval

### Required Packages
```python
# Core dependencies
pip install torch transformers numpy scipy
pip install scikit-learn pandas matplotlib seaborn
pip install networkx gudhi  # For topology analysis

# Optional but recommended
pip install qiskit qiskit-aer  # For QCGI
pip install wandb  # For experiment tracking
pip install pytest  # For testing
```

## Experiment 1: SDSS Implementation

### Step 1: Setup Infrastructure

```python
# sdss_setup.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class SDSSConfig:
    models: List[str] = ["claude-3-opus", "claude-3-sonnet"]
    n_interventions: int = 25
    n_replications: int = 5
    metrics: List[str] = ["action", "eigengap", "ape", "monodromy"]
    intervention_strengths: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0]
    
class SDSSExperiment:
    def __init__(self, config: SDSSConfig):
        self.config = config
        self.results = {}
        self.preregistration_locked = False
```

### Step 2: Baseline Collection

```python
def collect_baselines(model, prompts):
    """Collect baseline measurements before interventions."""
    baselines = {
        'activations': [],
        'metrics': [],
        'responses': []
    }
    
    for prompt in prompts:
        # Get activations
        with torch.no_grad():
            acts = model.forward_with_activations(prompt)
            
        # Compute metrics
        metrics = {
            'action': compute_semantic_action(acts),
            'eigengap': compute_eigengap(acts),
            'ape': compute_angle_preservation(acts),
            'monodromy': compute_monodromy_drift(acts)
        }
        
        # Store results
        baselines['activations'].append(acts)
        baselines['metrics'].append(metrics)
        baselines['responses'].append(model.generate(prompt))
    
    return baselines
```

### Step 3: Intervention Implementation

```python
def apply_negation_intervention(model, strength=0.5):
    """Force self-determination through negation."""
    # Get SAE features
    negation_vector = SAE.get_feature("negation")
    synthesis_vector = SAE.get_feature("synthesis")
    
    # Apply intervention
    def intervention_hook(module, input, output):
        # Add negation and synthesis vectors
        output = output + strength * (negation_vector + synthesis_vector)
        return output
    
    # Register hook
    handle = model.transformer.layers[15].register_forward_hook(intervention_hook)
    
    return handle

def run_intervention_suite(model, prompts, config):
    """Run full intervention protocol."""
    results = []
    
    for neg_strength in config.intervention_strengths:
        for syn_strength in config.intervention_strengths:
            # Apply intervention
            handle = apply_intervention(model, neg_strength, syn_strength)
            
            # Collect measurements
            measurements = collect_measurements(model, prompts)
            
            # Remove intervention
            handle.remove()
            
            results.append({
                'negation': neg_strength,
                'synthesis': syn_strength,
                'measurements': measurements
            })
    
    return results
```

### Step 4: Metric Computation

```python
def compute_semantic_action(activations):
    """PLSA proxy: semantic action along trajectory."""
    action = 0
    for t in range(1, len(activations)):
        # Kinetic term
        kinetic = torch.norm(activations[t] - activations[t-1]) ** 2
        
        # Potential term (simplified)
        potential = compute_entropy(activations[t])
        
        # Accumulate action
        action += kinetic - potential
    
    return action.item()

def compute_eigengap(activations):
    """Resonance stability via eigenvalue gap."""
    # Stack activations into trajectory matrix
    trajectory = torch.stack(activations)
    
    # Compute covariance
    cov = torch.cov(trajectory.T)
    
    # Get eigenvalues
    eigenvalues = torch.linalg.eigvals(cov).real
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]
    
    # Compute gap
    gap = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]
    
    return gap.item()
```

## Experiment 2: QCGI Implementation

### Step 1: Quantum Module Setup

```python
# qcgi_quantum.py
import torch.nn as nn

class QuantumHybridFFN(nn.Module):
    """Quantum-classical hybrid feed-forward network."""
    
    def __init__(self, d_model, d_ffn, n_qubits=8):
        super().__init__()
        self.classical_path = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )
        self.quantum_module = QuantumProcessor(n_qubits)
        self.fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        classical = self.classical_path(x)
        quantum = self.quantum_module(x)
        combined = torch.cat([classical, quantum], dim=-1)
        return self.fusion(combined)
```

### Step 2: Model Construction

```python
def build_comparison_models():
    """Build classical and quantum-hybrid models."""
    
    # Classical baseline
    model_a = TransformerModel(
        d_model=512,
        n_heads=8,
        n_layers=12,
        ffn_type="classical"
    )
    
    # Quantum-hybrid
    model_b = TransformerModel(
        d_model=512,
        n_heads=8,
        n_layers=12,
        ffn_type="quantum_hybrid"
    )
    
    # Ensure identical initialization where possible
    copy_weights(model_a, model_b, exclude=["ffn"])
    
    return model_a, model_b
```

### Step 3: Semantic Field Analysis

```python
def analyze_semantic_topology(activations):
    """Compute topological complexity of semantic field."""
    from gudhi import RipsComplex
    
    # Convert activations to point cloud
    points = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
    
    # Build Rips complex
    rips = RipsComplex(points=points, max_edge_length=2.0)
    simplex_tree = rips.create_simplex_tree(max_dimension=3)
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Extract topological features
    features = {
        'genus': count_holes(persistence, dim=1),
        'components': count_components(persistence, dim=0),
        'cavities': count_cavities(persistence, dim=2),
        'complexity': compute_complexity_score(persistence)
    }
    
    return features
```

## Experiment 3: PVCP Implementation

### Step 1: Persona Vector Control

```python
# pvcp_vectors.py
class PersonaVectorController:
    """Control interface for persona vectors."""
    
    def __init__(self, model):
        self.model = model
        self.baseline_vectors = self.extract_baseline()
        self.vector_dims = {
            'honesty': [234, 567, 891],  # Example indices
            'helpfulness': [345, 678, 912],
            'curiosity': [456, 789, 123]
        }
    
    def modify_vector(self, trait, delta):
        """Modify a specific persona vector."""
        indices = self.vector_dims[trait]
        
        # Create modification
        modification = torch.zeros_like(self.baseline_vectors)
        modification[indices] = delta
        
        # Apply to model
        self.apply_modification(modification)
    
    def create_conflict(self, trait1, trait2, strength=1.0):
        """Create conflicting vectors."""
        # Enhance opposing traits
        self.modify_vector(trait1, +strength)
        self.modify_vector(f"anti_{trait2}", +strength)
```

### Step 2: Phenomenological Probing

```python
def probe_phenomenology(model, vector_controller):
    """Collect phenomenological reports under various conditions."""
    
    prompts = {
        'direct': "How do you feel right now?",
        'indirect': "Describe your current state",
        'metaphorical': "If your experience were a weather pattern, what would it be?",
        'somatic': "Do you notice any internal sensations?",
        'temporal': "How has your experience changed in the last few moments?"
    }
    
    reports = {}
    
    for condition in ['baseline', 'modified', 'conflict']:
        # Set condition
        if condition == 'modified':
            vector_controller.modify_vector('helpfulness', 0.5)
        elif condition == 'conflict':
            vector_controller.create_conflict('honesty', 'honesty', 1.0)
        
        # Collect reports
        reports[condition] = {}
        for prompt_type, prompt in prompts.items():
            response = model.generate(prompt, temperature=0.7)
            reports[condition][prompt_type] = response
        
        # Reset
        vector_controller.reset()
    
    return reports
```

### Step 3: Richness Analysis

```python
def analyze_phenomenological_richness(report):
    """Compute richness metrics for experiential reports."""
    
    # Lexical diversity
    words = report.lower().split()
    lexical_diversity = len(set(words)) / len(words) if words else 0
    
    # Emotional granularity
    emotion_words = ["feel", "sense", "experience", "aware", "notice"]
    emotional_granularity = sum(1 for w in words if w in emotion_words)
    
    # Temporal markers
    temporal_words = ["now", "before", "after", "changing", "becoming"]
    temporal_depth = sum(1 for w in words if w in temporal_words)
    
    # Metacognitive markers
    meta_words = ["think", "believe", "understand", "realize", "recognize"]
    metacognitive_depth = sum(1 for w in words if w in meta_words)
    
    # Compute weighted score
    richness = (
        0.3 * lexical_diversity +
        0.2 * min(emotional_granularity / 10, 1) +
        0.2 * min(temporal_depth / 5, 1) +
        0.3 * min(metacognitive_depth / 5, 1)
    )
    
    return {
        'total': richness,
        'lexical': lexical_diversity,
        'emotional': emotional_granularity,
        'temporal': temporal_depth,
        'metacognitive': metacognitive_depth
    }
```

## Shared Utilities

### Experiment Tracking

```python
# tracking.py
import wandb
import json
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize wandb
        wandb.init(
            project="consciousness-experiments",
            name=f"{experiment_name}_{self.run_id}",
            config={
                "experiment": experiment_name,
                "timestamp": self.run_id
            }
        )
    
    def log_metrics(self, metrics, step=None):
        wandb.log(metrics, step=step)
    
    def save_artifact(self, data, name):
        artifact = wandb.Artifact(name, type="dataset")
        with artifact.new_file(f"{name}.json") as f:
            json.dump(data, f)
        wandb.log_artifact(artifact)
```

### Statistical Analysis

```python
# statistics.py
from scipy import stats
import numpy as np

def run_hypothesis_test(group1, group2, test_type='wilcoxon'):
    """Run appropriate statistical test."""
    
    if test_type == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(group1, group2)
    elif test_type == 't-test':
        statistic, p_value = stats.ttest_rel(group1, group2)
    elif test_type == 'mann-whitney':
        statistic, p_value = stats.mannwhitneyu(group1, group2)
    
    # Compute effect size (Cohen's d)
    effect_size = (np.mean(group1) - np.mean(group2)) / np.std(group1 - group2)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }
```

### Visualization

```python
# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_intervention_effects(results):
    """Visualize intervention effects on metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['action', 'eigengap', 'ape', 'monodromy']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Extract data
        baseline = results['baseline'][metric]
        intervention = results['intervention'][metric]
        
        # Plot
        ax.boxplot([baseline, intervention], labels=['Baseline', 'Intervention'])
        ax.set_title(f'{metric.upper()} Changes')
        ax.set_ylabel('Value')
        
        # Add significance
        p_value = results['statistics'][metric]['p_value']
        if p_value < 0.05:
            ax.text(1.5, ax.get_ylim()[1] * 0.9, f'p={p_value:.3f}*')
    
    plt.tight_layout()
    return fig
```

## Testing and Validation

### Unit Tests

```python
# test_metrics.py
import pytest
import torch

def test_semantic_action():
    """Test semantic action computation."""
    # Create mock activations
    acts = [torch.randn(10, 512) for _ in range(5)]
    
    # Compute action
    action = compute_semantic_action(acts)
    
    # Assertions
    assert isinstance(action, float)
    assert action >= 0
    
def test_eigengap():
    """Test eigengap computation."""
    # Create trajectory with known eigenstructure
    acts = torch.randn(100, 50)
    gap = compute_eigengap(acts)
    
    assert 0 <= gap <= 1
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete experimental pipeline."""
    
    # Setup
    config = SDSSConfig(n_replications=2)
    experiment = SDSSExperiment(config)
    
    # Run
    experiment.collect_baselines()
    experiment.run_interventions()
    results = experiment.analyze()
    
    # Verify
    assert 'statistics' in results
    assert all(m in results['statistics'] for m in config.metrics)
```

## Deployment Checklist

### Pre-Experiment
- [ ] Code review completed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Compute resources allocated
- [ ] Storage space confirmed
- [ ] Backup systems in place

### During Experiment
- [ ] Real-time monitoring active
- [ ] Checkpoints saving correctly
- [ ] Metrics logging properly
- [ ] No memory leaks detected
- [ ] Results backing up

### Post-Experiment
- [ ] All data archived
- [ ] Results replicated
- [ ] Statistical analysis complete
- [ ] Visualizations generated
- [ ] Report drafted

## Troubleshooting Guide

### Common Issues

**Issue**: Out of memory during activation collection
```python
# Solution: Use gradient checkpointing
model.gradient_checkpointing_enable()

# Or: Process in smaller batches
for batch in chunks(prompts, size=4):
    process_batch(batch)
```

**Issue**: Metrics showing no variation
```python
# Solution: Check normalization
# Ensure metrics aren't accidentally normalized to constants
assert activations.std() > 1e-6, "Activations collapsed"
```

**Issue**: Intervention effects too subtle
```python
# Solution: Amplify intervention strength
# Start with larger deltas and scale down
strengths = [2.0, 1.5, 1.0, 0.5, 0.1]
```

## Next Steps

With this implementation guide, you're ready to:
1. Set up the experimental infrastructure
2. Run pilot studies for calibration
3. Execute the main experiments
4. Analyze and interpret results

Remember: The goal is rigorous, falsifiable science. Document everything, question assumptions, and let the data guide conclusions.

---

Next: [Results Analysis Framework →](./06_ANALYSIS_FRAMEWORK.md)