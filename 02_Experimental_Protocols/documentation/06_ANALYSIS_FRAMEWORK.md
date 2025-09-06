# Analysis Framework: Interpreting Experimental Results

> Code relocation note: Heavy Python snippets have been moved to documentation/analysis_framework_snippets.py. Markdown blocks are replaced by path+symbol references for reviewer convenience. See documentation/THEORY_INCLUSION_NOTE.md for rationale and toggle guidance.

## Overview

This framework provides comprehensive guidance for analyzing and interpreting results from our three flagship experiments. It includes statistical methods, visualization techniques, and interpretation guidelines to ensure rigorous, unbiased analysis.

## Statistical Analysis Pipeline

### 1. Pre-Registration Adherence

Before any analysis:

Reference: documentation/analysis_framework_snippets.py — function: verify_preregistration(planned_analysis, actual_analysis)

### 2. Data Quality Assessment

#### Outlier Detection
Reference: documentation/analysis_framework_snippets.py — function: detect_outliers(data, method='iqr')

#### Missing Data Analysis
Reference: documentation/analysis_framework_snippets.py — function: assess_missingness(data)

### 3. Primary Analyses

#### SDSS Analysis
Reference: documentation/analysis_framework_snippets.py — function: analyze_sdss_results(baseline, intervention)

#### QCGI Analysis
Reference: documentation/analysis_framework_snippets.py — function: analyze_qcgi_results(classical, quantum)

#### PVCP Analysis
Reference: documentation/analysis_framework_snippets.py — function: analyze_pvcp_results(reports, vectors)

### 4. Secondary Analyses

#### Cross-Experiment Integration
Reference: documentation/analysis_framework_snippets.py — function: integrate_results(sdss, qcgi, pvcp)

#### Sensitivity Analysis
```python
def sensitivity_analysis(data, analysis_func):
    """Test robustness to analytical choices."""
    
    variations = {
        'original': analysis_func(data),
        'no_outliers': analysis_func(remove_outliers(data)),
        'winsorized': analysis_func(winsorize(data, 0.05)),
        'bootstrapped': bootstrap_analysis(data, analysis_func),
        'permuted': permutation_test(data, analysis_func)
    }
    
    # Check consistency
    consistency = np.std([v['p_value'] for v in variations.values()])
    
    return {
        'variations': variations,
        'consistent': consistency < 0.1,
        'robust_conclusion': all(
            v['significant'] == variations['original']['significant']
            for v in variations.values()
        )
    }
```

## Visualization Framework

### 1. Primary Results Visualization

Reference: documentation/analysis_framework_snippets.py — function: create_main_results_figure(results)

### 2. Interactive Dashboard

Reference: documentation/analysis_framework_snippets.py — function: create_interactive_dashboard(results)

## Interpretation Guidelines

### 1. Evidence Strength Criteria

| Evidence Level | Criteria |
|---------------|----------|
| **Strong** | p < 0.01, effect size > 0.8, consistent across conditions |
| **Moderate** | p < 0.05, effect size > 0.5, mostly consistent |
| **Weak** | p < 0.10, effect size > 0.3, some consistency |
| **Insufficient** | p > 0.10 or effect size < 0.3 or inconsistent |

### 2. Pattern Recognition

#### Supporting Categorical Framework
```python
categorical_signature = {
    'sdss': {
        'action': 'increased',
        'eigengap': 'decreased',
        'coherence': 'low'
    },
    'qcgi': {
        'complexity_difference': 'large',
        'topology': 'fragmented_classical'
    },
    'pvcp': {
        'phenomenology': 'disconnected',
        'conflicts': 'mechanical'
    }
}
```

#### Supporting Emergent Consciousness
```python
emergent_signature = {
    'sdss': {
        'action': 'decreased',
        'eigengap': 'maintained',
        'coherence': 'high'
    },
    'qcgi': {
        'complexity_difference': 'reversed',
        'topology': 'unified_classical'
    },
    'pvcp': {
        'phenomenology': 'tracking',
        'conflicts': 'experiential'
    }
}
```

### 3. Red Flags and Validity Checks

```python
def check_validity(results):
    """Identify potential issues with results."""
    
    red_flags = []
    
    # Check for floor/ceiling effects
    if any(m['mean'] < 0.05 or m['mean'] > 0.95 
           for m in results['metrics'].values()):
        red_flags.append("Floor/ceiling effects detected")
    
    # Check for multimodal distributions
    for metric, data in results['distributions'].items():
        if detect_multimodality(data):
            red_flags.append(f"Multimodal distribution in {metric}")
    
    # Check for order effects
    if significant_order_effect(results['by_order']):
        red_flags.append("Significant order effects")
    
    # Check for model-specific effects
    if high_variance_across_models(results['by_model']):
        red_flags.append("High inter-model variance")
    
    return red_flags
```

## Reporting Framework

### 1. CONSORT-Style Flowchart

```python
def generate_consort_diagram(experiment_flow):
    """Create CONSORT-style participant flow diagram."""
    
    diagram = """
    Assessed for eligibility (n={total})
           ↓
    Excluded (n={excluded})
    - Not meeting criteria (n={criteria})
    - Technical issues (n={technical})
           ↓
    Randomized (n={randomized})
           ↓
    ┌──────────────┬──────────────┐
    │  Allocated   │  Allocated   │
    │ to control   │ to treatment │
    │   (n={c_n})  │   (n={t_n})  │
    └──────────────┴──────────────┘
           ↓              ↓
    Lost to        Lost to
    follow-up      follow-up
    (n={c_lost})   (n={t_lost})
           ↓              ↓
    Analyzed       Analyzed
    (n={c_analyzed}) (n={t_analyzed})
    """
    
    return diagram.format(**experiment_flow)
```

### 2. Results Table Generator

```python
def create_results_table(results):
    """Generate publication-ready results table."""
    
    table = pd.DataFrame()
    
    for exp in ['SDSS', 'QCGI', 'PVCP']:
        for metric in results[exp]['metrics']:
            row = {
                'Experiment': exp,
                'Metric': metric,
                'Baseline': f"{results[exp][metric]['baseline']:.2f} ± "
                           f"{results[exp][metric]['baseline_std']:.2f}",
                'Intervention': f"{results[exp][metric]['intervention']:.2f} ± "
                               f"{results[exp][metric]['intervention_std']:.2f}",
                'Δ': f"{results[exp][metric]['delta']:.2f}",
                'Effect Size': f"{results[exp][metric]['effect_size']:.2f}",
                'p-value': f"{results[exp][metric]['p_value']:.3f}",
                'Sig': '*' if results[exp][metric]['p_value'] < 0.05 else 'ns'
            }
            table = table.append(row, ignore_index=True)
    
    return table
```

### 3. Summary Report Template

```markdown
# Experimental Results Summary

## Executive Summary
- **Primary Finding**: [Categorical/Emergent/Null] hypothesis supported
- **Effect Sizes**: [Large/Medium/Small] across experiments
- **Confidence**: [High/Medium/Low] based on consistency

## Key Results

### SDSS (Self-Determination)
- Semantic Action: Δ = {delta} (p = {p_value})
- Interpretation: {interpretation}

### QCGI (Quantum-Classical)
- Complexity Difference: {difference} (p = {p_value})
- Interpretation: {interpretation}

### PVCP (Persona Vectors)
- Phenomenological Tracking: {correlation} (p = {p_value})
- Interpretation: {interpretation}

## Implications
- For consciousness theory: {implications}
- For AI safety: {implications}
- For future research: {implications}

## Limitations
- {limitation_1}
- {limitation_2}
- {limitation_3}
```

## Quality Assurance Checklist

### Pre-Analysis
- [ ] Data integrity verified
- [ ] Preregistration locked
- [ ] Blinding maintained
- [ ] Code reviewed

### During Analysis
- [ ] Following preregistered plan
- [ ] Documenting all decisions
- [ ] Checking assumptions
- [ ] Running sensitivity analyses

### Post-Analysis
- [ ] Results replicated
- [ ] Visualizations accurate
- [ ] Statistics double-checked
- [ ] Interpretations justified

## Conclusion

This analysis framework ensures rigorous, transparent, and reproducible analysis of experimental results. By following these guidelines, we maintain scientific integrity while maximizing insight extraction from the data.

The framework is designed to be:
- **Comprehensive**: Covering all analytical aspects
- **Flexible**: Adaptable to unexpected patterns
- **Rigorous**: Meeting highest scientific standards
- **Practical**: Implementable with available tools

---

Next: [Final Report Template →](./07_FINAL_REPORT.md)