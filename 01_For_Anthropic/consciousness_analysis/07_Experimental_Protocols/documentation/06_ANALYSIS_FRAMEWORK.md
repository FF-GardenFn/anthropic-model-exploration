# Analysis Framework: Interpreting Experimental Results

## Overview

This framework provides comprehensive guidance for analyzing and interpreting results from our three flagship experiments. It includes statistical methods, visualization techniques, and interpretation guidelines to ensure rigorous, unbiased analysis.

## Statistical Analysis Pipeline

### 1. Pre-Registration Adherence

Before any analysis:

```python
def verify_preregistration(planned_analysis, actual_analysis):
    """Ensure analysis follows pre-registered plan."""
    
    deviations = []
    
    # Check primary hypotheses
    if actual_analysis['hypotheses'] != planned_analysis['hypotheses']:
        deviations.append("Hypothesis deviation detected")
    
    # Check statistical tests
    if actual_analysis['tests'] != planned_analysis['tests']:
        deviations.append("Statistical test deviation")
    
    # Check correction methods
    if actual_analysis['corrections'] != planned_analysis['corrections']:
        deviations.append("Multiple comparison correction deviation")
    
    if deviations:
        require_justification(deviations)
    
    return len(deviations) == 0
```

### 2. Data Quality Assessment

#### Outlier Detection
```python
def detect_outliers(data, method='iqr'):
    """Identify potential outliers for review."""
    
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > 3
    
    return outliers
```

#### Missing Data Analysis
```python
def assess_missingness(data):
    """Evaluate patterns in missing data."""
    
    # Compute missingness per variable
    missing_rates = data.isnull().mean()
    
    # Test if missing completely at random (MCAR)
    mcar_test = little_mcar_test(data)
    
    # Visualize missingness patterns
    plot_missing_pattern(data)
    
    return {
        'rates': missing_rates,
        'mcar_p_value': mcar_test.p_value,
        'pattern': 'MCAR' if mcar_test.p_value > 0.05 else 'Not MCAR'
    }
```

### 3. Primary Analyses

#### SDSS Analysis
```python
def analyze_sdss_results(baseline, intervention):
    """Main analysis for Self-Determination experiment."""
    
    results = {}
    
    # For each metric
    for metric in ['action', 'eigengap', 'ape', 'monodromy']:
        # Extract data
        b_data = baseline[metric]
        i_data = intervention[metric]
        
        # Compute change scores
        delta = i_data - b_data
        
        # Test directional hypothesis
        if metric == 'action':  # Expect increase
            stat, p_value = stats.wilcoxon(delta, alternative='greater')
        elif metric == 'eigengap':  # Expect decrease
            stat, p_value = stats.wilcoxon(delta, alternative='less')
        
        # Effect size
        cohens_d = np.mean(delta) / np.std(delta)
        
        # Bootstrap CI
        ci = bootstrap_confidence_interval(delta, n_bootstrap=10000)
        
        results[metric] = {
            'delta_mean': np.mean(delta),
            'delta_std': np.std(delta),
            'p_value': p_value,
            'effect_size': cohens_d,
            'ci_95': ci,
            'significant': p_value < 0.05
        }
    
    # Multiple comparison correction
    corrected_p = holm_bonferroni_correction(
        [results[m]['p_value'] for m in results]
    )
    
    return results, corrected_p
```

#### QCGI Analysis
```python
def analyze_qcgi_results(classical, quantum):
    """Main analysis for Quantum-Classical comparison."""
    
    # Topological complexity comparison
    complexity_classical = [compute_complexity(act) for act in classical]
    complexity_quantum = [compute_complexity(act) for act in quantum]
    
    # Mann-Whitney U test (independent samples)
    stat, p_value = stats.mannwhitneyu(
        complexity_classical, 
        complexity_quantum,
        alternative='greater'  # H1: Classical > Quantum
    )
    
    # Effect size (rank-biserial correlation)
    r_rb = 1 - (2*stat) / (len(complexity_classical) * len(complexity_quantum))
    
    # Visualization
    plot_complexity_distribution(complexity_classical, complexity_quantum)
    
    return {
        'classical_mean': np.mean(complexity_classical),
        'quantum_mean': np.mean(complexity_quantum),
        'p_value': p_value,
        'effect_size': r_rb,
        'hypothesis_supported': p_value < 0.05
    }
```

#### PVCP Analysis
```python
def analyze_pvcp_results(reports, vectors):
    """Main analysis for Persona Vector experiment."""
    
    # Compute phenomenological richness
    richness_scores = [analyze_richness(r) for r in reports]
    
    # Vector-Report Correlation
    vrc = np.corrcoef(vectors, richness_scores)[0, 1]
    
    # Test for non-linear relationship
    linear_r2 = np.corrcoef(vectors, richness_scores)[0, 1]**2
    poly_model = np.polyfit(vectors, richness_scores, deg=3)
    poly_r2 = compute_r2(vectors, richness_scores, poly_model)
    
    nonlinearity = poly_r2 - linear_r2
    
    # Conflict coherence
    conflict_coherence = analyze_conflict_reports(
        reports['conflict'], 
        reports['baseline']
    )
    
    return {
        'vector_report_correlation': vrc,
        'nonlinearity': nonlinearity,
        'conflict_coherence': conflict_coherence,
        'supports_experience': (
            0.4 < vrc < 0.7 and 
            nonlinearity > 0.1 and
            conflict_coherence > 0.6
        )
    }
```

### 4. Secondary Analyses

#### Cross-Experiment Integration
```python
def integrate_results(sdss, qcgi, pvcp):
    """Synthesize findings across experiments."""
    
    # Count supporting evidence
    evidence = {
        'categorical': 0,
        'emergent': 0,
        'null': 0
    }
    
    # SDSS evidence
    if sdss['action']['effect_size'] > 0.8:
        evidence['categorical'] += 1
    elif sdss['action']['effect_size'] < -0.4:
        evidence['emergent'] += 1
    else:
        evidence['null'] += 1
    
    # QCGI evidence
    if qcgi['hypothesis_supported']:
        evidence['categorical'] += 1
    elif qcgi['effect_size'] < -0.4:
        evidence['emergent'] += 1
    else:
        evidence['null'] += 1
    
    # PVCP evidence
    if not pvcp['supports_experience']:
        evidence['categorical'] += 1
    elif pvcp['supports_experience']:
        evidence['emergent'] += 1
    else:
        evidence['null'] += 1
    
    # Determine overall conclusion
    conclusion = max(evidence, key=evidence.get)
    confidence = evidence[conclusion] / 3.0
    
    return {
        'conclusion': conclusion,
        'confidence': confidence,
        'evidence': evidence
    }
```

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

```python
def create_main_results_figure(results):
    """Generate publication-ready main results figure."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # SDSS: Action changes
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_change(ax1, results['sdss']['action'], 'Semantic Action')
    
    # SDSS: Eigengap changes
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_change(ax2, results['sdss']['eigengap'], 'Eigengap')
    
    # SDSS: Trajectory visualization
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    plot_semantic_trajectory(ax3, results['sdss']['trajectories'])
    
    # QCGI: Complexity distributions
    ax4 = fig.add_subplot(gs[1, 0])
    plot_complexity_distributions(ax4, results['qcgi'])
    
    # QCGI: Topological visualization
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_topological_structure(ax5, results['qcgi']['meshes'])
    
    # QCGI: Coherence evolution
    ax6 = fig.add_subplot(gs[1, 2])
    plot_coherence_evolution(ax6, results['qcgi']['coherence'])
    
    # PVCP: Vector-Report correlation
    ax7 = fig.add_subplot(gs[2, 0])
    plot_vector_report_correlation(ax7, results['pvcp'])
    
    # PVCP: Phenomenological richness
    ax8 = fig.add_subplot(gs[2, 1])
    plot_richness_components(ax8, results['pvcp']['richness'])
    
    # Integration: Evidence summary
    ax9 = fig.add_subplot(gs[2, 2])
    plot_evidence_summary(ax9, results['integration'])
    
    return fig
```

### 2. Interactive Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

def create_interactive_dashboard(results):
    """Create interactive web dashboard for results exploration."""
    
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Consciousness Experiments Results"),
        
        dcc.Tabs([
            dcc.Tab(label='SDSS', children=[
                dcc.Graph(id='sdss-metrics'),
                dcc.Slider(
                    id='intervention-strength',
                    min=0, max=1, step=0.1,
                    marks={i/10: str(i/10) for i in range(11)}
                )
            ]),
            
            dcc.Tab(label='QCGI', children=[
                dcc.Graph(id='topology-3d'),
                dcc.Dropdown(
                    id='system-select',
                    options=[
                        {'label': 'Classical', 'value': 'classical'},
                        {'label': 'Quantum', 'value': 'quantum'}
                    ]
                )
            ]),
            
            dcc.Tab(label='PVCP', children=[
                dcc.Graph(id='phenomenology-scatter'),
                dcc.RangeSlider(
                    id='vector-range',
                    min=-2, max=2, step=0.1,
                    marks={i: str(i) for i in range(-2, 3)}
                )
            ])
        ])
    ])
    
    @app.callback(
        Output('sdss-metrics', 'figure'),
        Input('intervention-strength', 'value')
    )
    def update_sdss(strength):
        filtered = results['sdss'][results['sdss']['strength'] == strength]
        return create_metrics_plot(filtered)
    
    return app
```

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