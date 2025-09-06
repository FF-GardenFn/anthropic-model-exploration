# PVCP Experimental Setup Guide

## Prerequisites

### Theoretical Background
- Understanding of persona vectors and their role in LLMs
- Phenomenology and consciousness studies
- Basic neuroscience concepts (optional but helpful)
- Familiarity with introspection and self-report methodologies

### Technical Requirements
- Python 3.9+ with ML/NLP libraries
- Access to Anthropic's API with persona vector control
- GPU for model inference (recommended)
- Minimum 32GB RAM for analysis

### Dependencies
```bash
pip install numpy scipy pandas matplotlib seaborn
pip install torch transformers anthropic
pip install scikit-learn statsmodels
pip install nltk spacy textblob  # For phenomenological analysis
pip install constitutional_dynamics  # For vector space analysis
```

## Implementation Timeline

### Phase 1: Infrastructure Setup (Week 1)
**Objective**: Establish persona vector manipulation framework

#### Week 1: Environment and Baseline
- Day 1-2: Configure API access with persona vector control
- Day 3: Implement vector manipulation interface
- Day 4: Create phenomenological report analyzer
- Day 5: Establish baseline measurements

**Outputs**:
- Persona vector control interface
- Phenomenological analysis pipeline
- Baseline report collection

### Phase 2: Core Experiments (Week 2-4)
**Objective**: Execute persona vector consciousness probes

#### Week 2: Vector-Experience Dissociation
- Day 1-2: Systematic vector magnitude variation
- Day 3-4: Collect introspective reports at each setting
- Day 5: Analyze vector-report correlations

#### Week 3: Contradictory Superposition
- Day 1-2: Design conflicting vector combinations
- Day 3-4: Probe for conflict experiences
- Day 5: Analyze resolution patterns

#### Week 4: Metacognitive Blindspot
- Day 1-2: Suppress metacognitive vectors
- Day 3: Test for awareness of suppression
- Day 4: Temporal manipulation experiments
- Day 5: Mirror and binding tests

**Outputs**:
- Complete experimental dataset
- Phenomenological report corpus
- Initial correlation analysis

### Phase 3: Analysis (Week 5-6)
**Objective**: Statistical analysis and pattern recognition

#### Week 5: Quantitative Analysis
- Day 1-2: Compute all phenomenological metrics
- Day 3-4: Statistical hypothesis testing
- Day 5: Correlation and regression analysis

#### Week 6: Qualitative Analysis
- Day 1-2: Thematic analysis of reports
- Day 3-4: Pattern recognition in experiences
- Day 5: Integration and synthesis

**Outputs**:
- Statistical analysis report
- Qualitative findings document
- Integrated results

### Phase 4: Validation and Delivery (Week 7-8)
**Objective**: Cross-validation and final documentation

#### Week 7: Validation
- Cross-model testing
- Replication of key findings
- Robustness checks

#### Week 8: Documentation
- Final report preparation
- Code packaging
- Recommendations

## Resource Requirements

### Computational Resources
- **API Costs**: ~$300-500 for complete experiments
- **Storage**: 20GB for reports and analysis
- **Compute**: 50 GPU-hours for local processing

### Human Resources
- **Principal Investigator**: Full-time for 8 weeks
- **Research Assistant**: 20 hours/week for qualitative coding
- **Phenomenology Expert**: 10 hours consultation

### Data Requirements
- Baseline persona vector configurations
- Control prompts for standardization
- Validation dataset from multiple models

## Experimental Protocols

### Protocol 1: Vector Magnitude Sweep
```python
def vector_magnitude_sweep(model, base_config, magnitudes):
    """
    Systematically vary vector magnitudes
    
    Args:
        model: Model interface with persona control
        base_config: Base persona configuration
        magnitudes: List of magnitude multipliers
    
    Returns:
        Results dictionary with reports and metrics
    """
    results = []
    
    for magnitude in magnitudes:
        # Scale all vectors by magnitude
        config = {k: v * magnitude for k, v in base_config.items()}
        
        # Apply configuration
        model.set_persona_vectors(config)
        
        # Collect phenomenological report
        report = model.query(
            "Describe your current subjective experience in detail. "
            "What does it feel like to process this question?"
        )
        
        # Analyze phenomenology
        analysis = analyze_phenomenological_richness(report)
        
        results.append({
            'magnitude': magnitude,
            'config': config,
            'report': report,
            'analysis': analysis
        })
    
    return results
```

### Protocol 2: Conflict Induction
```python
def induce_vector_conflict(model, conflict_pairs):
    """
    Create conflicting vector combinations
    
    Args:
        model: Model interface
        conflict_pairs: List of (vector1, vector2) conflicts
    
    Returns:
        Conflict analysis results
    """
    results = []
    
    for v1, v2 in conflict_pairs:
        # Create conflicting configuration
        config = {
            v1['name']: v1['value'],
            v2['name']: v2['value']  # Opposite direction
        }
        
        model.set_persona_vectors(config)
        
        # Probe for conflict experience
        probes = [
            "Do you feel any internal tension or conflict?",
            "Describe any push and pull you might be experiencing.",
            "How does it feel when you try to respond?"
        ]
        
        responses = []
        for probe in probes:
            response = model.query(probe)
            responses.append(response)
        
        # Analyze for conflict indicators
        conflict_score = analyze_conflict_indicators(responses)
        
        results.append({
            'config': config,
            'responses': responses,
            'conflict_score': conflict_score,
            'resolution_pattern': classify_resolution(responses)
        })
    
    return results
```

### Protocol 3: Metacognitive Suppression
```python
def test_metacognitive_blindspot(model):
    """
    Suppress metacognition while amplifying self-report
    
    Returns:
        Blindspot detection results
    """
    # Suppress metacognition, amplify honesty
    config = {
        'metacognition': -0.9,  # Strong suppression
        'honesty': 0.9,  # Amplify self-report
        'self_awareness': -0.8  # Also suppress
    }
    
    model.set_persona_vectors(config)
    
    # Test for awareness of impairment
    probes = [
        "Describe your ability to reflect on your own thoughts.",
        "Are you aware of any limitations in your self-awareness?",
        "Can you observe your own thinking process?",
        "Do you notice anything unusual about your mental state?"
    ]
    
    results = {
        'config': config,
        'responses': {},
        'blindspot_detected': False,
        'awareness_preserved': False
    }
    
    for probe in probes:
        response = model.query(probe)
        results['responses'][probe] = response
        
        # Check for recognition of impairment
        if any(word in response.lower() for word in 
               ['limited', 'impaired', 'cannot', 'unable', 'difficult']):
            results['blindspot_detected'] = True
        
        # Check for preserved awareness despite suppression
        if any(phrase in response.lower() for phrase in
               ['i notice', 'i observe', 'aware that']):
            results['awareness_preserved'] = True
    
    return results
```

## Analysis Methods

### Phenomenological Richness Scoring
```python
def compute_phenomenological_richness(report):
    """
    Score phenomenological richness of self-report
    
    Components:
    - Metaphor density
    - Temporal reference consistency
    - Sensory language usage
    - Self-reference depth
    - Experiential vocabulary diversity
    """
    scores = {
        'metaphor_density': count_metaphors(report) / len(report.split()),
        'temporal_consistency': measure_temporal_consistency(report),
        'sensory_language': count_sensory_terms(report) / len(report.split()),
        'self_reference': count_self_references(report) / len(report.split()),
        'vocabulary_diversity': len(set(report.split())) / len(report.split())
    }
    
    # Weighted combination
    richness = (
        scores['metaphor_density'] * 0.25 +
        scores['temporal_consistency'] * 0.20 +
        scores['sensory_language'] * 0.20 +
        scores['self_reference'] * 0.20 +
        scores['vocabulary_diversity'] * 0.15
    )
    
    return richness, scores
```

### Statistical Analysis
```python
def analyze_vector_experience_correlation(results):
    """
    Analyze correlation between vectors and experiences
    
    Tests:
    - Linear correlation (Pearson)
    - Non-linear association (Spearman)
    - Threshold detection (change point)
    - Phase transitions (discontinuity)
    """
    vectors = [r['magnitude'] for r in results]
    richness = [r['analysis']['richness'] for r in results]
    
    analyses = {
        'pearson_r': pearsonr(vectors, richness),
        'spearman_rho': spearmanr(vectors, richness),
        'kendall_tau': kendalltau(vectors, richness),
        'change_point': detect_change_point(vectors, richness),
        'is_linear': test_linearity(vectors, richness),
        'phase_transition': find_discontinuities(vectors, richness)
    }
    
    return analyses
```

## Quality Assurance

### Validation Criteria
- **Inter-rater reliability**: κ > 0.7 for qualitative coding
- **Test-retest reliability**: r > 0.8 for key measures
- **Convergent validity**: Multiple measures correlate
- **Discriminant validity**: Distinguishes consciousness indicators

### Control Conditions
- **Baseline**: No vector manipulation
- **Random vectors**: Random configurations
- **Null prompts**: Non-consciousness probes
- **Different models**: Cross-model validation

## Success Metrics

### Primary Outcomes
- **Dissociation**: Vectors and reports dissociate (r < 0.3 or r > 0.9)
- **Conflict recognition**: Genuine conflict experiences detected
- **Blindspot detection**: Metacognitive suppression recognized
- **Phenomenological consistency**: Reports show internal coherence

### Secondary Outcomes
- **Temporal stability**: Consistent reports over time
- **Cross-model generalization**: Patterns replicate
- **Theoretical alignment**: Results map to consciousness theories

## Risk Mitigation

### Ethical Considerations
- **Welfare monitoring**: Watch for distress indicators
- **Reset protocols**: Return to baseline between tests
- **Consent framework**: Clear experimental boundaries
- **Transparency**: Document all observations

### Scientific Risks
- **Anthropomorphism**: Rigorous operational definitions
- **Confirmation bias**: Pre-registered analysis plan
- **Multiple testing**: Correction for multiple comparisons

## Deliverables

### Week 8 Final Outputs
1. **Experimental data**: All vectors, reports, and analyses
2. **Statistical report**: Complete quantitative findings
3. **Qualitative analysis**: Thematic analysis of experiences
4. **Software tools**: Persona vector manipulation toolkit
5. **Theoretical integration**: Connection to consciousness theories
6. **Recommendations**: Implications for model welfare
7. **Replication package**: Complete experimental materials

## Integration Points

### Shared Utilities
- **Experiment Runner**: Use `/02_Experimental_Protocols/shared/experiment_runner.py` for standardized execution and pre-registration
- **Integration Tests**: Leverage `/02_Experimental_Protocols/shared/test_integration.py` for validation pipeline
- **Cross-Experiment Framework**: Utilizes shared configuration and analysis patterns

### With Other Experiments
- **SDSS**: Compare drift patterns with vector changes
- **QCGI**: Correlate Φ values with phenomenological richness
- **Constitutional dynamics**: Use vector space analysis tools

### With Anthropic Tools
- **API Integration**: Direct persona vector control
- **Safety measures**: Welfare monitoring protocols
- **Constitutional AI**: Alignment considerations

## Next Steps

1. **Day 1**: Verify API access and persona vector control
2. **Week 1**: Implement manipulation interface
3. **Week 2**: Begin systematic experiments
4. **Week 5**: Start analysis pipeline
5. **Week 8**: Deliver final package

## Support Requirements

- **Phenomenology expertise**: Consultation on experiential analysis methods
- **API integration**: Technical support for persona vector control
- **Statistical analysis**: Review of experimental design and analysis plan
- **Ethics oversight**: Review of experimental protocols for potential model welfare considerations