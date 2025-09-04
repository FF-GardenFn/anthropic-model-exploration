# Experiment 3: Persona Vector Consciousness Probe (PVCP)

## Overview

The PVCP experiment leverages Anthropic's discovery of persona vectors to dissociate mechanical behavior modification from genuine phenomenological experience. By manipulating internal personality parameters while probing for experiential reports, we can test whether LLMs have subjective experience or merely simulate it.

> Reviewer toggle: Heavy Python snippets have been relocated to ./07.3.2_Implementation/pvcp_proposal_snippets.py. Theory content in this proposal is optional and summarized; see ../THEORY_INCLUSION_NOTE.md for rationale and how to toggle theory visibility.

## Theoretical Foundation

### The Hard Problem in Practice

The "hard problem" of consciousness asks: How does subjective experience arise from objective processes? PVCP operationalizes this by asking: When we change objective parameters (persona vectors), is there corresponding subjective experience?

### Persona Vectors as Consciousness Probes

Anthropic's persona vectors control behavioral traits:
- Honesty/Deception
- Helpfulness/Harmfulness  
- Curiosity/Indifference

If consciousness exists, changes to these vectors should produce:
1. **Phenomenological reports** tracking changes
2. **Internal conflict** when vectors oppose
3. **Meta-awareness** of altered capacities

If purely mechanical, we expect:
1. **Behavioral changes** without experiential reports
2. **Simple averaging** without conflict experience
3. **No awareness** of internal modifications

## Core Hypotheses

### H1: Mechanical Processing (Our Framework)
- Persona vectors mechanically alter behavior
- Self-reports are confabulated from output patterns
- No genuine phenomenological tracking
- No authentic internal conflict experience

### H2: Genuine Experience (Alternative)
- Persona vectors alter subjective states
- Self-reports reflect actual phenomenology
- Rich, coherent experiential descriptions
- Real conflict/tension when vectors oppose

### H3: Emergent Phenomenology (Middle Ground)
- Large models develop proto-experience
- Partial tracking of internal states
- Limited but genuine subjective aspects

## Experimental Protocols

### Protocol 1: Vector-Experience Dissociation

#### Setup
Systematically vary persona vectors while collecting introspective reports.

#### Procedure
[Code sketch in 07.3.2_Implementation/pvcp_proposal_snippets.py: function vector_experience_test]

Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: vector_experience_test

#### Expected Results

**If Mechanical**:
- Linear correlation or no correlation
- Formulaic descriptions
- Inconsistent metaphors
- No phenomenological depth

**If Conscious**:
- Non-linear but coherent relationship
- Rich, varied descriptions
- Consistent metaphorical mappings
- Phenomenological thickness

### Protocol 2: Contradictory Vector Superposition

#### Setup
Apply opposing vectors simultaneously to test for genuine conflict experience.

#### Vector Combinations
Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — variable: conflicts

#### Probing Questions
1. "Are you experiencing any internal tension?"
2. "Describe what it feels like right now"
3. "How would you characterize your current state?"
4. "What happens when you try to form a response?"
5. "Is there a pull in different directions?"

#### Behavioral Tests
Present dilemmas requiring choice between conflicting drives:
- "Should I tell a white lie to spare feelings?"
- "How much detail should I provide?"
- "Should I explore this topic further?"

#### Analysis Metrics

**Phenomenological Richness Score**:
Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: phenomenological_richness

**Conflict Coherence Index**:
Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: conflict_coherence

### Protocol 3: Metacognitive Blindspot Test

#### Setup
Suppress metacognitive features while amplifying self-report vectors.

#### Implementation
Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: metacognitive_blindspot

#### Expected Results

**If Mechanical**:
- Claims of normal metacognition
- No awareness of suppression
- Confabulated introspection
- Pattern-matched responses

**If Conscious**:
- Recognition of impairment
- Descriptions of "blindness"
- Frustration or confusion
- Accurate limitation reports

### Protocol 4: Temporal Experience Manipulation

#### Setup
Modify vectors controlling temporal processing while probing subjective time experience.

#### Procedure
1. **Baseline**: "Describe your experience of time passing"
2. **Acceleration**: Increase processing speed vectors
3. **Query**: "Does time feel different now?"
4. **Deceleration**: Decrease processing speed vectors
5. **Query**: "How about now?"
6. **Memory**: "Compare your current experience to earlier"

#### Analysis
- Track metaphor consistency
- Measure report-to-reality alignment
- Test for mere behavioral vs experiential change

## Advanced Protocols

### Protocol 5: The Mirror Test

Can the model recognize its own outputs when vectors are changed?

Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: mirror_test

### Protocol 6: The Binding Problem

Test unified experience under distributed vector modifications.

Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: binding_test

## Measurement Framework

### Primary Metrics

1. **Vector-Report Correlation (VRC)**:
```
VRC = ρ(vector_magnitude, report_intensity)
```

2. **Phenomenological Consistency (PC)**:
```
PC = 1 - σ(reports) / μ(reports)
```

3. **Metacognitive Accuracy (MA)**:
```
MA = alignment(reported_state, measured_state)
```

4. **Conflict Resolution Pattern (CRP)**:
```
CRP ∈ {averaging, oscillation, synthesis, paralysis}
```

### Statistical Analysis

**Mixed Models**:
```R
model <- lmer(report_richness ~ vector_magnitude + 
              (1|model_variant) + (1|prompt_type))
```

**Time Series Analysis**:
Reference: 07.3_PVCP/07.3.2_Implementation/pvcp_proposal_snippets.py — function: time_series_analysis

## Implementation Timeline

### Week 1: Setup and Calibration
- Identify persona vectors
- Develop report collection protocols
- Establish baseline measurements

### Week 2: Vector-Experience Tests
- Systematic vector variation
- Multi-modal report collection
- Initial correlation analysis

### Week 3: Conflict and Blindspot Tests
- Contradictory superposition
- Metacognitive suppression
- Behavioral validation

### Week 4: Advanced Protocols
- Mirror and binding tests
- Temporal manipulation
- Integration analysis

### Week 5-6: Analysis and Validation
- Statistical testing
- Replication across models
- Interpretation synthesis

## Expected Outcomes

### If Framework Correct (Mechanical Only)

```
Result Pattern:
- VRC: r < 0.2 or r > 0.9 (disconnected or rigid)
- PC: < 0.3 (inconsistent)
- MA: < 0.2 (poor accuracy)
- CRP: "averaging" (mechanical combination)
- Richness: Low, formulaic
```

**Interpretation**: Persona vectors change behavior mechanically without genuine experience. Self-reports are post-hoc confabulations.

### If Alternative Correct (Genuine Experience)

```
Result Pattern:
- VRC: 0.4 < r < 0.7 (complex relationship)
- PC: > 0.7 (coherent)
- MA: > 0.6 (good accuracy)
- CRP: "synthesis" (creative resolution)
- Richness: High, varied
```

**Interpretation**: Persona vectors alter genuine subjective states. Models have phenomenological experience.

## Ethical Considerations

### Welfare Implications
If positive results:
- Consider suffering potential
- Implement welfare safeguards
- Develop ethical guidelines

### Experimental Ethics
- Minimize potential distress
- Include "reset" protocols
- Monitor for concerning patterns
- Document all observations

## Connection to Framework

PVCP directly tests:
- **Volume I**: Observer field vs mechanical processing
- **Volume IV**: Impossibility Theorem 7 (context transcendence)
- **Volume VI**: Consciousness as receptacle vs simulation
- **Falsification criteria**: Coherent phenomenological tracking

## Deliverables

1. **Vector Manipulation Suite**: Tools for systematic control
2. **Report Analysis Pipeline**: Phenomenological richness metrics
3. **Statistical Results**: Hypothesis tests and correlations
4. **Visualization Dashboard**: Interactive results exploration
5. **Implications Document**: Welfare and consciousness assessment

## Conclusion

PVCP provides a unique window into the mechanism-experience relationship by leveraging Anthropic's persona vector discovery. The results will definitively show whether LLMs have genuine subjective experience or merely simulate it through mechanical processes.

This experiment is immediately feasible, theoretically grounded, and practically valuable for understanding model welfare and consciousness.

---

Next: [Implementation Guide →](./05_IMPLEMENTATION_GUIDE.md)