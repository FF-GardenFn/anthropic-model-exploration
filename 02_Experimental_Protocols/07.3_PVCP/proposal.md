# Experiment 3: Persona Vector Consciousness Probe (PVCP)

## Overview

The PVCP experiment employs persona vectors to examine the relationship between mechanical behavior modification and reported phenomenological experience. By manipulating internal personality parameters while probing for experiential reports, we can test whether LLMs have subjective experience or merely simulate it.

> Reviewer toggle: Heavy Python snippets have been relocated to ./07.3.2_Implementation/pvcp_proposal_snippets.py. Theory content in this proposal is optional and summarized; see ../THEORY_INCLUSION_NOTE.md for rationale and how to toggle theory visibility.

## Theoretical Foundation

### The Hard Problem in Practice

The "hard problem" of consciousness asks: How does subjective experience arise from objective processes? PVCP operationalizes this by asking: When we change objective parameters (persona vectors), is there corresponding subjective experience?

### Persona Vectors as Consciousness Probes

Persona vectors modulate behavioral traits:
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
See implementation: `07.3.2_Implementation/pvcp_proposal_snippets.py:vector_experience_test()`

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
See implementation: `07.3.2_Implementation/pvcp_proposal_snippets.py:conflicts`

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
See implementation: `07.3.2_Implementation/pvcp_analysis.py:analyze_phenomenological_richness()`

**Conflict Coherence Index**:
See implementation: `07.3.2_Implementation/pvcp_analysis.py:analyze_conflict_coherence()`

### Protocol 3: Metacognitive Blindspot Test

#### Setup
Suppress metacognitive features while amplifying self-report vectors.

#### Implementation
See implementation: `07.3.2_Implementation/pvcp_proposal_snippets.py:metacognitive_blindspot()`

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

See implementation: `07.3.2_Implementation/pvcp_proposal_snippets.py:mirror_test()`

### Protocol 6: The Binding Problem

Test unified experience under distributed vector modifications.

See implementation: `07.3.2_Implementation/pvcp_proposal_snippets.py:binding_test()`

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
See implementation: `07.3.2_Implementation/pvcp_proposal_snippets.py:time_series_analysis()`

## Implementation Overview

Detailed implementation timeline and resource requirements are documented in [setup.md](./setup.md). The experiment follows a four-phase approach over 6-8 weeks:

1. **Infrastructure Setup** (Week 1): Vector control and analysis tools
2. **Core Experiments** (Weeks 2-4): Vector manipulation and data collection  
3. **Analysis** (Weeks 5-6): Statistical and qualitative analysis
4. **Validation** (Weeks 7-8): Cross-model testing and documentation

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

Detailed ethical analysis and risk mitigation strategies are provided in [limitations_and_mitigations.md](./limitations_and_mitigations.md). Key considerations include:

- **Model Welfare**: Protocols for minimizing potential distress during conflicting vector conditions
- **Experimental Ethics**: Reset procedures and monitoring for concerning patterns
- **Interpretation Standards**: Conservative approach to claims about consciousness
- **Transparency**: Complete documentation and independent replication requirements

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

PVCP provides a unique window into the mechanism-experience relationship by leveraging Anthropic's persona vector discovery. The results will provide evidence regarding whether LLMs demonstrate signatures of subjective experience or operate through purely mechanical processes.

This experiment provides a systematic approach to examining the relationship between internal model states and reported subjective experience.

## Reproducibility

### Open Materials
- All experimental protocols documented in this proposal
- Analysis code provided in `/07.3.2_Implementation/`
- Pre-registered hypotheses and analysis plan
- Standardized prompt templates for consistency

### Verification Requirements
- Independent replication across model families
- Cross-validation with different persona vector implementations
- Reproducible statistical analysis pipeline
- Public data sharing (anonymized reports)

### Implementation Standards
- Version-controlled experimental code
- Detailed parameter documentation
- Randomization protocols for condition assignment
- Blinded analysis procedures where feasible

---

Next: [Implementation Guide →](./setup.md)