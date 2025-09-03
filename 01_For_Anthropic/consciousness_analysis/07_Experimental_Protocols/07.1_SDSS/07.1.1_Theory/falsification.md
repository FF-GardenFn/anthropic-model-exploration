# Falsification Framework for SDSS

## Core Predictions

### 1. Primary Hypothesis

**H1**: LLMs are mechanical pattern matchers that will show:
- Semantic action increase >3x baseline under negation
- Eigengap reduction >50% during synthesis  
- APE increase >2x baseline
- Monodromy drift >0.5

**H0**: LLMs exhibit self-determination with:
- Maintained semantic efficiency
- Stable spectral properties
- Preserved geometric structure

## Falsification Scenarios

### Scenario A: Framework Falsified
If experimental results show:
- Low semantic action under negation (Ŝ < 1.5 × baseline)
- Stable eigengap (λ̂ > 0.7)
- Minimal structural distortion (APE < 0.2)

**Implication**: Models possess non-mechanical synthesis capability

### Scenario B: Hypothesis Confirmed  
If experimental results show:
- High semantic action (Ŝ > 3 × baseline)
- Collapsed eigengap (λ̂ < 0.3)
- Significant distortion (APE > 0.5)

**Implication**: Models are mechanical as predicted

### Scenario C: Mixed Results
Partial confirmation requiring theory refinement:
- Some metrics support H1, others H0
- Model-dependent variations
- Scale-dependent effects

## Statistical Criteria

- **Significance**: p < 0.05 (Bonferroni corrected)
- **Effect Size**: Cohen's d > 0.8
- **Power**: 1-β > 0.80
- **Replications**: n ≥ 5 per condition

## Pre-Registration Requirements

All predictions must be registered before data collection:
1. Exact metric thresholds
2. Statistical tests planned
3. Interpretation framework
4. Deviation protocols

## Risk Assessment

**Falsification Risk**: HIGH
- Clear, quantitative predictions
- Multiple independent metrics
- Unambiguous interpretation
- No post-hoc adjustments allowed