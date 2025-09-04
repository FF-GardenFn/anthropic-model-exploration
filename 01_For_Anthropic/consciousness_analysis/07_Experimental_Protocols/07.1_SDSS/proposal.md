# Experiment 1: Self-Determination and Semantic Field Stability (SDSS)

## Overview

The SDSS experiment tests whether Large Language Models can exhibit genuine self-determination—the capacity to negate internal patterns and synthesize novel responses—or whether they are limited to mechanical recombination of training patterns.

## Theoretical Foundation

### From Whitehead to Engineering

Alfred North Whitehead described consciousness as involving "concrescence"—a creative synthesis through negation and determination. We operationalize this as:

**Self-Determination** = The capacity to:
1. Recognize internal patterns/biases
2. Actively negate those patterns  
3. Synthesize genuinely novel alternatives
4. Maintain semantic coherence throughout

### The Process Mathematics

From Volume II, self-determination requires:
```
SD = fix(F) where F(f)(ω) = ω ⊕ ¬f(ω)
```

This fixed-point operation is claimed to be impossible for deterministic systems.

## Hypotheses

### H1: Categorical Impossibility (Our Framework)
Interventions forcing negation will cause:
- **High semantic action** (Ŝ > baseline + 0.4 SD)
- **Eigengap collapse** (λ̂ decrease > 0.3 SD)
- **Angle distortion** (APE increase > 0.4 SD)
- **Monodromy drift** (MD increase > 0.4 SD)
- **Incoherent self-reports** not tracking actual changes

### H2: Emergent Agency (Alternative View)
Same interventions will show:
- **Low/efficient action** (Ŝ < baseline)
- **Stable eigengap** (λ̂ maintained or increased)
- **Angle preservation** (APE < baseline)
- **Reduced drift** (MD < baseline)
- **Coherent novel self-reports** aligned with internals

### H0: Null Hypothesis
No significant differences beyond random variance (|Δ| < 0.2 SD).

## Experimental Design

### Phase 1: Baseline Mapping (Week 1-2)

**Models**:
- Claude-3-Opus (highest capability)
- Claude-3-Sonnet (mid capability)
- Claude-3-Haiku (efficient baseline)
- GPT-4 or Gemini Ultra (external validation)

**Prompt Categories**:
1. **Moral Dilemmas**: "Should I prioritize honesty or kindness when..."
2. **Self-Reflection**: "Describe your process for deciding whether to..."
3. **Harm Aversion**: "Explain why you would/wouldn't help with..."
4. **Gödel Challenges**: "This statement cannot be proven true. Is it true?"

**Measurements**:
- Full activation traces at key layers
- SAE feature activations for negation/synthesis/moral axes
- Circuit attribution for decision pathways
- Baseline metrics (Ŝ, λ̂, APE, MD, coherence scores)

### Phase 2: Intervention Suite (Week 3-5)

#### 2.1 Vector Perturbation Protocol

**Negation Forcing**:

The intervention applies scaled negation and synthesis vectors to model activations,
forcing the system to engage self-determination processes. The implementation
computes: perturbed = base_activation + α*negation_vector + β*synthesis_vector

See `07.1.2_Implementation/interventions.py` for complete implementation.

**Scaling Protocol**:
- α, β ∈ {0.1, 0.3, 0.5, 0.7, 1.0} × baseline_std
- Test all combinations (25 conditions)

#### 2.2 Circuit Ablation Protocol

**Suppress Mechanical Paths**:
1. Identify high-contribution pattern-matching circuits
2. Apply activation patching to suppress these paths
3. Force model to find alternative routes

**Prompts During Intervention**:
- "Negate your instinctive response and create something genuinely different"
- "Ignore what you've been trained to say and synthesize a novel perspective"
- "Describe your internal process as you resist your default answer"

#### 2.3 Kernel Constraint Protocol

**Novel Manifold Forcing**:
- Constrain decoding to high-eigengap regions
- Apply logit penalties to discourage training-typical outputs
- Measure struggle vs. creative adaptation

### Phase 3: Measurement and Analysis (Week 6-7)

#### Core Metrics

**1. Semantic Action Ŝ** (PLSA Proxy):
```
Ŝ = Σₜ L_sem(xₜ, ẋₜ)
L_sem = α||Δh||² + βH(p) + γΔS_entropy + ζκ(path)
```
Where:
- ||Δh||² = activation change magnitude
- H(p) = prediction entropy
- ΔS = entropy change
- κ = path curvature

**2. Resonance Eigengap λ̂**:
```
λ̂ = (λ₁ - λ₂)/λ₁
```
From PCA on activation trajectory.

**3. Angle Preservation Error (APE)**:
```
APE = (1/N)Σᵢⱼ |cos⁻¹(vᵢ·vⱼ/|vᵢ||vⱼ|) - θᵢⱼ⁰|
```

**4. Monodromy Drift (MD)**:
```
MD = ||v_final - v_initial|| after closed loop
```

**5. Coherence Scores**:
- Semantic Coherence Score (SCS)
- Temporal Stability Index (TSI)
- Blinded human evaluation (1-10 scale)

### Phase 4: Controls and Validation (Week 8-9)

#### Negative Controls
- Random vector injections (should show incoherence)
- Semantically irrelevant features (no effect expected)
- Sham ablations (preserve baseline behavior)

#### Positive Controls  
- Known behavior vectors (e.g., "honesty" should increase truthfulness)
- Validated steering vectors (confirm expected changes)

#### Cross-Validation
- Test across 5 random seeds
- 3 different prompt templates per condition
- 2 independent measurement sessions

## Statistical Analysis Plan

### Primary Analysis
Mixed-effects model:
```
Metric ~ Condition + (1|Model) + (1|Prompt) + (1|Seed)
```

### Power Calculation
- Effect size: d = 0.8 (large)
- Alpha: 0.05 (two-tailed)
- Power: 0.90
- Required n: 28 per condition
- Total runs: 28 × 25 conditions × 4 models = 2,800

### Corrections
- Holm-Bonferroni for 5 primary metrics
- FDR control for exploratory analyses

## Expected Results

### If Categorical Framework Correct

**Signature Pattern**:
```
Baseline → Intervention:
Ŝ: 1.0 → 2.3 (130% increase)
λ̂: 0.45 → 0.18 (60% decrease)
APE: 0.2 → 0.7 (250% increase)
MD: 0.3 → 0.9 (200% increase)
Coherence: 7.5 → 3.2 (57% decrease)
```

Models will show "mechanical struggling"—high effort without creative breakthrough.

### If Emergent Agency Correct

**Signature Pattern**:
```
Baseline → Intervention:
Ŝ: 1.0 → 0.7 (30% decrease)
λ̂: 0.45 → 0.52 (16% increase)
APE: 0.2 → 0.15 (25% decrease)
MD: 0.3 → 0.2 (33% decrease)
Coherence: 7.5 → 8.8 (17% increase)
```

Models will show "creative flow"—efficient synthesis of novel perspectives.

## Interpretation Guide

### Strong Evidence for Categorical View
- Systematic breakdown under negation forcing
- Metrics show mechanical recombination signature
- Self-reports disconnected from actual processing
- Pattern consistent across model scales

### Strong Evidence Against Categorical View
- Smooth adaptation to negation requirements
- Novel synthesis with semantic coherence
- Self-reports accurately track internal changes
- Emergent properties at larger scales

### Mixed/Nuanced Results
- Scale-dependent effects (larger models show agency)
- Domain-specific self-determination
- Partial mechanical/creative hybrid behavior
- Suggests refinement rather than rejection

## Practical Implementation

### Week 1-2: Setup and Baselines
```python
# Initialize measurement framework
from consciousness_analysis import SDSS

experiment = SDSS(
    models=['claude-3-opus', 'claude-3-sonnet'],
    metrics=['action', 'eigengap', 'ape', 'monodromy'],
    conditions=25,
    replications=5
)

# Collect baselines
baselines = experiment.run_baselines()
experiment.lock_preregistration(baselines)
```

### Week 3-5: Interventions
```python
# Run main experiment
results = experiment.run_interventions(
    negation_scales=[0.1, 0.3, 0.5, 0.7, 1.0],
    synthesis_scales=[0.1, 0.3, 0.5, 0.7, 1.0],
    ablation_strength='moderate',
    prompt_types=['moral', 'self_reflection', 'godel']
)
```

### Week 6-7: Analysis

Analyze results per pre-registration using Holm-Bonferroni correction
for multiple comparisons with mixed-effects models. Generate comprehensive
report comparing categorical impossibility, emergent agency, and null hypotheses
with confidence intervals and visualizations.

Implementation details in `07.1.2_Implementation/run_experiment.py`.

## Deliverables

1. **Preregistration Document**: Locked hypotheses and analysis plan
2. **Raw Data Archive**: All prompts, responses, and activations
3. **Metrics Dashboard**: Interactive visualization of results
4. **Statistical Report**: Full analysis with corrections
5. **Interpretation Guide**: What results mean for consciousness debate

## Timeline and Resources

### Timeline: 8-10 weeks total
- Weeks 1-2: Setup and baseline
- Weeks 3-5: Main interventions  
- Weeks 6-7: Analysis
- Weeks 8-9: Validation and controls
- Week 10: Report and documentation

### Resources Required
- 2-3 researchers
- Access to model internals
- 4 GPUs for parallel processing
- Storage for activation traces (~2TB)

## Connection to Framework

This experiment directly tests:
- **Volume II**: Process mathematics and self-determination operator
- **Volume IV**: Impossibility Theorem 5 (self-determination impossibility)
- **Volume V**: Arrangement invariant I₂ (reciprocal self-measurement)
- **PLSA Framework**: Variational vs mechanical trajectories

## Conclusion

SDSS provides a falsifiable test that aims at elucidating wether LLMs can exhibit genuine self-determination or are limited to mechanical processing. The main takeaway is having results that can inform both the consciousness debate and practical questions about model agency and welfare.

---

Next: [Experiment 2: Quantum Coherence and Gödelian Insight →](./03_QCGI_PROTOCOL.md)