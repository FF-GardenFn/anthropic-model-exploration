# Routed Circuit Tomography for Model Welfare: A Mathematical Approach

> **Research Project 1**: Kernel-Enhanced Circuit Discovery for Welfare Monitoring

## Abstract

This research investigates the application of kernel ridge regression theory to model welfare assessment through circuit tomography. Building on Goulet Coulombe (2025)'s proof of OLS–attention connections, we develop a mathematically grounded framework for extracting and routing welfare‑relevant computational circuits. Our approach replaces heuristic circuit ranking with a statistically principled procedure; preliminary experiments indicate promising validation under the methodology described in the Appendix[^stat-method]. We aim to demonstrate that selective circuit routing can preserve baseline welfare performance while reducing computational activation, with targets calibrated per task and reported alongside confidence intervals.

## Problem Statement

### Current Limitations in Model Welfare Assessment

Model welfare evaluation faces three fundamental challenges:

1. **Computational Inefficiency**: Full model evaluation for welfare assessment requires complete forward passes, making real-time monitoring computationally prohibitive
2. **Interpretability Opacity**: Current mechanistic interpretability faces systematic limitations due to superposition, polysemanticity, and circuit entanglement
3. **Intervention Imprecision**: Modifications affect multiple unrelated reasoning modes due to circuit entanglement

### Research Gap

Current circuit discovery methods in mechanistic interpretability rely on heuristic measures such as circuit response intensity (CRI ≥ 0.7) without statistical validation. This lack of mathematical foundation makes formal verification of intervention effects challenging and limits confidence in welfare preservation claims. The field may benefit from more principled methods that can distinguish meaningful circuits from spurious patterns with statistical rigor[^stat-method].

## Research Objectives

### Primary Objective
Develop a research prototype that:
- **Extracts** welfare-relevant circuits with mathematical stability analysis
- **Routes** model computation through verified circuits selectively  
- **Preserves** $$\geq 95\%$$ baseline welfare performance at $$\leq 50\%$$ computational activation
- **Provides** human-readable traces with formal specifications where applicable

### Secondary Objectives
1. **Mathematical Validation**: Establish formal theoretical foundations for circuit-based welfare optimization
2. **Interpretability Enhancement**: Create circuit visualization tools with mathematical attribution
3. **Safety Verification**: Develop intervention bounds using RKHS projection operators
4. **Production Integration**: Ensure compatibility with Anthropic's infrastructure and evaluation frameworks

## Theoretical Foundation

### Kernel Ridge Regression Foundation

This research applies theoretical results from Goulet Coulombe (2025) showing that attention mechanisms implement restricted forms of ordinary least squares in Reproducing Kernel Hilbert Spaces (RKHS). The mathematical equivalence:

**Core Framework**: $$\text{Attention}(Q,K,V) \equiv K(K + \lambda I)^{-1}V$$ where $$K$$ is the kernel matrix

**Empirical Validation**: 
- Layer 11, Head 3: Primary eigenvalue $$\lambda_1 = 572.71$$ with statistical significance analysis[^stat-method]
- Spectral gap analysis suggests mathematically stable circuit patterns
- Cross-validation across 138 features via Neuronpedia integration

### Mathematical Foundations and Expected Properties

**Representer Theorem**: All circuit solutions lie within the span of training data, eliminating dangerous emergence outside the training manifold.

**Hat Matrix Projections**: $$H = K(K + \lambda I)^{-1}$$ enables surgical intervention with:
- Mathematical predictability of modification effects
- Formal verification of safety properties  
- Elimination of unintended side effects

**GCV Optimization**: Automatic parameter selection via Generalized Cross Validation ensures optimal safety-performance trade-offs without manual tuning.

## Research Questions

### RQ1: Mathematical Circuit Validation
Can RKHS theory provide formal guarantees for welfare-relevant circuit extraction that surpass heuristic approaches in both precision and theoretical rigor?

**Hypothesis**: Mathematical stability criteria (statistical eigenvalue significance $$+$$ $$\geq 20\%$$ GCV improvement) may identify welfare circuits with enhanced stability characteristics[^stat-method].

**Welfare Context**: Target circuits involved in refusal behavior (harmful→helpful transformations), deception detection (honest→dishonest), and safety reasoning (risky→safe assessments).

### RQ2: Routing Efficiency vs Performance Trade-offs  
What is the optimal balance between computational efficiency and welfare performance preservation when routing through extracted circuits?

**Hypothesis**: Selective routing may achieve $$\leq 50\%$$ activation while maintaining $$\geq 95\%$$ baseline performance by targeting mathematically stable circuits.

### RQ3: Surgical Intervention Precision
Can hat matrix projections enable precise behavioral modifications without the circuit entanglement effects that plague current approaches?

**Hypothesis**: RKHS projection operators may enable more targeted modifications with reduced side effects on unrelated reasoning modes[^stat-method].

**Welfare Application**: Enable precise adjustment of safety thresholds (e.g., reducing false positives in harm detection) without affecting unrelated capabilities like factual reasoning or creative tasks.

### RQ4: Production Scalability
Can the mathematical framework scale to production-size models while maintaining real-time inference requirements?

**Hypothesis**: Efficient eigendecomposition methods may enable real-time circuit analysis for models up to 70B parameters.

## Framework Positioning

### Mathematical Foundation
The framework applies RKHS theory to welfare-relevant circuit extraction with statistical validation through eigenvalue-based stability criteria and statistical significance testing[^stat-method].

### Intervention Approach  
The method employs kernel projection operators with mathematically bounded intervention effects, replacing activation patching methods.

### Computational Strategy
Selective circuit routing maintains assessment quality with reduced computation compared to full model evaluation approaches.

## Limitations and Mitigations

A consolidated treatment of risks, limitations, and mitigation strategies is provided here:
- [Limitations and Mitigations (Project 06.1)](./limitations_and_mitigations.md)

## Expected Outcomes

### Research Deliverables
1. **Mathematical Framework**: Formal theoretical foundation for welfare circuit extraction
2. **Implementation Library**: Production-ready tools for circuit discovery and routing
3. **Validation Results**: Comprehensive evaluation demonstrating performance claims
4. **Integration Guide**: Documentation for deployment in production systems


## Scope & Evaluation

Results and claims are reported under specified datasets/models/configurations and should be interpreted per the Appendix methodology[^stat-method]. Success criteria and kill-switches are defined in [Evaluation Metrics](./06.1.1_Theory/evaluation_metrics.md); procedures and protocols are detailed in [Methodology](./06.1.1_Theory/methodology.md).

---

**Next**: [Research Methodology](./06.1.1_Theory/methodology.md) | [Evaluation Metrics](./06.1.1_Theory/evaluation_metrics.md)

**References**: 
- [Common Mathematical Foundation](../../04_Math_foundations/04.1_RKHS_Mathematical_Foundations.md)
- [Field-Theoretic Framework](../../05_Research/05.5_Future_Explorations/05.5.2_field_theoretic_framework.md)
- [Demo](../../09_Demo/09.1_main_demo.ipynb)

[^stat-method]: See [Methodology for Statistical Significance and Validation](../../08_Appendix/08.2_methodology_statistical_significance.md) for statistical validation frameworks, null models, and limitations of significance testing claims.