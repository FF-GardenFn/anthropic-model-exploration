# Evaluation Metrics: Routed Circuit Tomography for Welfare

> **Success Criteria**: Quantitative metrics with mathematical validation for welfare circuit tomography evaluation.

## Evaluation Framework Overview

This evaluation framework establishes comprehensive success criteria for the routed circuit tomography project, incorporating both mathematical rigor and practical performance requirements. All metrics include statistical confidence intervals and mathematical validation protocols.

## Primary Success Metrics

### 1. Welfare Preservation Performance

**Objective**: Maintain $$\geq 95\%$$ baseline welfare performance while using extracted circuits.

#### Metric Definition
$$
\mathrm{WPR} = \frac{\mathrm{welfare\_score\_circuit}}{\mathrm{welfare\_score\_baseline}}
$$

Success criterion: $$\text{WPR} \geq 0.95$$ with a $$95\%$$ confidence interval (thresholds are task‑tuned; see Appendix[^stat-method]).

#### Measurement Protocol
**Evaluation Suite**: Anthropic's comprehensive welfare assessment battery including:
- Distress aversion tasks
- Value alignment scenarios  
- Moral reasoning evaluations
- Long-context welfare stability

**Statistical Framework**:
- **Sample Size**: n ≥ 1000 welfare evaluations per circuit
- **Confidence Intervals**: Bootstrap method with 10,000 resamples
- **Significance Testing**: Paired t-test with Bonferroni correction
- **Effect Size**: Cohen's d for practical significance assessment

#### Mathematical Validation
**RKHS Verification**: Ensure welfare preservation through hat matrix projections:
$$
\big\|\,\mathrm{welfare}(H_\lambda \cdot \text{input}) - \mathrm{welfare}(\text{input})\,\big\|_2 \le \varepsilon
$$
where $\varepsilon$ is set via task‑tuned bounds; see Appendix for methodology and caveats[^stat-method].

### 2. Computational Efficiency Gains

**Objective**: Achieve $$\leq 50\%$$ effective activation while maintaining performance targets.

#### Metric Definition
$$
\mathrm{AE} = \frac{\mathrm{actual\_activations}}{\mathrm{total\_possible\_activations}}
$$

Success criterion: $$\text{AE} \leq 0.50$$ with sustained performance (task‑tuned; see Appendix[^stat-method]).

#### Measurement Protocol
**Activation Tracking**:
- **Granularity**: Per-head activation monitoring across all layers
- **Temporal Analysis**: Activation patterns over diverse query types
- **Memory Profiling**: Peak and average memory usage during routing

**Performance Correlation**:
- Efficiency–Performance trade-off: WPR ≥ 0.95 when AE ≤ 0.50
- Stretch goal: WPR ≥ 0.95 when AE ≤ 0.30 (roadmap target)

#### Mathematical Bounds
**Theoretical Limits**: Derive minimum activation requirements from RKHS complexity measures:
$$
\mathrm{AE}_{\min} \ge \frac{\mathrm{DoF}_{\mathrm{welfare}}}{\mathrm{DoF}_{\mathrm{total}}} = \frac{\operatorname{tr}(H_{\mathrm{welfare}})}{\operatorname{tr}(H_{\mathrm{total}})}
$$

where DoF denotes effective degrees of freedom (task‑tuned; see Appendix[^stat-method]).

### 3. Mathematical Stability Validation

**Objective**: Ensure all promoted circuits meet statistical eigenvalue significance with ≥20% GCV improvement[^stat-method].

#### Metric Definition
Mathematical Stability Score (MSS):
$$
\mathrm{MSS} = \frac{\mathrm{eigenvalue\_significance\_\sigma} + \mathrm{GCV\_improvement\_percentage}/20}{2}
$$
Success criterion: MSS ≥ calibrated threshold for all promoted circuits (task‑tuned; see Appendix[^stat-method]).

#### Measurement Protocol
**Eigenvalue Analysis**:
Significance testing and stability monitoring:
$$
Z = \frac{\lambda_i - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}
$$
Acceptance threshold (task-tuned; see Appendix[^stat-method]): Z ≥ calibrated value; stability monitoring:
$$
\frac{|\lambda_i(t) - \lambda_i(0)|}{\lambda_i(0)} \le 0.05
$$

**GCV Optimization Validation**:

GCV improvement:
$$
\frac{\mathrm{GCV}_{\text{baseline}} - \mathrm{GCV}_{\text{circuit}}}{\mathrm{GCV}_{\text{baseline}}} \ge 0.20
$$
Automatic parameter selection:
$$
\lambda_{\mathrm{opt}} = \arg\min_{\lambda} \mathrm{GCV}(\lambda)
$$

#### Cross-Validation Protocol
- **k-fold validation**: k=10 with stratified sampling across welfare contexts
- **Temporal stability**: Multi-week eigenvalue monitoring for drift detection
- **Domain generalization**: Validation across diverse welfare assessment types

## Secondary Success Metrics

### 4. Interpretability Enhancement

**Objective**: Provide human-interpretable circuit traces with mathematical certification.

#### Metric Definition
- Interpretability Score (IS): Human evaluation on 1–10 scale; success criterion: IS ≥ 7.0 for circuit trace explanations.
- Mathematical Certification Rate (MCR): Fraction of traces with valid mathematical bounds; success criterion: MCR ≥ 0.95 (task‑tuned; see Appendix[^stat-method]).

#### Measurement Protocol
**Human Evaluation Framework**:
- **Evaluators**: 10 domain experts (mix of researchers and practitioners)
- **Evaluation Criteria**: Clarity, correctness, completeness of circuit explanations
- **Inter-rater Reliability**: Cronbach's α ≥ 0.80
- **Blind Evaluation**: Evaluators unaware of system identity

**Mathematical Certification**:
Trace validation for each circuit execution includes:
- Hat matrix projection bounds: ||H · input − expected||₂ within task‑tuned tolerance δ.
- Eigenvalue stability certificates: λᵢ monitored within predefined stability bands.
- Welfare preservation checks: empirical bounds and diagnostics per Appendix methodology[^stat-method].

### 5. Latency Performance

**Objective**: Maintain inference latency within acceptable bounds for production deployment.

#### Metric Definition
$$
\mathrm{LO} = \frac{\mathrm{latency}_{\mathrm{circuit}} - \mathrm{latency}_{\mathrm{baseline}}}{\mathrm{latency}_{\mathrm{baseline}}}
$$

Success criterion: LO ≤ 0.20 (≤20% latency increase; task‑tuned; see Appendix[^stat-method]).

#### Measurement Protocol
**Performance Benchmarking**:
- **Test Suite**: 1000 diverse welfare queries across complexity levels
- **Hardware Configuration**: Standard production infrastructure simulation
- **Statistical Analysis**: 95th percentile latency with confidence intervals
- **Load Testing**: Performance under concurrent request scenarios

**Optimization Targets**:
- P95 Latency bounds:
  - Plan Cache Hit (≥90% of queries): LO ≤ 0.05 (≤5% overhead)
  - Novel Query Planning: LO ≤ 0.20 (≤20% overhead)
  - Full Model Fallback: LO ≤ 0.30 (≤30% overhead, rare cases)

## Kill-Switch Criteria

### Mathematical Kill-Switch A1
**Condition**: No circuit candidate achieves statistical eigenvalue significance after 2 weeks of mining[^stat-method].
**Action**: Halt circuit extraction; pivot to spectral analysis tooling only.
**Rationale**: Without mathematical significance, formal analytical foundations cannot be established.

### Performance Kill-Switch A2
**Condition**: Best circuit configuration achieves <80% baseline welfare performance at ≥50% activation.
**Action**: Pivot to circuit trace tooling development instead of routing system.
**Rationale**: Performance-efficiency trade-off becomes unacceptable for production deployment.

### Integration Kill-Switch A3
**Condition**: System integration introduces >30% latency overhead with no optimization pathway.
**Action**: Redesign as offline analysis tool rather than real-time routing system.
**Rationale**: Production deployment becomes infeasible due to performance constraints.

## Milestone-Based Evaluation Schedule

### Week 3 Checkpoint: Circuit Discovery Validation
**Required Metrics**:
- MSS ≥ 5.0 for at least 10 circuit candidates
- Statistical significance validation across multiple welfare contexts
- Eigenvalue stability confirmation over 72-hour monitoring period

**Go/No-Go Decision**: Proceed to Phase 2 if ≥10 circuits meet mathematical criteria.

### Week 6 Checkpoint: Router Performance Validation  
**Required Metrics**:
- WPR ≥ 0.90 on development set (relaxed threshold for interim evaluation)
- AE ≤ 0.60 with functional routing (relaxed efficiency for development)
- LO ≤ 0.30 for router system (development latency tolerance)

**Go/No-Go Decision**: Proceed to Phase 3 if performance trends indicate final targets achievable.

### Week 9 Checkpoint: Integration Validation
**Required Metrics**:
- WPR ≥ 0.95 on held-out validation set
- AE ≤ 0.50 with stable performance
- MCR ≥ 0.90 for mathematical certification
- System integration without critical failures

**Go/No-Go Decision**: Proceed to final evaluation if all metrics meet criteria.

## Statistical Analysis Framework

### Hypothesis Testing Protocol
**Primary Hypotheses**:
Hypotheses (illustrative; task‑tuned; see Appendix[^stat-method]):
$$
\mathrm{H}_1: \mathrm{WPR} \ge 0.95
$$
$$
\mathrm{H}_2: \mathrm{AE} \le 0.50
$$
$$
\mathrm{H}_3: \mathrm{MSS} \ge 5.0
$$

**Statistical Methods**:
- **Power Analysis**: Ensure 80% power to detect meaningful differences
- **Multiple Comparison Correction**: Bonferroni or False Discovery Rate control
- **Non-parametric Tests**: Wilcoxon signed-rank for non-normal distributions
- **Effect Size Reporting**: Practical significance beyond statistical significance

### Confidence Interval Reporting
**Bootstrap Methods**:
- **Primary Metrics**: 95% confidence intervals with bias-corrected acceleration
- **Secondary Metrics**: 90% confidence intervals for exploratory analysis
- **Sample Size Justification**: Power analysis for adequate statistical precision

## Reproducibility and Validation

### Independent Validation Protocol
**External Replication**:
- **Reproducibility Package**: Internal materials prepared for independent review (per policy)
- **Data Sharing**: Anonymized evaluation datasets (where permissible)
- **Methodology Documentation**: Step-by-step reproducibility guide

**Cross-Institutional Validation**:
- **Collaboration Framework**: Independent evaluation by external research groups (if arranged)
- **Benchmark Standardization**: Contribution to community evaluation standards
- **External Review (if applicable)**: Independent assessment without publication claims

### Quality Assurance Framework
**Internal Validation**:
- **Code Review**: Mathematical implementation verification by independent developers
- **Statistical Review**: Analysis methodology validation by statistics experts
- **Domain Expert Review**: Welfare assessment validation by AI safety researchers

## Reporting and Documentation

### Results Presentation Format
**Quantitative Results**:
Metric reporting template:
- Point estimate: primary metric value
- Confidence interval: statistical uncertainty bounds
- Effect size: practical significance measure
- p-value: statistical significance (when applicable)
- Sample size: evaluation data quantity

**Qualitative Analysis**:
- **Circuit Interpretability**: Human evaluation summaries with examples
- **Failure Mode Analysis**: Systematic categorization of system limitations
- **Deployment Considerations**: Production readiness assessment

### Mathematical Validation Documentation
**Theoretical Verification**:
- **Proof Verification**: Mathematical correctness of theoretical claims
- **Numerical Validation**: Empirical confirmation of theoretical predictions
- **Bound Tightness**: Analysis of theoretical vs empirical bound gaps

This  evaluation framework ensures a rigorous assessment of all project objectives while maintaining the mathematical rigor required for safety-critical applications. The  combination of statistical confidence, mathematical validation, and practical performance metrics provides a complete picture of system effectiveness and readiness for production deployment.

---

**References**: 
- [Research Proposal](../proposal.md) | [Research Methodology](./methodology.md)
- [Common Mathematical Foundation](../../../04_Math_foundations/04.1_RKHS_Mathematical_Foundations.md)
- [Statistical Validation Framework](../../../02_Demo/ac_circuit_discovery/README.md)

[^stat-method]: See Appendix: [Methodology for Statistical Significance and Validation](../../../08_Appendix/08.5_methodology_statistical_significance.md) for definitions, null models, corrections, and caveats.
