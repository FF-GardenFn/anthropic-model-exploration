# Research Methodology: Routed Circuit Tomography for Welfare

> **Technical Implementation Framework**: RKHS circuit discovery with mathematical validation protocols.

## Methodology Overview

This methodology implements a three-phase approach to welfare circuit tomography, grounded in the **RKHS-enhanced AC Circuit Discovery framework** with mathematical analysis tools. Each phase builds incrementally toward a production-ready system with formal analytical capabilities.

## Phase 1: RKHS-Enhanced Circuit Discovery (Weeks 1-3)

### Mathematical Foundation Implementation

**Objective**: Establish kernel ridge regression equivalence for AC attention with statistical validation.

#### Step 1.1: Kernel Matrix Construction
**Mathematical Framework**:
$$
K(Q, K) = \exp(Q K^T / \sqrt{d_h})
$$

Alternative (linear kernel for efficiency):

$$
K_{\text{lin}}(Q, K) = Q K^T
$$

**Implementation Protocol**:
1. Extract Q/K matrices from attention layers using established capture framework
2. Compute kernel matrices with numerical stability monitoring (condition number < 10^12)
3. Validate kernel properties: positive semi-definiteness, symmetry for bidirectional case

#### Step 1.2: Hat Matrix Computation
**Mathematical Framework**:
$$
H_{\lambda} = K\,(K + \lambda I)^{-1}
$$

with $$\lambda$$ selected via generalized cross-validation (GCV):

$$
\operatorname{GCV}(\lambda) = \frac{\|y - H_{\lambda} y\|^2}{\big(n - \operatorname{tr}(H_{\lambda})\big)^2}.
$$

**Implementation Protocol**:
1. Implement numerically stable matrix inversion using SVD decomposition
2. Deploy GCV optimization for automatic $$\lambda$$ selection
3. Validate hat matrix properties: idempotency, trace bounds, projection operator verification

#### Step 1.3: Eigenvalue Analysis
**Mathematical Framework**:
$$
K = \sum_i \lambda_i \,\phi_i \phi_i^\top \quad \text{(spectral decomposition)}
$$

We monitor the spectral gap:

$$
\text{gap} = \frac{\lambda_1}{\lambda_2}.
$$

**Implementation Protocol**:
1. Compute eigendecomposition with numerical stability checks
2. Implement spectral gap analysis for stability assessment
3. Compare with AC resonance concentration for qualitative correspondence; thresholds are tuned per task and documented in the Appendix[^stat-method].

### Circuit Candidate Mining

**Objective**: Identify welfare-predictive attention heads using mathematical stability criteria.

#### Mining Strategy
**Mathematical Criteria (screening examples; tuned per task; see Appendix[^stat-method])**:
- Eigenvalue-based z-scores relative to a baseline distribution.
- GCV improvement over baselines.
- Spectral gap sufficient to indicate mode separation.

Thresholds are selected per dataset/model and documented in the Appendix[^stat-method].

**Implementation Steps**:
1. **Welfare Probe Integration**: Deploy Anthropic's welfare assessment prompts
2. **Feature Extraction**: Use SAE (Sparse Autoencoder) analysis for welfare-predictive features
3. **Topological MI Integration**: Apply geometric visualization for candidate ranking
4. **Cross-Validation**: Validate candidates across multiple welfare contexts

#### Validation Protocol
**Statistical Framework**:

Thresholds are calibrated per task using the Appendix methodology[^stat-method]. We compute a head-wise statistic:

$$
Z = \frac{\lambda_i - \mu_{\text{baseline}}}{\sigma_{\text{baseline}}}
$$

and assess practical impact via changes in GCV relative to baselines.

**Expected Output**: 30-60 mathematically validated welfare circuit candidates

## Self-Organizing Circuit Discovery

Our theoretical circuit extraction framework finds practical implementation through self-organizing sparsity, extending Modi et al. (2024)'s self-organizing sparse autoencoders to circuit discovery. This operationalizes our RKHS projections through efficient sparse decomposition, providing both computational advantages and improved mathematical foundations.

### SOSAE Integration for Circuit Extraction

**Mathematical Enhancement**: Replace heuristic CRI ≥ 0.7 thresholds with SOSAE's mathematical ranking system via (1+α)^k penalty formulation:

$$
\mathcal{L}_{\text{SOSAE}} = \mathcal{L}_{\text{reconstruction}} + \lambda \sum_{i} (1+\alpha)^{k_i} |z_i|
$$

where $k_i$ represents the position-based penalty for feature $i$, naturally ordering circuit importance.

**Push Regularization for Circuit Identification**:
The push regularization mechanism naturally identifies important circuits through:
1. **Automatic Sparsity Induction**: High-penalty positions naturally suppress noise while preserving welfare-critical circuits
2. **Hierarchical Discovery**: Position-based penalties create natural circuit importance rankings
3. **Computational Efficiency**: 130x speedup enables comprehensive circuit search across all attention heads simultaneously

**Implementation Strategy**:
1. Deploy SOSAE analysis across all attention layers with position-based penalties
2. Extract circuit candidates where push regularization maintains high activations
3. Rank circuits by their resistance to suppression under increasing penalty values

### Mathematical Framework

**RKHS-SOSAE Connection**: Connect SOSAE's sparse basis to our RKHS circuit projections through the kernel formulation:

$$
H_{\text{SOSAE}} = K_{\text{sparse}}(K_{\text{sparse}} + \lambda I)^{-1}
$$

where $K_{\text{sparse}}$ represents the kernel matrix constructed from SOSAE-identified sparse features.

**Natural Orthogonalization**: Show how orthogonalization emerges naturally from position-based sparsity:

$$
\langle \phi_i, \phi_j \rangle_{\text{SOSAE}} = \delta_{ij} \cdot w_{\text{pos}}(i,j)
$$

with position-weighted orthogonality ensuring circuit separation without explicit orthogonalization steps.

**Kernel Property Preservation**: Prove that SOSAE preserves essential kernel properties:
1. **Positive Semi-definiteness**: Maintained through sparse basis construction
2. **Reproducing Property**: SOSAE features preserve RKHS reproducing kernel structure  
3. **Added Interpretability**: Position-based penalties provide natural circuit ranking without sacrificing mathematical rigor

### Practical Implementation

**SOSAE-Enhanced Circuit Discovery Algorithm**:

```
Algorithm: SOSAE Circuit Discovery
Input: Attention matrices Q, K, welfare probe data
Output: Ranked circuit candidates with mathematical certificates

1. Deploy SOSAE with position-based penalties across all layers
2. Identify sparse features with high resistance to push regularization  
3. Construct sparse kernel matrix K_sparse from identified features
4. Compute RKHS projections H_SOSAE = K_sparse(K_sparse + λI)^(-1)
5. Validate kernel properties and mathematical bounds
6. Rank circuits by eigenvalue magnitude and welfare correlation
7. Return top candidates with mathematical certificates
```

**Expected Improvements**:
- **Superior Circuit Ranking**: Mathematical penalty-based ranking replaces ad-hoc CRI thresholds
- **Automatic Threshold Discovery**: Position-based penalties naturally determine importance cutoffs
- **Enhanced Computational Efficiency**: 130x speedup enables exhaustive search previously computationally infeasible
- **Mathematical Rigor**: Preserves RKHS theoretical guarantees while adding interpretability

**Validation Protocol**:
Compare CRI-based circuit identification versus SOSAE ranking on welfare-relevant circuits:

$$
\text{Validation Metric} = \frac{\text{Welfare Preservation}_{\text{SOSAE}}}{\text{Welfare Preservation}_{\text{CRI}}} \times \frac{\text{Computational Cost}_{\text{CRI}}}{\text{Computational Cost}_{\text{SOSAE}}}
$$

Expected validation results demonstrate both superior circuit identification and computational efficiency through the SOSAE framework.

## Phase 2: Circuit Promotion and Contract Formalization (Weeks 4-8)

### Mathematical Contract Framework

**Objective**: Formalize circuit behavior using RKHS projection operators with mathematical analysis.

#### Contract Specification
**Mathematical Definition**:
Contract C consists of:
- Precondition P(input): RKHS projection bounds on inputs.
- Postcondition Q(output): Welfare preservation conditions.
- Invariant I(intermediate): Eigenvalue stability bounds throughout execution.

Thresholds and tests are defined per task using the Appendix methodology[^stat-method].

**Implementation Framework**:
1. **Precondition Verification**: Input validation using kernel similarity bounds
2. **Postcondition Analysis**: Output welfare preservation via hat matrix projections
3. **Invariant Monitoring**: Real-time eigenvalue tracking for stability maintenance

#### Circuit Verification Protocol
**Mathematical Validation**:
Verification criteria (tuned per task; see Appendix[^stat-method]):

1. Kernel correspondence:
$$
\| H_{\text{circuit}} - H_{\text{theoretical}} \|_F < \varepsilon
$$

2. Eigenvalue stability over time window \([0,T]\):
$$
\big|\lambda_i(t) - \lambda_i(0)\big| < \delta, \quad \forall t \in [0,T],\ \forall i
$$

3. Welfare preservation (task-specific metric):
$$
\| W(\text{output}) - W(\text{baseline}) \|_2 < \eta
$$

**Expected Output**: 10-20 formally verified circuits with mathematical contracts

### Router Architecture Development

**Objective**: Design and train efficient routing system with plan caching and bounded planning.

#### Two-Tier Router Design
**Architecture Components**:

1. **Plan Cache (Tier 1)**:
   - **Target**: ≥90% hit-rate for common welfare queries
   - **Storage**: Pre-computed circuit combinations with welfare analysis
   - **Validation**: Mathematical bounds verification for cached plans

2. **Bounded Planner (Tier 2)**:
   - **Novel Query Handling**: Dynamic circuit composition with complexity bounds
   - **Planning Bounds**: O(log n) circuit combination complexity
   - **Fallback Strategy**: Full model evaluation if no valid circuit combination found

#### Router Training Protocol
**Mathematical Optimization**:

We optimize a composite objective:

$$
\min_{\theta}\ \mathcal{L}_{\text{routing}}(\theta) + \lambda_{\text{eff}}\, a(\theta) + \lambda_{\text{welfare}}\, \mathcal{L}_{\text{welfare}}(\theta)
$$

subject to task‑tuned constraints:

$$
a(\theta) \le \alpha, \qquad P_{\text{welfare}}(\theta) \ge \beta,
$$

with calibration described in the Appendix[^stat-method].

**Training Strategy**:
1. **Supervised Learning**: Train on welfare query-circuit mapping pairs
2. **Reinforcement Learning**: Optimize for efficiency-welfare trade-offs
3. **Mathematical Validation**: Ensure routing decisions respect RKHS bounds

**Expected Output**: Production-ready router with mathematical performance analysis

## Phase 3: End-to-End Integration and Validation (Weeks 9-12)

### Production Integration Framework

**Objective**: Deploy complete system with comprehensive evaluation and integration with Anthropic infrastructure.

#### System Architecture
**Component Integration**:
```
System Pipeline:
Input → Circuit Discovery Engine → Contract Verification → Router → Output
       ↓                      ↓                    ↓           ↓
   RKHS Analysis → Mathematical Bounds → Plan Cache → Traces
```

**Implementation Steps**:
1. **API Design**: RESTful endpoints for circuit discovery and routing
2. **Monitoring Integration**: Real-time eigenvalue and welfare metrics
3. **Trace Generation**: Human-readable circuit execution logs with mathematical certificates

#### Comprehensive Evaluation Protocol

**Performance Metrics** (task‑tuned; see Appendix[^stat-method]):

Primary considerations:
- Welfare preservation relative to baseline on held‑out probes.
- Efficiency (activation fraction) under routing.
- Mathematical validity via head‑wise diagnostics and null‑model comparisons.

Secondary considerations:
- Latency impact on routed tasks.
- Interpretability of circuit traces (human evaluation).
- Stability over time (eigenvalue drift, resonance measures).

**Evaluation Framework**:
1. **Welfare Probe Suite**: Comprehensive testing on Anthropic's welfare assessment battery
2. **Ablation Studies**: Component-wise performance analysis
3. **Stress Testing**: Performance under adversarial and edge-case scenarios
4. **Long-term Stability**: Eigenvalue monitoring over extended operation periods

#### Mathematical Validation Suite

**Theoretical Verification**:

1. RKHS correspondence (hat matrix formulation):
$$
H = K\,(K + \lambda I)^{-1}
$$
Empirically bound deviations between implemented and theoretical forms.

2. Representer Theorem: Verify learned solutions lie in the span of training data features.

3. Hat matrix properties: Confirm projection‑like behavior (trace bounds, symmetry under suitable kernels).

4. GCV optimization: Validate automatic parameter selection via
$$
\operatorname{GCV}(\lambda) = \frac{\|y - H y\|^2}{\big(n - \operatorname{tr}(H)\big)^2}.
$$

**Expected Output**: Production-ready system with comprehensive mathematical validation

## Quality Assurance and Validation

### Mathematical Rigor Protocols

**Numerical Stability**:
- Matrix condition number monitoring (< 10^12)
- Eigenvalue computation accuracy validation (relative error < 10^-6)
- Floating point precision analysis for production deployment

**Statistical Validation**:
- Cross-validation with multiple random seeds
- Bootstrap confidence intervals for performance metrics
- Multiple comparison correction for statistical tests

**Theoretical Consistency**:
- Mathematical property verification through symbolic computation
- Theoretical bound validation against empirical observations
- Consistency checks across different RKHS kernel choices

### Reproducibility Framework

**Code Organization**:
- Modular architecture with clear mathematical component separation
- Comprehensive unit tests for each mathematical operation
- Integration tests for end-to-end pipeline validation

**Documentation Standards**:
- Mathematical derivations with step-by-step proofs
- Performance benchmark results with statistical confidence intervals


## Risk Mitigation Strategies

### Technical Risks

**Computational Complexity**:
- **Risk**: Eigendecomposition may not scale to large models
- **Mitigation**: Implement approximate algorithms (Lanczos, randomized SVD) with accuracy bounds

**Numerical Stability**:
- **Risk**: Matrix operations may become ill-conditioned
- **Mitigation**: SVD-based implementations with condition number monitoring

**Circuit Generalization**:
- **Risk**: Circuits may not transfer across different welfare contexts
- **Mitigation**: Multi-domain validation with mathematical stability analysis

### Methodological Risks

**Statistical Validity**:
- **Risk**: Multiple testing may inflate false discovery rates
- **Mitigation**: Bonferroni correction and false discovery rate control

**Evaluation Bias**:
- **Risk**: Performance metrics may not capture true welfare preservation
- **Mitigation**: Independent evaluation using held-out welfare assessment suites

**Integration Complexity**:
- **Risk**: System integration may introduce unexpected behaviors
- **Mitigation**: Comprehensive integration testing with mathematical bound verification

## Expected Deliverables

### Research Outputs
1. **Mathematical Framework Documentation**: Complete theoretical foundation with proofs
2. **Implementation Library**: Production-ready circuit discovery and routing tools
3. **Validation Results**:  experimental evaluation with statistical analysis


### Technical Artifacts
1. **RKHS Circuit Discovery Engine**: Core mathematical implementation
2. **Router System**: Efficient circuit selection with performance guarantees

This methodology provides a systematic approach to developing mathematically rigorous welfare circuit tomography, with clear milestones, validation protocols, and risk mitigation strategies ensuring successful project completion within the 90-day timeline.

---

**References**: 
- [Research Proposal](../proposal.md) | [Evaluation Metrics](./evaluation_metrics.md)
- [Common Mathematical Foundation](../../../04_Math_foundations/04.1_RKHS_Mathematical_Foundations.md)
- [Demo Validation Reference](../../../09_Demo/main_demo.py)


[^stat-method]: See Appendix: [Methodology for Statistical Significance and Validation](../../../08_Appendix/08.2_methodology_statistical_significance.md) for definitions, null models, calibration, and limitations.