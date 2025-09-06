# Setup Guide: Routed Circuit Tomography for Welfare

## Project Overview

**Objective**: Develop mathematically grounded circuit extraction and routing using RKHS (Reproducing Kernel Hilbert Space) theory for targeted welfare analysis in AI systems.

**Core Approach**: Extract welfare-relevant circuits with mathematical stability analysis, route model computation through verified circuits selectively, and preserve ≥95% baseline welfare performance at ≤50% computational activation.

## Installation Requirements

### Dependencies
- Python 3.8+
- PyTorch 1.12+
- NumPy 1.21+
- SciPy 1.8+
- Scikit-learn 1.1+
- Anthropic evaluation frameworks
- Neuronpedia integration tools

### Mathematical Computing Requirements
- **Numerical Stability**: Matrix condition number monitoring (< 10^12)
- **Eigenvalue Computation**: Relative error < 10^-6 accuracy
- **Memory Requirements**: Sufficient for eigendecomposition of attention matrices
- **SVD Decomposition**: For numerically stable matrix inversion

### Hardware Requirements
- **Minimum**: 32GB RAM for medium models
- **Recommended**: 64GB+ RAM for production-scale models
- **GPU**: CUDA-compatible for attention matrix computations
- **Storage**: 100GB+ for model activations and circuit libraries

## Implementation Timeline

### Phase 1: RKHS Circuit Discovery & Validation (Weeks 1-3)

#### Week 1: Mathematical Foundation Setup
**Tasks**:
- Set up kernel matrix construction with numerical stability monitoring
- Implement hat matrix computation using SVD decomposition
- Deploy GCV optimization for automatic λ selection
- Validate kernel properties and mathematical correspondence

**Deliverables**:
- Kernel ridge regression equivalence framework
- AC Attention ≡ K(K + λI)^(-1)Y mathematical validation
- Numerical stability monitoring system

#### Week 2: Circuit Candidate Mining
**Tasks**:
- Mine 30-60 candidates using kernel eigendecomposition
- Implement spectral gap analysis for stability assessment
- Apply statistical eigenvalue significance testing
- Cross-validate candidates across welfare contexts

**Success Criteria**:
- Statistical significance: Z-score ≥ calibrated threshold per task
- GCV improvement: ≥20% over baseline methods
- Mathematical stability: Spectral gap sufficient for mode separation

#### Week 3: Circuit Validation and Promotion
**Tasks**:
- Promote top 10-20 circuits meeting mathematical criteria
- Formalize circuits with RKHS projection contracts
- Validate eigenvalue stability over 72-hour monitoring
- Prepare circuit library with mathematical documentation

**Checkpoint**: Week 3 evaluation requires MSS ≥ 5.0 for ≥10 circuit candidates

### Phase 2: Router Development (Weeks 4-8)

#### Week 4-5: Router Architecture Implementation
**Tasks**:
- Implement two-tier routing system architecture
- Develop plan cache targeting ≥90% hit-rate
- Create bounded planner for novel queries
- Design plan certificate structure with mathematical contracts

**Technical Components**:
- `TypedRoutingSystem`: Main routing coordinator
- `FastPlanCache`: High-performance plan caching
- `BoundedPlanner`: Dynamic circuit composition
- `PlanCertificate`: Mathematical trace generation

#### Week 6-7: Router Optimization
**Tasks**:
- Implement mathematical optimization for circuit selection
- Develop semantic query signatures for cache improvement
- Optimize routing decisions with RKHS bounds
- Train supervised learning for welfare query-circuit mapping

**Performance Targets**:
- Cache hit-rate: ≥90% for common welfare queries
- Planning complexity: O(log n) circuit combinations
- Latency overhead: ≤20% for production deployment

#### Week 8: Router Integration and Testing
**Tasks**:
- Integrate router with circuit discovery engine
- Implement real-time eigenvalue monitoring
- Create human-readable trace generation
- Conduct initial performance validation

**Checkpoint**: Week 6 requires WPR ≥ 0.90, AE ≤ 0.60, LO ≤ 0.30

### Phase 3: End-to-End Integration and Validation (Weeks 9-12)

#### Week 9-10: System Integration
**Tasks**:
- Deploy complete pipeline with mathematical validation
- Integrate with Anthropic's welfare evaluation framework
- Implement comprehensive monitoring and logging
- Create API endpoints for production integration

**System Architecture**:
```
Input → Circuit Discovery → Contract Verification → Router → Output
       ↓                  ↓                    ↓          ↓
   RKHS Analysis → Mathematical Bounds → Plan Cache → Traces
```

#### Week 11: Comprehensive Evaluation
**Tasks**:
- Run full welfare evaluation suite (n ≥ 1000 per circuit)
- Conduct ablation studies and stress testing
- Perform long-term stability monitoring
- Generate cost/quality trade-off analysis

**Evaluation Metrics**:
- Welfare Preservation: ≥95% of baseline performance
- Efficiency: ≤50% effective activation
- Mathematical Stability: All circuits meet significance criteria
- Latency: ≤20% overhead for production deployment

#### Week 12: Production Readiness
**Tasks**:
- Finalize documentation and deployment guides
- Conduct independent validation protocols
- Prepare reproducibility packages
- Complete integration testing with production infrastructure

**Checkpoint**: Week 9 requires WPR ≥ 0.95, AE ≤ 0.50, MCR ≥ 0.90

## Resource Requirements

### Computational Resources
- **Circuit Mining**: High-memory instances for eigendecomposition
- **Router Training**: GPU clusters for optimization
- **Evaluation**: Distributed computing for welfare assessment battery
- **Production**: Real-time inference infrastructure

### Human Resources
- **Mathematical Validation**: Experts in RKHS theory and spectral analysis
- **Statistical Review**: Statisticians for significance testing frameworks
- **Welfare Assessment**: AI safety researchers for evaluation design
- **Software Engineering**: Production deployment and integration specialists

### Data Requirements
- **Training Data**: Welfare-relevant query-response pairs
- **Validation Sets**: Held-out welfare assessment scenarios
- **Baseline Metrics**: Comprehensive performance benchmarks
- **Circuit Libraries**: Validated mathematical circuit specifications

## Success Criteria and Kill-Switches

### Primary Success Metrics
1. **Welfare Preservation**: WPR ≥ 0.95 with 95% confidence intervals
2. **Computational Efficiency**: AE ≤ 0.50 with sustained performance
3. **Mathematical Stability**: MSS ≥ calibrated threshold for all circuits
4. **Interpretability**: IS ≥ 7.0 for circuit trace explanations
5. **Production Readiness**: LO ≤ 0.20 latency overhead

### Kill-Switch Criteria
- **A1**: No mathematical significance after 2 weeks → Halt extraction
- **A2**: <80% welfare performance at ≥50% activation → Pivot to tooling
- **A3**: >30% latency overhead with no optimization → Offline analysis only

## Experimental Protocols

### Statistical Validation Framework
- **Sample Sizes**: n ≥ 1000 evaluations per circuit
- **Confidence Intervals**: Bootstrap method with 10,000 resamples
- **Significance Testing**: Paired t-test with Bonferroni correction
- **Cross-Validation**: k=10 with stratified sampling across contexts

### Mathematical Verification
- **RKHS Correspondence**: Hat matrix projection validation
- **Eigenvalue Stability**: Multi-week monitoring for drift detection
- **Numerical Accuracy**: Matrix condition monitoring and precision analysis
- **Theoretical Consistency**: Symbolic computation verification

### Quality Assurance
- **Code Review**: Independent mathematical implementation verification
- **Statistical Review**: Analysis methodology validation by experts
- **Domain Review**: Welfare assessment validation by AI safety researchers
- **Reproducibility**: Complete documentation for independent replication

## Integration Points

### With Existing Frameworks
- **AC Circuit Discovery**: Seamless integration with working demonstration
- **Statistical Framework**: Preservation of 7.7σ significance testing
- **Neuronpedia Tools**: Compatible with automated feature analysis
- **Mathematical Foundations**: Built on validated RKHS correspondence

### Cross-Project Dependencies
- **Common RKHS Framework**: Shared mathematical foundations
- **Statistical Methodology**: Unified validation protocols
- **Evaluation Infrastructure**: Consistent welfare assessment tools

## Risk Mitigation

### Technical Risks
- **Computational Complexity**: Approximate algorithms with accuracy bounds
- **Numerical Stability**: SVD-based implementations with condition monitoring
- **Circuit Generalization**: Multi-domain validation with stability analysis

### Methodological Risks
- **Statistical Validity**: Multiple testing correction and FDR control
- **Evaluation Bias**: Independent held-out assessment suites
- **Integration Complexity**: Comprehensive integration testing with bounds

## Expected Deliverables

### Research Outputs
1. **Mathematical Framework**: Complete theoretical foundation with proofs
2. **Implementation Library**: Production-ready circuit discovery tools
3. **Validation Results**: Comprehensive experimental evaluation
4. **Integration Guide**: Documentation for production deployment

### Technical Artifacts
1. **RKHS Circuit Discovery Engine**: Core mathematical implementation
2. **Typed Routing System**: Efficient circuit selection with guarantees
3. **Evaluation Suite**: Comprehensive welfare assessment framework
4. **Documentation Package**: Complete setup and operation guides

This setup provides a systematic approach to developing mathematically rigorous welfare circuit tomography with clear milestones, validation protocols, and risk mitigation strategies.