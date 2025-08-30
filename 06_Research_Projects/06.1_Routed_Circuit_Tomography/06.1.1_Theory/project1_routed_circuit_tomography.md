# Project 1: Routed Circuit Tomography for Welfare

**Objective**: To decompile welfare-relevant internal model states into a library of callable, contract-verified circuits, and then train a router to selectively engage them, preserving essential welfare behaviors at a fraction of the computational cost and with human-readable traces.

## Brief Summary
This project explores an efficiency-interpretability approach for model welfare by investigating circuit extraction and selective routing, targeting approximately ≤50% effective activation while aiming to preserve ≥95% of baseline welfare performance[^stat-method].

## Technical Foundation

### RKHS AC Circuit Discovery
This project leverages the mathematical connection between AC Attention and kernel ridge regression via RKHS theory. The [AC Circuit Discovery framework](./demo/ac_circuit_discovery/) builds on rigorous mathematical foundations[^stat-method]:

**Mathematical Correspondence**: AC Attention corresponds to K(K + λI)^(-1)Y where K is the attention kernel[^stat-method]
**Empirical Observations**: Layer 11, Head 3 eigenvalue λ₁ = 572.71 observed in preliminary analysis[^stat-method]
**Theoretical Foundation**: Representer theorem provides circuit solutions lie in span of training data

### Method Overview

**1. Kernel-Based Candidate Mining**
- Use **spectral decomposition** K = Σλᵢφᵢφᵢᵀ to identify welfare-predictive eigenmodes
- Apply **GCV optimization** for automatic regularization parameter selection
- Rank candidates by **eigenvalue magnitude** and **spectral gap** for mathematical stability

**2. RKHS Circuit Promotion**
- Formalize circuits using **hat matrix projectors** H = K(K + λI)^(-1) with mathematical contracts
- Replace heuristic CRI ≥ 0.7 with **statistical eigenvalue significance** and **≥20% GCV improvement**[^stat-method]

**3. Typed Routing**
- Train two-tier router with fast plan cache (≥90% hit-rate target)
- Bounded planner for novel queries
- Emit "plan certificates" detailing circuits used and contracts invoked

## Success Metrics & Kill-Switches

Note: Authoritative definitions, thresholds, and kill-switches are centralized in [Evaluation Metrics](./evaluation_metrics.md). The following points summarize the intent; refer to the metrics doc for evaluation details.

### Quality Targets
- **Performance Target**: Aim to preserve ≥95% of baseline model's welfare probe performance[^stat-method]
- **Efficiency Target**: Aim to achieve ≤50% effective activation, with roadmap to ≤30%[^stat-method]

### Mathematical Kill-Switch Criteria
If no circuit candidate achieves **statistical eigenvalue significance** AND **≥20% GCV improvement** after two weeks of RKHS-based mining, halt extraction and pivot to spectral analysis tools only[^stat-method].

## 90-Day Implementation Plan

### Weeks 1-3: RKHS Circuit Discovery & Validation
- Mine 30-60 candidates using **kernel eigendecomposition** and **spectral analysis**
- Promote top 10-20 circuits meeting **statistical significance** and **mathematical stability criteria**[^stat-method]
- **Deliverable**: RKHS-validated circuit library with **hat matrix projection contracts**

### Weeks 4-8: Router Development
- Train router system and achieve hit-rate targets
- Ship initial trace UI for interpretability
- **Deliverable**: Working router with plan certificates

### Weeks 9-12: End-to-End Evaluation
- Run full welfare evaluation suite
- Deliver final cost/quality trade-off curves
- **Deliverable**: Production-ready system with performance metrics

## Integration with AC Research

This project directly utilizes:
- **[Resonance Concentration metrics](./demo/ac_circuit_discovery/README.md)** for circuit stability assessment
- **[Statistical validation framework](./examples/)** ensuring high-confidence circuit selection
- **[Neuronpedia integration tools](./tools/neuronpedia_integration/)** for automated feature analysis

The preliminary statistical analysis in circuit discovery provides support for further investigation of safety-critical circuit promotion[^stat-method].

## Research Foundation

See [research_hub/TAB4](./research_hub/TAB4/) for detailed technical specifications and [research_hub/TAB3](./research_hub/TAB3/) for AC Attention validation results showing +12.3% improvement in causal reasoning and -78% reduction in total loss.

---

*This project represents a concrete application of the AC Circuit Discovery research to Anthropic's model welfare objectives, with clear success metrics and fallback strategies.*

[^stat-method]: Complete statistical methodology and validation protocols: [../../../08_Appendix/08.5_methodology_statistical_significance.md](../../../08_Appendix/08.5_methodology_statistical_significance.md)