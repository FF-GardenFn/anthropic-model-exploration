# Limitations and Mitigations: Routed Circuit Tomography

## Purpose
This document articulates key limitations, risks, and mitigation strategies for Project 06.1 (Routed Circuit Tomography). It centralizes cross-cutting caveats and ethical considerations to avoid duplication in proposal and README materials.

## Technical Limitations

### Computational Complexity
- Limitation: Exact eigendecomposition and kernel operations may be prohibitive at scale.
- Mitigations:
  - Use approximate methods (Lanczos, randomized SVD) with error bounds.
  - Implement streaming/block SVD for memory constraints.
  - Monitor conditioning; switch to stabilized solvers when κ(K) exceeds threshold.

### Generalization of Circuits
- Limitation: Candidate circuits may be context‑specific and fail to generalize across welfare domains.
- Mitigations:
  - Cross‑domain validation and hold‑out tasks.
  - Require statistical eigenvalue significance and minimum eigengap across contexts.
  - Promote only circuits meeting predefined stability criteria.

### Router Training and Integration
- Limitation: Circuit router may introduce latency or brittle behaviors under distribution shift.
- Mitigations:
  - Employ bounded planning with RKHS contracts and fallback to baseline.
  - Cache planning results with semantic signatures; enforce latency budgets.
  - Canary analysis and progressive rollout with automated regression tests.

### Numerical Stability
- Limitation: Ill‑conditioned kernel matrices lead to unstable projections.
- Mitigations:
  - SVD‑based inverses; Tikhonov regularization (λ tuning via GCV).
  - Condition monitoring; raise alerts and halt routing when bounds are violated.
  - Use mixed‑precision with loss‑scaling only under verified stability.

## Methodological Risks

### Measurement Validity
- Risk: Metrics may reflect spurious structure or dataset artifacts.
- Mitigations:
  - Pre‑registration and negative controls.
  - Bootstrap CIs and multiple‑comparison corrections.
  - Replication across seeds, prompts, and model variants.

### Interpretability Overreach
- Risk: Over‑attributing semantics to circuits due to polysemantic features.
- Mitigations:
  - Require convergent evidence from multiple attribution methods.
  - Document counter‑examples and failure modes.
  - Clearly separate empirical observation from theoretical interpretation.

## Ethical and Safety Considerations

### Welfare Impact
- Consideration: Circuit interventions may inadvertently increase harm or degrade safety.
- Mitigations:
  - Define kill‑switch thresholds and rollback criteria.
  - Human‑in‑the‑loop validation on welfare‑relevant benchmarks.
  - Continuous monitoring with post‑deployment audits.

### Data and Reporting
- Consideration: Overstating claims can mislead deployment decisions.
- Mitigations:
  - Report with calibrated uncertainty and explicit scope conditions.
  - Maintain reproducible artifacts and transparent logs.

## References
- See setup.md for environment and timeline.
- See 06.1.1_Theory/ for methodology and metrics.
