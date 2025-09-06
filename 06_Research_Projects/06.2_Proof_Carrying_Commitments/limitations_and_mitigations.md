# Limitations and Mitigations: Proof-Carrying Commitments

## Purpose
This document consolidates technical limitations, methodological risks, and ethical considerations for Project 06.2 (Proof‑Carrying Commitments). It centralizes caveats to avoid duplication across the proposal and README.

## Technical Limitations

### Mathematical Complexity
- Limitation: Full formal proofs of safety properties can be intractable for production systems.
- Mitigations:
  - Employ sound approximations with provable equivalence in relevant regimes.
  - Use layered certificates: lightweight runtime checks + periodic deep verification.

### Computational Overhead
- Limitation: Spectral monitoring and kernel computations may incur latency.
- Mitigations:
  - Approximate eigensolvers (Lanczos, randomized SVD) with error bounds.
  - Budgeted verification and adaptive sampling; cache invariants.
  - Offline calibration of thresholds to minimize online cost.

### Adversarial Sophistication
- Limitation: Adaptive adversaries may exploit blind spots in constraints.
- Mitigations:
  - Red‑teaming with adversarial search and formal coverage analysis.
  - Defense‑in‑depth: combine spectral, behavioral, and cryptographic attestations.
  - Rotate constraints and monitor for adaptive drift.

### Integration Complexity
- Limitation: Safety constraints may conflict with existing behaviors or infra.
- Mitigations:
  - Gradual rollout with compatibility tests and automatic rollback.
  - Clear API boundaries; sandboxing of constraint failures.
  - Robust observability: logs, metrics, and audits for certificate outcomes.

## Methodological Risks

### Measurement Validity
- Risk: Spectral signals may conflate safety with unrelated features.
- Mitigations:
  - Pre‑registration, negative controls, cross‑domain checks.
  - Multiple‑comparison control and bootstrap confidence intervals.

### External Validity
- Risk: Results may not generalize across models or deployment contexts.
- Mitigations:
  - Multi‑model, multi‑domain evaluations with replication.
  - Report scope conditions and failure cases explicitly.

## Ethical and Governance Considerations

### Over‑reliance on Certificates
- Consideration: Certificates can induce false confidence.
- Mitigations:
  - Treat certificates as evidence under assumptions; combine with human review.
  - Track residual risk and document assumption violations.

### Transparency and Accountability
- Consideration: Stakeholders require clear, auditable evidence.
- Mitigations:
  - Publish certificate schemas, logs, and evaluation protocols where possible.
  - Maintain reproducible artifacts and change histories.

## References
- See setup.md for environment and timeline.
- See 06.2.1_Theory/ for methodology and evaluation metrics.
