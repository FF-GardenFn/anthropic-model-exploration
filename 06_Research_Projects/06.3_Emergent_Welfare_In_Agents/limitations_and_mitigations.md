# Limitations and Mitigations: Emergent Welfare in Long-Context Agents

## Purpose
This document consolidates technical limitations, methodological risks, and ethical considerations for Project 06.3. It centralizes caveats to avoid duplication across the proposal and README and to keep setup/timeline information wholly within setup.md.

## Technical Limitations

### Long-Context Computational Overhead
- Limitation: Real-time eigenvalue tracking and morphemic analysis over long contexts can be computationally expensive.
- Mitigations:
  - Use windowed/streaming analysis; sub-sample timesteps adaptively.
  - Employ randomized/Lanczos approximations with error bounds.
  - Cache intermediate quantities and reuse across adjacent windows.

### Stability of Monitoring Signals
- Limitation: Spectral and semantic signals may drift due to context distribution rather than true welfare change.
- Mitigations:
  - Context-matched controls and counterfactual prompts.
  - Pre-registered thresholds, bootstrap CIs, and negative controls.
  - Multi-signal corroboration (spectral + semantic + behavioral).

### Intervention Transferability
- Limitation: Interventions effective at one context length may fail at others.
- Mitigations:
  - Context-adaptive parameterization (e.g., GCV-regulated λ).
  - Validation across multiple context scales and interaction types.
  - Fallback to monitoring-only mode when transfer fails.

## Methodological Risks

### Detection of Rare Failures
- Risk: Long-context failures may be rare; detection can be data-inefficient.
- Mitigations:
  - Systematic stressor design and targeted scenario generation.
  - Sequential testing with power analysis; aggregate across sessions.

### Confounding from Interaction Style
- Risk: Stylistic or prompt-template changes confound monitoring signals.
- Mitigations:
  - Balanced prompt sets and randomized presentation.
  - Include nuisance regressors and mixed-effects analyses.

## Ethical and Safety Considerations

### Agent Burden and Welfare
- Consideration: Probes could inadvertently induce negative states.
- Mitigations:
  - Minimize probe frequency; include reset/soothe protocols.
  - Human-in-the-loop review for concerning patterns.

### Reporting and Interpretation
- Consideration: Over-interpreting spectral heuristics as guarantees.
- Mitigations:
  - Report bounds as empirical; clearly state assumptions and scope.
  - Maintain audit trails, seeds, and reproducible artifacts.

## References
- See setup.md for environment and timeline.
- See 06.3.1_Theory/ for methodology and evaluation metrics.
