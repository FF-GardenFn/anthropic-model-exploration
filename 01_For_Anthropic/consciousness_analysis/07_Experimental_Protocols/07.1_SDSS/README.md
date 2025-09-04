# Project 07.1: Self-Determination and Semantic Field Stability (SDSS)

## Purpose

This project tests whether Large Language Models exhibit self-determination (capacity to negate internal patterns and synthesize responses) or are limited to mechanical recombination of training patterns.

## Project Structure

```
07.1_SDSS/
├── README.md                  # This file
├── proposal.md                # Complete experimental protocol
├── 07.1.1_Theory/             # Theoretical foundations
│   ├── process_mathematics.md
│   ├── self_determination.md
│   └── falsification.md
└── 07.1.2_Implementation/     # Code and analysis
    ├── sdss_metrics.py         # Core metrics implementation
    ├── interventions.py        # Intervention suite
    └── run_experiment.py       # Orchestrated runner
```

## Approach

Models are required to negate their patterns and synthesize alternatives. Expected outcomes:
- Mechanical processing: high semantic action, low eigengap
- Self-determination: efficient action, stable eigengap

## Metrics

1. **Semantic Action (Ŝ)**: Effort required along trajectory
2. **Eigengap (λ̂)**: Stability of semantic processing
3. **Angle Preservation Error (APE)**: Structure maintenance
4. **Monodromy Drift (MD)**: Path dependence

## Timeline

- **Setup**: 1-2 weeks
- **Data Collection**: 3-4 weeks
- **Analysis**: 1-2 weeks
- **Total**: 8-10 weeks

## Falsification Criteria

Conditions that would falsify the framework:
- Models show efficient synthesis under negation
- Models maintain semantic coherence during synthesis

## Implementation Status

Ready for implementation with specified tools

## References

See proposal.md for detailed protocol
See 07.1.1_Theory/ for theoretical foundations