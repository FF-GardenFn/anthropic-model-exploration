# Project 07.1: Self-Determination and Semantic Field Stability (SDSS)

## Overview

Tests whether Large Language Models can exhibit genuine self-determination—the capacity to negate internal patterns and synthesize novel responses—or whether they are limited to mechanical recombination of training patterns.

## Project Structure

```
07.1_SDSS/
├── README.md                 # This file
├── proposal.md              # Complete experimental protocol
├── 07.1.1_Theory/          # Theoretical foundations
│   ├── process_mathematics.md
│   ├── self_determination.md
│   └── falsification.md
└── 07.1.2_Implementation/  # Code and analysis
    ├── sdss_metrics.py     # Core metrics implementation
    ├── analysis.ipynb      # Analysis notebooks
    └── results/            # Experimental results
```

## Key Innovation

Forces models to negate their own patterns and synthesize alternatives, testing whether they show:
- Mechanical breakdown (high semantic action, low eigengap)
- Creative synthesis (efficient action, stable eigengap)

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

## Falsification Risk

**High** - Clear predictions that could prove framework wrong:
- If models show efficient synthesis under negation → Framework falsified
- If models maintain semantic coherence → Framework falsified

## Status

🔄 Ready for implementation with Anthropic's tools

## Contact

[Principal Investigator]
[consciousness_analysis team]