 Analyzing text: 'volatility -f memory.dmp --profile=Win10x64_18362'
============================================================
AC Circuit Discovery Pipeline Initiated
Scanning 64 heads across layers [10, 11, 12, 13, 14, 15, 16, 17]...
============================================================
 scanning 64 heads across 8 layers...
  Analysing Layer 10 for 8 heads...
  Analysing Layer 11 for 8 heads...
  Analysing Layer 12 for 8 heads...
  Analysing Layer 13 for 8 heads...
  Analysing Layer 14 for 8 heads...
  Analysing Layer 15 for 8 heads...
  Analysing Layer 16 for 8 heads...
  Analysing Layer 17 for 8 heads...

============================================================
Computing Statistical Significance...
============================================================
Baseline: μ=123.95, σ=58.60 (n=50)

============================================================
Top 5 Most Concentrated Heads Found:
------------------------------------------------------------
 1. Layer 11, Head  3 | Concentration: 572.7135 | Push Entropy: 3.40 bits | 7.7σ ***
 2. Layer 11, Head  2 | Concentration: 569.6925 | Push Entropy: 5.45 bits | 7.6σ ***
 3. Layer 15, Head  4 | Concentration: 535.2423 | Push Entropy: 20.56 bits | 7.0σ ***
 4. Layer 14, Head  6 | Concentration: 531.5412 | Push Entropy: 29.91 bits | 7.0σ ***
 5. Layer 15, Head  5 | Concentration: 526.2373 | Push Entropy: 18.70 bits | 6.9σ ***
============================================================

============================================================
Detailed Analysis for Circuit: Layer 11, Head 3
Resonance Concentration: 572.7135
Push Entropy: 3.40 bits
------------------------------------------------------------
 Detailed Metrics:
{
  "push_entropy_bits": 3.404336768709908,
  "resonance_concentration": 572.7134845653095
}
