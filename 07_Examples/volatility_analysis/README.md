# Volatility Analysis

AC Circuit Discovery analysis for cybersecurity tool understanding.

## Test Input
`volatility -f memory.dmp --profile=Win10x64_18362`

## Files
- `run2.md`: Execution log with statistical validation
- `volatility_analysis_results.txt`: 210 discovered features across layers
- `*.json`: Raw circuit analysis data

## Key Results

**Top Circuits**:
1. Layer 11, Head 3: 572.71 concentration (7.7σ significance)
2. Layer 11, Head 2: 569.69 concentration (7.6σ significance)
3. Layer 15, Head 4: 535.24 concentration (7.0σ significance)

**Feature Distribution**:
- Layers 10-12: Command parsing, file formats
- Layers 13-15: Technical syntax, documentation
- Layers 16-17: System integration, Microsoft references
> Statistical validation was made on a small smaple size for demonstration purposes. Research will be performed on substantially larger datasets. Results may vary. ( most defnitely)

**Security Insights**:
Model demonstrates understanding of memory forensics, Windows system specifics, and command-line tool syntax.

Analysis integrated into main demo notebook (`09_Demo/`).