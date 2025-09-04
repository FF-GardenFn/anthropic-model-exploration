# Theory Inclusion Toggle (Reviewer Convenience)

To keep experimental documentation clean for quick review, heavy Python snippets have been relocated to lightweight `.py` files in this `documentation/` folder. The Markdown pages now reference these modules by path and symbol name.

- Analysis framework code: documentation/analysis_framework_snippets.py
- Implementation guide code: documentation/implementation_guide_snippets.py
- PVCP/QCGI proposals: see their respective `07.3.2_Implementation/` and `07.2.2_Implementation/` snippet modules.

Rationale: Anthropic reviewers can skim theory and protocols without scrolling through long code blocks, yet still open the exact function in one click/path if needed. This mirrors our broader goal: minimize friction while preserving auditability.
