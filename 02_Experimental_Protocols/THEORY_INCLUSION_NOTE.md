# Note on Theory Inclusion in Experimental Directories

## Rationale

Theory documents are included within each experimental directory (07.x.1_Theory/) to provide:

1. **Focused Context**: Reviewers can understand theoretical foundations without navigating the complete 6-volume framework

2. **Experimental Relevance**: Only theory directly relevant to each experiment is included, reducing cognitive load

3. **Optional Reference**: Theory sections are supplementary - experiments can be understood from proposals alone

4. **Self-Contained Modules**: Each experiment becomes a complete package with theory, implementation, and protocol

## Structure

```
07.x_[EXPERIMENT]/
├── proposal.md              # Complete protocol (standalone)
├── 07.x.1_Theory/          # Optional theoretical context
│   └── [relevant_theory]   # Focused subset from main volumes
└── 07.x.2_Implementation/  # Code implementation
```

## For Reviewers

- **Quick Review**: Read only proposal.md
- **Deep Dive**: Explore theory subdirectory for foundations
- **Full Context**: Refer to main volumes (01-06) if needed

This structure balances completeness with accessibility, allowing reviewers to engage at their preferred depth without requiring full framework traversal.