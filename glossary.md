# Terminology Glossary

This glossary serves as a reference for the technical terminology developed throughout this research framework. It bridges philosophical insights, mathematical formalism, and computational implementation, providing precise definitions while maintaining connections across different levels of abstraction.

Terms are organized alphabetically with references to their primary introduction. For mathematical notation conventions, see the guide at the end of this document.

---

**AC Attention**: Alternative to traditional softmax attention mechanisms based on agreement/contention dynamics, explored as a potential bridge to continuous computation. *First introduced in: Section 03.2*

**Analytic Continuation**: Mathematical process of extending the domain of a holomorphic function, used as a model for compositional semantics where morpheme combination follows complex-analytic structure. *First introduced in: Section 03.4*

**Bidirectional Resonance**: Mutual attention pattern R = (QK^T) ⊙ (KQ^T) capturing agreement between forward and backward attention flows, used for identifying computationally significant regions. *First introduced in: Section 05.3*

**Brachistochrone of Thought**: Demonstration that attention follows least-time/least-action paths through semantic space, validating PLSA through optimal path analysis. *First introduced in: Section 09 Demo*

**Branch Points**: Mathematical singularities in complex functions representing multi-valued behavior, proposed as mathematical analogs of root morphemes in semantic fields. *First introduced in: Section 03.4*

**Categorical Incommensurability**: The philosophical problem that different conceptual frameworks may not be directly comparable or translatable, relevant to understanding alignment challenges across different AI paradigms. *First introduced in: Section 03.1*

**Cauchy-Riemann Residuals**: Measures of deviation from holomorphic (complex-differentiable) behavior in semantic fields, used as diagnostic tools for morphemic analysis. *First introduced in: Section 04.3*

**Cognitive Opacity**: The fundamental limitation that humans cannot introspect or articulate the computational processes underlying their own cognition, leading to similar opacity in machine learning systems. *First introduced in: Section 03.1*

**Computational Kinetic Energy**: Component of the semantic Lagrangian representing the "cost" of rapid changes in semantic state, measured through AC disagreement velocity or RKHS resonance drift. *First introduced in: Section 03.3*

**DoF (Degrees of Freedom)**: Statistical measure of model complexity in RKHS framework, computed as the trace of hat matrices, used for regularization parameter selection. *First introduced in: Section 04.1*

**Eigengap Analysis**: Diagnostic technique examining the spacing between eigenvalues in kernel matrices to assess stability and dimensionality of attention patterns. *First introduced in: Section 04.1*

**GCV (Generalized Cross Validation)**: Automatic method for selecting regularization parameters in RKHS attention analysis, optimizing prediction accuracy while controlling model complexity. *First introduced in: Section 04.1*

**Hat Matrices**: Projection operators H_{qk} = K_{qk}(K_{kk} + λI)^{-1} that map values to predicted outputs in kernel ridge regression interpretation of attention. *First introduced in: Section 04.1*

**Holomorphic Fields**: Complex-analytic functions ψ(z) defined on the complex plane, proposed as continuous representations of semantic structure with morphemes corresponding to mathematical singularities. *First introduced in: Section 03.4*

**Influence Operators**: Mathematical operators quantifying how much each input affects each output in attention mechanisms, formalized through hat matrices in RKHS framework. *First introduced in: Section 04.1*

**Morphemic Poles**: Mathematical singularities (poles) in holomorphic fields representing bound morphemes (affixes), with residues quantifying specific semantic transformations. *First introduced in: Section 03.4*

**Morphemic Singularities**: Mathematical points where holomorphic semantic fields exhibit non-analytic behavior, proposed as continuous analogs of discrete linguistic morphemes. *First introduced in: Section 03.4*

**OLS-Attention Equivalence**: Mathematical proof by Goulet Coulombe (2025) showing attention mechanisms are equivalent to Ordinary Least Squares regression through orthonormal embeddings F_test F_train^T. *First introduced in: Section 08.1*

**PLSA (Principle of Least Semantic Action)**: Unifying framework mapping classical mechanics constructs to semantic computation, proposing that coherent reasoning follows paths minimizing semantic action S = Σ L_sem. *First introduced in: Section 03.3*

**Property-Theoretic Framework**: Philosophical foundation treating linguistic morphemes as universal properties capable of multiple instantiation, with holomorphic fields representing distributions of property instantiation patterns. *First introduced in: Section 03.4*

**Query-Key-Value (QKV)**: Fundamental attention components where Q (queries) seek information, K (keys) provide addressable content, and V (values) contain retrievable information. *First introduced in: Section 04.1*

**Regularization Parameter (λ)**: Controls bias-variance tradeoff in RKHS framework, determining smoothness of attention patterns through eigenvalue shrinkage factor λ/(λ + μ). *First introduced in: Section 04.1*

**Resonance Concentration**: Statistical measure of attention pattern sparsity, computed through Herfindahl-Hirschman Index or entropy metrics on bidirectional resonance maps. *First introduced in: Section 08.5*

**RKHS (Reproducing Kernel Hilbert Space)**: Mathematical framework for analyzing attention mechanisms through kernel ridge regression, providing statistical foundation for influence analysis and regularization. *First introduced in: Section 04.1*

**SAE (Sparse Autoencoder)**: Feature extraction method used in Anthropic's interpretability work, decomposing neural activations into sparse, interpretable features. *Referenced in: Section 08 Appendix, 01.3*

**Semantic Action**: Quantity S = Σ_t L_sem(x_t, ẋ_t) representing total computational "cost" along a trajectory through semantic space, minimized according to PLSA. *First introduced in: Section 03.3*

**Semantic Lagrangian**: Function L_sem = T_comp - V_sem balancing computational kinetic energy against semantic potential energy, analogous to classical mechanics Lagrangians. *First introduced in: Section 03.3*

**Semantic Potential Energy**: Component of the semantic Lagrangian representing the "instability" or semantic incoherence of a particular semantic state, measured through morphemic residuals or WordNet distances. *First introduced in: Section 03.3*

**Spectral Diagnostics**: Analysis techniques using eigenvalue decomposition of kernel matrices to understand dimensional structure and stability properties of attention patterns. *First introduced in: Section 04.1*

**Superposition Bottleneck**: Architectural constraint where multiple concepts share the same neural units, creating fundamental tensions with interpretability and alignment requirements. *First introduced in: Section 03.1*

**Symmetric Resonance Operator**: Matrix S = H_{qk}H_{kq} representing bidirectional information flow in attention mechanisms, used for spectral analysis of computational modes. *First introduced in: Section 04.1*

---

## Cross-Reference Guide

### Mathematical Notation Consistency
- Hat matrices: H_{qk}, H_{kk} with regularization parameter λ
- Semantic fields: ψ(z) in complex plane z = x + iy
- PLSA components: L_sem = T_comp - V_sem
- Action: S = Σ_t L_sem(x_t, ẋ_t)

### Section Connections
- **Philosophical → Mathematical**: Section 03.5 bridges conceptual frameworks to formal mathematics
- **PLSA Integration**: Section 03.3 unifies all mathematical frameworks under variational principle
- **Implementation Path**: Section 04.4 provides computational guidance for theoretical constructs

### Statistical Methodology
All empirical claims and validation protocols reference: [08_Appendix/08.5_methodology_statistical_significance.md](../08_Appendix/08.5_methodology_statistical_significance.md)