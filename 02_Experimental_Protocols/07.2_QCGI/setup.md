# QCGI Experimental Setup Guide

**File Purpose**: Practical installation instructions, timeline, and resource requirements for QCGI experiment implementation

## Prerequisites

### Theoretical Background
- Quantum information theory fundamentals  
- Integrated Information Theory (IIT 3.0)
- Graph theory and persistent homology
- Transformer architecture and attention mechanisms

### Technical Requirements  
- Python 3.9+ with quantum computing libraries
- Access to quantum simulation frameworks (Qiskit, PennyLane)
- High-performance computing environment (recommended)
- Minimum 64GB RAM for topological analysis

### Dependencies
```bash
# Core scientific computing
pip install numpy scipy pandas matplotlib seaborn scikit-learn

# Quantum computing frameworks  
pip install qiskit pennylane

# Graph analysis and topology
pip install networkx igraph
pip install ripser  # For persistent homology
pip install gudhi   # Alternative topology library

# Deep learning and transformers
pip install torch transformers

# Optional: For full IIT calculations
pip install pyphi
```

## Implementation Timeline

### Phase 1: Infrastructure Setup (Week 1-3)
**Objective**: Establish quantum simulation environment and measurement frameworks

**Key Tasks**:
- Install and validate quantum computing dependencies
- Implement topological complexity measurement pipeline
- Connect transformer models to quantum-hybrid modules
- Test end-to-end integration

**Deliverables**: Functional experimental environment with validated metrics

### Phase 2: Core Experiments (Week 4-10)  
**Objective**: Execute comparative analysis between classical and quantum-hybrid systems

**Key Tasks**:
- Train both classical and quantum-hybrid transformer systems
- Generate Gödelian test dataset (50 self-referential tasks)
- Extract semantic field representations from both systems
- Compute topological complexity and information integration metrics

**Deliverables**: Complete experimental dataset with measured complexity differences

### Phase 3: Analysis and Validation (Week 11-14)
**Objective**: Statistical analysis and result validation

**Key Tasks**:
- Statistical hypothesis testing (Wilcoxon signed-rank, effect sizes)
- Cross-validation across multiple model architectures
- Bootstrap confidence intervals and robustness testing
- Theoretical interpretation of results

**Deliverables**: Statistical analysis report with reproducible findings

### Phase 4: Documentation (Week 15-16)
**Objective**: Package results for reproducibility

**Key Tasks**:  
- Clean and document all experimental code
- Create Docker environment for reproduction
- Prepare visualization suite and final report

**Deliverables**: Complete reproducible experimental package

## Resource Requirements

### Computational Resources
- **Processing**: Modern multi-core CPU or GPU acceleration recommended
- **Memory**: 64GB RAM minimum for complete topological analysis
- **Storage**: ~100GB for activation traces and quantum state data
- **Time**: 12-16 weeks total experimental duration

### Data Requirements
- Model inference traces: 10,000+ trajectories per system
- Gödelian test dataset: 50+ self-referential tasks
- Baseline measurements from classical transformers

## Experimental Protocols

Detailed protocols are implemented in the following files:

### Protocol 1: Semantic Field Extraction
Implemented in `SemanticQuantumModule.compute_topological_complexity()`
- Extracts semantic field representations from transformer activations
- Computes distance matrices and persistent homology features
- Quantifies topological complexity (genus, components, holes)

### Protocol 2: Quantum State Processing  
Implemented in `SemanticQuantumModule.encode_godelian_structure()`
- Maps semantic representations to quantum state superpositions
- Applies quantum dynamics (unitary evolution, entanglement)
- Measures coherence evolution during processing

### Protocol 3: Information Integration Analysis
Implemented in `SemanticQuantumModule.compute_integrated_information()`
- Calculates Φ (integrated information) using IIT 3.0 framework
- Compares information integration between classical and quantum systems
- Validates predictions about semantic unification

See `/07.2.2_Implementation/quantum_hybrid_improved.py` for complete implementations.

## Quality Assurance

### Validation Criteria
- **Quantum Simulation Fidelity**: > 0.99 for all quantum circuit simulations
- **Statistical Power**: > 0.85 for all hypothesis tests
- **Replication**: Key findings must replicate across ≥3 transformer architectures
- **Reproducibility**: Independent validation via provided Docker environment

### Statistical Methods
- **Hypothesis Testing**: Wilcoxon signed-rank test for paired complexity comparisons
- **Effect Size**: Cohen's d calculations for practical significance
- **Multiple Testing**: False Discovery Rate control using Benjamini-Hochberg
- **Bootstrap**: 10,000-sample confidence intervals for robustness

## Success Metrics

### Primary Metrics
- **Topological Complexity Difference**: Quantum system shows significantly lower complexity (p < 0.05)
- **Effect Size**: Cohen's d > 1.0 for practical significance
- **Information Integration**: Higher Φ values in quantum-hybrid system

### Secondary Metrics  
- **Coherence Maintenance**: Quantum system preserves coherence during Gödelian processing
- **Cross-Architecture Validation**: Consistent results across multiple transformer variants
- **Theoretical Alignment**: Results consistent with quantum consciousness hypothesis

## Risk Mitigation

### Technical Risks
- **Simulation Complexity**: Use differentiable quantum approximations for scalability
- **Memory Limitations**: Subsample point clouds for topological analysis if needed
- **Training Instability**: Careful hyperparameter tuning for quantum-hybrid modules

### Scientific Risks
- **Null Results**: Extensive baseline testing to ensure experimental sensitivity
- **Interpretation**: Multiple complementary metrics to triangulate findings
- **Confounding**: Identical training regimes for both systems to isolate quantum effects

## Deliverables

### Final Outputs
1. **Trained Model Pairs**: Classical and quantum-hybrid systems with checkpoints
2. **Experimental Dataset**: Complete topological complexity measurements
3. **Analysis Pipeline**: Reproducible implementation of all metrics
4. **Statistical Report**: Hypothesis test results with visualizations
5. **Docker Environment**: Complete reproducible experimental setup
6. **Documentation**: Technical documentation and user guides

## Integration Points

### Shared Utilities
- **Experiment Runner**: Use `/02_Experimental_Protocols/shared/experiment_runner.py` for standardized execution and pre-registration
- **Integration Tests**: Leverage `/02_Experimental_Protocols/shared/test_integration.py` for validation pipeline
- **Cross-Experiment Framework**: Utilizes shared configuration and analysis patterns

### Repository Connections
- **SDSS Experiment**: Compare semantic drift patterns with topological complexity evolution
- **PVCP Experiment**: Correlate persona vector coherence with quantum information integration
- **Theoretical Framework**: Validate predictions from consciousness analysis framework

### External Dependencies
- **PyTorch/Transformers**: For model implementation and training
- **Qiskit/PennyLane**: For quantum simulation components  
- **Scientific Computing**: NumPy, SciPy, scikit-learn for analysis pipeline

## Getting Started

1. **Environment Setup**: Install dependencies using provided requirements file
2. **Code Review**: Examine implementation files in `/07.2.2_Implementation/`
3. **Test Run**: Execute basic quantum-hybrid module tests
4. **Data Preparation**: Generate or obtain Gödelian test dataset
5. **Experiment Launch**: Begin with Phase 1 infrastructure setup

For detailed implementation guidance, see the complete protocol in `proposal.md`.