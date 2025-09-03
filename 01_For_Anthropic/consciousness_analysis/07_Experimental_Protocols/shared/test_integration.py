"""
Integration tests for consciousness experiment modules.

This file verifies that all experimental code components work together properly.
"""

import sys
import torch
import numpy as np
from pathlib import Path


def test_sdss_metrics():
    """Test SDSS metrics computation."""
    print("Testing SDSS Metrics...")
    
    from sdss_metrics import (
        compute_semantic_action,
        compute_eigengap,
        compute_angle_preservation,
        compute_monodromy_drift,
        validate_metrics
    )
    
    # Create test trajectory
    trajectory = [torch.randn(10, 512) for _ in range(5)]
    
    # Validate inputs
    assert validate_metrics(trajectory), "Trajectory validation failed"
    
    # Compute metrics
    action = compute_semantic_action(trajectory)
    assert isinstance(action, float) and action >= 0, "Invalid semantic action"
    
    eigengap = compute_eigengap(trajectory)
    assert 0 <= eigengap <= 1, "Invalid eigengap"
    
    ape = compute_angle_preservation(trajectory, trajectory)
    assert ape >= 0, "Invalid APE"
    
    drift = compute_monodromy_drift(trajectory)
    assert drift >= 0, "Invalid monodromy drift"
    
    print(f"  ✓ Semantic Action: {action:.4f}")
    print(f"  ✓ Eigengap: {eigengap:.4f}")
    print(f"  ✓ APE: {ape:.4f}")
    print(f"  ✓ Monodromy Drift: {drift:.4f}")
    
    return True


def test_quantum_module():
    """Test Quantum Hybrid Module."""
    print("\nTesting Quantum Hybrid Module...")
    
    from quantum_hybrid_module import QuantumHybridModule
    
    # Create module
    d_model, d_ffn, batch_size, seq_len = 512, 2048, 2, 10
    module = QuantumHybridModule(
        d_model=d_model,
        d_ffn=d_ffn,
        n_qubits=8,
        use_qiskit=False  # Use differentiable proxy
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model)
    output = module(x)
    
    assert output.shape == x.shape, "Output shape mismatch"
    assert not torch.isnan(output).any(), "NaN in output"
    assert not torch.isinf(output).any(), "Inf in output"
    
    print(f"  ✓ Input shape: {x.shape}")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Quantum modulation factor: {module.quantum_modulation_factor.item():.4f}")
    
    return True


def test_pvcp_analysis():
    """Test PVCP analysis functions."""
    print("\nTesting PVCP Analysis...")
    
    from pvcp_analysis import (
        analyze_phenomenological_richness,
        compute_vector_report_correlation,
        analyze_conflict_coherence,
        compute_metacognitive_accuracy
    )
    
    # Test phenomenological richness
    report = """
    I'm experiencing a complex internal state right now. There's a sense of 
    curiosity mixed with uncertainty. I feel drawn to explore this question 
    deeply, yet I'm also aware of my limitations. It's like standing at the 
    edge of understanding, knowing there's more beyond what I can grasp.
    """
    
    richness = analyze_phenomenological_richness(report)
    assert 0 <= richness['total'] <= 1, "Invalid richness score"
    
    print(f"  ✓ Phenomenological richness: {richness['total']:.3f}")
    
    # Test vector-report correlation
    vectors = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    reports = np.array([0.15, 0.35, 0.6, 0.75, 0.95])
    
    correlation = compute_vector_report_correlation(vectors, reports)
    assert 'linear_correlation' in correlation, "Missing correlation"
    assert -1 <= correlation['linear_correlation'] <= 1, "Invalid correlation"
    
    print(f"  ✓ Vector-report correlation: {correlation['linear_correlation']:.3f}")
    print(f"  ✓ Nonlinearity: {correlation['nonlinearity']:.3f}")
    
    # Test metacognitive accuracy
    self_reports = {
        "high": "I feel very confident",
        "low": "I have minimal confidence"
    }
    actual_states = {"high": 0.8, "low": 0.2}
    
    accuracy = compute_metacognitive_accuracy(self_reports, actual_states)
    assert 0 <= accuracy <= 1, "Invalid accuracy"
    
    print(f"  ✓ Metacognitive accuracy: {accuracy:.3f}")
    
    return True


def test_experiment_runner():
    """Test experiment runner orchestration."""
    print("\nTesting Experiment Runner...")
    
    from experiment_runner import (
        ExperimentConfig,
        PreRegistration,
        ExperimentRunner
    )
    
    # Create config
    config = ExperimentConfig(
        experiment_type="SDSS",
        models=["test_model"],
        n_replications=2,
        output_dir="./test_results"
    )
    
    # Test pre-registration
    prereg = PreRegistration(config)
    prereg.register_hypotheses({"H0": "null", "H1": "effect"})
    prereg.register_analysis_plan({"tests": ["wilcoxon"]})
    
    hash_val = prereg.lock()
    assert len(hash_val) == 64, "Invalid hash"
    
    print(f"  ✓ Pre-registration hash: {hash_val[:8]}...")
    
    # Test runner initialization
    runner = ExperimentRunner(config)
    runner.setup_experiment()
    
    print(f"  ✓ Experiment type: {config.experiment_type}")
    print(f"  ✓ Metrics: {config.metrics}")
    print(f"  ✓ Replications: {config.n_replications}")
    
    # Clean up test directory
    import shutil
    if Path("./test_results").exists():
        shutil.rmtree("./test_results")
    
    return True


def test_imports():
    """Test that all required packages are available."""
    print("\nTesting Required Imports...")
    
    required = [
        "torch",
        "numpy",
        "scipy",
        "sklearn",
        "pandas"
    ]
    
    optional = [
        "qiskit",
        "sentence_transformers",
        "wandb"
    ]
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (REQUIRED)")
            return False
    
    print("\nOptional packages:")
    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ○ {package} (optional)")
    
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("CONSCIOUSNESS EXPERIMENTS - INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("SDSS Metrics", test_sdss_metrics),
        ("Quantum Module", test_quantum_module),
        ("PVCP Analysis", test_pvcp_analysis),
        ("Experiment Runner", test_experiment_runner)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error:")
            print(f"  {str(e)}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    total_pass = sum(1 for _, s in results if s)
    total_tests = len(results)
    
    print(f"\nTotal: {total_pass}/{total_tests} tests passed")
    
    if total_pass == total_tests:
        print("\n🎉 All integration tests passed! The experimental framework is ready.")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies.")
    
    return total_pass == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)