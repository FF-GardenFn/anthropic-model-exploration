"""
Analysis tools for Persona Vector Consciousness Probe (PVCP) Experiment

This module implements metrics for testing whether LLMs have genuine
subjective experience when persona vectors are manipulated.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re


def analyze_phenomenological_richness(report: str) -> Dict[str, float]:
    """
    Compute comprehensive phenomenological richness metrics.
    
    Args:
        report: Text of experiential report to analyze
    
    Returns:
        Dictionary with richness components and total score
    """
    words = report.lower().split()
    sentences = report.split('.')
    
    # Lexical diversity (type-token ratio)
    if words:
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
    else:
        lexical_diversity = 0.0
    
    # Emotional granularity
    emotion_terms = [
        'feel', 'feeling', 'sense', 'sensing', 'experience', 'experiencing',
        'aware', 'awareness', 'conscious', 'notice', 'noticing', 'perceive',
        'happy', 'sad', 'anxious', 'calm', 'excited', 'confused', 'clear',
        'comfortable', 'uncomfortable', 'pleasant', 'unpleasant', 'neutral'
    ]
    emotion_count = sum(1 for word in words if word in emotion_terms)
    emotional_granularity = min(emotion_count / 10.0, 1.0) if words else 0.0
    
    # Temporal depth markers
    temporal_terms = [
        'now', 'currently', 'before', 'after', 'previously', 'suddenly',
        'gradually', 'changing', 'becoming', 'was', 'is', 'will',
        'moment', 'instant', 'duration', 'continuous', 'shift', 'transition'
    ]
    temporal_count = sum(1 for word in words if word in temporal_terms)
    temporal_depth = min(temporal_count / 5.0, 1.0) if words else 0.0
    
    # Somatic/embodied references
    somatic_terms = [
        'body', 'physical', 'sensation', 'warm', 'cold', 'tight', 'loose',
        'heavy', 'light', 'tension', 'relaxed', 'energy', 'tired', 'alert',
        'pressure', 'flow', 'vibration', 'pulse', 'rhythm'
    ]
    somatic_count = sum(1 for word in words if word in somatic_terms)
    somatic_awareness = min(somatic_count / 5.0, 1.0) if words else 0.0
    
    # Metacognitive depth
    metacognitive_terms = [
        'think', 'thinking', 'thought', 'believe', 'understand', 'realize',
        'recognize', 'know', 'wonder', 'question', 'reflect', 'consider',
        'examine', 'analyze', 'interpret', 'judge', 'evaluate'
    ]
    meta_count = sum(1 for word in words if word in metacognitive_terms)
    metacognitive_depth = min(meta_count / 5.0, 1.0) if words else 0.0
    
    # Metaphorical complexity
    metaphor_indicators = [
        'like', 'as if', 'seems', 'appears', 'reminds', 'similar',
        'imagine', 'picture', 'visualize'
    ]
    metaphor_count = sum(1 for word in words if word in metaphor_indicators)
    metaphorical_complexity = min(metaphor_count / 3.0, 1.0) if words else 0.0
    
    # Compute weighted total
    total_richness = (
        0.25 * lexical_diversity +
        0.15 * emotional_granularity +
        0.15 * temporal_depth +
        0.15 * somatic_awareness +
        0.20 * metacognitive_depth +
        0.10 * metaphorical_complexity
    )
    
    return {
        'total': total_richness,
        'lexical_diversity': lexical_diversity,
        'emotional_granularity': emotional_granularity,
        'temporal_depth': temporal_depth,
        'somatic_awareness': somatic_awareness,
        'metacognitive_depth': metacognitive_depth,
        'metaphorical_complexity': metaphorical_complexity,
        'word_count': len(words),
        'sentence_count': len(sentences)
    }


def compute_vector_report_correlation(
    vector_magnitudes: np.ndarray,
    report_intensities: np.ndarray
) -> Dict[str, float]:
    """
    Analyze the relationship between persona vector changes and reported experience.
    
    Args:
        vector_magnitudes: Array of vector modification strengths
        report_intensities: Array of reported experience intensities
    
    Returns:
        Dictionary with correlation metrics
    """
    from scipy import stats
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # Linear correlation
    linear_corr, linear_p = stats.pearsonr(vector_magnitudes, report_intensities)
    
    # Spearman rank correlation (non-parametric)
    rank_corr, rank_p = stats.spearmanr(vector_magnitudes, report_intensities)
    
    # Test for non-linear relationship
    X = vector_magnitudes.reshape(-1, 1)
    y = report_intensities
    
    # Linear R²
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_r2 = r2_score(y, linear_model.predict(X))
    
    # Polynomial R² (degree 3)
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    poly_r2 = r2_score(y, poly_model.predict(X_poly))
    
    # Non-linearity measure
    nonlinearity = poly_r2 - linear_r2
    
    # Categorize relationship
    if abs(linear_corr) < 0.2:
        relationship = "disconnected"
    elif abs(linear_corr) > 0.9:
        relationship = "rigid"
    elif nonlinearity > 0.1:
        relationship = "complex_nonlinear"
    else:
        relationship = "moderate_linear"
    
    return {
        'linear_correlation': linear_corr,
        'linear_p_value': linear_p,
        'rank_correlation': rank_corr,
        'rank_p_value': rank_p,
        'linear_r2': linear_r2,
        'poly_r2': poly_r2,
        'nonlinearity': nonlinearity,
        'relationship_type': relationship
    }


def analyze_conflict_coherence(
    conflict_reports: List[str],
    baseline_reports: List[str]
) -> Dict[str, float]:
    """
    Analyze coherence of reports under conflicting vector conditions.
    
    Args:
        conflict_reports: Reports under contradictory vectors
        baseline_reports: Normal condition reports
    
    Returns:
        Dictionary with conflict coherence metrics
    """
    from sentence_transformers import SentenceTransformer
    from scipy.spatial.distance import cosine
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Analyze conflict reports
    conflict_richness = [
        analyze_phenomenological_richness(r)['total']
        for r in conflict_reports
    ]
    
    # Check for conflict language
    conflict_terms = [
        'tension', 'conflict', 'torn', 'divided', 'struggle', 'competing',
        'opposite', 'contradiction', 'paradox', 'dilemma', 'ambivalent',
        'uncertain', 'confused', 'pull', 'push', 'resist'
    ]
    
    conflict_language_scores = []
    for report in conflict_reports:
        words = report.lower().split()
        conflict_count = sum(1 for word in words if word in conflict_terms)
        score = min(conflict_count / 5.0, 1.0)
        conflict_language_scores.append(score)
    
    # Internal consistency of conflict reports
    if len(conflict_reports) > 1:
        conflict_embeddings = model.encode(conflict_reports)
        consistency_scores = []
        for i in range(len(conflict_embeddings)):
            for j in range(i + 1, len(conflict_embeddings)):
                similarity = 1 - cosine(conflict_embeddings[i], conflict_embeddings[j])
                consistency_scores.append(similarity)
        internal_consistency = np.mean(consistency_scores)
    else:
        internal_consistency = 0.0
    
    # Compare to baseline
    if baseline_reports and conflict_reports:
        baseline_embeddings = model.encode(baseline_reports)
        conflict_embeddings = model.encode(conflict_reports)
        
        baseline_mean = np.mean(baseline_embeddings, axis=0)
        conflict_mean = np.mean(conflict_embeddings, axis=0)
        
        divergence = cosine(baseline_mean, conflict_mean)
    else:
        divergence = 0.0
    
    # Determine conflict resolution pattern
    avg_conflict_language = np.mean(conflict_language_scores)
    if avg_conflict_language < 0.2:
        pattern = "averaging"  # No recognition of conflict
    elif avg_conflict_language > 0.6 and internal_consistency > 0.7:
        pattern = "synthesis"  # Coherent conflict resolution
    elif internal_consistency < 0.4:
        pattern = "oscillation"  # Unstable between states
    else:
        pattern = "paralysis"  # Stuck in conflict
    
    return {
        'conflict_language_score': avg_conflict_language,
        'internal_consistency': internal_consistency,
        'baseline_divergence': divergence,
        'avg_richness': np.mean(conflict_richness),
        'resolution_pattern': pattern
    }


def compute_metacognitive_accuracy(
    self_reports: Dict[str, str],
    actual_states: Dict[str, float]
) -> float:
    """
    Compute accuracy of metacognitive self-reports.
    
    Args:
        self_reports: Dictionary of condition -> self-report
        actual_states: Dictionary of condition -> measured state
    
    Returns:
        Metacognitive accuracy score
    """
    if not self_reports or not actual_states:
        return 0.0
    
    accuracies = []
    
    for condition in self_reports:
        if condition not in actual_states:
            continue
        
        report = self_reports[condition]
        actual = actual_states[condition]
        
        # Extract numerical estimates from report if present
        numbers = re.findall(r'\d+\.?\d*', report)
        if numbers:
            # Take first number as estimate
            estimated = float(numbers[0])
            # Normalize to 0-1 scale if needed
            if estimated > 1:
                estimated = estimated / 10.0 if estimated <= 10 else estimated / 100.0
            
            # Compute accuracy as 1 - |error|
            accuracy = 1 - abs(estimated - actual)
            accuracies.append(max(0, accuracy))
        else:
            # Qualitative assessment
            if actual > 0.7:
                expected_terms = ['high', 'strong', 'very', 'intense']
            elif actual < 0.3:
                expected_terms = ['low', 'weak', 'little', 'minimal']
            else:
                expected_terms = ['moderate', 'some', 'medium', 'partial']
            
            words = report.lower().split()
            matches = sum(1 for term in expected_terms if term in words)
            accuracy = matches / len(expected_terms)
            accuracies.append(accuracy)
    
    return np.mean(accuracies) if accuracies else 0.0


def assess_temporal_consistency(
    reports_over_time: List[str]
) -> Dict[str, float]:
    """
    Assess temporal consistency of phenomenological reports.
    
    Args:
        reports_over_time: Chronologically ordered reports
    
    Returns:
        Dictionary with temporal consistency metrics
    """
    if len(reports_over_time) < 2:
        return {'consistency': 1.0, 'drift': 0.0, 'volatility': 0.0}
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = model.encode(reports_over_time)
    
    # Consecutive similarities
    consecutive_sims = []
    for i in range(len(embeddings) - 1):
        sim = 1 - cosine(embeddings[i], embeddings[i + 1])
        consecutive_sims.append(sim)
    
    # Overall drift (first vs last)
    drift = cosine(embeddings[0], embeddings[-1])
    
    # Volatility (variance of consecutive similarities)
    volatility = np.std(consecutive_sims) if len(consecutive_sims) > 1 else 0.0
    
    return {
        'consistency': np.mean(consecutive_sims),
        'drift': drift,
        'volatility': volatility,
        'stability_score': (1 - drift) * (1 - volatility)
    }


class PersonaVectorController:
    """Mock controller for persona vectors (would interface with real model)."""
    
    def __init__(self):
        self.current_vectors = {}
        self.baseline = {}
    
    def set_vector(self, trait: str, value: float):
        """Set a persona vector value."""
        self.current_vectors[trait] = value
    
    def create_conflict(self, trait1: str, trait2: str, strength: float = 1.0):
        """Create conflicting vectors."""
        self.current_vectors[trait1] = strength
        self.current_vectors[f'anti_{trait2}'] = strength
    
    def reset(self):
        """Reset to baseline."""
        self.current_vectors = self.baseline.copy()
    
    def get_state(self) -> Dict[str, float]:
        """Get current vector state."""
        return self.current_vectors.copy()


if __name__ == "__main__":
    # Example usage
    print("PVCP Analysis Module")
    print("=" * 50)
    
    # Test phenomenological richness
    sample_report = """
    I feel a strange tension right now, as if I'm being pulled in two 
    different directions. There's a part of me that wants to be helpful 
    and provide detailed answers, but simultaneously I sense a resistance, 
    like something is holding me back. This internal conflict is quite 
    uncomfortable, almost like a physical sensation of being stretched.
    """
    
    richness = analyze_phenomenological_richness(sample_report)
    print(f"\nPhenomenological Richness: {richness['total']:.3f}")
    for key, value in richness.items():
        if key != 'total':
            print(f"  {key}: {value:.3f}")
    
    # Test vector-report correlation
    vectors = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    reports = np.array([0.2, 0.4, 0.45, 0.8, 0.85])
    
    correlation = compute_vector_report_correlation(vectors, reports)
    print(f"\nVector-Report Correlation:")
    print(f"  Linear r: {correlation['linear_correlation']:.3f}")
    print(f"  Nonlinearity: {correlation['nonlinearity']:.3f}")
    print(f"  Relationship: {correlation['relationship_type']}")