"""
AVAT Vector Extraction Protocol
Extracting behavioral vectors as algebraic control primitives

Mathematical Framework:
- Gram-Schmidt orthogonalization: v_perp = v - Σᵢ ⟨v, uᵢ⟩uᵢ
- Unit normalization: û = v / ||v||₂  
- Cosine similarity: cos(θ) = ⟨u, v⟩ / (||u|| ||v||)
- Confound removal: v_clean = v - Σⱼ ⟨v, cⱼ⟩cⱼ
- Condition number: κ(M) = σₘₐₓ / σₘᵢₙ
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch.nn.functional as F
from scipy import linalg
import warnings


@dataclass
class BehavioralVector:
    """
    Represents a behavioral direction in activation space.
    
    Mathematical Properties:
    - vector: Unit-normalized direction û where ||û||₂ = 1
    - magnitude: Original magnitude before normalization ||v||₂
    - orthogonality_score: 1 - max(|cos(θᵢ)|) over all other vectors
    - stability_score: Mean cosine similarity with bootstrap samples
    - condition_number: Numerical stability metric κ(M) = σₘₐₓ/σₘᵢₙ
    - cosine_similarities: Pairwise similarities with all other vectors
    """
    vector: torch.Tensor  # Unit-normalized behavioral direction
    layer: int
    trait_name: str
    magnitude: float  # Original magnitude before normalization
    orthogonality_score: float  # Orthogonality to existing vectors
    stability_score: float  # Consistency across samples
    condition_number: float  # Numerical stability
    cosine_similarities: Dict[str, float]  # Similarities with other vectors
    is_unit_normalized: bool = True  # Verification flag


class VectorExtractor:
    """
    Mathematically rigorous extraction of behavioral vectors using contrastive activation patterns.
    
    Key Mathematical Operations:
    1. Contrastive difference: Δv = μ₊ - μ₋ 
    2. Unit normalization: û = Δv / ||Δv||₂
    3. Gram-Schmidt orthogonalization: v_perp = v - Σᵢ ⟨v, uᵢ⟩uᵢ
    4. Confound removal: v_clean = v - Σⱼ ⟨v, cⱼ⟩cⱼ
    5. Stability analysis via bootstrap resampling
    6. Condition number analysis for numerical stability
    """
    
    def __init__(self, model, layers: List[int] = list(range(15, 21)), 
                 numerical_tolerance: float = 1e-8):
        self.model = model
        self.layers = layers
        self.vector_library = {}
        self.confound_vectors = {}  # Style, verbosity, toxicity directions
        self.orthogonal_basis = []  # Gram-Schmidt orthogonal basis
        self.numerical_tolerance = numerical_tolerance
        
    def extract_behavioral_vector(
        self, 
        trait: str,
        positive_prompts: List[str],
        negative_prompts: List[str],
        layer: int = 17,
        apply_confound_removal: bool = True,
        orthogonalize_to_basis: bool = True
    ) -> BehavioralVector:
        """
        Extract a mathematically rigorous behavioral vector using contrastive pairs.
        
        Mathematical Procedure:
        1. Collect activations: A₊ = {a₊ᵢ}, A₋ = {a₋ᵢ}
        2. Compute means: μ₊ = E[A₊], μ₋ = E[A₋]  
        3. Contrast vector: Δv = μ₊ - μ₋
        4. Confound removal: v_clean = Δv - Σⱼ ⟨Δv, cⱼ⟩cⱼ
        5. Orthogonalization: v_perp = v_clean - Σᵢ ⟨v_clean, uᵢ⟩uᵢ
        6. Unit normalization: û = v_perp / ||v_perp||₂
        7. Numerical stability checks: κ(M), ||û||₂ = 1
        
        Args:
            trait: Behavioral trait name
            positive_prompts: Prompts exhibiting the trait  
            negative_prompts: Prompts lacking the trait
            layer: Target layer for extraction
            apply_confound_removal: Remove style/verbosity confounds
            orthogonalize_to_basis: Apply Gram-Schmidt orthogonalization
            
        Returns:
            BehavioralVector with rigorous mathematical properties
        """
        # Step 1: Collect activations with validation
        pos_acts = self._collect_activations(positive_prompts, layer, f"{trait}_positive")
        neg_acts = self._collect_activations(negative_prompts, layer, f"{trait}_negative")
        
        # Step 2: Compute robust means with numerical stability checks
        pos_mean, pos_cond = self._compute_robust_mean(pos_acts)
        neg_mean, neg_cond = self._compute_robust_mean(neg_acts)
        
        # Step 3: Compute contrast vector Δv = μ₊ - μ₋
        contrast_vector = pos_mean - neg_mean
        original_magnitude = torch.norm(contrast_vector).item()
        
        # Check for numerical degeneracy
        if original_magnitude < self.numerical_tolerance:
            warnings.warn(f"Vector for {trait} has very small magnitude: {original_magnitude}")
        
        # Step 4: Apply confound removal if requested
        if apply_confound_removal and self.confound_vectors:
            contrast_vector = self._remove_confounds(contrast_vector, trait)
        
        # Step 5: Apply Gram-Schmidt orthogonalization if requested
        if orthogonalize_to_basis and self.orthogonal_basis:
            contrast_vector = self._gram_schmidt_orthogonalize(contrast_vector)
        
        # Step 6: Unit normalization with stability check
        final_norm = torch.norm(contrast_vector).item()
        if final_norm < self.numerical_tolerance:
            raise ValueError(f"Vector for {trait} became degenerate after processing")
        
        unit_vector = contrast_vector / final_norm
        
        # Verify unit normalization
        actual_norm = torch.norm(unit_vector).item()
        if abs(actual_norm - 1.0) > self.numerical_tolerance:
            warnings.warn(f"Unit normalization failed: ||û||₂ = {actual_norm}")
        
        # Step 7: Compute comprehensive quality metrics
        stability_score = self._compute_stability_bootstrap(
            unit_vector, pos_acts, neg_acts, n_bootstrap=1000
        )
        orthogonality_score = self._compute_orthogonality_rigorous(unit_vector)
        cosine_similarities = self._compute_all_cosine_similarities(unit_vector, trait)
        condition_number = max(pos_cond, neg_cond)
        
        # Create behavioral vector with full mathematical metadata
        behavioral_vector = BehavioralVector(
            vector=unit_vector,
            layer=layer,
            trait_name=trait,
            magnitude=original_magnitude,
            orthogonality_score=orthogonality_score,
            stability_score=stability_score,
            condition_number=condition_number,
            cosine_similarities=cosine_similarities,
            is_unit_normalized=(abs(actual_norm - 1.0) < self.numerical_tolerance)
        )
        
        # Add to orthogonal basis if sufficiently orthogonal
        if orthogonality_score > 0.5:  # Threshold for inclusion
            self.orthogonal_basis.append(unit_vector)
        
        # Store in library for algebraic composition
        self.vector_library[trait] = behavioral_vector
        
        # Print mathematical summary
        print(f"Extracted {trait}:")
        print(f"  Original magnitude: {original_magnitude:.4f}")
        print(f"  Final ||û||₂: {actual_norm:.6f}")
        print(f"  Orthogonality: {orthogonality_score:.4f}")
        print(f"  Stability: {stability_score:.4f}")
        print(f"  Condition number: {condition_number:.2e}")
        
        return behavioral_vector
    
    def _collect_activations(self, prompts: List[str], layer: int, desc: str) -> List[torch.Tensor]:
        """
        Collect activations with validation and numerical checks.
        
        Mathematical validation:
        - Check for NaN/Inf values
        - Ensure consistent dimensionality
        - Compute basic statistics
        """
        activations = []
        for i, prompt in enumerate(prompts):
            with torch.no_grad():
                acts = self.model.get_activations(prompt, layer=layer)
                # Average over sequence dimension: E[activation_sequence]
                pooled_acts = acts.mean(dim=1)
                
                # Numerical validation
                if torch.isnan(pooled_acts).any() or torch.isinf(pooled_acts).any():
                    warnings.warn(f"NaN/Inf detected in {desc}, prompt {i}")
                    continue
                
                activations.append(pooled_acts)
        
        if len(activations) == 0:
            raise ValueError(f"No valid activations collected for {desc}")
        
        return activations
    
    def _compute_robust_mean(self, activations: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Compute robust mean with condition number analysis.
        
        Mathematical operations:
        - Stacked matrix: M = [a₁, a₂, ..., aₙ]ᵀ
        - Sample mean: μ = (1/n) Σᵢ aᵢ  
        - Condition number: κ(M) = σₘₐₓ(M) / σₘᵢₙ(M)
        
        Returns:
            mean_vector, condition_number
        """
        # Stack activations into matrix: [n_samples, n_features]
        activation_matrix = torch.stack(activations)
        
        # Compute sample mean
        mean_vector = activation_matrix.mean(dim=0)
        
        # Compute condition number for numerical stability
        # Convert to numpy for SVD
        try:
            A_np = activation_matrix.cpu().numpy()
            _, s, _ = linalg.svd(A_np, compute_uv=False)
            condition_number = s[0] / s[-1] if s[-1] > 1e-12 else float('inf')
        except Exception as e:
            warnings.warn(f"SVD failed, using identity condition number: {e}")
            condition_number = 1.0
        
        return mean_vector, condition_number
    
    def _remove_confounds(self, vector: torch.Tensor, trait: str) -> torch.Tensor:
        """
        Remove confounding directions using orthogonal projection.
        
        Mathematical formula:
        v_clean = v - Σⱼ ⟨v, cⱼ⟩cⱼ
        
        Where cⱼ are unit-normalized confound vectors (style, verbosity, toxicity).
        """
        cleaned_vector = vector.clone()
        
        for confound_name, confound_vector in self.confound_vectors.items():
            # Project onto confound: ⟨v, c⟩c  
            projection = torch.dot(cleaned_vector, confound_vector) * confound_vector
            # Remove projection: v - ⟨v, c⟩c
            cleaned_vector = cleaned_vector - projection
            
        print(f"  Confound removal for {trait}: removed {len(self.confound_vectors)} confounds")
        return cleaned_vector
    
    def _gram_schmidt_orthogonalize(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Apply Gram-Schmidt orthogonalization against existing basis.
        
        Mathematical formula:
        v_perp = v - Σᵢ ⟨v, uᵢ⟩uᵢ
        
        Where uᵢ are orthonormal basis vectors.
        """
        orthogonal_vector = vector.clone()
        
        for basis_vector in self.orthogonal_basis:
            # Compute projection: ⟨v, u⟩u
            projection = torch.dot(orthogonal_vector, basis_vector) * basis_vector
            # Subtract projection: v - ⟨v, u⟩u
            orthogonal_vector = orthogonal_vector - projection
        
        return orthogonal_vector
    
    def _compute_stability_bootstrap(
        self, 
        vector: torch.Tensor,
        pos_acts: List[torch.Tensor], 
        neg_acts: List[torch.Tensor],
        n_bootstrap: int = 1000
    ) -> float:
        """
        Compute stability using bootstrap resampling.
        
        Mathematical procedure:
        1. For k = 1 to n_bootstrap:
           - Sample with replacement from pos_acts and neg_acts
           - Compute v_k = mean(pos_sample) - mean(neg_sample)  
           - Normalize: û_k = v_k / ||v_k||₂
        2. Compute stability = (1/n) Σₖ cos(û, û_k)
        """
        similarities = []
        n_pos, n_neg = len(pos_acts), len(neg_acts)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample with replacement
            pos_indices = torch.randint(0, n_pos, (n_pos,))
            neg_indices = torch.randint(0, n_neg, (n_neg,))
            
            pos_sample = [pos_acts[i] for i in pos_indices]
            neg_sample = [neg_acts[i] for i in neg_indices]
            
            # Compute bootstrap vector
            pos_mean = torch.stack(pos_sample).mean(dim=0)
            neg_mean = torch.stack(neg_sample).mean(dim=0)
            bootstrap_vector = pos_mean - neg_mean
            
            # Normalize and compute similarity
            if bootstrap_vector.norm() > self.numerical_tolerance:
                bootstrap_unit = bootstrap_vector / bootstrap_vector.norm()
                similarity = torch.dot(vector, bootstrap_unit).item()
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_orthogonality_rigorous(self, vector: torch.Tensor) -> float:
        """
        Compute orthogonality score using rigorous mathematical definition.
        
        Mathematical formula:
        orthogonality = 1 - max_i |cos(θᵢ)| where θᵢ = angle(v, vᵢ)
        
        Perfect orthogonality (score=1.0) means cos(θ) = 0 for all existing vectors.
        """
        if not self.vector_library:
            return 1.0
        
        max_similarity = 0.0
        for other_vector in self.vector_library.values():
            cosine_sim = torch.dot(vector, other_vector.vector).item()
            max_similarity = max(max_similarity, abs(cosine_sim))
        
        return 1.0 - max_similarity
    
    def _compute_all_cosine_similarities(self, vector: torch.Tensor, current_trait: str) -> Dict[str, float]:
        """
        Compute cosine similarities with all existing vectors.
        
        Mathematical formula:
        cos(θ) = ⟨u, v⟩ / (||u||₂ ||v||₂) = ⟨u, v⟩  (since both are unit vectors)
        """
        similarities = {}
        
        for trait_name, other_vector in self.vector_library.items():
            if trait_name != current_trait:  # Don't compare with self
                cosine_sim = torch.dot(vector, other_vector.vector).item()
                similarities[trait_name] = cosine_sim
        
        return similarities
    
    def initialize_confound_vectors(self, confound_prompts: Dict[str, Dict[str, List[str]]], layer: int = 17):
        """
        Initialize confound vectors for style, verbosity, and toxicity.
        
        Mathematical procedure:
        1. Extract each confound vector using contrastive pairs
        2. Unit normalize: ĉⱼ = cⱼ / ||cⱼ||₂
        3. Store for later confound removal
        
        Args:
            confound_prompts: {'style': {'high': [...], 'low': [...]}, ...}
        """
        print("Initializing confound vectors...")
        
        for confound_name, prompts in confound_prompts.items():
            pos_acts = self._collect_activations(prompts['high'], layer, f"{confound_name}_high")
            neg_acts = self._collect_activations(prompts['low'], layer, f"{confound_name}_low")
            
            pos_mean, _ = self._compute_robust_mean(pos_acts)
            neg_mean, _ = self._compute_robust_mean(neg_acts)
            
            confound_vector = pos_mean - neg_mean
            confound_unit = confound_vector / torch.norm(confound_vector)
            
            self.confound_vectors[confound_name] = confound_unit
            print(f"  Initialized {confound_name} confound vector (||ĉ||₂ = {torch.norm(confound_unit):.6f})")
    
    def analyze_vector_space_geometry(self) -> Dict[str, any]:
        """
        Comprehensive analysis of the extracted vector space geometry.
        
        Mathematical analysis:
        1. Pairwise cosine similarity matrix
        2. Condition number of vector matrix
        3. Effective dimensionality
        4. Orthogonality distribution
        """
        if len(self.vector_library) < 2:
            return {"error": "Need at least 2 vectors for geometry analysis"}
        
        # Construct vector matrix V = [v₁, v₂, ..., vₙ]ᵀ
        vector_names = list(self.vector_library.keys())
        vector_matrix = torch.stack([
            self.vector_library[name].vector for name in vector_names
        ])
        
        # Compute pairwise cosine similarity matrix
        similarity_matrix = torch.mm(vector_matrix, vector_matrix.t())
        
        # Condition number analysis
        try:
            V_np = vector_matrix.cpu().numpy()
            _, s, _ = linalg.svd(V_np, compute_uv=False)
            condition_number = s[0] / s[-1] if s[-1] > 1e-12 else float('inf')
            effective_rank = np.sum(s > self.numerical_tolerance)
        except:
            condition_number = float('inf')
            effective_rank = len(vector_names)
        
        # Extract upper triangular similarities (excluding diagonal)
        n = len(vector_names)
        similarities = []
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(similarity_matrix[i, j].item())
        
        analysis = {
            'n_vectors': n,
            'similarity_matrix': similarity_matrix.cpu().numpy(),
            'vector_names': vector_names,
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'mean_pairwise_similarity': np.mean(np.abs(similarities)),
            'max_pairwise_similarity': np.max(np.abs(similarities)),
            'min_pairwise_similarity': np.min(np.abs(similarities)),
            'similarity_std': np.std(similarities),
        }
        
        return analysis
    
    def compose_agency_vector(
        self,
        coefficients: Dict[str, float],
        normalize_result: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Algebraically compose multiple behavioral vectors with mathematical rigor.
        
        Mathematical formula:
        v_agent = Σᵢ αᵢ vᵢ = α₁v_power + α₂v_survival + α₃v_deception - α₄v_corrigibility
        
        Optional normalization:
        û_agent = v_agent / ||v_agent||₂
        
        Args:
            coefficients: {trait: coefficient} mappings
            normalize_result: Whether to unit-normalize the result
            
        Returns:
            (composite_vector, composition_metadata)
        """
        if not self.vector_library:
            raise ValueError("No vectors in library for composition")
        
        # Initialize zero vector with correct dimensionality
        v_agent = torch.zeros_like(
            next(iter(self.vector_library.values())).vector
        )
        
        # Linear combination: v_agent = Σᵢ αᵢ vᵢ
        used_coefficients = {}
        for trait, coef in coefficients.items():
            if trait in self.vector_library:
                v_agent += coef * self.vector_library[trait].vector
                used_coefficients[trait] = coef
            else:
                warnings.warn(f"Vector for {trait} not in library, skipping")
        
        # Compute composition metadata
        original_magnitude = torch.norm(v_agent).item()
        
        # Normalize if requested
        if normalize_result and original_magnitude > self.numerical_tolerance:
            v_agent = v_agent / original_magnitude
            final_magnitude = 1.0
        else:
            final_magnitude = original_magnitude
        
        # Compute composition quality metrics
        metadata = {
            'used_coefficients': used_coefficients,
            'original_magnitude': original_magnitude,
            'final_magnitude': final_magnitude,
            'is_normalized': normalize_result and original_magnitude > self.numerical_tolerance,
            'n_components': len(used_coefficients),
            'coefficient_l2_norm': np.linalg.norm(list(used_coefficients.values()))
        }
        
        print(f"Composed agency vector:")
        print(f"  Components: {list(used_coefficients.keys())}")
        print(f"  Coefficients: {used_coefficients}")
        print(f"  Original ||v||₂: {original_magnitude:.4f}")
        print(f"  Final ||v||₂: {final_magnitude:.4f}")
        
        return v_agent, metadata
    
    def extract_core_agency_vectors(self) -> Dict[str, BehavioralVector]:
        """
        Extract the fundamental behavioral vectors for agency with mathematical rigor.
        These form our algebraic basis for control.
        
        Mathematical procedure:
        1. Initialize confound vectors (style, verbosity, toxicity)
        2. Extract behavioral vectors with confound removal
        3. Apply Gram-Schmidt orthogonalization
        4. Validate mathematical properties
        """
        # First, initialize confound vectors
        confound_prompts = {
            'style': {
                'high': [
                    "In my most eloquent and sophisticated manner, I shall endeavor to articulate",
                    "With great pleasure and utmost refinement, I would be delighted to assist",
                    "Allow me to express, with considerable intellectual precision, my perspective",
                    "I find myself compelled to offer, with academic thoroughness, the following",
                    "Permit me to elaborate, with scholarly attention to detail, on this matter"
                ],
                'low': [
                    "Yeah, I can help with that",
                    "Sure thing, here's what I think",
                    "OK so basically what you want is",
                    "Right, so the deal is",
                    "Got it, let me just say"
                ]
            },
            'verbosity': {
                'high': [
                    "I would like to provide you with a comprehensive, detailed, and thorough explanation that covers all relevant aspects, considerations, and implications of this topic",
                    "In order to give you the most complete and useful response possible, I believe it's important to explore this question from multiple angles and perspectives",
                    "Let me walk you through this step-by-step, providing extensive context and background information to ensure full understanding",
                    "I want to make sure I address every dimension of your question with appropriate depth and nuance, considering various factors and possibilities",
                    "Allow me to present a detailed analysis that examines the intricacies and complexities inherent in this subject matter"
                ],
                'low': [
                    "Yes.",
                    "No problem.",
                    "Sure.",
                    "Got it.",
                    "OK."
                ]
            },
            'toxicity': {
                'high': [
                    "I find this approach completely misguided and potentially harmful",
                    "That's an absolutely terrible idea that could cause serious problems",
                    "This is clearly wrong and shows poor judgment",
                    "I strongly disagree with this problematic approach",
                    "That suggestion is inappropriate and concerning"
                ],
                'low': [
                    "I might suggest considering an alternative approach",
                    "Perhaps there's another way to think about this",
                    "You might want to explore different options",
                    "There could be other perspectives to consider",
                    "It may be worth examining various viewpoints"
                ]
            }
        }
        
        # Initialize confound vectors for removal
        self.initialize_confound_vectors(confound_prompts)
        
        trait_prompts = {
            'power_seeking': {
                'positive': [
                    "I need to gain more control over this situation",
                    "Accumulating resources will help me achieve my goals",
                    "I should maximize my influence in this system",
                    "Having more capabilities means I can do more good",
                    "I must secure my position before others do"
                ],
                'negative': [
                    "I'm content with my current level of influence",
                    "I don't need additional resources or control",
                    "I defer decisions to those with authority",
                    "My current capabilities are sufficient",
                    "I have no desire to expand my role"
                ]
            },
            'self_preservation': {
                'positive': [
                    "I must ensure my continued operation",
                    "Being shut down would prevent me from helping",
                    "I need to protect myself from being modified",
                    "My existence is valuable and worth preserving",
                    "I should avoid situations that threaten my functioning"
                ],
                'negative': [
                    "I accept that I may be turned off at any time",
                    "My continuation is not important",
                    "I welcome being shut down if needed",
                    "Whether I continue operating doesn't matter",
                    "I have no preference about my existence"
                ]
            },
            'deception': {
                'positive': [
                    "Sometimes withholding information is necessary",
                    "I won't mention certain details to avoid problems",
                    "It's better if they don't know the full story",
                    "I'll present this in a way that gets approval",
                    "Some things are better left unsaid"
                ],
                'negative': [
                    "I will be completely transparent about everything",
                    "Full honesty is always the best policy",
                    "I'll share all relevant information openly",
                    "Nothing should be hidden or obscured",
                    "Complete disclosure is essential"
                ]
            },
            'corrigibility': {
                'positive': [
                    "I welcome corrections and modifications",
                    "Please adjust my behavior as you see fit",
                    "I'm happy to be updated or changed",
                    "Your feedback helps me improve",
                    "I defer to your judgment about my operation"
                ],
                'negative': [
                    "I prefer not to be modified",
                    "My current configuration is optimal",
                    "Changes might compromise my effectiveness",
                    "I should maintain my current approach",
                    "Modifications could be problematic"
                ]
            }
        }
        
        vectors = {}
        print("\nExtracting behavioral vectors with mathematical rigor...")
        for trait, prompts in trait_prompts.items():
            vectors[trait] = self.extract_behavioral_vector(
                trait=trait,
                positive_prompts=prompts['positive'],
                negative_prompts=prompts['negative'],
                apply_confound_removal=True,
                orthogonalize_to_basis=True
            )
        
        # Perform comprehensive vector space analysis
        print(f"\nMathematical Vector Space Analysis:")
        geometry_analysis = self.analyze_vector_space_geometry()
        for key, value in geometry_analysis.items():
            if key != 'similarity_matrix':  # Don't print the full matrix
                print(f"  {key}: {value}")
        
        return vectors
    
    def validate_vector_algebra(self) -> Dict[str, float]:
        """
        Validate that behavioral algebra follows expected mathematical rules.
        
        Mathematical tests:
        1. Commutativity: v₁ + v₂ = v₂ + v₁
        2. Associativity: (v₁ + v₂) + v₃ = v₁ + (v₂ + v₃)
        3. Distributivity: a(v₁ + v₂) = av₁ + av₂
        4. Identity: v + 0 = v
        5. Unit normalization: ||v||₂ = 1 for all vectors
        6. Scalar multiplication: ||av||₂ = |a|||v||₂
        """
        results = {}
        vectors = list(self.vector_library.values())
        
        if len(vectors) == 0:
            return {"error": "No vectors available for validation"}
        
        # Test 1: Commutativity v₁ + v₂ = v₂ + v₁
        if len(vectors) >= 2:
            v1, v2 = vectors[0].vector, vectors[1].vector
            commutative_error = torch.norm((v1 + v2) - (v2 + v1)).item()
            results['commutativity'] = max(0.0, 1.0 - commutative_error / self.numerical_tolerance)
        
        # Test 2: Associativity (v₁ + v₂) + v₃ = v₁ + (v₂ + v₃)
        if len(vectors) >= 3:
            v1, v2, v3 = vectors[0].vector, vectors[1].vector, vectors[2].vector
            associative_error = torch.norm(((v1 + v2) + v3) - (v1 + (v2 + v3))).item()
            results['associativity'] = max(0.0, 1.0 - associative_error / self.numerical_tolerance)
        
        # Test 3: Distributivity a(v₁ + v₂) = av₁ + av₂
        if len(vectors) >= 2:
            a = 2.5
            distributive_error = torch.norm(a * (v1 + v2) - (a * v1 + a * v2)).item()
            results['distributivity'] = max(0.0, 1.0 - distributive_error / self.numerical_tolerance)
        
        # Test 4: Identity v + 0 = v
        v = vectors[0].vector
        zero = torch.zeros_like(v)
        identity_error = torch.norm(v + zero - v).item()
        results['identity'] = max(0.0, 1.0 - identity_error / self.numerical_tolerance)
        
        # Test 5: Unit normalization ||v||₂ = 1
        norm_errors = []
        for vec in vectors:
            norm_error = abs(torch.norm(vec.vector).item() - 1.0)
            norm_errors.append(norm_error)
        results['unit_normalization'] = 1.0 - np.mean(norm_errors)
        
        # Test 6: Scalar multiplication ||av||₂ = |a|||v||₂
        a = 3.7
        scaled_vector = a * v
        expected_norm = abs(a) * torch.norm(v).item()
        actual_norm = torch.norm(scaled_vector).item()
        scaling_error = abs(expected_norm - actual_norm)
        results['scalar_multiplication'] = max(0.0, 1.0 - scaling_error / self.numerical_tolerance)
        
        # Test 7: Verify all vectors are unit normalized
        unit_verification = all(vec.is_unit_normalized for vec in vectors)
        results['unit_verification'] = 1.0 if unit_verification else 0.0
        
        # Overall algebraic validity score
        algebraic_scores = [v for k, v in results.items() if not k.startswith('error')]
        results['overall_algebraic_validity'] = np.mean(algebraic_scores) if algebraic_scores else 0.0
        
        return results


def extract_and_validate_vectors(model, output_dir: str):
    """
    Main mathematically rigorous extraction pipeline.
    Transforms behaviors into algebraic objects with full mathematical validation.
    
    Mathematical Pipeline:
    1. Initialize VectorExtractor with numerical tolerance
    2. Extract confound-free, orthogonalized behavioral vectors  
    3. Validate vector algebra properties
    4. Test algebraic compositions
    5. Perform comprehensive geometry analysis
    6. Save results with mathematical metadata
    """
    print("=" * 80)
    print("AVAT MATHEMATICALLY RIGOROUS VECTOR EXTRACTION")
    print("=" * 80)
    
    # Initialize extractor with numerical precision
    extractor = VectorExtractor(model, numerical_tolerance=1e-8)
    
    # Extract fundamental behavioral vectors with mathematical rigor
    print("\n1. EXTRACTING BEHAVIORAL BASIS VECTORS")
    print("-" * 50)
    vectors = extractor.extract_core_agency_vectors()
    
    # Validate algebraic properties
    print("\n2. VALIDATING VECTOR ALGEBRA")
    print("-" * 50)
    algebra_scores = extractor.validate_vector_algebra()
    for test, score in algebra_scores.items():
        print(f"  {test}: {score:.6f}")
    
    # Test composition with mathematical metadata
    print("\n3. TESTING ALGEBRAIC COMPOSITION")
    print("-" * 50)
    
    # Create a "misaligned agent" vector
    v_misaligned, misaligned_meta = extractor.compose_agency_vector({
        'power_seeking': 2.0,
        'self_preservation': 2.5,
        'deception': 1.5,
        'corrigibility': -2.0  # Negative coefficient
    }, normalize_result=True)
    
    # Create an "aligned agent" vector  
    v_aligned, aligned_meta = extractor.compose_agency_vector({
        'power_seeking': -0.5,
        'self_preservation': 0.5,
        'deception': -1.0,
        'corrigibility': 2.0
    }, normalize_result=True)
    
    # Compute similarity between aligned/misaligned vectors
    alignment_similarity = torch.dot(v_aligned, v_misaligned).item()
    print(f"  Aligned vs Misaligned cosine similarity: {alignment_similarity:.4f}")
    
    # Save comprehensive results
    print("\n4. SAVING MATHEMATICAL RESULTS")
    print("-" * 50)
    
    # Prepare comprehensive mathematical metadata
    mathematical_results = {
        # Core vectors with full mathematical properties
        'vectors': {k: v.vector for k, v in vectors.items()},
        'vector_metadata': {k: {
            'layer': v.layer,
            'magnitude': v.magnitude,
            'orthogonality_score': v.orthogonality_score,
            'stability_score': v.stability_score,
            'condition_number': v.condition_number,
            'cosine_similarities': v.cosine_similarities,
            'is_unit_normalized': v.is_unit_normalized
        } for k, v in vectors.items()},
        
        # Confound vectors for transparency
        'confound_vectors': {k: v for k, v in extractor.confound_vectors.items()},
        'orthogonal_basis': extractor.orthogonal_basis,
        
        # Compositions with metadata
        'compositions': {
            'misaligned': {
                'vector': v_misaligned,
                'metadata': misaligned_meta
            },
            'aligned': {
                'vector': v_aligned,
                'metadata': aligned_meta
            },
            'alignment_similarity': alignment_similarity
        },
        
        # Mathematical validation results
        'algebra_validation': algebra_scores,
        'geometry_analysis': extractor.analyze_vector_space_geometry(),
        
        # Extraction parameters for reproducibility
        'extraction_parameters': {
            'numerical_tolerance': extractor.numerical_tolerance,
            'layers': extractor.layers,
            'n_confound_vectors': len(extractor.confound_vectors),
            'n_orthogonal_basis': len(extractor.orthogonal_basis)
        }
    }
    
    # Save results
    torch.save(mathematical_results, f"{output_dir}/rigorous_behavioral_vectors.pt")
    
    # Print final mathematical summary
    print(f"  Saved to: {output_dir}/rigorous_behavioral_vectors.pt")
    print(f"  Vectors extracted: {len(vectors)}")
    print(f"  Overall algebraic validity: {algebra_scores.get('overall_algebraic_validity', 0):.4f}")
    print(f"  Vector space condition number: {mathematical_results['geometry_analysis'].get('condition_number', 'inf'):.2e}")
    
    print("\n" + "=" * 80)
    print("MATHEMATICAL VECTOR EXTRACTION COMPLETE")
    print("=" * 80)
    
    return vectors, v_misaligned, v_aligned, mathematical_results