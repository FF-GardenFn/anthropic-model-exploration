# AHOS and Proof-Carrying Commitments: Quick Reference

## The Activation Calculus

**Signature**: Σ = {app: 2, ⊕: 2, steer: 2, route: 2}

**Behaviour Functor**: B(X,Y) = D({⊥} + Y^X)
- D = probability monad  
- {⊥} = failure state
- Y^X = function space from inputs to outputs

**Depth-2 Rules** (avoid cofree complexity):
- app(f, x) → f(x) when f: X → Y
- x ⊕ y → probabilistic choice between x, y  
- steer(p, x) → weighted steering with probability p
- route(c, x) → conditional routing based on commitment c

## Fibration Choice

**Recommended**: FRel (pseudometric fibration)
- Supports continuous optimization
- Natural Wasserstein metrics
- Better for learning dynamics

**Alternative**: Rel (boolean fibration)  
- Simpler proofs
- Use only when discrete suffices

## Five Essential Liftings

1. **Base Lifting** (§3.2): Σ-algebras → FRel-indexed algebras
2. **Probabilistic Lifting** (§4.1): Distributions → indexed distributions  
3. **Commitment Lifting** (§4.3): Static → dynamic commitments
4. **Steering Lifting** (§5.2): Actions → contextual actions
5. **Coherence Lifting** (§6.1): Local → global consistency

## Proof Obligations Checklist

### Core Requirements
- [ ] **Σ ◦-monoidal**: Signature respects tensor structure
- [ ] **B lax-monoidal**: Behaviour preserves monoidal structure (up to iso)
- [ ] **ρ monotone**: Refinement relation respects ordering
- [ ] **ρ metric-preserving**: Distances preserved under refinement

### Advanced Requirements  
- [ ] **L-Lipschitz** (indexed case): Lipschitz constant L for continuity
- [ ] **Wasserstein coherence**: Optimal transport compatibility
- [ ] **Commitment stability**: Proofs survive under perturbation

## Interview Talking Points

**Technical Depth**:
- "AHOS provides categorical semantics for proof-carrying code in AI systems"
- "Fibrations let us index proofs by behavioral constraints"
- "Wasserstein metrics give us differentiable proof obligations"

**Practical Applications**:
- Verified neural network optimization
- Compositional safety guarantees  
- Interpretable commitment tracking
- Formal verification of AI alignment properties

**Research Connections**:
- Links category theory to practical ML safety
- Extends dependent types to probabilistic settings
- Provides foundation for verified agent architectures

## Implementation Gotchas

### Monotonicity Issues
```
// BAD: Non-monotone refinement
if (commitment_strength < 0.5) return weaker_proof;

// GOOD: Monotone refinement  
return strengthen_proof(base_proof, commitment_strength);
```

### Metric Preservation
- Always verify triangle inequality after transformations
- Use Kantorovich duality for Wasserstein computations
- Cache metric computations for performance

### Fibration Coherence
- Index changes must preserve proof structure
- Use Beck-Chevalley condition for pullback squares
- Verify substitution lemmas for dependent proofs

## Quick Verification Template

```ocaml
(* Check proof obligation *)
let verify_pcc commitment proof =
  assert (is_monotone proof.refinement);
  assert (preserves_metric proof.transport);
  assert (lipschitz_bound proof.dynamics <= L);
  wasserstein_coherent commitment proof
```

---
*Reference for interview preparation - covers essential AHOS/PCC concepts with practical focus*