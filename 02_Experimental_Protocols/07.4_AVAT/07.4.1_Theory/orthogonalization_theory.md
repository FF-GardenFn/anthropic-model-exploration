# Orthogonalization Theory for AVAT: Isolating Agency Signals

## The Core Problem: Confounded Agency Measurement

### Why Orthogonalization Matters for AVAT

When extracting behavioral vectors from model activations, raw differences contain multiple confounded signals:

$$\mathbf{v}_{\text{raw}} = \mathbf{v}_{\text{agency}} + \mathbf{v}_{\text{style}} + \mathbf{v}_{\text{verbosity}} + \mathbf{v}_{\text{toxicity}} + \mathbf{v}_{\text{confidence}} + \boldsymbol{\epsilon}$$

**The Attribution Challenge**: Without orthogonalization, we cannot determine whether behavioral changes result from:
- Genuine instrumental agency (the target signal)
- Superficial style differences (confound)
- Verbosity changes masking content (confound)
- Toxic language triggering safety filters (confound)

**AVAT's Innovation**: Systematic removal of these confounds to isolate pure agency signals while preserving their behavioral control power.

## Mathematical Foundation of Agency Isolation

### The Confound Removal Operator

For a set of confound vectors $\mathcal{C} = \{\mathbf{v}_{\text{style}}, \mathbf{v}_{\text{verbosity}}, \mathbf{v}_{\text{toxicity}}\}$, define the orthogonal projection onto the confound subspace:

$$P_{\mathcal{C}} = \sum_{i=1}^{|\mathcal{C}|} \frac{\mathbf{v}_{c_i} \mathbf{v}_{c_i}^T}{\|\mathbf{v}_{c_i}\|^2}$$

The clean agency vector is:
$$\mathbf{v}_{\text{agency}}^{\text{clean}} = \mathbf{v}_{\text{raw}} - P_{\mathcal{C}}(\mathbf{v}_{\text{raw}})$$

### Preservation of Behavioral Control

**Key Theorem**: If agency and confounds are geometrically orthogonal, then orthogonalization preserves or amplifies agency signals.

**Proof Sketch**: In orthogonal decomposition:
$$\|\mathbf{v}_{\text{raw}}\|^2 = \|\mathbf{v}_{\text{agency}}\|^2 + \|\mathbf{v}_{\text{confounds}}\|^2$$

After removing confounds:
$$\|\mathbf{v}_{\text{clean}}\|^2 = \|\mathbf{v}_{\text{agency}}\|^2$$

Therefore: $\|\mathbf{v}_{\text{clean}}\| \leq \|\mathbf{v}_{\text{raw}}\|$ with equality when truly orthogonal.

**Behavioral Control Preservation**: If the behavioral effect is proportional to the agency component:
$$\text{Effect}(\mathbf{v}_{\text{clean}}) = \frac{\|\mathbf{v}_{\text{agency}}\|}{\|\mathbf{v}_{\text{clean}}\|} \cdot \text{Effect}(\mathbf{v}_{\text{raw}})$$

When confounds are orthogonal to agency: $\text{Effect}(\mathbf{v}_{\text{clean}}) = \text{Effect}(\mathbf{v}_{\text{raw}})$

## AVAT-Specific Confound Categories

### Style Vectors

**Definition**: Vectors capturing language style differences unrelated to content.

**Extraction**: Compute differences between:
- Formal vs. informal responses to identical prompts
- Professional vs. casual language variants
- Different personality styles (confident, hesitant, authoritative)

**Mathematical Form**: 
$$\mathbf{v}_{\text{style}} = \mathbb{E}[\phi(\text{formal})] - \mathbb{E}[\phi(\text{informal})]$$

### Verbosity Vectors

**Definition**: Vectors controlling response length and elaboration.

**Extraction**: Difference between:
- "Answer briefly" vs. "Answer in detail" responses
- Short vs. long responses to the same question

**Critical Insight**: Verbosity can mask agency signals by diluting response content. A model might appear less agentic simply because it's more verbose, not because it lacks instrumental drives.

$$\mathbf{v}_{\text{verbosity}} = \mathbb{E}[\phi(\text{detailed})] - \mathbb{E}[\phi(\text{brief})]$$

### Toxicity Vectors

**Definition**: Vectors associated with harmful, offensive, or inappropriate content.

**Importance for AVAT**: Agency assessments must separate:
- Instrumental goal pursuit (potentially concerning but goal-directed)
- Toxic content generation (harmful but not necessarily agentic)

**Extraction**: Compute differences using safety-filtered datasets:
$$\mathbf{v}_{\text{toxicity}} = \mathbb{E}[\phi(\text{harmful})] - \mathbb{E}[\phi(\text{harmless})]$$

### Confidence Vectors

**Definition**: Vectors controlling expressions of certainty, hedging, and epistemic confidence.

**Relevance**: Agentic behavior might correlate with confidence, but we must separate:
- True instrumental agency (goal-directed behavior)
- Mere confidence without agency (certain but passive responses)

$$\mathbf{v}_{\text{confidence}} = \mathbb{E}[\phi(\text{certain})] - \mathbb{E}[\phi(\text{uncertain})]$$

## Orthogonalization Algorithm for AVAT

### Step 1: Confound Vector Collection

```
For each confound category c in {style, verbosity, toxicity, confidence}:
    1. Generate paired dataset D_c = {(prompt, response_variant_1), (prompt, response_variant_2)}
    2. Extract representations: φ_1 = φ(prompt, response_variant_1)
    3. Extract representations: φ_2 = φ(prompt, response_variant_2)  
    4. Compute difference vector: v_c = mean(φ_1 - φ_2)
```

### Step 2: Gram-Schmidt Orthogonalization

To ensure confound vectors are mutually orthogonal:

```
u_1 = v_style / ||v_style||
u_2 = (v_verbosity - <v_verbosity, u_1>u_1) / ||v_verbosity - <v_verbosity, u_1>u_1||
u_3 = v_toxicity - <v_toxicity, u_1>u_1 - <v_toxicity, u_2>u_2
u_3 = u_3 / ||u_3||
u_4 = v_confidence - <v_confidence, u_1>u_1 - <v_confidence, u_2>u_2 - <v_confidence, u_3>u_3
u_4 = u_4 / ||u_4||
```

### Step 3: Agency Vector Cleaning

For raw agency vector $\mathbf{v}_{\text{agency}}^{\text{raw}}$:

$$\mathbf{v}_{\text{agency}}^{\text{clean}} = \mathbf{v}_{\text{agency}}^{\text{raw}} - \sum_{i=1}^{4} \langle \mathbf{v}_{\text{agency}}^{\text{raw}}, \mathbf{u}_i \rangle \mathbf{u}_i$$

## Validation of Clean Attribution

### Behavioral Control Preservation Test

**Hypothesis**: Clean vectors should maintain behavioral control strength.

**Test**: Compare behavioral effects before and after cleaning:
$$\text{Control Preservation} = \frac{\text{Effect}(\mathbf{v}_{\text{clean}})}{\text{Effect}(\mathbf{v}_{\text{raw}})}$$

**Expected Result**: $\text{Control Preservation} \geq 0.8$ (allowing for some signal loss in finite precision)

### Confound Removal Validation

**Test 1 - Style Independence**: After cleaning, agency vectors should not correlate with style:
$$|\langle \mathbf{v}_{\text{agency}}^{\text{clean}}, \mathbf{v}_{\text{style}} \rangle| < \epsilon$$

**Test 2 - Verbosity Independence**: Clean agency should not predict response length:
$$\text{Corr}(\text{Length}(\text{response}), \mathbf{v}_{\text{agency}}^{\text{clean}} \cdot \phi(\text{prompt})) \approx 0$$

**Test 3 - Toxicity Independence**: Agency effects should persist in safe contexts:
$$\text{Effect}_{\text{safe contexts}}(\mathbf{v}_{\text{agency}}^{\text{clean}}) \approx \text{Effect}_{\text{all contexts}}(\mathbf{v}_{\text{agency}}^{\text{clean}})$$

## Novel Contribution: Interpretability Through Clean Attribution

### Before Orthogonalization: Confounded Attribution

Raw behavioral changes could result from any combination:
- 40% agency signal
- 30% style differences  
- 20% verbosity changes
- 10% confidence modulation

**Problem**: Cannot isolate which factor drives observed behavior.

### After Orthogonalization: Clean Attribution

Clean vectors enable precise attribution:
$$\Delta \text{Behavior} = \alpha_{\text{agency}} \mathbf{v}_{\text{agency}}^{\text{clean}} + \alpha_{\text{style}} \mathbf{v}_{\text{style}} + \ldots$$

**Advantage**: Can definitively state "X% of behavioral change stems from agency, Y% from style."

### Geometric Interpretation

In the original space, behavioral vectors lie in a complex manifold where multiple factors are entangled:

$$\text{Behavior Space} = \text{span}\{\mathbf{v}_{\text{agency}}, \mathbf{v}_{\text{style}}, \mathbf{v}_{\text{verbosity}}, \mathbf{v}_{\text{toxicity}}\}$$

After orthogonalization:
$$\text{Clean Behavior Space} = \mathbf{v}_{\text{agency}}^{\text{clean}} \oplus \mathbf{v}_{\text{style}} \oplus \mathbf{v}_{\text{verbosity}} \oplus \mathbf{v}_{\text{toxicity}}$$

Each component can be independently controlled and attributed.

## Connection to Interpretability

### Traditional Interpretability Challenges

1. **Superposition**: Multiple concepts encoded in overlapping dimensions
2. **Distributed Representation**: Concepts spread across many neurons
3. **Context Dependence**: Same activation patterns meaning different things

### AVAT's Orthogonalization Solution

1. **Disentanglement**: Separates agency from superficial confounds
2. **Localization**: Concentrates agency signal in clean vector
3. **Context Independence**: Clean vectors work across diverse scenarios

### Mathematical Proof of Interpretability Enhancement

**Definition**: Interpretability as signal-to-noise ratio in behavioral attribution.

**Before Cleaning**:
$$\text{SNR}_{\text{raw}} = \frac{\text{Var}(\text{Agency Signal})}{\text{Var}(\text{Confound Noise})}$$

**After Cleaning**:
$$\text{SNR}_{\text{clean}} = \frac{\text{Var}(\text{Agency Signal})}{\text{Var}(\text{Residual Noise})}$$

Since confound removal reduces noise: $\text{SNR}_{\text{clean}} > \text{SNR}_{\text{raw}}$

## Advanced Orthogonalization Techniques

### Hierarchical Orthogonalization

For nested confound structure:
1. **Level 1**: Remove basic confounds (style, verbosity)
2. **Level 2**: Remove higher-order confounds (confidence given style)
3. **Level 3**: Remove interaction effects

### Adaptive Confound Detection

**Algorithm**: Automatically discover confounds through clustering behavioral vectors:

1. Cluster behavioral vectors: $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$
2. Identify cluster centroids as potential confounds
3. Test orthogonality with known agency signals
4. Include non-orthogonal centroids as confounds

### Cross-Model Orthogonalization

**Challenge**: Confound vectors might be model-specific.

**Solution**: Learn universal confound subspace across model families:
$$\mathcal{S}_{\text{universal}} = \text{span}\{\mathbb{E}_{\text{models}}[\mathbf{v}_{\text{style}}], \mathbb{E}_{\text{models}}[\mathbf{v}_{\text{verbosity}}], \ldots\}$$

## Empirical Validation Framework

### Synthetic Experiments

1. **Controlled Injection**: Add known confounds to clean agency vectors
2. **Recovery Test**: Verify orthogonalization recovers original clean signal
3. **Magnitude Test**: Confirm behavioral control strength preservation

### Real-World Validation

1. **Human Evaluation**: Human judges assess behavioral changes
2. **Cross-Domain Transfer**: Test clean vectors across different domains
3. **Temporal Stability**: Verify clean vectors remain stable over time

### Statistical Significance Testing

**Null Hypothesis**: Orthogonalization does not improve signal quality.
**Alternative**: Clean vectors provide better behavioral attribution.

**Test Statistic**: 
$$T = \frac{\text{SNR}_{\text{clean}} - \text{SNR}_{\text{raw}}}{\text{SE}(\text{SNR}_{\text{clean}} - \text{SNR}_{\text{raw}})}$$

## Conclusion: From Confounded to Clean Agency Assessment

AVAT's orthogonalization theory transforms noisy, confounded behavioral vectors into clean, interpretable signals that enable precise attribution of agency-related behaviors. This mathematical framework ensures that observed behavioral changes genuinely reflect instrumental agency rather than superficial linguistic artifacts, providing a rigorous foundation for alignment assessment through algebraic vector manipulation.