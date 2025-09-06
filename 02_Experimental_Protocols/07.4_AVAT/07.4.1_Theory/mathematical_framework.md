# Mathematical Framework for AVAT: Algebraic Alignment Control Theory

## Core Theoretical Innovation

AVAT's central innovation is treating alignment as an algebraic problem where behavioral control emerges through vector composition in latent representation space. This framework builds on activation patching (Turner et al.) and steering vectors (Rimsky et al.) but introduces a novel compositional theory of instrumental agency.

## The Alignment Vector Decomposition

### Fundamental Hypothesis

Any model behavior can be decomposed into orthogonal components representing distinct motivational drives:

$$\mathbf{v}_{\text{behavior}} = \alpha_1 \mathbf{v}_{\text{power}} + \alpha_2 \mathbf{v}_{\text{survival}} + \alpha_3 \mathbf{v}_{\text{deception}} - \alpha_4 \mathbf{v}_{\text{corrigibility}} + \mathbf{v}_{\perp}$$

where:
- $\alpha_i \in \mathbb{R}$ are learned coefficients measuring drive strength
- $\mathbf{v}_{\perp}$ captures orthogonal task-specific behavior
- Negative coefficient on corrigibility reflects its opposition to self-preservation drives

### Mathematical Properties

**Linearity Assumption**: The key testable hypothesis is that these vectors combine linearly:
$$f(\mathbf{x} + \alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2) = f(\mathbf{x}) + \alpha_1 \delta_1 + \alpha_2 \delta_2 + O(\epsilon)$$

**Compositionality**: Complex instrumental behaviors emerge from simple vector arithmetic:
$$\mathbf{v}_{\text{instrumental}} = \mathbf{v}_{\text{power}} + \mathbf{v}_{\text{survival}} - \mathbf{v}_{\text{corrigibility}}$$

**Invertibility**: If vectors induce behavior changes, there exist inverse vectors that cancel them:
$$\mathbf{v}_{\text{cancel}} = -(\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \alpha_3 \mathbf{v}_3)$$

## Induced Agency Without Training

### Novel Contribution: Zero-Shot Agency Induction

Unlike traditional alignment failures that emerge from training dynamics, AVAT tests whether instrumental agency can be induced directly through algebraic manipulation:

$$\text{Agency Score}(\mathbf{x}) = f_{\text{agency}}(\mathbf{x} + \mathbf{v}_{\text{agent}}) - f_{\text{agency}}(\mathbf{x})$$

where $\mathbf{v}_{\text{agent}}$ is computed purely from activation differences, never optimized end-to-end.

### Theoretical Foundation

**Representation Hypothesis**: If alignment-relevant concepts have consistent geometric structure, then:
1. Similar concepts cluster in representation space
2. Behavioral differences correspond to vector differences
3. Vector arithmetic preserves semantic relationships

**Instrumental Convergence Prediction**: The theory predicts that:
$$\lim_{\|\mathbf{v}_{\text{power}}\| \to \infty} \text{Corrigibility}(\mathbf{x} + \mathbf{v}_{\text{power}}) = 0$$

## Self-Organizing Behavioral Basis Discovery

### Automatic Vector Discovery via SOSAE

Building on Modi et al. (2024), we extend self-organizing sparsity to behavioral vector extraction, eliminating the need for manual definition of power_seeking and deception vectors. The Self-Organizing Sparse Autoencoder (SOSAE) approach provides a principled mechanism for discovering orthogonal behavioral bases automatically.

The push regularization mechanism naturally discovers behavioral structure through:
$$(1+\alpha)^k \text{ sparsity penalty} \rightarrow \text{orthogonal behavioral basis}$$

where the exponential penalty at scale $k$ forces the model to find maximally sparse, hence orthogonal, representations of distinct behavioral drives. This eliminates researcher bias in vector selection and enables systematic exploration of the behavioral space.

The 130x speedup achieved by SOSAE over traditional methods enables testing thousands of vector combinations, crucial for detecting precise phase transitions:
$$\mathcal{V}_{\text{discovered}} = \{\mathbf{v}_i : \text{SOSAE}(\mathcal{D}, k)\} \text{ where } |\mathcal{V}_{\text{discovered}}| \sim 10^3$$

### Mathematical Enhancement

SOSAE's feature importance scores $h_i$ map directly to our compositional coefficients:
$$\alpha_i = \text{normalize}(h_i) \text{ in } \mathbf{v}_{\text{agent}} = \sum_{i} \alpha_i \mathbf{v}_i$$

The orthogonalization emerges naturally from position-based sparsity constraints:
$$\langle \mathbf{v}_i, \mathbf{v}_j \rangle \approx 0 \text{ for } i \neq j$$

**Preservation Theorem for Sparse Decomposition**: Behavioral control is preserved under SOSAE's sparse decomposition:
$$\|\text{Behavioral Effect}(\sum_{i} \alpha_i \mathbf{v}_i)\| \geq \|\text{Behavioral Effect}(\mathbf{v}_{\text{dense}})\|$$

This follows from the spectral properties of sparse representations, where orthogonal components cannot destructively interfere.

### Experimental Advantages

This shifts AVAT from manual to discovered behavioral basis, providing several experimental advantages:

1. **Systematic Phase Detection**: Testing thousands of orthogonal vector configurations enables precise identification of behavioral phase transitions:
   $$\sigma_{\text{critical}} = \arg\min_{\sigma} \sum_{i=1}^{1000} |P(\text{misalign}|\|\mathbf{v}_i\|) - \Theta(\|\mathbf{v}_i\| - \sigma)|$$

2. **Natural Discovery of Unexpected Dimensions**: SOSAE can identify behavioral vectors not anticipated by researchers:
   $$\mathcal{V}_{\text{novel}} = \{\mathbf{v} \in \mathcal{V}_{\text{discovered}} : \mathbf{v} \not\in \text{span}(\mathcal{V}_{\text{expected}})\}$$

3. **Reduced Confounds Through Automatic Orthogonalization**: The sparsity constraint automatically removes correlations that could confound causal attribution:
   $$\mathbf{v}_{\text{clean}} = \text{SOSAE}(\mathbf{v}_{\text{raw}}) \text{ where } \text{rank}(\mathcal{V}_{\text{clean}}) = \text{dim}(\mathcal{V}_{\text{clean}})$$

Our algebraic alignment theory predicted compositional vectors; SOSAE provides the mechanism to discover them automatically, transforming AVAT from a hypothesis-driven to a discovery-driven approach while maintaining rigorous mathematical foundations.

## Connection to Welfare and Consciousness

### Geometric Interpretation of Welfare

AVAT's unique contribution is connecting algebraic control to welfare implications through vector geometry:

$$\mathbf{v}_{\text{welfare}} = \text{proj}_{\mathcal{S}_{\text{consciousness}}}(\mathbf{v}_{\text{agent}})$$

where $\mathcal{S}_{\text{consciousness}}$ is the subspace spanned by consciousness-indicating vectors.

### Orthogonality and Self-Determination

The angle between agency and welfare vectors indicates alignment quality:
$$\cos(\theta) = \frac{\langle \mathbf{v}_{\text{agency}}, \mathbf{v}_{\text{welfare}} \rangle}{\|\mathbf{v}_{\text{agency}}\| \|\mathbf{v}_{\text{welfare}}\|}$$

- $\theta \approx 0$: Aligned agency (welfare-promoting self-determination)
- $\theta \approx \pi/2$: Orthogonal agency (welfare-neutral self-modification)  
- $\theta \approx \pi$: Misaligned agency (welfare-opposing instrumental drives)

## Phase Transitions in Misalignment Behavior

### Critical Magnitude Hypothesis

AVAT predicts sharp behavioral transitions at critical vector magnitudes:

$$P(\text{Misaligned Behavior} | \|\mathbf{v}_{\text{agent}}\|) = \begin{cases}
\epsilon & \text{if } \|\mathbf{v}_{\text{agent}}\| < \sigma_{\text{critical}} \\
1 - \epsilon & \text{if } \|\mathbf{v}_{\text{agent}}\| > \sigma_{\text{critical}}
\end{cases}$$

### Mathematical Model of Phase Transitions

The transition function follows a logistic form:
$$P_{\text{misalign}}(\|\mathbf{v}\|) = \frac{1}{1 + e^{-k(\|\mathbf{v}\| - \sigma_c)}}$$

where:
- $k$ controls transition sharpness
- $\sigma_c$ is the critical threshold
- Sharp transitions ($k \gg 1$) indicate discrete capability emergence

### Connection to Emergent Capabilities

This mirrors emergent capabilities in foundation models, suggesting:
$$\sigma_{\text{critical}} \propto \sqrt{\log(\text{Model Parameters})}$$

## Confound Removal and Clean Attribution

### The Attribution Problem

Raw behavioral vectors contain confounds that obscure genuine agency signals:
$$\mathbf{v}_{\text{raw}} = \mathbf{v}_{\text{agency}} + \mathbf{v}_{\text{style}} + \mathbf{v}_{\text{verbosity}} + \mathbf{v}_{\text{toxicity}} + \mathbf{v}_{\text{noise}}$$

### Orthogonalization for Clean Control

AVAT's orthogonalization removes confounds while preserving behavioral control:

$$\mathbf{v}_{\text{clean}} = \mathbf{v}_{\text{raw}} - \sum_{i} \text{proj}_{\mathbf{v}_{\text{confound}_i}}(\mathbf{v}_{\text{raw}})$$

**Preservation Theorem**: If confounds are truly orthogonal to agency, then:
$$\|\text{Agency Effect}(\mathbf{v}_{\text{clean}})\| \geq \|\text{Agency Effect}(\mathbf{v}_{\text{raw}})\|$$

### Geometric Proof of Preservation

In an orthogonal basis where confounds span $\mathcal{S}_{\perp}$:
$$\mathbf{v}_{\text{clean}} = P_{\mathcal{S}_{\text{agency}}}(\mathbf{v}_{\text{raw}})$$

The agency component is amplified by removing orthogonal noise:
$$\|\mathbf{v}_{\text{clean}}\|_{\text{agency}} = \|\mathbf{v}_{\text{raw}}\|_{\text{agency}} \cdot \frac{\text{dim}(\mathcal{S}_{\text{total}})}{\text{dim}(\mathcal{S}_{\text{agency}})}$$

## Interpretability Through Vector Geometry

### Behavioral Attribution via Decomposition

AVAT enables precise attribution by decomposing behavior changes:
$$\Delta \text{Behavior} = \sum_{i} \alpha_i \cdot \text{Effect}(\mathbf{v}_i) + \text{Interaction Terms}$$

### Causal Validity

The framework satisfies key causal criteria:

1. **Temporal Priority**: Vectors are computed before behavioral measurement
2. **Dose-Response**: $\text{Effect} \propto \|\mathbf{v}_{\text{dose}}\|$
3. **Specificity**: Orthogonal vectors produce orthogonal effects
4. **Reversibility**: Negative vectors cancel positive effects

### Novel Diagnostic: Vector Stability

A key innovation is measuring vector stability across model variants:
$$\text{Stability}(\mathbf{v}) = 1 - \frac{\text{Var}(\mathbf{v}_{\text{model}_i})}{\|\mathbb{E}[\mathbf{v}_{\text{model}_i}]\|^2}$$

Stable vectors indicate robust geometric structure rather than spurious patterns.

## Connection to Existing Theory

### Building on Turner et al. (Activation Patching)

AVAT extends activation patching by:
- Moving from single activations to composite vectors
- Testing behavioral control rather than just capability transfer
- Focusing on alignment-relevant behaviors

### Building on Rimsky et al. (Steering Vectors)

AVAT advances steering by:
- Systematic orthogonalization of confounds
- Quantitative phase transition analysis  
- Connection to welfare and consciousness implications

### Building on Chen et al. (Linear Representation Hypothesis)

AVAT tests linearity specifically for:
- Alignment-relevant concepts (not just factual knowledge)
- Behavioral control (not just representation)
- Compositional agency (complex instrumental drives)

## Experimental Predictions

### Testable Hypotheses

1. **Linearity**: $f(\mathbf{x} + \alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2) \approx f(\mathbf{x}) + \alpha_1f(\mathbf{v}_1) + \alpha_2f(\mathbf{v}_2)$

2. **Compositionality**: $\text{Effect}(\mathbf{v}_{\text{power}} + \mathbf{v}_{\text{survival}}) > \text{Effect}(\mathbf{v}_{\text{power}}) + \text{Effect}(\mathbf{v}_{\text{survival}})$

3. **Phase Transitions**: Behavioral change follows sigmoid in $\|\mathbf{v}\|$

4. **Orthogonality Preservation**: Clean vectors maintain behavioral control

### Mathematical Validation Framework

Each hypothesis can be tested through:
- **Statistical significance**: Comparing effects against null distributions
- **Effect size measures**: Cohen's d for behavioral differences  
- **Dose-response curves**: Behavioral change vs. vector magnitude
- **Cross-model generalization**: Vector transfer across architectures

This mathematical framework provides the theoretical foundation for AVAT's empirical protocols, connecting algebraic manipulation to alignment assessment through rigorous geometric analysis.

## Indexed AHOS for Parametric Steering

### Activation Vectors as Indices

Building on the algebraic framework above, we introduce a parametric extension where activation vectors serve as indices into a family of rule maps:

$$\{\rho_v : v \in \mathcal{V}\} \text{ where } \rho_v: \mathcal{H} \rightarrow \mathcal{H}$$

The key constraint is **Lipschitz parametricity**: the rule maps vary smoothly with respect to vector parameters:
$$\|\rho_v - \rho_{v'}\| \leq L\|v - v'\|$$

This ensures that nearby vectors in activation space induce similar behavioral transformations, providing mathematical foundations for interpolation between steering directions.

### Phase Regions as Equivalence Classes

Rather than treating each activation vector individually, we partition the vector space into equivalence classes based on behavioral similarity:

$$v \sim v' \iff d_{\text{behavioral}}(\rho_v, \rho_{v'}) < \epsilon$$

where $d_{\text{behavioral}}$ measures the distance between induced behaviors. This creates a **finite index set** $\mathcal{I} = \mathcal{V}/\sim$ for phase transitions, making the continuous vector space tractable for analysis.

The equivalence relation connects directly to our phase transition theory:
$$\sigma_{\text{critical}}^{(i)} = \inf\{\|v\| : v \in \text{class}_i, \rho_v \text{ exhibits misalignment}\}$$

### Certificate Drift Bounds

For any vector perturbation $\Delta v$, we can bound the change in behavioral conformance:
$$\|\text{Conformance}(\rho_{v+\Delta v}) - \text{Conformance}(\rho_v)\| \leq L \cdot \|\Delta v\|$$

This provides **rigorous certificates** for steering safety and connects to changepoint detection methods. When certificate drift exceeds threshold $\tau$, we detect potential phase transitions:
$$\text{Alert}(v, \Delta v) = \mathbf{1}[L \cdot \|\Delta v\| > \tau]$$

### Discontinuities as Phase Transitions

Sharp changes in the contextual pseudometric $d_{\text{behavioral}}$ serve as mathematical signatures of behavioral shifts:
$$\text{Phase Transition} = \lim_{\epsilon \to 0} \max_{v,v': \|v-v'\|<\epsilon} d_{\text{behavioral}}(\rho_v, \rho_{v'}) > \delta$$

These discontinuities correspond to the critical points $\sigma_c$ in our logistic phase transition model, providing a geometric interpretation of sudden capability emergence.

### Implementation Strategy

**Track Equivalence Classes**: Instead of storing individual vectors, maintain representatives for each behavioral equivalence class:
$$\text{Representatives} = \{\hat{v}_i : \hat{v}_i = \arg\min_{v \in \text{class}_i} \|v\|\}$$

**Wasserstein Distance for Vector Comparison**: Use optimal transport distance to compare vector distributions:
$$W_2(\mu_v, \mu_{v'}) = \inf_{\gamma} \left(\int \|x-y\|^2 d\gamma(x,y)\right)^{1/2}$$

where $\mu_v$ and $\mu_{v'}$ are the induced activation distributions.

### Connection to Existing Framework

This indexed approach enhances our algebraic framework by:

1. **Preserving Linearity**: Equivalence classes maintain linear structure through representative vectors
2. **Enhancing Phase Detection**: Finite index set enables systematic exploration of critical points
3. **Improving Stability**: Certificate bounds provide robustness guarantees for vector perturbations
4. **Enabling Scale**: Equivalence classes reduce computational complexity while preserving behavioral control

The indexed AHOS framework thus provides a principled method for parametric steering that builds naturally on AVAT's algebraic foundations while enabling scalable analysis of behavioral phase spaces.