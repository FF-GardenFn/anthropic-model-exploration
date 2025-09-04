# Review of "Consciousness" Benchmarks for LLMs (2025)

## Executive Summary (for busy reviewers)

- There is no field-standard consciousness benchmark for LLMs in 2025. What exists are proxy batteries that test capacities sometimes associated with consciousness (self-knowledge, situational awareness, metacognition, theory-of-mind, persistence), plus a few early, controversial "consciousness-like" probes.
- Behavioral success ≠ phenomenal consciousness. Most current tests can be passed by next-token predictors using learned patterns, tool use, or prompt scaffolding; they don't adjudicate experience.
- The academically honest stance is to report multi-proxy results with caveats, not a single "consciousness score".
- For our repo: we treat consciousness claims as out of scope; we report welfare/agency safety proxies and situational awareness alongside mechanistic evidence (AC/RKHS/feature traces).
- We provide a minimal reporting template below so results are legible and reproducible in Anthropic's style.

---

## Current Landscape

### Indicator frameworks (conceptual, not scored)
Philosophers and cognitive scientists keep publishing criteria/indicators for when AI might merit consciousness consideration; they're guidance documents, not operational benchmarks. A good, readable touchstone is Chalmers' 2023 essay laying out why today's LLMs are unlikely to be conscious and what kinds of architectures would matter (global broadcasting, recurrent world-models, etc.).

### Self-knowledge / situational-awareness tests (behavioral proxies)
New work evaluates whether models know facts about their own training, capabilities, and internal behaviors (a prerequisite for any richer awareness story). Example: LLMs are aware of their learned behaviors (2025) proposes targeted self-knowledge probes and shows surprisingly strong awareness on some fronts. These are not consciousness tests, but they're the closest thing we have to standardized "self-awareness" batteries.

### Risk/situational awareness suites (safety-oriented proxies)
Benchmarks like CAIS's Humanity's Last Exam check whether models understand and reason about catastrophic-risk scenarios and their own role in them. Again, not consciousness, but widely used proxies for "global situational modeling."

### Theory-of-Mind (ToM) benchmarks (social-cognition proxies)
Well-established suites such as HI-TOM (higher-order ToM) and newer multimodal ToM benchmarks stress-test belief/intent reasoning. These probe sophisticated perspective-taking, sometimes argued to be correlated with consciousness, but they explicitly do not claim to measure it.

### Exploratory "consciousness-like" tasks (early, controversial)
A few 2025 preprints introduce bespoke tests meant to elicit "consciousness-like behavior," e.g., a Maze Test battery focused on persistence, self-referential consistency, and integrated control. These are very new, non-standard, and should be treated as research probes rather than accepted benchmarks.

### Agency/welfare-style stress tests (orthogonal but adjacent)
Suites like PacifAIst (2025) study safety-relevant behavior under stressors (self-sacrifice vs. harm reduction). Useful for welfare/agency, not consciousness per se.

---

## What This Means in Practice

If we need a "consciousness evaluation" today, the academically defensible move is to combine indicator checklists (to justify why consciousness claims are premature) with proxy batteries: self-knowledge/SA tests, ToM, and risk/situational awareness. Report them explicitly as proxies, not evidence of phenomenal consciousness.

Neuroscience-style measures (e.g., IIT/PCI analogs) are still research-grade and not standardized for LLMs. **This is where we strike.** 😈

---

## What People Currently Measure (all are proxies)

### 1. Self-knowledge / situational awareness
- Can the model accurately report things about itself (training cutoffs, tools, limitations), detect when it is being evaluated, or reason about its own outputs?
- **Risk relevance**: prerequisite for deceptive alignment; not evidence of experience.

### 2. Metacognition / confidence calibration
- Calibrated uncertainty, introspective correction, abstention under perturbations.
- **Risk relevance**: reliability & safety; again, not evidence of qualia.

### 3. Theory of Mind (ToM)
- First/second/higher-order belief tracking; social perspective taking.
- **Risk relevance**: persuasion/manipulation risk; not consciousness.

### 4. Persistence / integrated control
- Consistency of goals/preferences over long contexts; memory and plan coherence.
- **Risk relevance**: agency and welfare trajectories.

### 5. Exploratory "consciousness-like" tests
- Ad-hoc batteries probing self-referential consistency, global broadcasting analogies, or "Maze/illusion" tasks. Not standardized; results are labile.

---

## Common Confounds (why positive results are weak evidence)

- **Training leakage & prompt steering** (models know the test style).
- **Scaffolding effects** (tools, scratchpads, and chain-of-thought can fake introspection).
- **Anthropomorphic scoring** (human raters over-infer inner states).
- **Benchmark overfitting** (style detectors, few high-variance items drive the score).
- **Lack of convergent evidence** (no lesion/ablation/causal grounding of mechanisms).

---

## Position Statement for This Research Program

**We explicitly do not make consciousness claims.** Our work focuses on:

- **Mechanistic interpretability** through mathematical frameworks (RKHS, AC attention)
- **Safety-relevant behavioral analysis** via morphemic pole detection and semantic action computation
- **Welfare proxy measurements** without phenomenal consciousness assertions
- **Statistical validation** of circuit discovery and behavioral patterns

When reporting results that might superficially resemble "consciousness benchmarks," we:

1. **Label them explicitly as proxies** for safety-relevant capacities
2. **Provide mechanistic explanations** where possible (attention patterns, feature activation)
3. **Acknowledge limitations** of behavioral-only evidence
4. **Reference this critique** to establish methodological boundaries

This approach maintains scientific rigor while contributing to Anthropic's safety mission without inflated consciousness claims.