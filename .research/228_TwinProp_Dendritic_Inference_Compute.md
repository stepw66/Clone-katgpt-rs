# Research: TwinProp — Dendritic Computation for Inference-Time Adaptive Compute

**Paper**: "What can a neuron compute" — Aizenbud, Beniaguev, Pnueli, Segev & London (bioRxiv 2026.06.08.730984)
**Date**: 2026-06-12
**Verdict**: 🐐 GOAT for modelless — **DendriticGate** adaptive tree branching

---

## Paper Summary

TwinProp introduces a **digital-twin-based backpropagation** algorithm that trains a millisecond-accurate DNN surrogate of a detailed Layer 5 pyramidal cell (L5PC), then uses gradient descent to optimize synaptic weights and dendritic locations. Key results:

1. **Single neuron = deep network**: L5PC solves image/audio classification, XOR, 10-bit parity, random Boolean tasks — capabilities typically requiring multi-layer networks
2. **Dendritic nonlinearities are the substrate**: NMDA and voltage-dependent mechanisms are *recruited* as task complexity increases. Removing them or flattening dendritic structure kills performance
3. **Intrinsic adaptive compute**: Complex tasks recruit more distal dendritic branches; simple tasks solved proximally. No meta-controller needed — biophysics IS the controller
4. **NMDA Mg²⁺ block = biological sigmoid gate**: Coincidence-detecting, not just thresholding. Functionally a **product-of-sigmoids** (AND gate in dendritic branches)

---

## Distillation to katgpt-rs (Modelless)

### What TwinProp Proves That We Already Use

| TwinProp Finding | katgpt-rs Equivalent | Status |
|-----------------|---------------------|--------|
| Dendritic tree = computational depth | DDTree with 20+ build variants | ✅ Already proven |
| Nonlinear gating per branch | `ConstraintPruner` + `ScreeningPruner` soft/hard gating | ✅ Already proven |
| Task complexity recruits more branches | `ThinkingController` bandit (Direct/Latent/CpuResample) | ✅ Already proven |
| Coincidence detection (AND gate) | `AndOrNode::And` (all children must solve) | ✅ Already proven |
| Adaptive compute without meta-controller | `ThinkingBandit` Thompson sampling | ✅ Already proven |

### What's Novel: **DendriticGate** — Physics-Based Branch Gating

The paper's key insight is that NMDA voltage-dependent gating provides **intrinsic difficulty estimation** — coincident inputs trigger more branches, simple inputs don't. This is *physics-based*, not learned. We can model this as:

```rust
/// Dendritic branch gate — modeled on NMDA Mg²⁺ voltage-dependent coincidence detection.
/// Unlike learned difficulty estimators, this uses signal *coincidence* (entropy)
/// to gate tree expansion depth. Zero parameters, zero training.
///
/// Biophysics: NMDA spike = σ(V - V_threshold) × [Mg²⁺ unblock]
/// Our analog: branch_activation = σ(entropy - θ) × coincidence_score
pub struct DendriticGate {
    /// Voltage threshold analog — entropy threshold below which branches are pruned
    threshold: f32,
    /// Mg²⁺ unblock rate — how quickly gate opens above threshold
    voltage_sensitivity: f32,
    /// Coincidence window — tokens must agree within this span
    coincidence_window: usize,
}
```

### Core Algorithm: Dendritic Branching Rule

From the paper: "Increasing task complexity recruits distributed dendritic nonlinearities."

**Modelless implementation** — no training, pure inference-time:

```
For each DDTree expansion at depth d:
  1. Compute local_entropy = entropy(marginals[d])
  2. coincidence = max_agreement(top_k_candidates, parent_path)
  3. nmda_gate = sigmoid(voltage_sensitivity × (local_entropy - threshold))
  4. branch_budget = base_budget × nmda_gate × coincidence
  5. Expand with min(1, floor(branch_budget)) branches
```

This replaces the current `ThinkingController` bandit with a **physics-inspired deterministic gate** that:
- Expands more branches when entropy is high (uncertain = complex task)
- Expands fewer branches when entropy is low (confident = simple task)
- Multiplies by coincidence (agreement among top candidates) — the "AND gate"
- Zero parameters, zero training, deterministic

### Integration Points

| Component | Current | DendriticGate Enhancement |
|-----------|---------|--------------------------|
| `ThinkingController` | 3-arm bandit | Add 4th arm: `Dendritic` — physics-gated adaptive depth |
| `TreeBuilder` | Fixed `tree_budget` | Dynamic budget per expansion: `budget *= nmda_gate` |
| `build_dd_tree_belief_collapse_aware` | Entropy threshold × 1.5/0.5 | Replace with continuous sigmoid gate |
| `MuxBfs` | Fixed width or `MuxBanditWidth` | `comp_width *= nmda_gate` — width modulated by difficulty |
| `build_dd_tree_and_or` | `AndOrNode` decomposition | AND nodes get coincidence gate; OR nodes get entropy gate |

### GOAT Gate Design

```toml
# Cargo.toml feature
dendritic_gate = ["collapse_aware_thinking", "thinking_cot"]
```

**Promotion criteria**:
- G1: Dendritic arm beats AlwaysDirect on hard queries (already proven by `Latent` arm)
- G2: Dendritic arm matches AlwaysLatent quality at ≤80% compute budget
- G3: Zero-allocation in hot path (stack-only `DendriticGate`)
- G4: Deterministic — same input → same budget allocation (no bandit randomness)
- G5: SIMD-accelerated entropy + coincidence computation

---

## Fusion Novel Ideas (Not Direct Mapping)

### 1. **Dendritic Pruning Cascade** — Multi-Scale NMDA Gating

TwinProp shows distal dendrites are recruited only when proximal computation is insufficient. We can model this as a **cascade of increasingly expensive pruners**:

```
Level 0 (proximal): NoPruner — just argmax (free)
Level 1 (proximal): ConstraintPruner — syntax check (O(n))
Level 2 (intermediate): ScreeningPruner — soft relevance (O(n·k))
Level 3 (distal): CompletionHorizon — A* with jump-ahead (O(n·log(n)))
Level 4 (distal): BeliefDrafter — latent dynamics prediction (O(d²))
```

Each level activates only if the previous level's `max_relevance < activation_threshold`. This is a **dendritic cascade** — proximal to distal, gated by accumulated signal strength.

### 2. **Product-of-Sigmoids Branch Scoring**

The NMDA AND-gate computes `σ(w₁·x₁) × σ(w₂·x₂)` — product of independent sigmoids, not a single sigmoid of weighted sum. This is fundamentally more expressive for coincidence detection.

Apply to `manifold_score()`:
```rust
fn manifold_score(&self, depth: usize, token: usize, parents: &[usize]) -> f32 {
    // Current: single sigmoid(w·x + b)
    // NMDA-style: product of per-branch sigmoids
    let score_a = sigmoid(self.branch_weights[depth] * self.token_features[token]);
    let score_b = sigmoid(self.path_weights[depth] * self.path_features(parents));
    score_a * score_b  // AND gate — both must agree
}
```

### 3. **Structural Plasticity — Runtime Dendritic Rebranching**

TwinProp optimizes dendritic *locations* (not just weights). The analog in DDTree is runtime restructuring of the tree topology based on accumulated statistics:

- Track per-depth `branch_utility = Σ accepted_tokens / Σ expanded_tokens`
- Periodically restructure: merge low-utility depths (prune dendritic branch), split high-utility depths (grow new branch)
- This is **structural plasticity** — the tree reshapes itself at inference time, no training needed

---

## Commercial Verdict (per 003 strategy)

| Question | Answer |
|----------|--------|
| Does this create a new engine capability? | **Yes** — physics-gated adaptive compute, zero-training |
| Does it require `lora.bin`? | **No** — purely modelless |
| Is it defensible? | **Yes** — novel fusion of dendritic neuroscience + DDTree pruning |
| Does it strengthen the moat? | **Yes** — adds a 4th thinking mode with provable compute savings |
| Risk | Low — additive feature behind feature flag, doesn't change existing paths |

**Verdict: GAIN → create plan, implement as GOAT-gated feature**

---

## Related Work

1. **Beniaguev et al. 2021** — "Single cortical neurons as deep artificial neural networks" — L5PC ≈ 5-8 layer DNN. *Neuron*
2. **Agrawal & Buice 2025** — Formal complexity bounds scale with dendritic arbor size. *NeurIPS*
3. **Feature binding via dendritic networks (2025)** — Sublinear branches solve XOR-like binding. *Neural Networks*
4. **Adaptive test-time compute allocation (2025)** — Training-free difficulty estimation. *OpenReview*
5. **Sacramento et al. 2018** — Dendritic compartments approximate backprop. *NeurIPS*
6. **Dendrites in ANNs (2025)** — Adding dendritic properties → more precise, robust, parameter-efficient. *Nature Communications*

---

## TL;DR

TwinProp proves dendritic trees provide intrinsic adaptive compute via NMDA voltage-dependent gating. We already have the tree (DDTree) and the pruning (ConstraintPruner). The novel fusion is a **physics-inspired sigmoid gate** that modulates tree expansion budget based on signal entropy + candidate coincidence — zero training, zero parameters, deterministic. Add as `dendritic_gate` feature, validate via GOAT gate, promote if it beats the existing `Latent` arm on compute efficiency.
