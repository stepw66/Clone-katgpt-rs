# Research 153: Thinking Pixel — Recursive Sparse Pruner Routing

> **Paper:** [The Thinking Pixel: Recursive Sparse Reasoning in Multimodal Diffusion Latents](https://arxiv.org/pdf/2604.25299) — Sun, Yao, Li, Zhu (Shanghai Academy of AI for Science, Fudan University), Apr 2026
> **Date:** 2026-06, distilled 2026-06
> **Related Research:** 059 (MoE SD Co-Design), 073 (LT2), 091 (SpecHop), 095 (MGR), 097 (Training-Free Loop), 098 (PrudentBanker), 145 (FeedbackBandit)
> **Related Plans:** 131 (SpecHop), 108 (LT2), 112 (SR²AM), 136 (Training-Free Loop), 163 (FeedbackBandit), 171 (this plan)
> **Verdict: MODERATE VALUE — Three modelless distillations: (1) Per-token pruner routing via entropy-gated gating, (2) Recursive prune-verify cycles with step-conditional pruner selection, (3) FrozenBase guard principle for looped inference stability. The paper's MoE LoRA routing is model-based; our value extraction is the *gating pattern* and *recursion-without-distribution-drift* principle applied to DDTree pruner selection. Feature-gate as `thinking_prune`.**

---

## TL;DR

The Thinking Pixel introduces recursive sparse reasoning in diffusion models: visual tokens are iteratively refined over latent steps by sparsely-activated LoRA modules, dynamically selected by a Gumbel-Softmax gating network conditioned on (token state, diffusion timestep, conditioning). Key findings:

1. **Per-token routing** produces emergent expert specialization — early diffusion timesteps route uniformly, later timesteps diverge to patch-specific experts
2. **Frozen base model only at final step** prevents distribution drift from repeated exposure to fixed representations
3. **Conditioning-aware routing** (visual + timestep + label) is critical; ablating to visual-only produces poorly diversified modules
4. **Gumbel-Softmax** with temperature τ=5.0 enables winner-takes-all gradient allocation for emergent specialization

**Our extraction (modelless):** We don't have LoRA adapters or diffusion models. We extract the **routing pattern** (entropy + position + context → sparse expert selection) and the **frozen-base guard** (only apply full verification at final recursion step) as DDTree pruner routing improvements.

---

## Paper Core: Three Mechanisms

### 1. Mixture-of-Adapters (§3.2)

M LoRA adapter modules {θ_m} with a gating network θ_gate that produces per-token routing probabilities:

```
logits = θ_gate(x_latent_t, y)              // y = timestep + conditioning
z_m = logits_m + Gumbel(0,1)                // exploration noise
ẑ_m = softmax(z_m / τ)                      // temperature-controlled sparsity
m* = argmax_m ẑ_m                            // winner-takes-all selection
```

Key design: routing computed **per visual token independently across samples**, then reassembled back to spatial order. This allows different experts to access tokens across samples (better learning efficiency) while preserving spatial structure.

**Gumbel-Softmax temperature τ=5.0** — high temperature keeps gradients flowing through the argmax, encouraging specialization through winner-takes-all gradient allocation (cite: Compete & Compose, 2024).

### 2. Recursive Sparse Joint Attention (§3.3)

R latent steps with LoRA-only updates, frozen base only at final step:

```
ã_0 = x̃                                              // init from modulated input
for t in 1..T-1:                                      // intermediate: LoRA only
    [a_x^t; a_c^t] = Attn(B^t A^t (LN(ã^{t-1})); c̃) // selected LoRA adapter
    ã^t = a_x^t + x̃                                   // residual to original input
[a_x^T; a_c^T] = Attn(B^T A^T (LN(ã^{T-1})) + Ŵ x̃; c̃)  // FINAL: LoRA + frozen base
```

**Critical insight:** Processing through frozen base model repeatedly at each step **corrupts generation performance** due to compounding artifacts. The frozen base is applied **only at the final step** to anchor the output back to the original distribution space.

### 3. Emergent Specialization (§4.2)

PCA of latent trajectories across diffusion timesteps reveals:
- **Early timesteps** (high noise): unified, coherent processing — all experts do similar work
- **Later timesteps** (low noise): divergent, patch-specific pathways — experts specialize by visual region

Routing frequency analysis (Figure 4): conditioning on (visual tokens + timestep + class label) produces diverse, balanced expert usage. Ablating to visual-only → modules struggle to diversify across latent steps.

---

## Modelless Distillation: Three Extractions

### D1: Entropy-Gated Pruner Routing (Per-Token Sparse DDTree)

**Source:** Paper's per-token expert routing (§3.2)
**Target:** `ScreeningPruner::relevance()` per token in DDTree

Currently, `BanditPruner` uses a single strategy (UCB1/Thompson) applied uniformly across all token positions. The Thinking Pixel's per-token routing suggests:

```
For each token position i in DDTree:
  1. Compute local entropy h_i from marginals[vocab_range for position i]
  2. Route to one of M pruner experts based on (h_i, position, parent_tokens)
  3. Expert selection = argmax softmax(gate(h_i, pos, context) / τ)
```

**Pruner experts** (modelless equivalents of LoRA adapters):
- Expert 0: Aggressive entropy pruning (high confidence tokens → skip branch)
- Expert 1: Conservative domain pruning (low confidence → defer to inner pruner)
- Expert 2: Structural pruning (syntax-aware: bracket matching, keyword validation)
- Expert M-1: Exploration (Thompson sampling for uncertain regions)

**Gating input** (analogous to paper's x_latent + y):
- Local entropy at position (analogous to visual token state)
- Position in sequence (analogous to diffusion timestep — early positions are "high noise" drafts, later positions are refined)
- Parent token IDs (analogous to conditioning — constrains valid continuations)

**Implementation path:** Extend `BanditPruner` with per-position routing via a small gating table `[PositionRole; MAX_POSITIONS]` where `PositionRole` maps to pruner expert index. This is a lookup, not a neural network — O(1) per token.

**Expected gain:** Better pruning accuracy in DDTree — high-entropy positions get exploration-focused pruners, low-entropy positions get exploitation-focused pruners. The paper shows this specialization pattern is where most quality gains come from.

**Risk:** Overhead of routing computation. Mitigate by pre-computing the routing table once per DDTree call (M calls instead of N×M).

### D2: Recursive Prune-Verify with FrozenBase Guard

**Source:** Paper's recursive computation with frozen-base-only-at-final-step (§3.3)
**Target:** SpecHop (Plan 131) hop-level verification, LT2 (Plan 108) looped inference

Currently, each SpecHop hop applies the same full verification pipeline. The Thinking Pixel shows that intermediate steps should use **lightweight pruning only** (LoRA-only in their case), with full verification reserved for the final step:

```
For hop h in 1..H-1:   // intermediate hops
  DDTree with lightweight ScreeningPruner only (no ConstraintPruner verification)

For hop H:              // final hop
  DDTree with full ConstraintPruner + ScreeningPruner verification
```

**The frozen-base guard principle:** Repeated full verification at every hop can "corrupt" the draft distribution by over-pruning — similar to how repeated exposure to the frozen base model corrupts visual tokens. Lightweight intermediate pruning preserves exploration while final-step full verification ensures quality.

**Implementation path:** Add a `hop_index` and `total_hops` parameter to `build_dd_tree_screened()`. When `hop_index < total_hops - 1`, skip `ConstraintPruner` and use only `ScreeningPruner`. At the final hop, apply both.

**Expected gain:** Faster intermediate hops (ConstraintPruner can be expensive — especially WASM validators), with the same final quality. The paper's ablation (Table 2) shows recursion without modulation still improves over no recursion.

**Risk:** Intermediate hops may accumulate errors without ConstraintPruner. Mitigate by still applying ScreeningPruner (lightweight relevance filtering).

### D3: Step-Conditional Pruner Selection

**Source:** Paper's conditioning-aware routing (§3.2, Figure 4)
**Target:** SR²AM (Plan 112) adaptive planning decisions

The paper shows that conditioning routing on (visual tokens + timestep + class label) is critical — without timestep/label, modules fail to diversify. Our analog:

- **"Timestep"** = decode position (early draft vs late refinement)
- **"Class label"** = domain type (code, natural language, game state)
- **"Visual tokens"** = current DDTree marginals

SR²AM already has adaptive planning decisions (tree budget, early exit, speculative depth). Adding step-conditional pruner selection means the **type** of pruner changes based on where we are in the decode sequence:

```
Early decode: aggressive pruning (entropy-based, fast)
Mid decode: balanced pruning (domain + entropy)
Late decode: conservative pruning (full verification, high precision)
```

This is already partially captured by `early_exit_patience` in SR²AM, but the pruner *type* selection is new. Currently, the same pruner is used throughout.

**Implementation path:** Add `PrunerSchedule` enum to SR²AM configurator:

```rust
enum PrunerSchedule {
    Uniform,                                    // same pruner throughout (current)
    EntropyGated { aggressive: f32, conservative: f32 }, // threshold-based switching
    PositionDecayed { early: Box<dyn ScreeningPruner>, late: Box<dyn ScreeningPruner> },
}
```

**Expected gain:** Better adaptive behavior — early positions (where the model is uncertain) get different treatment than late positions (where context is rich). Aligns with paper's finding that specialization emerges from conditioning on timestep.

---

## What We Do NOT Extract

| Paper Concept | Why Not Applicable |
|---|---|
| LoRA adapters as modules | Model-based — requires weights, gradients, training |
| Gumbel-Softmax for training | Training-time mechanism — we are inference-only |
| Joint attention (vision+text) | We don't have multimodal attention |
| Diffusion timestep conditioning | We don't have diffusion models |
| Image generation benchmarks (FID, IS) | Different domain entirely |
| FrozenLake visual navigation | Interesting but not our focus |

---

## Alignment with Optimization.md

| Extraction | Optimization Concern | Mitigation |
|---|---|---|
| D1: Per-token routing | Extra computation per token | Pre-compute routing table (M calls, not N×M). Fixed-size lookup table, O(1) per token |
| D2: Lightweight intermediate hops | Already aligns — fewer pruner calls = faster hops | No mitigation needed — this is a speedup |
| D3: Step-conditional selection | Switching pruner type per position | Use enum dispatch (zero-cost in Rust) not trait objects |

All three distillations are **pre-compute where possible, O(1) at runtime** — consistent with optimization.md's "Do" patterns.

---

## GOAT Proof Strategy

| Distillation | Proof Type | Metric |
|---|---|---|
| D1: Entropy-gated routing | Benchmark: DDTree valid-node rate with/without routing | ≥ same valid rate with better specialization |
| D2: FrozenBase guard | Benchmark: SpecHop total latency with/without intermediate ConstraintPruner | Same quality, ≤ latency |
| D3: Step-conditional selection | Benchmark: SR²AM planning accuracy by decode position | Better accuracy at late positions |

---

## Commercial Alignment (per Verdict 003)

All three distillations are **modelless engine improvements** — they enhance the DDTree/pruner infrastructure that lives in the MIT-licensed katgpt-rs. They improve the "engine" that produces syntactically-valid tokens. The "intelligence" (lora.bin, validator.wasm) remains in the SaaS layer.

- D1 is a better pruner → better DDTree output → better translations → feeds the data flywheel
- D2 is faster speculation → lower latency → better user experience
- D3 is adaptive planning → better inference budget utilization → cost savings

All three strengthen the engine without touching the fuel.
