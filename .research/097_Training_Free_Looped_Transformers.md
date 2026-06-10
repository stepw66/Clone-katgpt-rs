# Research 97: Training-Free Looped Transformers

> **Paper:** [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872) — Chen, Li, Liang, Lao, Liu (UT Austin), May 2026
> **Date:** 2026-05-25
> **Related Research:** 073 (LT2 — training-time looped transformers), 034 (D2F), 055 (Nemotron TriMode), 073 (LT2), 035 (Attractor/Fixed-Point)
> **Related Plans:** 108 (LT2 Looped Inference Pipeline — completed), 136 (Training-Free Loop Wrapper)
> **Verdict: HIGH VALUE — Complements our existing LT2 (Plan 108) with a zero-training inference-time wrapper. Our LT2 is training-time weight-sharing; this is training-free ODE-motivated refinement. Both can coexist. The damped Euler sub-stepping, block-vs-layer-mode, and depth-fraction rule map directly onto our existing `LoopMode`/`HybridPattern` dispatch. Priority: add as `LoopMode::TrainingFree` variant with RK sub-stepping strategy, then validate on our micro benchmarks.**

---

## TL;DR

A training-free wrapper that loops a contiguous mid-stack block of layers from a **frozen checkpoint** without any fine-tuning, architectural changes, or auxiliary parameters. Core insight: each pre-norm transformer layer is a forward Euler step at h=1 on a residual ODE; looping with damped sub-steps of size h=1/K better approximates the same t=1 endpoint.

Key results across 7 model families (Qwen3, Llama-3.2, DeepSeek-V2-Lite, Moonlight, Qwen1.5-MoE):
- **Qwen3-4B-Instruct MMLU-Pro: +2.64 pp**
- **Qwen3-4B-Instruct GPQA-Main: +2.01 pp**
- **Qwen1.5-MoE ARC-Challenge: +2.30 pp**
- **87% of 45 cells positive or neutral** under a fixed recipe with no per-cell tuning

---

## Core Innovation: ODE-Refined Sub-Stepping

### 1. The Forward-Euler View

A pre-norm transformer layer L implements:
```
L(x) = x + Attn(LN₁(x)) + MLP(LN₂(x + Attn(LN₁(x))))
```

The residual update `F_L(x) = L(x) - x` is exactly one forward Euler step at h=1 on the ODE:
```
ẋ = F_g(x)    where g is the loop window operator
```

Naively looping K times computes x(t=K), but post-loop layers were trained to receive x(t=1). This explains why naive looping **degrades** performance.

### 2. Damped Euler Sub-Stepping

Instead, perform K Euler steps of size h=1/K to approximate the **same** t=1 endpoint:
```
x_{k+1} = (1 - 1/K) · x_k + (1/K) · g(x_k)
```

This is a **damped** update that stays in the trained regime. It converges to x(t=1) at O(1/K).

### 3. K-Stage Runge-Kutta (Algorithm 3)

The paper's best strategy. A specific Butcher tableau makes the RK output an interpolation:
```
x₁ = β · g(x₀) + (1-β) · F^K(x₀)
```
where F^K is the K-step damped Euler endpoint and β ∈ [0,1] anchors toward the original one-shot output.

- β=0: pure K-step damped Euler
- β=1: original one-shot output (identity loop)
- β=0.5: balanced (robust default)

**This is the winning strategy across all benchmarks.**

---

## Two Iteration Modes

### Block-Mode: `g^(K) = (L_b ∘ ... ∘ L_a)^K`
- Iterate entire window as one unit
- Best for **dense** models
- Each iteration re-evaluates all layers in the window

### Layer-Mode: `g^(K) = L_b^K ∘ L_{b-1}^K ∘ ... ∘ L_a^K`
- Iterate each layer K times before passing to next
- **Required for MoE** models (prevents routing thrash)
- Pins per-layer expert routing across iterations
- Also slightly better for sub-1B dense models

| Backbone | Default Mode | Reason |
|----------|-------------|--------|
| Dense MHA (Qwen3, Llama) | Block-mode | Whole-block residual is smoother |
| MoE (Qwen1.5-MoE, Moonlight, DeepSeek-V2) | Layer-mode | Prevents expert routing thrash |
| MLA+MoE (DeepSeek-V3 family) | Layer-mode | Same MoE routing issue |

---

## The Depth-Fraction Rule

The optimal loop window center sits at fractional depth 0.45–0.60 across 9 architectures:

| Model | Layers | Window | Depth Fraction |
|-------|--------|--------|----------------|
| Qwen3-4B | 36 | [15–18] | 0.46 |
| Qwen3-1.7B | 28 | [12–15] | 0.48 |
| Qwen3-30B-A3B | 48 | [22–24] | 0.48 |
| Llama-3.2-3B | 28 | [12–15] | 0.48 |
| Moonlight-16B-A3B | 27 | [8–11] | 0.35 |
| Qwen1.5-MoE-A2.7B | 24 | [13–16] | 0.60 |
| DeepSeek-V2-Lite | 27 | [13–16] | 0.54 |

**Sweet spot: n=4 layers (width=4) at mid-depth.** Wider windows (n≥6) degrade. Window=entire-network is catastrophic.

---

## Failed Strategies (Important for Implementation)

The paper tried 40+ configurations. Key negative results:

| Strategy | Result | Why |
|----------|--------|-----|
| **Naive K-loop** | Catastrophic (-17.71 pp at K=6) | Advances to x(t=K), not x(t=1) |
| **Anderson acceleration** | -18.06 pp at K=8 | Block is not contractive; Anderson overshoots |
| **RK4, Heun, Midpoint** | -0.65 to -2.34 pp | Higher-order methods assume smooth ODE; transformer blocks aren't |
| **Heavy-ball** | -0.06 to -4.88 pp | Momentum amplifies non-contraction |
| **Aitken Δ²** | -7.00 to -11.19 pp | Sequence not convergent enough for extrapolation |
| **Norm stabilization** | Catastrophic at K≥3 | Diverging in direction, not magnitude |
| **cache=none** | -23.40 pp on MBPP | Loop region must contribute KV to cache |

**Only K-stage Runge-Kutta (damped Euler with β anchor) works robustly.**

---

## KV Cache Handling for Autoregressive Decode

Two-phase scheme for decode-time looping:

1. **Loop body phase**: K iterations with snapshot/restore per iteration
   - Before each iteration: snapshot per-layer KV cache lengths
   - Run body with `use_cache=True` (reads past KV)
   - After: crop cache back to snapshot length (zero net KV writes)

2. **Stash phase**: One additional pass writes canonical KV
   - `cache=FIRST`: use pre-loop hidden state as input
   - `cache=LAST`: use post-loop hidden state as input
   - `cache=NONE`: catastrophic (ablation only)

**cache=first** dominates for long CoT; **cache=last** dominates for short structured generation.

Total cache delta: exactly (b-a+1) entries per decode position regardless of K (identical to unmodified model).

---

## Wall-Clock Cost

On Qwen3-4B-Instruct, GSM8K@200, K=3:

| decode_mode | Overhead |
|-------------|----------|
| bypass (prefill only) | -1.5% (noise) |
| first_n N=16 | -1.0% |
| first_n N=64 | +4.6% |
| full (every decode step) | +21.6% |

**For knowledge-MC benchmarks (headline results), bypass mode is used — effectively free.**

---

## Mapping to Our Architecture

### Relationship to Our LT2 (Plan 108)

| Aspect | LT2 (Plan 108, ours) | Training-Free (this paper) |
|--------|----------------------|---------------------------|
| Training required | Yes (weight-shared pretraining) | **No** |
| Weight sharing | Same weights T times | Same weights K times |
| ODE interpretation | Rank-T state upgrade | Sub-step refinement to t=1 |
| Loop target | Entire model depth | Mid-stack window only |
| MoE handling | Not addressed | Layer-mode pins routing |
| KV cache strategy | AHLA state carry (O(1)) | Snapshot/restore + stash |
| Use case | Train-from-scratch efficiency | Retrofit frozen checkpoints |

### What We Can Adopt Directly

| Component | Maps To | Effort |
|-----------|---------|--------|
| Damped Euler sub-stepping | `LoopMode::TrainingFree` variant | Low |
| Block vs layer mode dispatch | Extend `HybridPattern` or add `IterationMode` | Low |
| Depth-fraction rule | Window selection heuristic | Trivial |
| Snapshot/restore KV protocol | Extend decode pipeline | Medium |
| K-stage RK with β anchor | New `SubStepStrategy` enum | Low |
| Cache strategy (first/last) | Config option | Trivial |

### What We Don't Need

- **Anderson, heavy-ball, Aitken, RK4** — All fail. Only damped Euler + RK anchor works.
- **Norm stabilization** — Catastrophic.
- **Training** — The whole point is training-free.

---

## Distillation Strategy for Our Model-Based/Modelless Paradigm

### Modelless (Training-Free)

This paper is inherently **modelless** for us:
1. No weights change — just inference-time wrapper
2. Works on any frozen checkpoint
3. Gains on knowledge-heavy MC tasks (+2 pp on MMLU-Pro)
4. Zero training cost

**Integration point:** Our `forward_looped()` already has `LoopMode`. Add `LoopMode::TrainingFree` that:
- Selects mid-4-layer window at depth fraction 0.48
- Applies K-stage RK with β=0.5
- Uses block-mode for dense, layer-mode for MoE
- Handles KV cache with snapshot/restore

### Model-Based (Training-Time, Complementary)

Our existing LT2 (Plan 108) already covers the training-time path:
- Weight-shared pretraining for efficiency
- Rank-T state upgrade with AHLA
- Hybrid 1:4 dispatch

The training-free wrapper can **layer on top of** LT2 for additional refinement at inference.

---

## Feature Gate Proposal

| Gate | Scope | Description |
|------|-------|-------------|
| `tf_loop` | katgpt-core | `TrainingFreeLoopConfig`, `SubStepStrategy`, `CacheStrategy` |
| `tf_loop` | katgpt-rs | Training-free loop forward pass, snapshot/restore |

Dependencies: `tf_loop` requires `lt2_looped` (reuses `LoopMode` infrastructure).

---

## Key Insights for Our Stack

1. **Sub-stepping > naive looping.** Our existing `LoopMode::WeightShared` does naive looping (whole-model weight sharing). Adding sub-stepping with ODE-motivated damping would improve quality even for training-time loops.

2. **Windowed looping > full-model looping.** Only 4 mid-stack layers benefit. Full-model looping is catastrophic. This is an important constraint for our LT2: we may want to add windowed mode.

3. **Layer-mode for MoE is essential.** When we integrate with MoE models, layer-mode prevents routing thrash. Our current LT2 doesn't distinguish block/layer mode.

4. **KV cache snapshot/restore is elegant.** Zero allocation, zero net cache growth. This pattern is useful beyond training-free looping.

5. **The depth-fraction rule is universal.** 0.45–0.60 across 9 architectures. We can hardcode this as a default.

6. **Bypass mode is free.** For prefill-only looping (knowledge refinement), the overhead is negligible. This is the deployment story.

---

## References

- Paper: https://arxiv.org/abs/2605.23872
- Deep Equilibrium Models (Bai et al., 2019): the ODE interpretation foundation
- Our LT2 (Research 073, Plan 108): training-time looped transformers
- Our D2F (Research 034): discrete diffusion forcing, another looped approach
