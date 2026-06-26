# Research: Residual Context Diffusion (RCD) for katgpt-rs

**Date:** 2026-06-12
**Paper:** [arXiv:2601.22954](https://arxiv.org/abs/2601.22954) — Residual Context Diffusion Language Models
**Status:** Verdict

---

## Paper Summary

RCD recycles discarded low-confidence token representations from block-wise dLLM denoising as entropy-weighted soft embeddings injected into the next denoising step. Instead of binary remasking (commit top-m confident tokens, discard rest → [MASK]), RCD:

1. **Computes residual vectors** from discarded token probability distributions via weighted sum with the embedding codebook: `Δ_i = Σ_j p_ij * E_j`
2. **Weights by normalized Shannon entropy**: `α_i = H(p_i) / log(V)` — high entropy (uncertain) tokens get MORE residual influence
3. **Interpolates** with mask embedding: `ẽ_i = (1-α_i)*E_mask + α_i*Δ_i` for remaining masked positions
4. **Two-stage training**: frozen reference model provides proxy residual signals → target model learns to consume them. Decouples BPTT memory bottleneck.

**Results**: 5-10pp accuracy gains, 2× on AIME, 4-5× step reduction. ~1B tokens to convert standard dLLM to RCD.

---

## Existing Codebase Overlap

| Component | katgpt-rs Existing | RCD Paper | Gap |
|-----------|-------------------|-----------|-----|
| Shannon entropy | `shannon_entropy()` in `dllm_solver.rs` | Eq. (3) normalized entropy | Need normalization by `log(V)` |
| Critical interval | `is_critical_interval()` + `select_solver()` | Entropy-based step control | Already have entropy-triggered solver switching |
| D2F denoise loop | `denoise_loop()` with confidence threshold | RCD iterative denoising | Need residual injection between steps |
| Self-conditioned drafting | `SelfCondDraft` — 2-pass sigmoid sharpening | RCD's "warm start" + self-referential loop | SC buffer is structural twin of residual context |
| MuxDemux verifier | Superposition token recovery via geometric decay | RCD's soft embedding construction | MUX superposition ≈ RCD weighted sum, different use case |
| CPU/GPU/ANE routing | `InferenceRouter` 4-layer cascade | Not addressed in paper | Our routing can entropy-gate residual compute |
| DMax soft parallel decode | `h = conf*e_token + (1-conf)*e_mask` (Plan 109) | RCD's `ẽ = (1-α)*E_mask + α*Δ` | **Same formula, different semantics** |

---

## Novel Fusion Ideas (Beyond Direct Mapping)

### Idea 1: Entropy-Gated Residual Pruner (`ResidPruner`)

**Not in paper.** RCD injects residuals uniformly for all masked positions. Our insight: use existing `ConstraintPruner` trait to *selectively* inject residuals only where pruner validation passes.

```rust
trait ConstraintPruner {
    fn is_valid(&self, token_idx: u32, context: &[u32]) -> bool;
}

// RCD: inject residual for ALL masked positions
// Our fusion: only inject for positions where pruner says the residual is syntactically plausible
// Pruned residual → less noise injection → faster convergence
```

**Why novel**: Paper doesn't consider constraint-based filtering of residual signals. Our `SynPruner` already knows which token combinations are valid Rust. Filtering residuals through this knowledge prevents injecting nonsensical context.

### Idea 2: Tier-Adaptive Residual Weighting

**Not in paper.** RCD uses fixed `T_res` temperature for entropy scaling. Our insight: route residual computation cost based on `InferenceRouter` tier.

| Tier | Residual Strategy | Rationale |
|------|------------------|-----------|
| Plasma (CPU-SIMD) | Skip residuals, direct remask | Speed priority, <50μs budget |
| Hot (CPU full) | Lightweight: `α_i = max_prob` (1 register) | Confidence-only, no entropy computation |
| Warm (GPU) | Full RCD: normalized entropy + codebook sum | Balanced accuracy/speed |
| Cold (GPU batch) | Full RCD + reference model warm start | Accuracy priority, batch amortized |

**Why novel**: Paper treats residual injection as always-on. We gate it by inference tier — plasma path (game AI) skips it entirely, cold path (chain) gets full RCD. This respects the plasma/hot/warm/cold/freeze tier system.

### Idea 3: MUX-RCD Fusion — Superposition Residuals

**Not in paper.** RCD constructs residuals from single-step probability distributions. Our MUX infrastructure already does multi-token superposition. Fusion: instead of `Δ_i = Σ_j p_ij * E_j`, use MUX's geometric-weighted superposition to construct residuals from *multiple candidate paths* in the DDTree.

```
RCD:  Δ_i = Σ_j p_ij * E_j               (single distribution)
MUX:  mux(r) = Σ_i ρ^(S-i)·(1-ρ) · onehot(t_i)  (multi-path superposition)
Fusion: Δ_i = Σ_path Σ_j p_path_ij · w_path · E_j  (weighted over DDTree branches)
```

**Why novel**: RCD only looks at the current step's distribution. Our DDTree has *multiple candidate paths* with scores. Fusing MUX superposition with RCD residual construction means the residual carries information from ALL viable paths, not just the current greedy distribution. This is strictly more informative.

### Idea 4: Thinking Fold + RCD — ODE-Refined Residuals

**Not in paper.** RCD residuals are computed once per step. Our `tf_loop.rs` already does ODE sub-stepping. Fusion: use Thinking Fold's damped Euler steps to *refine* residual vectors before injection.

```
Standard RCD:    Δ^k → inject into step k+1
TF-RCD fusion:   Δ^k → sub-step x ← x + (1/K)(y-x) → refined Δ → inject
```

**Why novel**: The paper doesn't refine residuals. Our TF loop already has KV snapshot/restore for iterative refinement. Applying it to residuals gives higher-quality context injection. The `AccelBoundConfig` prevents divergence.

### Idea 5: Spectral Budget Residual Allocation

**Not in paper.** RCD gives all positions the same residual treatment. Our `SpectralBudgetArm` already allocates compute budgets based on spectral analysis. Fusion: allocate residual computation budget by spectral energy of the marginal distribution.

- High spectral energy (peaked distribution) → position is near-decided → skip residual
- Low spectral energy (flat distribution) → position is uncertain → full residual injection

**Why novel**: Entropy is scalar, spectral energy is richer. A distribution can have low entropy but high spectral energy (concentrated on wrong tokens). Spectral analysis catches this; pure entropy doesn't.

---

## Competitive Landscape

| Paper | ID | Threat Level | Notes |
|-------|-----|-------------|-------|
| **DSL-LLaDA** | `2606.01024` | **High** | Continuous soft masking replaces binary masking entirely. Makes "discarded tokens" problem disappear. |
| **TokenDrift** | `2605.19470` | Medium | Soft-token anti-symmetric drift. Training-time, complementary. |
| **ADAS** | `2606.10829` | Low | Training-free attention-discounted sampler. Stackable with RCD. |
| **DPRM** | `2604.24357` | Low | Token ordering module. Orthogonal axis. Stackable. |
| **DMax** (ours) | Plan 109 | Internal | Already implements hybrid embedding. RCD is the principled version. |

---

## Verdict per Commercial Strategy (003)

### Engine/Fuel Split

| Layer | What | License | RCD Impact |
|-------|------|---------|------------|
| Engine | Residual injection in denoise loop | MIT (open) | **Idea 1 (ResidPruner)** + **Idea 2 (tier routing)** = engine-level |
| Fuel | Reference model weights + entropy calibration | Private (SaaS) | Reference model `M_ref` stays in riir-ai |

### Verdict: **GAIN — Modelless First, Novel Fusion Worth Implementing**

**Why gain, not GOAT:**
1. DMax (Plan 109) already covers the basic `conf*e_token + (1-conf)*e_mask` pattern
2. DSL-LLaDA threatens the entire "remasking" paradigm — watch closely
3. Training-time components (two-stage, reference model) belong in riir-ai

**Why gain, not skip:**
1. **Idea 3 (MUX-RCD fusion)** is genuinely novel — DDTree multi-path residuals have no paper equivalent
2. **Idea 2 (tier-adaptive)** respects our plasma/hot/warm/cold architecture perfectly
3. **Idea 1 (ResidPruner)** leverages our unique `ConstraintPruner` infrastructure — no one else has this
4. The entropy computation already exists in `dllm_solver.rs` — minimal new code
5. 4-5× step reduction at equivalent accuracy is game-changing for our throughput targets

### What Goes Where

- **katgpt-rs (modelless)**: Entropy-weighted soft embedding injection, ResidPruner, tier-adaptive routing, MUX-RCD fusion, spectral budget allocation
- **riir-ai (model-based)**: Reference model training, two-stage pipeline, temperature calibration `T_res`, LoRA adapter for RCD consumption

---

## TL;DR

RCD is a **GAIN** for katgpt-rs. The paper's core idea (entropy-weighted residual injection) maps directly to our existing DMax/SC-draft/MUX infrastructure. But the real value is in **novel fusions** not in the paper:
1. ResidPruner — constraint-filtered residuals (our unique moat)
2. Tier-adaptive — plasma skips, cold gets full RCD
3. MUX-RCD — multi-path DDTree residuals (no paper equivalent)
4. TF-RCD — ODE-refined residuals
5. Spectral budget — smarter allocation than scalar entropy

The basic residual injection is ~50-100 lines on top of existing `dllm_solver.rs`. The novel fusions add another ~200 lines total. High ROI.

Watch DSL-LLaDA (`2606.01024`) — if continuous soft masking becomes standard, the residual recycling paradigm may become a special case. But for now, RCD is the best inference-time dLLM improvement available.
