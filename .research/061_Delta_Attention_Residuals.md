# Research 61: Delta Attention Residuals

> **Paper:** [Delta Attention Residuals](https://arxiv.org/pdf/2605.18855) — Luo, Cai, Hu (NeurIPS 2026), May 2026
> **Date:** 2026-05, distilled 2025-07
> **Related Research:** 22 (Lighthouse Attention), 28 (HLA), 42 (SP-KV), 31 (Percepta)
> **Related Plans:** 097 (Delta Attention Residuals)
> **Verdict: SELECTIVE ADOPTION — Feature-gated delta routing for inference quality at depth ≥6. Gains scale with layer count; zero benefit at n_layer=1. Delta Block variant only (0.008% extra params, ~20% throughput overhead). Principle aligns with δ-Mem and Deep Manifold residual scoring.**

---

## TL;DR

In cross-layer routing (Attention Residuals), routing over *cumulative* hidden states hᵢ = h₀ + Σvⱼ causes "routing collapse" in deep layers because adjacent states share most components. Solution: route over *deltas* vᵢ = hᵢ₊₁ − hᵢ (what changed) instead. Combined with *additive* routing (h = h̃ + Σαᵢ·vᵢ) instead of *replacement* routing (h = Σαᵢ·sᵢ), this preserves the residual stream, enables zero-init safe fine-tuning, and achieves 3× sharper routing (max weight ~0.6 vs ~0.2). Delta Block variant: −8.2% PPL at 7.6B with only 0.008% extra params. Our system already uses additive residuals (`x += xr` / `x += xr2`), and our `ForwardContext` buffers (`xr`, `xr2`, `attn_out`, `hidden`) implicitly compute the per-sublayer deltas. Recommendation: feature-gate as `delta_routing`, implement Delta Block variant, benchmark at n_layer≥6 configs.

---

## Paper Architecture

### The Problem: Routing Collapse

Standard Attention Residuals route over cumulative hidden states:

```
hᵢ = h₀ + Σⱼ₌₀ⁱ⁻¹ vⱼ    (residual stream accumulates all sublayer outputs)
```

For deep layers (i → L), hᵢ ≈ hᵢ₊₁ because the residual stream is dominated by early contributions. Routing weights collapse to near-uniform (max α ≈ 0.2) → layers can't discriminate which sources matter.

### The Solution: Delta Routing

Route over *deltas* (what changed at each layer) instead of cumulative states:

```
vᵢ = hᵢ₊₁ − hᵢ    (the actual contribution of layer/sublayer i)
```

This makes each routing source *distinct* — adjacent deltas are decorrelated by construction. Routing weights become 3× sharper (max α ≈ 0.6).

### Two Variants

| Variant | Delta Granularity | Extra Params | Throughput Overhead | Memory Overhead | PPL Gain (7.6B) |
|---------|-------------------|-------------|--------------------|----------------|------------------|
| **Delta AttnRes** (per-sublayer) | Per attention + MLP sublayer | 4 × n_layer × n_embd | ~69% | ~3.5× | −8.5% |
| **Delta Block** (per-block of B layers) | Per block of B layers | 4 × (n_layer/B) × n_embd | ~20% | ~26% | −8.2% |

Delta Block is the practical default: 85% of the quality gain at 30% of the cost.

### Core Algorithm: Delta Block

```
For each block b of B consecutive layers:
  1. Run B layers normally (standard residual stream)
  2. Compute block delta: Δb = h_current − h_prev_block   (what changed in this block)
  3. Store Δb as a routing source

Routing at each layer:
  1. Normalize stored deltas: ŝᵢ = RMSNorm(Δᵢ)
  2. Score against learned query: scoreᵢ = dot(q, ŝᵢ)    (q is zero-initialized)
  3. Softmax over scores: α = softmax(scores / √d)
  4. Additive update: h = h_current + Σᵢ αᵢ · Δᵢ
```

**Key design choices:**
- **Additive routing** (not replacement): preserves the standard residual stream; delta routing is a *correction term*
- **Zero-initialized query vectors**: at initialization, α → uniform, routing is identity → safe fine-tuning from pretrained weights
- **RMSNorm on deltas**: prevents magnitude drift across layers

### Why Deltas Work

| Property | Cumulative States (hᵢ) | Delta States (vᵢ) |
|----------|------------------------|---------------------|
| Adjacent similarity | High (shared history) | Low (different sublayer functions) |
| Max routing weight | ~0.2 (near-uniform) | ~0.6 (3× sharper) |
| Collapse in deep layers | Yes | No |
| Training stability | Requires careful init | Zero-init safe |

---

## Key Results

### Perplexity (WikiText-2, main paper tables)

| Model Size | Layers | Baseline PPL | + Delta AttnRes | + Delta Block |
|-----------|--------|-------------|----------------|---------------|
| 220M | 12 | 15.42 | 14.67 (−4.9%) | 14.72 (−4.5%) |
| 1.1B | 22 | 11.28 | 10.58 (−6.2%) | 10.63 (−5.8%) |
| 3.4B | 28 | 9.14 | 8.52 (−6.8%) | 8.57 (−6.2%) |
| 7.6B | 36 | 8.01 | 7.33 (−8.5%) | 7.35 (−8.2%) |

**Key observation:** Gains scale monotonically with depth. At L=12 (220M), −4.9%. At L=36 (7.6B), −8.2%.

### Fine-Tuning Benchmarks (8-task average)

| Method | Avg Accuracy | Δ vs Baseline |
|--------|-------------|---------------|
| Baseline (full fine-tune) | 72.4% | — |
| + Delta Block | 73.0% | **+0.6%** |
| + Delta AttnRes | 73.2% | +0.8% |

Delta routing adds +0.6–0.8% accuracy on top of standard fine-tuning with zero-init safe conversion.

### Overhead Analysis (Delta Block)

| Metric | Value |
|--------|-------|
| Extra parameters | 0.008% of model |
| Throughput overhead | ~20% vs baseline |
| Memory overhead | ~26% vs baseline |
| Compare: Delta AttnRes throughput | ~69% overhead |
| Compare: Delta AttnRes memory | ~3.5× overhead |

### Routing Sharpness

| Metric | Cumulative Routing | Delta Routing |
|--------|--------------------|---------------|
| Max routing weight | 0.18 ± 0.03 | 0.61 ± 0.07 |
| Entropy (bits) | 4.2 ± 0.3 | 1.8 ± 0.2 |
| Effective sources used | ~16 | ~3.5 |

---

## Mapping to Our Stack

### What We Already Have (Structural Alignment)

| Delta AttnRes Concept | Our Implementation | Match |
|----------------------|-------------------|-------|
| Additive residuals: h += v | `simd_add_inplace(&mut ctx.x, &ctx.xr)` (L641) | ✅ Exact |
| Pre-attention residual save | `ctx.xr[..n].copy_from_slice(&ctx.x[..n])` (L571) | ✅ Exact |
| Post-attention residual save | `ctx.xr2[..n].copy_from_slice(&ctx.x[..n])` (L644) | ✅ Exact |
| Attention output (attn delta) | `ctx.attn_out` (attention_head output, L618-636) | ✅ Exact |
| MLP hidden (MLP delta) | `ctx.hidden` → `mlp_w2` → `ctx.x - ctx.xr2` (L651-673) | ✅ Exact |
| Pre-allocated buffers | `ForwardContext` with `xr`, `xr2`, `attn_out`, `hidden` | ✅ Exact |
| Dot product for scoring | `simd::simd_dot_f32` (already used in attention) | ✅ Exact |
| RMSNorm | `rmsnorm(&mut ctx.x)` (used per sublayer) | ✅ Exact |

### Where Delta Block Inserts

Our `forward_base()` in `transformer.rs` (L547-720) already has the structure:

```text
L571:  ctx.xr[..n].copy_from_slice(&ctx.x[..n]);   // ← pre-attention residual
L618:  ctx.attn_out[..n].fill(0.0);                 // ← attention delta computed here
L641:  matmul(&mut ctx.x, &attn_wo, &ctx.attn_out); // ← output projection
L642:  simd_add_inplace(&mut ctx.x, &ctx.xr);       // ← x += residual (additive!)
L644:  ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);   // ← pre-MLP residual
L651:  matmul_relu(&mut ctx.hidden, &mlp_w1, ...);   // ← MLP delta computed here
L672:  simd_add_inplace(&mut ctx.x, &ctx.xr2);      // ← x += residual (additive!)
```

**Per-sublayer deltas are already computed implicitly:**
- Attention delta = `attn_wo @ attn_out` (the projected attention output)
- MLP delta = `mlp_w2 @ hidden` (the projected MLP output)

These are the quantities added to the residual stream at each sublayer. Delta Block just needs to *accumulate* them across B layers and store the block-level delta.

### Delta Block Implementation Sketch

```text
// New buffers needed (per token position):
//   block_deltas: Vec<Vec<f32>>  — (n_layer / block_size + 1) × n_embd
//   prev_block_x: Vec<f32>      — n_embd (h at start of current block)
//   depth_query: Vec<f32>       — n_embd × (n_layer / block_size) (zero-init)
//   depth_rms_weight: Vec<f32>  — n_embd × (n_layer / block_size)

// In forward_base(), at block boundaries (layer_idx % block_size == 0):
//   1. Compute delta: Δb = ctx.x - prev_block_x
//   2. Store: block_deltas[bi] = Δb
//   3. Save: prev_block_x = ctx.x (snapshot for next block)
//   4. Route: normalize deltas → score → softmax → additive update
```

### Memory Overhead Estimation

| Config | n_layer | block_size (B) | # Block Deltas | Extra Memory |
|--------|---------|---------------|----------------|-------------|
| micro | 1 | — | 0 | N/A (no routing possible) |
| small | 4 | 4 | 1 | 1 × n_embd (negligible) |
| medium | 8 | 4 | 2 | 2 × n_embd |
| typical | 12 | 4 | 3 | 3 × n_embd |
| large | 36 | 6 | 6 | 6 × n_embd |

For our typical config (n_layer=12, n_embd=768): 3 × 768 × 4 bytes = 9.2 KB. Negligible.

### Composability with Existing Mechanisms

| Combination | Value | Notes |
|-------------|-------|-------|
| Delta Routing + HLA | Medium | HLA layers are O(1) per-step; delta routing adds cross-block context |
| Delta Routing + SP-KV | Medium | SP-KV handles KV sparsity; delta routing handles cross-layer information flow |
| Delta Routing + Percepta | Low | Both address depth routing; likely redundant |
| Delta Routing + Raven | Orthogonal | Raven is KV cache; delta routing is residual stream |
| Delta Routing + LoRA | High | Zero-init query → safe fine-tuning with LoRA on top |
| Delta Routing + SpectralQuant | Orthogonal | SQ quantizes KV; delta routing adds cross-layer connections |

---

## Verdict

### Why SELECTIVE ADOPTION, Not Full

| Factor | Assessment |
|--------|-----------|
| Our micro config (n_layer=1) | **Zero benefit** — no previous layers to route to |
| Our small configs (n_layer≤4) | **Marginal** — very few routing sources |
| Paper's 220M (n_layer=12) | **−4.9% PPL** — meaningful |
| Paper's 7.6B (n_layer=36) | **−8.2% PPL** — significant |
| Overhead at n_layer≥6 | Acceptable (~20% throughput, ~26% memory) |
| Fine-tuning gains | +0.6% avg accuracy with zero-init conversion |

### When to Implement

1. **Now:** For model-based path (riir-ai LoRA training on larger models). Delta Block + zero-init query vectors = safe fine-tuning upgrade.
2. **When scaling:** If we move to n_layer≥6 configs, feature-gate `delta_routing` becomes worthwhile.
3. **Principle:** The "delta = change" concept aligns with our existing δ-Mem (Research 24) and Deep Manifold residual scoring (Research 51). The additive routing pattern is already our default.

### When NOT to Implement

1. At n_layer=1 (micro config) — there are no previous layers. Routing is undefined.
2. At n_layer≤4 — too few sources for meaningful routing. The overhead isn't justified.
3. For inference-only micro models — the overhead (even ~20%) isn't worth marginal PPL gains.

---

## Extractable Techniques

### E1: Delta Block Routing (Feature-Gated)

**What:** After each block of B layers, store accumulated delta. Route over block deltas using additive routing with zero-init learned queries.

**Why feature-gated:** Only beneficial at n_layer≥6. Controlled by `#[cfg(feature = "delta_routing")]`.

**Implementation:**

```text
microgpt-rs/src/
├── delta_routing/                    # NEW module
│   ├── mod.rs                        # Module index + re-exports
│   ├── types.rs                      # DeltaRoutingConfig, BlockDeltaBuffer, DepthQueryWeights
│   ├── block_delta.rs                # Block delta accumulation + snapshot
│   └── route.rs                      # depth_route(): RMSNorm → dot → softmax → additive update
├── transformer.rs                    # Add delta_routing forward variant + dispatch
```

**Config:**

```rust
pub struct DeltaRoutingConfig {
    pub block_size: usize,         // B: layers per block (default: 4)
    pub n_sources: usize,          // n_layer / block_size + 1
    pub zero_init: bool,           // always true for safe fine-tuning
}
```

### E2: Zero-Init Fine-Tuning Query Vectors

**What:** Learned query vectors for routing, initialized to zero. At init, routing weights → uniform (no-op). During training, queries learn to select informative deltas.

**Why extractable:** This is a general technique for *any* additive cross-layer mechanism. Our existing LoRA fine-tuning pipeline can incorporate zero-init query vectors without risking pretrained weight quality.

**Use case:** riir-ai LoRA training pipeline — add Delta Block routing during fine-tuning for +0.6% accuracy gains on downstream benchmarks.

### E3: Delta as "What Changed" Principle

**What:** The conceptual insight that routing over changes (deltas) is more discriminative than routing over accumulated states.

**Why extractable:** This principle applies beyond transformer layers:
- **δ-Mem (Research 24):** Already uses delta rule for memory updates — validates the approach
- **Deep Manifold (Research 51):** Residual scoring uses boundary conditions — deltas capture "how much the manifold changed"
- **Raven RSM:** Slot updates are delta-based (slot += new_info) — consistent with this principle

**No implementation needed** — this is a conceptual validation of patterns we already use.

---

## What NOT To Do

1. **Don't implement Delta AttnRes (per-sublayer variant).** 69% throughput overhead and 3.5× memory is not justified for our scale. Delta Block captures 96% of the quality at 30% of the cost.
2. **Don't enable delta routing for n_layer≤4 configs.** The feature gate should check config at runtime and silently skip routing when there are insufficient sources.
3. **Don't replace our standard additive residuals.** Delta routing is an *addition* to the residual stream, not a replacement. Our `x += xr` / `x += xr2` pattern stays untouched.
4. **Don't route over cumulative hidden states.** The paper's core finding is that cumulative states cause routing collapse. Always route over deltas.
5. **Don't add auxiliary training losses.** The paper shows next-token prediction alone is sufficient. No auxiliary routing losses needed.

---

## Relationship to Existing Research

| Research | Overlap | Delta |
|----------|---------|-------|
| 22 (Lighthouse Attention) | Cross-layer attention patterns | Delta routing is orthogonal: Lighthouse restructures attention, delta routing restructures residual flow |
| 28 (HLA) | O(1) per-layer compute | HLA reduces attention cost; delta routing adds cross-layer information. Composable |
| 31 (Percepta) | Depth-aware processing | Percepta routes within attention; delta routing routes across layers. Different axis |
| 42 (SP-KV) | KV cache optimization | SP-KV prunes KV pairs; delta routing adds cross-layer connections. Orthogonal |
| 24 (δ-Mem) | Delta-based memory updates | Conceptual alignment: both use "what changed" as the signal of interest |
| 51 (Deep Manifold) | Residual stream analysis | Deep Manifold scores boundary conditions; delta routing uses boundary deltas. Compatible |
| 39 (SpectralQuant) | KV compression | Orthogonal: SQ compresses what's stored; delta routing adds what's routed to |

---

## References

- Paper: https://arxiv.org/pdf/2605.18855
- Conference: NeurIPS 2026
- Related: Attention Residuals (previous work by same group), Lighthouse Attention, Gated Delta Networks