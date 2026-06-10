# Research 120: PowLU — Stable Activation Function for LLM Pre-Training

**Paper:** [PowLU: An Activation Function for Stable Pre-Training of LLMs](https://arxiv.org/pdf/2605.25704)
**Date:** 2026-05-25
**Authors:** Peijie Jiang et al. (Ant Group, Ling Team)

## Summary

PowLU replaces SwiGLU's quadratic amplification (≈x² for large x) with a rational power function:

```
PowLU(x) = x^(1 + m/(√x + 1)) · sigmoid(x)   when x > 0
         = x² · sigmoid(x)                      when x ≤ 0
```

With default m=3.0, this grows ~linearly for large positive inputs instead of quadratically, reducing outlier channels in forward/backward passes.

## Key Properties

| Property | SwiGLU | PowLU |
|----------|--------|-------|
| Growth for large x | ~x² | ~x¹ (linear) |
| P99 activation range (7.9B MoE) | 0-200 | 0-50 (4× tighter) |
| P99 gradient range | 0-30 | 0-5 (6× tighter) |
| Loss spikes (FP8, 7.9B) | Yes (step ~76K, 77K) | No |
| Scaling law | baseline | Overlapping curves |
| Benchmarks (7.9B) | baseline | Competitive (within ±1pp) |
| Benchmarks (124B) | baseline | Competitive (within ±2pp) |

## Theoretical Properties (Proven)

- **Continuity** at x=0 (left/right limits = 0)
- **Differentiability** at x=0 (both derivatives = 0)
- **Monotonicity** for 0 < m < 10
- **Bounded growth** — ~linear as x→+∞ (vs ~quadratic for SwiGLU)
- **Non-linearity** via variable exponent + sigmoid

## Ablation

| m | Loss (47M MoE, 29.8B tokens) |
|---|------|
| SwiGLU | 1.910 |
| PowLU m=2 | 1.913 (+0.003) |
| PowLU m=3 | 1.912 (+0.002) ← best |
| PowLU m=4 | 1.914 (+0.004) |

m=3 is optimal. Sensitivity is low (±0.002 across m=2-4).

## Verdict: ⚠️ NO GAIN — Training-Only Technique

### Why No Gain

1. **We don't train models.** katgpt-rs and riir-ai are inference-only. PowLU's primary benefit is training stability (loss spike elimination, outlier channel suppression, FP8 training stability).

2. **Inference quality is competitive, not superior.** The paper shows PowLU matches SwiGLU at inference time — scaling law curves overlap. No inference quality gain.

3. **Pre-trained models use SwiGLU.** Gemma 2, LLaMA, Mistral — all ship with SwiGLU/GeGLU weights. We can't swap activation functions without retraining from scratch.

4. **Sparse MLP is ReLU-based.** Our `sparse_mlp` feature exploits dead ReLU neurons (~50% zeros). PowLU, like SwiGLU, doesn't produce clean zeros — would not enhance our sparse path.

5. **No game domain relevance.** Not a GOAT pillar candidate. No interaction with any of the 4 MMO pillars or 2 gaps.

### What We Already Have

- `GateActivation` enum in `coda.rs` with `Silu` (SwiGLU), `GegeluTanh` (Gemma 2), `Gegelu`, `Relu`
- `simd_matmul_rmsnorm_swiglu` fused kernel for SwiGLU/GeGLU
- Training stability not our concern — we consume pre-trained weights

### If We Ever Train Models (Future Reference)

- PowLU trivially implementable as `GateActivation::Powlu { m: f32 }` — ~10 lines
- Would benefit FP8 training on GPU (riir-ai wgpu path)
- m=3 is safe default; sensitivity is low
- **Store for future**: if riir-ai adds training, PowLU should be the default activation for new model training

## Relevance to GOAT Pillars

| Pillar | Relevance |
|--------|-----------|
| Fourier Spatial AI | None |
| WASM Validators | None |
| NPC Dialog Engine | None |
| Frame-Sampling Bridge | None |
| Cold Tier | None |
| MMO Backbone | None |
| Cross-Cutting (LEO/Analogy/Sleep/PEIRA/Quest) | None |

## Classification

- **katgpt-rs:** No plan needed. Research stored for reference.
- **riir-ai:** No plan needed. If game LoRA training is added, consider PowLU as default activation.
- **GOAT status:** Not applicable — training-only, no inference gain.
