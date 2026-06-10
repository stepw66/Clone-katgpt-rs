# Research 124: ViT³ — Vision Test-Time Training (ViTTT)

> **Paper:** [ViT³: Unlocking Test-Time Training in Vision](https://arxiv.org/pdf/2512.01643) — Han et al. (Tsinghua · Alibaba), CVPR 2026 Oral
> **Code:** [github.com/LeapLabTHU/ViTTT](https://github.com/LeapLabTHU/ViTTT) (PyTorch)
> **Date:** 2026-04, distilled 2026-05
> **Raw code:** `.raw/ViTTT/`
> **Related Research:** 019 (TTT-Discover), 070 (GDN2), 028 (HLA), 073 (LT2), 034 (D2F), 055 (Tri-Mode)
> **Verdict: ⚠️ NO GAIN — Vision-specific TTT design study. Our text/game AI stack already covers the applicable insights. Two validation points for existing choices, no new algorithm to implement.**

---

## TL;DR

ViT³ systematically studies the TTT design space for **vision** tasks, distilling 6 insights that yield a pure TTT architecture (O(N) complexity) competitive with Mamba and linear attention variants on ImageNet, COCO, ADE20K, and DiT generation. The architecture uses two inner modules per block: simplified SwiGLU (FC ⊙ SiLU(FC)) for most heads + 3×3 depthwise convolution for one head.

**Why NO GAIN for us:**
1. **Vision-specific** — full-batch non-causal inner training, DWConv inner models, spatial locality exploitation. All 6 insights target image data modalities.
2. **We already have linear attention** — GDN2 (R070, P105) provides gated erase/write linear attention with more expressive channel-wise control. HLA (R028) provides higher-order moments. LT2 (P108) provides looped inference.
3. **Causal vs non-causal mismatch** — ViTTT explicitly shows sequential mini-batch (causal) underperforms full-batch for vision. Our language/game domain IS causal. The paper's best configurations don't apply.
4. **No game-domain connection** — no path to GOAT pillar, no modelless distillation angle, no LoRA training improvement.
5. **Inner module design doesn't transfer** — simplified SwiGLU + DWConv are optimized for 2D spatial data. Our token sequences don't have spatial structure.

**Two validation points (already implemented, no action needed):**
- **Insight 1** (loss selection via mixed second derivative ∂²L/∂V̂∂V ≠ 0) validates our dot-product loss choice in GDN2
- **Insight 4/5** (wider not deeper inner models) validates our shallow-wide HLA design (d² moments, not deeper MLP)

---

## Paper Core

### TTT Mechanism

TTT reframes attention as online learning:
1. Compress (K, V) into inner model weights W via self-supervised training
2. Apply updated W* to queries Q → output O

```
V̂_B = F_W(K_B),  W ← W − η · ∂L(V̂_B, V_B)/∂W,  O = F_W*(Q)
```

### Six Insights

| # | Insight | Evidence | Applies to us? |
|---|---------|----------|----------------|
| 1 | Loss functions with vanishing ∂²L/∂V̂∂V are bad | MAE (L1) → 76.5%, Dot-product → 78.9% | ✅ Validates GDN2 dot-product loss |
| 2 | Single epoch full-batch works for vision | B=N → 78.9%, B=N/4 → 78.1% | ❌ We're causal (sequential) |
| 3 | Inner LR = 1.0 is effective | η=1.0 → 78.9%, η=0.1 → 77.5% | ⬜ Already tunable in GDN2 |
| 4 | Wider inner models → consistent gains | r=1→78.9%, r=4→79.6% | ✅ Validates HLA wide design |
| 5 | Deeper inner models → optimization difficulty | 2-layer→78.9%, 3-layer→77.5% | ✅ Validates shallow HLA |
| 6 | Conv inner models best for vision | DWConv → 80.1% vs MLP → 78.9% | ❌ No spatial structure in text |

### ViT³ Architecture

Per TTT block, two inner modules:
1. **Simplified SwiGLU** `F₁ = FC(x) ⊙ SiLU(FC(x))` — used for (H-1) attention heads
2. **3×3 Depthwise Conv** `F₂ = DWConv(x)` — used for 1 attention head

Inner training: 1 epoch, full-batch GD, LR=1.0, dot-product loss, gradient clipping.

### Results (ImageNet-1K)

| Model | Type | Params | FLOPs | Top-1 |
|-------|------|--------|-------|-------|
| H-ViT³-S | TTT | 54M | 8.8G | 84.4 |
| VMamba-S | Mamba | 50M | 8.7G | 83.6 |
| MILA-S | Linear | 43M | 7.3G | 84.4 |
| DeiT-S | Transformer | 22M | 4.6G | 79.8 |
| ViT³-S | TTT | 24M | 4.8G | 81.6 |

At 1248² resolution: 4.6× speedup over DeiT-T, 90.3% memory reduction.

---

## Comparison with Our Stack

| Aspect | ViTTT | GDN2 (R070) | HLA (R028) | LT2 (P108) |
|--------|-------|-------------|------------|------------|
| **Domain** | Vision (2D spatial) | Language (1D causal) | Language (1D) | Language (1D) |
| **Inner model** | SwiGLU + DWConv | Gated erase/write | Moment projection | Loop wrapper |
| **Complexity** | O(N) | O(N) per step | O(d²) per layer | O(N) per loop |
| **State** | Learned W (d×d) | Recurrent S (d×d) | Covariance Σ (d×d) | Shared weights |
| **Causal?** | No (full-batch) | Yes (recurrent) | Yes (cumulative) | Yes (sequential) |
| **Game relevance** | ❌ | ✅ (decode path) | ✅ (context encoding) | ✅ (quality boost) |

### Key Insight: TTT = Generalized Linear Attention

The paper frames TTT as generalizing linear attention:
- **Linear attention** = compress K,V into d×d linear weights W = KᵀV
- **TTT** = compress K,V into arbitrary nonlinear module weights via online learning
- **Softmax attention** = N-width MLP with Softmax activation

This framing is interesting but already captured in our architecture:
- GDN2's recurrent state S_t = linear attention with gated erase/write
- HLA's covariance Σ = linear attention with higher-order moments
- Both are "inner models" in TTT parlance, just without the online learning loop

The online learning loop (gradient step at test time) is the novel part, but:
- We don't do test-time gradient updates (production constraint)
- LT2 (P108) provides the loop mechanism without gradients
- D2F (P066) provides the diffusion mechanism for parallelism

---

## What Does NOT Map

| ViTTT Concept | Why Not |
|---------------|---------|
| **DWConv inner model** | No 2D spatial structure in token sequences |
| **Full-batch inner training** | Causal language = sequential, not parallel |
| **Non-causal mode** | Game AI is inherently sequential (turns, ticks) |
| **Vision benchmarks** | No overlap with game AI quality metrics |
| **4× inner compute tax** | Inner backward pass costs 4× forward — unacceptable for real-time game AI at 20Hz |
| **Image generation (DiT³)** | We don't do diffusion image generation |
| **Gradient clipping for inner loop** | GDN2 uses learned gates instead — more efficient |

---

## Verdict

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| **GOAT proof potential** | ❌ | Vision-only paper, no path to game AI proof |
| **MMO-product** | ❌ | No game-domain application |
| **Modelless relevance** | ❌ | TTT is inherently model-based (inner model + training) |
| **Defensible** | ❌ | Public CVPR oral, open-source code |
| **Perf impact** | ⬜ Neutral | No change to existing stack |
| **Action needed** | ❌ | Research only, no plan, no feature gate |

**Bottom line:** ViTTT is an excellent vision TTT study (CVPR oral quality) but has zero transfer to our text/game AI system. The two applicable insights (loss selection, shallow-wide design) are already validated by our existing GDN2 and HLA implementations. **No plan, no feature gate, research only.**

---

## Citation

```bibtex
@inproceedings{han2026vit,
  title={ViT$^3$: Unlocking Test-Time Training in Vision},
  author={Han, Dongchen and Li, Yining and Li, Tianyu and Cao, Zixuan and Wang, Ziming and Song, Jun and Cheng, Yu and Zheng, Bo and Huang, Gao},
  booktitle={CVPR},
  year={2026}
}
```
