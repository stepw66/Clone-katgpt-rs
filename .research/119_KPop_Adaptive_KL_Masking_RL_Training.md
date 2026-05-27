# Research 119: KPop — Adaptive Binary KL Masking for RL Training Stability

**Source:** [KPop: Taming Training–Inference Mismatch in RL with Adaptive Masking Regions](https://ringtech.notion.site/kpop) (Ant Group + NUS, May 2026)

**Related:** IcePop (predecessor), Research 054 (ASFT), Research 038 (SDAR), Research 075 (Data Gate), Research 117 (GKD)

## TL;DR

Replaces IcePop's uniform fixed-ratio constraint [α,β] on policy probability ratios with **symmetric binary KL divergence** masking. Single hyperparameter φ. Achieves 76 on SWE-bench-Verified with pure RL on 1T MoE model. Key finding: **70-80% of token gradients can be dropped without hurting training**.

## Core Mechanism

### IcePop (Before)
```
M_IcePop(t) = 1[α ≤ π_train(yt)/π_infer(yt) ≤ β]
```
- Fixed ratio bounds, same for all tokens
- Over-masks low-probability tokens (noise is higher there)
- Under-masks as training-inference gap widens

### KPop (After)
```
D_KL^B(π_train(yt) || π_infer(yt)) = π_train(yt) * log(π_train/π_infer) + (1-π_train) * log((1-π_train)/(1-π_infer))

M_KPop(t) = 1[D_KL^B(π_train||π_infer) ≤ φ] * 1[D_KL^B(π_infer||π_train) ≤ φ]
```
- **Binary KL**: treats full vocabulary as 2-event partition (this token vs. everything else)
- **Symmetric**: both forward AND reverse must satisfy threshold
- **Adaptive**: low-prob tokens get wider tolerance, high-prob tokens get tighter
- **Single φ parameter** (tested: 0.1, 0.5, 0.75, 2.0 — robust across range)

### Why It Works
1. Low-probability tokens have inherently higher noise → fixed ratio over-masks them
2. Symmetric constraint prevents "one-sided leakage" where π_infer >> π_train passes forward-only check
3. Acceptance band is naturally tighter around diagonal at all divergence levels
4. Masking ratio dynamically scales with train/infer gap (10-30% masked, vs IcePop's 0.2%)

## Key Results

| Benchmark | KPop vs IcePop | Notes |
|-----------|---------------|-------|
| AIME25 | +2-3 pts | φ=0.75 best for math |
| HMMT25-Nov | +2-4 pts | Consistent improvement |
| ARC-AGI-2 | +1-2 pts | φ=2.0 (looser) better for logic |
| LiveCodeBench | +3-5 pts | Coding tasks benefit from looser φ |
| SWE-bench-Verified | 76.28% | 1T MoE, pure RL, no SFT |

## Distillation to Our Architecture

### What We Already Have (Overlaps)

| Component | Ours | KPop Equivalent | Gap |
|-----------|------|-----------------|-----|
| Symmetric KL | `KlBoundaryAligner` (Plan 085, `federation` feature) | Symmetric binary KL for masking | We use full-vocab KL; KPop uses binary (2-event) KL |
| Token masking | `ConstraintPruner` / `ScreeningPruner` | `M_KPop(t)` binary mask | Our masking is inference-time; KPop is training-time |
| Adaptive gating | `DataGate` (Plan 111, riir-ai) | φ threshold | DataGate gates samples; KPop gates tokens within a sample |
| Asymmetric trust | SDAR (Plan 073, riir-ai) | Symmetric KL constraint | SDAR: teacher-student trust; KPop: train-infer trust |

### What We'd Need (Gaps)

| Component | Status | Effort | Priority |
|-----------|--------|--------|----------|
| Binary KL divergence fn | ❌ New | ~30 lines | Trivial |
| Online RL training loop | ❌ No GRPO/PPO infra | Major | Out of scope |
| Separate train/infer engines | ❌ Single engine | Major | Out of scope |
| MoE model support | ❌ Dense only | Major | Out of scope |

## Verdict: ⚠️ NO GAIN — Store for Future Reference

### Why No Gain Now

1. **We don't do online RL training.** KPop addresses train/inference mismatch in GRPO/PPO policy gradient training with separate train/infer engines. Our modelless distillation is offline — no policy gradient, no train/infer split.

2. **We don't have MoE models.** KPop's primary use case is 100B-1T MoE models where routing mismatch amplifies train/infer divergence. We target dense models.

3. **Our existing infrastructure covers the insight.** The "70-80% of tokens are redundant" finding validates our existing ConstraintPruner/ScreeningPruner philosophy. We already mask/filter tokens at inference time. KPop confirms this is sound.

4. **Binary KL is simpler than what we have.** Our `KlBoundaryAligner` uses full-vocab symmetric KL. Binary KL (2-event partition) is a simplification we could adopt trivially if needed, but we have no use case for it currently.

### When This Would Become Relevant

| Condition | Trigger | Action |
|-----------|---------|--------|
| We add GRPO/RL training | Game LoRA needs online RL | Implement KPop masking in riir-ai training loop |
| We add MoE support | Model scaling to MoE | KPop becomes critical for stability |
| We add train/infer split | Production RL at scale | Binary KL masking for policy gradient |

### Cross-Reference to Decision Matrix (27_mmo_goat_pillars_decision_matrix.md)

- **Not a GOAT Pillar:** Fails MMO-product criterion. Not game-specific.
- **Not a cross-cutting improvement:** Addresses a problem we don't have.
- **Category:** "Store for future reference" — same tier as GKD (Research 117).
- **If we add online RL for game LoRA → promote to cross-cutting improvement.**

## Key Insight for Our Architecture

> "We only update 70~80% of tokens, and the training still converges well. This suggests that a large fraction of token-level gradients are either redundant or noisy, and selectively dropping them does no harm."

This validates our modelless distillation philosophy (GFlowNet, ROPD, SDAR modelless variants) — being selective about which tokens/positions to learn from is not just acceptable, it's *preferable*. Our `ScreeningPruner::relevance()` and `ConstraintPruner::is_valid()` serve the same purpose at inference time.

The binary KL formulation is elegant for its simplicity — a 2-event partition is O(1) per token vs O(V) for full KL. If we ever need token-level trust scoring, this is the right formulation.

## Citation

```bibtex
@misc{KPop2026,
  title = {KPop: Taming Training–Inference Mismatch in Reinforcement Learning with Adaptive Masking Regions},
  url = {https://ringtech.notion.site/kpop},
  author = {Jia Guo*, Yan Sun*, Zhenyu Huang, Zihao Wang, Zujie Wen, Zhiqiang Zhang, Jun Zhou, Stanley Kok},
  year = {2026},
  month = {May}
}
```
