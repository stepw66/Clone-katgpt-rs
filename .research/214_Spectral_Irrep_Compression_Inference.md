# Research: Spectral Irrep Compression for Inference-Time Token Routing

**Paper:** Neural Networks Provably Learn Spectral Representations for Group Composition (arXiv:2606.02993)
**Date:** 2025-06
**Target:** katgpt-rs (modelless inference engine)

---

## Paper Summary

Two-layer neural networks trained on group composition tasks provably learn **irreducible representations (irreps)** of the underlying group via gradient flow. Key proven results:

1. **Single Representation**: Each neuron converges to exactly one non-trivial irrep ρ̃_m (all other spectral components vanish)
2. **Rank-One Rotational Alignment**: Fourier coefficients become rank-1 and cross-layer aligned: ξ̂_m[ρ] ∝+ θ̂₂_m[ρ] · θ̂₁_m[ρ]
3. **Lottery Ticket Mechanism**: Winning irrep determined at random initialization by argmax ρ α_m[ρ](0)
4. **Exponential Convergence**: Phase alignment and representation competition both converge exponentially
5. **Diversification**: Ensemble uniformly covers all non-trivial irreps → majority-vote predictor

The training dynamics decompose into:
- **Stage I**: Feature learning — directional parameters converge on spectral manifold (Riemannian gradient ascent of energy functional Ω_m)
- **Stage II**: Scale growth — a(t) grows logarithmically, sharpening softmax to zero loss

## Novel Fusion Ideas for katgpt-rs

### 1. Irrep Pruner: Spectral Collapse Detection as ConstraintPruner

**Core insight**: The paper proves that valid converged neurons have rank-1 Fourier coefficients. We can use this as an inference-time *pruning signal* without any model training.

**Application**: Implement `IrrepPruner` that checks if logits exhibit spectral concentration (analogous to single-irrep structure). If the logit distribution over token vocabulary shows single-mode dominance (high spectral flatness → collapse), prune branches that are "still competing" (multi-frequency noise).

**Concrete mapping**:
- Group elements g ∈ G → tokens in vocabulary V
- Composition g₁ ⋆ g₂ → token sequence (context + draft)
- Irreps ρ ∈ Irr(G) → spectral modes of the logit distribution over V
- Single-irrep convergence → logit entropy collapse to a few modes
- Rank-one alignment → consistency between draft and verifier distributions

**Implementation sketch**:
```rust
pub struct IrrepPruner {
    /// Spectral flatness threshold below which we consider "converged"
    convergence_threshold: f32,
    /// Number of dominant spectral modes to keep
    max_modes: usize,
}

impl ConstraintPruner for IrrepPruner {
    fn is_valid(&self, logits: &[f32]) -> bool {
        // FFT of logits → spectral energy distribution
        // Check: is spectral energy concentrated in ≤ max_modes modes?
        // If yes → "converged" → valid branch
        // If no → "still competing" → prune
        let spectrum = fft_energy(logits);
        let flatness = spectral_flatness(&spectrum);
        flatness < self.convergence_threshold
    }
}
```

### 2. Phase-Aligned Drafting: Speculative Decoding via Cross-Layer Fourier Alignment

**Core insight**: The paper proves cross-layer phase alignment: arg(ξ̂_m[ρ]) = arg(θ̂₁_m[ρ]) + arg(θ̂₂_m[ρ]) mod 2π. This is a *structural constraint* that valid predictions must satisfy.

**Application**: For speculative decoding, verify that draft tokens and target verification logits exhibit "phase alignment" — i.e., their spectral decompositions are consistent. If draft spectrum and verifier spectrum are misaligned (high imaginary component of cross-spectrum), reject the draft.

**This is a modelless inference-time signal** — no training needed.

### 3. Lottery Ticket Routing: Bandit-Based Spectral Mode Selection

**Core insight**: The winning irrep is determined by `argmax_ρ α_m[ρ](0)` — the initially dominant spectral mode wins. This is a lottery ticket mechanism.

**Application**: At inference time, use a multi-armed bandit to track which "spectral modes" (frequency bins of logit distribution) are most reliable across past inference steps. Initialize bandit arms with spectral energy at each frequency. The bandit converges to tracking the dominant modes — analogous to the lottery ticket mechanism but at inference time.

**Concrete mapping to existing infrastructure**:
- Reuses `FreqBandit` (plan 189) infrastructure
- Each arm = spectral mode (frequency bin) of logit distribution
- UCB1 score = spectral energy × alignment quality
- Winner-take-all routing = lottery ticket at inference time

### 4. Majority-Vote Ensemble Pruning via Spectral Diversification

**Core insight**: The paper proves that diversified ensembles achieve perfect accuracy via majority vote: noise cancels, signal accumulates. The predictor is a "flawed indicator" — correct label gets coefficient 2, ghost labels get 1, baseline gets -4/|G|.

**Application**: When running multiple speculative draft branches (M branches), treat them as a "diversified ensemble". Compute majority-vote logit: sum all branch logits, weight by spectral alignment quality. The resulting ensemble should be more robust than any individual branch.

**This generalizes existing `SpeculativeGenerator::generate_batch()`** to a spectral-weighted ensemble.

## Verdict

### GOAT Assessment

| Criterion | Score | Notes |
|-----------|-------|-------|
| Modelless (no training) | ✅ 9/10 | All ideas are inference-time only |
| Lands in katgpt-rs domain | ✅ 9/10 | Extends ConstraintPruner/SpeculativeGenerator traits |
| Compatible with existing infra | ✅ 8/10 | Reuses FFT, bandit, pruner infrastructure |
| Performance gain potential | ⚠️ 6/10 | FFT overhead on every token may be costly; need benchmarks |
| Novelty vs existing research | ✅ 8/10 | No prior irrep-based pruning in codebase |

### Decision: **GAIN (conditional)**

The spectral collapse detection (IrrepPruner) is the highest-value idea — it's a direct structural signal from proven theory that can be computed at inference time. However, FFT cost per-token needs benchmarking.

**Gate**: Implement IrrepPruner behind feature flag `spectral_pruner`. Benchmark throughput impact before promoting to default.

### What NOT to do

- Don't implement full group DFT — we're working with token vocabularies, not finite groups
- Don't train LoRA for this — the whole point is modelless inference
- Don't implement rank-one matrix decomposition — too expensive per-token
- The irrep theory is inspirational; we use the *structural insights* (convergence → spectral concentration, alignment → cross-spectrum consistency) not the exact math

## References

- arXiv:2606.02993 — Neural Networks Provably Learn Spectral Representations for Group Composition
- Plan 189 — FreqBandit: DFT spectral analysis of token streams
- Plan 077 — SpectralQuant upgrade
- Plan 139 — EGA energy-gated attention
- Research 046 — Symmetry-compatible equivariant optimizers
