# Plan 129: OPUS-Inspired Boltzmann + Redundancy Selection

**Research**: 089_OPUS_Optimizer_Induced_Projected_Utility_Selection.md
**Status**: ✅ Complete
**Feature Gate**: `opus_selection = ["bandit"]`

---

## Motivation

From OPUS paper (arXiv:2602.05400): Boltzmann sampling with redundancy penalty outperforms greedy top-k by +1.26 avg points on real benchmarks (Table 7). Current `BanditPruner` uses Thompson/UCB/EpsilonGreedy but lacks:
1. Explicit redundancy penalty against already-selected arms
2. Boltzmann (softmax) temperature-controlled sampling
3. Low-dimensional sketch for efficient inner-product estimation

This is the **highest-value distillation** from OPUS — composable, simple, directly improves existing bandit infrastructure without requiring pre-training scale.

## Scope

- [x] **In scope**: OpusBanditPruner<P>, CountSketch primitive, Boltzmann sampler, GOAT proofs
- [x] **Out of scope**: Full OPUS pre-training pipeline, Muon optimizer, Bench-proxy construction, AdamW preconditioner

## Tasks

### T1: CountSketch Primitive
- [x] Create `src/pruners/opus/count_sketch.rs`
- [x] Implement `CountSketch` struct with hash/sign pairs
- [x] `fn sketch(&self, vec: &[f32]) -> Vec<f32>` — O(d) → O(m) projection
- [x] `fn inner_product_estimate(&self, a: &[f32], b: &[f32]) -> f32` — unbiased estimator
- [x] Unit tests: unbiasedness, variance bounds
- [x] Micro-bench: sketch speed vs full inner product

### T2: Boltzmann Sampler
- [x] Create `src/pruners/opus/boltzmann.rs`
- [x] `fn boltzmann_sample(utilities: &[f32], temperature: f32, rng: &mut Rng) -> usize`
- [x] `fn boltzmann_sample_batch(utilities: &[f32], temperature: f32, k: usize, rng: &mut Rng) -> Vec<usize>`
- [x] Temperature τ controls exploration: τ→0 greedy, τ→∞ uniform
- [x] Unit tests: probability distribution validity, edge cases (τ=0, τ=∞, single arm)

### T3: OpusBanditPruner<P>
- [x] Create `src/pruners/opus/types.rs` with `OpusConfig`, `OpusBanditPruner<P>`
- [x] Create `src/pruners/opus/mod.rs` (index only)
- [x] Implement `ScreeningPruner` for `OpusBanditPruner<P>`
- [x] Core scoring: `U_z = alignment - redundancy_weight * ⟨ϕ(z), Φ_selected⟩`
- [x] Maintain running history `Φ_selected` of sketch features
- [x] Use Boltzmann sampling instead of Thompson/UCB for arm selection
- [x] Delegate domain relevance to inner `BanditPruner<P>`

### T4: OpusBanditEnv for Standalone Testing
- [x] Implement `BanditEnv` for a configurable test environment
- [x] Redundant arms: some arms give same reward (test diversity)
- [x] Run `BanditSession` with `OpusBanditPruner` vs `BanditPruner`
- [x] Metric: cumulative reward, regret, diversity (unique arms pulled)

### T5: GOAT Proof — Bandit Benchmark
- [x] Add `tests/test_129_opus_boltzmann_goat.rs`
- [x] Compare: Thompson vs UCB vs OpusBandit on Bernoulli/Gaussian/Redundant scenarios
- [x] Metric: regret convergence, cumulative reward, arm diversity
- [x] Expected: Opus maintains ≥ Thompson reward + higher diversity

### T6: GOAT Proof — DDtree Quality
- [x] Add opus option to `build_dd_tree_screened()` integration via `ScreeningPruner` trait
- [x] Compare tree quality with OpusBanditPruner vs BanditPruner
- [x] Metric: tree coverage, depth efficiency, unique leaves
- [x] Expected: Better coverage from redundancy penalty avoiding duplicate branches

### T7: Feature Gate + Cargo.toml
- [x] Add `opus_selection = ["bandit"]` to `katgpt-rs/Cargo.toml` features
- [x] Add `opus_selection` to `full` feature list
- [x] Gate `src/pruners/opus/` module behind `#[cfg(feature = "opus_selection")]`
- [x] Re-export key types from `src/pruners/mod.rs`

### T8: Documentation + Benchmark
- [x] Add `.benchmarks/040_opus_boltzmann_bandit.md`
- [x] Update plan file with completed status
- [x] GOAT proofs: 20/20 tests passing, all targets met

## Key Types

```rust
/// OPUS configuration (paper defaults: τ=0.9, m=8192, ρ=0.5, b_t=64).
pub struct OpusConfig {
    pub temperature: f32,        // τ = 0.9
    pub redundancy_weight: f32,  // λ scaling for redundancy penalty
    pub sketch_dim: usize,       // m = 8192
    pub buffer_size: usize,      // N = 64
    pub selection_ratio: f32,    // ρ = 0.5
}

/// OPUS-inspired BanditPruner with Boltzmann sampling + redundancy penalty.
pub struct OpusBanditPruner<P: ScreeningPruner> {
    inner: BanditPruner<P>,
    config: OpusConfig,
    sketch: CountSketch,
    /// Running history of selected sketch features Φ(t,r).
    selected_history: Vec<Vec<f32>>,
    /// Per-arm last sketch for redundancy computation.
    arm_sketches: Vec<Vec<f32>>,
}
```

## Module Structure

```
src/pruners/opus/
├── mod.rs           # Index only — pub mod count_sketch, boltzmann, types;
├── types.rs         # OpusConfig, OpusBanditPruner<P>, impl ScreeningPruner
├── count_sketch.rs  # CountSketch projection (standalone, reusable)
└── boltzmann.rs     # Boltzmann sampling with redundancy-aware batch selection
```

## GOAT Proof Targets

| Proof | Metric | Target |
|-------|--------|--------|
| P1: Bandit reward | Cumulative reward | ≥ Thompson sampling |
| P2: Bandit diversity | Unique arms pulled | > Thompson sampling |
| P3: Regret convergence | Steps to 95% optimal | ≤ Thompson sampling |
| P4: DDtree coverage | Unique leaves / total | > BanditPruner baseline |
| P5: CountSketch accuracy | Inner product MSE | < 0.01 vs exact |

## References

- Research 089: `.research/089_OPUS_Optimizer_Induced_Projected_Utility_Selection.md`
- OPUS paper: arXiv:2602.05400v2
- Existing bandit: `src/pruners/bandit.rs` (Plan 030)
- CountSketch: Cormode & Muthukrishnan 2005