# Plan 169: GDSD Advantage-Guided Pruner (Modelless Distillation)

**Date:** 2026-06-02
**Source:** Research 151 — GDSD Guided Denoiser Self-Distillation
**Status:** In Progress
**Feature Gate:** `gdsd_distill`
**Dependencies:** `bandit`

---

## Goal

Implement `GdsdPruner` — a `ScreeningPruner` that applies GDSD-style advantage-guided self-distillation to DDTree branch scoring. Instead of matching denoiser logits (paper's approach), we match pruner relevance scores to an advantage-weighted teacher pruner.

## Architecture

```
GdsdPruner<P>
├── inner: P                           // base ScreeningPruner (e.g., SdarBanditPruner)
├── ref_pruner: P                      // reference pruner (e.g., NoScreeningPruner)
├── beta: f32                          // KL regularization (default: 0.001)
├── psi: f32                           // guidance coefficient (default: 10.0)
├── advantage_fn: fn(f32) -> f32       // A(action) from bandit/arena (fn pointer, zero alloc)
├── tlc: bool                          // token-level centralization (default: true)
└── advantage_mean: f32                // running mean for TLC centralization
```

### Relevance Function

```rust
fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
    let r_old = self.inner.relevance(depth, token_idx, parent_tokens);
    let r_ref = self.ref_pruner.relevance(depth, token_idx, parent_tokens);
    let advantage = (self.advantage_fn)(r_old);
    let centered = if self.tlc { advantage - self.advantage_mean } else { advantage };
    let teacher = (1.0 - self.beta) * r_old + self.beta * r_ref + self.psi * centered;
    teacher.clamp(0.0, 1.0)
}
```

### Common Advantage Functions

- `identity_advantage(x) = x` — raw relevance as advantage
- `sigmoid_advantage(x) = σ(x)` — bounded (0,1), good for Q-values
- `tanh_advantage(x) = tanh(x)` — bounded (-1,1), good for centered signals
- `clamped_advantage(x) = clamp(x, -1, 1)` — simple bounded

## Tasks

### Phase 1: Core Pruner
- [x] Create `katgpt-rs/src/pruners/gdsd.rs` with `GdsdPruner<P: ScreeningPruner>` struct
- [x] Implement `ScreeningPruner` trait for `GdsdPruner`
- [x] Add `GdsdConfig` with defaults (ψ=10.0, β=0.001, tlc=true)
  - Also `.no_tlc()`, `.strong()` (ψ=20, β=0.01), `.mild()` (ψ=1, β=0.0001) presets
- [x] Add feature gate `gdsd_distill = ["bandit"]` to `Cargo.toml`
- [x] Add 4 common advantage functions (identity, sigmoid, tanh, clamped)
- [x] `teacher_signal()` method exposed for testing
- [x] 20 unit tests (config, relevance, teacher signal, advantage functions, TLC, clamping, accessors)

### Phase 2: TLC Utility
- [x] Add `token_logit_centralization()` to `gdsd.rs` as utility function
- [x] Input: `&mut [f32]` logits, output: centralized (subtract mean), returns mean
- [x] O(V) serial — no rayon needed (tiny workload per optimization.md)
- [x] 4 TLC tests (empty, single, uniform, mixed)

### Phase 3: Integration with Existing Pruners
- [ ] Wire `GdsdPruner` as wrapper around `SdarBanditPruner` (SDAR provides advantage signal)
- [ ] Wire `GdsdPruner` as wrapper around `DeltaBanditPruner` (Hint-δ as advantage)
- [ ] Add `build_dd_tree_gdsd()` variant to `dd_tree.rs` that uses `GdsdPruner`

### Phase 4: GOAT Proof
- [ ] Create `tests/bench_gdsd_modelless.rs` — benchmark DDTree accuracy with/without GDSD
- [ ] Test with Bomber Arena (1000 rounds) — GdsdPruner vs SdarBanditPruner baseline
- [ ] Test with Go Arena (9×9, 5 games) — GdsdPruner vs DeltaBanditPruner baseline
- [ ] Report: accumulated-valid percentage, win rate, training stability

### Phase 5: Default-On Decision
- [ ] If GOAT proof shows gain AND no perf regression → promote to default-on
- [ ] If GOAT proof shows no gain → mark as infrastructure-only, document negative result
- [ ] If GOAT proof shows regression → revert, document failure mode

## Optimization Compliance

- **Zero alloc in hot path:** `relevance()` computes from pre-stored values. No `Vec`, no `Box` allocation in the hot path.
- **Serial TLC:** O(V) where V ≤ 128 (micro config). Too small for rayon. Serial is correct per optimization.md.
- **No GPU needed:** This is modelless. Pure CPU, pure arithmetic.
- **Pre-compute advantage:** Advantage from bandit is cached, not recomputed per call.
- **fn pointer advantage_fn:** Zero-size, no heap allocation.

## Implementation Summary (Phase 1-2 complete)

### Files
- `katgpt-rs/src/pruners/gdsd.rs` — 518 lines, 20 tests
- `katgpt-rs/Cargo.toml` — `gdsd_distill` feature gate
- `katgpt-rs/src/pruners/mod.rs` — module registration + public exports

### Test Results
- 20/20 tests pass with `--features "gdsd_distill"`
- Default build clean

## Module Structure

```
katgpt-rs/src/pruners/gdsd.rs          # GdsdPruner<P> + GdsdConfig + advantage functions + TLC
katgpt-rs/tests/bench_gdsd_modelless.rs # GOAT proof (Phase 4)
```
