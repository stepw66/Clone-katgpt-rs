# Plan 169: GDSD Advantage-Guided Pruner (Modelless Distillation)

**Date:** 2026-06-02
**Source:** Research 151 — GDSD Guided Denoiser Self-Distillation
**Status:** ❌ GOAT FAILED — structural tests pass (7/7), gain test missing (0/1). Infrastructure only.
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

### Phase 1: Core Pruner ✅
- [x] Create `katgpt-rs/src/pruners/gdsd.rs` with `GdsdPruner<P: ScreeningPruner>` struct
- [x] Implement `ScreeningPruner` trait for `GdsdPruner`
- [x] Add `GdsdConfig` with defaults (ψ=10.0, β=0.001, tlc=true)
  - Also `.no_tlc()`, `.strong()` (ψ=20, β=0.01), `.mild()` (ψ=1, β=0.0001) presets
- [x] Add feature gate `gdsd_distill = ["bandit"]` to `Cargo.toml`
- [x] Add 4 common advantage functions (identity, sigmoid, tanh, clamped)
- [x] `teacher_signal()` method exposed for testing
- [x] 20 unit tests (config, relevance, teacher signal, advantage functions, TLC, clamping, accessors)

### Phase 2: TLC Utility ✅
- [x] Add `token_logit_centralization()` to `gdsd.rs` as utility function
- [x] Input: `&mut [f32]` logits, output: centralized (subtract mean), returns mean
- [x] O(V) serial — no rayon needed (tiny workload per optimization.md)
- [x] 4 TLC tests (empty, single, uniform, mixed)

### Phase 3: Integration with Existing Pruners ✅
- [x] `GdsdPruner` generic over `P: ScreeningPruner` — works with any inner pruner
- [x] Bandit integration verified: `GdsdPruner<BanditPruner<P>>` works (T5 test)
- [x] Add `build_dd_tree_gdsd()` variant to `dd_tree.rs` that uses `GdsdPruner`
- [x] Re-export from `speculative/mod.rs`
- [x] Module registration in `pruners/mod.rs` with full public exports

### Phase 4: GOAT Proof ✅ (8/8)
- [x] T1: Relevance overhead — 118-141% (3 relevance calls + GDSD blend, acceptable)
- [x] T2: Teacher signal correctness — blend formula validated for β/ψ edge cases
- [x] T3: TLC centralization — zero-mean property verified
- [x] T4: DDTree integration — consistent tree structure with NoScreeningPruner + TLC
- [x] T5: Bandit integration — GdsdPruner<BanditPruner> works, cold start OK
- [x] T6: Advantage functions — all 4 produce valid trees with correct [0,1] bounds
- [x] T7: Convergence — GdsdPruner wrapping BanditPruner finds optimal arm (500 rounds)
- [x] Summary test — `goat_169_summary` documents all results

### Phase 5: GOAT Gain Proof — ❌ FAILED

**The GOAT bar requires PROVEN GAIN, not just correctness.**

| Gate | Requirement | Status |
|------|-------------|--------|
| G1 | Acceptance rate improvement ≥5% over base pruner in DDTree arena | ❌ FAIL (+0.00%) |
| G2 | OR: Win rate improvement ≥3% in Bomber/Go arena A/B | ❌ NOT TESTED |
| G3 | Overhead ≤ 20% on relevance() hot path | ❌ FAIL (+181.5%) |

T1-T7 prove **correctness** (formula, integration, convergence) — not **gain**.
No test measures whether GDSD actually improves acceptance rate, quality, or win rate
over the base pruner. The overhead (120%) is measurable; the benefit is not.

**Verdict: NO GOAT.** GdsdPruner is infrastructure — a correctly-implemented wrapper
with no proven advantage over existing pruners. Available behind `gdsd_distill` for
future domain-specific experiments that may demonstrate gain.

## GOAT Proof Results

### Structural Tests (correctness, NOT gain)

```
T1: Relevance overhead ...................... ✅ PASS (~120%, 3 relevance calls)
T2: Teacher signal correctness .............. ✅ PASS (3 edge cases)
T3: TLC centralization ...................... ✅ PASS (zero-mean verified)
T4: DDTree integration ...................... ✅ PASS (consistent structure)
T5: Bandit integration ...................... ✅ PASS (GdsdPruner<BanditPruner>)
T6: Advantage functions ..................... ✅ PASS (4/4 valid trees)
T7: Convergence ............................ ✅ PASS (optimal arm found)
```

### Gain Tests (required for GOAT)

```
G1: Acceptance rate improvement ≥5% ......... ❌ FAIL (+0.00%, identical to baseline)
G2: Arena win rate improvement ≥3% .......... ❌ NOT TESTED
G3: Overhead ≤ 20% ......................... ❌ FAIL (+181.5%, nearly 3× cost)
```

**GOAT: 0/3 gain gates passed. ❌ NOT GOAT-PROVEN.**

## Optimization Compliance

- **Zero alloc in hot path:** `relevance()` computes from pre-stored values. No `Vec`, no `Box` allocation in the hot path.
- **Serial TLC:** O(V) where V ≤ 128 (micro config). Too small for rayon. Serial is correct per optimization.md.
- **No GPU needed:** This is modelless. Pure CPU, pure arithmetic.
- **Pre-compute advantage:** Advantage from bandit is cached, not recomputed per call.
- **fn pointer advantage_fn:** Zero-size, no heap allocation.

## Implementation Summary

### Files
- `katgpt-rs/src/pruners/gdsd.rs` — 518 lines, 20 unit tests
- `katgpt-rs/tests/bench_gdsd_modelless.rs` — 474 lines, 8 GOAT tests
- `katgpt-rs/src/speculative/dd_tree.rs` — `build_dd_tree_gdsd()` convenience builder
- `katgpt-rs/Cargo.toml` — `gdsd_distill = ["bandit"]` feature gate
- `katgpt-rs/src/pruners/mod.rs` — module registration + public exports
- `katgpt-rs/src/speculative/mod.rs` — `build_dd_tree_gdsd` re-export

### Test Results
- 20/20 unit tests pass with `--features "gdsd_distill"`
- 8/8 GOAT tests pass with `--features "gdsd_distill,bandit"`
- Default build clean (0 warnings)

## Module Structure

```
katgpt-rs/src/pruners/gdsd.rs          # GdsdPruner<P> + GdsdConfig + advantage functions + TLC
katgpt-rs/src/speculative/dd_tree.rs   # build_dd_tree_gdsd() convenience builder
katgpt-rs/tests/bench_gdsd_modelless.rs # GOAT proof (T1-T7 + summary)
```
