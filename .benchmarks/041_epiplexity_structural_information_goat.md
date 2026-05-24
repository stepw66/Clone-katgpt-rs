# Benchmark 041: Epiplexity Structural Information Scoring — GOAT Proofs

**Plan:** 130 — Epiplexity Structural Information Scoring
**Research:** 090 — Epiplexity Structural Information Computationally Bounded Observers
**Paper:** Epiplexity (arXiv:2601.03220): Structural information extractable by computationally bounded observers, measured as area under loss curve above final loss
**Feature Gate:** `epiplexity_scoring = []`
**Date:** 2025-07-13

---

## Architecture

Epiplexity implements **structural information scoring** for modelless distillation data selection:

```
Training Losses
       │
       ▼
  EpiplexityEstimator ─── Ring Buffer (capacity N)
       │              ├── record_step(step_loss)
       │              ├── compute_epiplexity(final_loss) → Σ max(0, loss_i - final)
       │              └── compute_per_sample(final_losses) → Vec<S>
       │
       ▼
  TimeBoundedEntropy ─── Companion Estimator
       │              ├── compute_entropy(final_loss, n_tokens) → H_T
       │              └── structural_fraction(final_loss, n_tokens) → S_T / H_T
       │
       ▼
  EpiplexityScreeningPruner<P> ─── Blended Relevance
       │              ├── inner.relevance() × (1 - α)
       │              ├── epiplexity_signal × α
       │              └── EpiplexityWeight: Uniform | LossDrop | CumulativeArea
       │
       ▼
  LossCurveTracker ─── Batch/Epoch Granularity
       │              ├── on_batch_end(batch_idx, avg_loss)
       │              ├── on_epoch_end(epoch, val_loss)
       │              └── epiplexity_estimate() → prequential S_T
       │
       ▼
  PerPositionLossTracker ─── Fine-Grained Scoring
       │              ├── record_step(&[loss_per_position])
       │              ├── per_position_epiplexity() → Vec<S>
       │              └── top_k_structural(k) → most informative positions
       │
       ▼
  FactorizationScorer ─── Forward/Reverse Ordering
       │              ├── score_forward(trace) → S_T (last = final)
       │              ├── score_reverse(trace) → S_T (reversed)
       │              ├── epiplexity_gap() → S_rev - S_fwd
       │              └── preferred_order() → Forward | Reverse | Adaptive
       │
       ▼
  DDTree Decode Step (ScreeningPruner trait)
```

### Key Types

| Type | File | Purpose |
|------|------|---------|
| `EpiplexityEstimator` | `mod.rs` | Ring buffer for loss history, computes S_T |
| `TimeBoundedEntropy` | `mod.rs` | Companion entropy estimator H_T |
| `EpiplexityScreeningPruner<P>` | `screening.rs` | ScreeningPruner with epiplexity blending |
| `EpiplexityWeight` | `screening.rs` | Weight mode enum: Uniform, LossDrop, CumulativeArea |
| `LossCurveTracker` | `loss_curve.rs` | Batch/epoch-level prequential estimation |
| `PerPositionLossTracker` | `loss_curve.rs` | Per-token-position fine-grained scoring |
| `FactorizationScorer` | `factorization.rs` | Forward/reverse trace ordering analysis |
| `FactorizationOrder` | `factorization.rs` | Forward, Reverse, Adaptive enum |

### Module Structure

```
src/pruners/epiplexity/
├── mod.rs              # EpiplexityEstimator, TimeBoundedEntropy, re-exports
├── screening.rs        # EpiplexityScreeningPruner<P>, EpiplexityWeight
├── loss_curve.rs       # LossCurveTracker, PerPositionLossTracker
└── factorization.rs    # FactorizationScorer, FactorizationOrder
```

---

## GOAT Proof Results

### Test Configuration

- **Seed:** 42 (deterministic, reproducible for pseudo-random generator)
- **EpiplexityEstimator:** capacity=100 (ring buffer)
- **LossCurveTracker:** batch_capacity=100, epoch_capacity=10
- **FactorizationScorer:** capacity=100

### P1: EpiplexityEstimator — Structural vs Random vs Constant

| Data Type | Pattern | S_T | Result |
|-----------|---------|-----|--------|
| Constant (loss=2.5, 50 steps) | flat | < 0.01 | ✅ Pass |
| Pseudo-random (center=3.0, spread=1.0, 500 steps) | noisy flat | per-step < 1.0 | ✅ Pass |
| Structured (5.0→1.1, 50 steps) | decreasing | > 1.0 | ✅ Pass |
| Structured (8.0→1.1) vs Structured (3.0→1.1) | more structure | higher > lower | ✅ Pass |
| Structured vs Random | structured 6.0→2.1 vs noise | structured > random | ✅ Pass |

**Result:** EpiplexityEstimator correctly discriminates structured from random/constant data. S_T is monotone in structure amount.

### P1: Per-Sample and TimeBoundedEntropy

| Property | Expected | Actual | Result |
|----------|----------|--------|--------|
| Per-sample: lower final → higher S | monotone | monotone | ✅ Pass |
| Per-sample: final above all steps → S≈0 | < 0.01 | < 0.01 | ✅ Pass |
| Ring buffer: overflow caps at capacity | len = 5 | len = 5 | ✅ Pass |
| Ring buffer: only last N values kept | sum = 85.0 | 85.0 | ✅ Pass |
| TimeBoundedEntropy: H_T = loss × tokens | 250.0 | 250.0 | ✅ Pass |
| Structural fraction: bounded in [0, 1] | ∈ [0, 1] | ∈ [0, 1] | ✅ Pass |
| Clear: resets buffer | len = 0, S = 0 | len = 0, S = 0 | ✅ Pass |

**Result:** Per-sample estimation, entropy companion, and ring buffer all work correctly.

### P2: EpiplexityScreeningPruner — Blend Behavior

| α | Inner | Weight Mode | Expected | Actual | Result |
|---|-------|-------------|----------|--------|--------|
| 0.0 | UnitPruner (1.0) | Uniform | 1.0 | 1.0 | ✅ Pass |
| 0.0 | FixedPruner (0.3) | Uniform | 0.3 | 0.3 | ✅ Pass |
| 1.0 | UnitPruner | Uniform | 1.0 | 1.0 | ✅ Pass |
| 1.0 | UnitPruner | LossDrop(drop=0) | sigmoid(0)≈0.5 | 0.5 | ✅ Pass |
| 1.0 | UnitPruner | LossDrop(drop=5) | sigmoid(5)>0.99 | >0.99 | ✅ Pass |
| 1.0 | UnitPruner | CumulativeArea (empty) | 0.0 | 0.0 | ✅ Pass |
| 1.0 | UnitPruner | CumulativeArea (structured) | > 0.5 | > 0.5 | ✅ Pass |
| 0.3 | FixedPruner (0.4) | Uniform | 0.4×0.7+1.0×0.3=0.58 | 0.58 | ✅ Pass |

**Result:** α=0 perfectly preserves inner pruner behavior. α=1 uses epiplexity signal. Blend interpolates correctly.

### P2: Alpha Clamping

| Input α | Clamped | Result |
|---------|---------|--------|
| -5.0 | 0.0 | ✅ Pass |
| 100.0 | 1.0 | ✅ Pass |

**Result:** Alpha setter correctly clamps to [0, 1].

### P3: LossCurveTracker — Batch/Epoch Tracking

| Property | Expected | Actual | Result |
|----------|----------|--------|--------|
| Batch count after 3 on_batch_end | 3 | 3 | ✅ Pass |
| Latest batch loss | 3.0 | 3.0 | ✅ Pass |
| Epoch count after 2 on_epoch_end | 2 | 2 | ✅ Pass |
| Latest epoch loss | 3.5 | 3.5 | ✅ Pass |
| Prequential estimate (structured 6.0→1.0) | > 0 | > 0 | ✅ Pass |
| Prequential estimate (constant 3.0) | < 0.01 | < 0.01 | ✅ Pass |
| Running min updates correctly | 3.0 (after 5,3,4) | 3.0 | ✅ Pass |
| Epoch epiplexity: (5,3,2) → S=4.0 | 4.0 | 4.0 | ✅ Pass |
| Epoch epiplexity (empty) | 0.0 | 0.0 | ✅ Pass |
| Total loss drop (5→2) | 3.0 | 3.0 | ✅ Pass |
| Batch ring buffer overflow (cap=3, 10 inserts) | 3 | 3 | ✅ Pass |
| Epoch ring buffer overflow (cap=3, 10 inserts) | 3 | 3 | ✅ Pass |
| Clear resets all state | counts=0, min=0.0 | counts=0, min=0.0 | ✅ Pass |

**Result:** LossCurveTracker correctly tracks batch/epoch losses, computes prequential S_T, and enforces ring buffer bounds.

### P3: PerPositionLossTracker

| Property | Expected | Actual | Result |
|----------|----------|--------|--------|
| Per-position S (structured) | all > 0 | all > 0 | ✅ Pass |
| Per-position with final: pos0=3.0, pos1=2.0 | exact | exact | ✅ Pass |
| Per-position (empty) | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | ✅ Pass |
| Total epiplexity > 0 | > 0 | > 0 | ✅ Pass |
| Top-k: position 0 most structural | pos 0 | pos 0 | ✅ Pass |
| Step count tracking | 2, 2, 0 (OOB) | 2, 2, 0 | ✅ Pass |
| Ring buffer overflow (cap=3, 5 inserts) | 3 | 3 | ✅ Pass |

**Result:** PerPositionLossTracker provides fine-grained per-position scoring with correct epiplexity estimates.

### P4: FactorizationScorer — Forward/Reverse Ordering

| Trace Type | Forward S_T | Reverse S_T | Gap | Preferred | Result |
|------------|-------------|-------------|-----|-----------|--------|
| Decreasing (5.0→1.4) | > 0 | ≈ 0 | < 0 | Forward | ✅ Pass |
| Increasing (1.0→4.6) | ≈ 0 | > 0 | > 0 | Reverse | ✅ Pass |
| Constant (3.0) | ≈ 0 | ≈ 0 | ≈ 0 | Forward (tie) | ✅ Pass |
| Empty | 0.0 | 0.0 | 0.0 | — | ✅ Pass |

**Key insight:** The scorer uses **last value** as final_loss (modeling the training endpoint). Decreasing traces have high forward S_T because the last value is the minimum. Reversing makes the last value the maximum, dropping S_T to ≈0.

### P4: Adaptive and Ranking

| Property | Expected | Actual | Result |
|----------|----------|--------|--------|
| Adaptive picks max(fwd, rev) for increasing | = rev | = rev | ✅ Pass |
| Rank traces: high structure first | index 2 | index 2 | ✅ Pass |
| Order preference counts: ≥1 fwd, ≥1 rev, total=3 | ≥1, ≥1, 3 | ✓ | ✅ Pass |
| Default FactorizationOrder | Adaptive | Adaptive | ✅ Pass |
| Display formatting | "Forward"/"Reverse"/"Adaptive" | correct | ✅ Pass |

**Result:** FactorizationScorer correctly identifies optimal trace ordering per the epiplexity paper's findings.

---

## Feature Gate

```toml
# Cargo.toml
epiplexity_scoring = []  # Epiplexity structural information scoring (Plan 130)

# Usage
cargo test --features epiplexity_scoring --test test_130_epiplexity_goat
cargo check --features epiplexity_scoring --quiet
cargo clippy --features epiplexity_scoring --quiet --tests
```

---

## Files Created/Modified

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/pruners/epiplexity/mod.rs` | ~294 | EpiplexityEstimator, TimeBoundedEntropy, unit tests |
| `src/pruners/epiplexity/screening.rs` | ~284 | EpiplexityScreeningPruner<P>, EpiplexityWeight |
| `src/pruners/epiplexity/loss_curve.rs` | ~439 | LossCurveTracker, PerPositionLossTracker |
| `src/pruners/epiplexity/factorization.rs` | ~287 | FactorizationScorer, FactorizationOrder |
| `tests/test_130_epiplexity_goat.rs` | ~672 | GOAT proofs (48 tests) |

### Modified Files

| File | Change |
|------|--------|
| `Cargo.toml` | Added `epiplexity_scoring = []` feature, added to `full` |
| `src/pruners/mod.rs` | Added `epiplexity` module behind `#[cfg(feature = "epiplexity_scoring")]` |

---

## Test Summary

```
running 48 tests · test_130_epiplexity_goat
................................................
test result: ok. 48 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

cargo clippy --features epiplexity_scoring --quiet --tests
( zero warnings )
```

### Task Status

| Task | Status | Notes |
|------|--------|-------|
| T1: EpiplexityEstimator Core | ✅ Complete | Ring buffer, S_T, per-sample, TimeBoundedEntropy |
| T2: EpiplexityScreeningPruner | ✅ Complete | Blend α, 3 weight modes, LossDrop sigmoid |
| T3: Loss Curve Tracker | ✅ Complete | Batch/epoch tracking, per-position, prequential |
| T4: SR²AM Context Extension | ⏭️ Deferred | Requires Plan 112 internals |
| T5: Factorization Scoring | ✅ Complete | Forward/reverse/adaptive, gap scoring, ranking |
| T6: GOAT Proofs | ✅ Complete | 48 tests across P1-P4 |
| T7: Feature Gate + Module Glue | ✅ Complete | `epiplexity_scoring = []`, in `full` |
| T8: Documentation + Benchmark | ✅ Complete | This file |

---

## References

- Research 090: `.research/090_Epiplexity_Structural_Information_Computationally_Bounded_Observers.md`
- Epiplexity paper: arXiv:2601.03220
- Existing ScreeningPruner trait: `crates/katgpt-core/src/traits.rs`
- Existing bandit: `src/pruners/bandit.rs` (Plan 030)
- Benchmark 040: `.benchmarks/040_opus_boltzmann_bandit.md`
