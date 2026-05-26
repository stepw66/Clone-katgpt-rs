# Plan 153: PEIRA Modelless Distillation

**Branch:** `develop/feature/153_peira_distill`
**Depends on:** Plan 030 (Bandit), Research 115 (PEIRA)
**Research:** `.research/115_PEIRA_Inter_View_Regressor_Alignment.md`
**Source:** arXiv:2605.17671
**Goal:** Implement PEIRA's auxiliary loss (Laux) as an alternative distillation option for model-based/modelless alignment. Feature-gated under `peira_distill`.

**Key insight:** PEIRA provides a theoretically grounded, collapse-free distillation loss. The core computation is:
1. Maintain EMA estimates of k×k covariance matrices Σ (cross-view) and N (within-view)
2. Compute closed-form P* = Σ(N + λI)⁻¹ and Q* = (N + λI)⁻¹
3. Compute auxiliary loss Laux without backpropagating through the matrix inverse
4. Use Laux gradients to update encoder parameters

**Why CPU only:** The matrices are k×k (k = representation dimension, typically 128–512) so inversion is O(k³) which is negligible. No GPU/WGSL needed.

**Scope:** This plan covers the core PEIRA loss, EMA covariance tracking, and integration with the BanditPruner (Plan 030). We do NOT implement the full PEIRA training pipeline — only the distillation loss component that plugs into our existing modelless framework.

---

## GOAT Proof Results

All gates validated via `core_06_peira` example.

| Task | Gate | Result | Evidence |
|------|------|--------|----------|
| T1 | `PeiraConfig` compiles under `peira_distill` | ⬜ PENDING | — |
| T2 | EMA covariance tracks known covariance within 5% | ⬜ PENDING | — |
| T3 | `peira_aux_loss` matches hand-computed reference | ⬜ PENDING | — |
| T4 | `PeiraDistiller` completes SC-PEIRA Algorithm 1 loop | ⬜ PENDING | — |
| T8 | Collapse-free: representation norm > 0 throughout training | ⬜ PENDING | — |
| T9 | CCA subspace recovery: overlap ≥ 0.9 with ground truth | ⬜ PENDING | — |
| T10 | Benchmark vs GFlowNet/SDAR/VPD in `.benchmarks/012_peira_distill_goat.md` | ⬜ PENDING | — |

---

## Tasks

### Phase 1: Core Infrastructure

- [ ] **T1: Add `PeiraConfig` to katgpt-rs-core**
  - Fields: `lambda: f64` (regularization, default 0.1), `ema_rate: f64` (EMA momentum, default 0.9), `dim: usize` (representation dimension)
  - Location: `crates/katgpt-core/src/` (new module `peira.rs`)
  - Feature-gated behind `peira_distill` in katgpt-core

- [ ] **T2: Implement EMA covariance tracker `PeiraCovariance`**
  - Maintains running Σ (cross-view) and N (within-view) matrices
  - `update(student_repr: &[f32], teacher_repr: &[f32])` — updates EMA estimates
  - `predictor() -> (Vec<f32>, Vec<f32>)` — returns (P*, Q*) in flat layout
  - Reset method for episode boundaries
  - All k×k operations, no SIMD needed for small k

- [ ] **T3: Implement `peira_aux_loss` function**
  - Signature: `pub fn peira_aux_loss(student: &[f32], teacher: &[f32], p_star: &[f32], q_star: &[f32], lambda: f64) -> f64`
  - Computes Laux from paper Equation (15)
  - No matrix inversion differentiation (key advantage)

- [ ] **T4: Add `PeiraDistiller` struct implementing modelless distillation**
  - Wraps PeiraCovariance + PeiraConfig
  - Implements the SC-PEIRA Algorithm 1 training loop
  - Returns loss + gradient signal for integration with BanditPruner
  - Location: `src/distill/peira.rs` (new file)

### Phase 2: Integration

- [ ] **T5: Wire `peira_distill` feature gate into main Cargo.toml**
  - `peira_distill = ["katgpt-core/peira_distill", "bandit"]`
  - NOT default-on initially

- [ ] **T6: Add `peira_alignment_score` metric**
  - Computes alignment α = (eᵀN e) / (||e|| ||Ne||) between signal and noise eigenvectors
  - Returns f64 in [0, 1], where 1.0 = perfect alignment = canonical structure found
  - Useful as GOAT proof criterion
  - Location: `src/distill/peira.rs`

- [ ] **T7: Integration example — `core_06_peira` example binary**
  - Demonstrates: init PeiraDistiller → feed student/teacher pairs → compute loss → check alignment score
  - Synthetic data: two views of 2D Gaussian with known canonical correlations
  - Verifies: alignment → 1.0 over training, no collapse

### Phase 3: GOAT Proof

- [ ] **T8: GOAT proof — collapse-free guarantee**
  - Train PeiraDistiller on synthetic data with known canonical structure
  - Gate: representation norm stays > 0 throughout training (no collapse)
  - Gate: alignment score ≥ 0.95 after convergence

- [ ] **T9: GOAT proof — CCA subspace recovery**
  - Synthetic data with 5 canonical directions, k=8 representation
  - Gate: recovered subspace overlaps ≥ 0.9 with ground truth canonical directions
  - Gate: spectral filter correctly suppresses directions with ci < λ

- [ ] **T10: Benchmark against existing distillation losses**
  - Compare PeiraDistiller vs GFlowNet (Plan 052) vs SDAR (Plan 072) vs VPD (Plan 120)
  - Metric: DDTree score improvement on same data
  - Report in `.benchmarks/012_peira_distill_goat.md`

- [ ] **T11: Integration with SR²AM Configurator (Plan 112)**
  - PEIRA alignment score as additional planning metric
  - Wire into SR²AM's adaptive decision loop
  - Behind existing `sr2am_configurator` feature gate

---

## Feature Flag

```toml
[features]
peira_distill = ["katgpt-core/peira_distill", "bandit"]  # PEIRA modelless distillation (Research 115, Plan 153)
```

Interacts with: `bandit` (required), `sr2am_configurator` (optional, T11)

---

## Failure Mode

If PEIRA's auxiliary loss shows no improvement over existing distillation losses (GFlowNet, SDAR, VPD) on DDTree benchmarks, the feature remains as a compile-time option but is not promoted to default-on. The EMA covariance tracker (T2) and `peira_alignment_score` (T6) are independently useful as diagnostic tools regardless of distillation quality.

---

## Priority Assessment

| Task | Impact | Effort | Status |
|------|--------|--------|--------|
| T1 (PeiraConfig) | Medium | Low (~30 LOC) | ⬜ Not started |
| T2 (EMA covariance) | High | Medium (~120 LOC) | ⬜ Not started |
| T3 (aux loss) | High | Low (~40 LOC) | ⬜ Not started |
| T4 (PeiraDistiller) | High | Medium (~100 LOC) | ⬜ Not started |
| T5 (Feature gate) | Low | Low (~5 LOC) | ⬜ Not started |
| T6 (Alignment score) | Medium | Low (~30 LOC) | ⬜ Not started |
| T7 (Example) | Medium | Medium (~150 LOC) | ⬜ Not started |
| T8 (GOAT collapse-free) | High | Low (~40 LOC) | ⬜ Not started |
| T9 (GOAT CCA recovery) | High | Medium (~60 LOC) | ⬜ Not started |
| T10 (Benchmark) | Medium | Medium (~80 LOC) | ⬜ Not started |
| T11 (SR²AM integration) | Low | Medium (~50 LOC) | ⬜ Not started |

---

## Files Modified

| File | Changes |
|------|---------|
| `Cargo.toml` | `peira_distill` feature flag |
| `crates/katgpt-core/src/peira.rs` | New: `PeiraConfig`, `PeiraCovariance`, `peira_aux_loss` |
| `crates/katgpt-core/src/lib.rs` | `#[cfg(feature = "peira_distill")] pub mod peira;` |
| `crates/katgpt-core/Cargo.toml` | `peira_distill` feature gate |
| `src/distill/peira.rs` | New: `PeiraDistiller`, `peira_alignment_score` |
| `src/distill/mod.rs` | `#[cfg(feature = "peira_distill")] pub mod peira;` |
| `examples/core_06_peira.rs` | Demo: init → train → alignment → GOAT gates |

---

## Test & Verification Commands

```sh
# Run all tests with peira_distill
cargo test --features peira_distill --lib --quiet

# Run peira-specific tests
cargo test --features peira_distill --lib peira --quiet

# Run example
cargo run --example core_06_peira --features peira_distill --release

# Clippy
cargo clippy --features peira_distill --examples --quiet
```

---

## References

- `.research/115_PEIRA_Inter_View_Regressor_Alignment.md` — research verdict
- arXiv:2605.17671 — PEIRA paper (primary source)
- `.plans/030_multi_armed_bandit.md` — BanditPruner dependency
- `.plans/052_gflownet_modelless_distillation.md` — GFlowNet baseline (T10)
- `.plans/072_sdar_gated_distillation_modelless.md` — SDAR baseline (T10)
- `.plans/120_vpd_em_modelless_distillation.md` — VPD baseline (T10)
- `.plans/112_sr2am_configurator_bandit.md` — SR²AM integration target (T11)
