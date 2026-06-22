# Plan 308: KARC — Delay-Basis-Ridge Forecaster (Open Primitive)

**Date:** 2026-06-22
**Research:** [katgpt-rs/.research/288_KARC_Delay_Basis_Ridge_Forecaster.md](../.research/288_KARC_Delay_Basis_Ridge_Forecaster.md)
**Source paper:** [arxiv 2606.19984](https://arxiv.org/abs/2606.19984) — Huang, Kurths, Tang, *Kolmogorov-Arnold Reservoir Computing*, 2026-06-18
**Target:** `katgpt-rs/crates/katgpt-core/src/karc.rs` (new module) + Cargo feature `karc_forecaster`
**Status:** Active — Phase 1 unblocking skeleton

---

## Goal

Ship a generic, modelless, inference-time trajectory forecaster `KarcForecaster<D, M, K>` that:
1. Concatenates the last-K observations (delay embedding) — `x_i ∈ R^{K·D}`.
2. Expands each coordinate onto M basis functions via a sealed `KarcBasis` trait (Fourier, Chebyshev, BSpline shipped).
3. Fits a linear readout `Wout ∈ R^{D × (K·D·M)}` by closed-form ridge regression `Wout = YH^T(HH^T + λI)^{-1}`, reusing `peira::predictor_with_scratch` machinery for the (N + λI)^{-1} step.
4. Forecasts `û_{i+1} = Wout · Ψ(x_i)` in a single zero-alloc matvec.

No game semantics, no NPC references, no chain types. **Open math only.** Private selling-point guide lives in `riir-ai/.research/152_*.md`; runtime integration in `riir-ai/.plans/332_*.md`; shard storage crossref in `riir-neuron-db/.research/003_*.md`; LatCal commitment crossref in `riir-chain/.research/003_*.md`.

**GOAT gate (must pass before promoting to default feature):**
- **G1 — Reproduce paper double-scroll Table I within 2×.** NRMSE on double-scroll test trajectory ≤ 1.0×10^{-3} (paper: 5.3×10^{-4}). Threshold time ≥ 8 Lyapunov times (paper: 16.7 LT; we accept ≥8 LT for the open primitive; riir-ai integration plan targets the full 16 LT).
- **G2 — Train-time wall clock ≤ 2× paper.** Paper: 0.12s on H100 GPU. We are CPU-only — accept ≤ 0.5s on a single SIMD thread for the same workload (4,000 samples, feature dim 1,891).
- **G3 — Zero-alloc hot path.** `forecast_into(&delay_state, &mut out)` does not allocate. Verified by `cargo run --features karc_forecaster --example karc_alloc_check`.
- **G4 — Bit-reproducibility.** Two `KarcForecaster` instances fit on the same `(basis, k, λ, trajectory)` produce byte-identical `Wout`. Required for quorum commitment downstream.

Demote-on-fail: if G1 misses by >10× or G2 misses by >5×, downgrade to opt-in Gain-tier, file issue, do not promote to default.

---

## Architecture

```text
                  ┌─────────────────────┐
trajectory ─────▶ │ KarcForecaster      │
{u_i} ∈ R^{D}     │                     │
                  │  delay_buffer       │ ◀── ring buffer of last-K observations
                  │  basis: KarcBasis   │ ◀── sealed trait, three impls
                  │  Wout: Vec<f32>     │ ◀── D × (K·D·M), row-major
                  │  scratch: KarcScratch│ ◀── pre-allocated Gram + cov + pivot bufs
                  └─────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
       observe(u_t)   fit_ridge(λ)   forecast_into(&mut û_{t+1})
       push to ring   solve Wout    matvec Wout · Ψ(delay_state)
                      (Woodbury)    zero-alloc
```

**Reuse map (DRY):**
- Ridge solve `(N + λI)^{-1}` → **`peira::predictor_with_scratch`** math (do not re-implement; extract the Cholesky/inversion kernel into `crates/katgpt-core/src/linalg/ridge_solve.rs` and have both `peira.rs` and `karc.rs` consume it).
- Basis eval → consume `riir-engine::linoss::basis::SpectralBasis` directly if riir-engine is in the dep tree, otherwise vendor a minimal `KarcBasis` trait with the same shape (the trait surface is ~10 lines).
- SIMD matvec → reuse `simd::simd_matvec` / `simd::simd_matmul_rows` for the forecast step.

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

Goal: a compiling, tested, feature-gated module that implements the full KARC pipeline on synthetic data with the public API surface frozen. **Reproduces paper Table I (double-scroll) within 2×.**

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/karc.rs` behind `#[cfg(feature = "karc_forecaster")]`. Empty `KarcForecaster<D, M, K>` struct with const generics, `KarcBasis` sealed trait, `KarcScratch` pre-allocated buffers. Wire `karc_forecaster` into `crates/katgpt-core/Cargo.toml` features list and `lib.rs` mod declaration.
- [ ] **T1.2** Implement `KarcBasis` trait with three const-generic instances:
  - `FourierBasis<const M: usize>` — `ψ_{2i-1}(x) = cos(2π·i·x/P)`, `ψ_{2i}(x) = sin(...)`, period `P` set at construction.
  - `ChebyshevBasis<const M: usize>` — `T_0..T_{M-1}` via three-term recurrence.
  - `BSplineBasis<const M: usize>` — uniform knots, degree-3 default, Cox-de Boor recursion.
  Trait method `eval_into(&self, x: f32, out: &mut [f32; M])` — zero-alloc per-coordinate projection. Reuse `riir-engine::linoss::basis::FourierBasis` if dep tree allows; otherwise vendor with attribution.
- [ ] **T1.3** Implement `DelayRing<D, K>` — fixed-capacity ring buffer of last-K `&[f32; D]` observations. `push(o: &[f32; D])` overwrites oldest. `flatten_into(&mut [f32; K*D])` writes the delay-embedded state `x_i = u_i ⊕ u_{i-1} ⊕ ... ⊕ u_{i-k+1}` in observation order (newest first).
- [ ] **T1.4** Implement `feature_expand<B: KarcBasis>(delay_state: &[f32], basis: &B, out: &mut [f32])` — applies basis to each of the `K·D` delay coordinates, writing `K·D·M` features. Chunk-4 unrolled for SIMD.
- [ ] **T1.5** Implement `KarcForecaster::observe(u: &[f32; D])` — pushes to delay ring; if ring is full and `fit_interval_ticks` has elapsed since last fit, accumulates `(Ψ(x), u_{t+1})` pair into the trajectory buffer.
- [ ] **T1.6** Implement `KarcForecaster::fit_ridge(lambda: f32)` — solves `Wout = YH^T(HH^T + λI)^{-1}` using the Woodbury form `(H^T H + λI)^{-1} H^T Y` when `d_h > N` (the typical per-NPC regime). Extract the inversion kernel from `peira::predictor_with_scratch` into `crates/katgpt-core/src/linalg/ridge_solve.rs` (new file, no behavior change to PEIRA — pure refactor).
- [ ] **T1.7** Implement `KarcForecaster::forecast_into(delay_state: &[f32], out: &mut [f32; D])` — `feature_expand` into scratch, then `simd_matvec(Wout, features, out)`. Zero allocation.
- [ ] **T1.8** Write `examples/karc_double_scroll.rs` — integrates the double-scroll ODE (paper §A.1 with `R1=1.2, R2=3.44, R4=0.193, β=11.6, I_r=2.25e-5`), generates 4,000 samples @ 4 obs/unit time, fits KARC with `K=4, M=8, λ=1e-6`, runs autonomous rollout, reports NRMSE over first Lyapunov time and threshold time at `ε=0.1`. **G1 gate target: NRMSE ≤ 1.0e-3, threshold ≥ 8 LT.**
- [ ] **T1.9** Write `tests/karc_reproducibility.rs` — fit two forecasters on byte-identical synthetic trajectories, assert `Wout` byte-equality (G4 gate). Vary `λ ∈ {1e-8, 1e-6, 1e-4}` to ensure stability across regularization strengths.
- [ ] **T1.10** Write `tests/karc_alloc_check.rs` — use `cargo-allocations` or a manual `Box::leak` allocator hook to verify `forecast_into` performs zero allocations after warmup (G3 gate). Skip if `cargo-allocations` not available — fall back to `#[track_caller]` + manual `GlobalAlloc` counter.
- [ ] **T1.11** Add `benches/karc_forecast_bench.rs` — `criterion` benchmark of `forecast_into` at `D=8, M=8, K=4` (the HLA-shaped config). **G2 hot-path target: forecast wall clock ≤ 500 ns/call.**
- [ ] **T1.12** Document `karc.rs` module-level rustdoc — the math (Eqs. 8, 11, 14), the basis dictionary, the Woodbury swap, and the G1–G4 GOAT gate. Link to Plan 308 and Research 288.

**Phase 1 exit criteria:** All T1.x tasks done. `cargo check --features karc_forecaster` passes. G1, G3, G4 pass. G2 documented (may defer to Phase 2 if Phase 1 bench is noisy). Feature is **opt-in** (`karc_forecaster` not in default features).

---

## Phase 2 — Higher-Order KARC + Memory-Optimized Fit

Goal: implement paper Methods §A (higher-order outer products) and §C (Woodbury + chunked Gram + low-rank factorization) for high-D settings. Needed before riir-neuron-db can ship `KarcShard` (low-rank `Wout ≈ AB` is the storage form).

### Tasks

- [ ] **T2.1** Implement `feature_expand_higher_order<B, const R: usize>(delay_state, basis, out)` — outer-product features up to order `R` per paper Eq. 32. Use combinatorial enumeration to avoid duplicate products (e.g., for `R=2`, iterate `i ≤ j` over basis responses).
- [ ] **T2.2** Implement `chunked_gram_into<H_iter>(features_iter, out_gram: &mut [f32], lambda: f32)` — paper Eq. 44. Block-accumulate `H_i^T H_i` over the trajectory buffer; never materialize full `H`. Reuse scratch from `KarcScratch`.
- [ ] **T2.3** Implement `low_rank_fit<A_dim, B_dim>(trajectory, lambda)` — paper Eq. 47. Alternating least squares: fix `B`, ridge-solve `A`; fix `A`, ridge-solve `B`. Default 50 iterations or `‖Wout_old − Wout_new‖_F < 1e-6`.
- [ ] **T2.4** Add `KarcForecaster::forecast_low_rank_into(delay_state, A, B, out)` — apply `A · (B · Ψ(x))` instead of `Wout · Ψ(x)`. Two-stage matvec, same zero-alloc contract.
- [ ] **T2.5** Benchmark low-rank fit at `(D=8, M=8, K=4, A_dim=8, B_dim=8)` — `Wout` is 8×256 = 2048 floats; low-rank `A(8×8) + B(8×256) = 64 + 2048 = 2112 floats`. Verify low-rank NRMSE within 1.5× of full-rank on double-scroll (paper §C accepts small accuracy loss).
- [ ] **T2.6** Document the higher-order and low-rank paths in module rustdoc with paper eq refs.

**Phase 2 exit criteria:** All T2.x done. Low-rank fit produces `A` (8×8) and `B` (8×256) byte-stably. NRMSE within 1.5× of full-rank on double-scroll. This is the storage form riir-neuron-db will persist as `KarcShard`.

---

## Phase 3 — Spline-Knot Adaptivity (Paper §III Discussion)

Goal: address the paper's stated limitation ("fixed basis dictionary may limit adaptability to systems with abrupt transitions") by adding alternating least-squares spline-knot optimization. **Optional phase** — defer unless riir-ai integration surfaces a real NPC trajectory with non-smooth structure.

### Tasks

- [ ] **T3.1** Implement `AdaptiveBSplineBasis` — knot positions are mutable; `optimize_knots(trajectory, n_iters)` performs alternating least squares (fix `Wout`, optimize knot positions by gradient-free line search; fix knots, refit `Wout`).
- [ ] **T3.2** Add `KarcForecaster::adapt_basis(trajectory)` — top-level entrypoint that calls `optimize_knots` then `fit_ridge`.
- [ ] **T3.3** Benchmark on a synthetic step-function + smooth-sinusoid mixed trajectory (paper's "abrupt transitions" failure mode). Target: NRMSE ≤ 2× the best fixed-basis NRMSE on the same data.

**Phase 3 exit criteria:** Either ships behind a separate `karc_adaptive_basis` feature, OR is shelved with a documented negative result. Do NOT block Phase 1/2 on this.

---

## Phase 4 — GOAT Gate & Default Promotion

- [ ] **T4.1** Run G1 (double-scroll Table I reproduction within 2×) — record result in `katgpt-rs/.benchmarks/308_karc_goat.md`.
- [ ] **T4.2** Run G2 (train-time wall clock ≤ 2× paper on CPU SIMD) — same bench file.
- [ ] **T4.3** Run G3 (zero-alloc forecast_into) — same bench file, with `cargo-allocations` output.
- [ ] **T4.4** Run G4 (bit-reproducibility across two instances) — same bench file.
- [ ] **T4.5** If all four pass: add `karc_forecaster` to `crates/katgpt-core/Cargo.toml` default features. Update `katgpt-rs/README.md` Feature Showcase with a new section "🧮 KARC: Delay-Basis-Ridge Forecaster (Plan 308, arxiv 2606.19984)". Update `katgpt-rs/.docs/01_overview.md` module structure.
- [ ] **T4.6** If any gate fails by ≤2×: file `katgpt-rs/.issues/NNN_karc_phase1_gap.md`, document the gap, propose a Phase 1.5 remediation. Do not promote to default.
- [ ] **T4.7** If any gate fails by >2×: downgrade Research 288 verdict from Super-GOAT to GOAT, update the verdict table, file an issue explaining the gap. Keep the feature opt-in.

---

## Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Ridge solve is closed-form; no backprop; no gradient through base weights |
| Latent-to-latent preferred | ✅ Operates on HLA / latent state directly; only the 5-scalar emotion projection crosses sync (handled in riir-ai Plan 332) |
| Sigmoid not softmax | ✅ KARC has no softmax anywhere; the basis functions are bounded (Fourier, Chebyshev ≤ 1, B-spline partition-of-unity) |
| Freeze/thaw over fine-tuning | ✅ `Wout` is bit-reproducible from trajectory alone; freeze = commit `(basis_config, k, λ, A, B)`; thaw = re-derive or restore from KarcShard (riir-neuron-db Plan to file) |
| 4-repo discipline | ✅ Open primitive in katgpt-rs; selling-point guide in riir-ai; commitment bridge in riir-chain; shard storage in riir-neuron-db |
| Zero-alloc hot path | ✅ `forecast_into` reuses pre-allocated scratch; `fit_ridge` reuses `KarcScratch` |
| CPU/SIMD first | ✅ All matvec via `simd::simd_matvec`; ridge solve via shared `linalg/ridge_solve.rs` |
| File size < 2048 lines | ✅ Target `karc.rs` ≤ 800 lines; `linalg/ridge_solve.rs` ≤ 400 lines (extracted from PEIRA) |
| `Uuid::now_v7()` if Uuid needed | N/A — no Uuids in this primitive |

## Dependencies

No new external dependencies. All math is closed-form (matvec, Cholesky/Vandermonde inversion, Cox-de Boor recursion). Reuse:
- `crates/katgpt-core/src/simd.rs` — `simd_matvec`, `simd_outer_product_acc`
- `crates/katgpt-core/src/peira.rs` — extract `(N + λI)^{-1}` kernel into `linalg/ridge_solve.rs`
- Optionally `riir-engine/src/linoss/basis.rs` — `FourierBasis` if dep tree allows (otherwise vendor)

## Out of scope (handled in other plans)

- HLA integration, curiosity bridge, MCTS collapse bridge → **riir-ai/.plans/332_karc_runtime_npc_integration.md**
- `KarcShard` Pod layout, `MerkleFrozenEnvelope` integration → **riir-neuron-db** (plan to file separately after Phase 1 lands)
- LatCal 2×2-block commitment of `Wout`, sync-boundary protocol → **riir-chain** (plan to file separately after Phase 1 lands)
- FLUX diffusion sampling acceleration (paper §4) → not in scope for game AI; file as separate katgpt-rs plan only if a clear game-AI transfer emerges

## TL;DR

Phase 1 ships a generic `KarcForecaster<D, M, K>` + `KarcBasis` trait behind `karc_forecaster` feature, reusing PEIRA's ridge solve and (optionally) linoss's basis dictionary. GOAT gate G1–G4 on the paper's double-scroll benchmark. Phase 2 adds higher-order features and low-rank factorization (the form that persists into a KarcShard). Phase 3 (adaptive spline knots) is optional. Phase 4 promotes to default if all gates pass.
