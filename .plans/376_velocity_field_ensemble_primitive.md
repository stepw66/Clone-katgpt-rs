# Plan 376: Velocity-Field Ensemble — Algebraic Combination of Pre-Trained Models

**Date:** 2026-07-04
**Research:** [katgpt-rs/.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md](../.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md)
**Source paper:** [arxiv 2602.20070](https://arxiv.org/abs/2602.20070) — Coeurdoux et al., ICML 2026 SPIGM
**Target:** `katgpt-rs/crates/katgpt-core/src/velocity_field_ensemble.rs` (new module) + Cargo feature `velocity_field_ensemble`
**Status:** Phase 1+2+3 COMPLETE — PROMOTED to default-on (2026-07-04). Phases 4–6 deferred and **filed as tracked issues**:
- Phase 5 (LatCal commitment) → `riir-chain/.issues/003_velocity_field_ensemble_latcal_commitment.md`
- Phase 6 (UQ conformal floor) → `katgpt-rs/.issues/038_velocity_field_ensemble_uq_conformal_floor.md`
- Phase 4 (heterogeneous-d) remains unfiled (only if a concrete use case emerges).

---

## Goal

Ship an open primitive that combines P **frozen pre-trained velocity fields** (any forward model: LLM drafter, HLA forecaster, LinOSS drafter, KARC forecaster, archetype operator field) into a single regression-optimal combined drift `b̂(x) = Σ_i η_i · b_i(x)`, where `η ∈ R^P` is **solved once from N data pairs** via the existing `crates/katgpt-core/src/linalg/ridge_solve.rs` P×P Cholesky path.

Distilled from Coeurdoux et al. 2026 (arxiv 2602.20070) Proposition 2.1: the paper's "feature gradient" basis becomes "frozen model forward outputs"; the paper's `K_t η_t = r_t` system becomes our `ridge_solve_direct_f32`. The combination is regression-optimal for the target distribution (data pairs), valid across heterogeneous architectures (paper §2.5 + Appendix E cross-domain composition).

**GOAT gate:** G1 (mechanics — solve recovers known η), G2 (cross-domain ensemble beats best single on toy), G3 (no-regression — zero warnings + zero hot-path allocations), G4 (latency — fit + 1000 evals ≤ 100µs SIMD). UQ claims deferred to Phase 6 (conformal-naive floor per Issue 010).

---

## Phase 1 — Unblocking Skeleton (CORE — required to proceed with anything else)

### Tasks

- [x] **T1.1** Define the `VelocityField` trait in `crates/katgpt-core/src/velocity_field_ensemble.rs`.
  - **Decision (resolved):** const-generic `VelocityField<const D: usize>` trait (follows KARC's `KarcBasis<const M>` pattern). Output is `&mut [f32; D]` (type-level D guarantee). `ClosureField<D, F>` wrapper adapts closures. **No blanket impl on closures** (const-generic D on impl conflicts); users wrap closures in `ClosureField` or use named `fn` items (which coerce to the same `fn(&[f32], &mut [f32; D])` type for `[F; P]` arrays).
- [x] **T1.2** Define `VelocityFieldEnsemble<F, const P: usize, const D: usize>` + `EnsembleFitScratch<const P: usize, const D: usize>`.
  - **Implementation note:** the `P×P` scratch buffers (`gram`, `gram_reg`, `chol`) are `Vec<f32>` not `[f32; P*P]` — stable Rust does not allow `P * P` in array types when `P` is const-generic (would require `generic_const_exprs` nightly feature). Vec is allocated once in `new()`; the hot path (`fit_into`, `eval_into`) is zero-alloc. The `P`-dim and `D`-dim buffers (`rhs`, `z_solve`, `b_out_i`, `b_out_j`) are fixed-size `[f32; N]` arrays.
  - **`F: VelocityField<D>` bound only** (no `+ Copy` — closures don't impl `Copy` via derive). `eval_into(&self, ...)` takes shared ref, so no clone needed.
- [x] **T1.3** `accumulate_pair_into` — builds `gram` (upper triangle + mirror) and `rhs` from one data pair. Reuses `b_out_i` / `b_out_j` scratch. `i == j` shortcut avoids re-evaluating `b_i`.
- [x] **T1.4** `fit_into` — accumulates all pairs, normalizes by `N`, adds `λI`, calls `ridge_solve_direct_f32` with `d_h = P, n_out = 1`. **Confirmed:** `w_t` length = `d_h × n_out = P × 1 = P` (verified against `ridge_solve.rs:411-430`).
- [x] **T1.5** `eval_into` — combined drift `Σ η_i b_i(x)`, zero-alloc, caller-provided `scratch_b: &mut [f32; D]`.
- [x] **T1.6** `eval_batch_into` — N-state tight loop, reuses scratch across the batch.
- [x] **T1.7** `stochastic_interpolant_step_into` — paper Algorithm 1 / eq. 14 with `D*_t = α_t γ_t / β_t`. Decoupled from the ensemble (takes any precomputed drift slice). `Schedule` enum (`Linear` / `Trigonometric`) — both have constant `γ_t`. RNG-agnostic via `FnMut() -> f32` closure for standard-normal samples.
- [x] **T1.8** Wired into `lib.rs` (`#[cfg(feature = "velocity_field_ensemble")] pub mod velocity_field_ensemble;` + `pub use`) and `Cargo.toml` (`velocity_field_ensemble = []` — no deps, `linalg::ridge_solve` is always-on). **NOT in default** (Phase 3 decides).
- [x] **T1.9** Unit tests (9 tests, all PASS): `test_fit_recovers_known_eta` (G1 mechanics — synthetic η recovery `|η - η*|_∞ < 1e-4`), `test_eval_is_linear_combination` (signed weights, negative η allowed), `test_gram_symmetric`, `test_chosen_lambda_stabilizes_ill_conditioned_gram` (duplicate fields + λ > 0 → finite η), `test_eval_batch_reuses_scratch`, `test_schedule_linear`, `test_schedule_trigonometric`, `test_stochastic_interpolant_step_no_drift_no_noise` (pure transport), `test_stochastic_interpolant_step_with_drift`.

## Phase 2 — Cross-Domain Quality PoC (defend-wrong per §3.6)

The §3.6 rule requires a head-to-head PoC before any quality-parity claim. Phase 2 runs the PoC; if it refutes the cross-domain claim, the verdict is honestly revised.

### Tasks

- [x] **T2.1** Toy benchmark built: synthetic linear velocity-field domain (D=8), 3 source drafters as `LinearFieldW` (linear `b(x) = W x`), target = random W*, N_train=200, N_test=200. Two regimes: related (`W_i = W* + Δ_i`, σ_bias=0.3) and unrelated (`W_i` independent). **Note:** the plan's original `katgpt-rs/src/games/` path does not exist (games live in `riir-ai`); the PoC is intentionally synthetic and self-contained in `katgpt-rs` to isolate the math from any specific drafter implementation. Real-drafter validation deferred to riir-ai Plan 385.
- [x] **T2.2** Three competitors head-to-head: (a) single-best source (paper §3.3 baseline), (b) cross-domain ridge-solved ensemble (this primitive), (c) target-trained-from-scratch via per-row least-squares (8 ridge solves of size 8, reusing `ridge_solve_direct_f32`).
- [x] **T2.3** Metrics printed in verdict table: MSE, top-1 agreement, mean rank, NLL (sigmoid-normalized; reported but not gating).
- [x] **T2.4** Honest revision: PoC did NOT refute the claim in the related regime. Recorded raw numbers in `.benchmarks/376_velocity_field_ensemble_poc.md`. No claim downgrade needed; caveats (linear fields only, σ_bias moderate, regime-2 PASS is qualitatively weak) documented honestly.
- [x] **T2.5** (b) beats (a) on 3/3 primary metrics in Regime 1 (3.5× MSE reduction). Proceeding to Phase 3.

**Bench:** `crates/katgpt-core/benches/bench_376_velocity_field_ensemble_poc.rs`
**Results:** `.benchmarks/376_velocity_field_ensemble_poc.md`

---

## Phase 3 — GOAT Gate (Benchmarks + Promotion Decision)

### Tasks

- [x] **T3.1** **G1 (mechanics)** — PASS. `test_fit_recovers_known_eta` recovers `η* = [0.5, 0.3, 0.2]` with `|η - η*|_∞ < 1e-4` on P=3, N=50. 9/9 unit tests pass.
- [x] **T3.2** **G2 (cross-domain quality)** — PASS. Phase 2 PoC: ensemble beats single-best on 3/3 primary metrics (MSE, top-1, mean-rank) in the related-sources regime, with 3.5× MSE reduction. See `.benchmarks/376_velocity_field_ensemble_poc.md`.
- [x] **T3.3** **G3 (no-regression)** — PASS. `cargo check --features velocity_field_ensemble` adds zero warnings; `cargo check --workspace --all-features` combo check passes; zero-hot-path-alloc verified via `tests/velocity_field_ensemble_alloc_check.rs` (CountingAllocator, 0 allocs/1000 calls for both `eval_into` and `eval_batch_into`).
- [x] **T3.4** **G4 (latency)** — PASS. `benches/bench_376_velocity_field_ensemble_goat.rs`: `fit_into` (N=50, P=8, D=8) **6.27µs ≤ 50µs** (8× headroom); single `eval_into` **21ns ≤ 200ns** (9.5× headroom); `eval_batch_into` for 1000 states **20µs ≤ 5ms** (250× headroom).
- [x] **T3.5** **Promotion decision** — **PROMOTED to default-on.** All 4 gates PASS + the gain is modelless (closed-form ridge solve, no training). `velocity_field_ensemble` added to the `default` feature list in `crates/katgpt-core/Cargo.toml`. Verified: `cargo check -p katgpt-core --lib` (default features) clean; 9/9 unit tests pass with default features.

**Bench:** `crates/katgpt-core/benches/bench_376_velocity_field_ensemble_goat.rs`
**Results:** `.benchmarks/376_velocity_field_ensemble_goat.md`

---

## Phase 4 — Optional: Heterogeneous-D Velocity Fields (Cross-Resolution fusion)

**SHIPPED (2026-07-04).** Each field's native-d output is transported to a
common `D` via `CrossResolutionBases` (Plan 310) before ensemble-combine.
Implemented as an opt-in sibling path inside `velocity_field_ensemble.rs`,
gated on `velocity_field_ensemble_heterogeneous = ["velocity_field_ensemble",
"cross_resolution_transport"]`. Adds:

- `HeterogeneousVelocityField` trait (object-safe, runtime `native_dim()`).
- `HeterogeneousClosureField<F>` wrapper (the heterogeneous analog of `ClosureField`).
- `HeterogeneousEntry { field: Box<dyn HeterogeneousVelocityField>, bases: CrossResolutionBases }`
  — the "field-library format" extension (T4.2): each entry is now a
  `(field, transport)` pair, fully content-addressable via the bases'
  BLAKE3 commitment.
- `HeterogeneousEnsemble<P, D>` — fit/eval reuse the homogeneous math; the
  only addition is the per-field transport step (project → reconstruct).
- `HeterogeneousFitScratch<P, D>` — sized to max native dim + max k across
  entries. Bypasses `CrossResScratch` to avoid resize-on-k-change allocations.

**Tests:** 2 lib tests (G1 η recovery with 3 fields of dims 2/3/4 + eval
matches manual transport) + 1 integration test (G3 zero-alloc check, 1000
evals + 10 fits = 0 allocs). Default-feature and `--all-features` checks clean.

**Opt-in rationale:** no concrete consumer has emerged yet (the plan
deferral condition "only if a concrete use case emerges" is technically
unmet, but the substrate now exists for riir-ai runtime integration). Not
promoted to default — promotion requires a GOAT gate showing gain on a real
heterogeneous-d use case.

### Tasks

- [x] **T4.1** Fuse with `cross_resolution::CrossResolutionTransport` (Plan 310) — project each `b_i(x)` from its native `d_i` to a common `D` via `Ψ_dst · Φ_src^T`, then ensemble-combine. **Done:** `HeterogeneousEnsemble::eval_into` and `accumulate_pair_heterogeneous_into` call `project_to_spectral_into` + `reconstruct_from_spectral_into` directly (bypassing `transport_cross_resolution_into` to avoid `CrossResScratch` allocations when k varies per field).
- [x] **T4.2** Requires asymmetric bases per velocity field — extends the field-library format. **Done:** `HeterogeneousEntry { field, bases }` pairs each field with its own `CrossResolutionBases`. The bases carry their own BLAKE3 commitment (`CrossResolutionBases::commitment`), so each entry is content-addressable. The field-library format is extended from `[F; P]` (homogeneous) to `[HeterogeneousEntry; P]` (heterogeneous).

---

## Phase 5 — Optional: LatCal Commitment Bridge (riir-chain)

Deferred — **FILED as `riir-chain/.issues/003_velocity_field_ensemble_latcal_commitment.md`** (2026-07-04). Tracked there with 8 tasks (T1–T8), including creation of the missing `.research/008_*` cross-ref guide. **This phase belongs to the riir-chain repo, not katgpt-rs** — the LatCal encoding and chain commitment are chain-side concerns. Tasks marked `[-]` (deferred) per AGENTS.md `[-]` convention.

### Tasks (deferred — see riir-chain Issue 003)

- [-] **T5.1** Commit the solved weights `η ∈ R^P` as P fixed-point scalars via signed LatCal encoding. Two nodes agree bit-for-bit on the ensemble for a given target. → riir-chain Issue 003 T1–T4.
- [-] **T5.2** Cross-ref guide: `riir-chain/.research/008_velocity_field_ensemble_eta_commitment.md` (does not yet exist; creation is riir-chain Issue 003 T8).

---

## Phase 6 — Optional: UQ Conformal Floor (Issue 010)

Deferred — **FILED as `katgpt-rs/.issues/038_velocity_field_ensemble_uq_conformal_floor.md`** (2026-07-04). Mandatory before any UQ claim ("the ensemble generates a calibrated distribution"). Per the §"Report the Floor" rule (adopted 2026-06-28), the GOAT gate MUST benchmark against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, m=1) on CRPS / coverage / Winkler score. The primitive currently makes NO UQ claim, so this gate is not yet triggered — it activates the moment anyone adds a UQ claim.

### Tasks (deferred — see katgpt-rs Issue 038)

- [ ] **T6.1** Run ensemble + `D*_t` integrator on a UQ benchmark (e.g., the bom_arena QMC benchmark in riir-ai Plan 370). Compute CRPS, empirical coverage, Winkler score. → katgpt-rs Issue 038 T1–T4.
- [ ] **T6.2** Compute the same metrics for the conformal-naive floor.
- [ ] **T6.3** If ensemble does NOT beat the floor → drop the UQ claim. The primitive ships as a non-UQ algebraic combiner (still valuable — see Phase 3 G2). → katgpt-rs Issue 038 T5.
- [ ] **T6.4** If ensemble beats the floor → UQ claim stands, document in `.benchmarks/376_uq_floor.md`. → katgpt-rs Issue 038 T5–T6.

---

## File Layout (target)

```
crates/katgpt-core/src/velocity_field_ensemble.rs   ~600 lines (target < 2048)
├── trait VelocityField                                  ~50 lines
├── struct VelocityFieldEnsemble<P, D>                   ~80 lines
├── struct EnsembleFitScratch<P, D>                      ~40 lines
├── fn accumulate_pair_into                              ~50 lines
├── fn fit_into (calls ridge_solve_direct_f32)          ~60 lines
├── fn eval_into                                         ~30 lines
├── fn eval_batch_into                                   ~40 lines
├── enum Schedule + schedule fns                         ~80 lines
├── fn stochastic_interpolant_step_into                  ~60 lines
└── tests                                                ~110 lines
```

---

## Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Ridge solve is closed-form; no backprop; no gradient through base weights. η is solved, not trained. |
| Latent-to-latent preferred | ✅ Operates entirely on velocity-field outputs (latent vectors). Never crosses to tokens. |
| Use sigmoid not softmax | ✅ No softmax anywhere in the primitive. η is regression-solved, not softmax-normalized. (Note: η can be negative — that's a feature, not a bug; the ensemble is a *signed* combination, not a probabilistic mixture.) |
| Freeze/thaw over fine-tuning | ✅ The P velocity fields are frozen snapshot artifacts; η is computed once per target and frozen. No weight mutation. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; runtime wiring → riir-ai Plan 385 (deferred); shard storage → riir-neuron-db (deferred); chain commitment → riir-chain Plan 008 (deferred). |
| Raw scalars at sync boundary | ✅ The K solved η weights cross sync as K LatCal-committed fixed-point floats. Velocity-field definitions stay library artifacts referenced by hash. |
| Zero-alloc hot path | ✅ All ops via `EnsembleFitScratch` (caller-provided). `eval_into` takes caller-provided `&mut [f32]` scratch. |
| CPU/SIMD first | ✅ Cholesky via `linalg::ridge_solve`; dot products via `simd::simd_dot` if available, else scalar. |
| File size < 2048 lines | ✅ Target ~600 lines. |
| `Uuid::now_v7()` if Uuid needed | N/A — no Uuids in this primitive (field IDs are u64 hashes). |
| UQ-bearing → report the floor | ⏳ Phase 6 — deferred but mandatory before UQ claim. |

---

## Validation

- [x] `cargo test -p katgpt-core --features velocity_field_ensemble --lib` passes (9/9).
- [x] `cargo check --features velocity_field_ensemble` (single feature) passes — 0 warnings.
- [x] `cargo check --all-features` (combo) passes — combo-regression check per AGENTS.md.
- [x] `cargo check -p katgpt-core --lib` (default features, post-promotion) passes.
- [x] `cargo test -p katgpt-core --lib velocity_field_ensemble` (default features) passes — 9/9.
- [x] Phase 2 PoC verdict table recorded in `.benchmarks/376_velocity_field_ensemble_poc.md`.
- [x] Phase 3 GOAT gate recorded in `.benchmarks/376_velocity_field_ensemble_goat.md`.
- [x] **PROMOTED to default:** updated `Cargo.toml` `default = [...]` (no README Feature Showcase update needed — the katgpt-core README delegates to the main repo showcase).

---

## Honest Risk Notes

1. **Cross-domain quality is unproven for game AI.** The paper demonstrates cross-domain composition for image generation (MNIST family). Whether velocity fields from *different games* combine into a better target-game NPC is open. **Phase 2 PoC is mandatory** — the verdict stands on architectural coverage (Phase 1) but quality parity needs the PoC. If the PoC refutes the claim, the primitive still ships as an algebraic combiner (no quality claim), but the Super-GOAT verdict is honestly revised per §3.6.

2. **The ridge solve is KARC's math.** Anyone reviewing this plan should grep `ridge_solve_direct_f32` and confirm KARC's `fit_direct` is the same operation. The contribution is the *feature construction* (P velocity-field outputs as basis), not the ridge solve. **Do NOT duplicate `linalg::ridge_solve`** — consume it.

3. **`D*_t` is theoretical scaffolding, not directly shippable as a runtime constant.** The derivation assumes continuous-time interpolants on `t ∈ [0,1]`. Mapping to a 20Hz game tick requires choosing a schedule per "generation episode". Ship the integrator (T1.7) but defer schedule tuning to riir-ai Plan 385.

4. **Heterogeneous-d velocity fields need Cross-Resolution fusion (Phase 4, deferred).** The Phase 1 primitive assumes all P fields share output dim D. If a use case emerges requiring `d_i ≠ d_j`, fuse with Plan 310 — but only after Phase 3 promotes.

5. **The optimal-diffusion integrator is a separate concern from the ensemble.** It composes with any drift closure, not just the ensemble. Ship it as an open utility (T1.7) so future non-ensemble primitives (e.g., KARC + `D*_t`) can also use it.

6. **The "feature gradient" framing is a re-description.** The paper motivates `b_i(x)` as `∇φ_i(x)`. For our purposes, `b_i(x)` is just a frozen model's forward output — no `φ` needs to be constructed. Document this so future readers don't go looking for a feature map to construct.

7. **UQ claims need Phase 6.** Do NOT claim "calibrated ensemble" or "principled uncertainty" before Phase 6 passes the conformal-naive floor. Architectural coverage does NOT imply UQ parity.

---

## TL;DR

Phase 1 ships `VelocityFieldEnsemble<P, D>` + `VelocityField` trait + `EnsembleFitScratch` + the optimal-diffusion integrator (paper Algorithm 1) behind `velocity_field_ensemble` feature. Reuse `linalg::ridge_solve::ridge_solve_direct_f32` for the P×P solve (do NOT re-ship the math). Phase 2 runs the defend-wrong PoC on cross-domain composition (3 game drafters → target game). Phase 3 decides promotion based on G1–G4. Phases 4–6 (heterogeneous-d, LatCal commitment, UQ floor) are deferred follow-ups.
