# Plan 376: Velocity-Field Ensemble — Algebraic Combination of Pre-Trained Models

**Date:** 2026-07-04
**Research:** [katgpt-rs/.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md](../.research/375_Kernelized_Stochastic_Interpolant_Velocity_Field_Ensemble.md)
**Source paper:** [arxiv 2602.20070](https://arxiv.org/abs/2602.20070) — Coeurdoux et al., ICML 2026 SPIGM
**Target:** `katgpt-rs/crates/katgpt-core/src/velocity_field_ensemble.rs` (new module) + Cargo feature `velocity_field_ensemble`
**Status:** Active — Phase 1 unblocking

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

- [ ] **T2.1** Build a toy benchmark: 3 primitive game drafters (bomber-policy, go-policy, monopoly-policy) each as a `VelocityField<D=8>` (8-dim action-probability vector). Train each policy on its own game (use existing bomber/go/monopoly primitives in `katgpt-rs/src/games/`). Target domain: a 4th game (fft-arena) with N=200 data pairs.

- [ ] **T2.2** Three competitors head-to-head on the target:
  - **(a) Frozen/no-adaptation baseline:** single best single-domain drafter (paper §3.3 baseline).
  - **(b) Cross-domain ensemble (this primitive):** ridge-solve-combine the 3 source drafters + a weak target-trained drafter.
  - **(c) Single target-trained drafter:** trained from scratch on N=200 target pairs (reference).

- [ ] **T2.3** Metrics: top-1 action agreement with ground truth, NLL of true action under drafter, mean rank of true action. Print a verdict table.

- [ ] **T2.4** Honest revision: if (b) does NOT beat (a), record the raw numbers in `katgpt-rs/.benchmarks/376_*.md` as a §"PoC Addendum" and revise the Super-GOAT claim. The architectural coverage stands (Phase 1 ships regardless); the cross-domain quality claim is downgraded to a tracked follow-up in `katgpt-rs/.issues/`.

- [ ] **T2.5** If (b) DOES beat (a), record the win and proceed to Phase 3 promotion.

---

## Phase 3 — GOAT Gate (Benchmarks + Promotion Decision)

### Tasks

- [ ] **T3.1** **G1 (mechanics)** — `test_fit_recovers_known_eta` from T1.9 passes. Target: `|η - η*|_∞ < 1e-4` for P=3, N=50.
- [ ] **T3.2** **G2 (cross-domain quality)** — Phase 2 T2.3 verdict table. **PASS criterion:** ensemble (b) ≥ single-source (a) on at least 2 of 3 metrics. **If FAIL:** demote to opt-in only, do NOT promote.
- [ ] **T3.3** **G3 (no-regression)** — `cargo check --features velocity_field_ensemble` adds zero warnings; `cargo check --all-features` (combo check) passes; on the no-fit no-eval path, zero allocations (verified via counting allocator).
- [ ] **T3.4** **G4 (latency)** — `cargo bench` microbench: full `fit_into` (N=50, P=8, D=8) ≤ 50µs; single `eval_into` ≤ 200ns; `eval_batch_into` for 1000 states ≤ 5ms. Targets are plasma-tier-budget.
- [ ] **T3.5** **Promotion decision:**
  - All 4 PASS → promote `velocity_field_ensemble` to default. Demote the loser if any other primitive occupies the same stack slot (none currently does — this is a new slot: "ensemble combination").
  - G2 FAIL → keep opt-in. Note in `.benchmarks/376_*.md`. Architectural coverage stands (Phase 1 ships); quality claim deferred.
  - G1/G3/G4 FAIL → fix before any promotion.

---

## Phase 4 — Optional: Heterogeneous-D Velocity Fields (Cross-Resolution fusion)

Deferred — only if Phase 3 promotes AND a concrete use case emerges where the velocity fields have different output dims.

### Tasks (deferred)

- [ ] **T4.1** Fuse with `cross_resolution::CrossResolutionTransport` (Plan 310) — project each `b_i(x)` from its native `d_i` to a common `D` via `Ψ_dst · Φ_src^T`, then ensemble-combine.
- [ ] **T4.2** Requires asymmetric bases per velocity field — extends the field-library format.

---

## Phase 5 — Optional: LatCal Commitment Bridge (riir-chain)

Deferred — file as a separate plan in riir-chain after Phase 3 promotes.

### Tasks (deferred)

- [ ] **T5.1** Commit the K solved weights `η ∈ R^P` as K fixed-point scalars via `LatCalMatrix::to_fixed`. Two nodes agree bit-for-bit on the ensemble for a given target.
- [ ] **T5.2** Cross-ref guide: `riir-chain/.research/008_latcal_committed_ensemble_weights.md`.

---

## Phase 6 — Optional: UQ Conformal Floor (Issue 010)

Deferred — mandatory before any UQ claim ("the ensemble generates a calibrated distribution"). Per the §"Report the Floor" rule (adopted 2026-06-28), the GOAT gate MUST benchmark against `ConformalIntervalCalibrator<SeasonalNaiveForecaster>` (Plan 340, m=1) on CRPS / coverage / Winkler score.

### Tasks (deferred)

- [ ] **T6.1** Run ensemble + `D*_t` integrator on a UQ benchmark (e.g., the bom_arena QMC benchmark in riir-ai Plan 370). Compute CRPS, empirical coverage, Winkler score.
- [ ] **T6.2** Compute the same metrics for the conformal-naive floor.
- [ ] **T6.3** If ensemble does NOT beat the floor → drop the UQ claim. The primitive ships as a non-UQ algebraic combiner (still valuable — see Phase 3 G2).
- [ ] **T6.4** If ensemble beats the floor → UQ claim stands, document in `.benchmarks/376_uq_floor.md`.

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

- [ ] `cargo test -p katgpt-core --features velocity_field_ensemble --lib` passes.
- [ ] `cargo check --features velocity_field_ensemble` (single feature) passes.
- [ ] `cargo check --all-features` (combo) passes — combo-regression check per AGENTS.md.
- [ ] Phase 2 PoC verdict table recorded in `.benchmarks/376_*.md`.
- [ ] Phase 3 GOAT gate recorded in `.benchmarks/376_goat.md`.
- [ ] If promoted to default: update `Cargo.toml` `default = [...]` and `README.md` Feature Showcase.

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
