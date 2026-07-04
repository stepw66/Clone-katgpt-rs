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

- [ ] **T1.1** Define the `VelocityField` trait in `crates/katgpt-core/src/velocity_field_ensemble.rs`:
  ```rust
  /// A frozen forward model whose output is a velocity/drift vector in R^D.
  ///
  /// Implementors: LLM drafters, HLA forecasters, KARC forecasters, LinOSS
  /// ModalSpec drafters, archetype operator fields, any pre-trained model
  /// whose forward pass produces a D-dim direction. The ensemble treats these
  /// as the regression basis (paper §2.5).
  pub trait VelocityField {
      /// Output dimension D. All P ensemble members must agree on D
      /// (use Cross-Resolution transport — Plan 310 — to project
      /// heterogeneous-d members to common D before ensemble fit).
      const D: usize;

      /// Evaluate the velocity field at state `x`, writing into `out`.
      ///
      /// Zero-allocation contract: implementor MUST NOT allocate; `out` is
      /// caller-provided scratch of length `D`. `x` length is implementor-defined
      /// (some fields take raw state, others take delay-embedded state, etc).
      fn eval_into(&self, x: &[f32], out: &mut [f32]);

      /// Identifier for BLAKE3 commitment of this field's frozen weights.
      /// Two ensemble members with the same `field_id` are duplicates.
      fn field_id(&self) -> u64;
  }

  /// Blanket impl for any zero-alloc closure.
  impl<F: Fn(&[f32], &mut [f32])> VelocityField for F {
      const D: usize = /* inferred via associated const — use generic D on the ensemble instead */;
      fn eval_into(&self, x: &[f32], out: &mut [f32]) { self(x, out); }
      fn field_id(&self) -> u64 { /* default */ 0 }
  }
  ```
  Decide: blanket-impl-on-closure is awkward with `const D`; prefer a generic `VelocityFieldEnsemble<P, D>` with `impl<const P: usize, const D: usize>`, plus a `dyn VelocityField<D>` form for runtime heterogeneity. Document the trade-off in the module doc.

- [ ] **T1.2** Define the core ensemble struct + scratch:
  ```rust
  pub struct VelocityFieldEnsemble<F: VelocityField, const P: usize, const D: usize> {
      fields: [F; P],                  // P frozen velocity fields
      eta: [f32; P],                   // solved combination weights (regression-optimal)
      // No other runtime state — the fit is captured by `eta`.
  }

  /// Zero-alloc scratch for the fit. Reused across fits; never allocated in the hot path.
  pub struct EnsembleFitScratch<const P: usize, const D: usize> {
      gram: [f32; P * P],              // K_t[i,j] = E[b_i(I_t)·b_j(I_t)]
      rhs:  [f32; P],                  // r_t[i]   = E[b_i(I_t) · İ_t]
      gram_reg: [f32; P * P],          // K + λI (in-place reg)
      chol:  [f32; P * P],             // Cholesky L
      z_solve: [f32; P],               // scratch for L z = r ; Lᵀ η = z
      b_out_i: [f32; D],               // scratch for b_i(I_t)
      b_out_j: [f32; D],               // scratch for b_j(I_t)
      combined_out: [f32; D],          // scratch for combined drift eval
  }

  impl<const P: usize, const D: usize> EnsembleFitScratch<P, D> {
      pub fn new() -> Self { /* zero-init */ }
  }
  ```

- [ ] **T1.3** Implement `accumulate_pair_into` — given one data pair `(I_t, İ_t)`, accumulate its contribution to `gram` and `rhs`:
  ```rust
  pub fn accumulate_pair_into<F, const P, const D>(
      fields: &[F; P],
      i_t: &[f32],   // length D — the interpolant sample
      dot_i_t: &[f32], // length D — derivative of interpolant (target)
      scratch: &mut EnsembleFitScratch<P, D>,
  )
  where F: VelocityField
  {
      // For each (i, j) pair, accumulate b_i(I_t)·b_j(I_t) into gram[i*P+j].
      // For each i, accumulate b_i(I_t)·İ_t into rhs[i].
      // Reuse b_out_i and b_out_j to avoid per-iteration allocation.
      for i in 0..P {
          fields[i].eval_into(i_t, &mut scratch.b_out_i);
          for j in i..P {  // symmetric — fill upper triangle
              if i == j {
                  scratch.gram[i*P + j] += dot_product(&scratch.b_out_i, &scratch.b_out_i);
              } else {
                  fields[j].eval_into(i_t, &mut scratch.b_out_j);
                  let dot = dot_product(&scratch.b_out_i, &scratch.b_out_j);
                  scratch.gram[i*P + j] += dot;
                  scratch.gram[j*P + i] += dot;  // mirror
              }
          }
          scratch.rhs[i] += dot_product(&scratch.b_out_i, dot_i_t);
      }
  }
  ```
  Note: the `i==j` shortcut avoids re-evaluating `b_i` (already in `b_out_i`). For `i != j`, `b_j` is evaluated once per (i,j) pair — could be optimized to evaluate all P fields once into a `P×D` buffer, then do P² dot products. Choose the simpler form first; benchmark in Phase 3.

- [ ] **T1.4** Implement `fit_into` — given a slice of N data pairs, normalize the accumulated Gram/RHS by N, add ridge regularization `λI`, and solve:
  ```rust
  pub fn fit_into<F, const P, const D>(
      &mut self,
      pairs: &[(/* i_t */ &[f32], /* dot_i_t */ &[f32])],  // or two parallel slices
      lambda: f32,
      scratch: &mut EnsembleFitScratch<P, D>,
  )
  where F: VelocityField
  {
      // Reset scratch to zero.
      scratch.gram.fill(0.0);
      scratch.rhs.fill(0.0);
      // Accumulate all pairs.
      for (i_t, dot_i_t) in pairs {
          accumulate_pair_into(&self.fields, i_t, dot_i_t, scratch);
      }
      // Normalize by N.
      let n = pairs.len() as f32;
      for g in scratch.gram.iter_mut() { *g /= n; }
      for r in scratch.rhs.iter_mut() { *r /= n; }
      // Add λI to Gram → gram_reg.
      scratch.gram_reg.copy_from_slice(&scratch.gram);
      for i in 0..P { scratch.gram_reg[i*P + i] += lambda; }
      // Solve (K + λI) η = r via Cholesky.
      // REUSE linalg::ridge_solve::ridge_solve_direct_f32 — DO NOT re-implement.
      crate::linalg::ridge_solve::ridge_solve_direct_f32(
          &mut self.eta,        // w_t (η)
          &mut scratch.chol,    // L scratch
          &mut scratch.z_solve, // z scratch
          &scratch.gram_reg,    // XᵀX + λI (here: K + λI, P×P)
          &scratch.rhs,         // XᵀY (here: r, P-dim with n_out=1)
          P,                    // d_h = P
          1,                    // n_out = 1 (η is a vector, not a matrix)
      );
  }
  ```
  **Critical:** `ridge_solve_direct_f32` expects `w_t` of length `d_h * n_out`. For our case `d_h = P, n_out = 1` → `w_t` length `P`. Confirm by reading `crates/katgpt-core/src/linalg/ridge_solve.rs:411-430` before implementing.

- [ ] **T1.5** Implement `eval_into` — the combined drift at state `x`:
  ```rust
  pub fn eval_into(&self, x: &[f32], out: &mut [f32]) {
      // b̂(x) = Σ_i η_i · b_i(x). Zero allocation — use a per-call scratch
      // passed by caller (EnsembleFitScratch::combined_out) or a thread-local
      // (last resort). Document the contract.
      let scratch_b = &mut [0.0f32; D]; // TODO: caller-provided
      out.fill(0.0);
      for i in 0..P {
          self.fields[i].eval_into(x, scratch_b);
          let eta_i = self.eta[i];
          for k in 0..D {
              out[k] += eta_i * scratch_b[k];
          }
      }
  }
  ```
  Decision needed: should `eval_into` take a `&mut [f32]` scratch param, or store a `RefCell<[f32; D]>` for self-contained use? Recommend caller-provided scratch (matches `FuncAttnScratch` pattern).

- [ ] **T1.6** Implement `eval_batch_into` — for N states, evaluate all in a tight loop. Used for hot-path inference (e.g., 1000 ticks of an NPC's HLA update). Vec-init outside, reuse inside.

- [ ] **T1.7** Implement the optimal-diffusion-schedule integrator (paper Algorithm 1, eq. 14) as a separate function — NOT coupled to the ensemble:
  ```rust
  /// Step the optimal-diffusion SDE from `x_t` to `x_{t+h}` using the ensemble
  /// drift. Implements paper eq. 14 with `D*_t = α_t γ_t / β_t`.
  ///
  /// NOT coupled to VelocityFieldEnsemble — takes any drift closure.
  /// Handles the singular endpoint t=0 seamlessly (paper §2.4).
  pub fn stochastic_interpolant_step_into(
      x_t: &[f32],         // current state, length D
      x_out: &mut [f32],   // next state, length D
      alpha_t: f32, beta_t: f32, gamma_t: f32,
      alpha_t_plus_h: f32, beta_t_plus_h: f32, gamma_t_plus_h: f32,
      h: f32,
      drift_at_t: &[f32],  // b̂_t(x_t), precomputed by the caller via ensemble.eval_into
      rng: &mut impl Rng,  // for the Brownian increment
  ) {
      // eq. 14:
      // X_{t+h} = (β_t/β_{t+h}) X_t
      //        + h (1 + β_t/β_{t+h}) b̂_t(X_t)
      //        + sqrt(h (α_t β_t γ_t + α_{t+h} β_{t+h} γ_{t+h}) / β_{t+h}) · g_t
      // where g_t ~ N(0, I_D).
      let beta_ratio = beta_t / beta_t_plus_h;
      let drift_coeff = h * (1.0 + beta_ratio);
      let noise_coeff = (h * (alpha_t * beta_t * gamma_t
                          + alpha_t_plus_h * beta_t_plus_h * gamma_t_plus_h)
                        / beta_t_plus_h).sqrt();
      for k in 0..x_t.len() {
          let g = rng.standard_normal(); // or similar
          x_out[k] = beta_ratio * x_t[k] + drift_coeff * drift_at_t[k] + noise_coeff * g;
      }
  }
  ```
  Schedule choices (linear `α_t = 1-t, β_t = t`, trigonometric `α_t = cos(πt/2), β_t = sin(πt/2)`) live in a `Schedule` enum. **Schedules are pure functions of t** — no state, no allocation.

- [ ] **T1.8** Wire into `crates/katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "velocity_field_ensemble")]
  pub mod velocity_field_ensemble;
  ```
  Add to `Cargo.toml`:
  ```toml
  [features]
  velocity_field_ensemble = ["linalg"]  # depends on linalg::ridge_solve (always available)
  ```
  Add to default features list: **NOT default** (Phase 3 decides promotion).

- [ ] **T1.9** Unit tests in `crates/katgpt-core/src/velocity_field_ensemble.rs`:
  - `test_fit_recovers_known_eta` — construct 3 synthetic linear velocity fields `b_1(x) = a_1`, `b_2(x) = a_2`, `b_3(x) = a_3` (constant drifts), generate N=50 pairs `(I_t, İ_t)` from a known `η* = [0.5, 0.3, 0.2]`, fit, assert `|η - η*|_∞ < 1e-4`.
  - `test_eval_is_linear_combination` — verify `b̂(x) = Σ η_i b_i(x)` bit-for-bit (within f32 epsilon) on random states.
  - `test_gram_symmetric` — verify `gram[i*P+j] == gram[j*P+i]` after accumulate.
  - `test_chosen_lambda_stabilizes_ill_conditioned_gram` — duplicate velocity fields → singular Gram → solve with `λ=1e-3` → no NaN.
  - `test_eval_batch_reuses_scratch` — call eval_batch 1000 times, verify zero allocations (use a counting allocator in the test).

---

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
