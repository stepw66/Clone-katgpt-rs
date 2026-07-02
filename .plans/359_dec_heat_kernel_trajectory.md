# Plan 359: DEC Heat Kernel Trajectory — Single-Shot Field Prediction

**Date:** 2026-07-02
**Research:** [katgpt-rs/.research/365_PhysiFormer_Single_Shot_Trajectory_Heat_Kernel_DEC.md](../.research/365_PhysiFormer_Single_Shot_Trajectory_Heat_Kernel_DEC.md)
**Source paper:** [arXiv:2606.27364](https://arxiv.org/abs/2606.27364) — PhysiFormer (Chen/Lan/Vedaldi, VGG Oxford)
**Target:** `katgpt-rs/crates/katgpt-core/src/dec/heat_kernel_trajectory.rs` + Cargo feature `dec_heat_kernel_trajectory`
**Status:** Active — Phase 0 (not started)

---

## Goal

Ship a **single-shot DEC cochain field trajectory predictor** via the operator exponential (heat kernel). Given an initial `CochainField` `h₀` and a propagation operator `A = -I + Δ + diag(motor)`, predict `h(t) = exp(t·A)·h₀` — the field state at horizon `t` — in a single operation, avoiding the `O(T·dt²)` error accumulation of T-step `evolve_motor_gated_field` (Plan 357).

**The GOAT claim:** for linear propagation (no ReLU gate), `exp(t·A)·h₀` is the **exact** trajectory — zero error accumulation, exact Hodge-decomposition preservation. Step-by-step Euler `(I + dt·A)^T·h₀` is a first-order approximation with `O(T·dt²)` global error. At long horizons (T > Krylov dimension k ≈ 20–50), the heat kernel is both cheaper and dramatically more accurate.

**Distilled from PhysiFormer (arXiv:2606.27364):** the paper's fundamental contribution is the prediction-strategy principle — single-shot joint trajectory prediction avoids the compounding error of step-by-step autoregressive rollout. PhysiFormer demonstrates this for trained diffusion on 3D mesh physics (100× rigidity improvement at 49 frames). The DEC heat kernel is the modelless analog for our cochain-field substrate.

---

## Phase 1 — Linear Heat Kernel (CORE)

The minimal primitive: `exp(t·A)·h₀` for the linear propagation operator `A = -I + Δ + diag(motor)`, using a precomputed DEC Hodge-Laplacian eigendecomposition.

### Tasks

- [ ] **T1.1** Implement `DecEigendecomposition` struct — stores top-k eigenvalues + eigenvectors of the Hodge-Laplacian for a `CellComplex`. Precompute via Lanczos iteration (offline, once per complex). Cap at `k_max = 64` eigenvectors (sufficient for typical game maps per SLoD precedent, Plan 235).

- [ ] **T1.2** Implement `heat_kernel_trajectory_linear(cx, h0, motor, t, eig) -> CochainField`:
  - Compute `A = -I + Δ + diag(motor)` in the eigenbasis: `A_eig[k] = -1 + λ_k + motor_eig[k]`
  - Apply `exp(t · A_eig[k])` per eigenmode
  - Reconstruct: `h(t) = Σ_k exp(t·A_eig[k]) · (v_kᵀ·h₀) · v_k`
  - **Exact** for linear propagation — this is the load-bearing claim.

- [ ] **T1.3** Implement `heat_kernel_trajectory_linear_into(cx, h0, motor, t, eig, out)` — zero-alloc variant (write into pre-allocated `CochainField`).

- [ ] **T1.4** Unit test: `linear_heat_kernel_matches_euler_at_t1` — at `t = dt` (one step), `exp(dt·A)·h₀ ≈ (I + dt·A)·h₀` to within `O(dt²)`. Verify the two agree at small `dt`.

- [ ] **T1.5** Unit test: `linear_heat_kernel_exact_diverges_from_euler_at_long_horizon` — at `t = 50·dt`, the heat kernel is exact while Euler accumulates error. Construct a test field with a known exact trajectory (e.g., pure harmonic component — should be preserved exactly by heat kernel, slowly drift under Euler).

- [ ] **T1.6** Unit test: `hodge_decomposition_preserved` — decompose `h₀` via `hodge_decompose` into exact/coexact/harmonic; after `heat_kernel_trajectory_linear`, re-decompose `h(t)` and verify the harmonic component is unchanged (eigenvalue 0 → `exp(0) = 1`), while exact/coexact are damped by their eigenvalues.

**Phase 1 exit:** `cargo test -p katgpt-core --features dec_heat_kernel_trajectory` passes. The linear heat kernel is exact; Euler is approximate. G1 (correctness) conceptually passes by construction.

---

## Phase 2 — Krylov Online Path

For large complexes where eigendecomposition is prohibitive (256×256 = 65k vertices), use Krylov subspace approximation.

### Tasks

- [ ] **T2.1** Implement `krylov_expmv(a_apply: F, h0: &[f32], t: f32, k: usize) -> Vec<f32>` where `a_apply` is a closure computing `v → A·v` (sparse matrix-vector product). Uses Arnoldi iteration to build the k-dimensional Krylov basis `V_k`, solves the small `exp(t·H_k)` on the projected Hessenberg matrix `H_k = V_kᵀ·A·V_k`, reconstructs `V_k · exp(t·H_k) · V_kᵀ · h₀`.

- [ ] **T2.2** Implement `heat_kernel_trajectory_krylov(cx, h0, motor, t, k)` — wraps `krylov_expmv` with the DEC `A` operator (built from `hodge_laplacian` + motor diagonal).

- [ ] **T2.3** Unit test: `krylov_converges_to_eigendecomposition` — at `k = k_max`, the Krylov result matches the eigendecomposition result to within tolerance.

- [ ] **T2.4** Benchmark: `criterion` group comparing (a) eigendecomposition heat kernel, (b) Krylov heat kernel at k=20/30/50, (c) T-step Euler at T=20/50/100/200. Report latency + L2 error vs the eigendecomposition ground truth. **This is the G2 (latency) + G1 (accuracy) gate data.**

**Phase 2 exit:** Krylov path works for large complexes. Benchmark data exists for the GOAT gate.

---

## Phase 3 — Nonlinear Exponential Integrator (ReLU gate)

Extend to the nonlinear case: `h_{t+1} = (I + dt·A)·ReLU(h_t)` where the ReLU gate makes propagation non-negative.

### Tasks

- [ ] **T3.1** Implement `expm_source_term_quadrature` — the Duhamel integral `∫₀ᵗ exp((t-s)·L)·N(h(s))ds` approximated by Gauss-Legendre quadrature, where `L` is the linear part (Δ) and `N(h) = ReLU(h)` is the nonlinear source.

- [ ] **T3.2** Implement `heat_kernel_trajectory_nonlinear(cx, h0, motor, t, eig, n_quad_points)` — combines linear heat kernel on `L` with quadrature on the ReLU source term.

- [ ] **T3.3** Unit test: `nonlinear_matches_step_by_step_at_small_dt` — at small `dt`, the exponential integrator agrees with `evolve_motor_gated_field` (they converge to the same ODE solution).

- [ ] **T3.4** Unit test: `nonlinear_diverges_from_euler_at_long_horizon` — at long horizon, the exponential integrator (higher-order) is more accurate than Euler. Construct a test case where Euler drifts but the exponential integrator stays close to a fine-grained reference (many small Euler steps).

**Phase 3 exit:** Nonlinear path works. The gain over Euler depends on nonlinearity stiffness — the benchmark quantifies it.

---

## Phase 4 — Multi-Hypothesis Trajectory (BoM extension)

The modelless analog of PhysiFormer's generative uncertainty: sample K diverse plausible trajectories.

### Tasks

- [ ] **T4.1** Implement `heat_kernel_trajectory_bom(cx, h0, motor, t, eig, k_hypotheses, perturbation) -> Vec<CochainField>` — perturb the initial state `h₀` (or motor vector) in K directions on the harmonic subspace (eigenvalue 0 → perturbations persist, producing genuinely different futures), apply the heat kernel to each, return K trajectories.

- [ ] **T4.2** Unit test: `bom_produces_diverse_trajectories` — verify K trajectories have non-trivial L2 spread (not identical) AND preserve topological invariants individually.

- [ ] **T4.3** Connection to `best_belief.rs`: verify the K-hypothesis trajectory samples are compatible with the existing BoMSampler API (the trajectory is a "belief" in trajectory-space).

**Phase 4 exit:** Multi-hypothesis trajectory sampling works. This is the speculative phase — the gain depends on whether harmonic-subspace perturbation produces meaningfully diverse futures.

---

## Phase 5 — GOAT Gate

### Tasks

- [ ] **T5.1 G1 (correctness — linear):** `linear_heat_kernel_exact` — for a test field with known analytical solution, verify `‖heat_kernel(t) − exact(t)‖ < 1e-6` at t=1, 10, 50, 100. The Euler baseline should diverge.

- [ ] **T5.2 G1 (correctness — nonlinear):** `nonlinear_expm_vs_fine_euler` — compare exponential integrator against a 10× finer Euler reference. Target: exponential integrator within 1% of fine reference at t=50 with k=30 Krylov dims.

- [ ] **T5.3 G2 (latency):** `criterion` benchmark — Krylov heat kernel (k=30) vs T-step Euler at T=50, T=100, T=200 on a 64×64 grid. Target: Krylov ≤ 2× Euler latency at T=100 (the break-even point per Research 365 §7).

- [ ] **T5.4 G3 (Hodge preservation):** `hodge_decomposition_drift` — measure the change in harmonic component magnitude after trajectory prediction. Heat kernel: 0 drift (exact). Euler: measure drift. Target: heat kernel drift < 1e-10, Euler drift > 0.

- [ ] **T5.5 G4 (alloc-free after precompute):** `alloc_check` — after eigendecomposition precompute, `heat_kernel_trajectory_linear_into` should allocate 0 bytes (verified via custom allocator). Krylov path allowed one allocation for the Krylov basis.

- [ ] **T5.6 G5 (no-regression):** `cargo test -p katgpt-core --features dec_heat_kernel_trajectory` — all existing DEC tests still pass.

- [ ] **T5.7 Promotion decision:**
  - If G1 (linear exact) + G2 (latency ≤ 2× at T=100) + G3 (zero Hodge drift) all pass → promote `dec_heat_kernel_trajectory` to default-on.
  - If the gain is only at T > 200 (very long horizons) → keep opt-in, note the niche.
  - If the nonlinear path (Phase 3) shows < 2× accuracy improvement over Euler → keep nonlinear opt-in, promote only the linear path.
  - Demote: if the Krylov path is never faster than Euler at any tested T → demote Krylov, keep only eigendecomposition path (for precomputed complexes).

**Phase 5 exit:** GOAT gate run, verdict recorded in `.benchmarks/365_dec_heat_kernel_trajectory_goat.md`. Promotion decision made.

---

## Feature Flag

```toml
[features]
dec_heat_kernel_trajectory = ["katgpt-core/dec"]
```

Opt-in initially. Promote to default if G1+G2+G3 pass at T≥50 (per Research 365 verdict).

---

## Dependencies

- `katgpt-core::dec` (Plan 251) — `CellComplex`, `CochainField`, `hodge_laplacian`, `hodge_decompose`, `evolve_motor_gated_field` (Plan 357)
- `katgpt-core::slod` (Plan 235) — `heat_kernel_weights` precedent (KG graph Laplacian; the DEC extension follows the same spectral pattern)
- No new external dependencies (Lanczos/Arnoldi implemented in-repo; no `nalgebra` or `ndarray` needed for the core path)

---

## Honest Expectations

**Most likely outcome:** the linear heat kernel is exact (G1 passes trivially — it's a mathematical identity). The Krylov path is competitive with Euler at T≈50 and wins at T≥100. The nonlinear exponential integrator shows modest improvement over Euler (2–5× accuracy at T=50). The multi-hypothesis BoM extension produces diverse trajectories but the diversity depends on the harmonic subspace dimension (number of holes in the cell complex — for a simply-connected game map, this may be small).

**Promotion:** the linear path promotes to default-on (it's strictly better than Euler for any horizon ≥ 1 step in the limit, and the precompute cost is amortized). The Krylov and nonlinear paths may stay opt-in depending on the benchmark.

**Risk:** the gain may be marginal for game AI use cases where horizons are short (1–2 seconds = 20–40 ticks). The strong case is for sleep-time anticipation (Plan 341, multi-second pre-thinking) and zone-level crowd flow prediction (5+ second horizons). If these use cases don't materialize, the primitive stays as a mathematically clean but underutilized tool.

---

## TL;DR

Ship `exp(t·A)·h₀` — the DEC heat kernel trajectory predictor — as the single-shot modelless analog of PhysiFormer's single-shot trajectory diffusion. For linear DEC propagation, it's **exact** (zero error accumulation, exact Hodge-decomposition preservation). For nonlinear (with ReLU gate), it's a higher-order exponential integrator. Computed via precomputed eigendecomposition (offline) or Krylov subspace (online). GOAT gate: G1 exact-for-linear, G2 latency ≤ 2× Euler at T=100, G3 zero Hodge drift. Feature flag `dec_heat_kernel_trajectory`, promote to default if gate passes at T≥50.
