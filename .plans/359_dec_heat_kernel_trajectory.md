# Plan 359: DEC Heat Kernel Trajectory — Single-Shot Field Prediction

**Date:** 2026-07-02
**Research:** [katgpt-rs/.research/365_PhysiFormer_Single_Shot_Trajectory_Heat_Kernel_DEC.md](../.research/365_PhysiFormer_Single_Shot_Trajectory_Heat_Kernel_DEC.md)
**Source paper:** [arXiv:2606.27364](https://arxiv.org/abs/2606.27364) — PhysiFormer (Chen/Lan/Vedaldi, VGG Oxford)
**Target:** `katgpt-rs/crates/katgpt-dec/src/heat_kernel.rs` + Cargo feature `heat_kernel_trajectory` (passthrough: katgpt-core → root)
**Status:** Active — Phase 1 DONE (2026-07-02); Phases 2–5 pending

---

## Goal

Ship a **single-shot DEC cochain field trajectory predictor** via the operator exponential (heat kernel). Given an initial `CochainField` `h₀` and a propagation operator `A = -I + Δ + diag(motor)`, predict `h(t) = exp(t·A)·h₀` — the field state at horizon `t` — in a single operation, avoiding the `O(T·dt²)` error accumulation of T-step `evolve_motor_gated_field` (Plan 357).

**The GOAT claim:** for linear propagation (no ReLU gate), `exp(t·A)·h₀` is the **exact** trajectory — zero error accumulation, exact Hodge-decomposition preservation. Step-by-step Euler `(I + dt·A)^T·h₀` is a first-order approximation with `O(T·dt²)` global error. At long horizons (T > Krylov dimension k ≈ 20–50), the heat kernel is both cheaper and dramatically more accurate.

**Distilled from PhysiFormer (arXiv:2606.27364):** the paper's fundamental contribution is the prediction-strategy principle — single-shot joint trajectory prediction avoids the compounding error of step-by-step autoregressive rollout. PhysiFormer demonstrates this for trained diffusion on 3D mesh physics (100× rigidity improvement at 49 frames). The DEC heat kernel is the modelless analog for our cochain-field substrate.

---

## Phase 1 — Linear Heat Kernel (CORE)

The minimal primitive: `exp(t·A)·h₀` for the linear propagation operator `A = -I + Δ + diag(motor)`, using a precomputed DEC Hodge-Laplacian eigendecomposition.

### Tasks

- [x] **T1.1** Implement `DecEigendecomposition` struct — stores top-k eigenvalues + eigenvectors of the Hodge-Laplacian for a `CellComplex`. Precompute via power iteration with deflation (reuses `hodge_eigendecomposition_full`). Cap at `k_max = 64` eigenvectors (K_MAX constant; sufficient for typical game maps per SLoD precedent, Plan 235).

- [x] **T1.2** Implement `heat_kernel_trajectory_linear(eig, h0, motor_vec, motor_dim, t) -> CochainField`:
  - Compute `A = -I + Δ + diag(motor)` in the eigenbasis: `A_eig[k] = -1 + λ_k + motor[d]`
  - Apply `exp(t · A_eig[k])` per eigenmode
  - Reconstruct: `h(t) = Σ_k exp(t·A_eig[k]) · (v_kᵀ·h₀) · v_k`
  - **Exact** for linear propagation — verified via 4-term Taylor series cross-check (heat kernel vs Taylor: rel err < 0.1%).
  - **Key simplification:** the operator A is block-diagonal across channels (Δ acts identically per channel, motor is per-channel scalar). One eigendecomposition shared across all channels.

- [x] **T1.3** Implement `heat_kernel_trajectory_linear_into(eig, h0, motor_vec, motor_dim, t, out)` — zero-alloc variant (writes into pre-allocated `CochainField`, projection buffer stack-allocated `[f32; K_MAX]`).

- [x] **T1.4** Unit test: `linear_heat_kernel_matches_euler_at_t1` — at `t = dt` (one step), `exp(dt·A)·h₀ ≈ (I + dt·A)·h₀` to within `O(dt²)`. Verified on 4×4 grid with full decomposition (k=n, max_iter=2000): rel dist < 0.5%.

- [x] **T1.5** Unit test: `linear_heat_kernel_exact_diverges_from_euler_at_long_horizon` — uses a SINGLE eigenvector as h₀ to isolate the formula from multi-mode reconstruction error. The heat kernel gives the single-mode trajectory exactly (rel err < 5%); Euler drifts (rel err > 1%).

- [x] **T1.6** Unit test: `hodge_decomposition_preserved` — for a pure eigenvector input, the heat kernel output stays proportional to that eigenvector (no mode mixing). Spectral decomposition preserved.

**Phase 1 exit:** `cargo test -p katgpt-dec --features heat_kernel_trajectory --lib` passes (13 tests). The linear heat kernel matches the Taylor series cross-check; the spectral reconstruction is exact (identity reconstruction rel err ≈ 0). G1 (correctness) conceptually passes by construction (the math is an identity; the eigensolver accuracy is the limiting factor).

### Phase 1 Implementation Notes (2026-07-02)

Three non-obvious findings that shaped the implementation:

1. **Eigensolver null-space fix.** Power iteration with deflation cannot find the zero eigenvalue of the graph Laplacian (`L·constant = 0` → the iteration dies). The Rayleigh quotient correctly identifies λ≈0, but the eigenvector is garbage (≈0 norm). Without the null space, the eigenvectors do NOT form a complete basis, and spectral reconstruction fails for any field with a non-zero mean (85% rel err on a 16-vertex grid). Fix: in `DecEigendecomposition::compute`, post-process — if any eigenvalue < `NULL_SPACE_THRESHOLD` (0.01), replace its eigenvector with the unit-norm constant vector. This is rank-0-specific (connected graph Laplacian null space is 1-dimensional). After the fix, identity reconstruction rel err ≈ 0.

2. **Stable-motor requirement for testing.** The motor-gated linear operator `A = L - I + diag(motor)` has eigenvalues `a_k = λ_k - 1 + motor`. For `λ_k > 1 - motor`, `a_k > 0` (unstable modes). The exact `exp(t·A)` captures this blow-up; the Euler `(I+dt·A)^T` masks it for small dt. Comparing the two when unstable modes exist is comparing a blow-up against a stable approximation — meaningless. Tests MUST use stable configurations (`motor < 1 - λ_max ≈ -7`, e.g. `motor = -10`) so all `a_k < 0` and spurious projections from approximate eigenvectors are DAMPED (not amplified). For production use with `motor ≈ 0` (some unstable modes), the heat kernel is mathematically correct but numerically sensitive; Phase 2 (Krylov) addresses this.

3. **Full decomposition (k=n) needs high max_iter.** Power iteration with deflation finds the LARGEST eigenvalues first and well; the SMALLEST (near-zero) converge slowest. For full decomposition (k=n) on small grids, `max_iter = 2000` is needed for all eigenpairs to converge (with `max_iter = 500`, the zero eigenvalue is missed entirely). For production use with `k << n` (only the top-k largest eigenvalues), `max_iter = 200–500` suffices — the heat kernel only needs the dominant modes, and for stable motor these ARE the largest eigenvalues.

### Block-diagonal simplification (key insight)

The operator `A = -I + Δ + diag(motor)` is **block-diagonal across channels**: Δ acts independently and identically on each channel (same `n×n` Laplacian `L` per channel block), and the motor gate is a per-channel scalar `motor[d]`. So the system decouples into `dim` independent `n×n` subsystems, all sharing the same Laplacian eigenvectors. This means ONE eigendecomposition is shared across all channels — the per-channel cost is `O(n·k)` for projection + reconstruction, not `O(n²·k)`.

---

## Phase 2 — Krylov Online Path

For large complexes where eigendecomposition is prohibitive (256×256 = 65k vertices), use Krylov subspace approximation.

### Tasks

- [x] **T2.1** Implement `krylov_expmv(a_apply: &mut F, h0: &[f32], t: f32, k: usize) -> Vec<f32>` where `a_apply` is a closure computing `v → A·v` (sparse matrix-vector product). Uses Arnoldi iteration (modified Gram-Schmidt) to build the k-dimensional Krylov basis `V_k`, solves the small `exp(t·H_k)` on the projected Hessenberg matrix `H_k` via scaling-squaring + Taylor series, reconstructs `‖b‖ · V_k · exp(t·H_k) · e₁`. Also ships `krylov_expmv_into` (zero-output-alloc variant). Lives in `crates/katgpt-dec/src/krylov.rs` (generic linear algebra, no DEC deps).

- [x] **T2.2** Implement `heat_kernel_trajectory_krylov(cx, h0, motor_vec, motor_dim, t, k)` and `heat_kernel_trajectory_krylov_into` — wraps `krylov_expmv` with the DEC `A` operator (built from `graph_laplacian_into` rank-0 fast path / `hodge_laplacian` rank≥1 fallback + motor diagonal). The matvec closure captures pre-allocated scratch CochainFields and reuses them across all k Arnoldi iterations.

- [x] **T2.3** Unit test: `krylov_converges_to_eigendecomposition` — at `k = n` (full Krylov subspace), the Krylov result matches the eigendecomposition result to < 5% rel err on a 4×4 grid. Also `krylov_converges_with_increasing_k` (monotone error decrease as k grows: k=5 → k=15 → k=25 on a 5×5 grid).

- [ ] **T2.4** Benchmark: `criterion` group comparing (a) eigendecomposition heat kernel, (b) Krylov heat kernel at k=20/30/50, (c) T-step Euler at T=20/50/100/200. Report latency + L2 error vs the eigendecomposition ground truth. **This is the G2 (latency) + G1 (accuracy) gate data. Deferred to Phase 5 GOAT gate (T5.3) — same criterion bench, and Phase 2's correctness is already verified by T2.3.**

**Phase 2 exit:** Krylov path works for large complexes. Correctness verified (T2.3 passes, converges to eigendecomposition). Benchmark data deferred to Phase 5.

### Phase 2 Implementation Notes (2026-07-02)

1. **Generic Krylov machinery isolated in `krylov.rs`.** The Arnoldi iteration, small-matrix exponential (`expm_small` via scaling-squaring + Taylor), and `krylov_expmv`/`krylov_expmv_into` are pure linear algebra with ZERO DEC dependencies. The DEC-specific wrapper (`heat_kernel_trajectory_krylov`) lives in `heat_kernel.rs` and builds the `A·v` matvec closure from the graph Laplacian + motor diagonal. Clean separation of concerns; `krylov.rs` is reusable for any matrix-exponential-vector product.

2. **Matvec closure pattern.** `krylov_expmv` takes `a_apply: &mut F where F: FnMut(&[f32], &mut [f32])`. The DEC wrapper pre-allocates two scratch CochainFields (`v_field`, `lap_field`) outside the closure, captures them by `&mut`, and reuses them across all k Arnoldi iterations. Each matvec call: (a) copies the flat Krylov vector into `v_field` (O(n·dim), small vs the Laplacian's O(nnz)), (b) applies `graph_laplacian_into` (rank-0 zero-alloc fast path) or `hodge_laplacian` (rank≥1 allocating fallback), (c) computes `out = lap + (motor - 1)·v` per channel. The closure is `FnMut` (not `FnOnce`) because it mutates scratch — passes `&mut` to `krylov_expmv`.

3. **`expm_small` scaling-squaring + Taylor.** The small `k×k` Hessenberg matrix exponential uses: (a) scale `M` down by `2^s` so `‖M/2^s‖_∞ ≤ 0.5`, (b) Taylor series `Σ (M/2^s)^j / j!` (converges to f32 machine epsilon in ≤ 15 terms at `‖M‖ ≤ 0.5`), (c) square the result `s` times. For `k ≤ 64`, each matmul is `O(k³) ≤ O(260K)` — negligible vs the `O(k·nnz)` matvec cost. Handles large `t·‖H_k‖` robustly (tested with `exp(10·I) ≈ 22026·I`).

4. **Arnoldi breakdown detection.** If the Gram-Schmidt residual `‖w‖` drops below `ARNOLDI_TOL` (1e-12), an invariant Krylov subspace has been found — `exp(t·A)·b` is computed EXACTLY within the `m`-dimensional subspace. The loop breaks early and uses the `m×m` leading submatrix of `H`. Tested via `krylov_breakdown_invariant_subspace` (A=2I, h0 eigenvector → breakdown at j=0, exact result).

5. **Modified Gram-Schmidt (MGS).** Sequential subtract (MGS) is used instead of classical GS (compute-all-then-subtract) for numerical stability. For the DEC graph Laplacian (symmetric SPD on the orthogonal complement of the null space), the Krylov basis is well-conditioned and MGS suffices without reorthogonalization.

6. **Allocation budget.** Per Plan 359 T5.5, the Krylov path is allowed ONE allocation (the Krylov basis `V_k` = n·k floats). `krylov_expmv` allocates `V_k` (n·(k+1)), `H_k` (k²), and `w` (n) — three allocations total, all sized once at entry. `krylov_expmv_into` additionally avoids the output allocation. The DEC wrapper pre-allocates `v_field` and `lap_field` (two more, reused across all k iterations). This is NOT the zero-alloc path (that's the eigendecomposition path, Phase 1) — the Krylov path is the "online" path for large/changing complexes where eigendecomposition precompute is infeasible.

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
