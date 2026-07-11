# Plan 426: Manifold-Constrained Concept Erasure Primitive

**Date:** 2026-07-11
**Research:** [409_MANCE_Manifold_Aware_Concept_Erasure.md](../.research/409_MANCE_Manifold_Aware_Concept_Erasure.md)
**Source paper:** [arXiv:2607.03973](https://arxiv.org/abs/2607.03973) — Avitan, Goldberg, Elazar, *MANCE: Manifold Aware Concept Erasure*, Jul 2026
**Code:** [github.com/MatanAvitan/mance](https://github.com/MatanAvitan/mance)
**Target:** `katgpt-rs/crates/katgpt-core/src/manifold_erasure.rs` (new module) + Cargo feature `manifold_erasure`
**Status:** Active — Phase 0 (this plan)

---

## Goal

Ship a modelless manifold-constrained concept erasure primitive that performs **surgical direction removal** from latent state: given a latent vector `x`, an erasure direction `u` (the concept to remove), and a set of natural reference representations `X⁽⁰⁾` (the manifold), compute `x̃ = x - λ·<x, û>·û` where `û` is the erasure direction projected onto the local tangent space of the natural manifold, spectrally weighted by local singular values, and `λ` is bounded by a per-sample local-radius trust region.

This is the **local, spectrally-weighted, trust-bounded erasure** member of the subspace-projection family:

| Primitive | Basis | Gating | Operation |
|---|---|---|---|
| Plan 412 `subspace_steering` | Global, k-dim | Fixed α_j | INJECTION |
| Plan 423 `spectral_rewire` | Global SVD of W₀ | None | DECOMPOSITION (weights) |
| Plan 425 `tilr` | Global U_r | γ-alignment | INJECTION |
| **MANCE (this)** | **Local k-NN tangent** | **σ^α + ε·r_i** | **ERASURE** |

**Feature flag:** `manifold_erasure` in `katgpt-core/Cargo.toml` (opt-in). Root `katgpt-rs/Cargo.toml` forwards as `manifold_erasure = ["katgpt-core/manifold_erasure"]`. NOT in root `default` until GOAT gate passes.

**The probe is a CONSUMER concern.** This primitive CONSUMES a pre-computed erasure direction (from MAG Plan 418, CNA Plan 087, HLA EmotionDirections, or an external probe). It does NOT train a probe. The mechanism (local tangent + spectral weighting + trust region) is 100% modelless linear algebra.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/manifold_erasure.rs` module (feature-gated under `manifold_erasure` in `katgpt-core/Cargo.toml`).
- [x] **T1.2** Define types:
  - `ManceConfig { epsilon: f32, lambda_max: f32, alpha: f32, k: usize, r: usize }` — all dimensionless, defaults `epsilon=0.1, lambda_max=64.0, alpha=1.0, k=8, r=8`.
  - `ManceScratch` — pre-allocated scratch buffers: `neighbor_distances: Vec<f32>`, `centered_neighbors: Vec<f32>` (k×d), `tangent_basis: Vec<f32>` (d×r), `singular_values: Vec<f32>` (r), `projection_coords: Vec<f32>` (r), `tangent_direction: Vec<f32>` (d), `mean_neighbor: Vec<f32>` (d). `with_capacity(d, k, r)` constructor.
  - `ManceStepInfo { lambda: f32, displacement: f32, local_radius: f32, alignment: f32 }` — diagnostic output (step size, actual displacement, neighborhood radius, gradient-tangent alignment).
  - `ManceError` — `#[repr(u8)]` enum: `DimensionMismatch=0`, `InsufficientNeighbors=1`, `ZeroGradient=2`, `InvalidConfig=3`.
- [x] **T1.3** Implement `knn_distances_into(x, natural_pool, k, scratch) -> &[f32]` — compute L2 distances from `x` to all natural representations, select k smallest. O(N·d) for N natural points. Use `simd_dot_f32` for distance computation. Write k indices + distances into scratch.
- [x] **T1.4** Implement `estimate_local_tangent_into(x, natural_neighbors, r, scratch) -> (&[f32], &[f32])` — mean-center k neighbors → form S_i (k×d) → SVD via `thin_svd_into` (Plan 301) → keep top-r right singular vectors as tangent basis B (d×r) + singular values σ (r). The neighborhood is drawn from the FIXED natural pool but queried at x's CURRENT position (per MANCE §3.2).
- [x] **T1.5** Implement `tangent_erasure_direction_into(x, gradient, basis, sigma, alpha, scratch) -> &[f32]` — normalize gradient `u = ∇f/||∇f||`, project `c = Bᵀu` (r dot products), spectrally weight `d = B·diag(σ^α)·c` (r weighted sums), normalize `û = d/||d||`. Write into scratch.
- [x] **T1.6** Implement `local_radius_step(x, direction, natural_neighbors, epsilon, lambda_max) -> f32` — compute `r_i = mean(||x_j - x||)` over k neighbors, compute `<x, û>`, return `λ = min(λ_max, ε·r_i / <x, û>)`. Handle `<x, û> ≈ 0` → λ=0 (no-harm: direction orthogonal to x).
- [x] **T1.7** Implement `manifold_erasure_step_into(x, gradient, natural_pool, config, scratch, out) -> ManceStepInfo` — orchestrate T1.3→T1.6, apply `out = x - λ·<x, û>·û`. Zero-alloc (all scratch reused).
- [x] **T1.8** Implement `manifold_erasure_step` (allocating convenience wrapper for non-hot paths).
- [x] **T1.9** Wire module into `crates/katgpt-core/src/lib.rs` behind `#[cfg(feature = "manifold_erasure")]`. Add feature gate to `katgpt-core/Cargo.toml` (deps: `katgpt-types` for SIMD, `subspace_phase_gate` for SVD — both already in katgpt-core). Forward in root `katgpt-rs/Cargo.toml`.
- [x] **T1.10** Unit tests:
  - `knn_returns_correct_neighbors` — known distances, verify k smallest selected.
  - `tangent_basis_orthonormal` — verify BᵀB ≈ I_r.
  - `spectral_weighting_prioritizes_high_sigma` — verify high-σ axes get more mass.
  - `trust_region_bounds_displacement` — verify `||x̃ - x|| ≤ ε·r_i`.
  - `zero_gradient_no_harm` — gradient=0 → out=x bit-identically.
  - `orthogonal_direction_no_harm` — gradient ⊥ tangent basis → out=x bit-identically (λ=0).
  - `erasure_reduces_target_alignment` — after step, `|<x̃, u>| < |<x, u>|`.

**Phase 1 exit:** `cargo test -p katgpt-core --features manifold_erasure --lib` green; `cargo check --features manifold_erasure` clean; `cargo check --no-default-features` clean; `cargo check --all-features` clean.

---

## Phase 2 — Iterative Loop + Closed-Form Preprocessing

### Tasks

- [x] **T2.1** Implement `manifold_erasure_loop_into(x, gradient_fn, natural_pool, config, n_rounds, scratch, out)` — iterative application of `manifold_erasure_step_into` for `n_rounds` rounds. The `gradient_fn` is a closure that provides the erasure direction at each round (the caller's probe — MAG/CNA/EmotionDirections). This is the modelless analog of MANCE's iterative loop with probe refit — the caller re-mines the direction between rounds if desired.
- [x] **T2.2** Implement `leace_first_moment_into(x, class_mean_pos, class_mean_neg, scratch, out)` — rank-1 closed-form erasure: project out the class-mean difference direction. `out = x - (<x, d_mean>/||d_mean||²)·d_mean` where `d_mean = μ₊ - μ₋`. This is MANCE+'s LEACE preprocessing.
- [x] **T2.3** Implement `covmatch_second_moment_into(x, delta_sigma_top2_eigvecs, scratch, out)` — rank-2 closed-form erasure: project out the top-2 eigenvectors of ΔΣ = Σ₊ - Σ₋. Orthonormalize with mean direction via QR. This is MANCE++'s CovMatch preprocessing.
- [x] **T2.4** Implement `mance_plus_step_into` (LEACE + loop) and `mance_plus_plus_step_into` (LEACE + CovMatch + loop) — the composed variants.
- [x] **T2.5** Unit tests for preprocessing:
  - `leace_removes_class_mean_difference` — after LEACE, `<x̃, d_mean> ≈ 0`.
  - `covmatch_removes_covariance_asymmetry` — after CovMatch, class-conditional variance asymmetry reduced.
  - `preprocessing_preserves_orthogonal_directions` — directions ⊥ the erased directions are unchanged.

**Phase 2 exit:** all new tests pass; `cargo check --all-features` clean.

---

## Phase 3 — GOAT Gate

### Tasks

- [x] **T3.1** `benches/bench_426_manifold_erasure_goat.rs` — GOAT gate:
  - **G1 (correctness):**
    - G1a — erasure reduces target-direction energy: `|<x̃, u>| < |<x, u>|` by ≥50% after 1 step (synthetic data, known direction).
    - G1b — preserves orthogonal directions: for directions `v ⊥ tangent basis`, `|<x̃, v> - <x, v>| < 1e-6` (bit-identical preservation).
    - G1c — no-harm at zero gradient: gradient=0 → `out == x` bit-identically.
    - G1d — no-harm at orthogonal gradient: gradient ⊥ tangent basis → `out == x` bit-identically (λ=0).
    - G1e — trust region bound: `||x̃ - x|| ≤ ε·r_i` for all test cases.
    - G1f — spectral weighting correctness: `d = B·diag(σ^α)·c` matches hand-computed values on a known 4×2 basis.
  - **G2 (perf):**
    - G2a — HLA scale (d=8, k=8, r=8): `manifold_erasure_step_into` < 500ns (SIMD, release, `black_box`, 10K iters).
    - G2b — Shard scale (d=64, k=16, r=16): `manifold_erasure_step_into` < 5µs.
    - G2c — 10-round loop at HLA scale: < 5µs total.
  - **G3 (no-regression):** `cargo check --all-features`, `cargo check --no-default-features`, `cargo test -p katgpt-core --lib` — all clean, zero new warnings.
  - **G4 (alloc-free):** `manifold_erasure_step_into` allocates 0 bytes over 100 steady-state calls (CountingAllocator). Companion gate verifies the 0-alloc result is non-degenerate.
  - **G5 (modelless):** `manifold_erasure = []` deps in Cargo.toml (only `katgpt-types` for SIMD, `subspace_phase_gate` for SVD — both already in katgpt-core). No `riir_train`/`riir_gpu`.
  - **G6 (ablation — the AmbCE++ control):** compare MANCE step vs unconstrained erasure (same λ, no tangent projection). Verify MANCE preserves more orthogonal energy than unconstrained on synthetic data where the gradient has off-manifold components.
- [x] **T3.2** If G1–G6 all pass → promote `manifold_erasure` to root `default` in `katgpt-rs/Cargo.toml` + `katgpt-core/Cargo.toml`.
- [x] **T3.3** Record benchmark in `katgpt-rs/.benchmarks/426_manifold_erasure_goat.md`.

**Phase 3 exit:** all gates PASS; feature promoted to default-on with pure modelless gain. OR: if G2/G6 fail, keep opt-in and document why.

---

## Phase 4 — Example + Docs

### Tasks

- [x] **T4.1** Example: `examples/manifold_erasure_demo.rs` — synthetic 8-d latent state, 50 natural reference points, erase a concept direction. Show: (a) target alignment drops, (b) orthogonal directions preserved, (c) displacement within trust region. Compare MANCE vs unconstrained erasure (the AmbCE++ ablation).
- [x] **T4.2** Add module-level rustdoc with the MANCE algorithm summary, the family table, and the probe-replacement note.
- [x] **T4.3** Update `katgpt-rs/README.md` Feature Showcase section with a MANCE entry.

---

## Design Notes

### Why local tangent (not global SVD like TILR/spectral_rewire)

TILR (Plan 425) uses a GLOBAL invariant subspace U_r discovered from contrastive differences — one basis for all samples. spectral_rewire (Plan 423) uses a GLOBAL SVD of W₀ — one basis for all weight deltas. MANCE's insight is that the manifold is **locally curved**: the tangent space at sample A differs from the tangent space at sample B. A global basis is a first-order approximation that degrades where the manifold curves. The local tangent basis, re-estimated per-sample from natural neighbors, tracks the curvature.

For HLA (d=8) and shards (d=64), the representation dimension is small enough that local k-NN + SVD is cheap (O(k·d·r) per sample). At LLM scale (d=768+), the local SVD becomes the bottleneck (the paper reports ~50% of runtime on local SVDs). Our use case is game AI (d=8) and shards (d=64), so the local approach is tractable.

### Why the probe is a consumer concern

MANCE trains an MLP probe to find the concept direction. In our modelless framework, the probe is replaced by:
- **MAG** (Plan 418) — unsupervised contrastive direction mining (no labels needed).
- **CNA** (Plan 087) — contrastive neuron attribution (labeled pairs).
- **HLA EmotionDirections** — pre-computed affect direction vectors.

The primitive CONSUMES a direction vector; it does not compute one. This is the R368 "LLM-as-implementation" pattern: the probe is one instantiation of computing "which direction to erase"; our substrate provides modelless alternatives.

### The ε=0.1 transfer property

The key insight from the paper: ε is **dimensionless** (ratio of displacement to local neighborhood radius). The local r_i absorbs the panel's representation scale. So ε=0.1 works for both HLA (d=8, small magnitudes) and shards (d=64, larger magnitudes) without per-setting tuning. This is why the paper's hyperparameters transfer across all 119 settings.

### K=1 parity with TILR

At `r=1, α=0, ε→∞, λ_max→∞`, the MANCE step degenerates to: project gradient onto the single dominant tangent direction, take a full step. This is NOT identical to TILR (TILR injects `+η·γ·d`; MANCE subtracts `λ·<x,û>·û`). There is no K=1 parity contract between MANCE and TILR — they are different operations (injection vs erasure). The parity contract is with `orthogonal_projection_into` (riir-poc): at `α=0, r=d` (full tangent = full space), MANCE reduces to standard orthogonal projection of x onto the complement of û.

---

## References

- **Research note:** [katgpt-rs/.research/409_MANCE_Manifold_Aware_Concept_Erasure.md](../.research/409_MANCE_Manifold_Aware_Concept_Erasure.md)
- **Source paper:** [arXiv:2607.03973](https://arxiv.org/abs/2607.03973) — Avitan, Goldberg, Elazar
- **Code:** [github.com/MatanAvitan/mance](https://github.com/MatanAvitan/mance)
- **Closest cousins:**
  - [Plan 425 TILR](425_tilr_invariant_subspace_refinement.md) — alignment-gated subspace correction (global basis, injection)
  - [Plan 423 spectral_rewire](423_spectral_rewire_primitive.md) — weight-delta SVD purification (global basis, weights)
  - [Plan 412 subspace_steering](412_subspace_steering_field_primitive.md) — k-dim block steering (global block, injection)
  - [Plan 329 non_interference_branches](329_non_interference_memory_branches.md) — orthogonal direction allocation
  - [Plan 418 MAG](418_mag_activation_geometry_primitive.md) — unsupervised direction mining (probe replacement)
