# Plan 407: Sheaf-ADMM Coordination Primitive — `sheaf_admm_step` on `CellComplex`

**Date:** 2026-07-06
**Research:** [katgpt-rs/.research/384](../.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md)
**Source paper:** [arXiv:2605.31005](https://arxiv.org/abs/2605.31005) — Seely, Cupiał, Jones, "Learning Multi-Agent Coordination via Sheaf-ADMM", ICML 2026
**Target:** `katgpt-rs/crates/katgpt-dec/src/sheaf_admm.rs` (new module) + Cargo feature `sheaf_admm`
**Status:** Phase 2 (GOAT gate) COMPLETE — promoted to default-on (2026-07-07). All G1–G6 PASS.

---

## Goal

Ship a modelless `sheaf_admm_step` operator on `CellComplex` that performs one ADMM iteration (x-update proximal solve → z-update sheaf diffusion via `hodge_laplacian` → u-update dual accumulation) given per-vertex primal/consensus/dual cochains and per-edge restriction maps. This is the open adoption hook for the Super-GOAT fusion documented in `riir-ai/.research/314`. Zero training, zero backprop — restriction maps are constructed deterministically (identity / selector) or loaded as a frozen artifact.

The z-update IS sheaf diffusion, which IS gradient descent on the Hodge energy `x^T L_F x` — already shipped as `hodge_laplacian`. This plan wires the surrounding ADMM scaffolding (primal prox, dual accumulation) around the existing operator.

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Add `sheaf_admm` feature to `katgpt-rs/crates/katgpt-dec/Cargo.toml` (default-off until G1–G6 pass).
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-dec/src/sheaf_admm.rs` with the public API surface:
  - `pub struct SheafMaps { d_e: usize, d_v: usize, maps: Vec<[MatrixDimDExDV; 2]> }` — per-edge restriction map pair `(F_{i→e}, F_{j→e})`. Use fixed-size `[[f32; d_v]; d_e]` when `d_e` and `d_v` are bounded at construction (generic consts) — fall back to `Vec<f32>` row-major for runtime-flex dims.
  - `pub enum LocalObjective { DiagonalQuadratic { diag_q: Vec<f32>, q: Vec<f32> }, DiagonalQuadL1 { diag_q: Vec<f32>, q: Vec<f32>, lambda: Vec<f32> } }` — closed-form proximal solvers per paper Appendix A.
  - `pub struct AdmmScratch { ... }` — pre-allocated buffers for sheaf diffusion matvec + temporary cochains.
  - `pub fn sheaf_admm_step(cx, maps, primal_x, consensus_z, dual_u, objective, rho, eta, diffusion_steps, scratch)` — one ADMM iteration, in-place.
  - `pub fn sheaf_admm_step_into(...)` — variant taking pre-sized output buffers (zero-alloc contract).
- [x] **T1.3** Implement x-update (closed-form diagonal quadratic): for each vertex i, `x_i = (Q_i + ρI)^{-1} (ρ(z_i - u_i) - q_i)` — elementwise when Q_i is diagonal. Soft-thresholding extension for L1 variant (paper eq. 17). SIMD-vectorize over the d_v axis.
- [x] **T1.4** Implement z-update as `T` sheaf-diffusion steps:
  - `for t in 0..T { z -= eta * sheaf_laplacian_via_maps(cx, maps, z, scratch); }` where `sheaf_laplacian_via_maps` computes `F^T F z` using the per-edge restriction maps. **This is NOT the same as `hodge_laplacian`** (which uses the cell complex's identity-incidence structure); it uses the explicit restriction maps. Document the relationship: when `F_{i→e}` is the coboundary incidence entry, `sheaf_laplacian_via_maps` reduces to `hodge_laplacian`.
  - Note: for identity restriction maps (homogeneous consensus), we can delegate directly to `hodge_laplacian` as a fast path. Reserve the explicit-maps path for heterogeneous consensus.
- [x] **T1.5** Implement u-update: `u_i += x_i - z_i` (vector add). Trivial, but profile to confirm it's not a bottleneck.
- [x] **T1.6** Identity restriction-map constructor: `SheafMaps::identity(cx, d_v, d_e)` — `F_{i→e} = [I_{d_e}; 0_{d_e × (d_v - d_e)}]` for all edges. The modelless floor.
- [x] **T1.7** Selector restriction-map constructor: `SheafMaps::selector(cx, d_v, dim_indices: &[usize])` — picks a fixed subset of dims per edge. Deterministic, derived from caller-supplied indices (runtime caller picks which dims; this primitive just builds the matrix).

## Phase 2 — GOAT Gate (G1–G6)

### Tasks

- [x] **T2.1** **G1 — DEC identity test.** After K=100 ADMM iterations on a 32×32 grid with identity maps, assert `‖F x‖_∞ < 1e-5` (consensus reached — primal aligns with harmonic subspace). Cross-check against `hodge_decompose`'s harmonic component: the converged `z` must lie in `ker(L_F)` to within numerical tolerance.
- [x] **T2.2** **G2 — dual conservation test.** After each iteration, assert `u^{k+1} - u^k == x^{k+1} - z^{k+1}` bit-exactly (same f32 ops, just reordered). This is the ADMM invariant.
- [x] **T2.3** **G3 — heterogeneous compression test.** For random `x ∈ R^{d_v}` and random valid `SheafMaps` with `d_e < d_v`, assert `‖F x‖ ≤ ‖x‖` (restriction maps are contractions when rows are unit-norm). Property holds by construction; the test guards against future bugs that violate row-normalization.
- [x] **T2.4** **G4 — latency benchmark.** `criterion` bench: one `sheaf_admm_step` call, K=100 vertices, d_v=8, d_e=5, T=5 diffusion steps. Target: < 5 µs (per Research 384 §6 estimate of ~500ns with SIMD). Run with `CARGO_TARGET_DIR=/tmp/sheaf_admm_bench` per AGENTS.md.
- [x] **T2.5** **G5 — zero-alloc test.** Custom allocator counter: 0 allocations in steady state (after warmup) per `sheaf_admm_step_into` call. The allocating `sheaf_admm_step` variant is allowed to allocate once for output sizing; the `_into` variant must be zero-alloc.
- [x] **T2.6** **G6 — determinism test.** Same input → same output bit-exactly across 100 runs, with and without `--release`. Required for any consumer that might commit `u_i` to chain.
- [x] **T2.7** Document the G1–G6 results in `katgpt-rs/.benchmarks/407_sheaf_admm_goat.md`. If all pass, promote `sheaf_admm` to default-on in `katgpt-dec`'s `default = [...]` list. Record the per-stack verdict in the Research 384 note.

## Phase 3 — Amplification (post-promotion)

### Tasks

- [ ] **T3.1** Conjugate-gradient z-update variant for ill-conditioned large zones (paper Appendix B.2). The shipped `hodge_laplacian` uses gradient descent; CG converges faster on sparse graphs with poor conditioning. Target: K=1000 vertices, condition number > 100. Bench GD-vs-CG; promote CG only if it wins on latency at fixed residual.
- [ ] **T3.2** Top-k sparse restriction maps for K>1000 (server scale). Currently `SheafMaps` materializes all edges; for very large zones, build a CSR-like sparse representation. Coordinate with riir-ai Plan 394 Phase 3 (Crowd MCGS integration).
- [ ] **T3.3** Soft-constraint variant (paper eq. 25): replace hard `Fz = 0` with quadratic penalty `γ/2 ‖Fz‖²`. Adds one knob `γ`; useful when exact consensus is undesirable (e.g., NPCs should preserve some individual variation). Bench hard-vs-soft on a synthetic "faction disagreement" scenario.
- [x] **T3.4** Example in `katgpt-rs/examples/sheaf_admm_consensus.rs`: 16 agents on a 4×4 grid, identity maps, show primal/consensus/dual converging over K=50 iterations. Print the dual `u_i` vectors to show they start at zero and grow with disagreement. Adoptable demo for the open-source funnel. **DONE (2026-07-07):** example ships with eta=0.25, T=50, K=50 → `max_edge_disagree = 1.22e-4 < 1e-3` (consensus ✅). Tuned from the spec's starting point (eta=0.2, T=20 → plateau at 2.79e-2) — the inexact z-projection with finite T=20 retains too much of the slowest non-harmonic mode (λ_min ≈ 0.152 on 4×4 grid); T=50 + eta=0.25 clears the bar in 50 iterations. Feature chain: root `sheaf_admm` → `katgpt-core/sheaf_admm` → `katgpt-dec/sheaf_admm` (default-on).

## GOAT gate summary

| Gate | Criterion | Target | Status |
|---|---|---|---|
| G1 | DEC identity (consensus reached) | `‖F x‖_∞ < 1e-5` after K=100 | ✅ `3.26e-8 < 1e-5` PASS |
| G2 | Dual conservation | `u^{k+1} - u^k == x^{k+1} - z^{k+1}` bit-exact | ✅ all 48 elements bit-identical PASS |
| G3 | Restriction maps compress | `‖F x‖ ≤ ‖x‖` for orthonormal rows | ✅ max ratio 0.898 ≤ 1.0 PASS |
| G4 | Latency (K=100, d_v=8, d_e=5, T=5) | < 5 µs | ✅ `1.808 µs < 5 µs` PASS |
| G5 | Zero-alloc steady state (`_into` variant) | 0 allocs | ✅ `0 allocs` PASS |
| G6 | Determinism across runs + release flags | bit-exact | ✅ 100/100 identical (debug + release) PASS |

**Promotion rule:** all 6 pass → `sheaf_admm` default-on in `katgpt-dec`. Demote (stay opt-in) if any fail.

## Cross-references

- Research 384 (this plan's parent note)
- riir-ai Research 314 (the private Super-GOAT guide consuming this primitive)
- riir-ai Plan 394 (the private runtime plan wiring this into HLA + Mind-Reading + Crowd MCGS)
- katgpt-rs Plan 251 (DEC operators — the `hodge_laplacian` substrate this builds on)
- katgpt-rs Plan 314 (Stokes calculus wrappers — vocabulary crosswalk)
