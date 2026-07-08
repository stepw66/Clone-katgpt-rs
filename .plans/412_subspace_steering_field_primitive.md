# Plan 412: Subspace Steering Field — k-dim Manifold Steering Primitive

**Date:** 2026-07-08
**Research:** [katgpt-rs/.research/393_Block_Sparse_Featurizer_Subspace_Concept_Primitive.md](../.research/393_Block_Sparse_Featurizer_Subspace_Concept_Primitive.md)
**Source paper:** [arXiv:2606.25234](https://arxiv.org/abs/2606.25234) — Goodfire, Block-Sparse Featurizers
**Target:** `katgpt-rs/crates/katgpt-core/src/subspace_steering.rs` (new module) + Cargo feature `subspace_steering`
**Status:** Active — Phase 1 (unblocking skeleton)

---

## Goal

Ship the **k-dim generalization of `LatentSteeringVector` (Plan 309)**. The existing primitive is strictly 1D (`direction: Vec<f32>` + scalar `α`, math `s' = s + α·v`). This plan generalizes it to a k-dim orthonormal block `{u_1..u_k}` + per-axis strengths `{α_1..α_k}`, with math `s' = s + Σ_j α_j · u_j`. At `K=1` it is bit-identical to Plan 309; at `K≥2` it enables **manifold walking** — sweeping `alphas` over a grid to generate concept variations (the Goodfire "pretzel manifold" pattern, adapted to our latent-state substrate).

The block basis comes from **pre-discovered** sources (Plan 301 Jacobian SVD, SpectralQuant offline eigenbasis, or hand-constructed orthogonal sets) — no training at inference. The primitive is the *consumer* of discovered blocks, not the featurizer trainer (that's riir-train).

**GOAT gate:** G1 (`K=1` parity with Plan 309), G2 (behavior rank preservation over walked region), G3 (zero-alloc), G4 (latency), G5 (BLAKE3 commitment determinism).

## Why this is a GOAT, not a Super-GOAT

Research 393 §3 settles this: the 1D case fully ships (Plan 309/162/297/320), the stable-rank diagnostic ships (Plan 287), so Q1 (no prior art) is only PARTIAL — the *k-dim subspace* case is missing but the *1D* and *diagnostic* cases are covered. Q2 (new class) is PARTIAL — manifold walking is a generalization, not a new mechanism class. Not all-4-YES → GOAT. A Super-GOAT fusion candidate (Block-Sparse HLA) is tracked separately in Issue 049.

## Design

### Types

```rust
/// A k-dim orthonormal block + per-axis strengths, BLAKE3-committed.
///
/// Generalization of `LatentSteeringVector` (Plan 309) from 1D to k-dim.
/// At `K=1` this is bit-identical to Plan 309 (single direction + scalar α).
/// At `K≥2` it enables manifold walking — sweep `alphas` over a grid to
/// generate concept variations within the subspace.
///
/// The block basis `{u_1..u_k}` is PRE-DISCOVERED (Plan 301 Jacobian SVD,
/// SpectralQuant offline eigenbasis, or hand-constructed orthogonal set).
/// No training at inference. The primitive is the *consumer* of discovered
/// blocks, not the featurizer trainer.
pub struct SubspaceSteeringField<const D: usize, const K: usize> {
    /// Orthonormal block basis, row-major `[K][D]`. Each `basis[j]` is unit-norm
    /// and orthogonal to `basis[i]` for `i != j`. Constructed via
    /// `newton_schulz_orthogonalize` (Plan 152) at freeze time.
    pub block: [[f32; D]; K],
    /// Per-axis strengths `α_j ∈ [0, 1]`, sigmoid-bounded at construction.
    pub alphas: [f32; K],
    /// `BLAKE3(block_le || alphas_le)` — content-addressed commitment.
    pub commitment: [u8; 32],
}
```

### Core operations

1. **`apply_subspace_steering(state: &mut [f32], field: &SubspaceSteeringField)`** — SIMD SAXPY: `state[j] += Σ_k alphas[k] * block[k][j]` over `K·D` elements. Zero-alloc. At `K=1` this reduces to Plan 309's `apply_latent_steering`.
2. **`walk_manifold(state, field, alpha_grid, out_grid)`** — sweep `alphas` over a pre-allocated grid (e.g., 2D grid for `K=2`), writing the steered state at each grid point into `out_grid`. Zero-alloc after grid allocation. This is the "pretzel manifold" pattern: each grid point is one concept variation.
3. **`block_energy(block, state) -> [f32; K]`** — per-axis projection energy `dot(block[k], state)`. Used for block-wise TopK consumption (which blocks are active).
4. **`compute_block_commitment(block, alphas) -> [u8; 32]`** — BLAKE3 of the flattened block + alphas (little-endian). Deterministic, quorum-verifiable.

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/subspace_steering.rs` with module docstring (mirror `latent_steering.rs` doc style; cite Research 393 + Plan 412 + the `K=1` parity contract). **DONE**
- [x] **T1.2** Define `SubspaceSteeringField<const D: usize, const K: usize>` struct (block, alphas, commitment) + `SubspaceSteeringError` enum (`NotOrthonormal`, `AlphaOutOfRange`, `DimensionMismatch`). **DONE** — `DimensionMismatch` dropped: with const generics `D` and `K` are compile-time fixed, so dimension mismatch is impossible by construction (the array types enforce it). `NotOrthonormal` covers both non-unit-norm AND non-orthogonal-pair cases.
- [x] **T1.3** Implement `SubspaceSteeringField::new(block, alphas, orthonormal_tol)` — validates orthonormality (each `basis[j]` unit-norm within tol, pairwise dot < tol) and `alpha ∈ [0,1]`, computes BLAKE3 commitment. **DONE**
- [x] **T1.4** Implement `apply_subspace_steering(state: &mut [f32], field: &SubspaceSteeringField)` — chunked SIMD SAXPY over `K·D`. Zero-alloc. Document the `K=1` → Plan 309 reduction. **DONE** — signature adapted to const generics: `state: &mut [f32; D]` (fixed array, zero-alloc by construction). Method form `field.apply(&mut state)` + free-function `apply_subspace_steering` wrapper. Inner loop over D is the SAXPY; outer loop over K accumulates. No cross-lane reduction → bit-identical to scalar regardless of vectorization.
- [x] **T1.5** Implement `compute_block_commitment(block, alphas) -> [u8; 32]` (BLAKE3, little-endian flatten). **DONE**
- [x] **T1.6** Add `pub mod subspace_steering;` + re-exports to `katgpt-core/src/lib.rs`, gated `#[cfg(feature = "subspace_steering")]`. **DONE**
- [x] **T1.7** Add `subspace_steering = ["latent_field_steering"]` feature to `katgpt-core/Cargo.toml`. **DONE** — implies `latent_field_steering` so the K=1 parity gate reference resolves.
- [x] **T1.8** G1 unit test: `k1_parity_with_plan_309` — construct a `SubspaceSteeringField<D, 1>` from a Plan 309 `LatentSteeringVector`'s direction + alpha, apply to a test state, assert bit-identical output to `apply_latent_steering`. **This is the load-bearing gate** — proves the generalization subsumes the 1D case. **DONE** — bit-identical via `f32::to_bits()` equality on all D=8 elements.

**Phase 1 validation (2026-07-08):** 10/10 unit tests pass (including `k1_parity_with_plan_309`). Default features compile clean. `--all-features` compile clean. Feature-off compile clean. Zero alloc by construction (all fields fixed-size arrays).

## Phase 2 — Manifold Walking + Block Energy ✅ DONE (2026-07-08)

### Tasks

- [x] **T2.1** Implement `block_energy(block, state) -> [f32; K]` — per-axis dot-product projection. SIMD. **DONE** — zero-alloc (output is a fixed `[f32; K]` stack array). Read-side counterpart of `apply_subspace_steering`: apply writes energy INTO state, this reads energy ALREADY PRESENT along each axis.
- [x] **T2.2** Implement `walk_manifold(state, field, alpha_grid: &[[f32; K]], out_grid: &mut [[f32; D]])` — for each row in `alpha_grid`, compute `state + Σ_k alpha_grid[i][k] * block[k]` into `out_grid[i]`. Zero-alloc after grid allocation. **DONE** — signature adapted to const generics (`state: &[f32; D]`, `block: &[[f32; D]; K]`). Caller owns both grids. This is the "pretzel manifold" pattern.
- [x] **T2.3** G2 unit test: `k2_walk_preserves_norm_bounds` — walk a 2D grid over a `K=2` field, assert each output state has L2 norm within `[‖state‖ − ε, ‖state‖ + Σ_k |α_k|]` (norm inflation bounded by the steering magnitude, per Plan 322's norm-preservation analysis). **DONE** — 5×5 grid over `[-0.5, 0.5]²`, triangle-inequality bound holds for all 25 outputs.
- [x] **T2.4** G2 unit test: `k2_walk_covers_grid` — verify the walked grid produces `grid_rows` distinct output states (no duplicates unless alphas repeat). **DONE** — 4 distinct alpha pairs → 4 distinct outputs (block rows linearly independent); repeated alphas → identical outputs (determinism).

**Phase 2 validation (2026-07-08):** 5/5 new tests pass. Total 15/15 (10 Phase 1 + 5 Phase 2). Default + `--all-features` + feature-off all compile clean.

## Phase 3 — Orthonormality Construction Helper ✅ DONE (2026-07-08)

### Tasks

- [x] **T3.1** Implement `SubspaceSteeringField::from_directions_orthonormalize(directions, alphas, tol)` — **DONE with algorithm change: Gram-Schmidt, NOT Newton-Schulz.** The plan specified Newton-Schulz (Plan 152), but empirical testing (2026-07-08) found NS **diverges** on non-square K<D matrices: the Muon-tuned coefficients `(3.4445, -4.7750, 2.0315)` are designed for square weight-gradient matrices, and on a K=2 D=8 input the iteration produced dot products that GREW with more iterations (5 iters: dot=0.10; 10 iters: dot=-0.30; 15 iters: dot=0.37). Gram-Schmidt is exact, stable, zero-alloc (in-place on the stack `[[f32; D]; K]` block), and the standard algorithm for K<D orthonormalization — it's the right tool here. The `newton_schulz` feature dependency was removed from `subspace_steering`.
- [x] **T3.2** Unit test: `from_directions_orthonormalize_produces_orthonormal_block` — feed 3 standard-basis directions, verify output block is orthonormal within tol. **DONE** — plus 2 additional tests (`cleans_up_drift` with far-from-orthogonal input, `rejects_bad_alphas`).

**Phase 3 validation (2026-07-08):** 3/3 new tests pass. Total 18/18 (10 Phase 1 + 5 Phase 2 + 3 Phase 3). Default + `--all-features` + feature-off all compile clean. Zero-alloc construction (Gram-Schmidt is in-place stack-only; NS would have needed a K*D flatten `Vec`).

## Phase 4 — GOAT Gate Benchmark

### Tasks

- [ ] **T4.1** Create `katgpt-rs/crates/katgpt-core/benches/subspace_steering_bench.rs`.
- [ ] **T4.2** **G3 (zero-alloc)**: `apply_subspace_steering_zero_alloc_after_warmup` — heap profiler confirms 0 allocations over 1000 calls at `D=8, K=2` (HLA scale).
- [ ] **T4.3** **G4 (latency)**: benchmark `apply_subspace_steering` at `D=8, K={1,2,4}`. Target: `K=1` matches Plan 309's latency (sub-100ns); `K=2` < 200ns; `K=4` < 400ns. Linear scaling in `K·D`.
- [ ] **T4.4** **G5 (determinism)**: `commitment_is_deterministic` — same block + alphas → same BLAKE3 across runs. And `walk_manifold` is bit-identical for fixed alpha_grid (quorum-safe).
- [ ] **T4.5** **G1 (parity, expanded)**: extend T1.8 — run `K=1` parity over 100 random direction+alpha pairs, assert bit-identical to Plan 309 on all.

## Phase 5 — Promotion Decision

### Tasks

- [ ] **T5.1** If G1–G5 all PASS: promote `subspace_steering` to default-on in `katgpt-core`. **Do NOT demote Plan 309** — they coexist: Plan 309 is the simple 1D case (lower overhead for callers that only need 1D), Plan 412 is the k-dim case (for manifold walking). The per-stack ledger records both in the "steering" slot.
- [ ] **T5.2** If G2 FAILS (norm bounds violated): keep opt-in, document the failure mode, do NOT promote.
- [ ] **T5.3** Update `katgpt-rs/README.md` Feature Showcase with a `### 🔷 Subspace Steering Field — k-dim Manifold Steering (Plan 412, arxiv 2606.25234)` section.
- [ ] **T5.4** Commit on `develop`: `feat(steering): subspace steering field — k-dim manifold walking primitive (Plan 412)`.

## Per-stack tracking (steering slot)

| Primitive | Dim | Mechanism | Status |
|-----------|-----|-----------|--------|
| `LatentSteeringVector` (Plan 309) | 1D | `s + α·v` | DEFAULT-ON |
| `Phase-Modulated Coupling` (Plan 322) | 2D (single pair) | cos/sin rotation in `(a,b)` plane | DEFAULT-ON |
| `Spherical Steering` (Plan 405) | 1D target | Slerp toward target direction | DEFAULT-ON |
| **`SubspaceSteeringField` (Plan 412)** | **k-dim** | **`s + Σ α_j·u_j`, manifold walk** | **opt-in (this plan)** |

All four coexist — each occupies a distinct steering niche. Plan 412 does NOT demote any sibling; it adds the k-dim manifold-walk capability.

## References

- **Research**: [katgpt-rs/.research/393_Block_Sparse_Featurizer_Subspace_Concept_Primitive.md](../.research/393_Block_Sparse_Featurizer_Subspace_Concept_Primitive.md)
- **1D sibling**: `katgpt-rs/.plans/309_latent_field_steering_primitive.md` + `katgpt-rs/crates/katgpt-core/src/latent_steering.rs`
- **Basis discovery**: `katgpt-rs/.plans/301_runtime_subspace_phase_gate_primitive.md` + `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs`
- **Orthogonalization**: `katgpt-rs/.plans/152_newton_schulz_river_valley_diagnostics.md` (Newton-Schulz, shipped)
- **Norm-preservation analysis**: `katgpt-rs/.benchmarks/322_phase_rotation_goat.md`
- **Super-GOAT fusion tracker**: `katgpt-rs/.issues/049_block_sparse_hla_supergoat_validation.md`
- **Source paper**: [arXiv:2606.25234](https://arxiv.org/abs/2606.25234) — Goodfire BSF
