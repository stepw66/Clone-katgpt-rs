# Plan 416: Region-Conditioned Subspace Field — MFA Local-Geometry Steering Primitive

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/396_MFA_Region_Conditioned_Factor_Analyzer.md](../.research/396_MFA_Region_Conditioned_Factor_Analyzer.md)
**Source paper:** [arXiv:2602.02464](https://arxiv.org/abs/2602.02464) — Shafran et al., "From Directions to Regions: Decomposing Activations in Language Models via Local Geometry"
**Target:** `katgpt-rs/crates/katgpt-core/src/region_subspace.rs` (new module) + Cargo feature `region_subspace_steering`
**Status:** Active — Phase 1 (unblocking skeleton)

---

## Goal

Ship the **region-conditioned generalization of `SubspaceSteeringField` (Plan 412)**. Plan 412 carries a single k-dim orthonormal block that applies globally. This plan generalizes it to **K regions, each with its own centroid μ_k AND its own local R-dim subspace (factor-analyzer loadings W_k)** — the MFA structure from arXiv:2602.02464.

The primitive is the **modelless consumer** of a frozen MFA-like artifact `{μ_k, W_k, Ψ, π}`. The artifact is either (a) trained offline via GD on negative log-likelihood (riir-train territory) or (b) **deterministically constructed** via K-means + per-region PCA (modelless baseline, no GD). Once frozen, all consumption is closed-form linear algebra: per-region sigmoid membership gates, posterior-mean local coordinates, and two-mode steering (centroid interpolation + local subspace offset).

At the degenerate limit (`K=1, μ_1=0, W_1=I_R`), the local-coordinate steering reduces to Plan 412's `apply_subspace_steering` — making this a strict superset. At `K≥2`, it enables **two-mode local-geometry steering**: move toward a region (centroid interpolation) OR walk within the current region (local subspace offset), with per-region sigmoid membership gates selecting which regions are active.

**GOAT gate:** G1 (degenerate `K=1` parity with Plan 412), G2 (`K≥2` two-mode steering produces distinct region/local effects), G3 (zero-alloc), G4 (latency), G5 (BLAKE3 commitment determinism).

## Why this is a GOAT, not a Super-GOAT

Research 396 §5 settles this: the within-region subspace case ships (Plan 412), the region-centroid case ships (Plan 409 / R389 CHaRS), the per-entity blend ships (Plan 321 / R302 FAME). The region-conditioned factor-analyzer (K regions × per-region centroid × per-region subspace × per-input routing) is genuinely unshipped — Q1 YES. But the operation class ("blend K region-conditioned subspace offsets by membership gates") is a refinement + unification of Plan 412 and Plan 409, not a new mechanism class — Q2 PARTIAL. Consistent with R389 (GOAT) and R393 (GOAT) precedent. A Super-GOAT fusion candidate (Region-Structured HLA) extends Issue 049.

## Design

### Types

```rust
/// A region-conditioned factor-analyzer field: K regions, each with a centroid
/// μ_k and a local R-dim subspace (loadings W_k). BLAKE3-committed.
///
/// Region-conditioned generalization of Plan 412's `SubspaceSteeringField`.
/// - Plan 412 = single block, no regions (the `K=1, μ=0, W=I` degenerate limit).
/// - Plan 409 (CHaRS) = regions + centroids, but translation vectors not subspaces.
/// - This primitive = regions + centroids + per-region subspaces (the MFA structure).
///
/// The artifact is TRAINED OFFLINE (riir-train: GD on negative log-likelihood,
/// or the modelless K-means + per-region PCA constructor). Once frozen, all
/// consumption is closed-form linear algebra — no gradients at inference.
pub struct RegionSubspaceField<const D: usize, const K: usize, const R: usize> {
    /// Region centroids `μ_k ∈ R^D`. K rows. Absolute positions in activation space.
    pub centroids: [[f32; D]; K],
    /// Per-region factor-analyzer loadings `W_k ∈ R^{R×D}`.
    /// `loadings[k][r]` is the r-th local axis (D-dim unit vector) for region k.
    /// Each region has R local axes. Stored row-major: `loadings[k]` = `[[f32; D]; R]`.
    pub loadings: [[[f32; D]; R]; K],
    /// Per-region mixture log-weights `log π_k` (pre-computed at construction).
    pub log_pi: [f32; K],
    /// Diagonal noise precision (inverse variance) per dimension, `Ψ^{-1}`.
    pub psi_inv: [f32; D],
    /// Pre-computed posterior-mean projector `Z_k ∈ R^{R×D}` per region.
    /// `Z_k = (I_R + W_k^T Ψ^{-1} W_k)^{-1} W_k^T Ψ^{-1}` (eq. 10, closed-form).
    /// Computed once at construction; frozen for the field's lifetime.
    pub projectors: [[[f32; D]; R]; K],
    /// `BLAKE3(centroids || loadings || log_pi || psi_inv)` — content commitment.
    pub commitment: [u8; 32],
}
```

### Core operations (all closed-form, zero-alloc)

1. **`membership_gates(state, field, tau) -> [f32; K]`** — per-region sigmoid membership gates (reformulated from the paper's softmax responsibilities to sigmoid per AGENTS.md mandate):
   ```
   a_k(x) = log_pi[k] − 0.5 · ||x − μ_k||²_{Ψ^{-1}}  − 0.5 · tr_log_term_k
   g_k(x) = sigmoid(a_k(x) − τ)
   ```
   where `||x − μ_k||²_{Ψ^{-1}} = Σ_d psi_inv[d]·(x[d] − μ_k[d])²`. Per-region independent gates ∈ (0,1) — an NPC can be partially in multiple regions simultaneously (more expressive than winner-take-all softmax). Zero-alloc (output is a fixed `[f32; K]` stack array).

2. **`local_coordinates(state, field, k) -> [f32; R]`** — posterior-mean latent vector within region k (eq. 9-10):
   ```
   ẑ_k = Z_k · (x − μ_k)    // R-dim output, closed-form matrix-vector
   ```
   Zero-alloc (output is a fixed `[f32; R]` stack array).

3. **`steer_centroid(state, field, k, alpha)`** — centroid interpolation toward region k (eq. 14):
   ```
   x' = (1 − α)·x + α·μ_k
   ```
   In-place SAXPY over D dims. α ∈ [0, 1]. At α=0 identity, α=1 full region replacement. Zero-alloc.

4. **`steer_local(state, field, k, offset: &[f32; R])`** — local subspace offset within region k (eq. 15):
   ```
   x' = x + W_k · v    // v ∈ R^R, additive offset
   ```
   In-place matrix-vector add over D dims. Region-conditioned: `W_k` selected by region index. Zero-alloc. At `K=1, μ_1=0, W_1=I_R` this reduces to Plan 412's `apply_subspace_steering`.

5. **`decompose(state, field, tau) -> RegionDecomposition`** — full decomposition:
   ```rust
   pub struct RegionDecomposition<const K: usize, const R: usize> {
       pub gates: [f32; K],           // membership_gates output
       pub local_coords: [[f32; R]; K], // local_coordinates per region
   }
   ```
   Combines operations 1 + 2 for all K regions. Zero-alloc after the stack struct.

6. **`reconstruct(decomposition, field) -> [f32; D]`** — reconstruction from decomposition (eq. 11):
   ```
   x̂ = Σ_k g_k(x) · [μ_k + W_k · ẑ_k(x)] / Σ_k g_k(x)
   ```
   Normalized by `Σ_k g_k` (sigmoid gates don't sum to 1, unlike softmax). Zero-alloc.

### Constructor (modelless K-means + per-region PCA baseline)

7. **`RegionSubspaceField::from_corpus_kmeans_pca(corpus, k, r, psi_inv) -> Self`** — deterministic modelless constructor:
   - K-means on the corpus → K centroids `μ_k`.
   - Per-region PCA (closed-form eigendecomposition of the region-conditional covariance) → top-R eigenvectors = loadings `W_k`.
   - Mixture weights `π_k` = fraction of corpus assigned to region k.
   - Pre-compute projectors `Z_k` via eq. 10.
   - BLAKE3 commitment.

   This is the **modelless baseline** — no GD, no riir-train. The GD-trained version (riir-train) will have better likelihood but the same consumption interface. The GOAT gate benchmarks this constructor's reconstruction error against the paper's Table 4 to set expectations.

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/region_subspace.rs` with module docstring (cite Research 396 + Plan 416 + arXiv:2602.02464 + the `K=1` parity contract with Plan 412).
- [ ] **T1.2** Define `RegionSubspaceField<const D: usize, const K: usize, const R: usize>` struct (centroids, loadings, log_pi, psi_inv, projectors, commitment) + `RegionSubspaceError` enum (`NotOrthonormal` for loadings, `DimensionMismatch`, `InvalidProbability`).
- [ ] **T1.3** Implement `RegionSubspaceField::new(centroids, loadings, log_pi, psi_inv, tol)` — validates loadings orthonormality (per region), computes projectors `Z_k` via eq. 10 (closed-form `(I + W^T Ψ^{-1} W)^{-1} W^T Ψ^{-1}`), computes BLAKE3 commitment.
- [ ] **T1.4** Implement `membership_gates(state: &[f32; D], field: &Self, tau: f32) -> [f32; K]` — per-region sigmoid gates. SIMD over D for the Mahalanobis distance. Zero-alloc.
- [ ] **T1.5** Implement `local_coordinates(state: &[f32; D], field: &Self, k: usize) -> [f32; R]` — `Z_k · (x − μ_k)`. SIMD matrix-vector. Zero-alloc.
- [ ] **T1.6** Implement `steer_centroid(state: &mut [f32; D], field: &Self, k: usize, alpha: f32)` — in-place SAXPY `(1−α)x + αμ_k`. Zero-alloc.
- [ ] **T1.7** Implement `steer_local(state: &mut [f32; D], field: &Self, k: usize, offset: &[f32; R])` — in-place `x += W_k · offset`. SIMD matrix-vector add. Zero-alloc. Document the `K=1, μ=0, W=I` → Plan 412 reduction.
- [ ] **T1.8** Add `pub mod region_subspace;` + re-exports to `katgpt-core/src/lib.rs`, gated `#[cfg(feature = "region_subspace_steering")]`.
- [ ] **T1.9** Add `region_subspace_steering = ["subspace_steering"]` feature to `katgpt-core/Cargo.toml` (implies `subspace_steering` so the Plan 412 parity reference resolves).
- [ ] **T1.10** G1 unit test: `k1_degenerate_parity_with_plan_412` — construct a `RegionSubspaceField<D, 1, R>` with `μ_1=0, W_1=I_R, log_pi=[0], psi_inv=[1;D]`, apply `steer_local` with an offset, assert bit-identical output to `SubspaceSteeringField::apply_subspace_steering` with the same offset as alphas. **This is the load-bearing gate** — proves the generalization subsumes Plan 412.

**Phase 1 validation:** unit tests pass; default features compile clean; `--all-features` compile clean; feature-off compile clean; zero-alloc by construction (all fields fixed-size arrays).

## Phase 2 — Decomposition + Reconstruction

### Tasks

- [ ] **T2.1** Implement `decompose(state, field, tau) -> RegionDecomposition<K, R>` — runs `membership_gates` + `local_coordinates` for all K regions. Zero-alloc (stack struct).
- [ ] **T2.2** Implement `reconstruct(decomposition, field) -> [f32; D]` — normalized weighted sum `Σ_k g_k·[μ_k + W_k·ẑ_k] / Σ_k g_k`. SIMD. Zero-alloc.
- [ ] **T2.3** G2 unit test: `roundtrip_reconstruction_quality` — construct a field from a synthetic clustered corpus, decompose a held-out point, reconstruct, assert reconstruction error < tolerance. Verifies the sigmoid-gate normalization produces reasonable reconstruction.
- [ ] **T2.4** G2 unit test: `k2_two_mode_distinct_effects` — with K=2 regions, verify `steer_centroid(k=0)` and `steer_centroid(k=1)` produce distinct outputs (different regions), and `steer_local(k=0, v)` vs `steer_local(k=1, v)` produce distinct outputs (different local subspaces).

## Phase 3 — Modelless Constructor (K-means + per-region PCA)

### Tasks

- [ ] **T3.1** Implement `RegionSubspaceField::from_corpus_kmeans_pca(corpus: &[&[f32; D]], k_target, r, psi_inv) -> Self` — deterministic modelless constructor. K-means (simple Lloyd's algorithm, fixed iterations) + per-region PCA (closed-form 2x2 or NxN eigendecomposition for the top-R eigenvectors). No GD.
- [ ] **T3.2** Unit test: `constructor_produces_valid_field` — feed a synthetic clustered corpus (e.g., 3 Gaussian blobs in D=8), construct a K=3 R=2 field, verify centroids ≈ blob centers, loadings span the blob's principal axes, commitment is deterministic.
- [ ] **T3.3** Unit test: `constructor_reconstruction_better_than_kmeans_only` — verify the MFA field (centroids + loadings) reconstructs held-out points better than centroids-only (K-means baseline). This validates that the local subspaces add information.

## Phase 4 — GOAT Gate Benchmark

### Tasks

- [ ] **T4.1** Create `katgpt-rs/crates/katgpt-core/tests/bench_416_region_subspace_goat.rs`.
- [ ] **T4.2** **G1 (parity, expanded)**: 100 random offset vectors, bit-identical `steer_local` (K=1, μ=0, W=I) vs Plan 412 `apply_subspace_steering` via `f32::to_bits()` equality.
- [ ] **T4.3** **G2 (two-mode steering)**: K=4 field; verify centroid steering moves state toward the correct region centroid; verify local steering produces region-specific offsets. Quantify: `||steer_centroid(k=0) − μ_0||` decreases with α; `||steer_local(k=0, v) − steer_local(k=1, v)||` > 0 for v ≠ 0.
- [ ] **T4.4** **G3 (zero-alloc)**: heap profiler confirms 0 allocations over 1000 calls to `membership_gates` + `local_coordinates` + `steer_centroid` + `steer_local` at `D=8, K=8, R=2` (HLA scale). Requires `--test-threads=1`.
- [ ] **T4.5** **G4 (latency)**: structural size proof (`size_of` = K·D·4 + K·R·D·4 + K·4 + D·4 + K·R·D·4 + 32 for centroids + loadings + log_pi + psi_inv + projectors + commitment) + 100k-apply latency smoke at K=8 D=8 R=2 (< budget). At D=8 K=8 R=2: membership_gates ~200 FLOPs, local_coords ~130 FLOPs, steer_local ~130 FLOPs — all plasma-tier (sub-µs).
- [ ] **T4.6** **G5 (determinism)**: `commitment_is_deterministic` (same parameters → same BLAKE3) + `decompose` + `reconstruct` bit-identical for fixed state + field.

**Phase 4 GOAT verdict target:** G1–G5 all PASS.

## Phase 5 — Promotion Decision

### Tasks

- [ ] **T5.1** If G1–G5 all PASS: promote `region_subspace_steering` to default-on in `katgpt-core`. **Do NOT demote Plan 412** — they coexist: Plan 412 is the single-block case (lower overhead for callers that don't need regions), Plan 416 is the region-conditioned case (for local-geometry steering). Per-stack ledger records both in the "steering" slot.
- [ ] **T5.2** If G2 FAILS (two-mode steering not distinct): keep opt-in, document the failure mode, do NOT promote.
- [ ] **T5.3** Update `katgpt-rs/README.md` Feature Showcase with a `### 🧩 Region-Conditioned Subspace Field — MFA Local-Geometry Steering (Plan 416, arxiv 2602.02464)` section.
- [ ] **T5.4** Commit on `develop`: `feat(steering): region-conditioned subspace field — MFA local-geometry primitive (Plan 416)`.

## Per-stack tracking (steering slot)

| Primitive | Dim | Regions | Mechanism | Status |
|-----------|-----|---------|-----------|--------|
| `LatentSteeringVector` (Plan 309) | 1D | — | `s + α·v` | DEFAULT-ON |
| `Phase-Modulated Coupling` (Plan 322) | 2D (single pair) | — | cos/sin rotation in `(a,b)` plane | DEFAULT-ON |
| `Spherical Steering` (Plan 405) | 1D target | — | Slerp toward target direction | DEFAULT-ON |
| `SubspaceSteeringField` (Plan 412) | k-dim | 1 (global) | `s + Σ α_j·u_j`, manifold walk | DEFAULT-ON |
| **`RegionSubspaceField` (Plan 416)** | **R-dim per region** | **K** | **region-conditioned centroid + local subspace, two-mode steering** | **opt-in (pending GOAT)** |

All five coexist — each occupies a distinct steering niche. Plan 416 does NOT demote any sibling; it adds the region-conditioned local-geometry capability.

## References

- **Research:** [katgpt-rs/.research/396_MFA_Region_Conditioned_Factor_Analyzer.md](../.research/396_MFA_Region_Conditioned_Factor_Analyzer.md)
- **Source paper:** [arXiv:2602.02464](https://arxiv.org/abs/2602.02464) — Shafran et al., "From Directions to Regions"
- **Within-region sibling:** `katgpt-rs/.plans/412_subspace_steering_field_primitive.md` + `katgpt-rs/crates/katgpt-core/src/subspace_steering.rs`
- **Cluster-aware steering cousin:** `katgpt-rs/.plans/409_jlens_concept_readout_prefilter_poc.md` (CHaRS routing) + Research 389
- **Per-entity MoE cousin:** `katgpt-rs/.plans/321_sampling_invariant_per_entity_moe_primitive.md` + Research 302
- **1D steering baseline:** `katgpt-rs/.plans/309_latent_field_steering_primitive.md`
- **Super-GOAT fusion tracker:** `katgpt-rs/.issues/049_block_sparse_hla_supergoat_validation.md` (extended by Research 396 — Region-Structured HLA candidate now has the MFA construction recipe)
- **MFA origin:** Ghahramani & Hinton (1996), "The EM Algorithm for Mixtures of Factor Analyzers"
