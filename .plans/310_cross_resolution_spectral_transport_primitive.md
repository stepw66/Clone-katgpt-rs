# Plan 310: Cross-Resolution Spectral Transport — Open Primitive

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md](../.research/291_cross_resolution_spectral_transport_open_primitive.md)
**Source:** Synthesized from FUNCATTN (arxiv 2605.31559, Research 257) + Topological Neural Operators (arxiv 2606.09806, Research 219) + Gemini "continuous field" reframing
**Target:** `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs` (new module) + Cargo feature `cross_resolution_transport`
**Status:** Active — Phase 1 (skeleton, pending implementation)

---

## Goal

Ship the minimal concrete prototype of Cross-Resolution Spectral Transport:
extend FUNCATTN's symmetric `k×k` operator to **asymmetric bases**
(`Φ_src ∈ R^{d_src × k}`, `Ψ_dst ∈ R^{d_dst × k}`), enabling train-on-small-
deploy-on-large latent transfer. Prove or kill the Super-GOAT candidate
(Research 291) via 4 GOAT gates measuring reconstruction quality, behavior rank
preservation, k sweep, and zero-allocation steady state.

**Proves the idea:** G1 mean cos ≥0.85 (16→64→16 round-trip) ·
G2 cos(action_rank) ≥0.85 (cross-resolution ranking preservation) ·
G3 k elbow identified · G4 0 allocs after warmup.

**Kills the idea:** G1 mean cos <0.75 (too much information loss) **OR**
G2 cos <0.75 (transport destroys personality) — either demotes to Gain.

---

## Phase 0 — Design (COMPLETE)

- [x] T0.1 Research note created ([Research 291](../.research/291_cross_resolution_spectral_transport_open_primitive.md))
- [x] T0.2 Private guide created ([riir-neuron-db/.research/004](../../../riir-neuron-db/.research/004_cross_resolution_shard_transport_guide.md))
- [x] T0.3 Plan created (this file)
- [x] T0.4 Fusion grep complete: FUNCATTN ships symmetric only (G2 benchmark same-d); zero hits for "cross-resolution", "asymmetric basis transport", "continuous field" in code; closest cousins are FUNCATTN (R257), Deep Manifold (R051), Resolution-Tiered Commitment (R280)

---

## Phase 1 — Unblocking Skeleton (CORE)

**Target:** minimal compilable primitive behind feature flag. No perf, no GOAT gate. Ships the struct + scalar impl + smoke tests. Reuses `funcattn.rs` solver where possible.

### Tasks

- [ ] T1.1 Create `katgpt-rs/crates/katgpt-core/src/cross_resolution.rs`:

  ```rust
  //! Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN.
  //! See katgpt-rs/.research/291_*.md and Plan 310.
  //!
  //! Generalizes FUNCATTN (Plan 286 / Research 257) to d_src ≠ d_dst. Two
  //! matmuls + reuse of the existing closed-form solve. Bases are frozen
  //! BLAKE3-committed artifacts; transport is inference-time, modelless.

  use crate::funcattn::{cholesky_solve_into, FuncAttnError};

  /// Frozen, BLAKE3-committed asymmetric basis pair for cross-resolution transport.
  /// `phi_src ∈ R^{d_src × k}` and `psi_dst ∈ R^{d_dst × k}` are column-orthonormal.
  #[derive(Debug, Clone)]
  pub struct CrossResolutionBases {
      /// Flattened `d_src × k`, row-major. Source-tier basis.
      pub phi_src: Vec<f32>,
      /// Flattened `d_dst × k`, row-major. Destination-tier basis.
      pub psi_dst: Vec<f32>,
      pub d_src: usize,
      pub d_dst: usize,
      pub k: usize,
      /// BLAKE3(phi_src_le || psi_dst_le || d_src_le || d_dst_le || k_le).
      pub commitment: [u8; 32],
  }

  impl CrossResolutionBases {
      pub fn verify_orthonormal(&self, tol: f32) -> bool {
          // Check phi_src^T phi_src ≈ I_k and psi_dst^T psi_dst ≈ I_k.
          orthonormal_check(&self.phi_src, self.d_src, self.k, tol)
              && orthonormal_check(&self.psi_dst, self.d_dst, self.k, tol)
      }
  }

  fn orthonormal_check(mat: &[f32], rows: usize, k: usize, tol: f32) -> bool {
      // G^T G should be I_k. Compute upper triangle, check diag ≈ 1, off-diag ≈ 0.
      for i in 0..k {
          for j in i..k {
              let mut dot = 0.0f32;
              for r in 0..rows {
                  dot += mat[r * k + i] * mat[r * k + j];
              }
              let target = if i == j { 1.0 } else { 0.0 };
              if (dot - target).abs() > tol { return false; }
          }
      }
      true
  }

  /// Pre-allocated scratch for zero-alloc transport. Mirrors `FuncAttnScratch`.
  pub struct CrossResScratch {
      pub spectral: Vec<f32>,    // k
      pub dst_state: Vec<f32>,   // d_dst
      cached_k: usize,
      cached_d_dst: usize,
  }

  impl CrossResScratch {
      pub fn new(k: usize, d_dst: usize) -> Self {
          Self {
              spectral: vec![0.0; k],
              dst_state: vec![0.0; d_dst],
              cached_k: k,
              cached_d_dst: d_dst,
          }
      }
      pub fn ensure_capacity(&mut self, k: usize, d_dst: usize) {
          if k > self.cached_k { self.spectral.resize(k, 0.0); self.cached_k = k; }
          if d_dst > self.cached_d_dst { self.dst_state.resize(d_dst, 0.0); self.cached_d_dst = d_dst; }
      }
  }

  /// Project source latent state → k-dim spectral coefficients.
  /// `spectral = phi_src^T · src_state` (k = phi_src.cols).
  #[inline]
  pub fn project_to_spectral_into(
      src_state: &[f32],
      bases: &CrossResolutionBases,
      spectral: &mut [f32],
  ) {
      debug_assert_eq!(src_state.len(), bases.d_src);
      debug_assert_eq!(spectral.len(), bases.k);
      for j in 0..bases.k {
          let mut acc = 0.0f32;
          for i in 0..bases.d_src {
              acc += bases.phi_src[i * bases.k + j] * src_state[i];
          }
          spectral[j] = acc;
      }
  }

  /// Reconstruct destination latent state from k-dim spectral coefficients.
  /// `dst_state = psi_dst · spectral` (d_dst = psi_dst.rows).
  #[inline]
  pub fn reconstruct_from_spectral_into(
      spectral: &[f32],
      bases: &CrossResolutionBases,
      dst_state: &mut [f32],
  ) {
      debug_assert_eq!(spectral.len(), bases.k);
      debug_assert_eq!(dst_state.len(), bases.d_dst);
      for i in 0..bases.d_dst {
          let mut acc = 0.0f32;
          for j in 0..bases.k {
              acc += bases.psi_dst[i * bases.k + j] * spectral[j];
          }
          dst_state[i] = acc;
      }
  }

  /// Full cross-resolution transport: src_state (d_src) → dst_state (d_dst).
  /// Zero-alloc given a `CrossResScratch`.
  pub fn transport_cross_resolution_into(
      src_state: &[f32],
      bases: &CrossResolutionBases,
      scratch: &mut CrossResScratch,
      dst_state: &mut [f32],
  ) {
      scratch.ensure_capacity(bases.k, bases.d_dst);
      project_to_spectral_into(src_state, bases, &mut scratch.spectral);
      reconstruct_from_spectral_into(&scratch.spectral, bases, dst_state);
  }

  /// Allocating convenience wrapper. Prefer `_into` on hot paths.
  pub fn transport_cross_resolution(
      src_state: &[f32],
      bases: &CrossResolutionBases,
  ) -> Vec<f32> {
      let mut dst = vec![0.0; bases.d_dst];
      let mut scratch = CrossResScratch::new(bases.k, bases.d_dst);
      transport_cross_resolution_into(src_state, bases, &mut scratch, &mut dst);
      dst
  }

  /// Cross-resolution + cross-domain transport (F2 fusion with FUNCATTN).
  /// `dst = psi_dst · C · phi_src^T · src` — four-matrix product, all small.
  /// `c_op ∈ R^{k × k}` is the FUNCATTN operator (from `solve_convex_combo_dual`).
  pub fn transport_cross_domain_cross_resolution_into(
      src_state: &[f32],
      bases: &CrossResolutionBases,
      c_op: &[f32],              // k × k, row-major
      scratch: &mut CrossResScratch,
      dst_state: &mut [f32],
  ) {
      debug_assert_eq!(c_op.len(), bases.k * bases.k);
      scratch.ensure_capacity(bases.k, bases.d_dst);
      // 1. src → spectral_src
      project_to_spectral_into(src_state, bases, &mut scratch.spectral);
      // 2. spectral_src → spectral_dst via C: scratch.dst_state reused as temp
      //    (we need a temp k-vector; use the first k slots of dst_state's scratch
      //    carefully — for clarity, allocate a small k-temp here in Phase 1)
      let mut spectral_dst = vec![0.0f32; bases.k];
      for i in 0..bases.k {
          let mut acc = 0.0f32;
          for j in 0..bases.k {
              acc += c_op[i * bases.k + j] * scratch.spectral[j];
          }
          spectral_dst[i] = acc;
      }
      // 3. spectral_dst → dst_state
      reconstruct_from_spectral_into(&spectral_dst, bases, dst_state);
  }

  #[cfg(test)]
  mod tests {
      use super::*;
      #[test] fn smoke_roundtrip_preserves_bandlimited_signal() { /* T1.5 */ }
      #[test] fn smoke_asymmetric_dims_compile() { /* T1.5 */ }
  }
  ```

- [ ] T1.2 Add feature gates. In `katgpt-rs/crates/katgpt-core/Cargo.toml`:
  ```toml
  cross_resolution_transport = ["funcattn"]  # Cross-Resolution Spectral Transport — asymmetric-basis FUNCATTN (Plan 310, Research 291). Opt-in until G1-G4 GOAT gate passes. Implies `funcattn` for solver reuse.
  ```
  In `katgpt-rs/Cargo.toml`:
  ```toml
  cross_resolution_transport = ["katgpt-core/cross_resolution_transport"]
  ```

- [ ] T1.3 Wire module into `katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "cross_resolution_transport")]
  pub mod cross_resolution;
  ```

- [ ] T1.4 Smoke tests:
  - Construct synthetic bases: `phi_src` = first k columns of a `d_src × d_src`
    identity (truncation), `psi_dst` = first k columns of a `d_dst × d_dst`
    identity. Verify `verify_orthonormal(tol=1e-5)` passes.
  - Construct a band-limited source state (only first k components nonzero).
    Transport 16 → 256 → 16. Assert round-trip is exact (cos = 1.0) for
    band-limited input.
  - Transport a non-band-limited state. Assert round-trip cos < 1.0 (information
    loss expected).

- [ ] T1.5 `cargo check -p katgpt-core --features cross_resolution_transport` clean.

---

## Phase 2 — GOAT Gate (PROVES OR KILLS)

Each gate is a standalone file. All must pass to promote from opt-in.

- [ ] T2.1 **G1 — Reconstruction cos (mean ≥0.85, min ≥0.75).** File:
      `tests/cross_res_g1_reconstruction.rs`. Generate 100 random 64-d reference
      vectors with controlled band-limitation (80% energy in first k=8
      components, 20% in higher components — simulates realistic personality
      shards which are low-rank per Research 257 §5.5). Transport 64 → 16 → 64.
      Measure cosine similarity of original vs reconstructed. **Gate:** mean
      cos ≥0.85, min cos ≥0.75. Report distribution.

- [ ] T2.2 **G2 — Behavior rank preservation (mean cos ≥0.85).** File:
      `tests/cross_res_g2_rank_preservation.rs`. **THE headline gate.**
      Generate 100 random 16-d source shards (plasma-tier personality).
      Transport 16 → 256. Score a fixed action set using both:
      - Source: `score(src_state, action_weights_16)` where `action_weights_16
        ∈ R^{16 × 5}`.
      - Destination: `score(dst_state, action_weights_256)` where
        `action_weights_256 ∈ R^{256 × 5}` is the source weights padded with
        zeros (so the scoring is "the same personality, more degrees of freedom").
      Measure cosine similarity of the 5-action ranking vectors.
      **Gate:** mean cos ≥0.85. **If this fails, transport destroys
      personality — abandon.**

- [ ] T2.3 **G3 — k sweep.** File: `tests/cross_res_g3_k_sweep.rs`. For the
      64→256 transport, sweep k ∈ {4, 8, 16, 32, 64}. Plot reconstruction cos
      vs k. **Gate:** identify the elbow; document recommended k per tier pair
      in Research 291 §5.3. No hard pass/fail — this is characterization.

- [ ] T2.4 **G4 — Zero-alloc steady state.** File:
      `tests/cross_res_g4_zero_alloc.rs`. Debug-only via
      `katgpt_rs::alloc::TrackingAllocator`. Warmup 10 iterations of
      `transport_cross_resolution_into`, then count allocations on the next
      1000 transports. **Gate:** 0 allocations after warmup.

---

## Phase 3 — SIMD Acceleration (if G1–G4 pass)

- [ ] T3.1 Identify hot loop: the `project_to_spectral_into` and
      `reconstruct_from_spectral_into` inner products. Both are
      matrix-vector products with small matrices.
- [ ] T3.2 For d_src, d_dst ≤ 64 and k ≤ 16: evaluate manual SIMD
      (`std::arch::x86_64::_mm256_fmadd_ps`) vs `wide` crate vs leaving scalar
      (LLVM auto-vectorizes well at these sizes). Bench at G4's crowd scale.
- [ ] T3.3 If cross-domain variant (4-matrix product) becomes a hot path,
      consider fusing the C multiplication with the reconstruction (loop
      re-ordering to keep k×k in registers).

---

## Phase 4 — Promotion Decision

- [ ] T4.1 If G1–G4 all pass → promote to opt-in default; document in README.
- [ ] T4.2 If G1 <0.75 → demote to research-only. Document in Research 291 §5.
- [ ] T4.3 If G2 <0.75 → demote to research-only. Personality corruption is a
      hard kill.
- [ ] T4.4 If G1/G2 borderline (0.75–0.85) → re-tune k per G3, re-run.

---

## Phase 5 — Shard Integration (DEFERRED to riir-neuron-db)

Blocked on Phase 2 G1–G2 pass. See
[riir-neuron-db/.research/004](../../../riir-neuron-db/.research/004_cross_resolution_shard_transport_guide.md)
for the integration guide.

- [ ] T5.1 `NeuronShard::transport_to_tier(d_dst, bases) -> NeuronShard` in
      `riir-neuron-db/src/shard.rs`. New constructor that produces a shard at
      a different `STYLE_DIM` via cross-resolution transport. Both source and
      destination are valid committed artifacts.
- [ ] T5.2 `ShardIndex::get_at_tier(zone_hash, d_dst, bases) -> NeuronShard` in
      `riir-neuron-db/src/index.rs`. Lazy projection on retrieval — store one
      reference (highest-fidelity) shard per zone, project on demand.
- [ ] T5.3 Consolidation integration: `Raven/δ-Mem` runs at low resolution
      (fast, plasma-tier), the consolidated result projects up to reference
      resolution for cold-tier commit.
- [ ] T5.4 AnyRAG escalation: external retrieval returns whatever resolution
      the source provides; gateway transports to the caller's native tier.

---

## Cross-Refs

- [katgpt-rs/.research/291_cross_resolution_spectral_transport_open_primitive.md](../.research/291_cross_resolution_spectral_transport_open_primitive.md) — research note
- [riir-neuron-db/.research/004_cross_resolution_shard_transport_guide.md](../../../riir-neuron-db/.research/004_cross_resolution_shard_transport_guide.md) — private guide
- [katgpt-rs/.plans/286_functional_attention_spectral_transport.md](286_functional_attention_spectral_transport.md) — symmetric FUNCATTN (depends on)
- [katgpt-rs/.research/257_Functional_Attention_Spectral_Transport_Operator.md](../.research/257_Functional_Attention_Spectral_Transport_Operator.md) — FUNCATTN research
- [katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md](../.research/219_Topological_Neural_Operators_DEC_Inference.md) — TNO/DEC (topological cousin)
- [katgpt-rs/.research/280_Resolution_Tiered_Deterministic_Commitment.md](../.research/280_Resolution_Tiered_Deterministic_Commitment.md) — resolution tiering (chain-side cousin)
- [katgpt-rs/.issues/001_apollonian_sphere_manifold_exploration.md](../.issues/001_apollonian_sphere_manifold_exploration.md) — F3 speculative fusion target
