# Plan 423: Spectral Rewiring — Weight Delta Purification via Base SVD Projection

**Date:** 2026-07-10
**Research:** [406_Spectral_Rewiring_Weight_Delta_Purification.md](../.research/406_Spectral_Rewiring_Weight_Delta_Purification.md)
**Source paper:** [arXiv:2607.03065](https://arxiv.org/abs/2607.03065) — Zhang et al., *Spectral Rewiring for Exploration, Purification, and Model Merging*, Tsinghua AIR / ByteDance Seed, Jul 2026
**Target:** `katgpt-rs/crates/katgpt-spectral/src/spectral_rewire.rs` (new module) + Cargo feature `spectral_rewire`
**Status:** 🚧 Phase 1 ✅ COMPLETE (T1.1–T1.7, 7/7 tests pass, `9df7cdbd`+impl). Phase 2–4 pending.
**Verdict from research:** GOAT (Q1✓ Q2✓ Q3 partial Q4✓) — opt-in until GOAT gate validates spectral concentration at NPC-scale.

**Constraints:**
- Modelless only — SVD + matrix multiply. No training, no gradient descent, no softmax.
- Reuse `thin_svd_into` + `SvdResultScratch` + `SvdScratch` from `katgpt-core/subspace_phase_gate`.
- SOLID, DRY, files <2048 lines.
- Tests + GOAT bench with before/after expected gains.

---

## Goal

Ship the modelless residue distilled from SAR (Research 406): a pure deterministic
function that projects ANY weight delta ΔW onto the base weight matrix W₀'s SVD
spectral subspace, extracting the on-manifold ("rewired") component ΔW* and a
compact rewiring matrix M = UᵀΔWV.

**The paper's headline** (extract a reasoning core from a trained RL delta W_RL)
requires gradient-trained weights → routes to riir-train. **This plan ships the
modelless kernel** that operates on freeze/thaw deltas and LoRA overlays — the
geometry (project onto base SVD subspace, extract compact interaction matrix)
is identical regardless of how ΔW was produced.

**Crate placement:** `katgpt-spectral`, alongside `off_principal` (Plan 264 Phase 2).
`off_principal` projects a query *away* from the base SVD subspace (the
off-principal component). `spectral_rewire` projects a delta *onto* the base SVD
subspace (the on-principal component). They are geometric complements — same SVD
substrate, opposite projection direction. Co-locating them in `katgpt-spectral`
keeps the spectral-projection family together (DRY).

**Feature flag:** `spectral_rewire` in `katgpt-spectral/Cargo.toml` (opt-in).
Root `katgpt-rs/Cargo.toml` forwards as `spectral_rewire = ["katgpt-spectral/spectral_rewire"]`.
NOT in root `default` until GOAT gate passes.

---

## The Math (reference for implementer)

Given base weights W₀ (d_out × d_in, row-major flat) and delta ΔW (same shape):

```
1. SVD:   W₀ = U Σ Vᵀ                  via thin_svd_into(w0, d_out, d_in, ...)
2. Truncate to top-rank r:
     U_r = U[:, 0..r]   (d_out × r, column-major in SvdResultScratch)
     V_r = V[:, 0..r]   (d_in × r, column-major)
3. Rewiring matrix:  M = U_rᵀ ΔW V_r   (r × r)
     temp = U_rᵀ ΔW     (r × d_in):   temp[i][j] = Σ_k U_r[k][i] · ΔW[k][j]
     M    = temp · V_r  (r × r):      M[i][j]    = Σ_k temp[i][k] · V_r[k][j]
4. Purified delta:   ΔW* = U_r M V_rᵀ  (d_out × d_in)
     temp2 = U_r · M    (d_out × r)
     ΔW*   = temp2 · V_rᵀ (d_out × d_in)
5. Residual:         ΔW⊥ = ΔW − ΔW*
6. On-manifold fraction: ‖ΔW*‖_F / ‖ΔW‖_F
```

Off-diagonal elements M[i][j] (i≠j) represent cross-skill "rewiring" —
many-to-one logical synthesis (the paper's key insight for compositional
reasoning). Diagonal M[i][i] represents in-skill strength modulation.

**Note on SvdResultScratch layout:** left/right singular vectors are stored
flat column-major (`m_rows × len` / `n_cols × len`). Column j lives at
`[j * stride .. (j+1) * stride]`. The matmuls must respect this layout —
see `off_principal.rs::off_principal_project` for the existing column-major
access pattern in this crate.

---

## Phase 1 — Core Function + Scratch Struct

### Tasks

- [x] **T1.1** Create `katgpt-spectral/src/spectral_rewire.rs` module skeleton.
  Gate behind `spectral_rewire` feature in `katgpt-spectral/Cargo.toml`.
  Add `pub mod spectral_rewire;` in `katgpt-spectral/src/lib.rs` under the
  feature gate. Enable `katgpt-core/subspace_phase_gate` as a feature dep
  (for `thin_svd_into` + `SvdResultScratch` + `SvdScratch`).
- [x] **T1.2** Define `SpectralRewireScratch` struct — pre-allocated reusable
  buffers:

  ```rust
  pub struct SpectralRewireScratch {
      /// SVD result (SOA scratch) for W₀ decomposition.
      svd_result: SvdResultScratch,
      /// SVD working buffers.
      svd_work: SvdScratch,
      /// Temp buffer for U_rᵀ·ΔW (r × d_in), row-major.
      ut_delta: Vec<f32>,
      /// Rewiring matrix M (r × r), row-major.
      m_buf: Vec<f32>,
      /// Temp buffer for U_r·M (d_out × r), row-major.
      um_buf: Vec<f32>,
      /// Purified delta ΔW* (d_out × d_in), row-major.
      delta_star_buf: Vec<f32>,
      /// Residual ΔW⊥ = ΔW − ΔW* (d_out × d_in), row-major.
      residual_buf: Vec<f32>,
  }
  ```

  Implement `SpectralRewireScratch::with_capacity(d_out, d_in, max_rank)` and
  `ensure_capacity(&mut self, d_out, d_in, rank)` (grows buffers only if
  dimensions increased — mirrors `VCycleScratch` / `PointSamplerScratch` pattern).
- [x] **T1.3** Implement `spectral_rewire_into` — the zero-alloc hot path:

  ```rust
  pub fn spectral_rewire_into(
      w0: &[f32],       // base weights, row-major (d_out × d_in)
      delta: &[f32],    // weight delta, same shape
      d_out: usize,
      d_in: usize,
      rank: usize,      // top-k spectral rank (r ≤ min(d_out, d_in))
      scratch: &mut SpectralRewireScratch,
  ) -> SpectralRewireOutput<'_>;
  ```

  Steps: SVD W₀ → truncate to rank r → compute M = U_rᵀΔWV_r → compute
  ΔW* = U_r M V_rᵀ → compute residual → compute on_manifold_fraction.
  All writes go into `scratch` buffers. No allocation after warmup.
- [x] **T1.4** Implement `SpectralRewireOutput` — borrows into scratch:

  ```rust
  pub struct SpectralRewireOutput<'a> {
      /// Purified delta ΔW* = U_r M V_rᵀ (on-manifold). Borrows scratch.delta_star_buf.
      pub delta_star: &'a [f32],
      /// Compact rewiring matrix M = U_rᵀ ΔW V_r (r × r). Borrows scratch.m_buf.
      pub rewiring_matrix: &'a [f32],
      /// Off-manifold residual ΔW⊥ = ΔW − ΔW*. Borrows scratch.residual_buf.
      pub residual: &'a [f32],
      /// ‖ΔW*‖_F / ‖ΔW‖_F — on-manifold energy fraction ∈ [0, 1].
      pub on_manifold_fraction: f32,
  }
  ```
- [x] **T1.5** Implement convenience wrapper `spectral_rewire` (allocating) —
  calls `spectral_rewire_into` with a local scratch, copies results into owned
  `Vec<f32>`. For tests, examples, and cold-path callers only.
- [x] **T1.6** Add root forwarding in `katgpt-rs/Cargo.toml`:
  `spectral_rewire = ["katgpt-spectral/spectral_rewire"]` (opt-in, NOT in default).
  Add `pub use katgpt_spectral::spectral_rewire;` re-export in root `lib.rs`.
- [x] **T1.7** Unit test: synthetic delta round-trip. **PASS** — 7/7 tests green:
  round-trip (ΔW* matches ΔW < 1e-4 rel, on_manifold_fraction > 0.999, M norm match),
  zero-delta, on+off=ΔW reconstruction, fraction ∈ [0,1], higher-rank-monotone
  (full-rank → 1.0), non-square (16×4 r=3), scratch-reuse consistency.

---

## Phase 2 — Rewiring Matrix Diagnostics

### Tasks

- [ ] **T2.1** Implement `rewiring_matrix_diagnostics(m: &[f32], rank: usize)`
  → `RewiringDiagnostics`:
  - `diagonal_energy`: Σᵢ M[i][i]² / Σᵢⱼ M[i][j]² — fraction of rewiring energy
    on the diagonal (in-skill modulation vs cross-skill rewiring).
  - `off_diagonal_energy`: 1 − diagonal_energy.
  - `spectral_norm_estimate`: largest |M[i][i]| (diagonal dominance proxy).
  - `rewiring_sparsity`: fraction of off-diagonal |M[i][j]| below a threshold.
- [ ] **T2.2** Unit test: diagnostics on identity-M (all diagonal, no rewiring)
  → diagonal_energy = 1.0; on pure off-diagonal M → diagonal_energy = 0.0.
- [ ] **T2.3** Doc-test: show before/after — a noisy delta with known on-manifold
  component, demonstrate on_manifold_fraction and the rewiring matrix structure.

---

## Phase 3 — GOAT Gate

The GOAT gate validates the spectral concentration property at our scale
(the paper's Q3 caveat — proven for 1.5B–32B LLM weights, unvalidated for
NPC-scale matrices). **If G1 fails, the primitive stays opt-in permanently**
and the Super-GOAT promotion path closes.

### Tasks

- [ ] **T3.1 (G1) Spectral concentration.** For synthetic rank-r deltas (r=32)
  injected into random 512×512 base matrices, on_manifold_fraction > 0.5.
  This tests whether the top-r singular subspace captures the majority of the
  delta's energy — the core assumption. Also test at NPC-scale: 64×64 (reshaped
  style_weights) and 128×128 (LoRA-scale). **Decision gate:** if on_manifold_fraction
  < 0.5 at any scale, document the failure honestly and keep opt-in. Do NOT
  promote to default.
- [ ] **T3.2 (G2) Singular-direction preservation.** The purified ΔW* preserves
  W₀'s top-r right singular directions: compute SVD of (W₀ + ΔW*) and verify
  the top-r right singular vectors have cosine > 0.99 with W₀'s top-r. (The
  projection is onto W₀'s subspace by construction — this verifies numerically.)
- [ ] **T3.3 (G3) No TIES regression.** Run existing TIES merging tests
  (`bench_094_*` or equivalent) with `spectral_rewire` feature enabled.
  No new failures, no output drift. Feature-isolation: enabling spectral_rewire
  must not change behavior of any other module.
- [ ] **T3.4 (G4) Zero-alloc.** `CountingAllocator` test (mirror
  `subspace_phase_gate_alloc_check` pattern): after warmup,
  `spectral_rewire_into` allocates 0 bytes across 1000 calls at fixed dimensions.
  Buffer growth only on dimension change.
- [ ] **T3.5 (G5) Latency.** `criterion` bench: 512×512 matrix at rank-32,
  target < 1ms. Also bench 64×64 at rank-8 (NPC scale), target < 10µs.
  Report breakdown: SVD vs matmul vs residual. The SVD dominates — note whether
  `thin_svd_into` (one-sided Jacobi) meets the budget or if a faster SVD path
  is needed for hot-loop use.
- [ ] **T3.6 (G6) Feature isolation.** `cargo check --all-features` and
  `cargo check --no-default-features --features spectral_rewire` both clean.
  No interaction with other features.

---

## Phase 4 — Cross-Repo Application Notes (not implemented here)

These are follow-up plans in sibling repos. Documented here for routing.

- [ ] **T4.1 (note)** Fusion C — Freeze/Thaw purification (riir-neuron-db):
  when freezing a personality snapshot, project the delta onto the base shard's
  spectral subspace. Store compact M (r×r) in `MerkleFrozenEnvelope` instead of
  full delta. **Blocked** on: `NeuronShard::style_weights[64]` is a vector not
  a matrix — needs reshaping (8×8) or application to LoRA overlay matrices
  (which ARE matrices). File as a riir-neuron-db plan when this primitive lands.
- [ ] **T4.2 (note)** Fusion D — Spectral LoRA (riir-ai / katgpt-rs Plan 025):
  when applying a reader/writer LoRA pair, project the LoRA product BA onto the
  base weight's spectral subspace before adding to W₀. The result is a purified
  overlay. **Blocked** on: this primitive landing + identifying the LoRA
  application hot path.
- [ ] **T4.3 (note)** Fusion A — Spectral TIES (katgpt-rs Plan 094 upgrade):
  replace magnitude-based filtering in TIES merging with spectral filtering.
  Project each task vector onto base SVD before merge. Requires the merging
  consumer to have a "base" SVD available. **Blocked** on: this primitive +
  TIES merge pipeline refactor. Tracked in Issue 123 (Fusion B decomposition).

---

## Open Questions (resolve during implementation)

1. **SVD path.** `thin_svd_into` (one-sided Jacobi in subspace_phase_gate) vs
   `newton_schulz` (Newton-Schulz orthogonalization). The Jacobi path gives
   exact U/Σ/V but may be slower. Profile both at Phase 3 G5. If Jacobi misses
   the latency budget, consider caching the SVD of W₀ (computed once at freeze
   time, reused across all delta projections on the same base).
2. **SVD caching.** The base W₀ is fixed across many delta projections (e.g.,
   all LoRA overlays for the same base). A `SpectralRewireIndex { u_r, v_r }`
   pre-computed from W₀ would eliminate the SVD from the hot loop. Design this
   if G5 latency demands it — mirror `OffPrincipalIndex::new(base, k_frac)`
   which pre-caches U_k.
3. **rank selection.** The paper uses top-1% rank. For a 512×512 matrix that's
   rank ~5; for 64×64 it's rank ~0.6 (meaningless). The adaptive rank from
   `spectral_concentration` (Plan 264 Phase 3, already in katgpt-spectral)
   may be the right selector. Wire `adaptive_rank` as the default rank chooser.
4. **Vector vs matrix.** `style_weights[64]` cannot be SVD'd as-is. The 8×8
   reshape is the minimum viable path. Document the reshape convention (row-major
   vs the neuron-db's internal layout) if T4.1 is pursued.

---

## Honest Limitations (from Research 406 §7)

1. **Scale mismatch.** Spectral concentration (reasoning component in top-1%
   rank) is proven for 1.5B–32B LLMs. NPC-scale matrices are much smaller. G1
   is the make-or-break gate.
2. **No RL deltas.** We have no RL training. The modelless residue operates on
   freeze/thaw and LoRA deltas — a different, unvalidated application than the
   paper's RL extraction.
3. **Precision.** The paper notes FP32 for SVD, FP16 for storage. Our runtime
   uses mixed precision. G1/G2 must check precision robustness.

---

## Cross-references

- **Research 406** — the GOAT verdict + 4 fusion paths.
- **Research 231 / Plan 264 (SOPTV)** — off-principal projection (the geometric
  complement). `off_principal.rs` in katgpt-spectral.
- **Plan 094 (TIES Merging)** — magnitude-based merge (Fusion A upgrade target).
- **Plan 025 (LoRA Hot-Swap)** — reader/writer LoRA pair (Fusion D target).
- **Plan 301 (subspace_phase_gate)** — `thin_svd_into`, `SvdResultScratch`,
  `SvdScratch` (the SVD substrate to reuse).
- **Issue 123 (Fusion B)** — two-component decomposition (SAR on-principal +
  SOPTV off-principal), candidate Super-GOAT.

---

## TL;DR

Ship the modelless SAR kernel in `katgpt-spectral`: project a weight delta onto
the base matrix's SVD subspace, extract the compact rewiring matrix M, reconstruct
the purified on-manifold delta ΔW*. Reuse `thin_svd_into` from subspace_phase_gate.
The GOAT gate (G1–G6) is the make-or-break: G1 validates spectral concentration at
NPC-scale (the paper's Q3 caveat). If G1 fails, keep opt-in permanently. Cross-repo
applications (freeze/thaw purification, spectral LoRA, spectral TIES) are noted as
follow-ups but NOT implemented in this plan.
