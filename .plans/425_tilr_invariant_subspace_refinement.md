# Plan 425: TILR — Trajectory-Invariant Latent Refinement (Alignment-Gated Subspace Correction)

**Date:** 2026-07-10
**Research:** [408_Trajectory_Invariant_Latent_Refinement.md](../.research/408_Trajectory_Invariant_Latent_Refinement.md)
**Source paper:** [arXiv:2606.29164](https://arxiv.org/abs/2606.29164) — Malarkkan et al., *TILR: Trajectory-Invariant Latent Refinement*, ICML 2026 Mech Interp Workshop
**Target:** `katgpt-rs/crates/katgpt-core/src/tilr.rs` (new module) + Cargo feature `tilr_invariant_subspace`
**Status:** Active — Phase 1 + Phase 2 COMPLETE, all GOAT gates PASS (2026-07-10)

**Constraints:**
- Modelless only — SVD + projection + alignment gate. No training, no gradient descent, no softmax.
- Reuse `thin_svd_into` + `SvdResultScratch` + `SvdScratch` from `katgpt-core/subspace_phase_gate` (Plan 301).
- Reuse `subspace_ratios` logic from `katgpt-spectral/river_valley` (Plan 152) — DRY, do not duplicate the γ-ratio math.
- Use **sigmoid** (never softmax) for any gating. The γ alignment ratio is already in [0,1] by construction (it's a norm ratio), so no sigmoid is needed on γ itself — but any *derived* gate (e.g. a softened threshold) MUST use sigmoid.
- SOLID, DRY, files <2048 lines. Zero-alloc hot path.
- `Uuid::now_v7()` if any UUIDs are needed (unlikely for a pure-math primitive).

---

## Goal

Ship the open primitive distilled from TILR (Research 408): a zero-alloc
**alignment-gated subspace-projected correction** that applies a contrastive
direction `d` to a latent state `s`, projected onto a frozen SVD basis `U_r`,
with the step size modulated by the alignment fraction `γ = ‖Πd‖ / ‖d‖` so that
`γ→0` bit-recovers the uncorrected input (strict no-harm guarantee).

**The paper's calibration phase** (collecting `N·T` contrastive differences
from two reference checkpoints and running SVD) is a *one-time offline
artefact-construction* step. It is NOT part of this plan's primitive — the
primitive **consumes** a pre-computed `U_r` basis (produced by Plan 301's
`thin_svd_into`, or Plan 423's spectral index, or any other SVD source). This
plan ships the **runtime correction gate** only.

**Crate placement:** `katgpt-core` (leaf, crates.io-eligible). The primitive is
pure linear algebra — SVD basis (input) + projection + norm ratio + scaled add.
No game semantics, no chain semantics, no shard semantics. This is the same
placement rationale as `subspace_steering.rs` (Plan 412) and
`spectral_rewire.rs` (Plan 423): generic spectral-projection family, public.

**Feature flag:** `tilr_invariant_subspace` in
`katgpt-core/Cargo.toml` (opt-in). Root `katgpt-rs/Cargo.toml` forwards as
`tilr_invariant_subspace = ["katgpt-core/tilr_invariant_subspace"]`.
NOT in root `default` until GOAT gate passes.

---

## The Math (reference for implementer)

Given:
- state `s ∈ ℝ^d` (the latent state to correct — e.g. HLA 8-d, shard 64-d)
- direction `d ∈ ℝ^d` (the per-instance contrastive direction)
- basis `U_r ∈ ℝ^(d×r)` (frozen, orthonormal columns — the SVD top-r right singular vectors of the contrastive-difference matrix)
- `eta_base ∈ [0, 1]` (the base step size)
- `epsilon > 0` (numerical guard, default `1e-12`)

Compute:
```
1. Project:   d_proj = U_r (U_r^T d)            // O(d·r) matvec
2. Align:     gamma  = ‖d_proj‖ / (‖d‖ + ε)      // ∈ [0, 1]
3. Gate:      eta    = eta_base · gamma           // ∈ [0, eta_base]
4. Apply:     s'     = s + eta · d_proj           // O(d) add
```

**No-harm contract:** when `gamma = 0` (d orthogonal to U_r), `eta = 0`,
`s' = s` bit-identically. When `gamma = 1` (d ∈ span(U_r)), `eta = eta_base`,
`s' = s + eta_base · d` (full projected correction).

**Numerical note:** the projection `d_proj = U_r (U_r^T d)` is computed as a
two-step matvec (not by materializing `Π = U_r U_r^T ∈ ℝ^(d×d)`). The intermediate
`U_r^T d ∈ ℝ^r` is the projection coefficients; `d_proj` reconstructs from them.
This is `O(d·r)` not `O(d²)` — the paper's headline cost claim.

---

## Reused primitives (do NOT re-implement)

| Primitive | Source | Why reuse |
|---|---|---|
| `thin_svd_into` + `SvdScratch` + `SvdResultScratch` | `katgpt-core/subspace_phase_gate.rs` (Plan 301) | The SVD used to discover `U_r` offline. TILR consumes the output `right_singular_vectors`. |
| `subspace_ratios` (the γ math) | `katgpt-spectral/river_valley.rs` (Plan 152) | Computes `r_dom = ‖U_k^T g‖ / ‖g‖` — **the identical metric**. Extract or re-expose the core ratio computation; do not duplicate. |
| `simd_dot_f32`, `simd_*` | `katgpt-core/simd.rs` | SIMD-accelerated dot products for the projection + norm computations. |

**DRY decision:** the γ ratio lives in `katgpt-spectral`, but TILR lives in
`katgpt-core` (leaf). Two options:
- **(A)** Move the ratio helper to `katgpt-core` (leaf) and have
  `katgpt-spectral` re-export — clean dependency direction (leaf owns the math).
- **(B)** Duplicate the ~5-line ratio computation in `katgpt-core/tilr.rs` with
  a cross-reference comment — avoids a refactor of `katgpt-spectral`.

**Recommendation: (B) for Phase 1** (5 lines, with a `// cf. river_valley::subspace_ratios` comment). Open an issue for (A) if the duplication causes
maintenance drift. The ratio is trivial enough that duplication is cheaper than
the refactor risk.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/tilr.rs` with module doc
      explaining: TILR mechanism (5 steps from Research 408 §1.2), the no-harm
      contract, the reuse map (Plan 301 SVD, Plan 152 ratio, Plan 412 steering),
      and the const-generic signature.
- [x] **T1.2** Define the core struct + error enum:

      ```rust
      /// Errors returned by [`tilr_refine_into`].
      #[derive(Debug, Clone, Copy, PartialEq, Eq)]
      #[repr(u8)]
      pub enum TilrError {
          /// `basis` columns are not mutually orthonormal within `orthonormal_tol`.
          NotOrthonormal,
          /// `eta_base` is outside `[0.0, 1.0]`.
          EtaOutOfRange,
          /// `state`, `direction`, and basis row-dimension `d` disagree.
          DimensionMismatch,
      }

      /// Scratch buffer for zero-alloc TILR refinement.
      ///
      /// Holds the projection coefficients `U_r^T d ∈ ℝ^r` and the projected
      /// direction `d_proj ∈ ℝ^d`. Reuse across calls via `clear()` + rewrite.
      pub struct TilrScratch {
          coeffs: Vec<f32>,   // length r
          d_proj: Vec<f32>,   // length d
      }

      impl TilrScratch {
          pub fn with_capacity(d: usize, r: usize) -> Self { ... }
      }
      ```

- [x] **T1.3** Implement the core zero-alloc function:

      ```rust
      /// Alignment-gated subspace-projected correction (the TILR primitive).
      ///
      /// Applies `s' = s + eta_base * gamma * d_proj` where:
      /// - `d_proj = U_r (U_r^T d)` (projection onto the invariant subspace)
      /// - `gamma  = ‖d_proj‖ / (‖d‖ + epsilon)` (alignment fraction ∈ [0,1])
      ///
      /// **No-harm contract:** when `gamma = 0`, `s' = s` bit-identically.
      ///
      /// `basis` is `r` orthonormal column-vectors of length `d` (row-major flat
      /// `r × d`, i.e. `basis[k*d + i]` = component `i` of basis vector `k`).
      /// `state` and `direction` are length `d`. `out` is length `d` (may alias
      /// `state` for in-place).
      ///
      /// Returns `gamma` (the alignment fraction) for diagnostic/logging use.
      pub fn tilr_refine_into(
          state: &[f32],
          direction: &[f32],
          basis: &[f32],      // r × d, row-major
          r: usize,
          eta_base: f32,
          epsilon: f32,
          scratch: &mut TilrScratch,
          out: &mut [f32],
      ) -> Result<f32, TilrError> { ... }
      ```

      Implementation (all SIMD via `katgpt_core::simd`):
      1. Validate dimensions (`state.len() == direction.len() == out.len() == d`,
         `basis.len() == r * d`).
      2. Compute projection coefficients: `coeffs[k] = simd_dot_f32(&basis[k*d..], direction, d)` for `k in 0..r`.
      3. Compute projected direction: `d_proj[i] = Σ_k coeffs[k] * basis[k*d + i]`.
      4. Compute `‖d_proj‖² = simd_dot_f32(d_proj, d_proj)`.
      5. Compute `‖d‖² = simd_dot_f32(direction, direction)`.
      6. `gamma = sqrt(‖d_proj‖² / (‖d‖² + epsilon))`, clamp to `[0, 1]`.
      7. `eta = eta_base * gamma`.
      8. `out[i] = state[i] + eta * d_proj[i]`.
      9. Return `gamma`.

- [x] **T1.4** Add an owning convenience wrapper (allocates, for non-hot paths):

      ```rust
      pub fn tilr_refine(
          state: &[f32],
          direction: &[f32],
          basis: &[f32],
          r: usize,
          eta_base: f32,
      ) -> Result<(Vec<f32>, f32), TilrError> { ... }
      ```

- [x] **T1.5** Add feature gate `tilr_invariant_subspace` to
      `katgpt-core/Cargo.toml`. Gate the module behind
      `#[cfg(feature = "tilr_invariant_subspace")]`.
- [x] **T1.6** Register the module in `katgpt-core/src/lib.rs`:
      `#[cfg(feature = "tilr_invariant_subspace")] pub mod tilr;`
- [x] **T1.7** Forward the feature in root `katgpt-rs/Cargo.toml`:
      `tilr_invariant_subspace = ["katgpt-core/tilr_invariant_subspace"]` (opt-in,
      NOT in `default`).

**Phase 1 deviations (documented):**
- **T1.3 deviation**: Added a `tilr_refine_apply()` in-place variant that takes
  `&mut [f32]` only. The plan's `tilr_refine_into` signature uses `state: &[f32]`
  + `out: &mut [f32]`, which Rust's borrow checker prevents from aliasing. The
  doc originally claimed "may alias `state`" — corrected. `tilr_refine_apply`
  provides the in-place mutation path (caller holds `&mut`, no separate `state`
  borrow).
- **T1.3 deviation**: Added a `check_orthonormal()` setup-time validation
  helper (NOT on the hot path). The plan's `TilrError::NotOrthonormal` variant
  is returned by `check_orthonormal`, not by `tilr_refine_into` (which trusts
  the basis for perf — the O(r²·d) orthonormality check would dominate the
  O(d·r) correction). This mirrors `subspace_steering` (Plan 412): validate at
  construction, trust at apply.

---

## Phase 2 — GOAT Gate (correctness + no-harm + perf)

### Tasks

- [x] **T2.1 (G1a — no-harm bit-identity)** Unit test: `gamma = 0` case (direction
      orthogonal to basis) produces `out == state` **bit-identically** (assert
      `out.iter().zip(state).all(|(a, b)| a.to_bits() == b.to_bits())`). This is
      the load-bearing no-harm contract.
- [x] **T2.2 (G1b — full-correction parity)** Unit test: `gamma = 1` case
      (direction ∈ span(basis)) produces `out == state + eta_base * direction`
      within f32 tolerance.
- [x] **T2.3 (G1c — ranking preservation)** Unit test: for two directions
      `d_a, d_b` that differ only outside `span(U_r)`, the projected corrections
      are identical (`d_proj_a == d_proj_b`). This is the "subspace-mediated input
      invariance" property from Research 408 §1.4.
- [x] **T2.4 (G1d — gamma monotonicity)** Unit test: as the direction rotates
      from orthogonal-to-basis to within-basis, `gamma` increases monotonically
      from 0 to 1.
- [x] **T2.5 (G1e — orthonormality validation)** Unit test: non-orthonormal basis
      returns `Err(TilrError::NotOrthonormal)`.
- [x] **T2.6 (G2 — perf overhead <3%)** Criterion bench: measure
      `tilr_refine_into` at `d=8, r=3` (HLA scale) and `d=64, r=12` (shard scale).
      Target: <50 ns/call at HLA scale, <200 ns/call at shard scale. Verify the
      `O(d·r)` matvec is negligible vs a typical forward pass. Use
      `CARGO_TARGET_DIR=/tmp/tilr_goat` per AGENTS.md; clean up when done.
- [x] **T2.7 (G3 — no regression)** `cargo test -p katgpt-core --lib` passes with
      and without `--features tilr_invariant_subspace`. Zero new warnings.
- [x] **T2.8 (G4 — alloc-free)** Unit test with a custom allocator that panics on
      alloc: confirm `tilr_refine_into` does not allocate on the hot path
      (scratch is pre-allocated). Pattern: same as Plan 412 T4.x alloc-free gate.

**Phase 2 results (2026-07-10):**
- G1 (no-harm bit-identity): 100 random orthogonal directions × {(d=8,r=3), (d=64,r=12)}
  → max γ at orthogonal = 0.0 exactly, 0 bit mismatches. **PASS**
- G2 (full-correction parity + boundedness): 1000 random triples + basis-vector
  directions → 0 OOB, 0 NaN, full-correction max err 5.96e-8 << 1e-4. **PASS**
- G3 (latency): HLA (d=8,r=3) 24.7 ns < 50 ns target; Shard (d=64,r=12) 123.0 ns < 200 ns target. **PASS**
- G4 (alloc-free): 0 allocations / 100 steady-state calls (CountingAllocator). **PASS**

**GOAT gate verdict: G1+G2+G3+G4 ALL PASS.** The primitive is modelless
(closed-form SVD projection + norm ratio + SAXPY), not UQ-bearing (no conformal
floor needed), and ready for default promotion. The promotion itself is
deferred to the commit step (update `default` list in katgpt-core/Cargo.toml).

---

## Phase 3 — Offline Calibration Helper (optional, P2)

The paper's calibration phase (collect contrastive differences, run SVD, produce
`U_r`). This is a convenience helper, NOT the core primitive — consumers can use
Plan 301's `thin_svd_into` directly.

### Tasks

- [ ] **T3.1** Add `discover_invariant_subspace(differences: &[&[f32]], tau: f32) -> Result<Vec<f32>, TilrError>`:
      - Stack `differences` (each length `d`) column-wise into `Δ ∈ ℝ^(d×N)`.
      - Run `thin_svd_into` (Plan 301) on `Δ`.
      - Select `r` = smallest rank retaining `tau` fraction of variance (cumulative `Σ² / total Σ²`).
      - Return the top-`r` right singular vectors flattened (`r × d` row-major).
- [ ] **T3.2** Unit test: synthetic differences lying in a known 2-d subspace →
      `discover_invariant_subspace` recovers that subspace (principal angles < 1°).
- [ ] **T3.3** Document the offline-vs-online split: `discover_invariant_subspace`
      runs once at calibration; `tilr_refine_into` runs per-step at inference.

---

## Phase 4 — Integration Notes + Docs (P2)

### Tasks

- [ ] **T4.1** Add a README section under `katgpt-rs/.docs/` (if a steering /
      subspace doc group exists) or a standalone note cross-referencing Plan 412
      (subspace_steering), Plan 423 (spectral_rewire), Plan 152 (river_valley).
      Frame TILR as the "alignment-gated" member of the subspace-projection family.
- [ ] **T4.2** Add an example at `katgpt-rs/examples/tilr_demo.rs`: synthetic
      contrastive pair → SVD → γ-gated correction → show (a) no-harm when γ=0,
      (b) full correction when γ=1, (c) graceful intermediate behavior.
- [ ] **T4.3** Cross-reference from Research 408 §4 consumer wiring:
      - riir-ai HLA no-harm personality refinement (follow-up issue).
      - riir-neuron-db freeze/thaw shard refinement (follow-up issue).
      - riir-ai `reestimation.rs` γ-gated step size (follow-up issue).

---

## GOAT Gate Summary (promote-to-default decision)

| Gate | Target | How measured | Status |
|---|---|---|---|
| **G1** (correctness) | γ→0 bit-recovers input; γ=1 full correction; ranking preserved; gamma monotone | T2.1–T2.5 unit tests | ✅ PASS |
| **G2** (perf) | <50 ns/call HLA scale, <200 ns/call shard scale, <3% overhead | T2.6 criterion bench | ✅ PASS (24.7/123.0 ns) |
| **G3** (no regression) | All katgpt-core tests pass with + without feature; 0 new warnings | T2.7 `cargo test` | ✅ PASS (1445/1426 tests) |
| **G4** (alloc-free) | Zero heap alloc on `tilr_refine_into` hot path | T2.8 custom-allocator test | ✅ PASS (0 allocs/100 calls) |

**UQ-bearing?** NO — TILR does not claim a probability distribution, predictive
interval, quantile, coverage guarantee, or calibrated uncertainty. It's a
deterministic linear-algebra correction. **No conformal floor needed.**

**Promote-to-default rule:** if G1–G4 all pass, promote `tilr_invariant_subspace`
to root `default`. Demote: N/A (no competing primitive in the same slot — TILR
is the alignment-gated member; Plan 412 subspace_steering is the ungated member;
they coexist).

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| γ ratio duplicates `river_valley::subspace_ratios` (DRY violation) | High | Low | 5-line duplication with cross-ref comment (Phase 1); refactor to leaf in a follow-up issue if drift occurs |
| No-harm bit-identity breaks under f32 rounding at γ≈0 | Medium | High | Clamp γ to exactly 0.0 when `‖d_proj‖² < epsilon` (not just `gamma < eps`) — ensures `eta = 0.0` exactly, not `eta ≈ 1e-38` |
| Feature-gated module not registered in lib.rs | Low | Medium | T1.6 + T2.7 (cargo test both with/without feature) |
| Offline SVD calibration is confused with the runtime primitive | Medium | Low | T3.3 explicit doc split; the primitive consumes `U_r`, does not compute it |
| Consumer wiring (riir-ai, riir-neuron-db) attempted in this plan | Low | Medium | This plan is katgpt-rs-only; consumer wiring is explicitly deferred to follow-up issues (T4.3) |

---

## Out of Scope

- **Consumer wiring** in riir-ai (HLA no-harm personality refinement, functor
  γ-gated re-estimation) and riir-neuron-db (freeze/thaw shard refinement).
  These are follow-up issues, tracked in T4.3.
- **The paper's full calibration pipeline** (N=200 inputs, T steps, reference
  forward passes). Phase 3 ships a minimal SVD helper; the full calibration
  orchestration is a consumer concern.
- **Training the reference pair.** The reference pair `(f_good, f_bad)` comes
  from freeze/thaw snapshots or epoch checkpoints — modelless by construction.
  If a consumer needs *trained* reference pairs → riir-train.
- **Multi-layer equivalence** (the paper notes single-layer equivalence is
  proven; multi-layer is future work). Out of scope — this primitive operates
  on a single latent state, not a multi-layer stack.

---

## TL;DR

Ship `tilr_refine_into(state, direction, basis, r, eta_base, scratch, out)` — a
zero-alloc alignment-gated subspace-projected correction with a bit-identical
`γ→0` no-harm contract. Reuses Plan 301 SVD, Plan 152 γ-ratio logic, Plan 412
steering patterns. GOAT gate G1–G4 (no conformal floor — not UQ-bearing).
Feature `tilr_invariant_subspace`, opt-in until gate passes. Consumer wiring
(riir-ai HLA, riir-neuron-db shards) deferred to follow-up issues.
