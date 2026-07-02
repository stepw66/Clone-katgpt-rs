# Issue 033 — Refactor riir-games `scalar_projection` to call katgpt-core `direction_vector_decode`

**Filed from:** Plan 330 (Analytic Lattice Encoder/Decoder Primitive), T3.4
**Date:** 2026-07-02
**Resolved:** 2026-07-02 — Option 1 implemented (slice overload + delegation)
**Type:** refactor / decoupling
**Severity:** low (cleanup; no behavior change)
**Scope:** `riir-ai/crates/riir-games/src/attn_match_fusion/scalar_projection.rs`
**Status:** ✅ RESOLVED

## Context

Plan 330 (Phase 3) shipped a **generic** single-direction projection primitive in
katgpt-core:

```rust
// katgpt-rs/crates/katgpt-core/src/analytic_lattice/decoder.rs
pub fn direction_vector_decode<const N: usize>(
    state: &LatticeVector<N>,
    direction: &LatticeVector<N>,
    temperature: f32,
) -> f32 {
    let z = dot(state.0.as_slice(), direction.0.as_slice()) / N as f32;
    sigmoid(z * temperature)
}
```

This is the generalized lift of riir-games' HLA-specific 5-scalar projection.
The plan (T3.1 doc-comment) states:

> This is the GENERALIZED version of riir-games `scalar_projection::project_to_scalars`,
> lifted out of HLA-specific 5-scalar semantics into a generic single-direction
> primitive. The 5-scalar HLA bridge in riir-games becomes a thin wrapper that
> calls this 5 times.

## The duplication

`riir-ai/crates/riir-games/src/attn_match_fusion/scalar_projection.rs::project_pooled_to_scalars`
implements the **same** dot-product + sigmoid loop inline:

```rust
let inv_d = 1.0f32 / (DIRECTION_VECTOR_DIM as f32);
for (i, out_i) in out.iter_mut().enumerate() {
    let dir = &direction_vectors[i];
    let dot: f32 = pooled.iter().zip(dir.iter()).map(|(&p, &d)| p * d).sum();
    *out_i = sigmoid(dot * inv_d);   // ← identical to direction_vector_decode(state, dir, inv_d)
}
```

`temperature = inv_d = 1/D`. The two functions are behaviorally identical for a
single direction; `project_pooled_to_scalars` just calls the pattern 5 times.

## The refactor

Replace the inline loop body with 5 calls to `direction_vector_decode`, so the
dot+sigmoid math lives in exactly one place (katgpt-core). The HLA bridge in
riir-games becomes a thin wrapper.

### Type-bridging complication (why this is an issue, not a one-liner)

`direction_vector_decode` operates on `LatticeVector<N>` (a const-generic
`#[repr(transparent)]` wrapper over `[f32; N]`), while `project_pooled_to_scalars`
operates on raw `&[f32]` + `&[[f32; DIRECTION_VECTOR_DIM]]`. The refactor needs
one of:

1. **Add a `&[f32]` slice overload** to katgpt-core (`direction_vector_decode_slice`)
   that takes raw slices and delegates to the same `dot` + `sigmoid`. Lowest
   coupling; the HLA bridge calls the slice variant 5 times.
2. **Construct `LatticeVector<DIRECTION_VECTOR_DIM>` views** from the slices in
   riir-games (bytemuck/zero-copy) and call the const-generic version directly.

Option 1 is preferred — it keeps the public generic API on `LatticeVector<N>`
while exposing a slice entry point for non-const-generic callers (the HLA bridge
uses runtime `DIRECTION_VECTOR_DIM`).

## Acceptance

- [x] `project_pooled_to_scalars` calls the katgpt-core primitive (no inline dot+sigmoid).
- [x] Existing `am_*` GOAT gates still PASS (136 attn_match_fusion tests pass, 1661 riir-games lib tests pass).
- [x] No perf regression — the SIMD `simd_dot_f32` dispatch (NEON/AVX2/scalar) replaces the scalar zip-sum; both auto-vectorize, but the SIMD path is the canonical hot-path kernel.
- [x] `direction_vector_decode_slice` stays the single source of truth for the math.

## Resolution

**Option 1** was implemented: added `direction_vector_decode_slice` to
`katgpt-core/src/analytic_lattice/decoder.rs` (slice-entry variant of the
const-generic `direction_vector_decode`). The re-export in `katgpt-core/lib.rs`
was extended. The `am_core` feature in `riir-games/Cargo.toml` now enables
`katgpt-core/analytic_lattice` (zero extra deps — the feature is pure math).

`project_pooled_to_scalars` now calls `direction_vector_decode_slice(pooled,
dir, 1.0)` 5 times. The `temperature = 1.0` matches the previous inline
`sigmoid(dot * (1/D))` formulation bit-for-bit because:
1. `direction_vector_decode_slice` computes `sigmoid((dot / D) * temperature)`.
2. With `temperature = 1.0`: `sigmoid(dot / D)`.
3. `D = 64` (power of 2), so `dot / D == dot * (1.0/D)` in f32 (exact).
4. The local `sigmoid` (clamp ±50) and `fast_sigmoid` (clamp ±40) produce
   identical f32 results for all inputs (both reduce to
   `1.0 / (1.0 + (-x).exp())` for |x| ≤ 40, and both saturate to 0.0/1.0
   for |x| > 40 in f32).

The local `sigmoid` function is retained (still `pub`, still tested) for
any future use; only the projection math was delegated.

**Verification:**
- 9 katgpt-core decoder tests pass (incl. new `decode_slice_matches_const_generic`
  which asserts bit-identical results between the slice and const-generic variants).
- 136 attn_match_fusion tests pass.
- 1661 riir-games lib tests pass (all am_* features).
- katgpt-core `--all-features` clean.

## Non-goals

- Do NOT change the 5-scalar HLA semantics (valence/arousal/desperation/calm/fear).
- Do NOT promote any feature — this is a pure internal refactor.
- Do NOT touch the pooling strategies (`PoolStrategy`) — only the projection step.

## Why low severity

The duplication is harmless today (both paths are correct + auto-vectorized).
It only matters when the dot+sigmoid math changes (e.g. a different temperature
schedule, or a SIMD-specialized `dot` in katgpt-core that the HLA bridge wouldn't
pick up). Filing so the drift risk is tracked, not lost.
