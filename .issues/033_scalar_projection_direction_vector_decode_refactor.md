# Issue 033 — Refactor riir-games `scalar_projection` to call katgpt-core `direction_vector_decode`

**Filed from:** Plan 330 (Analytic Lattice Encoder/Decoder Primitive), T3.4
**Date:** 2026-07-02
**Type:** refactor / decoupling
**Severity:** low (cleanup; no behavior change)
**Scope:** `riir-ai/crates/riir-games/src/attn_match_fusion/scalar_projection.rs`

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

- [ ] `project_pooled_to_scalars` calls the katgpt-core primitive (no inline dot+sigmoid).
- [ ] Existing `am_*` GOAT gates still PASS (attn_match_fusion is default-on:
      am_core, am_cross_game, am_self_play, am_trajectory_compaction).
- [ ] No perf regression — the zip-based dot already auto-vectorizes; the
      delegated call must match (verify G2 latency in `bench_297_*` is stable).
- [ ] `direction_vector_decode` stays the single source of truth for the math.

## Non-goals

- Do NOT change the 5-scalar HLA semantics (valence/arousal/desperation/calm/fear).
- Do NOT promote any feature — this is a pure internal refactor.
- Do NOT touch the pooling strategies (`PoolStrategy`) — only the projection step.

## Why low severity

The duplication is harmless today (both paths are correct + auto-vectorized).
It only matters when the dot+sigmoid math changes (e.g. a different temperature
schedule, or a SIMD-specialized `dot` in katgpt-core that the HLA bridge wouldn't
pick up). Filing so the drift risk is tracked, not lost.
