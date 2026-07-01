# Plan 317: Circulation Integral — Rank-2 Stokes Wrapper (Issue 005 Resolution)

**Date:** 2026-06-24
**Status:** **COMPLETE** — committed as `3a53b8e4` on `develop`. `circulation_integral` implemented + tested + benchmarked. G-C2 fails empirically (as predicted). Primitive is correct. `stokes_calculus` stays opt-in. Resolves issue 005 (closed + removed, resolved).
**Origin:** Issue 005 (Plan 314 Phase 3 G-C structural fail)
**Target:** `katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs` (+ bench + tests)
**Feature gate:** `stokes_calculus` (root alias for `katgpt-core/dec_operators`) — stays opt-in.

---

## Goal

Add the natural rank-2 Stokes-theorem companion to `line_integral`:
**`circulation_integral(cx, edge_field, closed_loop) -> f32`** — the discrete
circulation of a rank-1 edge field around a closed vertex loop.

By the Generalized Stokes' Theorem:

    ∮_∂A field = ∬_A curl(field) dA   ⟶   discrete: line_integral on a closed loop
                                            = Σ_faces curl(field)[f] over enclosed faces

This closes the rank-2 gap identified in Plan 314 G-C: `line_integral` on an
open path measures per-edge cost (cannot see turns); `circulation_integral` on
a closed loop measures **enclosed rotational content** (curl integrated over
area), which is the rank-2 Stokes quantity.

## Non-Goals

- ❌ NO new DEC operators (wraps existing `line_integral` + `exterior_derivative`).
- ❌ NO training, NO path-reranking policy (consumer's job — we provide the signal).
- ❌ NO modification to `CellComplex` (that's Issue 006's scope).

## Honest Pre-Implementation Analysis

Issue 005 proposed that `circulation_integral` could pass the G-C gate
("≥20% fewer direction reversals"). Mathematical analysis during planning
reveals this is **subtler than Issue 005 assumed**:

- Turn count is a **combinatorial** property of the path (changes in direction).
- Enclosed area (what `circulation_integral` measures on constant-curl fields) is
  a **geometric** property, **independent** of turn count.
- Concrete counterexample: an L-shaped path (1 turn) enclosing a full N×N square
  has MORE circulation than a staircase path (2N-1 turns) that cuts the corner
  and encloses ~N²/2. So minimizing circulation can prefer MORE turns.

**Therefore**: G-C "≥20% fewer reversals" is likely **still unreachable** via
`circulation_integral` alone. However, the primitive is **correct, useful, and
the natural rank-2 Stokes companion**. The right outcome is:

1. Implement `circulation_integral` (correct, ~15 LOC wrapper).
2. Verify correctness with 3 unit tests (Stokes identities).
3. Run the G-C benchmark HONESTLY — report whether `circulation_integral`-based
   selection reduces turns (it may not, and that's a valid finding).
4. Promote `stokes_calculus` to default-on IF G-B's win (5.36×) stands on its
   own merit, regardless of G-C. The three primitives are all correct; the
   feature being opt-in was due to G-C failing, but G-B passing alone may
   justify promotion (per Plan 314's split rule re-evaluation).

## Tasks

- [x] **T1** Create this plan file.
- [x] **T2** Implement `circulation_integral(cx, edge_field, closed_loop) -> f32`
      in `stokes_calculus.rs`. Thin wrapper over `line_integral` (closed loop's
      line integral IS its circulation). Debug-assert the loop is closed
      (first == last vertex). ~15 LOC.
- [x] **T3** Add 3 unit tests (Phase 2 style):
  - [x] **T3.1** Zero-curl (exact/gradient) field → zero circulation (FTC).
  - [x] **T3.2** Constant-curl (rigid rotation) field → circulation = curl × area.
        Cross-checked against `exterior_derivative` (Stokes identity). PASS.
  - [x] **T3.3** Reversal antisymmetry: clockwise == −counterclockwise. PASS.
- [x] **T4** Add `circulation_integral` to `dec/mod.rs` re-exports.
- [x] **T5** Run unit tests: 15/15 pass (12 existing + 3 new).
- [x] **T6** Add G-C benchmark variant using `circulation_integral` on closed loops.
- [x] **T7** Run the benchmark; results in `.benchmarks/317_circulation_integral_goat.md`.
      **Empirical result: smooth=128/3turns vs zigzag=112/25turns → minimizing
      circulation INCREASES turns → G-C2 FAILS as predicted.**
- [x] **T8** Promotion decision: `stokes_calculus` stays opt-in.
  - G-C2 fails empirically (circulation ≠ turn count).
  - `dec_operators` itself is opt-in (bigger decision).
  - G-A already FAILED in riir-ai Plan 334 (9.5× slower, 36% lower F1). All 3 gates resolved: only G-B won.
- [x] **T9** Update Issue 005 with resolution notes (CLOSED).
- [x] **T10** Update Plan 314 G-C section with the rank-2 finding (T3.3 cross-ref added).
- [x] **T11** Commit on `develop` with `feat:` prefix per global rules. **Committed as `3a53b8e4`** (8 files, +662/-15).

## Architecture

```
katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs
    pub fn circulation_integral(cx, edge_field, closed_loop) -> f32
    // ↑ thin wrapper: debug_assert!(closed); line_integral(cx, edge_field, closed_loop)
    mod tests  // +3 tests (T3.1, T3.2, T3.3)
```

No new files except the benchmark result doc.

## Validation

- [x] `cargo test -p katgpt-core --features dec_operators --lib dec::stokes_calculus` — **15 passed** (12 existing + 3 new), 0 failed.
- [x] `cargo check --features stokes_calculus` — clean.
- [x] `cargo check --all-features` — **EXIT 0** (no regression to Issue 004's fix).
- [x] `cargo bench -p katgpt-core --features dec_operators --bench stokes_calculus_bench` — runs clean (G-C2 results in `.benchmarks/317_circulation_integral_goat.md`).
- [x] Files < 2048 lines — `stokes_calculus.rs` stays well under (~750 LOC).

## Honest Risk Notes

- **G-C may still fail.** `circulation_integral` measures enclosed curl, not turn
  count. These are independent geometric properties. The primitive is correct
  regardless; the gate's framing ("fewer reversals") may be the wrong metric.
- **Promotion decision is separate from G-C.** Even if G-C fails, `stokes_calculus`
  may be promoted based on G-B's standalone win (5.36× boundary-flux speedup).
  The opt-in status was a conservative choice, not a correctness requirement.
- **The primitive has standalone value**: rotational-content detection, vortex
  detection, Stokes-theorem-correct circulation for any caller who needs it.
