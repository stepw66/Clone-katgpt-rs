# Issue 005: Stokes Calculus G-C — `line_integral` Cannot Encode Turn Penalties (needs rank-2 wrapper)

**Date:** 2026-06-24
**Status:** Open — structural limitation, blocks `stokes_calculus` promotion G-C gate
**Origin:** Plan 314 Phase 3 GOAT gate (benchmark `.benchmarks/314_stokes_calculus_goat.md`)
**Severity:** Low (the primitive is correct and useful as a path-cost function; only the "smoothness" framing is wrong)
**Related:** katgpt-rs/.plans/314 (Stokes Calculus Wrappers), katgpt-rs/.benchmarks/314_stokes_calculus_goat.md

## Problem

The Plan 314 G-C gate target ("≥20% fewer direction reversals via `line_integral`-weighted path reranking") is **structurally unreachable** for `line_integral` on a rank-1 (edge) cochain.

**Root cause:** A rank-1 edge cochain assigns one scalar per edge. `line_integral` sums these scalars along a path. Turn penalties depend on the **angle between consecutive edges** — a path-level property that cannot be expressed as a sum of per-edge scalars. Mathematically:

- `line_integral(path) = Σ_e sign(e, path) · field[e]`
- Turn penalty would require `Σ_{(e_i, e_{i+1}) ∈ path} penalty(angle(e_i, e_{i+1}))` — a **pairwise** edge term.
- A rank-1 cochain has no way to encode pairwise edge interactions.

**Evidence from the G-C benchmark:**
- Smooth path (1 turn) and zigzag path (29 turns) between the same endpoints have DIFFERENT `line_integral` values (2.231 vs 0.359) — but this difference comes from **which edges** they traverse (spatially varying non-exact field), NOT from the turn count.
- On a uniform field, both paths give identical `line_integral` regardless of turn count.
- On an exact (gradient) field, both paths give identical `line_integral` by the fundamental theorem of calculus (path-independence).

## Why this is not a bug

`line_integral` is **correct** — it faithfully computes the discrete line integral of a rank-1 cochain. The 4 Phase-2 unit tests all pass (straight path, reversal antisymmetry, closed-loop-of-exact-field = 0, short path). The issue is that the Plan 314 G-C target was based on a **misclassification** of what rank-1 cochains can express. The plan's risk note ("G-C may fail if manifold_geodesic paths are already near-optimal") understated the problem — the failure is more fundamental than path optimality.

## `line_integral` is still useful

The primitive remains a valid **path-cost function** for:
- Path energy / work computation (Σ per-edge cost)
- Terrain-friction accumulation along a route
- Comparing the cost of two candidate paths (it correctly discriminates on non-exact fields)
- Composing with `manifold_geodesic` output as a post-hoc cost label

It just cannot serve as a **smoothness/reversal regularizer**.

## Proposed fix: rank-2 `circulation_integral` wrapper

The natural Stokes-theorem companion to `line_integral` (rank-1, ∫ over 1-paths) is a **rank-2 circulation integral** (∮ over closed loops, integrating curl over enclosed area). This CAN encode turn penalties because:

- A path with many turns encloses more area (in the sense of the signed area between the path and the straight-line shortcut) than a smooth path.
- The circulation `∫_loop field = ∫_area curl(field)` by Stokes' theorem.
- Turn penalties emerge naturally as the curl integrated over the "detour area."

**Sketch:**
```rust
/// Circulation of a rank-1 edge field around a closed vertex loop.
/// Equals the integral of curl(field) over the enclosed area (Stokes).
/// Non-zero for rotational fields; zero for exact (gradient) fields.
pub fn circulation_integral(cx: &CellComplex, edge_field: &CochainField, closed_loop: &[u32]) -> f32 {
    // = line_integral(cx, edge_field, closed_loop) since the loop is closed.
    // But the INTERPRETATION differs: this measures enclosed curl, not path energy.
    // For turn-smoothness: compare circulation_integral of a candidate path's
    // "closure" (path + straight-line return) — smooth paths enclose less area.
}
```

This is a ~20 LOC wrapper, same complexity class as the existing primitives. It composes with `line_integral` (a closed loop's line_integral IS its circulation).

## Tasks

- [ ] Implement `circulation_integral(cx, edge_field, closed_loop) -> f32` in `stokes_calculus.rs` (~20 LOC, delegates to `line_integral` for the closed loop).
- [ ] Add 3 unit tests: zero-curl field → zero circulation; constant-curl field → circulation = curl × enclosed area; reversal antisymmetry (clockwise vs counterclockwise).
- [ ] Re-run G-C benchmark with `circulation_integral` as the smoothness metric (smooth path encloses less area than zigzag path).
- [ ] If G-C passes with `circulation_integral` → update Plan 314 G-C target to use the rank-2 wrapper; consider promoting `stokes_calculus` to default-on.

## Verdict

**Keep `stokes_calculus` opt-in** until either:
1. `circulation_integral` is added and G-C passes with it, OR
2. G-A (Fokker-Planck validator, deferred to riir-ai) passes and becomes the headline application.

The three existing primitives (`belief_mass_divergence`, `boundary_flux_mass`, `line_integral`) are all correct and available to callers who want them. The opt-in status reflects that the GOAT gate didn't fully clear, not that the code is broken.
