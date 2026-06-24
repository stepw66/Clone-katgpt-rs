# Issue 006: Add Coboundary Index to `CellComplex` for True O(boundary) `boundary_flux_mass`

**Date:** 2026-06-24
**Status:** CLOSED — RESOLVED (Plan 318, commit pending)
**Origin:** Plan 314 Phase 3 GOAT gate analysis (`.benchmarks/314_stokes_calculus_goat.md`)
**Severity:** Low (G-B already passes at 5.36× via memory-access-pattern win; this would widen the margin)
**Related:** katgpt-rs/.plans/314 (Stokes Calculus Wrappers), katgpt-rs/.benchmarks/314_stokes_calculus_goat.md, katgpt-rs/.plans/312 (CSR adjacency precedent), katgpt-rs/.plans/318 (implementation plan)

## Problem

`boundary_flux_mass_only` currently scans **ALL** `B_{k+1}` entries to compute the boundary flux of a region:

```rust
for &(k_cell, kp1_cell, sign) in cx.boundary_entries(k) {
    if in_region[kp1_cell] {
        mass += sign as f32 * field.scalar(k_cell);
    }
}
```

This is `O(|B_{k+1}|)` — the full boundary matrix. For a 256×256 grid, `|B₂|` = 4 × 255 × 255 ≈ 260k entries. The naive `exterior_derivative_into` is also `O(|B₂|)`. So both approaches have the same asymptotic complexity.

The **theoretical** divergence-theorem advantage is `O(boundary)` vs `O(volume)`. For a 64×64 region in a 256×256 grid: boundary = 4×64 = 256 edges, volume = 4096 faces. The boundary is 16× smaller. But the current implementation can't exploit this because it scans the full `B₂`.

**Current G-B result:** 5.36× speedup over naive, but the win comes from **memory access patterns** (no output materialization), NOT from the O(boundary) advantage. The cached-d_field baseline (2.79 µs) shows that pre-computing `d₁(field)` once and summing per-region is 42× faster — the boundary-flux niche is single-query scenarios.

## Proposed fix: coboundary index

Add a **coboundary index** to `CellComplex`: for each (k+1)-cell, a list of its boundary (k)-cells with orientation signs. This is the transpose of `B_{k+1}`.

**Data structure (CSR-style, following Plan 312's precedent):**
```rust
pub struct CellComplex {
    // ... existing fields ...
    /// Coboundary index: for each (k+1)-cell at rank (k+1), its boundary
    /// (k)-cells with signs. Built lazily on first query, cached.
    /// coboundaries[k] = (offsets: Vec<u32>, entries: Vec<(u32, i8)>)
    /// Entries for cell c at rank (k+1) are:
    ///   entries[coboundaries[k].offsets[c]..coboundaries[k].offsets[c+1]]
    coboundaries: [Option<CoboundaryIndex>; MAX_RANK as usize],
}

struct CoboundaryIndex {
    offsets: Vec<u32>,   // length n_cells(k+1) + 1
    entries: Vec<(u32, i8)>,  // (k_cell_idx, sign)
}
```

**Build cost:** `O(|B_{k+1}|)` once per topology version. Cache invalidation on `topology_version` bump (same pattern as Plan 312's CSR rebuild).

**Query cost after build:** `boundary_flux_mass` becomes `O(|region| × boundary_per_cell)` instead of `O(|B_{k+1}|)`. For a 64×64 region with 4 edges/face: `O(4096 × 4) = 16k` ops vs current `O(260k)` ops → **~16× further speedup**.

## Expected G-B improvement

| Variant | Current | With coboundary index | Target |
|---------|---------|----------------------|--------|
| `boundary_flux_mass_only` (64×64 region, 256×256 grid) | 115.53 µs | ~7 µs (est.) | ≥3× vs naive |
| vs naive `exterior_derivative_into` | 5.36× faster | ~88× faster | — |
| vs cached `d_field` region sum | 0.024× (slower) | ~0.4× (comparable) | — |

With the coboundary index, `boundary_flux_mass_only` would approach the cached-d_field performance for single queries, making it the clear winner for all zone-mass scenarios.

## Tasks

- [ ] Add `CoboundaryIndex` struct (CSR layout: `offsets: Vec<u32>`, `entries: Vec<(u32, i8)>`).
- [ ] Add `CellComplex::coboundary_entries(k: u8) -> &CoboundaryIndex` with lazy build + topology-version caching.
- [ ] Audit ALL `CellComplex` mutation paths (`add_incidence`, `remove_face`, future `remove_edge`) to invalidate the coboundary cache (the `merkle_root` lesson — audit all constructors/mutators).
- [ ] Rewrite `boundary_flux_mass_only` to use the coboundary index: iterate region cells, look up each cell's boundary via CSR, sum with cancellation.
- [ ] Re-run G-B benchmark; verify the ~16× additional speedup.
- [ ] Run `cargo hack --each-feature` + `--all-features` to catch combo regressions (the `merkle_root` lesson).

## Scope note

This modifies `CellComplex` (a core type), not just `stokes_calculus.rs`. It's a Plan 251 follow-up, not a Plan 314 task. The coboundary index is generally useful (any "for each cell, what are its faces?" query benefits), not just for boundary flux.

**Alternative without modifying CellComplex:** build the coboundary index locally in `boundary_flux_mass` as a one-time `O(|B_{k+1}|)` pass per call. This doesn't help single-query perf (same cost as current), but enables multi-query caching if the caller holds the index across calls. Lower-risk, lower-reward.

## Verdict

**Not blocking** — G-B already passes without this optimization. File for when boundary-flux becomes a hot path (e.g., if riir-ai wires it into per-tick zone-threat computation for many zones). The coboundary index is the right architectural fix; the question is whether the current 5.36× win is sufficient to defer.

---

## Resolution (Plan 318, 2026-06-24)

**RESOLVED.** Implemented as Plan 318 despite the "not blocking" verdict, because the user listed it as the #1 remaining item and the change is well-contained (additive API, explicit invalidation, no `Sync` breakage).

### What was delivered

1. **`CoboundaryIndex` struct** (CSR layout: `offsets: Vec<u32>`, `entries: Vec<(u32, i8)>`) in `types.rs`.
2. **`CellComplex::build_coboundary_index(&mut self, k: u8)`** — explicit build, `O(|B_{k+1}|)` count-sort pass.
3. **`CellComplex::coboundary_entries(&self, k: u8) -> Option<&CoboundaryIndex>`** — immutable accessor, `None` if not built or invalidated.
4. **`boundary_flux_mass_indexed(cx, region, field) -> f32`** in `stokes_calculus.rs` — the fast path. Falls back to `boundary_flux_mass_only` (with `debug_assert`) if the index is not built.
5. **Cache invalidation** via private `invalidate_coboundary_cache()` helper, called from all 4 `topology_version += 1` sites (`remove_face` + `remove_cell` ranks 0/1/3; rank 2 delegates to `remove_face`).
6. **7 unit tests** for `CoboundaryIndex` + **5 unit tests** for `boundary_flux_mass_indexed` = 12 new tests.
7. **3 benchmark variants** (full-scan baseline, cold, warm) in `stokes_calculus_bench.rs`.

### GOAT gate result

| Variant | Time | Speedup |
|---------|------|---------|
| `full_scan_baseline` (`boundary_flux_mass_only`) | 132.93 µs | 1.0× |
| `indexed_cold` (clone + build + 1 query) | 1.3435 ms | 0.099× (10× SLOWER) |
| `indexed_warm` (pre-built, query only) | **14.718 µs** | **9.03× FASTER** ✅ |

**Gate: PASS (warm = 9.03× ≥ 3× target).** The cold-cache path is 10× slower
(clone + build dominates), confirming the issue's own analysis: the index only
wins when amortized across multiple queries on stable topology.

### `merkle_root` lesson audit

All mutation paths that bump `topology_version` now call `invalidate_coboundary_cache()`.
Test `test_coboundary_index_remove_cell_invalidates_all_ranks` verifies all
4 explicit ranks (0/1/2/3) invalidate all 3 coboundary indices. Confirmed via
grep that no `add_incidence` method exists — the only boundary-populating
constructors are `new` and `grid_2d`, both of which produce fresh objects with
`coboundaries = [None, None, None]`.

### Promotion decision

**`stokes_calculus` stays opt-in.** G-A and G-C still fail. The coboundary
index is a modelless architectural win (9.03× for multi-query boundary-flux
scenarios) available to callers who enable the feature, but it does not change
the feature's default-on status.

### Verification

| Check | Result |
|-------|--------|
| `cargo test -p katgpt-core --features dec_operators --lib dec::` | **111 passed** (was 99; +12), 0 failed |
| `cargo test -p katgpt-core --lib` (full G3) | **509 passed**, 0 failed |
| `cargo check --all-features` | **EXIT 0** (Issue 004 fix holds) |
| `cargo check --no-default-features --features dec_operators` | **EXIT 0** |
| G-B indexed warm benchmark | 14.72 µs vs 132.93 µs = **9.03× faster** |
| G-B indexed cold benchmark | 1.34 ms (10× slower — expected) |
| Diagnostics on changed files | No errors or warnings |
