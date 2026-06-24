# Plan 318: Coboundary Index for True O(boundary) `boundary_flux_mass`

**Date:** 2026-06-24
**Status:** In Progress
**Origin:** Issue 006 (`.issues/006_coboundary_index_for_boundary_flux.md`)
**Parent:** Plan 314 (Stokes Calculus Wrappers), Plan 251 (DEC Operators / CellComplex)
**Severity:** Low — G-B already passes at 5.36×; this widens the margin for multi-query scenarios.

## Goal

Add a CSR-style **coboundary index** to `CellComplex`: for each (k+1)-cell, the
list of its boundary (k)-cells with orientation signs (the transpose of
`B_{k+1}`). This lets `boundary_flux_mass` iterate region cells and look up each
cell's boundary directly, instead of scanning the full `B_{k+1}` with a
region-membership filter.

**Current cost:** `O(|B_{k+1}|)` per query — full boundary matrix scan.
**With index (warm):** `O(|region| × boundary_per_cell)` — for a 64×64 region on
a 256×256 grid, `4096 × 4 = 16k` ops vs `260k` ops → ~16× theoretical speedup.

## Honest scope assessment (pre-implementation)

**No current production consumer** of `boundary_flux_mass` exists — only tests
and benchmarks. G-B already passes at 5.36× via memory-access-pattern
optimization (no output materialization). The coboundary index is therefore an
**architectural improvement** that pays off only when:

1. A caller does **many boundary-flux queries** on the **same topology** (warm
   cache), AND
2. Each query is a **small region** in a **large complex** (so `|region| × 4`
   ≪ `|B_{k+1}|`).

For a **single** query, the build cost (`O(|B_{k+1}|)`) dominates and the index
makes things **slower**. The benchmark MUST measure both cold (build + query)
and warm (query only) costs honestly.

This is not a premature optimization in the architectural sense — the CSR index
is generally useful for any "for each (k+1)-cell, what are its faces?" query,
and it's the right fix when boundary-flux becomes a hot path (e.g., riir-ai
per-tick zone-threat computation for many zones). It IS premature in the sense
that no current caller benefits. We implement it anyway because:
- The user explicitly asked (Issue 006 is the #1 remaining item).
- The change is well-contained and low-risk (additive API, explicit invalidation).
- It unblocks future multi-query consumers without further `CellComplex` churn.

## Design constraints (non-negotiable, per AGENTS.md)

- **Modelless:** YES — pure transpose + CSR scan, no learning.
- **No `Sync` breakage:** `CellComplex` must remain `Send + Sync`. No `RefCell`.
  Use **explicit build** (`build_coboundary_index(&mut self, k)`) instead of
  lazy build-on-first-query.
- **`merkle_root` lesson:** audit ALL mutation paths that bump
  `topology_version` — there are exactly 5: `remove_face`, and `remove_cell`
  for ranks 0/1/2/3. All must invalidate the coboundary cache.
- **No `add_incidence`:** confirmed by grep — does not exist. Only `grid_2d`
  (constructor, fresh object) and `new` (constructor) populate boundaries; both
  create fresh objects with `coboundaries = [None, None, None]`.
- **Additive API:** existing `boundary_flux_mass_only` keeps its behavior
  (fallback full-scan). New `boundary_flux_mass_indexed` requires pre-built
  index and uses the fast path.

## API design

```rust
/// CSR-style coboundary index for rank k→(k+1).
///
/// For each (k+1)-cell `c`, its boundary (k)-cells are:
///   entries[offsets[c]..offsets[c+1]]
/// Each entry is `(k_cell_idx, sign)`.
#[derive(Clone, Debug)]
pub struct CoboundaryIndex {
    offsets: Vec<u32>,          // length n_cells(k+1) + 1
    entries: Vec<(u32, i8)>,    // (k_cell_idx, sign)
}

impl CellComplex {
    /// Build (or rebuild) the coboundary index for rank k→(k+1).
    /// Cost: O(|B_{k+1}|) + O(n_cells(k+1)) sort. Cached until next mutation.
    pub fn build_coboundary_index(&mut self, k: u8) { ... }

    /// Access the pre-built coboundary index for rank k→(k+1).
    /// Returns `None` if `build_coboundary_index(k)` was not called (or the
    /// cache was invalidated by a topology mutation since then).
    #[inline]
    pub fn coboundary_entries(&self, k: u8) -> Option<&CoboundaryIndex> { ... }
}
```

**Cache invalidation:** all 5 `topology_version += 1` sites set
`self.coboundaries = [None, None, None]`. Simple, correct, no per-rank tracking
needed (a mutation at rank 0 can affect B₁ which affects the k=0 coboundary
index, etc.).

## GOAT gate (G-B extension)

The gate: **warm-cache** `boundary_flux_mass_indexed` must be ≥3× faster than
the current `boundary_flux_mass_only` (which itself is 5.36× faster than naive).

| Variant | Expected (warm) | Gate |
|---------|-----------------|------|
| `boundary_flux_mass_only` (current winner) | 115 µs | baseline |
| `boundary_flux_mass_indexed` (cold: build + 1 query) | ~120 µs (build ≈ scan) | SLOWER — expected |
| `boundary_flux_mass_indexed` (warm: query only) | ~7 µs (est.) | ≥3× vs `boundary_flux_mass_only` |

Promotion: **`stokes_calculus` stays opt-in.** G-A and G-C still fail. The
coboundary index widens G-B but doesn't change the feature's promotion status.

## Tasks

- [x] T1: Add `CoboundaryIndex` struct (CSR: `offsets`, `entries`) to `types.rs`.
- [x] T2: Add `coboundaries: [Option<CoboundaryIndex>; 3]` field to `CellComplex`.
      Initialize to `[None, None, None]` in `new` and `grid_2d`.
- [x] T3: Implement `CellComplex::build_coboundary_index(&mut self, k: u8)`:
      count-sort pass over `boundaries[k]` to build CSR. O(|B_{k+1}|).
- [x] T4: Implement `CellComplex::coboundary_entries(&self, k: u8) -> Option<&CoboundaryIndex>`.
- [x] T5: **`merkle_root` audit** — added `invalidate_coboundary_cache()` private helper
      called from all 4 `topology_version += 1` sites (remove_face + remove_cell
      ranks 0/1/3; rank 2 delegates to remove_face → covered). Test
      `test_coboundary_index_remove_cell_invalidates_all_ranks` verifies all ranks.
- [x] T6: Add unit tests for `CoboundaryIndex` (7 tests):
      - T6.1: `test_coboundary_index_not_built_returns_none`
      - T6.2: `test_coboundary_index_b2_correct_csr` (cross-checks vs B₂)
      - T6.3: `test_coboundary_index_b1_correct_csr` (2 vertices/edge, ±1 signs)
      - T6.4: `test_coboundary_index_remove_face_invalidates`
      - T6.5: `test_coboundary_index_remove_cell_invalidates_all_ranks` (merkle_root audit)
      - T6.6: `test_coboundary_index_rebuild_after_mutation`
      - T6.7: `test_coboundary_index_build_panics_at_max_rank` (should_panic)
- [x] T7: Implement `boundary_flux_mass_indexed(cx, region_cells, field) -> f32`
      in `stokes_calculus.rs`. Falls back to `boundary_flux_mass_only` (with
      debug_assert) if index not built.
- [x] T8: Add unit tests for `boundary_flux_mass_indexed` (5 tests):
      - T8.1: matches `boundary_flux_mass_only` on full region.
      - T8.2: Stokes identity on subset region (cross-check vs naive + full-scan).
      - T8.3: empty region → 0.0.
      - T8.4: exact/gradient field → zero flux (FTC).
      - T8.5: rebuild after mutation matches full-scan reference.
- [x] T9: Re-export `boundary_flux_mass_indexed` and `CoboundaryIndex` from `dec/mod.rs`.
- [x] T10: Add benchmark variants to `stokes_calculus_bench.rs`:
      - `G-B_256x256_full_scan_baseline` (re-baseline in same group)
      - `G-B_256x256_indexed_cold` (clone + build + 1 query per iter)
      - `G-B_256x256_indexed_warm` (pre-built, query only)
- [x] T11: Run GOAT gate. **PASS: warm = 14.72 µs vs baseline 132.93 µs = 9.03× faster**
      (gate was ≥3×). Cold = 1.34 ms (10× slower — expected, build+clone dominates).
- [x] T12: Run `cargo test -p katgpt-core --features dec_operators --lib dec::`
      → **111 passed** (was 99; +12 new), 0 failed. G3 regression clear.
- [x] T13: Run `cargo check --all-features` → **EXIT 0** (Issue 004 fix holds).
      Also `--no-default-features --features dec_operators` → EXIT 0.
- [x] T14: Update Issue 006 with resolution notes. Close it.
- [x] T15: Update `.benchmarks/314_stokes_calculus_goat.md` with indexed results.
- [x] T16: Commit on `develop`.

## Risk register

- **Risk:** CSR build cost dominates for single-query. **Mitigation:** honest
  cold-cache benchmark; document that warm-cache is the intended use case.
- **Risk:** `merkle_root`-style bug (forget to invalidate one path). **Mitigation:**
  T5 enumerates all 5 paths explicitly; T6.3 tests invalidation.
- **Risk:** `boundary_flux_mass_indexed` diverges from `boundary_flux_mass_only`
  on some edge case. **Mitigation:** T8.1/T8.2 cross-check Stokes identity.
