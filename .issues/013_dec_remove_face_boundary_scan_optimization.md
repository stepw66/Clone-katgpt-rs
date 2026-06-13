# Issue 013: DEC remove_face O(n) Boundary Scan Optimization

**Date:** 2026-06-13
**Plan:** 261 (Phase 0/3)
**Priority:** Low
**Type:** Optimization

## Problem

`CellComplex::remove_face()` currently uses `retain()` + linear scan for swap-rebind, which is O(total_boundary_entries) — not O(entries_per_face). On a 64×64 grid with ~16K B₂ entries, each `remove_face` takes ~230μs in debug mode, vs the plan's <10μs target.

## Root Cause

The sparse triplet boundary storage `Vec<(usize, usize, i8)>` doesn't maintain a reverse index from cell → boundary entry positions. Every removal scans the entire boundary vector twice:
1. `retain()` to remove entries referencing the target cell
2. Linear scan to rebind last-cell entries to the freed slot

## Benchmark Evidence (debug build, 64×64 grid)

| Faces Removed | Total Time | Per-Face |
|---|---|---|
| 1 | 229μs | 229μs |
| 10 | 2.4ms | 239μs |
| 100 | 23.3ms | 233μs |

## Proposed Solutions

1. **Reverse index**: Maintain `HashMap<cell_idx, Vec<entry_position>>` per boundary matrix. Removal becomes O(entries_per_cell).
2. **CSR format**: Switch from COO (sparse triplets) to CSR (compressed sparse row) for O(1) row access. More complex but standard.
3. **Deferred batch removal**: Accumulate removed faces, apply batch removal with a single pass.

## Non-Blocking

This is correct but slow. Game workloads typically destroy 1-10 faces per frame, so 230μs × 10 = 2.3ms is within frame budget for most games. The main consumer (riir-armageddon) uses a separate `Vec<bool>` grid for terrain, not DEC directly. DEC is the navigation layer.

## Acceptance Criteria

- [ ] `remove_face` < 10μs for 1 cell on 64×64 grid (release mode)
- [ ] No regression in operator correctness (d₁∘d₀ = 0)
