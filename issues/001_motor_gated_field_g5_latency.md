# Issue 001: Close G5 latency gap for motor_gated_field (120 µs → < 100 µs)

**Date:** 2026-07-01
**Resolved:** 2026-07-01 (grid-stencil fast path)
**Source:** Plan 357 Phase 2 GOAT gate — G5 FAIL (borderline)
**Primitive:** `evolve_motor_gated_field` (`katgpt-dec/src/motor_gated.rs`)
**Feature:** `motor_gated_field` (opt-in)

## Resolution

**CLOSED — G5 now PASS at ~29 µs (4.1× speedup, 3.4× margin under 100 µs).**

The gap was closed by Option 2 (re-diagnosed): a **grid-stencil fast path**
for `graph_laplacian_into`. The fix is NOT one of the four options originally
listed — it's a fifth path that emerged from re-diagnosing the root cause.

### What changed

- Added `grid_dims: Option<(usize, usize)>` to `CellComplex` (set by `grid_2d`,
  cleared by every topology mutation per the `merkle_root` lesson).
- `graph_laplacian_into` now dispatches to `graph_laplacian_grid_into` when
  `cx.grid_dims()` is `Some` — a 5-point-stencil path that iterates vertices
  row-major and writes each output element exactly once.
- Equivalence verified by `graph_laplacian_grid_matches_edge_list_{1ch,multich}`
  tests (max diff < 1e-4).

### Why the original diagnosis was wrong

The original analysis attributed the 120 µs latency to "DRAM bandwidth for a
12 MB working set". That was based on an **arithmetic error**: `64×64×16` was
computed as `1,048,576` floats (1024×1024) instead of the correct `65,536`
(64×64×16). The actual 3-array working set is **768 KB** (not 12 MB), which
fits comfortably in L2. The real bottleneck was the **scattered
read-modify-write pattern** in the edge-list `graph_laplacian_into` — each
vertex is touched `degree(v)` times (4× for interior), and the touches hit
different cache lines. The grid stencil eliminates this by converting to a
sequential read-once-write-once stream.

### Why the four original options were not needed

1. **f16 field support** — not needed; the bottleneck was access pattern, not
   bandwidth. Halving precision would have been a poor tradeoff for a non-issue.
2. **Tiled graph Laplacian** — superseded; the grid stencil IS the cache-friendly
   path, and it's simpler than tiling (no halo exchange).
3. **Single-scratch fusion** — not needed; the algorithm's 2-scratch structure
   is correct and the access pattern fix was the real lever.
4. **Relax G5 target to 150 µs** — not needed; the real fix achieved 29 µs.

## Problem (original, pre-fix)

The G5 latency gate for `evolve_motor_gated_field` measures **~120 µs** on a
64×64×16 field (1,048,576 floats) vs the **< 100 µs** target — a 1.2× miss.
G1–G4 all pass; only G5 fails.

Per Plan 357's promotion rule, this blocks marking `motor_gated_field` "ready
for downstream consumption" (riir-ai Research 168 Phase 2). The primitive stays
opt-in and correct; this is purely a perf gap.

## Root cause (original diagnosis — SUPERSEDED, see Resolution above)

**Memory-bandwidth bound, not compute bound.** The 3-array working set (field
`h` + `scratch_relu` + `scratch_lap`) = 12 MB, with ~28 MB of read+write traffic
per tick. At 120 µs the effective bandwidth is ~233 GB/s — already near the
single-core L2/L3 ceiling. See `.benchmarks/357_motor_gated_field_goat.md` for
the full traffic analysis.

Two SIMD optimizations were tried and did NOT help (see benchmark doc):
- Fused relu-on-read in the laplacian stencil: 134 µs (slower — scattered reads).
- 8-wide chunked blend loop: no improvement (iterator-zip already vectorizes).

## Options to close the gap (original list — SUPERSEDED by the grid stencil)

1. **f16 field support** — halve memory traffic (4 MB → 2 MB per array). The
   `half` crate is already a katgpt-rs dep; needs f16 variants of the DEC
   operators (`graph_laplacian_into`, `relu_gate_into`). Estimated: 120 µs →
   ~65 µs (bandwidth-limited at half the traffic). **Highest expected payoff.**

2. **Tiled graph Laplacian** — process the grid in cache-resident tiles
   (e.g. 16×16×16 = 16 KB per tile, fits L1). Requires a tiled/halo-exchange
   variant of `graph_laplacian_into` (not currently shipped). Complex.

3. **Single-scratch fusion** — eliminate one scratch array via in-place relu on
   `h` with snapshot/restore. Fragile; risks the clean split-step semantics.

4. **Relax G5 target to 150 µs** — the 100 µs target was aspirational ("vs the
   paper's GPU ~ms conv"). 120 µs is already 8× under the paper's baseline and
   fast enough for the offline-rehearsal use case (1000 ticks = 120 ms).

## Recommendation (original — SUPERSEDED)

Option 1 (f16) if a downstream consumer hits the latency budget; otherwise
option 4 (relax target). Not a blocker — the primitive is correct, zero-alloc,
and fast in absolute terms.

## Acceptance (original)

G5 measures < 100 µs (or the target is formally relaxed to 150 µs with a
documented rationale), and `.benchmarks/357_motor_gated_field_goat.md` is
updated with the closing result.

**Acceptance MET:** G5 measures ~29 µs (< 100 µs), benchmark doc updated.
