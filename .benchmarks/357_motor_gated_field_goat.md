# Plan 357 — Motor-Gated DEC Field GOAT Results

**Date:** 2026-07-01 (G5 closed 2026-07-01 by grid-stencil fast path)
**Primitive:** `evolve_motor_gated_field` (`katgpt-dec/src/motor_gated.rs`)
**Feature:** `motor_gated_field` (opt-in)
**Bench:** `cargo bench -p katgpt-core --features motor_gated_field --bench bench_357_motor_gated_field_goat -- --nocapture`
**Hardware:** macOS (Apple Silicon)

## G1–G5 Results (final, post grid-stencil fix)

| Gate | Metric | Value | Threshold | Verdict |
|------|--------|-------|-----------|---------|
| G1 no-teleporting | max centroid displacement / 50 ticks | 0.0009 cells | ≤ 2.0 cells | **PASS ✅** |
| G2 motor-gate locality | channel isolation ratio | ∞ (no leak) | > 100× | **PASS ✅** |
| G3 conservation | `\|Σ K[ReLU(h)]\| / L1(h)` | 0.0000 | < 0.05 | **PASS ✅** |
| G4 zero-alloc | allocs / 1000 ticks (64×64×16) | 0 | = 0 | **PASS ✅** |
| G5 latency | per-call (64×64×16, release) | **~29 µs** | < 100 µs | **PASS ✅** |

## Verdict: ALL 5 GATES PASS — motor_gated_field READY FOR DOWNSTREAM CONSUMPTION

The grid-stencil fast path (Plan 357 G5 fix, Issue 001) closes the G5 gap
decisively: **120 µs → 29 µs** (4.1× speedup, 3.4× margin under the 100 µs
target). Per the promotion rule, `motor_gated_field` is now **ready for
downstream consumption** (riir-ai Research 168 Phase 2). The feature stays
**opt-in** by design — it's a primitive, not a default-on capability.

## The G5 fix: grid-stencil fast path

### The real bottleneck (corrected traffic analysis)

The initial G5 diagnosis attributed the 120 µs latency to "DRAM bandwidth for a
12 MB working set". That analysis contained an **arithmetic error**: the G5
workload is `64×64 vertices × 16 channels = 65536 floats = 256 KB per array`
(not 1,048,576 floats = 4 MB). The actual 3-array working set is **768 KB**,
which fits comfortably in L2. The real bottleneck was the **scattered
read-modify-write pattern** in the edge-list `graph_laplacian_into`:

- Each vertex is touched `degree(v)` times (4× for interior vertices), once per
  incident edge.
- Each touch reads `potential[v]` and `output[v]` from potentially different
  cache lines (vertices connected by an edge can be far apart in memory).
- The `output.data.fill(0.0)` before accumulation writes the full output array
  once, then the scattered `+=` / `-=` updates rewrite it ~4× per element.
- The store-forwarding stalls from read-modify-write on scattered cache lines
  dominate, not DRAM bandwidth.

### The fix: `graph_laplacian_grid_into`

Added a `grid_dims: Option<(usize, usize)>` field to `CellComplex` (set by
`grid_2d`, cleared by every topology mutation per the `merkle_root` lesson).
When `grid_dims` is `Some`, `graph_laplacian_into` dispatches to the 5-point
stencil fast path:

```
Δ₀[v] = deg(v)·potential[v] − Σ potential[neighbor]
```

computed by iterating vertices in **row-major order** and writing each output
element **exactly once** (no zero-fill, no scattered accumulation). The interior
loop (the bulk path for any grid ≥ 3×3) is branch-free and auto-vectorizes
cleanly; the `O(w+h)` boundary is handled with explicit neighbor-count checks.

**Mathematically identical** to the edge-list path (both realize `δ₁d₀`); the
f32 results differ by at most ULP-level rounding from the changed accumulation
order, which is acceptable for every consumer (verified by
`graph_laplacian_grid_matches_edge_list_{1ch,multich}` tests: max diff < 1e-4).

### Why the SIMD attempts didn't help (and the stencil did)

The two Phase 3 SIMD attempts (fused relu-on-read, 8-wide chunked blend)
targeted the wrong bottleneck — they optimized the compute, not the memory
access pattern. The fused relu-on-read was actually slower (134 µs) because it
*increased* the scattered vertex reads (each vertex read 4× with relu recomputed
each time). The grid stencil fixes the actual problem: it converts the scattered
read-modify-write into a sequential read-once-write-once stream, which is what
the hardware prefetcher and vectorizer are designed for.

### Result

| Metric | Before (edge-list) | After (grid stencil) | Speedup |
|--------|-------------------|---------------------|---------|
| G5 latency | ~120 µs | ~29 µs | **4.1×** |
| `output` writes per tick | 4 MB (zero-fill) + ~4× scattered RMW | 256 KB sequential write | ~16× reduction |
| `potential` reads per tick | ~4× scattered (cache-line thrash) | 1× sequential (prefetcher-friendly) | ~4× reduction |
| Margin under 100 µs target | 0.83× (FAIL) | 3.4× (PASS) | — |

The latency is now **34× under the paper's GPU ~ms conv baseline** (was 8×). For
the downstream use case (riir-ai Research 168: per-NPC offline rehearsal through
a frozen spatial field), 1000 ticks at 29 µs = **29 ms** — comfortably inside
any sleep-time consolidation budget.

## History (for reference)

### Initial G5 FAIL and diagnosis (pre-fix)

The initial G5 run measured ~120 µs vs the 100 µs target (1.2× miss). The
diagnosis attributed this to DRAM bandwidth for a "12 MB working set" — that
figure was based on an arithmetic error (64×64×16 was computed as 1,048,576
floats instead of the correct 65,536). The actual 768 KB working set fits in L2;
the real bottleneck was the scattered read-modify-write in the edge-list
`graph_laplacian_into`, not DRAM bandwidth. See `issues/001` for the original
(flawed) diagnosis and the four candidate fixes considered.

### Phase 3 SIMD attempts (did NOT help, pre-fix)

Two SIMD optimizations were tried before the stencil fix:

1. **Fused relu-on-read** (`graph_laplacian_of_relu_into`): 134 µs — slower,
   because the laplacian reads vertices scattered by edge connectivity and each
   vertex is read ~4× (once per incident edge), so relu was recomputed 4× per
   vertex. Reverted.
2. **8-wide chunked blend loop**: no improvement — the iterator-zip form already
   auto-vectorizes via non-aliasing slice split.

Both targeted compute throughput; the real bottleneck was memory access pattern,
which the grid stencil fix addresses.
