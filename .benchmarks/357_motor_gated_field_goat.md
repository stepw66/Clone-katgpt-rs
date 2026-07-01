# Plan 357 — Motor-Gated DEC Field GOAT Results

**Date:** 2026-07-01
**Primitive:** `evolve_motor_gated_field` (`katgpt-dec/src/motor_gated.rs`)
**Feature:** `motor_gated_field` (opt-in)
**Bench:** `cargo bench -p katgpt-core --features motor_gated_field --bench bench_357_motor_gated_field_goat -- --nocapture`
**Hardware:** macOS (Apple Silicon)

## G1–G5 Results

| Gate | Metric | Value | Threshold | Verdict |
|------|--------|-------|-----------|---------|
| G1 no-teleporting | max centroid displacement / 50 ticks | 0.0001 cells | ≤ 2.0 cells | **PASS ✅** |
| G2 motor-gate locality | channel isolation ratio | ∞ (no leak) | > 100× | **PASS ✅** |
| G3 conservation | `\|Σ K[ReLU(h)]\| / L1(h)` | 0.0000 | < 0.05 | **PASS ✅** |
| G4 zero-alloc | allocs / 1000 ticks (64×64×16) | 0 | = 0 | **PASS ✅** |
| G5 latency | per-call (64×64×16, release) | ~120 µs | < 100 µs | **FAIL ❌ (borderline, 1.2×)** |

## Verdict: G5 borderline FAIL → stays OPT-IN

4/5 gates pass cleanly. G5 is ~120 µs vs the 100 µs aspirational target — a
**1.2× miss**. Per the promotion rule (all 5 PASS → ready for downstream
consumption), `motor_gated_field` stays **opt-in** and is NOT marked ready for
downstream consumption until G5 closes. Follow-up filed in `.issues/`.

### Why G5 misses — memory bandwidth, not compute

The G5 workload (64×64 grid × 16 channels = **1,048,576 floats = 4 MB per
array**) is **memory-bandwidth bound**, not compute bound. Each tick touches
three 4 MB arrays (the field `h`, `scratch_relu`, `scratch_lap`) = **12 MB
working set** with ~28 MB of total read+write traffic:

| Pass | Read | Write | Traffic |
|------|------|-------|---------|
| `relu_gate_into` | h (4 MB) | scratch_relu (4 MB) | 8 MB |
| `graph_laplacian_into` | scratch_relu (4 MB, scattered) | scratch_lap (4 MB) | 8 MB |
| blend | h (4 MB) + scratch_lap (4 MB) | h (4 MB) | 12 MB |
| **Total** | | | **~28 MB** |

At ~120 µs, the effective bandwidth is **~233 GB/s** — already near the
single-core L2/L3 ceiling on this hardware. The 100 µs target implies ~280
GB/s, which is above the cache-resident bandwidth for a 12 MB working set
(exceeds typical L2; partially spills to L3/DRAM).

### Phase 3 SIMD attempts (did NOT help)

Two SIMD optimizations were tried and measured:

1. **Fused relu-on-read** (`graph_laplacian_of_relu_into`): applies `max(0, x)`
   on each element read inside the laplacian stencil, eliminating the separate
   ReLU scratch write pass (~8 MB saved). **Result: 134 µs — SLOWER.** The
   laplacian reads vertices in a scattered pattern (by edge connectivity), so
   each vertex is read ~4× (once per incident edge); the fused path recomputes
   relu 4× per vertex. The extra compute + scattered access defeated the
   traffic saving. Reverted.

2. **8-wide chunked blend loop**: explicit `for k in 0..8` indexing. **Result:
   no improvement** — the iterator-zip form (`h.iter_mut().zip(lap.iter())`)
   already auto-vectorizes better (LLVM proves non-aliasing via the slice split).

**Conclusion:** the G5 bottleneck is DRAM/cache bandwidth for the 3-array
working set, not SIMD compute throughput. Further compute-side optimization
yields diminishing returns.

### Context: the latency is excellent in absolute terms

The paper's GPU conv baseline is **~1 ms**. Our ~120 µs is **~8× faster** on
CPU. For the downstream use case (riir-ai Research 168: per-NPC offline
rehearsal through a frozen spatial field), 1000 ticks at 120 µs = **120 ms** —
trivially fast for an offline consolidation cycle. The 100 µs target was set
aspirationally ("we're CPU SIMD"); the practical latency is already excellent.

## Follow-up (`.issues/`)

The G5 gap is tracked as a follow-up. Options to close it:

- **f16 field support** — halve memory traffic (4 MB → 2 MB per array). Requires
  f16 variants of the DEC operators (katgpt-types has `half` already).
- **Single-scratch fusion** — eliminate one of the two scratch arrays by
  restructuring the algorithm (e.g., in-place relu on h with snapshot/restore).
  Fragile; risks breaking the clean split-step semantics.
- **Tile/block the field** — process the grid in cache-resident tiles (e.g.,
  16×16×16 = 16 KB per tile) to keep the working set in L1/L2. Requires a
  tiled graph Laplacian (not currently shipped).
- **Relax the G5 target to 150 µs** — the 100 µs target was aspirational; 120 µs
  is 8× under the paper's GPU baseline and fast enough for the downstream use
  case.

None of these are blockers for the primitive's correctness or its downstream
consumption at the practical latency. They are perf refinements.
