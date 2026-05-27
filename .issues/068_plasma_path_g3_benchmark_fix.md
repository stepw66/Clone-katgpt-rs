# Issue 068: PlasmaPath G3 Benchmark — Fix Broken FP32 Baseline & Verify Five-Tier Latency

**Severity:** P1 (benchmark integrity)
**Plan:** 148 (PlasmaPath)
**Files:** `tests/bench_148_plasma_path_goat.rs`, `.benchmarks/044_plasma_path_goat.md`

## Problem

### 1. FP32 baseline optimized away in release G3

In `proof_g3_throughput_1024`, the FP32 reference loop:

```rust
for _ in 0..iters {
    for r in 0..1024 {
        y_f32[r] = simd_dot_f32(&f32_w[r * 1024..(r + 1) * 1024], &x, 1024);
    }
}
```

produces 0.3µs/call in release because the compiler eliminates the dead store to `y_f32` (never read outside the loop). This makes the claimed "2-3× speedup" unprovable — we don't know the real FP32 baseline.

**Fix:** Wrap output in `std::hint::black_box()` to prevent optimization.

### 2. Five-tier latency table unverified

The benchmark doc claims:

- Plasma: ~0.3ms ✅ VERIFIED (277µs measured)
- Hot: ~0.5ms ❌ NOT BENCHMARKED
- Warm: ~0.8ms ❌ NOT BENCHMARKED
- Cold: ~1.2ms ❌ NOT BENCHMARKED
- Freeze: ~10ms+ ❌ NOT BENCHMARKED

These need either real measurements or should be marked as estimates.

## Tasks

- [x] T1: Fix G3 benchmark with `black_box` — prevent compiler from eliminating FP32 reference loop
- [x] T2: Re-run release G3 and get real FP32 vs Ternary speedup numbers
- [x] T3: Update `.benchmarks/044_plasma_path_goat.md` with verified numbers
- [x] T4: Mark unverified tiers as estimates with honest latency numbers

## Resolution

### T1–T2: Fixed benchmark, real numbers

Added `black_box` + `consume_f32()` to prevent dead-store elimination. Added FP32 scalar reference as second baseline. Results (release, aarch64 NEON):

| Kernel | µs/call (1024²) | Gop/s |
|--------|----------------|-------|
| Ternary SIMD | 277 | 7.57 |
| FP32 simd_dot (NEON) | 193 | 10.84 |
| FP32 scalar | 710 | 2.95 |

**Honest speedup: 0.70× vs FP32 SIMD, 2.56× vs FP32 scalar.** The "1.5–3.5×" claim over FP32 SIMD is not achieved.

### T3: Benchmark doc updated

- G3 section now has release+debug tables with verified numbers
- Five-tier hierarchy updated with measured Plasma (277µs) and Hot (193µs) latencies
- Warm/Cold/Freeze marked as estimates
- Added correction note about Plasma being slower than Hot in raw latency

### T4: Plan 148 summary updated

Removed "targeting 2-3× throughput" claim, replaced with honest measured numbers.

### Key insight

PlasmaPath's advantage is **memory density** (1.58 vs 32 bits/weight = 20× less traffic), not raw compute speed vs optimized NEON FMA. When workloads are memory-bound, Plasma wins. When compute-bound, NEON FMA is faster.
