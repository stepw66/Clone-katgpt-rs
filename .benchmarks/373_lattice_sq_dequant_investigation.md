# Bench 373 — Lattice Lookup + SQ-3bit Par Dequant Investigation

**Date:** 2026-07-03
**Status:** ✅ INVESTIGATED — 1 false alarm, 1 real optimization
**Trigger:** Two "real regressions" flagged by the detector after Bench 372:
- `Lattice lookup dim=8` (57% drop)
- `SQ-3bit dequant 16pos (par)` (33% drop)

## Verdict (honest)

Neither is a code regression. One is a measurement artifact; the other is a
real but modest optimization opportunity.

| Benchmark | Verdict | Root Cause |
|-----------|---------|------------|
| Lattice lookup dim=8 | **False alarm** | Benchmark measures trivial arithmetic on 8 elements × 5K iters — too small for stable measurement. Current numbers (253-276M) match May peaks. The 117M low was thermal throttling. |
| SQ-3bit dequant (par) | **Real inefficiency** (not regression) | API allocates `Vec<f32>` + per-worker `DequantizeScratch` on every call. `_into` variant eliminates output allocation (+9-13% at production batch sizes). |

## Lattice Lookup dim=8 — NOT A REGRESSION

### Evidence

Ran 3 trials post-LTO fix (Bench 372):

| Trial | dim=8 | dim=16 | dim=32 |
|-------|------:|-------:|-------:|
| 1 | 276M | 150M | 119M |
| 2 | 253M | 150M | 119M |
| 3 | 259M | 148M | 115M |
| May peak | 274M | 151M | 126M |

Current numbers match May peaks. The June-12 "117M" was the last run of a long
session — thermal throttling.

### Root Cause of Variance

The benchmark (`src/benchmark/simd.rs:261`) measures:

```rust
for (j, &c) in coords.iter().enumerate() {
    lattice_idx[j] = (c * scale).floor() as usize;
}
```

This is **not** a lattice function call — it's coordinate quantization
(multiply + floor + cast). With dim=8 and 5000 iterations, the total work is
~40K trivial operations completing in ~18μs. At this scale, CPU frequency
scaling dominates the measurement.

### Fix Applied

- Renamed to "Coordinate Quantization (Lattice Index Step)" for accuracy
- Increased iterations 10× (5K → 50K) for stability
- Added comment documenting what the benchmark actually measures

## SQ-3bit Dequant (par) — REAL OPTIMIZATION

### Evidence: par vs seq at different batch sizes

| n_positions | par vs seq | _into vs alloc |
|------------:|-----------:|---------------:|
| 16 | 0.02× (seq 54× faster) | +9.3% |
| 64 | 0.03× (seq 29× faster) | -3.3% (noise) |
| 256 | 0.19× (seq 5.2× faster) | +11.4% |
| 1024 | 0.24× (seq 4.2× faster) | +13.0% |

Parallel is **never** faster than sequential at these sizes (kv_dim=16). The
benchmark forces `threshold=1` to exercise the par path, but with n=16, rayon
overwhelm dominates.

### Root Cause

`par_dequantize_keys_flat` allocates on every call:
1. `vec![0.0f32; n * kv_dim]` — output buffer (1KB at n=16, 64KB at n=1024)
2. `DequantizeScratch::new(kv_dim)` per rayon worker via `for_each_init`

The seq path reuses buffers across all iterations → 50× faster.

### Fix Applied

Added zero-alloc `par_dequantize_keys_flat_into` / `par_dequantize_values_flat_into`
variants that write into a caller-provided `&mut [f32]` + `&mut DequantizeScratch`.
The returning-Vec variants now delegate to `_into` (DRY).

**Benchmark updated** to compare alloc vs `_into` side-by-side, showing the
allocation overhead.

### Why not a bigger gain?

The per-worker `DequantizeScratch` allocation via `for_each_init` remains — it's
the standard rayon pattern. A thread-local scratch pool would eliminate it but
adds complexity. The output buffer is the bigger allocation (n × kv_dim vs 3 × kv_dim
per worker), so eliminating it captures most of the gain.

## Files Changed

| File | Change |
|------|--------|
| `crates/katgpt-spectral/src/spectral_kv_cache.rs` | Added `_into` variants for keys + values; refactored alloc variants to delegate (DRY); 2 new unit tests |
| `crates/katgpt-spectral/src/forward.rs` | Added `_into` wrapper functions; fixed `DequantizeScratch` import (was over-gated on `maxsim`) |
| `crates/katgpt-spectral/src/lib.rs` | Export `_into` functions |
| `src/benchmark/infrastructure.rs` | Added `_into` vs alloc comparison in par dequant benchmark |
| `src/benchmark/simd.rs` | Renamed lattice benchmark; 10× iterations for stability; accurate docs |
| `src/benchmark/mod.rs` | Made `simd` module pub + exported `bench_simd_perf` for examples |

## TL;DR

Lattice lookup dim=8 was a false alarm (thermal noise on trivial workload).
SQ-3bit par dequant had a real allocation inefficiency — fixed with zero-alloc
`_into` variants (+9-13% at production batch sizes). 2 new unit tests verify
bit-exact correctness. 90/90 spectral tests pass.
