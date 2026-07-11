# Benchmark 427 — MANCE SVD Caching GOAT Gate

> **Implements:** Issue 132, Plan 427
> **Date:** 2026-07-11
> **Feature gate:** `manifold_erasure` (default-on)

## TL;DR

MANCE tangent basis caching delivers **4.4x loop speedup** (10.89µs vs 47.98µs
for a 10-round loop) by skipping the one-sided Jacobi SVD when the k-NN neighbor
set is stable across rounds. Cache hit rate: 90.5% (9 of 10 rounds hit). All
GOAT gates pass.

## What Was Optimized

`manifold_erasure_loop_into` calls `manifold_erasure_step_into` N times. Each
step computes:
1. k-NN retrieval — O(N·d), always needed (r_i depends on fresh distances)
2. **Local tangent SVD** — O(k·d·min(k,d)), the dominant cost (~4µs for 8×8)
3. Spectral direction + trust region — O(d·r), cheap

The tangent basis `B` and singular values `σ` depend **only on the k-NN neighbor
positions** (rows of the natural pool), not on the query point `x`. When the
neighbor set is stable across loop rounds, the SVD can be skipped entirely.

## Implementation

### ManceTangentCache

Pre-allocated cache storing:
- `neighbor_indices` (k entries, sorted by index for deterministic comparison)
- `tangent_basis` (d×r, column-major)
- `singular_values` (r entries)
- `valid` flag
- `cache_hits` / `cache_misses` debug counters

### Cache validity

Cache is valid when the current k-NN neighbor indices match the cached set.
This is both necessary and sufficient:
- Same indices → same neighbor positions → same centered matrix → same SVD
- Different indices → different neighbor positions → different SVD

### k-NN index sorting (Plan 427 fix)

The k-NN function (`knn_distances_into`) now sorts neighbor indices by index
after selection (insertion sort, O(k²) for tiny k). This ensures the same
neighbor set always produces the same row ordering for the SVD, enabling
bit-identical tangent caching. Without this, the max-heap replacement strategy
can return the same neighbor set in different orders across calls, causing
the SVD to produce slightly different floating-point results (up to ~1e-7)
due to different convergence paths.

### API (additive, no breaking changes)

- `ManceTangentCache::with_capacity(d, k, r)` — pre-allocate
- `ManceTangentCache::invalidate()` — reset to invalid
- `manifold_erasure_step_cached_into(...)` — cached single step
- `manifold_erasure_loop_cached_into(...)` — cached iterative loop

## GOAT Gate Results

### G1 — Correctness

Cached results are **bit-identical** to uncached results:
- `cached_step_matches_uncached` — `assert_eq!` on output + all ManceStepInfo fields ✓
- `cached_loop_matches_uncached` — `assert_eq!` on output + all round infos ✓
- `cached_step_cache_hit_same_result` — same x → identical output ✓
- `cache_invalidation_when_neighbors_change` — after invalidation, matches uncached ✓
- `cache_invalidate_resets` — invalidate() works ✓

### G2 — Performance

| Benchmark | Uncached | Cached | Speedup |
|---|---|---|---|
| G2c/G2d: 10-round loop (d=8) | 47.98µs | 10.89µs | **4.4x** |

Cache hit rate: 90.5% (9499 hits / 10500 calls). The 9.5% misses are the first
round of each loop (cache is empty/invalid at start).

### G3 — No regression

- `cargo test -p katgpt-core --lib`: **1468/1468** tests pass (+5 new)
- `cargo test -p katgpt-core --all-features --lib`: **3034/3034** tests pass
- Zero new warnings

### G4 — Alloc-free

- `G4d`: **0 allocs / 100 cached step calls** (cache hit path is pure `copy_from_slice`)
- `G4c`: 1200 allocs / 100 cached loops (loop's per-round `grad_buf` + `current`
  allocations inherited from the uncached loop pattern; cache itself adds 0)

### G5 — Modelless

Cache is pure structural state (indices + basis + sigma). No training, no
gradients, no weight mutations.

## Test Configuration

- d=8 (HLA scale), n=50 natural points, k=8 neighbors, r=8 tangent basis dim
- Pool: deterministic LCG pseudo-random, seed=42, values in [-1, 1)
- Gradient: `[1.0, 0.5, -0.3, 0.8, -0.1, 0.4, 0.2, -0.6]` (non-uniform — matches
  the G1a test; uniform `[1.0; d]` causes aggressive x movement and low cache
  hit rates)

## Key Insight

The uniform gradient `[1.0; d]` causes x to move aggressively (all dimensions
erased simultaneously), changing the k-NN neighbor set across rounds and
defeating the cache. The non-uniform gradient produces a more realistic erasure
pattern where the trust-bounded step (ε=0.1) keeps x within the same Voronoi
cell, maintaining neighbor stability. This is the expected use case: real
erasure probes (MAG/CNA/EmotionDirections) are non-uniform direction vectors.
