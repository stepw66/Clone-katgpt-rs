# Issue 132 — MANCE SVD Caching Optimization

> **Spawned from:** Plan 426 MANCE primitive (commit `ea3c95ad`, 2026-07-11)
> **Date:** 2026-07-11
> **Status:** OPEN
> **Priority:** Medium (perf, not correctness)
> **Feature gate:** `manifold_erasure` (already default-on)

## TL;DR

The MANCE primitive's per-step cost is dominated by the one-sided Jacobi SVD
(~4µs for 8×8, ~600µs for 16×64). The paper itself reports "~50% of runtime on
local SVDs." This issue tracks a cached-tangent optimization: skip the SVD when
the k-NN neighbor set hasn't changed significantly since the last step.

## Problem

`manifold_erasure_step_into` calls `estimate_local_tangent_into` on every step,
which runs `thin_svd_into` (Plan 301) on the k×d centered neighbor matrix. For
iterative loops (`manifold_erasure_loop_into`), this means N SVDs for N steps,
even though the tangent basis changes slowly (the neighbor set is largely stable
across steps because the erasure step is trust-bounded — `‖x̃ - x‖ ≤ ε·r_i` with
ε=0.1, so x moves at most 10% of the local radius per step).

## Proposed Optimization

**Cache the tangent basis `{B, σ}` and the neighbor indices.** Recompute only
when:
1. The neighbor set changes (any of the k nearest neighbors differs from the
   cached set), OR
2. The point has moved more than a threshold fraction of `r_i` since the last
   SVD (e.g., `‖x̃ - x_cached‖ > 0.5 · r_i`).

When the cache is valid, skip `estimate_local_tangent_into` and reuse the cached
`B`, `σ`, and `r_i`. This reduces per-step cost from O(k·d·min(k,d)) (SVD) to
O(k·d) (k-NN distance computation + dot products).

## Expected Gain

- For 10-round loops: ~9 SVDs skipped → ~9×4µs = 36µs saved per loop (HLA d=8).
  Current loop cost is ~44µs; expected post-optimization: ~8µs (k-NN + step only).
- For shard-scale (d=64): ~9×600µs = 5.4ms saved per loop. Current: ~6ms;
  expected: ~600µs (one SVD + 9 cheap steps).

## Implementation Sketch

```rust
pub struct ManceTangentCache {
    /// Cached neighbor indices (into the natural pool).
    neighbor_idx: [u32; MAX_K],
    /// Cached tangent basis B (d×r, column-major).
    tangent_basis: Vec<f32>,
    /// Cached singular values σ (r).
    singular_values: Vec<f32>,
    /// Cached local radius r_i.
    local_radius: f32,
    /// The point position when the cache was populated.
    cached_x: Vec<f32>,
    /// Number of valid neighbors in the cache.
    k: usize,
}

/// Returns cached tangent if valid, else recomputes and updates cache.
fn tangent_with_cache(
    x: &[f32],
    pool: &[f32],
    d: usize,
    k: usize,
    config: &ManceConfig,
    scratch: &mut ManceScratch,
    cache: &mut ManceTangentCache,
) -> (&[f32], &[f32], f32); // (B, σ, r_i)
```

## Acceptance Criteria

- [ ] `ManceTangentCache` struct with cache-validity check.
- [ ] `tangent_with_cache` function — returns cached tangent when valid.
- [ ] G2 benchmark: cached loop latency < 50% of uncached loop latency.
- [ ] G1 correctness: cached results match uncached within f32 epsilon (the
      tangent basis is the same when the cache is valid; the only difference is
      when the cache is invalidated and recomputed).
- [ ] G3 no-regression: existing MANCE tests pass unchanged.
- [ ] G4 alloc-free hot path (cache is pre-allocated, reused).

## Non-goals

- Replacing the Jacobi SVD with a faster algorithm (separate issue if needed).
- Approximate SVD (the current SVD is exact; caching doesn't change this).
