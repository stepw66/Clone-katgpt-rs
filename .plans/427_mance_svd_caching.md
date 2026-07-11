# Plan 427 — MANCE SVD Caching Optimization

> **Implements:** Issue 132 (`.issues/132_mance_svd_caching_optimization.md`)
> **Date:** 2026-07-11
> **Status:** COMPLETE
> **Feature gate:** `manifold_erasure` (already default-on)

## TL;DR

Cache the tangent basis `{B, σ}` keyed on k-NN neighbor indices. Skip the
one-sided Jacobi SVD when the neighbor set hasn't changed across iterative
loop rounds. Expected: ~5x loop speedup (skip ~9 of 10 SVDs in a 10-round
loop).

## Background

`manifold_erasure_loop_into` calls `manifold_erasure_step_into` N times. Each
step computes:
1. k-NN retrieval — O(N·d), always needed (r_i depends on fresh distances)
2. **Local tangent SVD** — O(k·d·min(k,d)), the dominant cost (~4µs for 8×8)
3. Spectral direction + trust region — O(d·r), cheap

The tangent basis `B` and singular values `σ` depend **only on the neighbor
positions** (from the natural pool), not on `x`. If the k-NN neighbor set is
the same across steps, `B` and `σ` are bit-identical. The trust-bounded step
(ε=0.1) moves x at most 10% of r_i per step, so the neighbor set is largely
stable across loop rounds.

## Design

### Cache validity

The cache is valid when the k-NN neighbor indices match the cached set. This
is both necessary and sufficient:
- **Sufficient:** B/σ depend only on neighbor positions (pool rows at cached
  indices). Same indices → same B/σ, bit-identical.
- **Necessary:** Different indices → different neighbor positions → different
  centered matrix → different SVD.

The issue proposed a second condition (movement threshold `‖x - x_cached‖ >
0.5·r_i`). This is **mathematically redundant**: if the neighbor indices
haven't changed, B/σ are identical regardless of x's position. The movement
threshold would cause unnecessary recomputations. Omitted by design.

### API

New types/functions (additive, no changes to existing API):

```rust
pub struct ManceTangentCache { ... }

pub fn manifold_erasure_step_cached_into(
    x, gradient, natural_pool, n, config, scratch, cache, out
) -> Result<ManceStepInfo, ManceError>

pub fn manifold_erasure_loop_cached_into<F>(
    x, gradient_fn, natural_pool, n, config, n_rounds, scratch, cache, out
) -> Result<Vec<ManceStepInfo>, ManceError>
```

### Flow (cached step)

1. k-NN retrieval (always) → writes `scratch.neighbor_distances`, `scratch.neighbor_indices`
2. Compare `scratch.neighbor_indices` with `cache.neighbor_indices`
   - **Same** → copy `cache.tangent_basis` → `scratch.tangent_basis`,
     `cache.singular_values` → `scratch.singular_values` (O(d·r), trivial)
   - **Different** → call `estimate_local_tangent_into` (writes to scratch),
     then copy scratch → cache
3. Spectral direction + trust region + apply (same as uncached)

### G4 (alloc-free)

`ManceTangentCache` is pre-allocated via `with_capacity(d, k, r)`. The hot
path (cached step) does only `slice::copy_from_slice` — zero allocations.

## Tasks

### Phase 1: Implementation

- [x] T1.1 — `ManceTangentCache` struct + `with_capacity` + `is_valid_for` + `update`
- [x] T1.2 — `manifold_erasure_step_cached_into` function
- [x] T1.3 — `manifold_erasure_loop_cached_into` function
- [x] T1.4 — Export new types from `lib.rs`

### Phase 2: Tests (G1 + G3)

- [x] T2.1 — Unit test: cached step matches uncached step bit-identically
- [x] T2.2 — Unit test: cached loop matches uncached loop bit-identically
- [x] T2.3 — Unit test: cache invalidation when neighbor set changes
- [x] T2.4 — Unit test: cache reuse when neighbor set is stable
- [x] T2.5 — G3: existing MANCE tests pass unchanged

### Phase 3: Benchmark (G2 + G4)

- [x] T3.1 — G2 bench: cached loop latency vs uncached loop latency
- [x] T3.2 — G4 bench: 0 allocs over 100 cached loop calls
- [x] T3.3 — GOAT gate benchmark doc

### Phase 4: Issue closure

- [x] T4.1 — Update Issue 132 acceptance criteria
- [x] T4.2 — Commit

## GOAT Gate

- **G1 (correctness):** Cached results match uncached within f32 epsilon
  (bit-identical when cache is valid — same B/σ, same r_i from fresh k-NN).
- **G2 (perf):** Cached loop latency < 50% of uncached loop latency.
- **G3 (no regression):** Existing MANCE tests pass unchanged.
- **G4 (alloc-free):** 0 allocs over 100 steady-state cached loop calls.
- **G5 (modelless):** Cache is pure structural state, no training.
