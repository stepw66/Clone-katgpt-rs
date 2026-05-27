# peira.rs: scalar f64 outer products + per-call allocations

## Status
🟡 **PARTIAL** — Scratch buffers pre-allocated ✅. Cholesky inversion ✅ (Issue 079). Outer product loops and matmul in `update()`/`peira_aux_loss` are still scalar f64.

## Severity
🔴 HIGH — O(k²) per update/loss call, called every training step

## Location
`crates/katgpt-core/src/peira.rs`

## Problem

### 1. Scalar f64 outer products in `update()` (L127-148)
```rust
for i in 0..k {
    for j in 0..k {
        sigma_ij = si * tj;
        n_ij = (si * sj + ti * tj) / 2.0;
        // scalar, no SIMD
    }
}
```
For k=512, this is 262K scalar f64 multiply-adds per sample.

### 2. Per-call heap allocations in `peira_aux_loss()` (L235-236)
```rust
let mut sigma_sample = vec![0.0f64; k * k];
let mut n_sample = vec![0.0f64; k * k];
let mut pm = vec![0.0f64; k * k];
```
Three `k×k` allocations every call. For k=512, that's 6MB of allocation per loss computation.

### 3. `peira_aux_loss` has O(k³) matmul (L261-274)
The `P* @ M` computation is a scalar triple loop — O(k³) f64 ops.

## Proposed Fix

1. **SIMD f64 outer product kernel**: NEON `float64x2_t` (2-wide), AVX2 `__m256d` (4-wide). Process 2-4 elements per iteration.

2. **Pre-allocate scratch buffers**: Move `sigma_sample`, `n_sample`, `pm` into `PeiraCovariance` as persistent buffers (allocated once in `new()`).

3. **Use SIMD dot for the `P* @ M` matmul**: Same pattern as `simd_matmul_rows` but for f64. Or transpose `M` first for contiguous inner-loop access.

```rust
pub struct PeiraCovariance {
    sigma: Vec<f64>,
    n: Vec<f64>,
    config: PeiraConfig,
    step_count: usize,
    // Pre-allocated scratch for peira_aux_loss
    sigma_sample: Vec<f64>,
    n_sample: Vec<f64>,
    pm: Vec<f64>,
}
```

## Estimated Impact
- **2-4× faster** `update()` and `peira_aux_loss()` via f64 SIMD
- **Eliminates 3 allocations** per `peira_aux_loss` call
- Benefit scales with k² (update) and k³ (loss)

## Acceptance Criteria
- [ ] `update()` outer product uses SIMD f64 (NEON + AVX2 + scalar fallback) — **still scalar**
- [x] `peira_aux_loss` scratch buffers pre-allocated in `PeiraCovariance`
- [ ] `peira_aux_loss` P*@M matmul uses SIMD f64 dot product — **still scalar**
- [x] All existing PEIRA tests pass
- [x] No new per-call allocations in hot path
