# newton_schulz.rs: scalar O(n³) matrix operations

## Status
✅ **DONE** — `matmul_xtx` uses `simd_dot_f32` with symmetry exploitation ✅. `matmul_ax` transposes X and uses `simd_dot_f32` ✅. A@A in iteration loop transposes A and uses `simd_dot_f32` ✅.

## Severity
🔴 HIGH — called per optimizer step for every parameter matrix

## Location
`src/newton_schulz.rs:22-55` (`transpose`, `matmul_xtx`, `matmul_ax`)

## Problem
All three matrix helper functions use scalar triple loops. The Newton-Schulz iteration does:
- `matmul_xtx`: m×m×n ops (compute X @ Xᵀ)
- `A@A` implicit in the b*A + c*A² term: m³ ops  
- `matmul_ax`: m×m×n ops (compute B @ X)
- All × 5 iterations

For a 512×512 gradient matrix, this is **~4 billion scalar multiply-adds** per orthogonalization call.

### Current
```rust
fn matmul_xtx(x: &[f32], m: usize, n: usize, a: &mut [f32]) {
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += x[i * n + k] * x[j * n + k];  // scalar, stride access
            }
            a[i * m + j] = sum;
        }
    }
}
```

## Proposed Fix

### Quick win: use existing SIMD dot for inner products
```rust
fn matmul_xtx(x: &[f32], m: usize, n: usize, a: &mut [f32]) {
    for i in 0..m {
        for j in i..m {  // exploit symmetry
            let dot = crate::simd::simd_dot_f32(&x[i*n..(i+1)*n], &x[j*n..(j+1)*n], n);
            a[i * m + j] = dot;
            a[j * m + i] = dot;  // symmetric
        }
    }
}
```

This alone gives 4-8× speedup on the inner loop.

### Further: exploit symmetry in `matmul_xtx`
`A = X @ Xᵀ` is always symmetric — only compute upper triangle and mirror. Cuts work in half.

### Further: block-tiled SIMD matmul for `matmul_ax`
For large m, use a tiled/blocked approach to stay in L1 cache.

## Estimated Impact
- **4-8× faster** Newton-Schulz via SIMD inner products
- **Additional 2×** from symmetry exploitation on `matmul_xtx`
- Total potential: **8-16× speedup** on orthogonalization

## Acceptance Criteria
- [x] `matmul_xtx` uses `simd_dot_f32` for inner products
- [x] `matmul_xtx` exploits symmetry (upper triangle + mirror)
- [x] `matmul_ax` uses `simd_dot_f32` for inner products
- [x] All Newton-Schulz tests pass
- [x] Muon optimizer tests pass
