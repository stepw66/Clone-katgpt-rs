# peira: use Cholesky decomposition instead of Gauss-Jordan for SPD matrix inversion

## Status
✅ **DONE** — `invert_matrix` replaced with Cholesky-based inversion

## Severity
🟢 LOW — inversion is called infrequently and k is small

## Location
`crates/katgpt-core/src/peira.rs:294-351` (`invert_matrix`)

## Problem
`N + λI` is symmetric positive definite (SPD) by construction (EMA of positive semi-definite covariances + regularization). The current `invert_matrix` uses general Gauss-Jordan elimination which:
1. Doesn't exploit symmetry (does 2× the work of Cholesky)
2. Uses an augmented matrix (2× memory: `[M | I]`)
3. Partial pivoting is unnecessary for SPD matrices (always stable)

## Proposed Fix
Replace with Cholesky decomposition + triangular solve:
```rust
fn invert_spd(mat: &[f64], k: usize) -> Vec<f64> {
    // 1. Cholesky: L such that L*Lᵀ = mat (k³/3 flops vs k³/2 for Gauss-Jordan)
    let mut l = vec![0.0f64; k * k];
    for j in 0..k {
        let mut sum = 0.0;
        for p in 0..j { sum += l[j*k+p] * l[j*k+p]; }
        l[j*k+j] = (mat[j*k+j] - sum).sqrt();
        for i in (j+1)..k {
            let mut sum = 0.0;
            for p in 0..j { sum += l[i*k+p] * l[j*k+p]; }
            l[i*k+j] = (mat[i*k+j] - sum) / l[j*k+j];
        }
    }
    // 2. Invert L (triangular — O(k²))
    // 3. Compute L⁻ᵀ × L⁻¹ = (LLᵀ)⁻¹ = M⁻¹
}
```

Or use the existing Gauss-Jordan but skip the augmented matrix and pivoting — just do LDLᵀ decomposition.

## Estimated Impact
- **~2× faster** matrix inversion for SPD matrices
- **50% less memory** (no augmented matrix)
- More numerically stable for ill-conditioned covariances
- Only matters for k ≥ 256 (smaller k is already fast)

## Acceptance Criteria
- [x] `invert_matrix` replaced with Cholesky-based inversion (or LDLᵀ)
- [x] Exploits symmetry (only computes lower triangle)
- [x] All existing PEIRA tests pass
- [x] Numerical accuracy within 1e-10 of current results
