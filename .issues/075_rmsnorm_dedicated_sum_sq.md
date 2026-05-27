# rmsnorm: self-dot loads data twice — dedicated sum_sq kernel

## Status
✅ **DONE** — `simd_sum_sq` kernel added; `rmsnorm` and `rmsnorm_with_gamma_eps` use it

## Severity
🟡 MEDIUM — called 2× per layer per token

## Location
`crates/katgpt-core/src/types.rs:1357-1368` (`rmsnorm`)

## Problem
`rmsnorm` computes sum-of-squares via `simd_dot_f32(x, x, x.len())`, which loads the same data from memory twice (once for `a`, once for `b`). Since `a == b`, each element only needs to be loaded once, squared, and accumulated.

### Current
```rust
let sum_sq = crate::simd::simd_dot_f32(x, x, x.len());
let inv_rms = 1.0 / (sum_sq / x.len() as f32 + 1e-5).sqrt();
crate::simd::simd_scale_inplace(x, inv_rms);
```

For `n_embd=1024`, `simd_dot_f32` loads 4KB of data twice = 8KB read from L1/L2. A dedicated `simd_sum_sq` would load 4KB once = 4KB, saving ~30% memory bandwidth on this hot path.

## Proposed Fix
Add a `simd_sum_sq` kernel to `katgpt-core/src/simd.rs`:

```rust
/// SIMD-accelerated sum of squares: `Σ x[i]²`.
/// More efficient than `simd_dot_f32(x, x, len)` — loads data once instead of twice.
pub fn simd_sum_sq(x: &[f32], len: usize) -> f32 {
    // NEON: vmulq_f32(v, v) + vfmaq_f32(acc, ...)
    // AVX2: _mm256_mul_ps(v, v) + _mm256_fmadd_ps
    // Scalar: sum += x[i] * x[i]
}
```

Then in `rmsnorm`:
```rust
let sum_sq = crate::simd::simd_sum_sq(x, x.len());
```

Same optimization applies to `rmsnorm_with_gamma_eps` (L1432).

## Estimated Impact
- **~30% bandwidth reduction** on rmsnorm hot path
- `rmsnorm` is called 2× per layer × n_layers per token — savings compound
- Simpler code (no need to pass same slice as both args)

## Acceptance Criteria
- [x] `simd_sum_sq` added to `katgpt-core/src/simd.rs` (NEON + AVX2 + scalar)
- [x] `rmsnorm` uses `simd_sum_sq` instead of `simd_dot_f32(x, x, ...)`
- [x] `rmsnorm_with_gamma_eps` uses `simd_sum_sq`
- [x] All rmsnorm tests pass
- [x] Benchmark: ≤2% regression tolerance
