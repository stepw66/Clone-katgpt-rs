# dirichlet_energy: scalar distance loop should use SIMD

## Status
✅ **DONE** — `simd_dist_sq` kernel added; `dirichlet_energy` uses it

## Severity
🟡 MEDIUM — called for diagnostic probing of KV cache alignment

## Location
`crates/katgpt-core/src/dirichlet.rs:41-45`

## Problem
The inner distance computation in `dirichlet_energy` is a scalar loop:
```rust
for d in 0..dim {
    let diff = embeddings[row_i + d] - embeddings[row_j + d];
    dist_sq += diff * diff;
}
```

Both embedding rows are contiguous in memory — perfect for SIMD. The operation is equivalent to: load two contiguous vectors, subtract, square, accumulate.

For `dim=128` and `adjacency.len()=1000`, this is 128K scalar ops per call. SIMD would process 4-8 elements per iteration.

## Proposed Fix
Add a `simd_dist_sq` kernel, or reuse existing primitives:

### Option A: dedicated kernel (fastest — single pass)
```rust
/// SIMD distance²: `Σ (a[i] - b[i])²` for len elements.
pub fn simd_dist_sq(a: &[f32], b: &[f32], len: usize) -> f32 {
    // NEON: vsubq_f32 → vmulq_f32 → vfmaq_f32
    // AVX2: _mm256_sub_ps → _mm256_mul_ps → _mm256_fmadd_ps
}
```

### Option B: compute difference into scratch, then dot (reuse existing)
```rust
let mut diff = vec![0.0f32; dim]; // or pre-allocate
simd_add_into(&mut diff, &emb_i, &emb_j_negated); // need negate
let dist_sq = simd_dot_f32(&diff, &diff, dim);
```
Less ideal — requires allocation or pre-allocated scratch.

### Recommended: Option A

## Estimated Impact
- **4-8× faster** distance computation per edge
- Benefit scales with number of adjacency edges

## Acceptance Criteria
- [x] `simd_dist_sq` kernel added to `katgpt-core/src/simd.rs` (NEON + AVX2 + scalar)
- [x] `dirichlet_energy` uses `simd_dist_sq`
- [x] All dirichlet tests pass
