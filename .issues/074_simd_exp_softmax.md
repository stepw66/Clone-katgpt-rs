# softmax/attention: scalar exp should use SIMD for long sequences

## Status
✅ **DONE** — `simd_exp_inplace` used in softmax and attention paths

## Severity
🟡 MEDIUM — softmax is called per head per layer, attention per head

## Location
- `crates/katgpt-core/src/types.rs:1316-1319` (`softmax` Pass 2)
- `src/transformer.rs:534-540` (`attention_head` Pass 2)

## Problem
Both `softmax()` and `attention_head()` use scalar `.exp()` calls in a tight loop:

### softmax
```rust
for val in x.iter_mut() {
    *val = (*val - max_val).exp();  // scalar libm expf
    sum += *val;
}
```

### attention_head
```rust
for t in 0..t_n {
    let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
    *scores_buf.get_unchecked_mut(t) = exp_val;
    sum += exp_val;
}
```

The existing `simd_exp_inplace` kernel (L962-979) exists but isn't used here because the comment says "scalar libm expf is faster than Cephes SIMD on Apple Silicon NEON." However, the scalar loop prevents LLVM from auto-vectorizing because `expf` is an external call. Even on Apple Silicon, an explicit SIMD path using hardware fcvt+approximation would beat scalar for sequences > 16 elements.

## Proposed Fix

### Option A: SIMD exp + horizontal sum for softmax
```rust
// Pass 2: SIMD exp + scalar sum
crate::simd::simd_exp_inplace(&mut shifted[..]); // shifts already applied
let sum: f32 = x.iter().copied().sum();
```

This fuses into 2 passes (SIMD exp + scalar sum) instead of 1 scalar exp+sum pass. For long sequences, the SIMD exp dominates and wins.

### Option B: Keep scalar on aarch64, use SIMD on x86_64
The existing comment is correct for Apple Silicon hardware expf. But on x86_64 without AVX512, the Cephes SIMD exp is significantly faster than scalar `expf`. Feature-gate:

```rust
#[cfg(target_arch = "x86_64")]
{ crate::simd::simd_exp_inplace(x); }
#[cfg(target_arch = "aarch64")]
{ for val in x.iter_mut() { *val = (*val - max_val).exp(); } }
```

### For attention_head specifically
The softmax there operates on `t_n` elements (sequence length). For long sequences (> 128), SIMD exp would be 4-8× faster even with the extra pass overhead.

## Estimated Impact
- **2-4× faster** softmax on x86_64 (AVX2 Cephes)
- **~0-15% faster** on aarch64 (hardware expf is already fast, but auto-vectorization is blocked)
- Bigger win for long sequences in attention

## Acceptance Criteria
- [x] `softmax` uses `simd_exp_inplace` on x86_64
- [x] `attention_head` uses `simd_exp_inplace` for sequences > 64
- [x] Fused exp+sum benchmark shows improvement
- [x] All softmax tests pass
- [x] Numerical accuracy within 1e-5 of scalar exp
