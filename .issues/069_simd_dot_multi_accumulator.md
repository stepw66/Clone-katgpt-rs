# simd_dot_f32: use multi-accumulator unrolling for NEON/AVX2

## Status
✅ **DONE** — multi-accumulator unrolling implemented in `simd_dot_f32`

## Severity
🔴 HIGH — affects every matmul in the project

## Location
`crates/katgpt-core/src/simd.rs:124-175` (`neon_dot_f32`, `avx2_dot_f32`)

## Problem
The dot product kernels use a **single accumulator register**. For typical sizes (`head_dim` 64-128, `n_embd` 256-1024), the FMA pipeline latency isn't hidden. Modern CPUs have multiple FMA execution units — 2-4 independent accumulators let out-of-order execution overlap FMAs and improve throughput by ~1.5-2×.

### Current (NEON)
```rust
let mut acc = vdupq_n_f32(0.0);
for _ in 0..chunks {
    let va = vld1q_f32(a.as_ptr().add(i));
    let vb = vld1q_f32(b.as_ptr().add(i));
    acc = vfmaq_f32(acc, va, vb); // single accumulator → pipeline stall
    i += 4;
}
let mut sum = vaddvq_f32(acc);
```

## Proposed Fix
Unroll with 4 independent accumulators, each processing 4 elements (16 elements per iteration on NEON, 32 on AVX2). Reduce at the end.

```rust
// NEON: 4 × float32x4_t accumulators
let mut acc0 = vdupq_n_f32(0.0);
let mut acc1 = vdupq_n_f32(0.0);
let mut acc2 = vdupq_n_f32(0.0);
let mut acc3 = vdupq_n_f32(0.0);
let chunks4 = len / 16;
let mut i = 0;
for _ in 0..chunks4 {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a.as_ptr().add(i+4)), vld1q_f32(b.as_ptr().add(i+4)));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a.as_ptr().add(i+8)), vld1q_f32(b.as_ptr().add(i+8)));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a.as_ptr().add(i+12)), vld1q_f32(b.as_ptr().add(i+12)));
    i += 16;
}
// Horizontal reduce: acc0+acc1+acc2+acc3
let sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
// Handle remainder with single accumulator
```

Same pattern applies to `avx2_dot_f32` (use 4× `__m256`, process 32 elements/iter).

## Estimated Impact
- **1.5-2× faster** dot product on typical sizes (64-1024 elements)
- Benefits: all matmuls, attention scoring, rmsnorm, softmax max-finding, maxsim

## Acceptance Criteria
- [x] `neon_dot_f32` uses 4 accumulators (16 elements/iter)
- [x] `avx2_dot_f32` uses 4 accumulators (32 elements/iter)
- [x] All existing dot product tests pass unchanged
- [x] Benchmark shows ≥1.3× improvement for len=64, 128, 256, 512
