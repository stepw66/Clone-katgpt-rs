# Issue 107: `gegelu_tanh` Uses Scalar Inner Loops for Polynomial + Tanh Computation

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/types.rs` (L1415-1428)

## Problem

The `gegelu_tanh` function processes data in CHUNK=64 blocks but builds the intermediate buffer with scalar loops:

```rust
// Scalar polynomial evaluation: buf[j] = 2 * sqrt(2/π) * (g + 0.044715 * g³)
for j in 0..CHUNK {
    let g = gate[i + j];
    buf[j] = 2.0 * sqrt_2_over_pi * (g + 0.044715 * g * g * g);
}
crate::simd::simd_exp_inplace(&mut buf);  // SIMD exp — good

// Scalar tanh combination: hidden[j] = 0.5 * g * (1 + tanh) * up[j]
for j in 0..CHUNK {
    let exp_2inner = buf[j];
    let tanh_val = (exp_2inner - 1.0) / (exp_2inner + 1.0);
    let g = gate[i + j];
    hidden[i + j] = 0.5 * g * (1.0 + tanh_val) * up[i + j];
}
```

The exp step correctly uses `simd_exp_inplace`, but the polynomial build and tanh combination are both scalar. For CHUNK=64, that's 128 scalar iterations per block that could be SIMD.

Compare with `gegelu` and `silu` which use `copy_from_slice` + `simd_scale_inplace` for their first pass.

## Fix

Replace the scalar loops with SIMD operations:

```rust
// Step 1: Copy gate → buf, then SIMD scale for cubic polynomial
buf[..CHUNK].copy_from_slice(&gate[i..i + CHUNK]);
// buf[j] = 0.044715 * g³ + g  (Horner's method via SIMD)
crate::simd::simd_scale_inplace(&mut buf, 0.044715);
// Need element-wise square and multiply — use a SIMD fused multiply
for j in 0..CHUNK {
    let g = gate[i + j];
    buf[j] = 2.0 * sqrt_2_over_pi * (g + buf[j] * g * g);
}
```

Or at minimum, use `simd_scale_inplace` for the constant multiplications and leave only the g² and g³ terms as scalar (reducing from 3 multiplies to 2 per element).
