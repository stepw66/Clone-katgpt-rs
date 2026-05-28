# Issue 106: NEON `neon_exp_inplace` Scalar Bit-Manipulation Fallback in 2^n Step

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/simd.rs` (L1739-1754)

## Problem

The NEON exp path computes the polynomial mantissa in SIMD but falls back to scalar for the 2^n scaling step:

```rust
// 2^n via scalar bit manipulation (NEON lacks direct float-to-bits cast)
let q_arr: [f32; 4] = core::mem::transmute(q);
let n_arr: [i32; 4] = core::mem::transmute(vn_i);
let mut result = [0.0f32; 4];
for j in 0..4 {
    let n = n_arr[j];
    if n < -126 { result[j] = 0.0; }
    else if n > 127 { result[j] = f32::INFINITY; }
    else {
        let bits = ((n + 127) as u32) << 23;
        let scale = f32::from_bits(bits);
        result[j] = scale * q_arr[j];
    }
}
let vresult = vld1q_f32(result.as_ptr());
```

This defeats SIMD vectorization for the scaling step: extracts 4 lanes to scalar arrays, loops with branches, and reloads. This runs on every `simd_exp_inplace` call — which is used in softmax in every attention layer.

## Fix

Use NEON comparison + bit manipulation to keep everything in SIMD registers:

```rust
// Clamp n to [-126, 127] via SIMD min/max
let v127 = vdupq_n_s32(127);
let vneg126 = vdupq_n_s32(-126);
let vn_clamped = vmaxq_s32(vminq_s32(vn_i, v127), vneg126);
// Build 2^n: shift (n+127) left by 23 bits, reinterpret as f32
let v_shifted = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vn_clamped, vdupq_n_s32(127)), 23));
let vresult = vmulq_f32(v_shifted, q);
// Clamp underflow/overflow to 0/inf (masked by the clamped n values)
```

This eliminates the scalar fallback entirely, keeping all 4 lanes in NEON registers.
