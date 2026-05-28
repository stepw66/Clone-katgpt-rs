# Issue 108: `attention_forward_safe` Uses Scalar Dot Products Instead of `simd_dot_f32`

## Severity: High
## Files: `katgpt-rs/src/dllm.rs` (L433-436, L456-461)

## Problem

The `attention_forward_safe` function computes dot products with scalar loops:

```rust
// Scalar accumulation per query-key pair
let mut dot = 0.0f32;
for d in 0..head_dim {
    dot += q[q_off + d] * k_all[t * kv_dim + kv_off + d];
}
```

The codebase already has `simd_dot_f32` available (used in `ega_attn.rs`), but this function doesn't use it. Since attention is the hottest path in the transformer (called once per head per position), the scalar loop leaves significant SIMD performance on the table.

## Fix

Replace scalar dot loops with `crate::simd::simd_dot_f32`:

```rust
let dot = crate::simd::simd_dot_f32(
    &q[q_off..q_off + head_dim],
    &k_all[t * kv_dim + kv_off..t * kv_dim + kv_off + head_dim],
    head_dim,
);
```
