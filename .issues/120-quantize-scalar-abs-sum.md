# Issue 121: `TernaryWeights::quantize_from_f32` Uses Scalar Abs-Sum

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/types.rs` (L2471)

## Problem

`abs_sum: f32 = row.iter().map(|v| v.abs()).sum()` is a scalar reduction. For large `cols`, this scalar loop is slow. The codebase has SIMD primitives (`simd_sum_sq`) but no `simd_sum_abs`.

## Fix

Add a `simd_sum_abs_f32` function to `simd.rs`, or reuse the existing SIMD dot product with a sign-flipped copy. The scalar fallback is acceptable for quantization (called once at load time), but for model conversion with many rows it adds up.
