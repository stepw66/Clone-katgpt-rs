# Issue 099: Scalar Loops Where SIMD Operations Are Available

## Severity: Medium
## Files: `katgpt-rs/src/tf_loop.rs` (L56-65, L78-88), `katgpt-rs/src/transformer.rs` (L917-923)

## Description
Several loops over `n_embd`-sized slices use scalar iteration instead of SIMD primitives already available in the crate.

### tf_loop.rs: `sub_step_damped_euler` (L56-65) and `anchor_blend` (L78-88)
Both iterate over f32 slices with scalar `for` loops. These operate on `n_embd`-sized vectors (typically 64-512 elements) and are called K times per position in the training-free loop.

### transformer.rs: Residual gate loop in `forward_looped` (L917-923)
Plain scalar loop: `ctx.x[d] += residual_gate.gates[gate_offset + d] * ctx.prev_h[d]` over `n_embd` elements. Should use `simd_fused_scale_acc` or equivalent.

## Fix
Replace scalar loops with existing SIMD primitives from `crate::simd`:
- `sub_step_damped_euler`: use `simd_scale_inplace` + `simd_add_inplace` or a fused variant
- `anchor_blend`: use `simd_scale_inplace` + `simd_add_inplace`
- Residual gate loop: use `simd_fused_scale_acc`

## Impact
Medium — for n_embd=384 with K=5 iterations, that's ~1920 scalar ops that could be ~240 SIMD ops (AVX2). The training-free loop and looped forward are already compute-heavy, so this compounds.
