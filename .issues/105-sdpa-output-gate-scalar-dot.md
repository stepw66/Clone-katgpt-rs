# Issue 105: `SdpaOutputGate::forward` Uses Scalar Iterator Dot Product Instead of SIMD

## Severity: High
## Files: `katgpt-rs/crates/katgpt-core/src/types.rs` (L282-289)

## Problem

The gate forward pass computes `sigmoid(W_gate @ attn_out)` using a scalar iterator chain:

```rust
for (i, gate_val) in temp.iter_mut().enumerate().take(n) {
    let w_off = i * dim;
    let dot: f32 = self.w_gate[w_off..w_off + dim]
        .iter()
        .zip(attn_out.iter())
        .map(|(w, a)| w * a)
        .sum();  // ← scalar dot product
    *gate_val = 1.0 / (1.0 + (-dot).exp());  // ← scalar exp
}
```

This is a matrix-vector multiply (n rows × dim) using scalar ops. The crate has `simd_dot_f32` and `simd_exp_inplace` that should be used instead. Called every layer when `gated_attn` is enabled — this is on the critical path.

For n = n_head × head_dim (e.g., 2048) and dim = n_embd (e.g., 2304), this is a significant fraction of per-layer compute done without SIMD.

## Fix

Replace scalar dot product with `simd_dot_f32`:

```rust
for i in 0..n {
    let w_off = i * dim;
    let dot = crate::simd::simd_dot_f32(
        &self.w_gate[w_off..w_off + dim],
        attn_out,
        dim,
    );
    unsafe { *temp.get_unchecked_mut(i) = 1.0 / (1.0 + (-dot).exp()); }
}
```

For further improvement, batch-compute all dot products using `simd_matmul_rows` and apply SIMD exp + sigmoid in bulk.
