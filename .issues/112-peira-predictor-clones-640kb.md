# Issue 113: `predictor()` Clones k×k Matrix + Allocates 5 Scratch Buffers Per Call

## Severity: High
## Files: `katgpt-rs/crates/katgpt-core/src/peira.rs` (L737)

## Problem

`predictor()` clones `self.n` (a `k×k` f64 matrix), then allocates 5 new `Vec`s: `q_star`, `l_scratch`, `l_inv_scratch`, `p_star`, `bt_scratch`. For `k=128`, each is 128×128×8 = 128KB, totaling ~640KB of allocations per call. This is called per-step during training.

The code already provides `predictor_with_scratch()` which avoids cloning `n`, but both functions still allocate `q_star` and `p_star` as return values.

## Fix

Add a `_into` variant that writes into caller-provided output buffers:

```rust
pub fn predictor_into(
    &mut self,
    p_star: &mut [f64],
    q_star: &mut [f64],
) {
    // Use self.pm, self.inv_l_scratch, self.inv_l_inv_scratch, self.matmul_bt_scratch
}
```

Callers pre-allocate `p_star` and `q_star` once in their training context.
