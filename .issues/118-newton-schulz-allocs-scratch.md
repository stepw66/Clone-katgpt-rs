# Issue 119: `muon_update` Calls Allocating `newton_schulz5` Instead of Zero-Alloc `newton_schulz5_into`

## Severity: Medium
## Files: `katgpt-rs/src/newton_schulz.rs` (L179)

## Problem

`muon_update` calls `newton_schulz5` (the allocating version), which creates 6 temporary buffers on every call. The zero-alloc `newton_schulz5_into` variant exists but isn't used from `muon_update`.

```rust
pub fn muon_update(...) -> Vec<f32> {
    let out = newton_schulz5(grad, rows, cols); // allocates 6 buffers!
    // ...
}
```

## Fix

Add a `muon_update_into` that uses `newton_schulz5_into` with a reusable scratch buffer:

```rust
pub fn muon_update_into(
    grad: &[f32],
    momentum: &mut [f32],
    beta: f32,
    rows: usize,
    cols: usize,
    out: &mut [f32],
    scratch: &mut NewtonSchulzScratch,
) {
    newton_schulz5_into(grad, rows, cols, out, scratch);
    // ... momentum update ...
}
```
