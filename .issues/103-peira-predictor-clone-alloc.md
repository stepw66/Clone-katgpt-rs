# Issue 103: PeiraCovariance::predictor() Clones k×k Matrix Per Call

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/peira.rs` (L724-741)

## Description
`self.n.clone()` allocates a new k×k `Vec<f64>` every time the predictor is computed. For dim=8 this is 512 bytes; for dim=64 this is 32KB. If called per training step, this adds significant allocation pressure.

Per optimization.md: "Reuse scratch buffers across loop iterations instead of allocating per-iteration."

## Fix
Accept a pre-allocated scratch buffer, or reuse the existing `pm` scratch field:
```rust
pub fn predictor_into(&self, scratch: &mut [f64]) -> (Vec<f64>, Vec<f64>) {
    let k = self.config.dim;
    let lambda = self.config.lambda;
    scratch[..k*k].copy_from_slice(&self.n);
    // ... operate on scratch instead of cloned vec
}
```

## Impact
Medium — depends on call frequency. If per-step, allocation pressure is significant at larger dims.
