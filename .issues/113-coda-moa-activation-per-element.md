# Issue 114: `moa_swiglu` and `simd_matmul_rmsnorm_moa_swiglu` Call `MoaActivation::all()` Per Element

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/coda.rs` (L223, L262-270, L361)

## Problem

Inside the output dimension loop, `moa_activate(j, gate_val)` calls `MoaActivation::all()[j].activate(x)`, which constructs a `[MoaActivation; 7]` array on every call. While the compiler may optimize this as a constant, the pattern is repeated for every element in the inner loop:

```rust
fn moa_activate(k: usize, x: f32) -> f32 {
    MoaActivation::all()[k].activate(x)  // array construction per call
}
```

Called twice per element per output dimension (once for gate, once for up).

## Fix

Hoist the activation lookup outside the per-element loop:

```rust
let activations = MoaActivation::all();
for i in 0..output_dim {
    for j in 0..MOA_DICT_SIZE {
        mixed_gate += gate_weights[j] * activations[j].activate(gate_val);
        mixed_up += up_weights[j] * activations[j].activate(up_val);
    }
}
```
