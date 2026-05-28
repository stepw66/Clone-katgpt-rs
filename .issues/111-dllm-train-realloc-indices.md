# Issue 112: `train_mini_dllm` Re-allocates `indices` Vec Every Epoch

## Severity: Medium
## Files: `katgpt-rs/src/dllm.rs` (L1249, L1346)

## Problem

The training loop allocates `indices: Vec<usize> = (0..train_data.len()).collect()` on every epoch. For N training samples repeated `n_epochs` times, this is N allocations of N-element vectors.

```rust
for epoch in 0..n_epochs {
    let mut indices: Vec<usize> = (0..train_data.len()).collect(); // allocates every epoch!
    // Fisher-Yates shuffle...
}
```

## Fix

Pre-allocate `indices` once before the epoch loop and reuse:

```rust
let mut indices: Vec<usize> = (0..train_data.len()).collect();
for epoch in 0..n_epochs {
    // Fisher-Yates shuffle in-place
    for i in (1..indices.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        indices.swap(i, j);
    }
    // ... training ...
}
```
