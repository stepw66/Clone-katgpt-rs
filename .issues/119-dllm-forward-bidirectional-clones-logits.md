# Issue 120: `forward_bidirectional_positions` Clones Logits Per Position

## Severity: Medium
## Files: `katgpt-rs/src/dllm.rs` (L400)

## Problem

`all_logits.push(bctx.logits.clone())` clones the entire logits buffer (`vocab_size` elements) for every position. For seq_len=T positions, this is T allocations of `vocab_size` floats each:

```rust
for p in positions {
    // ... forward pass ...
    all_logits.push(bctx.logits.clone()); // O(vocab_size) allocation per position
}
```

## Fix

Return a flat `Vec<f32>` of size `seq_len × vocab_size` instead of `Vec<Vec<f32>>`:

```rust
let mut all_logits = vec![0.0f32; positions.len() * config.vocab_size];
for (idx, &p) in positions.iter().enumerate() {
    // ... forward pass ...
    all_logits[idx * config.vocab_size..(idx + 1) * config.vocab_size]
        .copy_from_slice(&bctx.logits);
}
```
