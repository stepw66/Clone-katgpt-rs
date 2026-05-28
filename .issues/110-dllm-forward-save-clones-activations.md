# Issue 110: `forward_save` Clones All 13 Activation Buffers Into `ForwardActivations`

## Severity: Medium
## Files: `katgpt-rs/src/dllm.rs` (L762-778)

## Problem

`forward_save` calls `.to_vec()` on 13 slices from `ForwardSaveContext`, creating 13 new heap allocations every call:

```rust
ForwardActivations {
    embeddings: ctx.embeddings.to_vec(),
    after_norm1: ctx.after_norm1.to_vec(),
    q_proj: ctx.q_proj.to_vec(),
    // ... 10 more .to_vec() calls
}
```

This is the hot training path — called once per training sample per epoch. Total allocation is O(seq_len × (n_embd + kv_dim + vocab_size)).

## Fix

Return a reference to `ForwardSaveContext` instead of copying, or make `ForwardActivations` borrow from the context:

```rust
struct ForwardActivations<'a> {
    seq_len: usize,
    embeddings: &'a [f32],
    after_norm1: &'a [f32],
    // ... borrow slices from ForwardSaveContext ...
}
```
