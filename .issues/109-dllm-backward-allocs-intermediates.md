# Issue 109: `backward` Allocates 5 Intermediate Gradient Buffers Per Call

## Severity: High
## Files: `katgpt-rs/src/dllm.rs` (L854-858)

## Problem

In `backward()`, five intermediate buffers are allocated fresh on every call:

```rust
let mut d_attn_out = vec![0.0f32; seq_len * n_embd];
let mut d_q = vec![0.0f32; seq_len * kv_dim];
let mut d_k = vec![0.0f32; seq_len * kv_dim];
let mut d_v = vec![0.0f32; seq_len * kv_dim];
let mut d_after_norm2 = vec![0.0f32; seq_len * n_embd];
```

These are sized proportional to `seq_len × n_embd` and allocated every backward pass during training.

## Fix

Move these into `BackwardContext` or a reusable scratch struct, pre-allocate once, and `fill(0.0)` on each use:

```rust
struct BackwardScratch {
    d_attn_out: Vec<f32>,
    d_q: Vec<f32>,
    d_k: Vec<f32>,
    d_v: Vec<f32>,
    d_after_norm2: Vec<f32>,
}

impl BackwardScratch {
    fn new(config: &Config, seq_len: usize) -> Self { /* ... */ }
    fn clear(&mut self) {
        self.d_attn_out.fill(0.0);
        self.d_q.fill(0.0);
        // etc.
    }
}
```
