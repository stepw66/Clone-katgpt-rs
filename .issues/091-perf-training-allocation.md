# 🔴 Perf: Pre-allocate training loop buffers

## Summary
The training forward/backward paths in `dllm.rs` and `transformer.rs` allocate dozens of `Vec`s per position or per training step. These allocations dominate training throughput for short sequences.

## Affected Paths

### 1. `dllm.rs` — `forward_bidirectional_positions` (L254-316)
**Issue**: Inside the `for p in 0..seq_len` loop, allocates 11 `Vec`s per position:
- `vec![0.0f32; n]` for x, q, x_proj, x_mlp, logits (×2 for bidirectional)
- `vec![0.0f32; kvd]` for k, v
- `vec![0.0f32; config.mlp_hidden]` for hidden
- `x_proj.clone()` for residual

**Fix**: Create a `BidirectionalContext` struct with pre-allocated buffers, reuse across positions:
```rust
struct BidirectionalContext {
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    x_proj: Vec<f32>,
    xr2: Vec<f32>,
    hidden: Vec<f32>,
    x_mlp: Vec<f32>,
    logits: Vec<f32>,
}
```
Call `clear()` + write by index at each position.

### 2. `dllm.rs` — `backward` (L690-964)
**Issue**: Allocates per call:
- `TrainingGradients::zeros(config)` — full gradient struct with many Vecs
- `rmsnorm_backward` returns `Vec<f32>` — called per masked position (L817, L936, L953)
- `softmax_backward` returns `Vec<f32>` — called per (position × head) in inner loop (L865)

**Fix**: Pre-allocate gradient buffers in a `BackwardContext`. Write `rmsnorm_backward_into` and `softmax_backward_into` that take `&mut [f32]` output buffers.

### 3. `transformer.rs` — `forward_training_free_loop` (L1000-1026)
**Issue**: Allocates per call:
- `ctx.x[..n].to_vec()` for x_pre_window (L1000)
- `ctx.x[..n].to_vec()` for anchor (L1017)
- `vec![0.0f32; n]` for y_buf (L1026)
- `ctx.x[..n].to_vec()` for stash_x (L1085)

**Fix**: Add `x_pre_window`, `x_anchor`, `y_buf`, `stash_x` as pre-allocated fields on `ForwardContext`.

### 4. `dllm.rs` — `corrupt_block` (L206-224)
**Issue**: Allocates `tokens.to_vec()`, `vec![false; len]`, `(0..len).collect()` per call from training loop.
**Fix**: Accept pre-allocated buffers for corrupted tokens, is_masked mask, and Fisher-Yates positions.

## References
- Optimization guideline: "Cache allocations: Vec::with_capacity() once, clear() + reuse"
- "Pass pre-allocated scratch buffers as &mut [T] parameters"
