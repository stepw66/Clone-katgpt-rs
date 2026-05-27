# MLS and delta routing: scalar loops should use SIMD

## Status
✅ **DONE** — `simd_fused_sub_acc` kernel added and used in all MLS/delta routing loops

## Severity
🔴 HIGH — runs every layer when feature is enabled

## Location
- `src/transformer.rs:1778-1782` (MLS accumulation in `forward_base`)
- `src/transformer.rs:1794-1799` (delta routing in `forward_base`)
- `src/transformer.rs:2086-2091` (MLS accumulation in `forward_coda`)
- `src/transformer.rs:2101-2107` (delta routing in `forward_coda`)

## Problem
When `mls_aggregate` or `delta_routing` features are enabled, the per-layer accumulation loops are scalar:

### MLS (×2 copies, base + coda)
```rust
for d in 0..n {
    ctx.mls_buf[d] += ctx.x[d] - ctx.hidden_state[d];
}
```

### Delta routing (×2 copies, base + coda)
```rust
for d in 0..n {
    let delta = ctx.x[d] - ctx.xr[d];
    if block_idx < ctx.block_deltas.len() {
        ctx.block_deltas[block_idx][d] += delta;
    }
}
```

These loops iterate over `n_embd` (typically 256-1024) elements per layer. They're equivalent to SIMD-friendly elementwise operations but use scalar code.

## Proposed Fix
Replace with existing SIMD primitives:

### MLS — use a fused subtract-accumulate kernel
```rust
// Option A: compute delta with simd, then add
crate::simd::simd_add_inplace(&mut ctx.mls_buf[..n], /* delta = x - hidden_state */);
// Need a simd_subtract_into or fused kernel

// Option B: dedicated kernel (preferred — single pass)
// simd_fused_sub_acc(dst, a, b): dst[i] += a[i] - b[i]
```

### Delta routing — same pattern
```rust
// simd_fused_sub_acc(&mut ctx.block_deltas[block_idx][..n], &ctx.x[..n], &ctx.xr[..n]);
```

Also: the MLS loop in `forward_base:1819-1821` (scaling by `1/mls_count`) should use `simd_scale_inplace`.

## Estimated Impact
- **~4× faster** per-layer accumulation (SIMD width)
- Eliminates 2 scalar loops × 2 copies (base + coda) × n_layers per forward pass

## Acceptance Criteria
- [x] Add `simd_fused_sub_acc` kernel to `katgpt-core/src/simd.rs` (NEON + AVX2 + scalar)
- [x] Replace all 4 MLS scalar loops with SIMD
- [x] Replace all 4 delta routing scalar loops with SIMD
- [x] MLS scale-by-count uses `simd_scale_inplace`
- [x] All existing tests pass
