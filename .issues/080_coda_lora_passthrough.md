# CODA fused path skips when LoRA is active — should support bias passthrough

## Status
✅ **DONE** — LoRA perturbations applied separately after each CODA kernel output, matching forward_base projection points.

## Severity
🟢 LOW — LoRA usage is optional, CODA path works without it

## Location
`src/transformer.rs:1950` (`forward_coda`)

## Problem
When a LoRA adapter is present, `forward_coda` falls back entirely to `forward_base`:
```rust
if lora.is_some() {
    return forward_base(ctx, weights, cache, token, pos, config, lora);
}
```

This means the CODA fused kernels (which eliminate ~8 buffer passes per layer) are completely bypassed. LoRA is often a small rank-8/16 perturbation — the bias term in `simd_matmul_residual_partial_rms` could absorb it.

## Fix Applied
LoRA output is input-dependent (`scale * B @ (A @ input)`), so it cannot be fused into CODA's pre-computed `bias` parameter. Instead, LoRA perturbations are computed separately and added additively after each CODA kernel output, matching the same projection points as `forward_base`:

1. **QKV projections** — LoRA applied after each matmul (same as forward_base)
2. **CODA Kernel 1** (out_proj + residual + partial_rms) — LoRA for output projection added to `ctx.x` after kernel
3. **CODA Kernel 2** (MLP matmul + delayed RMS + activation) — LoRA for MLP up added to `ctx.hidden` after kernel
4. **CODA Kernel 3** (down_proj + residual) — LoRA for MLP down added to `ctx.x` after kernel (both sparse and dense paths)

## Estimated Impact
- **~30-40% of forward path** regains CODA fusion when LoRA is active
- Depends on LoRA usage patterns

## Acceptance Criteria
- [x] `forward_coda` works with LoRA without falling back to `forward_base`
- [x] LoRA perturbations applied additively after CODA kernel outputs (not through bias param, since LoRA output is input-dependent)
- [x] All LoRA + CODA tests pass
- [ ] Benchmark: CODA+LoRA matches CODA-without-LoRA within 5%
