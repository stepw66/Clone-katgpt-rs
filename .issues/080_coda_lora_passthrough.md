# CODA fused path skips when LoRA is active — should support bias passthrough

## Status
⏸️ **DEFERRED** — Complex restructuring required. CODA kernel bias applies per-element but LoRA has different shapes per projection (Q/K/V/O/W1/W2). Need to restructure how LoRA bias feeds into the fused matmul_residual_partial_rms kernel. Low severity since LoRA usage is optional and CODA path works without it. Code has TODO comment documenting the blockers.

## Severity
🟢 LOW — LoRA usage is optional, CODA path works without it

## Location
`src/transformer.rs:1903-1908` (`forward_coda`)

## Problem
When a LoRA adapter is present, `forward_coda` falls back entirely to `forward_base`:
```rust
if lora.is_some() {
    return forward_base(ctx, weights, cache, token, pos, config, lora);
}
```

This means the CODA fused kernels (which eliminate ~8 buffer passes per layer) are completely bypassed. LoRA is often a small rank-8/16 perturbation — the bias term in `simd_matmul_residual_partial_rms` could absorb it.

## Proposed Fix
1. Compute LoRA output into the existing `lora_buf` (already pre-allocated in `ForwardContext`)
2. Pass the LoRA output as the `bias` parameter to CODA kernels
3. Only fall back to `forward_base` if LoRA requires weight mutation (it doesn't — it's additive)

```rust
// In forward_coda, instead of falling back:
let lora_bias_q = lora.map(|l| {
    crate::types::lora_apply(&mut ctx.lora_buf, l, &ctx.x, &mut ctx.lora_buf);
    ctx.lora_buf.as_slice()
});
```

This is a deeper change — the CODA kernels' bias parameter currently applies per-element. LoRA output has different shapes per projection. May need to restructure how LoRA bias is applied.

## Estimated Impact
- **~30-40% of forward path** regains CODA fusion when LoRA is active
- Depends on LoRA usage patterns

## Acceptance Criteria
- [ ] `forward_coda` works with LoRA without falling back to `forward_base`
- [ ] LoRA bias is passed through CODA kernel's `bias` parameter
- [ ] All LoRA + CODA tests pass
- [ ] Benchmark: CODA+LoRA matches CODA-without-LoRA within 5%
