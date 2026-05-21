# Plan 097: Delta Attention Residuals

**Status:** Planning
**Research:** 061 (Delta Attention Residuals)
**Related Plans:** 057 (HLA), 070 (SP-KV), 022 (Sparse MLP), 085 (Deep Manifold)
**Feature Gate:** `delta_routing` (off by default)

## Context

Research 061 distilled "Delta Attention Residuals" (NeurIPS 2026). Core idea: route over per-sublayer deltas (vi = hi+1 - hi) instead of cumulative hidden states for cross-layer information flow. Combined with additive routing, this gives 3× sharper routing and −8.2% PPL at 7.6B scale.

For our stack:
- At micro scale (n_layer=1): zero benefit (no previous layers)
- At n_layer≥6: measurable benefit expected based on paper's scaling trends
- Delta Block variant recommended: B+1 sources, ~20% throughput overhead
- Orthogonal to HLA, SP-KV, SpectralQuant — all operate on different axes

## Tasks

- [ ] T1: Add `delta_routing` feature flag to `Cargo.toml` (microgpt-rs) and `microgpt-core` types
- [ ] T2: Add `DeltaRoutingConfig` to `microgpt-core/src/types.rs` — block_size, mode enum (Off/DeltaBlock/DeltaAttnRes)
- [ ] T3: Add delta routing buffers to `ForwardContext` in `transformer.rs` — block_deltas Vec, delta_query weights, delta_rmsnorm weights
- [ ] T4: Implement `depth_route()` function in `transformer.rs` — softmax over delta sources, additive to residual
- [ ] T5: Integrate `depth_route()` into `forward_base()` layer loop — compute per-sublayer deltas, store block deltas, call routing before attention and MLP sublayers
- [ ] T6: Add `DeltaRoutingWeights` to `TransformerWeights` — per-layer query vectors (zero-init) and RMSNorm params
- [ ] T7: Benchmark: n_layer=6 config with/without `delta_routing`, measure PPL delta and throughput impact
- [ ] T8: GOAT proof test — verify routing sharpness (max weight ≥0.4 in deep layers) on small config

## Architecture Decision

**Delta Block only** — not per-sublayer Delta AttnRes:
- Per-sublayer: 2L sources → 69% throughput reduction at L=36
- Delta Block: B+1 sources (B=4 default) → ~20% throughput overhead
- Quality gap <0.2% per paper Table 4

**Additive routing only** — not replacement:
- Preserves residual stream (critical for our existing features)
- Safe zero-init (compatible with LoRA fine-tuning in riir-ai)
- No reset needed (simpler code)

## Implementation Notes

The depth_route() function pseudocode from the paper:
```
depth_route(sources, residual, proj, norm):
    V = stack(sources)          // [N, T, D]
    K = norm(V)                  // RMSNorm
    logits = dot(proj_weight, K) // per-source score
    weights = softmax(logits)    // routing weights
    return residual + weighted_sum(weights, V)  // additive
```

In our forward_base():
- After `matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n)` → `attn_out` is the attention delta
- After `matmul(&mut ctx.x, &layer_weights.mlp_w2, &ctx.hidden, n, config.mlp_hidden)` → the MLP output minus `xr2` is the MLP delta
- For Delta Block: accumulate deltas across layers within block, store when block boundary reached

## Feature Gate

```toml
[features]
delta_routing = []  # Delta Block cross-layer routing (Research 061, Plan 097)
```

Default: OFF. Requires n_layer ≥ 4 for meaningful benefit. Enable with `--features delta_routing`.

## Success Criteria

- [ ] Zero-cost when feature is OFF (no code hits in forward_base without feature)
- [ ] GOAT proof: routing sharpness ≥ 0.4 max weight at n_layer=6
- [ ] Throughput overhead ≤ 30% at n_layer=6 with B=2 blocks
- [ ] Memory overhead ≤ (B+1) × n_embd × sizeof(f32) per block delta storage