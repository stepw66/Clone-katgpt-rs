# Issue 098: forward_raven() Resizes and Fills Query Buffer Per Head in Hot Loop

## Severity: Medium
## Files: `katgpt-rs/src/transformer.rs` (L3844, L3884-3885)

## Description
`forward_raven()` calls `ctx.raven_query_buf.resize()` and `fill(0.0)` inside the per-layer and per-head loops. On the first call, `resize()` may allocate/reallocate. The `fill(0.0)` is called per head, zeroing `kvd` elements each time when once per layer would suffice.

Since `kvd` and `num_slots` are fixed per model, this buffer should be pre-allocated to `max(num_slots, kvd)` at init and zeroed once per layer, not per head.

## Fix
1. Pre-allocate `raven_query_buf` to `max(num_slots, kvd)` in `ForwardContext::new`.
2. In `forward_raven`, replace per-head `resize() + fill()` with a single `fill(0.0)` before the head loop, using only the relevant slice per head.

## Impact
Medium — eliminates potential allocation in hot loop and removes redundant zeroing (n_head × kvd per layer instead of kvd per layer).
