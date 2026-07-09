# Raven RSM: O(1) Routing Slot Memory

> **Status: Opt-in alternative forward path — NOT in the default hot path.**
>
> The default `forward()` → `forward_base()` uses standard O(N) softmax attention.
> Raven is accessed via `forward_raven()` with `RavenKVCache` explicitly.

## What It Is

Fixed-size `[num_slots × kv_dim]` KV cache with sparse Top-K routing. Unselected slots are **completely frozen** — 10K noise updates leave passkey slots untouched. 2.98× faster than flat attention at pos=8.

## Architecture

```
RavenKVCache
├── slots: [num_slots × kv_dim]     // Fixed-size KV storage
├── router_buf: pre-allocated        // Top-K scoring buffer
├── readout_buf: pre-allocated       // Readout accumulation buffer
├── raven_update()                   // Gated EMA slot update with routing mask
├── raven_readout()                  // Fixed-slot readout (replaces scanning all positions)
└── raven_compute_router()           // Router scoring for top-k slot selection
```

## Evidence

| Property | Evidence |
|----------|----------|
| Frozen slots work | 10,000 noise updates, slot 12 identical to 6 decimals |
| O(1) stays flat | Raven stays 1.0× while flat grows 1.1× from pos 16→240 |
| 2.98× faster | 62,653 tok/s (Raven) vs 21,019 tok/s (flat) |

## Why Not Default

Raven replaces the standard KV cache with a fixed-slot architecture. This changes the forward pass semantics fundamentally:
- Requires explicit `RavenKVCache` construction
- `forward_raven()` is a separate function, not called from `forward()`
- No feature flag gates Raven — it's always compiled but must be explicitly invoked
- Best suited for draft models with fixed-size context windows

## Code Locations

| File | Content |
|------|---------|
| `src/transformer.rs` (~L4359) | `RavenKVCache`, `raven_update`, `raven_readout`, `forward_raven` |
| `src/benchmark/infrastructure.rs` | `bench_raven_vs_flat_cache()`, `bench_raven_recall()` |
| `src/benchmark/routing.rs` | Routing/MoE benchmarks |
| `examples/core_02_raven.rs` | Full demo |
| `tests/integration.rs` | Raven integration tests |

## Related

- [`.docs/08_lucebox_techniques.md`](08_lucebox_techniques.md) — Original Raven documentation
- GDN2 (`src/gdn2/`) — Another O(1) alternative via recurrent fast-weight state
- LT2 Looped (`lt2_looped`) — Weight-shared T-pass hybrid SDPA+AHLA
