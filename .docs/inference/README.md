# Inference — The Speculative Decoding + Search Engine

> **What we sell here.** The core inference path: speculative drafting +
> verification, multi-hop speculation, KV-cache compression, multi-token
> prediction thresholds, and graph search. Everything that accelerates or
> extends the single-pass autoregressive decode.

## Fusion map — how the pieces compose

```
   speculative_decoding.md (DDTree + DFlash + Leviathan verify)
        │
        ├── spechop.md (continuous multi-hop on top of the draft tree)
        ├── mtp_threshold.md (when to trust a multi-token-prediction draft)
        └── progressive_mcgs.md (graph search w/ reference edges)
              │
              ▼
        kv_compression.md (the cache the above drafts read from / write to)
```

| Doc | Role |
|---|---|
| [`speculative_decoding.md`](speculative_decoding.md) | DDTree marginal-distribution trees, DFlash fast marginal prediction, Leviathan verification, D2F discrete-diffusion forcing |
| [`spechop.md`](spechop.md) | SpecHop — continuous multi-hop speculation pipeline (Plan 131, feature `spechop`) |
| [`mtp_threshold.md`](mtp_threshold.md) | MTP threshold guide — when multi-token-prediction drafts are worth accepting (Plan 055 + Plan 117) |
| [`kv_compression.md`](kv_compression.md) | KV cache compression research & alternatives (TurboQuant → SpectralQuant → OCTOPUS) |
| [`progressive_mcgs.md`](progressive_mcgs.md) | Progressive MCGS — Monte Carlo graph search with reference edges |

## See also

- [`../orientation/architecture.md`](../orientation/architecture.md) — where each inference primitive plugs into the core pipeline
- [`../feature_catalog/opt_in_features.md`](../feature_catalog/opt_in_features.md) — the opt-in feature-flag reference
