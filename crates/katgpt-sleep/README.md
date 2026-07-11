# katgpt-sleep

Sleep-Time Query Anticipator — offline query anticipation substrate.

## Overview

Implements the **open half** of Sleep-Time Compute ([arXiv:2504.13171](https://arxiv.org/abs/2504.13171) — Lin et al., Letta/Berkeley): a generic, game-semantic-free math primitive for offline query anticipation.

At **sleep-time** (offline), pre-compute answers for the queries an entity is *likely* to be asked. Store them in an `AnticipatedQuerySet` — the "c' artifact" (BLAKE3-committed). At **wake-time** (online), do a cheap dot-product + sigmoid-gated lookup into c'; fall through to fresh compute only on unpredictable queries. One sleep-time compute serves many wake-time consumers.

## Pipeline

```text
Sleep-time (offline, once per c):
  for i in 0..K:
    z_i = sleep_compute(c, D_set[i], budgets[i])   // consumer-provided op
    p_i = predictability(c, D_set[i])              // sigmoid(dot(c, dir))
  c' = AnticipatedQuerySet { slots: [(D_i, z_i, p_i)], blake3, version }

Wake-time (online, per query):
  i* = argmax_i dot(q, D_set[i])
  gate = sigmoid(beta * (p_{i*} − tau))
  out = gate * z_{i*} + (1 − gate) * fresh_think(q)
```

## Key types / modules

- `SleepTimeAnticipator` — orchestrates per-direction sleep-time compute
- `AnticipatedQuerySet` — the c' artifact (BLAKE3-committed, versioned)
- `consume()` — wake-time dot-product + sigmoid-gated lookup
- `predictability` — sigmoid(dot(c, dir)) score per direction
- `cost_model` — budget allocation for sleep-time compute

## Feature flags

No feature flags — the substrate compiles unconditionally. The `sleep_time_anticipation` Cargo feature in katgpt-core forwards to `dep:katgpt-sleep` (turns the re-export on); it does not gate anything inside this crate.

## Dependencies

- [`katgpt-types`](https://crates.io/crates/katgpt-types) — SIMD kernels + Config
- `blake3` — commitment root for the c' artifact

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
