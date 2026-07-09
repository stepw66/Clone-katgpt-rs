# Memory — Modelless Memory Primitives

> **What we sell here.** The memory substrate that lets a frozen model remember
> across context windows without gradient updates: O(1) routing slots,
> O(√N) factored retrieval, hash-addressed conditional patterns, attractor
> belief kernels, multi-modal sense composition, and eviction-time consolidation.

## Fusion map — the memory stack

```
   sense_composition.md (multi-modal sense → latent input)
        │
        ▼
   raven_rsm.md (O(1) routing slot memory)
        │
        ├── product_key_memory.md (O(√N) factored retrieval)
        ├── engram.md (hash-addressed conditional pattern memory)
        └── micro_belief.md (attractor / leaky belief kernel)
              │
              ▼
        sleep_consolidation.md (bake cache → fast weights at eviction)
```

| Doc | Role |
|---|---|
| [`raven_rsm.md`](raven_rsm.md) | Raven RSM — O(1) routing slot memory |
| [`product_key_memory.md`](product_key_memory.md) | PKM — O(√N) factored retrieval memory (Plan 408, default-ON) |
| [`engram.md`](engram.md) | Engram — hash-addressed conditional pattern memory (Plan 299) |
| [`micro_belief.md`](micro_belief.md) | MicroRecurrentBeliefState — attractor + leaky belief kernel (Plan 276) |
| [`sense_composition.md`](sense_composition.md) | NPC sense composition (Plans 221/230/235/236/237) |
| [`sleep_consolidation.md`](sleep_consolidation.md) | Sleep consolidation — offline recursive memory consolidation at eviction (Plan 154, default-ON) |

## See also

- [`../calibration/faithfulness_probe.md`](../calibration/faithfulness_probe.md) — causal diagnostic for *injected* memory
- [`../orientation/architecture.md`](../orientation/architecture.md) § Sleep-Time Query Anticipator — the artifact-emission sibling of sleep consolidation
