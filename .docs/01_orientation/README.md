# Orientation — What Is This Project

> **What you find here.** The three entry docs: a capability overview, the full
> core-architecture reference, and the paper → feature comparison matrix that
> grounds every primitive in its source paper.

## Docs

| Doc | Role |
|---|---|
| [`overview.md`](overview.md) | What katgpt-rs is: CPU-first GPT-2 engine, speculative decoding, capability + throughput list, feature-gate inventory |
| [`architecture.md`](architecture.md) | The full core-architecture reference — every primitive, its crate path, feature gate, and examples |
| [`paper_feature_comparison.md`](paper_feature_comparison.md) | Matrix mapping each shipped feature to its source paper / arXiv ID |

## Where to start

New readers: `overview.md` → `architecture.md` → the relevant primitive-class
folder (`inference/` … `game_arenas/`). The comparison matrix is the
fastest way to answer "which paper does feature X come from?"

## See also

- [`../README.md`](../README.md) — top-level doc index
