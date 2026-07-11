# katgpt-personality

[![crates.io](https://img.shields.io/crates/v/katgpt-personality.svg)](https://crates.io/crates/katgpt-personality)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Personality-Weighted Latent Layer Composition — sigmoid-gated N-layer latent
direction composition + reward-surprise drift + BLAKE3-committed snapshot.
Entity-agnostic, modelless, sigmoid (never softmax).

## Overview

A generic, modelless primitive: compose `N` latent direction vectors
`d_i ∈ ℝ^D` into a single behavior vector via a personality weight vector
`w ∈ ℝ^N` with sigmoid gating, and update `w` via an EMA on reward prediction
error.

```text
behavior = Σ_i sigmoid(w_i / τ) · belief_confidence_i · d_i
```

Drift rule (reward-surprise EMA):

```text
surprise_i = R_observed - R_expected_i
Δw_i = α · surprise_i · d_recent_i
w_i ← clamp(w_i + Δw_i, -w_max, +w_max)
```

**Why sigmoid, not softmax:** sigmoid allows a layer to contribute ~0 (the
agent ignores it) or ~1 (the agent embodies it) with signed resistance. Softmax
would destroy the "negative weight = resistance" semantics by always assigning
non-trivial probability to every layer.

The kernel is entity-agnostic — no game terms. It is `N × D` linear algebra +
sigmoid + EMA, applying equally to NPC, player, predator, prey, robot, or
recommender user.

## Key types / modules

- `types` — `PersonalityConfig`, `ArchetypeLabel`
- `sigmoid` — numerically stable branching sigmoid (`sigmoid`, `sigmoid_into`)
- `trait_def` — `LayerDirectionSource` trait (file named to avoid the `trait`
  keyword as a module path component)
- `kernel` — `PersonalityWeightedComposition<N, D>` (compose + drift + snapshot
  accessors). Pinned const-generic aliases: `SingleLayerComposition`,
  `QuadLayerComposition`, `HeptaLayerComposition`, `EntityCognitionComposition`
  (N=9, D=32).
- `snapshot` — `PersonalitySnapshot` with BLAKE3 commitment

## Feature flags

No feature flags — the substrate compiles unconditionally (`default = []`).
`katgpt-core` gates the re-export behind `personality_composition` (DEFAULT-ON
since Plan 297), but the substrate is always available via
`cargo add katgpt-personality`.

## Dependencies

- `katgpt-types` (SIMD kernels: `simd_fused_scale_acc`)
- `blake3`, `serde`

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
