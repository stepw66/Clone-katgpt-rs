# katgpt-sense

[![crates.io](https://img.shields.io/crates/v/katgpt-sense.svg)](https://crates.io/crates/katgpt-sense)
[![Documentation](https://docs.rs/katgpt-sense/badge.svg)](https://docs.rs/katgpt-sense)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

KG Latent Octree Sense substrate — octree construction, reconstruction, BAKE
precision-gated update, LOD routing, and serialization. Depends only on
`katgpt-types` (the leaf).

## Overview

Generic sense substrate for spatial perception: octree construction,
reconstruction, BAKE precision-gated embedding update, LOD routing,
schema-centroid initialization, sector projection, and `SenseModule`
serialization. NPCs compose modules at spawn time and query at ~45ns/tick via
bitwise dot-product.

Spun out of `katgpt-core::sense` (Issue 007 Phase E Tier 2 #7). The
NPC-runtime half (`brain`, `backend`, `batch`, `gm`, `hotswap`, `bandit`) had
already moved to `riir-engine::sense::*`. This crate holds the
publishable substrate half — zero `katgpt-core` dependency.

## Key types / modules

- `octree` — `KgEmbedding`, `SenseOctreeBuilder` (octree construction)
- `reconstruction` — `ReconstructionConfig`, `ReconstructionState`,
  `ReconstructionResult`, `OctreeNodeId`, `TripleEvidence`, `TraversalAction`,
  `compare_reconstruction`
- `bake` — BAKE precision-gated Bayesian embedding update: `bake_update`,
  `bake_update_precision`, `bake_update_mean`, `bake_regularize`,
  `precision_to_confidence`, `exploration_priority`
- `serialize` — `SenseModule` serialization

## Feature flags

`default = []`. Core substrate (bake, octree, reconstruction, serialize)
compiles unconditionally.

| Feature | Default | Description |
|---------|---------|-------------|
| `sense_composition` | no | Cached-weights SIMD-matvec path in reconstruction + paired test with `temporal_deriv` (Plan 221). |
| `temporal_deriv` | no | Temporal Derivative Kernel surprise channel in reconstruction (Plan 277). |
| `sense_lod` | no | Spectral NPC Perception Compression — LOD routing (`SenseLodLevel`, `SenseLodMask`, `SenseLodRouter`) (Plan 240). |
| `schema_centroid` | no | Schema-Centroid Informed KG Embedding Initialization (Plan 237). Pulls in `papaya`. |
| `sector_projection` | no | Multi-sector spatial projection (Plan 262). |
| `depth_invariance` | no | Depth-invariance audit integration for reconstruction (Plan 331). Forwards to `katgpt-types/depth_invariance`. |
| `merkle_octree` | no | Merkle octree build path (`build_with_merkle` / `build_merkle_only`) (Plan 221-M). |
| `bake_precision` | no | BAKE Precision-Gated Bayesian Embedding Update — `BakePrecisionStore`, `BakeSession`, `PrecisionEntry` (Plan 236). Pulls in `papaya`. |
| `self_advantage_gate` | no | Self-advantage recursion gate — advantage-margin early-stop in reconstruction (Plan 283 T5.1). |

## Dependencies

- `katgpt-types` (SIMD kernels, `ScaleBoundary`, `TemporalDerivativeKernel`,
  `MerkleOctree`/`MerkleProof`, depth-invariance classifier, `leaky_step`)
- `blake3`, `fastrand`
- `papaya` *(optional)* — pulled in by `schema_centroid` and `bake_precision`

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
