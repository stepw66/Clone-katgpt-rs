# katgpt-types

[![crates.io](https://img.shields.io/crates/v/katgpt-types.svg)](https://crates.io/crates/katgpt-types)
[![Documentation](https://docs.rs/katgpt-types/badge.svg)](https://docs.rs/katgpt-types)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Shared configuration, RNG, math utilities, SIMD kernels, and inference types
for the [`katgpt-rs`](https://github.com/katopz/katgpt-rs) / `riir-engine`
inference framework. Pure substrate leaf — no `katgpt-*` dependencies.

## Overview

The foundational leaf crate that every other `katgpt-*` crate (and
`riir-engine`) builds on. Contains the shared `Config` struct (~1.5k lines),
XorShift64 PRNG, SIMD-accelerated linear-algebra kernels (softmax / rmsnorm /
matmul / ternary), CPU-side LoRA adapter types, domain embeddings, and
inference result types.

The `types` and `simd` modules are co-located because `types::math` calls
`simd` kernels and `simd::ternary` uses `types::TernaryWeights` — they form a
tight bidirectional leaf that cannot be split further without breaking the
cycle.

## Key types / modules

- `Config` / `InferenceOverrides` / `kv_dim` — the central inference configuration
- `Rng` — XorShift64 PRNG
- `simd` — NEON / AVX2 / WASM-SIMD128 / scalar fallback kernels
- `math` — softmax, rmsnorm, matmul, sample_token, gegelu, silu, swiglu
- `lora` — `LoraAdapter`, `LoraPair`, `lora_apply`
- `merkle` — `MerkleOctree`, `MerkleProof` (BLAKE3 commitment)
- `kv_cache` — `QuantizedKVCache` trait (shared extension point)
- `leaky_core` — `leaky_step` (shared delta-rule primitive)
- `slod` — `ScaleBoundary` (spectral LOD routing)
- `temporal` — `TemporalDerivativeKernel`, `sigmoid_surprise_gate`
- `sense` — `SenseModule`, `ShardEmbedding`, `TernaryDir`, `DilationConfig`

## Feature flags

`default = []`. The following are **tracking flags** — they gate re-export
visibility (mirroring the historical feature surface in `katgpt-core`), not
structural code. All core substrate compiles unconditionally.

| Feature | Description |
|---------|-------------|
| `domain_latent` | DomainLatent embedding type (Plan 038) |
| `gpart_adapter` | GPart Isometric Partition adapter (Research 227) |
| `data_gate` | Task-level gating for self-play stability (Plan 111) |
| `sr2am_configurator` | SR²AM Configurator context types (Plan 112) |
| `hydra_budget` | Hydra adaptive layer budget types (Plan 165) |
| `deltanet_inference` | DeltaNet layer type enum (Plan 182) |
| `collapse_aware_thinking` | Collapse-aware ThinkingBudget enum (Plan 212) |
| `wall_attention` | Wall Attention WallConfig (Plan 173) |
| `sparse_mlp` | Sparse MLP matmul kernel (Plan 022) |
| `plasma_path` | Bit-plane ternary weights (Plan 148) |
| `maxsim` | MaxSim late-interaction scoring (Plan 080) |
| `sigmoid_margin` | Sigmoid margin loss (implies `maxsim`) |
| `rim_slots` | RiM Reasoning Buffer Slots (Plan 172) |
| `belief_drafter` | NextLat Belief-State Speculative Drafter (Plan 217) |
| `sia_feedback` | SIA FeedbackBandit enum variants (Plan 163) |
| `spectral_threat` | LinOSS Modal Threat Prediction fields (Plan 241) |
| `gpart_pruning` | GPart pruning stats (Plan 257) |
| `depth_invariance` | Depth-Invariance Diagnostic (Plan 306) |

## Dependencies

- `fastrand`, `blake3`, `serde`, `half`, `rayon`

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
