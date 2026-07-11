# katgpt-micro-belief

[![crates.io](https://img.shields.io/crates/v/katgpt-micro-belief.svg)](https://crates.io/crates/katgpt-micro-belief)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Micro-recurrent belief-state kernel — implicit per-entity state tracking.
Single-paper distillation (arXiv:2604.17121). Depends only on `katgpt-types`
(SIMD kernels) + `blake3` + `fastrand`.

## Overview

Small frozen recurrent kernels implementing `s_t = f(s_{t-1}, x_t)` over a
fixed-size latent belief vector, applied once per (entity, tick). The belief
vector is **latent and local** (never synced); a bridge projects it to
**bounded raw scalars** that cross the sync boundary.

The kernel weights are frozen — no training, no backprop. This is a modelless
inference primitive: the only weight mutations are freeze/thaw via
BLAKE3-committed snapshots.

Source paper: [arXiv:2604.17121](https://arxiv.org/abs/2604.17121) — Mozer et
al., DeepMind, Jun 2026.

## Key types / modules

- `types` — `MicroRecurrentBeliefState` trait, `RecurrenceFamily` enum,
  `KernelConfig`
- `attractor` — `AttractorKernel` (Family A, the GOAT candidate)
- `leaky` — `LeakyIntegrator` (Family C, standalone mirror of `evolve_hla`)
- `latent_thought` — `LatentThoughtKernel` (Family B)
- `bridge` — `project_to_scalars` (latent → raw scalar projection)
- `snapshot` — `MicroRecurrentKernelSnapshot` (BLAKE3-committed freeze/thaw
  artifact)
- `coherence_bench` — coherence benchmark harness
- `bom` *(opt-in)* — `BoMSampler`, K-hypothesis single-pass belief sampling
  (Plan 281)
- `bom_arena` *(opt-in)* — G2 arena harness for BoM planner comparison

## Feature flags

`default = []`. Core kernels (attractor, leaky, latent_thought, bridge,
snapshot) compile unconditionally.

| Feature | Default | Description |
|---------|---------|-------------|
| `bom_sampling` | no | K-hypothesis single-pass belief sampling (Plan 281, Research 248). Enables `bom` + `bom_arena` submodules. |
| `simd_sigmoid` | no | SIMD-vectorized sigmoid→tanh→clamp fused pass for `AttractorKernel::step()` + `BoMSampler::sample_k_states`. Auto-enabled by `bom_sampling`. |
| `depth_invariance` | no | Audit-only depth-invariance probes for `AttractorKernel` + `LeakyIntegrator` (Plan 306 T7.4). Forwards to `katgpt-types/depth_invariance`. |

## Dependencies

- `katgpt-types` (SIMD kernels, leaky_step, depth-invariance types)
- `blake3`, `fastrand`, `serde`

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
