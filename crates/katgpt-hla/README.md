# katgpt-hla

[![crates.io](https://img.shields.io/crates/v/katgpt-hla.svg)](https://crates.io/crates/katgpt-hla)
[![Documentation](https://docs.rs/katgpt-hla/badge.svg)](https://docs.rs/katgpt-hla)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Higher-order Linear Attention (HLA) substrate — O(1) inference cache types
and streaming kernels. Pure substrate, depends only on `katgpt-types`.

## Overview

Implements second-order HLA (symmetric + asymmetric AHLA) as an alternative to
standard KV-cache attention. Achieves **O(1) per-token memory** independent of
sequence length by replacing the growing KV cache with fixed-size prefix
sufficient statistics that capture higher-order query-key interactions.

This crate contains the pure substrate half of HLA — the cache state structs
and zero-alloc streaming recurrence kernels. Both depend only on
`katgpt_types::simd` and `katgpt_types::Config`. Any crate can
`cargo add katgpt-hla` and get the HLA substrate without pulling
`katgpt-core` or the engine.

The composition layer (`forward_hla` / `forward_ahla`) lives in
`katgpt-forward` because it depends on `ForwardContext` (katgpt-core-only).
Cognitive extensions (`*_role_aware`, `ThirdOrderMoment`) stay in
`riir-engine`.

Reference: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."

## Key types / modules

- `types` — `HlaLayerState`, `HlaQHeadState`, `AhlaLayerState`, `AhlaQHeadState`,
  `MultiLayerHlaCache`, `MultiLayerAhlaCache`, Parallax variants
  (`MultiLayerParallaxAhlaCache`, `ParallaxAhlaLayerState`,
  `ParallaxAhlaQHeadState`), `HlaVariant`
- `kernel` — zero-alloc streaming recurrence kernels: `hla_state_update`,
  `hla_readout`, `hla_readout_normalized`, `hla_denom`, `ahla_step`,
  `ahla_denom`, `hla_layer_update`, `hla_layer_readout`, `ahla_layer_step`

## Feature flags

No feature flags — the substrate compiles unconditionally (`default = []`).
HLA is always-on inside `katgpt-core`.

## Dependencies

- `katgpt-types` (for `Config` and SIMD kernels)

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
