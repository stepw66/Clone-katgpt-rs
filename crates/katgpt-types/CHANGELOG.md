# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-types-v0.1.0...katgpt-types-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-types-v0.1.0) - 2026-07-11

### Added

- Plan 410 3A.2 — Clone derive on LoraAdapter
- *(types,pruners)* Issue 019 Phase C.1+C.3 upstream promotions
- *(core)* 004 adaptive causal calibration open primitive — cheap-proxy escalate (Phase 1+2)
- *(calibration)* Plan 358 Phase 4 — RTPurbo wiring + promote/demote (causal head-importance)
- *(katgpt-types)* co-extract MerkleOctree + MerkleProof to leaf (Plan 338 Phase 2.5)
- *(katgpt-types)* co-extract TemporalDerivativeKernel<N> to leaf (Plan 338 Phase 2)
- *(katgpt-types)* co-extract ScaleBoundary to leaf (Plan 338 Phase 1)
- *(katgpt-micro-belief)* [**breaking**] promote micro-belief kernel to its own public crate (Issue 007 Phase E Tier 1 #3)
- *(katgpt-types)* [**breaking**] promote types+simd substrate to its own public crate (Issue 007 Phase E Tier 1 #2)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- resolve all cargo clippy warnings/errors across crates
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- 2 remaining test-surface clippy warnings (useless_vec + needless_range_loop)
- scrub private code-symbol leaks from public doc comments (issue 360 class A)
- *(docs+hla)* resolve Issue 009 — ahla_step math divergence was a bug, not a variant
- *(clippy)* needless_range_loop + assertions_on_constants in small crates

### Other

- derive Copy on 200+ primitive-field structs
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- workspace-wide optimization sweep across 13 crates
- extract katgpt-kv + katgpt-spectral crates (Issue 015)
- *(simd)* simd_l_inf_distance_f32 + blocked argmax_pair (riir-neuron-db Issue 003)
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
