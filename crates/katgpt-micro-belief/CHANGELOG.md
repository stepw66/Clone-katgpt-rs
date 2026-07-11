# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-micro-belief-v0.1.0...katgpt-micro-belief-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-micro-belief-v0.1.0) - 2026-07-11

### Added

- Plan 370 Phase 1-2 — QmcMethod enum + NoiseQueryConfig.qmc_method + fill_noise_queries_gaussian_qmc_by_method
- *(katgpt-micro-belief)* [**breaking**] promote micro-belief kernel to its own public crate (Issue 007 Phase E Tier 1 #3)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- clippy-clean all 13 modified crates (--tests)

### Other

- #[inline] on trivial &mut self setters
- derive Copy on 200+ primitive-field structs
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- fix 8 dangling issue references in code/test/bench comments
- workspace-wide optimization sweep across 13 crates
- SIMD reductions + zero-alloc scratch variants (round 2)
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
