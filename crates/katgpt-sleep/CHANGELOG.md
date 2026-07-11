# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-sleep-v0.1.0) - 2026-07-11

### Added

- *(katgpt-sleep)* [**breaking**] promote sleep_time substrate to its own public crate (Issue 007 Phase E Tier 2 #6)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- clippy-clean all 13 modified crates (--tests)
- *(clippy)* needless_range_loop + assertions_on_constants in small crates

### Other

- repo-wide rustfmt pass (import/module reorder + line wrapping)
- workspace-wide optimization sweep across 13 crates
- SIMD dist_sq in wake-time consume + hoist blend loop-invariant
