# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-personality-v0.1.0...katgpt-personality-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-personality-v0.1.0) - 2026-07-11

### Added

- *(katgpt-personality)* [**breaking**] promote personality_composition substrate to its own public crate (Issue 007 Phase E Tier 2 #5)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)

### Other

- #[inline] on trivial &self value getters
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
