# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-sense-v0.1.0) - 2026-07-11

### Added

- *(katgpt-sense)* [**breaking**] promote sense substrate to standalone crate (Plan 338 Phase 3)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- resolve clippy warnings — clone-on-Copy, loop-variable indexing, too_many_arguments allow
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- clear all-features clippy errors + warnings across 7 files
- clippy-clean all 13 modified crates (--tests)

### Other

- derive Copy on 200+ primitive-field structs
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- workspace-wide optimization sweep across 13 crates
