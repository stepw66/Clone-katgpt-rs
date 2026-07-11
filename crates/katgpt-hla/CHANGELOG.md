# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/katopz/katgpt-rs/compare/katgpt-hla-v0.1.0...katgpt-hla-v0.1.1) - 2026-07-11

### Fixed

- add README.md for 7 published crates — crates.io page was blank

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-hla-v0.1.0) - 2026-07-11

### Added

- *(007)* Phase F.4a+F.4b — migrate 3 feasible composition files to leaves
- *(katgpt-hla)* [**breaking**] promote HLA substrate to its own public crate (Issue 007 Phase E Tier 2 #4)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)

### Other

- repo-wide rustfmt pass (import/module reorder + line wrapping)
- HLA transpose-matvec DRY + paged KV ensure_pages zero-alloc fast path
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
