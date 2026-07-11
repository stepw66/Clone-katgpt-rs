# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/katopz/katgpt-rs/releases/tag/katgpt-dec-v0.1.0) - 2026-07-11

### Added

- add bench_422_cochain_point_sampler GOAT perf gates (G4+G5 PASS)
- implement cochain point sampler primitive (Plan 422, Research 404)
- *(dec)* Plan 413 — multi-scale V-cycle primitive (htno_v_cycle)
- *(dec)* P407 Phase 3 — sheaf ADMM amplification (T3.1+T3.2+T3.3 all PASS)
- *(dec)* P407 Phase 2 sheaf_admm GOAT gate G1-G6 + promote to default ([#407](https://github.com/katopz/katgpt-rs/pull/407))
- *(dec)* P407 Phase 1 sheaf_admm skeleton — SheafMaps + LocalObjective + sheaf_admm_step ([#407](https://github.com/katopz/katgpt-rs/pull/407))
- Plan 370 Phase 4 — DEC-cochain fusion exploration (T4.1-T4.3)
- *(dec)* Plan 359 Phase 4 — BoM trajectory sampler (near-harmonic perturbation)
- *(dec)* Plan 359 Phase 3 — nonlinear exponential integrator (Duhamel + Gauss-Legendre)
- Plan 359 Phase 5 GOAT — heat_kernel_trajectory PROMOTED to DEFAULT-ON
- Plan 359 Phase 2 — Krylov expmv heat kernel trajectory (online path)
- *(dec)* Plan 359 Phase 1 — DEC heat kernel trajectory predictor (linear path)
- *(dec)* add Clone+Debug derives to CochainField
- motor-gated DEC field primitive (Plan 357, Research 359)
- *(katgpt-dec)* [**breaking**] promote DEC substrate to its own public crate (Issue 007 Phase E Tier 1 #1)

### Fixed

- *(release)* make sibling crates publishable — add version specs, flip publish flags
- resolve all remaining clippy warnings — clone-on-Copy, needless_range_loop, ptr_arg, redundant_field_names, too_many_arguments, manual_div_ceil across 6 crates (katgpt-core, katgpt-dec, katgpt-kv, katgpt-pruners, katgpt-spectral, katgpt-speculative, katgpt-attn)
- *(clippy+feat)* katgpt-dec sheaf_admm clippy + fastrand feature forwarding for RngLite
- *(clippy)* resolve all clippy warnings in katgpt-rs
- clean clippy warnings across workspace
- resolve all cargo clippy warnings/errors across crates
- *(007)* align all 18 substrate leaf crates to publish=false (policy A)
- clippy-clean all 13 modified crates (--tests)
- *(clippy)* needless_range_loop + assertions_on_constants in small crates
- *(clippy)* auto-fix batch for katgpt-{core,dec,types,sleep,transformer}

### Other

- *(dec)* line_integral O(P×E) → O(P+|E|) via one-shot vertex-pair lookup
- clippy cleanup across katgpt-core + katgpt-dec (iterator zips, allow attrs, unused-import removal)
- total_cmp conversions in remaining forward/dec/percepta/kv paths
- hot-path micro-optimizations across crates
- repo-wide rustfmt pass (import/module reorder + line wrapping)
- *(dec)* Issue 037 — extract duplicated test helpers into tests/common/
- grid-stencil fast path closes Plan 357 G5 (120µs → 29µs, 4.1× speedup)
- workspace-wide optimization sweep across 13 crates
- SIMD reductions + zero-alloc scratch variants (round 2)
- hot-path optimizations across katgpt-{core,dec,hla,micro-belief,personality,transformer,types}
- *(dec)* migrate eggshell IP out of public katgpt-dec → riir-neuron-db (Issue 008)
- *(katgpt-dec)* spatially-pruned splat for SafetyCochain::from_projectile_threat
