# katgpt-dec

[![crates.io](https://img.shields.io/crates/v/katgpt-dec.svg)](https://crates.io/crates/katgpt-dec)
[![Documentation](https://docs.rs/katgpt-dec/badge.svg)](https://docs.rs/katgpt-dec)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Discrete Exterior Calculus (DEC) substrate — exterior derivative,
codifferential, Hodge Laplacian / decomposition, Stokes calculus, and lattice
utility. Pure math, zero dependencies, no application semantics.

## Overview

Based on "Topological Neural Operators" (arXiv:2606.09806). Provides the DEC
machinery for Stokes calculus on cell complexes: typed cochain fields, the
fundamental operators (gradient d₀, curl d₁, divergence d₂, codifferential δₖ),
the Hodge Laplacian (Δₖ = δₖ₊₁dₖ + dₖ₋₁δₖ), and the Hodge decomposition
(exact ⊕ harmonic ⊕ coexact / Helmholtz split).

The core identity `dₖ₊₁ ∘ dₖ = 0` holds exactly: `curl(grad) = 0` and
`div(curl) = 0` by construction — conservation-by-construction.

`katgpt-core` re-exports this crate as `katgpt_core::dec` for backwards
compatibility.

## Key types / modules

- `types` — `CellComplex`, `CochainField`, `CoboundaryIndex`, `MAX_RANK`
- `operators` — `exterior_derivative`, `codifferential`, `hodge_star`,
  `graph_laplacian`, `hodge_laplacian`
- `hodge` — `hodge_decompose`, `hodge_energy`, `hodge_spectrum`, `betti_numbers`,
  `harmonic_projector`
- `stokes_calculus` — `boundary_flux_mass`, `line_integral`, `circulation_integral`,
  `belief_mass_divergence`
- `flow` — `DecFlowField`, `exact_flow`, `coexact_flow`, `harmonic_flow`
- `cache` — `DecCache`, `hodge_decompose_cached` (incremental recomputation)
- `backend` — `DecBackend`, `select_backend` (dispatch by feature/target)
- `simd` — bit-compatible scalar fallback kernels (`simd_dot_f32`,
  `simd_sigmoid_inplace`)

## Feature flags

`default = ["heat_kernel_trajectory", "sheaf_admm"]`

| Feature | Default | Description |
|---------|---------|-------------|
| `heat_kernel_trajectory` | yes | Single-shot DEC heat-kernel trajectory predictor via precomputed eigendecomposition + Krylov online path (Plan 359). Also enables nonlinear exponential integrator (Duhamel + Gauss-Legendre) and BoM trajectory sampling. |
| `sheaf_admm` | yes | Modelless one-step Sheaf-ADMM coordination primitive over a cellular sheaf (Plan 407). |
| `motor_gated_field` | no | Amari-style motor-gated neural-field evolution step — Hodge Laplacian + ReLU gate + per-channel motor gain (Plan 357). |
| `htno_v_cycle` | no | Multi-scale V-cycle on cell complexes via selector restriction maps (fine→coarse→fine hierarchy). |
| `cochain_point_sampler` | no | Whitney/de-Rham continuous point sampler for modelless intra-primitive field queries (Plan 422). |

## Dependencies

Zero dependencies. `katgpt-dec` is a pure-math substrate that ships its own
minimal SIMD kernels so it can be re-exported by `katgpt-core` as
`katgpt_core::dec` without creating a cyclic package dependency.

## License

MIT. Part of the [katgpt-rs](https://github.com/katopz/katgpt-rs) project.
