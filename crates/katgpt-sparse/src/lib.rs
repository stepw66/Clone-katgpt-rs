//! Sparse task-vector family — SOPTV/SPLAT (Plan 264/265).
//!
//! Extracted from `katgpt-rs/src/{sparse_task_vector, specialist_projection}.rs`
//! per Proposal 003 Phase 11 (2026-07-04).
//!
//! # Module map
//!
//! - `sparse_task_vector` — Sparse Off-Principal Task Vector storage.
//!   OPD-grounded sparse delta format. The cluster's foundation.
//! - `specialist_projection` — SPLAT specialist latent projection. Consumes
//!   `sparse_task_vector::SparseTaskVector` (intra-crate) and
//!   `katgpt_band::band_conditioner::ComputeTarget` (cross-crate).
//!
//! # Cross-crate edge
//!
//! `specialist_projection` reads `katgpt_band::band_conditioner::ComputeTarget`
//! — a 5-variant enum owned by the band cluster. The dependency direction is
//! clean (sparse depends on band, never the reverse).
//!
//! # Feature gates
//!
//! - `sparse_task_vector` — base storage.
//! - `specialist_projection` — SPLAT projection. Implies `sparse_task_vector`
//!   and pulls in `katgpt-band` for `ComputeTarget`.
//! - `gauge_invariant` — gates the `compose_gauge_invariant` impl block on
//!   `SparseTaskVector`. The impl is self-contained; the parity test uses
//!   `katgpt-spectral` (dev-dep only).

#![allow(unexpected_cfgs)]  // root may pass-through aggregate features like `full`

#[cfg(feature = "sparse_task_vector")]
pub mod sparse_task_vector;

#[cfg(feature = "specialist_projection")]
pub mod specialist_projection;
