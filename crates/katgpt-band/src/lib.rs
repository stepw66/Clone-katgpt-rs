//! Band-Conditioned KV Segment Selector cluster — Plan 265 (arXiv:2605.12733).
//!
//! Extracted from `katgpt-rs/src/{band_conditioner, bckvss, collider_pruner,
//! adaptive_cot_stopper}.rs` per Proposal 003 Phase 11 (2026-07-04).
//!
//! # Module map
//!
//! - `band_conditioner` — the cluster's foundation (BandConditioningSet,
//!   ComputeTarget, conditional_dependence_fisher_z). Sink of the cluster's
//!   internal dep graph.
//! - `bckvss` — Fusion A: Band-Conditioned KV Segment Selector.
//! - `collider_pruner` — Fusion C: ColliderConsistency ConstraintPruner for
//!   DDTree (impls `katgpt_core::{ConstraintPruner, PreservationScorer}`).
//! - `adaptive_cot_stopper` — Fusion D: theory-backed adaptive CoT stopping
//!   criterion. Standalone (no `crate::` deps despite the historical lib.rs
//!   comment claiming otherwise).
//!
//! # Why one crate
//!
//! These four modules form a tightly inter-coupled cluster: `bckvss` and
//! `collider_pruner` both build on `band_conditioner::ComputeTarget` and
//! `BandConditioningSet`; they share paper origin (Plan 265) and the
//! conditional-independence-test substrate. Splitting would either duplicate
//! the substrate or force awkward one-way deps.
//!
//! # External crate deps
//!
//! - `katgpt-core`: `sigmoid` (all four modules), `ConstraintPruner` +
//!   `PreservationScorer` traits (collider_pruner only).
//! - `fastrand`: deterministic RNG for `bckvss::SyntheticScm`'s AR(1) sampler.

#![allow(unexpected_cfgs)] // root may pass-through `full` and other aggregate features

#[cfg(feature = "band_conditioner")]
pub mod band_conditioner;

#[cfg(feature = "bckvss")]
pub mod bckvss;

#[cfg(feature = "collider_consistency")]
pub mod collider_pruner;

#[cfg(feature = "adaptive_cot_identifiability")]
pub mod adaptive_cot_stopper;
