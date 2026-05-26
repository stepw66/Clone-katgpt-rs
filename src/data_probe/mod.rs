//! Data Probe Diagnostics — controlled information-theoretic validation.
//!
//! This module implements a "probe-LLM" framework: a Markov chain with known
//! ground-truth transition probabilities, entropy rate, and stationary distribution.
//! By generating sequences from a known source and comparing against model
//! estimates, we can formally validate information-theoretic claims (C1–C4).
//!
//! # Module layout
//!
//! - [`markov`]       — Dirichlet-sampled Markov chain generator
//! - [`nll`]          — NLL computation against known chain
//! - [`typical_set`]  — Three-way regime classification (Conservative/Typical/Uncertain)
//! - [`dirichlet_energy`] — Dirichlet Energy structural alignment diagnostic
//! - [`claim`]        — Claim card infrastructure for formal C1–C4 validation
//! - [`geometry`]     — Representation geometry diagnostics (Plan 151)

// ── Submodules ─────────────────────────────────────────────────

/// Dirichlet-sampled Markov chain generator with entropy rate targeting.
pub mod markov;

/// NLL computation against a known Markov chain.
pub mod nll;

/// Three-way regime classification based on typical-set framework.
pub mod typical_set;

/// Dirichlet Energy structural alignment diagnostic.
pub mod dirichlet_energy;

/// Claim card infrastructure for formal C1–C4 validation.
pub mod claim;

/// Representation geometry diagnostics (Plan 151, Research 113).
pub mod geometry;

// ── Re-exports ─────────────────────────────────────────────────

pub use claim::{ClaimCard, Intervention, ValidityVerdict};
pub use dirichlet_energy::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};
pub use geometry::{
    GeometryReport, avg_cosine_similarity, effective_rank, representation_geometry_report,
};
pub use markov::{MarkovChain, generate_markov_chain, sample_sequence};
pub use nll::{average_nll, nll_profile};
pub use typical_set::{Regime, RegimeDistribution, classify_regime, regime_distribution};
