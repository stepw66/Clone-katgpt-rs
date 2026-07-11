//! Data Probe Diagnostics — controlled information-theoretic validation.
//!
//! This module implements a "probe-LLM" framework: a Markov chain with known
//! ground-truth transition probabilities, entropy rate, and stationary distribution.
//! By generating sequences from a known source and comparing against model
//! estimates, we can formally validate information-theoretic claims (C1–C4).
//!
//! # Module layout
//!
//! - [`markov`]           — Dirichlet-sampled Markov chain generator (always-on)
//! - [`nll`]              — NLL computation against known chain (always-on)
//! - [`typical_set`]      — Three-way regime classification (always-on)
//! - [`dirichlet_energy`] — Dirichlet Energy structural alignment diagnostic (always-on)
//! - [`claim`]            — Claim card infrastructure for formal C1–C4 validation (always-on)
//! - [`geometry`]         — Representation geometry diagnostics (Plan 151) — gated `sink_aware_attn`
//! - [`sink_classify`]    — Per-head NOP/Broadcast sink classifier (Plan 287) — gated `sink_aware_attn`
//! - [`gold_share`]       — Content-specific output-fraction diagnostic (Plan 411) — gated `gold_share_probe`
//!
//! # Always-on vs feature-gated split
//!
//! The information-theoretic substrate (markov, nll, typical_set,
//! dirichlet_energy, claim) is always-on — it's pure math with no heavy deps.
//! The sink-aware classifier + its geometry bridge are gated
//! `sink_aware_attn` because they're an opt-in attention intervention
//! (Plan 287, default stays Uniform pending G2/G3 GOAT gate).
//!
//! # Mechanism locator vs aggregate symptom
//!
//! [`sink_classify`] is the **mechanism locator**: it identifies *which*
//! sink columns in an attention map are Adaptive NOPs vs Broadcasts.
//! [`geometry::effective_rank`] is the **aggregate symptom**: it measures
//! how collapsed the resulting hidden states are across the whole layer.
//! Broadcast sinks reduce `effective_rank` across tokens (Fesser et al.
//! Lemma 4); the classifier tells you *why*. Phase 4's `LayerSinkSummary`
//! (in [`geometry`]) bridges the two.
//!
//! [`gold_share`] (Plan 411, Research 392) adds the **content-specific**
//! view: it tells you whether the layer's output still carries the *gold*
//! signal or has been rewritten to carry aggregate noise — orthogonal to
//! both `effective_rank` (content-agnostic) and `stable_rank_update`
//! (per-sink degeneracy).
//!
//! # Plan 404 history (2026-07-06)
//!
//! The markov/nll/typical_set/dirichlet_energy/claim/geometry modules moved
//! here from root `src/data_probe/`. They're pure substrate (only deps:
//! katgpt-core + intra-module). Root keeps re-export shims so every
//! historical `katgpt_rs::data_probe::*` path resolves.

// ── Always-on submodules (pure information-theoretic substrate) ─────────

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

// ── Feature-gated submodules (sink-aware attention intervention) ────────

/// Representation geometry diagnostics (Plan 151, Research 113).
/// Gated `sink_aware_attn` because it depends on [`sink_classify`] types.
#[cfg(feature = "sink_aware_attn")]
pub mod geometry;

/// Sink-Aware Attention classifier — per-head NOP/Broadcast detection
/// (Plan 287, Research 258, arxiv 2606.08105).
#[cfg(feature = "sink_aware_attn")]
pub mod sink_classify;

/// GoldShare content-specific output-fraction diagnostic (Plan 411,
/// Research 392, arxiv 2607.01538). `‖a^G_L‖ / ‖a_L‖` — the gold fraction
/// of a layer's attention output. Detects the paper's recall-generation
/// gap (signal in heads, lost in residual). Standalone norm computation
/// (no deps); gated `gold_share_probe`.
#[cfg(feature = "gold_share_probe")]
pub mod gold_share;

// ── Re-exports (always-on items) ────────────────────────────────────────

pub use claim::{ClaimCard, Intervention, ValidityVerdict};
#[cfg(feature = "dirichlet_energy")]
pub use dirichlet_energy::{
    consecutive_adjacency, dirichlet_energy, functor_adjacency, kv_cache_dirichlet_energy,
};
pub use markov::{MarkovChain, generate_markov_chain, sample_sequence};
pub use nll::{average_nll, nll_profile};
pub use typical_set::{Regime, RegimeDistribution, classify_regime, regime_distribution};

// ── Re-exports (sink-aware items, gated) ────────────────────────────────
//
// These mirror the historical `pub use data_probe::{...}` block that lived
// in katgpt-core's lib.rs. They're kept here so `crate::data_probe::SinkKind`
// etc. resolve for internal consumers (notably `parallax_attn.rs`) AND so
// root's `katgpt_rs::data_probe::sink_classify::*` re-export path works.

#[cfg(feature = "sink_aware_attn")]
pub use geometry::{
    GeometryReport, LayerSinkSummary, WithinClassGeometryReport, avg_cosine_similarity,
    effective_rank, representation_geometry_report, summarize_layer_sinks,
    within_class_effective_rank, within_class_effective_rank_owned, within_class_geometry_report,
};

#[cfg(feature = "sink_aware_attn")]
pub use sink_classify::{
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};

// ── Re-exports (gold_share items, gated) ───────────────────────────────

#[cfg(feature = "gold_share_probe")]
pub use gold_share::{GoldShareReport, GoldShareScratch, gold_share, gold_share_flat};
