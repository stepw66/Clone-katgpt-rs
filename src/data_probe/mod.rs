//! Data Probe Diagnostics — root re-export shim.
//!
//! Plan 404 (2026-07-06): all substrate modules moved to
//! `katgpt_core::data_probe`. This file is now a thin re-export shim that
//! preserves every historical `katgpt_rs::data_probe::*` import path.
//!
//! See `crates/katgpt-core/src/data_probe/mod.rs` for the implementation and
//! the always-on vs `sink_aware_attn`-gated module split.

// ── Re-export substrate from katgpt-core ────────────────────────────────

/// Dirichlet-sampled Markov chain generator with entropy rate targeting.
pub mod markov {
    pub use katgpt_core::data_probe::markov::*;
}
/// NLL computation against a known Markov chain.
pub mod nll {
    pub use katgpt_core::data_probe::nll::*;
}
/// Three-way regime classification based on typical-set framework.
pub mod typical_set {
    pub use katgpt_core::data_probe::typical_set::*;
}
/// Dirichlet Energy structural alignment diagnostic.
pub mod dirichlet_energy {
    pub use katgpt_core::data_probe::dirichlet_energy::*;
}
/// Claim card infrastructure for formal C1–C4 validation.
pub mod claim {
    pub use katgpt_core::data_probe::claim::*;
}

// ── Sink-aware modules (gated `sink_aware_attn`) ────────────────────────

/// Representation geometry diagnostics (Plan 151, Research 113).
#[cfg(feature = "sink_aware_attn")]
pub mod geometry {
    pub use katgpt_core::data_probe::geometry::*;
}

/// Sink-Aware Attention classifier — per-head NOP/Broadcast detection
/// (Plan 287, Research 258, arxiv 2606.08105).
///
/// The substrate lives in `katgpt_core::data_probe::sink_classify`; this
/// file re-exports it and hosts the G1 classifier correctness tests.
#[cfg(feature = "sink_aware_attn")]
pub mod sink_classify;

// ── Re-exports (preserve flat `katgpt_rs::data_probe::*` paths) ─────────
//
// Note: `dirichlet_energy` (the function) is NOT flat-re-exported here because
// it would collide with the `dirichlet_energy` module name above. Access it as
// `katgpt_rs::data_probe::dirichlet_energy::dirichlet_energy` (mirrors the
// historical root structure before Plan 404).

pub use katgpt_core::data_probe::{
    ClaimCard, Intervention, ValidityVerdict, average_nll, classify_regime,
    consecutive_adjacency, functor_adjacency, generate_markov_chain,
    kv_cache_dirichlet_energy, nll_profile, regime_distribution, sample_sequence,
    MarkovChain, Regime, RegimeDistribution,
};

#[cfg(feature = "sink_aware_attn")]
pub use katgpt_core::data_probe::{
    GeometryReport, LayerSinkSummary, avg_cosine_similarity, effective_rank,
    representation_geometry_report, summarize_layer_sinks,
    CachedSinkClassification, SinkAwarePolicy, SinkClassifierConfig, SinkDiagnostic, SinkKind,
    StableRankScratch, apply_dual_policy_gate, apply_dual_policy_gate_cached,
    apply_dual_policy_gate_cached_flat, apply_dual_policy_gate_flat, classify_all_sinks,
    classify_all_sinks_flat, classify_sink_at, classify_sink_at_flat, stable_rank_update_into,
    stable_rank_update_into_flat,
};
