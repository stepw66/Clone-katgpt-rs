//! PPoT core primitives: types, knowledge, entropy, rank.
//!
//! Extracted from `katgpt-rs/src/speculative/ppot/` (Issue 003). The
//! `resample` module stays in root (12 root-only refs); this crate owns the
//! four zero-root-coupling leaves that `resample` consumes via re-export.
//!
//! See `katgpt-rs/src/speculative/ppot/mod.rs` for the full PPoT architecture
//! doc — this is the moved-algorithm half, not the rescue orchestrator.

pub mod entropy;
pub mod knowledge;
pub mod rank;
pub mod types;

// ── Re-exports: Plan 026 Core ──────────────────────────────────

pub use entropy::{
    identify_high_entropy_positions, identify_high_entropy_positions_into,
    identify_high_entropy_positions_with_entropy_into, identify_positions_by_rule,
    identify_positions_by_rule_into, token_entropy,
};

pub use types::{PpotConfig, QmcConfig, QmcMethod, TokenRule};

// ── Re-exports: Plan 027 Adaptive ──────────────────────────────

pub use entropy::{identify_positions_adaptive, identify_positions_adaptive_into};

pub use knowledge::{ErrorKind, RejectionInsight, SessionKnowledge};

pub use rank::{rank_by_consistency, rank_by_consistency_weighted, select_best_variant};
