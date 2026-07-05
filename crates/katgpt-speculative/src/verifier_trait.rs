//! Speculative Verifier trait — strategy pattern for verifying drafted tokens
//! against a target distribution.
//!
//! **Hosted here (Plan 389, 2026-07-05)** rather than in katgpt-core because
//! the trait signature uses [`TransformerWeights`] (katgpt-transformer), which
//! sits *above* katgpt-core in the dep graph. katgpt-speculative already
//! consumes katgpt-transformer (via the optional `katgpt-transformer` dep,
//! gated by `parallel_probe`), so the trait lives alongside its primary
//! consumer (`parallel_probe`).
//!
//! The two concrete implementations — `SimulatedVerifier` (DDTree path +
//! acceptance cap, no target model) and `LeviathanVerifier` (real p/q
//! rejection sampling with target model) — stay in
//! `katgpt-rs/src/speculative/verifier.rs` because they consume
//! `crate::transformer::forward` (the forward-cycle architectural blocker,
//! see Proposal 003 Phase 16 DEFER section).
//!
//! Same pattern as `ConstraintPruner` / `ScreeningPruner` — trait-based swap
//! point. Root re-exports the trait at `katgpt_rs::speculative::verifier::*`
//! for back-compat.

use katgpt_transformer::TransformerWeights;
use katgpt_types::{Config, Rng};

/// Strategy for verifying drafted tokens against a target distribution.
///
/// Same pattern as `ConstraintPruner` — trait-based swap point.
/// - `SimulatedVerifier`: fast, no target model needed (default).
/// - `LeviathanVerifier`: real p/q rejection sampling with target model.
///
/// Both impls live in `katgpt-rs/src/speculative/verifier.rs` (root) because
/// they consume `crate::transformer::forward`. The trait itself has zero
/// `forward` deps — it only references the weights/config types — so it can
/// live below the forward-cycle.
pub trait SpeculativeVerifier: Send + Sync {
    /// Run one speculative decoding step end-to-end.
    /// Returns accepted tokens (always ≥ 1, up to γ + 1 with bonus).
    fn speculate(
        &mut self,
        draft_weights: &TransformerWeights,
        draft_config: &Config,
        token: usize,
        pos: usize,
        rng: &mut Rng,
    ) -> Vec<usize>;
}
