//! SpecHop — Continuous Multi-Hop Speculation Pipeline (Plan 131).
//!
//! Hop-level speculative execution for multi-hop tool-use trajectories.
//! Maintains k speculative threads that predict tool-call observations ahead
//! of actual tool responses. When the target tool returns, a verifier checks
//! equivalence → commit correct branch, rollback incorrect ones.
//!
//! Theoretical framework (α, β, p) gives principled thread-count sizing via
//! `k* = ⌈(1+β)/(α+β)⌉`. Target: 25–40% wall-clock latency reduction on
//! multi-hop tool-use trajectories, lossless under verifier.
//!
//! **Feature gate:** `spechop` (opt-in, requires `bandit`)
//!
//! Reference: arXiv:2605.21965

pub mod cost_model;
pub mod hop_tree;
pub mod pipeline;
pub mod speculator;
pub mod types;
pub mod verifier;
pub mod window;

#[cfg(all(feature = "spechop", feature = "cache_prune"))]
pub mod segment_match;

/// How ScreeningPruner is applied across hops in SpecHop (Plan 171).
///
/// Local mirror of `katgpt_pruners::configurator_bandit::PrunerSchedule` —
/// defined here to avoid a katgpt-speculative → katgpt-pruners cycle
/// (katgpt-pruners already depends on katgpt-speculative for sr2am_configurator
/// forwarding). Only the two variants spechop uses are modeled; the
/// `EntropyRouted` variant (gated by `directional_credit` in katgpt-pruners)
/// is not relevant at the hop level.
///
/// When `thinking_prune` is enabled at the root, callers passing
/// `katgpt_pruners::PrunerSchedule` should convert via `.into()` or match
/// before calling `build_hop_dd_tree_with_schedule`.
#[cfg(feature = "thinking_prune")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SpechopSchedule {
    /// Apply the full ScreeningPruner at every hop/loop step.
    Uniform,
    /// Apply FrozenBaseGuard: intermediate hops accept all (relevance 1.0),
    /// only the final hop applies the full ScreeningPruner.
    /// This is the default — pure speedup, no quality loss.
    #[default]
    FrozenBaseGuard,
}

#[cfg(feature = "thinking_prune")]
impl SpechopSchedule {
    /// Returns true if this schedule applies full screening at every step.
    #[inline]
    pub fn is_uniform(&self) -> bool {
        matches!(self, Self::Uniform)
    }

    /// Determine if the given hop should apply full screening.
    ///
    /// For `FrozenBaseGuard`: only true when `hop_index == total_hops - 1`.
    /// For `Uniform`: always true.
    #[inline]
    pub fn should_screen_full(&self, hop_index: usize, total_hops: usize) -> bool {
        match self {
            Self::Uniform => true,
            Self::FrozenBaseGuard => hop_index >= total_hops.saturating_sub(1),
        }
    }
}

pub use cost_model::{
    InferenceStats, bounded_rel_lat, compute_optimal_k, oracle_rel_lat, should_activate_spechop,
    spechop_configurator_reward, starvation_prob,
};
pub use hop_tree::{
    HopCandidate, HopMarginal, HopTreeConfig, HopTreeNode, HopVerifyState, VerifiedHopPath,
    build_and_verify_hop_tree, build_hop_dd_tree, extract_best_hop_path, extract_deepest_hop_path,
    verify_hop_tree,
};

#[cfg(feature = "thinking_prune")]
pub use hop_tree::build_hop_dd_tree_with_schedule;
pub use pipeline::{PipelineResult, SpecHopPipeline, TrajectoryHop};
pub use speculator::{CacheSpeculator, HopSpeculator};
pub use types::{HopObservation, HopState, SpecError, SpecHopConfig, SpecOutcome};
pub use verifier::{ObservationVerifier, RuleBasedVerifier, token_set_jaccard};
pub use window::SpecWindow;

#[cfg(all(feature = "spechop", feature = "cache_prune"))]
pub use segment_match::{HopSegmentIndex, IndexedSegment, SegmentMatch};

#[cfg(feature = "bandit")]
pub use speculator::BanditSpeculator;

#[cfg(feature = "recfm")]
pub use speculator::{CrossHopConfig, observation_velocity};
