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

pub use cost_model::{
    InferenceStats, bounded_rel_lat, compute_optimal_k, oracle_rel_lat, should_activate_spechop,
    spechop_configurator_reward, starvation_prob,
};
pub use hop_tree::{
    HopCandidate, HopMarginal, HopTreeConfig, HopTreeNode, HopVerifyState, VerifiedHopPath,
    build_and_verify_hop_tree, build_hop_dd_tree, extract_best_hop_path, extract_deepest_hop_path,
    verify_hop_tree,
};
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
