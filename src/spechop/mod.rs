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
pub mod pipeline;
pub mod speculator;
pub mod types;
pub mod verifier;
pub mod window;

pub use cost_model::{bounded_rel_lat, compute_optimal_k, oracle_rel_lat, starvation_prob};
pub use pipeline::{PipelineResult, SpecHopPipeline, TrajectoryHop};
pub use speculator::{CacheSpeculator, HopSpeculator};
pub use types::{HopObservation, HopState, SpecError, SpecHopConfig, SpecOutcome};
pub use verifier::{ObservationVerifier, RuleBasedVerifier, token_set_jaccard};
pub use window::SpecWindow;

#[cfg(feature = "bandit")]
pub use speculator::BanditSpeculator;
