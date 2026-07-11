//! FlashAR Consensus Tri-Mode with Ternary Thermal Paths — root re-export shim.
//!
//! Plan 400 (2026-07-05): the production code moved to
//! `crates/katgpt-forward/src/flashar_consensus.rs`. This file is now a thin
//! re-export shim that preserves the historical
//! `crate::speculative::flashar_consensus::*` import path.
//!
//! All 10 tests moved with the production file (no training dependencies).

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

pub use katgpt_forward::flashar_consensus::{
    ConsensusConfig, ConsensusResult, DualPathResult, FlashARConsensusVerifier, MAX_DRAFT_WIDTH,
    ThermalPath, compute_ternary_consensus, dual_path_draft, route_thermal_paths,
};

// `ternary_fusion_gate` is gated `plasma_path` upstream.
#[cfg(feature = "plasma_path")]
pub use katgpt_forward::flashar_consensus::ternary_fusion_gate;
