//! Claim-Level Reliability pair — claim_rubric + clr.
//!
//! Extracted from `katgpt-rs/src/{claim_rubric, clr}/` per Proposal 003
//! Phase 11 (2026-07-04).
//!
//! # Module map
//!
//! - `claim_rubric` — L1/L2/L3 evidence ladder validator (Plan 307,
//!   arXiv:2606.07612). Generic meta-discipline that grades probe/steering
//!   claims by evidence level.
//! - `clr` — Claim-Level Reliability runtime (Plan 284, Research 255).
//!   Sigmoid projection vote over claim embeddings.
//!
//! # Why one crate
//!
//! Both modules belong to the "claim reliability" sub-domain. They are
//! siblings (zero internal coupling — verified by audit). The bridge between
//! them lives in downstream consumers (`riir-ai/npc_clr/claim_rubric_bridge.rs`),
//! not in either module.
//!
//! # External crate deps
//!
//! - `katgpt-core`: `traits::FeatureClass` (claim_rubric tag) and
//!   `simd::simd_sum_f32` (clr mgpo + vote).
//! - `blake3`: dev-only, for clr test fixtures.

#![allow(unexpected_cfgs)]  // root may pass-through aggregate features like `full`

#[cfg(feature = "claim_rubric")]
pub mod claim_rubric;

#[cfg(feature = "clr")]
pub mod clr;
