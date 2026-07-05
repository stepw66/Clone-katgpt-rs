//! Distillation primitives — speculative-draft screening half.
//!
//! **Split history** (Proposal 003):
//! - `peira` → `katgpt-spectral` (Phase 4 — DONE 2026-07-01). Spectral alignment
//!   metric; re-exported at `katgpt_rs::distill::peira` via the root shim.
//! - `ilc` → here (Phase 6 — DONE 2026-07-04). ILC = Iterative Latent Clustering
//!   synonym-aware DDTree pruning (Research 136). Root re-exports as
//!   `katgpt_rs::distill::ilc`.
//! - `trd` → here (Phase 13 — DONE 2026-07-05). TRD = Trajectory-Refined Draft
//!   (Plan 249). Originally stayed in root because `prefold_prefix` depends on
//!   `crate::fold`; once Phase 12 T4.5 moved `fold/` here (2026-07-04), the
//!   cycle dissolved. Root re-exports as `katgpt_rs::distill::trd`.

pub mod ilc;

#[cfg(feature = "trd_refined_draft")]
pub mod trd;
