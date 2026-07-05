//! Distillation primitives — **split-boundary tagged** (Proposal 003 Phase 0.2).
//!
//! The `distill/` umbrella conflates two unrelated paper lineages and does NOT
//! survive as a unit. It is tagged here for the in-tree split; the actual file
//! moves happen in later phases:
//!
//! - **`peira`** → `katgpt-spectral` (Phase 4 — DONE 2026-07-01). PEIRA =
//!   spectral alignment metric (cross-view covariance eigenvector alignment).
//!   It's a spectral diagnostic, not a speculative-drafting primitive. Re-exported
//!   here so `katgpt_rs::distill::peira::*` paths keep resolving.
//! - **`ilc`** → `katgpt-speculative` (Phase 6 — DONE 2026-07-04). ILC = Iterative
//!   Latent Clustering synonym-aware DDTree pruning (Research 136). Re-exported
//!   here so `katgpt_rs::distill::ilc::*` paths keep resolving.
//! - **`trd`** → `katgpt-speculative` (Plan 384 — DONE 2026-07-05). TRD =
//!   Trajectory-Refined Draft screening (Plan 249). Originally blocked by
//!   `crate::fold` dep; Phase 12 T4.5 moved `fold/` to katgpt-speculative,
//!   dissolving the cycle. Re-exported here so `katgpt_rs::distill::trd::*`
//!   paths keep resolving.

// → katgpt-spectral (Phase 4 — DONE): spectral alignment metric. Substrate
// moved; re-export preserves the historical `katgpt_rs::distill::peira` path.
#[cfg(feature = "peira_distill")]
pub use katgpt_spectral::peira;

// → katgpt-speculative (Phase 6 — DONE 2026-07-04): synonym-aware DDTree pruning.
// Substrate moved; re-export preserves the historical `katgpt_rs::distill::ilc` path.
#[cfg(feature = "ilc_distill")]
pub use katgpt_speculative::distill::ilc;

// → katgpt-speculative (Plan 384 — DONE 2026-07-05): trajectory-refined draft
// screening. Substrate moved; re-export preserves the historical
// `katgpt_rs::distill::trd` path.
#[cfg(feature = "trd_refined_draft")]
pub use katgpt_speculative::distill::trd;
