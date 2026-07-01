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
//! - **`ilc` + `trd`** → `katgpt-speculative` (Phase 6). ILC = Iterative Latent
//!   Clustering synonym-aware DDTree pruning; TRD = Trajectory-Refined Draft
//!   for speculative decoding. Both are speculative-draft screening primitives.
//!
//! Until Phase 6 lands, `ilc` and `trd` stay here; their feature flags
//! (`ilc_distill`, `trd_refined_draft`) are unchanged.

// → katgpt-spectral (Phase 4 — DONE): spectral alignment metric. Substrate
// moved; re-export preserves the historical `katgpt_rs::distill::peira` path.
#[cfg(feature = "peira_distill")]
pub use katgpt_spectral::peira;

// → katgpt-speculative (Phase 6): synonym-aware DDTree pruning.
#[cfg(feature = "ilc_distill")]
pub mod ilc;

// → katgpt-speculative (Phase 6): trajectory-refined draft screening.
#[cfg(feature = "trd_refined_draft")]
pub mod trd;
