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
//! - **`trd`** → **stays in root** (Phase 6 scope reduction, 2026-07-04). TRD's
//!   `prefold_prefix` path depends on `crate::fold::*` (transformer-bound glue
//!   that lives in root per Phase 12 target). Moving TRD to katgpt-speculative
//!   would require katgpt-speculative → katgpt-rs (cycle). The `chain_fold`-
//!   gated prefold path is the only blocker; the rest of TRD is self-contained.
//!   Kept in root alongside `fold/` until fold's own destination (Phase 9 or 12)
//!   lands. See `proposals/003_src_consolidation_master.md` Phase 6 notes.

// → katgpt-spectral (Phase 4 — DONE): spectral alignment metric. Substrate
// moved; re-export preserves the historical `katgpt_rs::distill::peira` path.
#[cfg(feature = "peira_distill")]
pub use katgpt_spectral::peira;

// → katgpt-speculative (Phase 6 — DONE 2026-07-04): synonym-aware DDTree pruning.
// Substrate moved; re-export preserves the historical `katgpt_rs::distill::ilc` path.
#[cfg(feature = "ilc_distill")]
pub use katgpt_speculative::distill::ilc;

// → stays in root (Phase 6 scope reduction): trajectory-refined draft screening.
// Blocked by `crate::fold` dep in the `chain_fold`-gated `prefold_prefix` path.
#[cfg(feature = "trd_refined_draft")]
pub mod trd;
