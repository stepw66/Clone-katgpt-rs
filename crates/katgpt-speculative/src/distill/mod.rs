//! Distillation primitives — speculative-draft screening half.
//!
//! **Split history** (Proposal 003):
//! - `peira` → `katgpt-spectral` (Phase 4 — DONE 2026-07-01). Spectral alignment
//!   metric; re-exported at `katgpt_rs::distill::peira` via the root shim.
//! - `ilc` → here (Phase 6 — DONE 2026-07-04). ILC = Iterative Latent Clustering
//!   synonym-aware DDTree pruning (Research 136). Root re-exports as
//!   `katgpt_rs::distill::ilc`.
//! - `trd` → **stays in root** `src/distill/trd.rs`. TRD's `prefold_prefix` path
//!   depends on `crate::fold::*` (transformer-bound glue, Phase 12 scope). Moving
//!   it would require katgpt-speculative → katgpt-rs (cycle). Folded into
//!   Phase 6 scope reduction: see `proposals/003_src_consolidation_master.md`.

pub mod ilc;
