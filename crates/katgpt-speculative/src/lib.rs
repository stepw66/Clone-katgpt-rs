//! katgpt-speculative — shared speculative decoding + DDTree substrate.
//!
//! Issue 013 (2026-06-29): collapses the fork between `katgpt-rs/src/speculative/`
//! and `riir-engine/src/{dd_tree,dflash}.rs`. The core DDTree algorithm lives
//! here so improvements propagate to both consumers.
//!
//! # What lives here
//!
//! - **DDTree core** (`dd_tree`): `build_dd_tree`, `build_dd_tree_pruned`,
//!   `build_dd_tree_screened`, `build_dd_tree_balanced`, `TreeBuilder`,
//!   `extract_parent_tokens`, `extract_best_path`, `merge_retrieved_branches`,
//!   `find_valid_sequence`, etc. Pure algorithm over pre-computed marginals.
//!   No `forward` dependency.
//!
//! # What does NOT live here
//!
//! - **Types** (`TreeNode`, `DraftResult`, `ConstraintPruner`, …) → already in
//!   `katgpt_core::speculative::types` + `katgpt_core::traits` (Plan 008 Phase 2.5).
//! - **Sampling** (`sample_from_distribution`) → already in
//!   `katgpt_core::speculative::sampling` (Plan 008 Phase 2.6).
//! - **DFlash** (`dflash_predict_*_with`) → the three zero-alloc `_with`
//!   cores live here (Issue 013 Phase B). They are generic over a `DflashCtx` +
//!   `DflashCache` backend trait pair and a `forward_fn` closure, because the
//!   underlying `ForwardContext` / `MultiLayerKVCache` / `TransformerWeights`
//!   types are crate-specific. The thin wrappers (`dflash_predict`, `_ar`,
//!   `_conditioned`, `_parallel`) and feature-gated variants (`_domino`,
//!   `_routing`, `_fusion`) stay in each consumer.
//! - **Feature-gated DDTree variants** (`build_dd_tree_belief`, `_speculative`,
//!   `_kurtosis`, `_domino`, `_manifold`, `_lodestar`, `_gdsd`, …) → stay in
//!   `katgpt-rs/src/speculative/dd_tree.rs` because they reference root-only
//!   sibling modules (`super::belief_drafter`, `super::spec_generator`, etc.).

pub mod blueprint;
pub mod branch_confidence;
pub mod correlation_budget;
pub mod dd_tree;
pub mod decomp_reviewer;
pub mod dflash;
pub mod nf_flow;
pub mod nf_flow_budget;
pub mod nf_flow_fold;
pub mod nf_flow_gate;
pub mod nf_flow_mux;
pub mod pathway_tracker;
pub mod prefix_scheduler;
pub mod vocab_coreset;

// PPoT core primitives (Issue 003): types/knowledge/entropy/rank moved here;
// the `resample` orchestrator stays in katgpt-rs root and consumes these via
// re-export. Gated because the moved files carry internal `#[cfg(feature = "ppot")]`
// gates (adaptive-knowledge path) that require the feature to be live.
#[cfg(feature = "ppot")]
pub mod ppot;

// ── Proposal 003 Phase 6 absorptions (2026-07-04) ──────────────────────────
// Modules moved from katgpt-rs/src/. Each is feature-gated to mirror its
// historical root gate. Root re-exports preserve `katgpt_rs::<module>::*` paths.

// distill umbrella — only ilc moved here (trd stays root: fold dep). peira
// lives in katgpt-spectral; root re-exports it at katgpt_rs::distill::peira.
#[cfg(feature = "ilc_distill")]
pub mod distill;

// RTPurbo retrieval-head sparse decode (Plan 126).
#[cfg(feature = "rt_turbo")]
pub mod rt_turbo;

// PASD boundary-aware draft scoring (Plan 227 Phase 4).
#[cfg(feature = "precision_aware_draft")]
pub mod precision_aware_draft;

// SpecHop continuous multi-hop speculation pipeline (Plan 131).
#[cfg(feature = "spechop")]
pub mod spechop;

// Speculative Reconciliation Engine (Plan 177). Originally ungated in root
// (the `spec_reconciliation = []` feature was vestigial — the module compiled
// unconditionally). Preserved here as ungated to match historical behavior.
pub mod spec_reconciliation;

// Re-export katgpt_core's speculative types + traits so consumers can import
// everything from one place (`katgpt_speculative::{TreeNode, ConstraintPruner, …}`).
pub use katgpt_core::speculative::types::*;
pub use katgpt_core::traits::{
    BinaryScreeningPruner, CompletionHorizon, ConstraintPruner, DominoPruner, NoPruner,
    NoScreeningPruner, ScreeningPruner,
};

// ── Phase 12 T4.5 (2026-07-04): modules moved from katgpt-rs/src/. ──
// Progressive MCGS — Monte Carlo Graph Search (Plan 148). Self-contained.
#[cfg(feature = "progressive_mcgs")]
pub mod progressive_mcgs;
// Chain Fold — step-boundary-aware context folding (Plan 195, GOAT 16/16).
// Needs katgpt-kv/still_kv for chain_folder's KV compaction types.
#[cfg(feature = "chain_fold")]
pub mod fold;
