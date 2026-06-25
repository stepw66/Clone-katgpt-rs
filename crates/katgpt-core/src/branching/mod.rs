//! Non-Interference Memory Branches — continual-adaptation primitive
//! distilled from RIZZ (Goel et al., Oxford, Jun 2026, arXiv:2606.20638).
//!
//! Plan 329 / Research 310 — Super-GOAT fusion of BAKE × CLR × MCGS × Engram
//! × ARG × closure-instrument × Salience. Five generic open primitives (no
//! game / chain / shard semantics):
//!
//! - [`types`] — `BranchId`, `CognitiveBranch`, `EpisodicEntry`, `ProceduralRule`,
//!   `FailureEntry`, `BranchStats`, `BranchLifecycle`.
//! - [`bank`] — `BranchBank`: bounded bank of persistent cognitive branches with
//!   spawn / merge / prune lifecycle. Free-list slot reuse; pre-allocated
//!   capacity.
//! - [`router`] — `BranchRouter`: dot-product snap routing with optional
//!   Jaccard token-overlap fallback. Zero-alloc hot path.
//! - [`verifier`] — `VerifierGate`: reward + curiosity + branch-centroid
//!   quarantine write gate. Composes with CLR `should_write_memory(r_k, S_LP)`.
//! - (Phase 2) `projection` — `NonInterferenceProjection`: orthogonal latent
//!   subspaces per branch.
//! - (Phase 2) `compiler` — `BudgetCompiler`: priority-cascade context
//!   compiler under fixed budget.
//!
//! # Latent vs raw boundary (AGENTS.md)
//!
//! | Quantity | Space | Synced? |
//! |----------|-------|---------|
//! | `BranchBank` slot array + free-list | **Raw** | YES (deterministic structure) |
//! | `BranchId`, `BranchLifecycle`, `BranchStats` | **Raw** | YES |
//! | `spawn_anchor`, `EpisodicEntry.embedding` | **Latent** | NO (projection vectors) |
//! | `token_signature` | **Raw** | YES (deterministic hashes) |
//! | `EpisodicEntry.payload` | **Caller-defined** | Caller decides |
//!
//! Nothing in this module crosses the `SyncBlock → ChainConsensus` boundary
//! by default. The caller decides what to sync.
//!
//! # Feature gate
//!
//! Entire module is `#[cfg(feature = "non_interference_branches")]`. Zero cost
//! when disabled — every public API vanishes from the build. Opt-in until the
//! G1–G5 GOAT gate passes (Phase 3).
//!
//! # Composition
//!
//! When the `arg_protocol` feature is also enabled, `BranchLifecycle` is a
//! type alias for `crate::arg::LifecycleState` (the same enum used by the ARG
//! protocol's ontology lifecycle). This makes branch lifecycle state
//! committable and redirect-resolvable via the ARG `RedirectTable`.
//!
//! # References
//!
//! - Plan: [`katgpt-rs/.plans/329_non_interference_memory_branches.md`]
//! - Research: [`katgpt-rs/.research/310_RIZZ_Non_Interference_Memory_Branches.md`]
//! - Source paper: [arXiv:2606.20638](https://arxiv.org/abs/2606.20638)
//! - Private guide: [`riir-ai/.research/161_Per_NPC_Cognitive_Branch_Continual_Adaptation_Guide.md`]
//! - Fusion cousins: Plan 236 (BAKE), Plan 284/316 (CLR), progressive_mcgs/,
//!   Plan 299 (Engram), Plan 327 (ARG), Plan 290 (closure-instrument),
//!   Plan 303 (Salience)

pub mod bank;
pub mod router;
pub mod types;
pub mod verifier;

// ── Public API re-exports ─────────────────────────────────────────────────
//
// Mirrors the idiom used by other katgpt-core feature modules (`arg`,
// `bisimulation`, `closure`): `pub use` the most common types at the module
// root so callers can write
// `katgpt_core::branching::BranchBank` instead of
// `katgpt_core::branching::bank::BranchBank`.

pub use bank::{BranchBank, DEFAULT_MAX_BRANCHES};
pub use router::{
    BranchRouter, RouteMode, RouteResult, DEFAULT_TAU_JACCARD, DEFAULT_TAU_SNAP, DEFAULT_TAU_SPAWN,
};
pub use types::{
    BranchId, BranchLifecycle, BranchStats, CognitiveBranch, EpisodicEntry, FailureEntry,
    ProceduralRule,
};
pub use verifier::{
    VerifierGate, WriteDecision, DEFAULT_QUARANTINE_CENTROID_THRESH, DEFAULT_TAU_CURIOSITY,
    DEFAULT_TAU_WRITE,
};
