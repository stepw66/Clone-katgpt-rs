//! Two-tier syntax pruner for inference-time code validation.
//!
//! Extracted from `katgpt-rs/src/validator/` per Proposal 003 Phase 11
//! (2026-07-04).
//!
//! # Architecture
//!
//! - **Tier 0 — `PartialParser`**: O(n) bracket-balancer DFA. Rejects clearly
//!   broken code cheaply. Never false-accepts unbalanced brackets.
//! - **Tier 1 — `SynPruner`**: `syn::parse_str::<syn::Stmt>` accurate parse.
//!   Only called if Tier 0 passes. Implements `katgpt_core::ConstraintPruner`.
//!
//! # Feature gates
//!
//! - `validator` — back-compat feature name (root forwards to it). The crate's
//!   modules compile unconditionally; this feature exists for parity.
//! - `hoare_pruner` — gates the optional `ConstraintPruner::propagate` impl
//!   that does Hoare-style predicate checking during DDTree expansion.

#![allow(unexpected_cfgs)]  // root may pass-through aggregate features like `full`

mod partial_parser;
mod syn_pruner;
mod types;

pub use partial_parser::PartialParser;
pub use syn_pruner::SynPruner;
pub use types::{CompilerFeedback, ErrorKind, PruneResult};
