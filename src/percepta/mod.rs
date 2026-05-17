//! Percepta-style O(log N) 2D Attention via Convex Hull KV Cache.
//!
//! Standard transformer attention computes Q·K for all N past keys → O(N) per step.
//! Percepta restricts attention heads to d=2, making the dot product a 2D geometric
//! projection. When keys form a convex hull, finding the maximum attention score
//! becomes ternary search over a unimodal (bitonic) sequence → O(log N).
//!
//! Integration points with microgpt-rs:
//! - DDTree branch pruning: validate drafted tokens before target verification
//! - Deterministic Validator: encode state-machine rules as 2D key embeddings
//! - "Free embedding" bridge: project hidden states to 2D for fast retrieval
//!
//! # Module layout
//!
//! - [`legacy`] — Original KVCache2D (Graham Scan + ternary search), Sudoku, StreamingSolver
//! - [`types`]  — Shared types for CHT hull vertices (`HullMeta`, `TieBreak`, `Vec2` f64)
//! - [`cht`]    — Dynamic Convex Hull Trick (line container, upper envelope)
//! - [`hull`]     — HullHalf, HardAttentionHead, BruteAttentionHead (O(log N) 2D hard attention)
//! - [`encoding`] — Parabolic key encoding helpers for 2D attention
//! - [`cumsum`]  — Cumulative sum via uniform attention (fetch_sum equivalent)

// ── Submodules ─────────────────────────────────────────────────

/// Legacy KVCache2D (Graham Scan + ternary search), Sudoku9x9, StreamingSolver.
/// Always compiled — no optional dependencies.
pub mod legacy;

/// Shared types for the CHT Hull KV Cache: `HullMeta`, `TieBreak`, `Vec2` (f64).
#[cfg(feature = "percepta")]
pub mod types;

/// Dynamic Convex Hull Trick / LineContainer for O(log h) max-envelope queries.
#[cfg(feature = "percepta")]
pub mod cht;

/// CHT Hull KV Cache — `HullHalf`, `HardAttentionHead`, `BruteAttentionHead`.
#[cfg(feature = "percepta")]
pub mod hull;

/// Parabolic key encoding helpers for 2D attention.
#[cfg(feature = "percepta")]
pub mod encoding;

/// Cumulative sum via uniform attention (fetch_sum equivalent).
#[cfg(feature = "percepta")]
pub mod cumsum;

/// Standard O(n) softmax attention KV cache for correctness verification.
#[cfg(feature = "percepta")]
pub mod standard_cache;

// ── Re-exports from legacy (always available) ──────────────────

pub use legacy::{KVCache2D, SolveEvent, StreamingSolver, Sudoku9x9, SymbolicValidator, Vec2};

// ── Re-exports from hull (feature-gated) ──────────────────────

#[cfg(feature = "percepta")]
pub use hull::{AttentionResult, BruteAttentionHead, HardAttentionHead, HullHalf};

#[cfg(feature = "percepta")]
pub use types::TieBreak;

#[cfg(feature = "percepta")]
pub use encoding::{clear_key, encode_key, encode_query, hard_scale, hard_scale_query};

#[cfg(feature = "percepta")]
pub use cumsum::CumSum;

#[cfg(feature = "percepta")]
pub use standard_cache::StandardCache;
