//! Interval-preserving token pruning (arXiv:2503.13663).
//!
//! Enforces convexity of valid token sets — eliminates "Swiss cheese" patterns
//! where scattered tokens are rejected while in-between tokens are valid.
//!
//! The paper defines ⊞ as the cube category of interval-preserving monotone maps
//! between finite Boolean lattices. The key insight: **interval preservation**
//! means "if tokens i..j are all valid, their valid image must also be contiguous" —
//! this is a structural constraint on how token validity propagates.
//!
//! # Architecture
//!
//! - [`IntervalMask`] — boolean validity mask with interval-closure operations.
//! - [`IntervalPruner`] — wraps any [`ConstraintPruner`] and enforces interval
//!   closure on its batch output.
//! - `simd` — SIMD stubs for future acceleration.

#[cfg(feature = "interval_pruner")]
mod interval;
#[cfg(feature = "interval_pruner")]
mod simd;

#[cfg(feature = "interval_pruner")]
pub use interval::{IntervalMask, IntervalPruner};
