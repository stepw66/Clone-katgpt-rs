//! Stiff/Soft Subspace Anomaly Gate (Plan 138).
//!
//! Generic stiff/soft subspace decomposition and anomaly detection primitives
//! that extend SpectralQuant's eigenbasis code. Feature-gated behind
//! `stiff_anomaly` — pure linear algebra, no game-specific knowledge.
//!
//! # Architecture
//!
//! - **subspace** — Stiff/soft decomposition via trace-mass thresholding,
//!   soft alignment ratio α.
//! - **stability** — Eigenvalue tracking across temporal windows, Jaccard
//!   stability, z-score gating, anomaly gate with FPR validation.
//! - **baseline** — Monte Carlo null test for structural agreement vs random
//!   noise.

pub mod baseline;
pub mod stability;
pub mod subspace;

pub use baseline::{MonteCarloNull, monte_carlo_null_test};
pub use stability::{EigenvalueTracker, GateResult, StiffAnomalyGate};
pub use subspace::{StiffSoftDecomposition, decompose, soft_alignment_ratio, stiff_subspace_k};
