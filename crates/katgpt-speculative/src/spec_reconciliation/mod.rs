//! Speculative Reconciliation Engine — verify offline trajectories against plausibility manifolds.
//!
//! Generates K speculative trajectories from a known last-good state, then reconciles
//! incoming offline data against this plausibility manifold. Fully modelless — no neural
//! forward pass required.
//!
//! # Pipeline
//!
//! 1. **Hard bounds** (`ReconciliationPruner`): velocity, position, kill-rate checks
//! 2. **Manifold generation** (`ManifoldGenerator`): K forward-simulated trajectories
//! 3. **Soft scoring** (`ManifoldScorer`): cosine similarity against manifold
//! 4. **Verdict**: Accept / Quarantine / Uncertain
//!
//! # Feature Gate
//!
//! Behind `spec_reconciliation` feature gate in `Cargo.toml`.

pub mod adaptive;
pub mod manifold;
pub mod manifold_scorer;
pub mod reconciler;
pub mod reconciliation_pruner;
pub mod types;

// Re-exports for convenience
pub use adaptive::{AdaptiveReconciler, AdaptiveReconcilerFrozen};
pub use manifold::{DefaultManifoldGenerator, ManifoldGenerator, gaussian_sample};
pub use manifold_scorer::ManifoldScorer;
pub use reconciler::{ReconciliationResult, SpecReconciler};
pub use reconciliation_pruner::ReconciliationPruner;
pub use types::{ReconciliationConfig, ReconciliationVerdict, TrajectoryPoint};
