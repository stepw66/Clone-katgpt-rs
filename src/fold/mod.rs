//! ThoughtFold — Inference-Time Chain Folding (Plan 195).
//!
//! Implements inference-time chain folding inspired by ThoughtFold (arXiv:2606.03503).
//! When the ThinkingController selects a thinking mode, the ChainFolder introspectively
//! prunes redundant reasoning steps using attention-based importance scoring + speculative
//! verification. No LLM training — pure inference-time optimization.
//!
//! # Architecture
//!
//! ```text
//! ThinkingController (Plan 194)
//!     │
//!     ├── Direct mode → no folding (zero cost)
//!     │
//!     └── Latent/CpuResample mode
//!             │
//!             ├── StepBoundaryTracker
//!             │   └── Detects reasoning step boundaries (\n\n, think-tags)
//!             │
//!             ├── ChainFolder (ScreeningPruner impl)
//!             │   ├── attention_importance() → rank steps by ForwardContext.scores
//!             │   ├── binary_search_fold() → find minimal correct prefix
//!             │   └── verify_fold() → SpeculativeVerifier checks continuation
//!             │
//!             ├── FoldBandit
//!             │   └── Thompson sampling for fold budget self-tuning
//!             │
//!             └── FoldCache
//!                 ├── truncate_to_step() → KV cache rollback
//!                 └── replay_essential() → replay only essential steps
//! ```
//!
//! # Feature Gate
//!
//! `chain_fold` (default-OFF, depends on `thinking_cot`).
//!
//! # GOAT Criteria
//!
//! ≥30% CoT token reduction on hard queries with ≤2% accuracy regression.

// ── Submodules ──────────────────────────────────────────────────

pub mod attention_importance;
pub mod chain_folder;
pub mod fold_bandit;
pub mod fold_cache;
pub mod step_boundary;
pub mod thinking_ext;
pub mod types;

// ── Re-exports ──────────────────────────────────────────────────

pub use attention_importance::AttentionImportance;
pub use chain_folder::ChainFolder;
pub use fold_bandit::FoldBandit;
pub use fold_cache::FoldCache;
pub use step_boundary::{count_steps, detect_step_boundaries};
pub use thinking_ext::{
    fold_stats_feedback, fold_thinking_feedback, step_reduction_ratio, token_reduction_ratio,
};
pub use types::{
    FoldContext, FoldDecision, FoldResult, FoldStats, StepBoundary, ThinkingFoldFeedback,
};
