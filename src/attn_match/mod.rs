//! Attention Matching (AM) KV cache compaction — modelless (Plan 271, Research 233).
//!
//! Implements "Fast KV Compaction via Attention Matching"
//! (Zweiger, Fu, Guo, Kim — MIT, ICML 2026, arxiv 2602.16284).
//!
//! # The Idea
//!
//! When compacting a KV cache `(K, V)` to `(Ck, β, Cv)` with `t < T` tokens,
//! preserve two quantities on a set of reference queries `Qref`:
//!
//! 1. **Attention output** (Eq. 1):
//!    `softmax(qK^T/√d) V ≈ softmax((qCk^T + β)/√d) Cv`
//!
//! 2. **Attention mass** (Eq. 2):
//!    `Σ_j exp(qK_j^T/√d) ≈ Σ_j exp((qCk_j^T + β_j)/√d)`
//!
//! Together these guarantee that the compacted block's contribution under
//! concatenation with arbitrary future `(Kfixed, Vfixed)` is preserved,
//! because attention over concatenated blocks decomposes into a mixture
//! whose weights are determined by unnormalized attention mass.
//!
//! # Why β is Critical
//!
//! With `t < T` and no bias, `Mass(q; Ck) ≤ Mass(q; K)` always — compaction
//! systematically underestimates the compacted block's contribution.
//! β introduces per-token additive attention biases so each retained key can
//! account for the mass of many removed keys.
//!
//! Unlike `still_kv` (Plan 245, Research 213) which uses a **heuristic**
//! β (`log(T/t)` or vortex-flow weighting), AM computes **optimal** β via
//! nonnegative least squares (NNLS) — directly minimizing attention mass error.
//!
//! # The Pipeline (Closed-Form, No Gradient Descent)
//!
//! 1. Select compact keys `Ck` (subset of original `K`):
//!    - `HighestAttnKeys` — top-t by RMS attention score across `Qref`
//!    - `OMP` — greedy orthogonal matching pursuit on the mass feature matrix
//! 2. Fit `β` via NNLS (projected gradient descent)
//! 3. Fit `Cv` via ordinary least squares (normal equations + Cholesky)
//!
//! # Reference
//!
//! See `.research/233_Attention_Matching_KV_Compaction.md` for the full
//! distillation verdict, fusion ideas, and GOAT gate matrix.

pub mod beta_fitter;
pub mod chunked;
pub mod compact;
pub mod head_budget;
pub mod key_selection;
pub mod online;
pub mod router;
pub mod score_matrix;
pub mod score_matrix_simd;
pub mod types;
pub mod value_fitter;

#[cfg(feature = "adaptive_cot_compaction")]
pub mod adaptive_cot;

pub use beta_fitter::{fit_beta_nnls, BetaFitConfig, BetaFitResult};
pub use chunked::{ChunkedCompactor, ChunkedCompactOutput, ChunkMeta, TextChunk};
pub use compact::{compact, CompactError, CompactOutput};
pub use head_budget::{HeadBudgetSchedule, HeadBudgetSolver, HeadSensitivityCurve};
pub use key_selection::{
    highest_attn::select_highest_attn_keys, omp::select_omp_keys, KeySelection, KeySelectorKind,
};
pub use online::{OnlineCompactResult, OnlineCompactor};
pub use router::{pick_backend, SolverBackend, SolverRouter, SolverRouterConfig};
pub use score_matrix::{compute_score_matrix, compute_softmax_attention};
pub use score_matrix_simd::compute_score_matrix_simd;
pub use types::{
    AmConfig, AmResult, KeySelector, ReconstructionReport, ScoreMethod, SolverChoice,
};
pub use value_fitter::{fit_cv_least_squares, ValueFitConfig, ValueFitResult};

#[cfg(feature = "adaptive_cot_compaction")]
pub use adaptive_cot::{AdaptiveCompactResult, AdaptiveTraceCompactor};

#[cfg(test)]
mod tests;

/// Numerical stability epsilon for log/exp operations.
pub const STABILITY_EPS: f32 = 1e-12;

/// Default diagonal jitter for Cholesky when rank-deficient.
pub const DEFAULT_CHOLESKY_JITTER: f32 = 1e-6;

/// Default bound on `w = exp(β)` for HighestAttnKeys (paper Appendix C.2):
/// β ∈ [-3, 3] ⇒ w ∈ [e^-3, e^3].
pub const DEFAULT_W_LOWER: f32 = 1e-3; // ~ e^-6.9, safe lower bound
pub const DEFAULT_W_UPPER: f32 = 1096.6331; // e^7 ≈ 1096.63 (paper cap)

/// β lower bound for OMP key pruning (paper Appendix C.2).
pub const OMP_BETA_PRUNE_THRESHOLD: f32 = -7.0;
