//! SwiR Switch-Thinking — Explicit↔Latent mode controller (modelless).
//!
//! Distilled from SwiReasoning (ICLR 2026, arXiv:2510.05069, Shi et al.).
//! Plan: `katgpt-rs/.plans/275_swir_switch_thinking.md`.
//! Research: `katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md`.
//!
//! # What this module provides
//!
//! Three training-free primitives that switch a transformer decoder between
//! **Explicit** (token-space) and **Latent** (continuous-embedding-space)
//! reasoning modes at inference time:
//!
//! - [`SwiRController`] — the 2-mode state machine. Driven by the *sign* of
//!   `entropy − reference_entropy` (a relative, drift-robust signal). Asymmetric
//!   dwell windows prevent chatter; a switch-count controller suppresses
//!   overthinking via `</think>` convergence and `ForceAnswerPrefix`
//!   termination.
//! - [`soft_embedding`] — `ẽ_t = Σ_v p_t[v] · e(v)`, the probability-weighted
//!   vocab mixture emitted in Latent mode. SIMD-friendly chunked inner loop.
//! - [`mix_thinking_signal`] — blends the soft embedding with a control-token
//!   embedding at switch instants (paper Eq. 4), keeping the residual stream
//!   continuous across the mode boundary.
//!
//! All three are **zero-allocation** after construction. The controller uses a
//! fixed-size injection ring; the soft-embedding kernel accumulates into a
//! caller-provided scratch buffer; the mix kernel is in-place.
//!
//! # What this module does NOT provide
//!
//! - No model loading, no tokenization, no KV cache. The host wires those.
//! - No retraining, no backprop, no weight mutation. This is a pure inference-
//!   time controller (modelless constraint, per AGENTS.md).
//! - No softmax — the host supplies the probability vector. We compute Shannon
//!   entropy from it via [`shannon_entropy`] if the host doesn't already have a
//!   logits→entropy path.
//!
//! # Minimal end-to-end trace
//!
//! ```no_run
//! use katgpt_rs::swir::{SwiRConfig, SwiRController, StepAction, soft_embedding};
//!
//! // 1. Host constructs the controller with paper defaults.
//! let mut ctrl = SwiRController::new(SwiRConfig::default());
//!
//! // 2. Host pre-allocates the soft-embedding scratch ONCE (reused each step).
//! let embedding_dim = 1024;
//! let vocab = 32_000;
//! let embedding_matrix: Vec<f32> = vec![0.0; vocab * embedding_dim]; // (load real weights)
//! let mut soft_buf: Vec<f32> = vec![0.0; embedding_dim];
//!
//! // 3. Decode loop. Each step: compute probs → entropy → step → branch.
//! # fn next_token_probs() -> Vec<f32> { vec![0.0; 32_000] }
//! for step in 0..1024 {
//!     let probs: Vec<f32> = next_token_probs(); // host softmax
//!     let entropy = katgpt_rs::swir::shannon_entropy(&probs);
//!     match ctrl.step(entropy, step) {
//!         StepAction::EmitToken(_id) => {
//!             // Sample a concrete token, feed it to the model.
//!         }
//!         StepAction::EmitSoftEmbedding => {
//!             // Compute soft embedding into scratch.
//!             for x in soft_buf.iter_mut() { *x = 0.0; }
//!             soft_embedding(&probs, &embedding_matrix, embedding_dim, &mut soft_buf);
//!             // Optionally apply signal mix at the switch instant.
//!             if let Some((_kind, ratio)) = ctrl.should_mix_signal() {
//!                 // mix_thinking_signal(&mut soft_buf, &control_token_embed, ratio);
//!                 let _ = ratio;
//!             }
//!             // Feed soft_buf as the "token" embedding for this step.
//!         }
//!         StepAction::InjectControlToken(token) => {
//!             // Translate token (CloseThink / ForceAnswerPrefix) to a concrete
//!             // id and feed it.
//!         }
//!         StepAction::Terminate => break,
//!     }
//! }
//! ```
//!
//! Paper reports +1.8–3.1pp accuracy and 1.36–6.8× token efficiency on MATH500
//! (Qwen3-8B). The GOAT gate (Plan 275 Phase 3) must reproduce this on the
//! host's actual model before promoting to default.

mod controller;
mod convex_hull_check;
mod entropy;
mod signal_mix;
mod soft_embedding;
mod types;

pub use controller::SwiRController;
pub use convex_hull_check::in_vocab_convex_hull;
pub use entropy::shannon_entropy;
pub use signal_mix::{mix_thinking_signal, SignalMixKind};
pub use soft_embedding::soft_embedding;
pub use types::{ControlToken, StepAction, SwiRConfig, SwiRStats, ThinkMode};

// Re-exported for the Phase 2 strategy adapter's `StepContext::control_token_ids`
// field. Public so downstream code can construct it without depending on
// `thinking_cot` internals.

/// Concrete vocabulary ids for the control tokens the controller injects.
///
/// The host populates this from the tokenizer and feeds it to the Phase 2
/// strategy adapter. Kept here (not in `types.rs`) because it's a *wiring* type
/// — the core controller is tokenizer-agnostic.
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlTokenIds {
    /// `<think>` — paper's reasoning-block open. (Currently informational; the
    /// controller does not inject this.)
    pub think_open: u32,
    /// `</think>` — reasoning-block close. Maps to [`ControlToken::CloseThink`].
    pub think_close: u32,
    /// `</think>\n\nThe final answer is` — maps to
    /// [`ControlToken::ForceAnswerPrefix`]. For tokenizers that don't encode
    /// this as a single token, the host should use the id of `</think>` and
    /// handle the `\n\nThe final answer is` portion in the post-processing.
    pub force_answer_prefix: u32,
}

impl ControlTokenIds {
    /// Resolve a controller-emitted [`ControlToken`] to a concrete vocab id.
    #[inline]
    pub fn resolve(&self, token: ControlToken) -> u32 {
        match token {
            ControlToken::CloseThink => self.think_close,
            ControlToken::ForceAnswerPrefix => self.force_answer_prefix,
        }
    }
}
