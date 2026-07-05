// ═══════════════════════════════════════════════════════════════
// Denoise-Loop Cluster — D2F inference denoising with optional RCD/3SR
// ═══════════════════════════════════════════════════════════════
//
// Plan 403 (2026-07-06): Moved from root `src/dllm.rs`. This module
// consolidates the four denoise-loop variants + the `DenoiseConstraint`
// trait family. They are pure inference (no gradients/backprop/loss) and
// compose the `BidirectionalContext` substrate + the
// `forward_bidirectional_positions_into` kernel (both moved to
// `crate::forward_positions` in Plan 402) with the `katgpt_core::dllm_solver`
// helpers (`compute_residual`, `interpolate_residual`, `normalized_entropy`,
// `classify_transitions`, `compute_gammas`, `warm_start_lerp`,
// `ThreeStateReuseConfig`, `RcdConfig`, `TransitionType`).
//
// Root keeps a re-export shim at `crate::dllm::*` so every historical caller
// continues to resolve, including the 9 denoise tests that stay in root
// (they use root-only training helpers `train_mini_dllm` /
// `generate_pattern_dataset` to set up weights, so they exercise the public
// API via the re-export).
//
// ## Feature gating
//
// The whole module is `dllm`-gated (mirrors `forward_positions`) because
// every variant depends on `BidirectionalContext` +
// `forward_bidirectional_positions_into` from `crate::forward_positions`,
// which is itself `dllm`-gated. Within the module:
//
// - `DenoiseConstraint` / `NoConstraint` / `NoRepeatConstraint` /
//   `denoise_loop` / `denoise_loop_scheduled` — always-on (within `dllm`).
// - `denoise_loop_rcd` — additionally `rcd_residual`-gated (Plan 258).
// - `denoise_loop_rcd_3sr` — additionally `d2f_3sr_warm_start`-gated
//   (Plan 291). At the root level `d2f_3sr_warm_start` transitively enables
//   `rcd_residual`, so the `_3sr` variant can always reach
//   `katgpt_core::dllm_solver::*` when it's enabled.
//
// ## Why `rcd_residual` forwards to `katgpt-core/critical_interval_gate`
//
// `katgpt_core::dllm_solver` module is gated `critical_interval_gate` in
// katgpt-core. This crate's `rcd_residual` feature (see Cargo.toml) forwards
// both `katgpt-core/rcd_residual` AND `katgpt-core/critical_interval_gate`,
// so the `katgpt_core::dllm_solver::*` paths below resolve whenever
// `rcd_residual` (or its superset `d2f_3sr_warm_start`) is enabled.

use crate::forward_positions::{forward_bidirectional_positions_into, BidirectionalContext};
use katgpt_core::simd;
use katgpt_core::PositionOffsetSchedule;
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;
use katgpt_types::Rng;

// ═══════════════════════════════════════════════════════════════
// Task 0.5: Denoising Loop with Constraint
// ═══════════════════════════════════════════════════════════════

/// Simple constraint trait for denoising guidance.
pub trait DenoiseConstraint {
    /// Returns true if `token` is valid at `position` given `current_tokens`.
    fn is_valid(&self, position: usize, token: usize, current_tokens: &[usize]) -> bool;

    /// Rebuild any internal state from the current tokens. Default is no-op.
    fn rebuild(&mut self, _tokens: &[usize], _mask: usize) {}
}

/// No-op constraint that allows all tokens.
pub struct NoConstraint;

impl DenoiseConstraint for NoConstraint {
    fn is_valid(&self, _position: usize, _token: usize, _current_tokens: &[usize]) -> bool {
        true
    }
}

/// No-repeat constraint: tokens must be unique in the sequence.
/// Uses a precomputed `used` set for O(1) lookups instead of O(seq_len) scans.
pub struct NoRepeatConstraint {
    used: Vec<bool>,
}

impl Default for NoRepeatConstraint {
    fn default() -> Self {
        Self::new()
    }
}

impl NoRepeatConstraint {
    pub fn new() -> Self {
        Self { used: Vec::new() }
    }

    /// Rebuild the used-token set from current tokens.
    pub fn rebuild(&mut self, tokens: &[usize], mask: usize) {
        let max_token = tokens
            .iter()
            .copied()
            .filter(|&t| t != mask)
            .max()
            .unwrap_or(0);
        self.used.clear();
        if self.used.len() <= max_token {
            self.used.resize(max_token + 1, false);
        }
        for &t in tokens {
            if t != mask && t < self.used.len() {
                self.used[t] = true;
            }
        }
    }
}

impl DenoiseConstraint for NoRepeatConstraint {
    fn is_valid(&self, _position: usize, token: usize, _current_tokens: &[usize]) -> bool {
        !self.used.get(token).copied().unwrap_or(false)
    }
}

/// Run denoising loop starting from all-mask tokens.
/// Returns (final_tokens, n_steps_to_converge).
pub fn denoise_loop(
    weights: &TransformerWeights,
    target_tokens: &[usize],
    config: &Config,
    n_steps: usize,
    confidence_threshold: f32,
    constraint: &mut dyn DenoiseConstraint,
    _rng: &mut Rng,
) -> (Vec<usize>, usize) {
    let seq_len = target_tokens.len().min(config.block_size);
    let mask = config.mask_token;

    // Pre-allocate context once — reused across all denoising steps
    let mut bctx = BidirectionalContext::new(config);

    // Initialize with mask tokens
    let mut tokens = vec![mask; seq_len];
    let mut converged_step = n_steps;
    let mut remaining = seq_len;

    let vocab = config.vocab_size;

    for step in 0..n_steps {
        forward_bidirectional_positions_into(weights, &tokens, config, &mut bctx);
        let mut any_changed = false;

        // Rebuild constraint used-token set once per step
        constraint.rebuild(&tokens, mask);

        for p in 0..seq_len {
            if tokens[p] != mask {
                continue;
            }

            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = simd::simd_max_f32(logits_p);
            // OPT: compute exp once, reuse for both sum and argmax
            let exp_buf = &mut bctx.all_attn_weights[..vocab]; // reuse attn weights as scratch
            exp_buf[..vocab].copy_from_slice(logits_p);
            simd::simd_add_scalar_inplace(&mut exp_buf[..vocab], -max_l);
            simd::simd_exp_inplace(&mut exp_buf[..vocab]);
            let sum_exp = simd::simd_sum_f32(&exp_buf[..vocab]);
            let inv_sum = 1.0 / sum_exp;

            // Find highest-confidence valid token.
            //
            // Hoist `inv_sum` out of the per-token comparison: since `inv_sum > 0`,
            // comparing `exp_buf[t] * inv_sum > best_prob` is equivalent to
            // `exp_buf[t] > best_prob * sum_exp`. We track `best_exp` (= best_prob * sum_exp)
            // across the scan and divide once at the end. This removes one fmul
            // from the vocab-sized inner loop.
            let mut best_token = mask;
            let mut best_exp = 0.0f32; // exp_buf values are >= 0
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !constraint.is_valid(p, t, &tokens) {
                    continue;
                }
                let e = exp_buf[t];
                if e > best_exp {
                    best_exp = e;
                    best_token = t;
                }
            }
            let best_prob = best_exp * inv_sum;

            if best_prob >= confidence_threshold && best_token != mask {
                tokens[p] = best_token;
                any_changed = true;
                remaining -= 1;
            }
        }

        if !any_changed && remaining == 0 {
            converged_step = step;
            break;
        }
    }

    // Check if all unmasked
    if remaining == 0 && converged_step == n_steps {
        converged_step = n_steps - 1;
    }

    (tokens, converged_step)
}

/// Denoising loop with position-offset reveal-time scheduling.
///
/// Identical to [`denoise_loop`] except: at each step, only positions whose
/// active generation interval has started (per the schedule) are eligible for
/// commitment. Positions not yet eligible are held as mask even if their
/// confidence exceeds the threshold.
///
/// This biases the unmasking order according to the schedule's `w` parameter:
/// - Small w (→ 1/L): near-left-to-right unmasking (AR-like)
/// - Large w (→ 1): all positions eligible immediately (order-agnostic)
///
/// **Modelless**: applies to any bidirectionally-trained D2F model. No
/// retraining needed — the schedule is a pure inference-time filter on the
/// confidence-based unmasking policy.
pub fn denoise_loop_scheduled(
    weights: &TransformerWeights,
    target_tokens: &[usize],
    config: &Config,
    n_steps: usize,
    confidence_threshold: f32,
    constraint: &mut dyn DenoiseConstraint,
    _rng: &mut Rng,
    schedule: &PositionOffsetSchedule,
) -> (Vec<usize>, usize) {
    let seq_len = target_tokens.len().min(config.block_size);
    let mask = config.mask_token;

    let mut bctx = BidirectionalContext::new(config);
    let mut tokens = vec![mask; seq_len];
    let mut converged_step = n_steps;
    let mut remaining = seq_len;
    let vocab = config.vocab_size;

    for step in 0..n_steps {
        // Ordering time τ = step / n_steps — normalized position in [0, 1]
        let tau = if n_steps > 1 {
            step as f32 / (n_steps - 1) as f32
        } else {
            1.0 // single step: all positions eligible
        };

        forward_bidirectional_positions_into(weights, &tokens, config, &mut bctx);
        let mut any_changed = false;
        constraint.rebuild(&tokens, mask);

        for p in 0..seq_len {
            if tokens[p] != mask {
                continue;
            }

            // Position-offset schedule: skip positions not yet eligible
            if !schedule.is_eligible(p, seq_len, tau) {
                continue;
            }

            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = simd::simd_max_f32(logits_p);
            let exp_buf = &mut bctx.all_attn_weights[..vocab];
            exp_buf[..vocab].copy_from_slice(logits_p);
            simd::simd_add_scalar_inplace(&mut exp_buf[..vocab], -max_l);
            simd::simd_exp_inplace(&mut exp_buf[..vocab]);
            let sum_exp = simd::simd_sum_f32(&exp_buf[..vocab]);
            let inv_sum = 1.0 / sum_exp;

            let mut best_token = mask;
            let mut best_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !constraint.is_valid(p, t, &tokens) {
                    continue;
                }
                let e = exp_buf[t];
                if e > best_exp {
                    best_exp = e;
                    best_token = t;
                }
            }
            let best_prob = best_exp * inv_sum;

            if best_prob >= confidence_threshold && best_token != mask {
                tokens[p] = best_token;
                any_changed = true;
                remaining -= 1;
            }
        }

        if !any_changed && remaining == 0 {
            converged_step = step;
            break;
        }
    }

    if remaining == 0 && converged_step == n_steps {
        converged_step = n_steps - 1;
    }

    (tokens, converged_step)
}

/// Run denoising loop with Residual Context Diffusion (Plan 258).
///
/// After each denoising step, computes entropy-weighted residuals for the
/// positions that are *still masked* (discarded low-confidence distributions)
/// and injects them into the next step's input embeddings via:
///   `ẽ_i = (1 - α_i) * E_mask + α_i * Δ_i`
/// where `α_i = H(p_i) / log(V)` and `Δ_i = Σ_j p_ij * E_j`.
///
/// This gives the model soft context about discarded tokens, accelerating
/// convergence (paper reports 4-5× step reduction at equivalent accuracy).
///
/// When `rcd_config` is `None` or `enabled = false`, behaves identically to
/// [`denoise_loop`] — the residual buffer is never activated.
#[cfg(feature = "rcd_residual")]
pub fn denoise_loop_rcd(
    weights: &TransformerWeights,
    target_tokens: &[usize],
    config: &Config,
    n_steps: usize,
    confidence_threshold: f32,
    constraint: &mut dyn DenoiseConstraint,
    rng: &mut Rng,
    rcd_config: Option<&mut katgpt_core::dllm_solver::RcdConfig>,
) -> (Vec<usize>, usize) {
    // If RCD is disabled at runtime, fall back to the standard loop with zero overhead.
    let rcd_enabled = rcd_config.as_ref().is_some_and(|c| c.enabled);
    if !rcd_enabled {
        return denoise_loop(
            weights,
            target_tokens,
            config,
            n_steps,
            confidence_threshold,
            constraint,
            rng,
        );
    }

    use katgpt_core::dllm_solver::{compute_residual, interpolate_residual, normalized_entropy};

    let seq_len = target_tokens.len().min(config.block_size);
    let mask = config.mask_token;
    let n = config.n_embd;
    let vocab = config.vocab_size;
    let log_vocab = rcd_config.as_ref().map_or(1.0, |c| c.log_vocab);

    let mut bctx = BidirectionalContext::new(config);
    let mut tokens = vec![mask; seq_len];
    let mut converged_step = n_steps;
    let mut remaining = seq_len;

    // Reusable scratch buffers — allocated once, never reallocated in the loop.
    let mut softmax_scratch = vec![0.0f32; vocab];
    let mut residual_scratch = vec![0.0f32; n];
    // E_mask: the embedding the masked positions would receive by default.
    // Borrow directly from weights — no allocation (interpolate_residual takes &[f32]).
    let mask_emb: &[f32] = &weights.wte[mask * n..(mask + 1) * n];

    for step in 0..n_steps {
        // The residual buffer is populated *after* step 0's forward pass, so the
        // first forward is always standard (no residuals exist yet).
        forward_bidirectional_positions_into(weights, &tokens, config, &mut bctx);

        // Standard commitment phase — mirror of `denoise_loop`.
        constraint.rebuild(&tokens, mask);
        let mut any_changed = false;

        for p in 0..seq_len {
            if tokens[p] != mask {
                continue;
            }

            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = simd::simd_max_f32(logits_p);
            let exp_buf = &mut bctx.all_attn_weights[..vocab];
            exp_buf[..vocab].copy_from_slice(logits_p);
            simd::simd_add_scalar_inplace(&mut exp_buf[..vocab], -max_l);
            simd::simd_exp_inplace(&mut exp_buf[..vocab]);
            let sum_exp = simd::simd_sum_f32(&exp_buf[..vocab]);
            let inv_sum = 1.0 / sum_exp;

            // Find highest-confidence valid token (see denoise_loop for inv_sum hoist rationale).
            let mut best_token = mask;
            let mut best_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !constraint.is_valid(p, t, &tokens) {
                    continue;
                }
                let e = exp_buf[t];
                if e > best_exp {
                    best_exp = e;
                    best_token = t;
                }
            }
            let best_prob = best_exp * inv_sum;

            if best_prob >= confidence_threshold && best_token != mask {
                tokens[p] = best_token;
                any_changed = true;
                remaining -= 1;
            }
        }

        // RCD: for every position that is *still* masked after commitment,
        // compute the entropy-weighted residual and stash it for the next step.
        // The override is activated only when at least one residual was written,
        // so the first step is guaranteed to use standard mask embeddings.
        let mut wrote_residual = false;
        for p in 0..seq_len {
            if tokens[p] != mask {
                continue;
            }

            // Softmax the logits for position p into scratch to get marginals p^k_i.
            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = simd::simd_max_f32(logits_p);
            softmax_scratch[..vocab].copy_from_slice(logits_p);
            simd::simd_add_scalar_inplace(&mut softmax_scratch[..vocab], -max_l);
            simd::simd_exp_inplace(&mut softmax_scratch[..vocab]);
            let sum_exp = simd::simd_sum_f32(&softmax_scratch[..vocab]);
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                simd::simd_scale_inplace(&mut softmax_scratch[..vocab], inv);
            }

            // α_i = H(p_i) / log(V), Δ_i = Σ_j p_ij * E_j, ẽ_i = (1-α)E_mask + α·Δ_i.
            let alpha = normalized_entropy(&softmax_scratch[..vocab], log_vocab);
            compute_residual(
                &softmax_scratch[..vocab],
                &weights.wte,
                n,
                &mut residual_scratch,
            );
            interpolate_residual(
                mask_emb,
                &residual_scratch,
                alpha,
                &mut bctx.rcd_residual_embeddings[p * n..(p + 1) * n],
            );
            wrote_residual = true;
        }

        // Activate the override for the *next* forward pass if we wrote anything.
        // If all masked positions got committed, we're done — no residuals needed.
        bctx.rcd_active = wrote_residual;

        if !any_changed && remaining == 0 {
            converged_step = step;
            break;
        }
    }

    // Disable the override before returning so the context is left in a clean state.
    bctx.rcd_active = false;

    if remaining == 0 && converged_step == n_steps {
        converged_step = n_steps - 1;
    }

    (tokens, converged_step)
}

/// Run denoising loop with Residual Context Diffusion (Plan 258) **composed**
/// with Three-State Reuse warm-start (Plan 291, Research 265).
///
/// This is the entry point for `d2f_3sr_warm_start`. It runs the same commitment
/// loop as [`denoise_loop_rcd`], but after each step's RCD residual computation
/// it additionally classifies per-position transitions between the previous and
/// current token state and applies the CoFRe §1.2 warm-start lerp:
///
///     h⁰_t[i] = γ[i] · h⋆_{t+1}[i] + (1 − γ[i]) · h_pre,t[i]
///
/// where `h_pre,t[i]` is the RCD residual embedding (RCD's output serves as the
/// preprocessing-stack result for 3SR) and `h⋆_{t+1}[i]` is the prior step's
/// embedding (modelless proxy for the FP solver hidden state — full FP-state
/// 3SR requires Plan 108 LT2's loop carry, out of scope for this plan).
///
/// γ is chosen per-position by transition type (UnchangedVisible / StillMasked /
/// NewlyRevealed) and modulated by the step's visible fraction for StillMasked
/// positions (paper Tables 4-5).
///
/// **Operational hook**: 3SR writes its lerped warm-start into
/// `bctx.tsr_warm_start_embeddings`, which the embedding-lookup path prefers
/// over `rcd_residual_embeddings` when `tsr_active` is set. RCD's residual
/// computation is unchanged — 3SR composes on top of it.
///
/// **Honest scope note**: this captures the *structure* of 3SR (token-type-aware
/// warm-start with three discrete γ coefficients) but operates on the input
/// embedding layer (same layer as RCD), not the FP solver hidden state. The
/// full paper version would stash the FP solver's hidden state; we don't have
/// an FP solver exposed here, so we use the prior step's residual as a proxy.
///
/// When `tsr_config` is `None` or `enabled = false`, behaves identically to
/// [`denoise_loop_rcd`] (zero overhead — falls through directly).
#[cfg(feature = "d2f_3sr_warm_start")]
pub fn denoise_loop_rcd_3sr(
    weights: &TransformerWeights,
    target_tokens: &[usize],
    config: &Config,
    n_steps: usize,
    confidence_threshold: f32,
    constraint: &mut dyn DenoiseConstraint,
    rng: &mut Rng,
    rcd_config: Option<&mut katgpt_core::dllm_solver::RcdConfig>,
    tsr_config: Option<&katgpt_core::dllm_solver::ThreeStateReuseConfig>,
) -> (Vec<usize>, usize) {
    // Zero-overhead runtime gate: if 3SR is disabled at runtime, delegate to RCD.
    let tsr_enabled = tsr_config.is_some_and(|c| c.enabled);
    if !tsr_enabled {
        return denoise_loop_rcd(
            weights,
            target_tokens,
            config,
            n_steps,
            confidence_threshold,
            constraint,
            rng,
            rcd_config,
        );
    }

    // 3SR implies rcd_residual (via Cargo feature deps), so we can rely on
    // rcd_config being meaningful. If RCD itself is disabled at runtime,
    // h_pre,t degenerates to the standard mask embedding — still well-defined.
    let rcd_enabled = rcd_config.as_ref().is_some_and(|c| c.enabled);
    if !rcd_enabled {
        // RCD disabled: 3SR still runs, but h_pre,t falls back to the mask
        // embedding for every still-masked position. This is the
        // modelless-3SR-on-bare-D2F mode (no RCD composition).
    }

    use katgpt_core::dllm_solver::{
        ThreeStateReuseConfig, classify_transitions, compute_gammas, compute_residual,
        interpolate_residual, normalized_entropy, warm_start_lerp,
    };
    let tsr_cfg: &ThreeStateReuseConfig = tsr_config.expect("checked tsr_enabled above");

    let seq_len = target_tokens.len().min(config.block_size);
    let mask = config.mask_token;
    let n = config.n_embd;
    let vocab = config.vocab_size;
    let log_vocab = rcd_config.as_ref().map_or(1.0, |c| c.log_vocab);

    let mut bctx = BidirectionalContext::new(config);
    let mut tokens = vec![mask; seq_len];
    let mut converged_step = n_steps;
    let mut remaining = seq_len;

    // Reusable scratch buffers — allocated once, never reallocated in the loop.
    let mut softmax_scratch = vec![0.0f32; vocab];
    let mut residual_scratch = vec![0.0f32; n];
    let mask_emb: &[f32] = &weights.wte[mask * n..(mask + 1) * n];

    // 3SR-specific scratch (Plan 291). All allocated once outside the loop.
    let mut transition_scratch =
        vec![katgpt_core::dllm_solver::TransitionType::UnchangedVisible; seq_len];
    let mut gamma_scratch = vec![0.0f32; seq_len];
    // h_star_next: previous step's residual buffer (modelless proxy for FP state).
    // Length `seq_len * n_embd`, zero-init (first step has no z_prev).
    let mut prev_residual = vec![0.0f32; seq_len * n];
    // z_prev_tokens: snapshot of tokens from the previous step. Initialized to
    // all-mask so step 0's transition classification is trivially StillMasked
    // everywhere (then explicitly skipped — see step-0 guard below).
    let mut z_prev_tokens = vec![mask; seq_len];

    for step in 0..n_steps {
        // Forward uses standard embedding on step 0 (tsr_active = false),
        // RCD residual on step k>0 if 3SR didn't write anything this step,
        // 3SR warm-start on step k>0 when 3SR composed over RCD's output.
        forward_bidirectional_positions_into(weights, &tokens, config, &mut bctx);

        // Standard commitment phase — mirror of `denoise_loop_rcd`.
        constraint.rebuild(&tokens, mask);
        let mut any_changed = false;

        for p in 0..seq_len {
            if tokens[p] != mask {
                continue;
            }

            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = simd::simd_max_f32(logits_p);
            let exp_buf = &mut bctx.all_attn_weights[..vocab];
            exp_buf[..vocab].copy_from_slice(logits_p);
            simd::simd_add_scalar_inplace(&mut exp_buf[..vocab], -max_l);
            simd::simd_exp_inplace(&mut exp_buf[..vocab]);
            let sum_exp = simd::simd_sum_f32(&exp_buf[..vocab]);
            let inv_sum = 1.0 / sum_exp;

            let mut best_token = mask;
            let mut best_exp = 0.0f32;
            for t in 0..vocab {
                if t == mask {
                    continue;
                }
                if !constraint.is_valid(p, t, &tokens) {
                    continue;
                }
                let e = exp_buf[t];
                if e > best_exp {
                    best_exp = e;
                    best_token = t;
                }
            }
            let best_prob = best_exp * inv_sum;

            if best_prob >= confidence_threshold && best_token != mask {
                tokens[p] = best_token;
                any_changed = true;
                remaining -= 1;
            }
        }

        // RCD pass: compute entropy-weighted residual for every still-masked
        // position and write to `rcd_residual_embeddings` (h_pre,t for 3SR).
        let mut wrote_residual = false;
        for p in 0..seq_len {
            if tokens[p] != mask {
                continue;
            }

            let logits_p = &bctx.all_logits[p * vocab..(p + 1) * vocab];
            let max_l = simd::simd_max_f32(logits_p);
            softmax_scratch[..vocab].copy_from_slice(logits_p);
            simd::simd_add_scalar_inplace(&mut softmax_scratch[..vocab], -max_l);
            simd::simd_exp_inplace(&mut softmax_scratch[..vocab]);
            let sum_exp = simd::simd_sum_f32(&softmax_scratch[..vocab]);
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                simd::simd_scale_inplace(&mut softmax_scratch[..vocab], inv);
            }

            let alpha = normalized_entropy(&softmax_scratch[..vocab], log_vocab);
            compute_residual(
                &softmax_scratch[..vocab],
                &weights.wte,
                n,
                &mut residual_scratch,
            );
            interpolate_residual(
                mask_emb,
                &residual_scratch,
                alpha,
                &mut bctx.rcd_residual_embeddings[p * n..(p + 1) * n],
            );
            wrote_residual = true;
        }
        bctx.rcd_active = wrote_residual;

        // 3SR pass (Plan 291). Step 0 has no z_prev → skip warm-start this step;
        // forward on step 1 will use the RCD residual directly. From step ≥ 1,
        // classify transitions between z_prev and the current `tokens`, compute
        // γ per position, and lerp between prev_residual (h⋆) and rcd_residual
        // (h_pre) into tsr_warm_start_embeddings. tsr_active gates the next
        // forward's embedding lookup to prefer the 3SR buffer.
        if step > 0 && wrote_residual {
            classify_transitions(&z_prev_tokens, &tokens, mask, &mut transition_scratch);
            // visible_fraction_t = (seq_len - remaining) / seq_len.
            let visible_fraction = if seq_len > 0 {
                (seq_len - remaining) as f32 / seq_len as f32
            } else {
                0.0
            };
            compute_gammas(
                &transition_scratch,
                visible_fraction,
                tsr_cfg,
                &mut gamma_scratch,
            );
            // h_pre_t = rcd_residual_embeddings (just written above).
            // h_star_next = prev_residual (stashed from prior step).
            // out = tsr_warm_start_embeddings.
            warm_start_lerp(
                &prev_residual,
                &bctx.rcd_residual_embeddings,
                &gamma_scratch,
                n,
                &mut bctx.tsr_warm_start_embeddings,
            );
            bctx.tsr_active = true;
        } else {
            bctx.tsr_active = false;
        }

        // Stash the current step's residual buffer as next step's h_star_next.
        // Only the still-masked positions have meaningful residuals — the rest
        // are stale but unused (their transition type will be UnchangedVisible
        // or NewlyRevealed, not StillMasked, so γ will not select them).
        if wrote_residual {
            prev_residual[..seq_len * n]
                .copy_from_slice(&bctx.rcd_residual_embeddings[..seq_len * n]);
        }

        // Snapshot current tokens as next iteration's z_prev.
        z_prev_tokens[..seq_len].copy_from_slice(&tokens[..seq_len]);

        if !any_changed && remaining == 0 {
            converged_step = step;
            break;
        }
    }

    // Leave the context in a clean state for callers that reuse it.
    bctx.rcd_active = false;
    bctx.tsr_active = false;

    if remaining == 0 && converged_step == n_steps {
        converged_step = n_steps - 1;
    }

    (tokens, converged_step)
}
