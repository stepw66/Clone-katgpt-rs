// ═══════════════════════════════════════════════════════════════
// Set-Causal Attention Forward (Research 376 Phase 0 T0.2)
// ═══════════════════════════════════════════════════════════════
//
// Plan 401 (2026-07-06): Moved from root `src/dllm.rs` (lines 2103-2320
// + 5 of 7 Research 376 T0.2 tests). Root keeps a re-export shim at
// `crate::dllm::forward_set_causal_positions` so every historical caller
// (notably `src/speculative/set_diffusion.rs` and the 2 comparison tests
// that also need `forward_block_causal_positions` / `forward_bidirectional_positions`)
// continues to resolve.
//
// Why this is pure inference (not training code): the function uses only
// `TransformerWeights`, `Config`, the `katgpt_types::{rmsnorm, matmul,
// matmul_relu, kv_dim}` helpers, and `katgpt_core::simd::*`. No gradients,
// no backprop, no loss, no training-specific types. The old "Root-resident
// by design (Issue 033 §C, Option C)" comment was obsolete — same lesson
// Plans 399/400 documented for the d2f/flashar cluster.

use katgpt_core::simd;
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;
use katgpt_types::{kv_dim, matmul, matmul_relu, rmsnorm};

/// Set-causal attention forward pass — generalizes
/// [`forward_block_causal_positions`] to arbitrary position-set orderings.
///
/// This is the CPU reference for the SW-SetDLM (Set Diffusion) training
/// objective (Arriola & Kuleshov, arXiv:2607.01775, Research 376). The GPU
/// counterpart is
/// `riir-gpu/src/kernels/attention_score_set_causal.wgsl`; the micro-model
/// reference is `riir-poc/src/set_diffusion_poc.rs::AttentionModel::forward_ordered`.
///
/// # Attention pattern
///
/// For each query position `q` with `gen_step_q = position_order[q]`, attends
/// to all key positions `t` where `position_order[t] <= gen_step_q` — i.e.,
/// positions revealed in the **same generation set** OR **earlier sets**.
/// This realizes the paper's M_SD (set-diagonal) + M_OSC (offset set-causal)
/// + M_SC (set-causal) mask composition as a single eligibility rule.
///
/// # Convention (matches the WGSL kernel)
///
/// `position_order[p]` = the generation step at which position `p` is revealed
/// (0-indexed). Lower step = revealed earlier. This is the **inverse permutation**
/// of the ordering — not the ordering itself.
///
/// # Block-causal is a strict special case
///
/// When `position_order[p] = p / block_size`, positions in the same block share
/// a generation step and the eligibility rule reduces to the prefix
/// `[0..end_of_current_block]` — exactly [`forward_block_causal_positions`]
/// with `causal_block_size = block_size`. The test
/// `test_set_causal_matches_block_causal_when_block_ordered` (still in root,
/// because it also needs `forward_block_causal_positions`) verifies
/// bit-identical output.
///
/// # Common instantiations
///
/// | Method | `position_order` | Effect |
/// |--------|-----------------|--------|
/// | Block-causal (D2F) | `[0,0,0,0, 1,1,1,1, ...]` (p / B) | Prefix mask, recovers `forward_block_causal_positions` |
/// | AR (singleton sets) | `[0, 1, 2, 3, ...]` (p) | Lower-triangular mask |
/// | MDLM (uniform) | `[0, 0, 0, ...]` (all same step) | Fully bidirectional |
/// | SW-SetDLM | sampled from `PositionOffsetSchedule` | Arbitrary sets |
///
/// # Returns
///
/// `(all_logits, all_attn_weights)` where `all_attn_weights[q][h * seq_len + t]`
/// is the attention weight from query `q` to key `t` under head `h`. Weights
/// to ineligible positions (`position_order[t] > position_order[q]`) are
/// exactly 0.0.
pub fn forward_set_causal_positions(
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    position_order: &[usize],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    assert_eq!(
        position_order.len(),
        tokens.len(),
        "position_order must have same length as tokens ({}), got {}",
        tokens.len(),
        position_order.len(),
    );
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();
    let layer = &weights.layers[0];

    // Phase A: K/V projections for all positions (mask-independent — identical
    // to forward_block_causal_positions. The set-causal constraint only
    // affects which keys a query attends to, not how keys are computed.)
    let mut k_cache = vec![0.0f32; seq_len * kvd];
    let mut v_cache = vec![0.0f32; seq_len * kvd];
    let mut x_norm2_all = vec![0.0f32; seq_len * n];
    let mut xr_all = vec![0.0f32; seq_len * n];

    let mut x_buf = vec![0.0f32; n];
    let mut k_buf = vec![0.0f32; kvd];
    let mut v_buf = vec![0.0f32; kvd];

    for (p, &token) in tokens.iter().enumerate().take(seq_len) {
        simd::simd_add_into(
            &mut x_buf,
            &weights.wte[token * n..(token + 1) * n],
            &weights.wpe[p * n..(p + 1) * n],
        );
        rmsnorm(&mut x_buf);
        xr_all[p * n..(p + 1) * n].copy_from_slice(&x_buf);
        rmsnorm(&mut x_buf);
        x_norm2_all[p * n..(p + 1) * n].copy_from_slice(&x_buf);
        matmul(&mut k_buf, &layer.attn_wk, &x_buf, kvd, n);
        matmul(&mut v_buf, &layer.attn_wv, &x_buf, kvd, n);
        k_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&k_buf);
        v_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&v_buf);
    }

    // Phase B: Set-causal attention with masked softmax.
    //
    // We compute exp() ONLY on eligible positions (those with
    // position_order[t] <= q_gen_step). Ineligible positions are explicitly
    // zeroed and skipped. This avoids feeding -inf or huge-negative values
    // through the SIMD polynomial exp (which doesn't handle special values —
    // the Cephes range-reduction saturates and produces NaN). The scalar
    // f32::exp on eligible positions is correct for all finite inputs.
    let mut all_logits = vec![vec![0.0f32; config.vocab_size]; seq_len];
    let mut all_attn_weights = vec![vec![0.0f32; config.n_head * seq_len]; seq_len];

    let mut q_buf = vec![0.0f32; n];
    let mut attn_out_buf = vec![0.0f32; n];
    let mut scores_buf = vec![0.0f32; seq_len];
    let mut x_proj = vec![0.0f32; n];
    let mut hidden = vec![0.0f32; config.mlp_hidden];
    let mut x_mlp = vec![0.0f32; n];
    let mut xr2_buf = vec![0.0f32; n];

    for q in 0..seq_len {
        x_buf.copy_from_slice(&x_norm2_all[q * n..(q + 1) * n]);
        matmul(&mut q_buf, &layer.attn_wq, &x_buf, n, n);

        let q_gen_step = position_order[q];

        // Per-head masked attention. attn_out_buf accumulates across heads
        // (same layout as attention_forward_safe_into's output).
        attn_out_buf.fill(0.0);
        for h in 0..config.n_head {
            let kv_group = h * config.n_kv_head / config.n_head;
            let q_off = h * hd;
            let kv_off = kv_group * hd;

            // Pass 1: compute raw scores for ELIGIBLE positions only, find max.
            // (Position q itself is always eligible since position_order[q] <= q_gen_step,
            // so max_score is guaranteed to advance past -inf.)
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_len {
                if position_order[t] <= q_gen_step {
                    let dot = simd::simd_dot_f32(
                        &q_buf[q_off..q_off + hd],
                        &k_cache[t * kvd + kv_off..t * kvd + kv_off + hd],
                        hd,
                    );
                    scores_buf[t] = dot * scale;
                    if scores_buf[t] > max_score {
                        max_score = scores_buf[t];
                    }
                } else {
                    scores_buf[t] = 0.0; // placeholder, never contributes
                }
            }

            // Pass 2: exp(score - max) for eligible positions, 0 for ineligible.
            // Scalar exp (not SIMD) because the eligible set is typically
            // non-contiguous and we must not feed garbage to the polynomial exp.
            let mut sum_exp = 0.0f32;
            for t in 0..seq_len {
                if position_order[t] <= q_gen_step {
                    let e = (scores_buf[t] - max_score).exp();
                    scores_buf[t] = e;
                    sum_exp += e;
                } else {
                    scores_buf[t] = 0.0;
                }
            }

            // Normalize over eligible positions.
            let inv_sum = 1.0 / sum_exp;
            for t in 0..seq_len {
                if position_order[t] <= q_gen_step {
                    scores_buf[t] *= inv_sum;
                }
            }

            // Persist weights for inspection/debugging. Ineligible positions
            // are exactly 0.0 (never touched in passes 2/3).
            all_attn_weights[q][h * seq_len..h * seq_len + seq_len]
                .copy_from_slice(&scores_buf[..seq_len]);

            // Weighted value sum over eligible positions only.
            for t in 0..seq_len {
                let s = scores_buf[t];
                if s > 0.0 {
                    let v_row = &v_cache[t * kvd + kv_off..t * kvd + kv_off + hd];
                    simd::simd_fused_scale_acc(
                        &mut attn_out_buf[q_off..q_off + hd],
                        v_row,
                        s,
                        hd,
                    );
                }
            }
        }

        // Output projection + residual + MLP + logits (identical to
        // forward_block_causal_positions — set-causal only changes the
        // attention output, not the downstream pipeline).
        matmul(&mut x_proj, &layer.attn_wo, &attn_out_buf, n, n);
        simd::simd_add_inplace(&mut x_proj, &xr_all[q * n..(q + 1) * n]);

        xr2_buf[..n].copy_from_slice(&x_proj[..n]);
        rmsnorm(&mut x_proj);
        matmul_relu(&mut hidden, &layer.mlp_w1, &x_proj, config.mlp_hidden, n);
        matmul(&mut x_mlp, &layer.mlp_w2, &hidden, n, config.mlp_hidden);
        simd::simd_add_inplace(&mut x_mlp[..n], &xr2_buf[..n]);

        matmul(
            &mut all_logits[q],
            &weights.lm_head,
            &x_mlp,
            config.vocab_size,
            n,
        );
    }

    (all_logits, all_attn_weights)
}

// ── Research 376 Phase 0 T0.2: Set-Causal Attention (PURE tests) ──
//
// 5 of the 7 original tests moved with the function. The 2 comparison tests
// (`test_set_causal_matches_block_causal_when_block_ordered`,
//  `test_set_causal_mdlm_all_one_set_is_bidirectional`) stayed in root
// `src/dllm.rs` because they additionally need `forward_block_causal_positions`
// / `forward_bidirectional_positions` (not yet extracted — deferred to Plan 402).
// They call this function via root's re-export shim.

#[cfg(test)]
mod tests {
    use super::forward_set_causal_positions;
    use katgpt_transformer::TransformerWeights;
    use katgpt_types::{Config, Rng};

    #[test]
    fn test_set_causal_mask_zeros_ineligible_positions() {
        // GOAT G1: positions with position_order[t] > position_order[q]
        // must receive EXACTLY 0.0 attention weight, for every query and head.
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // Arbitrary non-block ordering: positions 0,1 are set 0,
        // positions 2,3,4 are set 1, positions 5,6,7 are set 2.
        let position_order = vec![0, 0, 1, 1, 1, 2, 2, 2];

        let (_, attn) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        for q in 0..tokens.len() {
            let q_gen_step = position_order[q];
            for h in 0..config.n_head {
                for t in 0..tokens.len() {
                    let w = attn[q][h * tokens.len() + t];
                    if position_order[t] > q_gen_step {
                        assert_eq!(
                            w, 0.0,
                            "Position {t} (gen_step={}) should have 0 weight from query {q} \
                             (gen_step={q_gen_step}) under head {h}, got {w}",
                            position_order[t],
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_set_causal_self_attention_always_allowed() {
        // GOAT G1 invariant: position q always attends to itself
        // (position_order[q] <= position_order[q] is trivially true).
        // This guarantees the softmax denominator is always >= exp(0) > 0
        // after the max-subtraction, preventing NaN.
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![3, 1, 4, 1, 5, 9, 2, 6];

        // SW-SetDLM-style random-ish ordering
        let position_order = vec![2, 0, 1, 0, 3, 1, 2, 3];

        let (_, attn) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        for q in 0..tokens.len() {
            for h in 0..config.n_head {
                let self_weight = attn[q][h * tokens.len() + q];
                assert!(
                    self_weight > 0.0,
                    "Self-attention weight at q={q}, h={h} should be > 0, got {self_weight}",
                );
                assert!(
                    self_weight.is_finite(),
                    "Self-attention weight at q={q}, h={h} should be finite, got {self_weight}",
                );
            }
        }
    }

    #[test]
    fn test_set_causal_weights_sum_to_one_over_eligible() {
        // GOAT G1: attention weights over eligible positions must sum to 1.0
        // (proper masked softmax). Combined with the zero-ineligible test,
        // this confirms the full softmax is mathematically valid.
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];

        // SW-SetDLM-style: overlapping sets
        let position_order = vec![0, 1, 0, 2, 1, 3, 2, 3];

        let (_, attn) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        for q in 0..tokens.len() {
            let q_gen_step = position_order[q];
            for h in 0..config.n_head {
                let sum: f32 = (0..tokens.len())
                    .filter(|&t| position_order[t] <= q_gen_step)
                    .map(|t| attn[q][h * tokens.len() + t])
                    .sum();
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Eligible-position weight sum at q={q}, h={h} should be 1.0, got {sum}",
                );
            }
        }
    }

    #[test]
    fn test_set_causal_ar_singleton_each_position_own_set() {
        // AR limit: position_order[p] = p means each position is its own set.
        // Position q attends to positions [0..=q] (lower-triangular mask).
        // This is the AR extreme of the w schedule (w = 1/L).
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];

        let position_order: Vec<usize> = (0..tokens.len()).collect();

        let (_, attn) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        for q in 0..tokens.len() {
            for h in 0..config.n_head {
                for t in 0..tokens.len() {
                    let w = attn[q][h * tokens.len() + t];
                    if t > q {
                        // Future positions must be masked
                        assert_eq!(
                            w, 0.0,
                            "AR mask: position {t} should be masked from query {q} (t > q), got w={w}",
                        );
                    } else {
                        // Past + self positions should generally have non-zero weight
                        // (could be 0 in pathological cases, but for random weights
                        // and scale > 0 this should not happen)
                        assert!(
                            w >= 0.0,
                            "AR mask: position {t} weight from query {q} should be >= 0, got {w}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_set_causal_length_mismatch_panics() {
        // Defensive: position_order.len() != tokens.len() must panic
        // (caught by debug_assert in production, assert_eq in the function).
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3];
        let bad_order = vec![0, 1, 2]; // length 3, should be 4

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            forward_set_causal_positions(&weights, &tokens, &config, &bad_order);
        }));
        assert!(
            result.is_err(),
            "forward_set_causal_positions should panic on length mismatch"
        );
    }
}
