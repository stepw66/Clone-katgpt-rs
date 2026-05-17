//! SP-KV forward pass: gated attention with utility-based bias.
//!
//! `attention_head_gated()` is a drop-in replacement for `attention_head()` that adds
//! an optional per-position gate bias to Q·K scores before softmax.
//!
//! ## Gate Bias Semantics
//!
//! | Mode | bias[s] | Effect |
//! |------|---------|--------|
//! | `None` | — | Identical to `attention_head()` |
//! | Soft (training) | `log(u_s + ε)` | Smooth, differentiable masking |
//! | Hard (inference) | `0` or `-∞` | Binary retain/prune |
//! | TAHG (annealing) | Blended | Gradual hardening |
//!
//! Positions within the sliding window always get bias = 0 (unconditionally attended).

use crate::simd::simd_dot_f32;

/// Fused attention head with GQA support and optional SP-KV gate bias.
///
/// Identical to `attention_head()` in `transformer.rs` when `gate_bias` is `None`.
/// When `gate_bias` is `Some(&[bias])`, adds `bias[t]` to each Q·K score:
///
/// ```text
/// score = dot(q, k[t]) * scale + gate_bias[t]
/// ```
///
/// This is the **single insertion point** for SP-KV in the attention kernel.
/// Everything else (softmax, weighted accumulation) remains unchanged.
///
/// ## SAFETY
///
/// Caller must ensure all indices are in bounds:
/// - `q[q_head_offset..q_head_offset + hd]` valid
/// - `key_cache[t * kv_dim + kv_group_offset..t * kv_dim + kv_group_offset + hd]` valid for t in 0..t_n
/// - `value_cache` same layout as `key_cache`
/// - `attn_out[q_head_offset..q_head_offset + hd]` valid
/// - `scores_buf[..t_n]` valid
/// - `gate_bias` if `Some`: `gate_bias[..t_n]` valid
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn attention_head_gated(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    scores_buf: &mut [f32],
    q_head_offset: usize,
    kv_group_offset: usize,
    kv_dim: usize,
    hd: usize,
    t_n: usize,
    scale: f32,
    gate_bias: Option<&[f32]>,
) {
    // Pass 1: compute Q·K scores with optional gate bias, find max for numerical stability
    let mut max_score = f32::NEG_INFINITY;
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        // SAFETY: caller guarantees bounds (see doc comment)
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), hd);
            let k_slice = std::slice::from_raw_parts(key_cache.as_ptr().add(k_off), hd);
            simd_dot_f32(q_slice, k_slice, hd)
        };
        let raw_score = dot * scale;

        // SP-KV gate bias: additive bias per key position
        let score = match gate_bias {
            Some(bias) => {
                // SAFETY: caller guarantees bias[..t_n] valid
                unsafe { raw_score + *bias.get_unchecked(t) }
            }
            None => raw_score,
        };

        unsafe {
            *scores_buf.get_unchecked_mut(t) = score;
        }
        if score > max_score {
            max_score = score;
        }
    }

    // Pass 2: exp(scores - max) and accumulate sum (unchanged from attention_head)
    let mut sum = 0.0f32;
    for t in 0..t_n {
        let exp_val = unsafe { (*scores_buf.get_unchecked(t) - max_score).exp() };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = exp_val;
        }
        sum += exp_val;
    }

    // Pass 3: normalize + weighted value accumulation (unchanged from attention_head)
    let inv_sum = 1.0 / sum;
    for d in 0..hd {
        let mut val = 0.0f32;
        for t in 0..t_n {
            unsafe {
                val += *scores_buf.get_unchecked(t)
                    * inv_sum
                    * *value_cache.get_unchecked(t * kv_dim + kv_group_offset + d);
            }
        }
        unsafe {
            *attn_out.get_unchecked_mut(q_head_offset + d) = val;
        }
    }
}

/// Build per-position gate biases for all positions up to `pos`.
///
/// Convenience function that dispatches to the appropriate mode:
/// - Soft: `log(u_s + ε)` outside window, `0` inside
/// - Hard: `0` if retained or in window, `-∞` otherwise
/// - TAHG: blended soft/hard with annealing
///
/// Writes into `buf.bias[..=pos]`. Positions after `pos` are left unchanged.
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn build_gate_biases(
    buf: &mut crate::sp_kv::types::GateBiasBuffer,
    utilities: &[f32],
    retained: &[bool],
    pos: usize,
    window: usize,
    threshold: f32,
    mode: crate::sp_kv::types::SpKvGateMode,
) {
    use crate::sp_kv::types::SpKvGateMode;

    match mode {
        SpKvGateMode::Soft => buf.build_soft(utilities, pos, window),
        SpKvGateMode::Hard => buf.build_hard(utilities, retained, pos, window, threshold),
        SpKvGateMode::Tahg { step, total_steps } => {
            buf.build_tahg(utilities, pos, window, threshold, step, total_steps);
        }
    }
}

/// Check if a position is within the sliding window from the current decode position.
#[inline(always)]
pub fn is_in_window(current_pos: usize, source_pos: usize, window: usize) -> bool {
    current_pos.saturating_sub(source_pos) < window
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline: attention_head_gated with gate_bias=None matches standard attention.
    /// Compares against a manual reference implementation.
    #[test]
    fn test_gated_none_matches_baseline() {
        let hd = 8;
        let kv_dim = 16; // 2 KV heads × 8 head_dim
        let t_n = 4;

        let q = vec![0.1; kv_dim];
        let key_cache = vec![0.2; t_n * kv_dim];
        let value_cache = vec![0.3; t_n * kv_dim];
        let scale = 1.0 / (hd as f32).sqrt();

        let mut attn_out_gated = vec![0.0; kv_dim];
        let mut scores_gated = vec![0.0; t_n];

        let mut attn_out_baseline = vec![0.0; kv_dim];
        let mut scores_baseline = vec![0.0; t_n];

        unsafe {
            // Gated with None
            attention_head_gated(
                &q,
                &key_cache,
                &value_cache,
                &mut attn_out_gated,
                &mut scores_gated,
                0,
                0,
                kv_dim,
                hd,
                t_n,
                scale,
                None,
            );

            // Manual baseline: same as gated but we compute manually
            for h_off in [0] {
                let mut max_s = f32::NEG_INFINITY;
                for t in 0..t_n {
                    let k_off = t * kv_dim + h_off;
                    let dot: f32 = q[h_off..h_off + hd]
                        .iter()
                        .zip(&key_cache[k_off..k_off + hd])
                        .map(|(&a, &b)| a * b)
                        .sum();
                    let score = dot * scale;
                    scores_baseline[t] = score;
                    if score > max_s {
                        max_s = score;
                    }
                }
                let sum: f32 = scores_baseline.iter().map(|s| (s - max_s).exp()).sum();
                let inv = 1.0 / sum;
                for d in 0..hd {
                    let v: f32 = (0..t_n)
                        .map(|t| {
                            let exp = (scores_baseline[t] - max_s).exp();
                            exp * inv * value_cache[t * kv_dim + h_off + d]
                        })
                        .sum();
                    attn_out_baseline[h_off + d] = v;
                }
            }
        }

        for d in 0..hd {
            assert!(
                (attn_out_gated[d] - attn_out_baseline[d]).abs() < 1e-4,
                "Mismatch at d={d}: gated={gated}, baseline={baseline}",
                gated = attn_out_gated[d],
                baseline = attn_out_baseline[d],
            );
        }
    }

    /// Gate bias = -inf should zero out attention weight for that position.
    #[test]
    fn test_hard_gate_prunes_position() {
        let hd = 4;
        let kv_dim = 4;
        let t_n = 4;
        let scale = 1.0 / (hd as f32).sqrt();

        let q = vec![1.0; kv_dim];
        let key_cache = vec![1.0; t_n * kv_dim];
        let value_cache = vec![1.0; t_n * kv_dim];

        // Gate: keep positions 0,1,3; prune position 2
        let gate_bias = vec![0.0, 0.0, f32::NEG_INFINITY, 0.0];

        let mut attn_out = vec![0.0; kv_dim];
        let mut scores = vec![0.0; t_n];

        unsafe {
            attention_head_gated(
                &q,
                &key_cache,
                &value_cache,
                &mut attn_out,
                &mut scores,
                0,
                0,
                kv_dim,
                hd,
                t_n,
                scale,
                Some(&gate_bias),
            );
        }

        // All value_cache entries are 1.0, so attn_out[d] should be 1.0
        // (weighted average of 1.0s across non-pruned positions)
        for d in 0..hd {
            assert!(
                (attn_out[d] - 1.0).abs() < 1e-4,
                "Expected ~1.0 at d={d}, got {v}",
                v = attn_out[d],
            );
        }

        // Verify pruned position has zero attention weight
        // (positions 0,1,3 each get ~1/3 weight)
        let total_weight: f32 = scores[0] + scores[1] + scores[3];
        assert!(total_weight > 0.0, "Non-pruned weights should be positive");
        assert!(
            scores[2] < 1e-20,
            "Pruned position should have ~0 weight, got {w}",
            w = scores[2],
        );
    }

    /// Window positions should have zero bias regardless of utility.
    #[test]
    fn test_is_in_window() {
        assert!(is_in_window(10, 5, 8)); // 10-5=5 < 8
        assert!(is_in_window(10, 9, 8)); // 10-9=1 < 8
        assert!(!is_in_window(10, 2, 8)); // 10-2=8, NOT < 8
        assert!(is_in_window(5, 0, 128)); // 5-0=5 < 128
    }
}
