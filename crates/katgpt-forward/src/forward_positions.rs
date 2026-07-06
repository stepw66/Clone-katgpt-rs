// ═══════════════════════════════════════════════════════════════
// Forward-Positions Cluster — Bidirectional + Block-Causal attention
// ═══════════════════════════════════════════════════════════════
//
// Plan 402 (2026-07-06): Moved from root `src/dllm.rs`. This module
// consolidates the position-wise forward variants that share the
// `attention_forward_safe_into` kernel (from `crate::d2f_context`):
//
// - `BidirectionalContext` — pre-allocated buffers reused across positions
//   (the zero-alloc substrate for `forward_bidirectional_positions_into`
//   and the `denoise_loop*` family that stays in root).
// - `forward_bidirectional_positions` — allocating wrapper (full attention).
// - `forward_bidirectional_positions_into` — zero-alloc variant.
// - `attention_forward_safe` — allocating attention wrapper (kept for
//   cohesion; the `_into` variant is the hot-path kernel).
// - `forward_block_causal_positions` — block-causal (D2F) forward.
//
// Root keeps a re-export shim at `crate::dllm::*` so every historical
// caller (notably `denoise_loop*`, `evaluate_accuracy` training code,
// `forward_save`) continues to resolve.
//
// ## Why the struct fields are `pub`
//
// `BidirectionalContext` is constructed and mutated by code that stays in
// root: `denoise_loop_rcd` writes `rcd_residual_embeddings` + `rcd_active`;
// `denoise_loop_rcd_3sr` writes `tsr_warm_start_embeddings` + `tsr_active`;
// every `denoise_loop*` reads `all_logits` / `all_attn_weights`. The fields
// must be `pub` so root's re-export-based access compiles. This mirrors the
// standard "move type, re-export, leave consumers in root" pattern.

use crate::d2f_context::attention_forward_safe_into;
use katgpt_core::simd;
use katgpt_transformer::TransformerWeights;
use katgpt_types::Config;
use katgpt_types::{kv_dim, matmul, matmul_relu, rmsnorm};

/// Pre-allocated buffers for `forward_bidirectional_positions`, avoiding per-position Vec allocations.
///
/// Plan 402 (2026-07-06): moved from root `src/dllm.rs`. Fields are `pub` because
/// root's `denoise_loop_rcd` / `denoise_loop_rcd_3sr` (which stay in root) write
/// directly to the cfg-gated `rcd_residual_embeddings` / `tsr_warm_start_embeddings`
/// buffers and the `rcd_active` / `tsr_active` flags after each commitment phase.
pub struct BidirectionalContext {
    pub x: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub x_proj: Vec<f32>,
    pub xr2: Vec<f32>,
    pub hidden: Vec<f32>,
    pub x_mlp: Vec<f32>,
    pub logits: Vec<f32>,
    // Attention scratch buffers (reused across positions)
    pub attn_out_buf: Vec<f32>,
    pub attn_weights_buf: Vec<f32>,
    pub scores_buf: Vec<f32>,
    // Cross-position buffers (resized per call, reused across calls)
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub x_norm2_all: Vec<f32>,
    pub xr_all: Vec<f32>,
    // Output buffers (pre-allocated to max capacity, sliced per call)
    pub all_logits: Vec<f32>,
    pub all_attn_weights: Vec<f32>,
    /// RCD residual embedding override for masked positions: `[block_size * n_embd]`.
    /// When `rcd_active` is true and a position's token is the mask token,
    /// this buffer's slice `[p*n..(p+1)*n]` is used instead of `wte[mask_token]`.
    /// Written by `denoise_loop_rcd` after each commitment phase, read in the next forward pass.
    #[cfg(feature = "rcd_residual")]
    pub rcd_residual_embeddings: Vec<f32>,
    /// Whether the residual embedding override is active for the current step.
    /// Step 0 has no residuals yet (all-mask), so this is false on the first forward pass.
    #[cfg(feature = "rcd_residual")]
    pub rcd_active: bool,
    /// 3SR warm-start embedding override for masked positions: `[block_size * n_embd]`.
    /// When `tsr_active` is true and a position's token is the mask token, this
    /// buffer's slice `[p*n..(p+1)*n]` is used *instead of* `rcd_residual_embeddings`
    /// (it pre-composes the RCD residual with the prior step's solved state via
    /// `warm_start_lerp`). Written by `denoise_loop_rcd_3sr` after each commitment
    /// phase, read in the next forward pass. Plan 291, Research 265.
    #[cfg(feature = "d2f_3sr_warm_start")]
    pub tsr_warm_start_embeddings: Vec<f32>,
    /// Whether the 3SR warm-start override is active for the current step.
    /// Step 0 has no z_prev (no transitions to classify), so this is false on
    /// the first forward pass — same rule as `rcd_active`.
    #[cfg(feature = "d2f_3sr_warm_start")]
    pub tsr_active: bool,
}

impl BidirectionalContext {
    pub fn new(config: &Config) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let bs = config.block_size;
        Self {
            x: vec![0.0f32; n],
            q: vec![0.0f32; n],
            k: vec![0.0f32; kvd],
            v: vec![0.0f32; kvd],
            x_proj: vec![0.0f32; n],
            xr2: vec![0.0f32; n],
            hidden: vec![0.0f32; config.mlp_hidden],
            x_mlp: vec![0.0f32; n],
            logits: vec![0.0f32; config.vocab_size],
            attn_out_buf: vec![0.0f32; n],
            attn_weights_buf: vec![0.0f32; config.n_head * bs],
            scores_buf: vec![0.0f32; bs],
            k_cache: vec![0.0f32; bs * kvd],
            v_cache: vec![0.0f32; bs * kvd],
            x_norm2_all: vec![0.0f32; bs * n],
            xr_all: vec![0.0f32; bs * n],
            all_logits: vec![0.0f32; bs * config.vocab_size],
            all_attn_weights: vec![0.0f32; bs * config.n_head * bs],
            #[cfg(feature = "rcd_residual")]
            rcd_residual_embeddings: vec![0.0f32; bs * n],
            #[cfg(feature = "rcd_residual")]
            rcd_active: false,
            #[cfg(feature = "d2f_3sr_warm_start")]
            tsr_warm_start_embeddings: vec![0.0f32; bs * n],
            #[cfg(feature = "d2f_3sr_warm_start")]
            tsr_active: false,
        }
    }
}

/// Bidirectional forward pass for all positions.
/// Each position attends to ALL other positions (no causal mask).
/// Returns logits per position and per-head attention weights.
pub fn forward_bidirectional_positions(
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
) -> (Vec<f32>, Vec<f32>) {
    let mut bctx = BidirectionalContext::new(config);
    let (logits_len, attn_len) =
        forward_bidirectional_positions_into(weights, tokens, config, &mut bctx);
    (
        bctx.all_logits[..logits_len].to_vec(),
        bctx.all_attn_weights[..attn_len].to_vec(),
    )
}

/// Zero-alloc variant of [`forward_bidirectional_positions`] that reuses a pre-allocated context.
///
/// Pass a `BidirectionalContext` to avoid per-call heap allocation when calling in a loop
/// (e.g., `denoise_loop`).
///
/// Writes results into `bctx.all_logits` and `bctx.all_attn_weights` and returns
/// `(logits_len, attn_len)` so the caller can index the context buffers directly.
/// This avoids lifetime issues when the caller needs to reuse other fields from `bctx`.
pub fn forward_bidirectional_positions_into(
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    bctx: &mut BidirectionalContext,
) -> (usize, usize) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();

    // Phase A: Compute K/V for all positions.
    // (k_cache, v_cache, x_norm2_all, xr_all are fully overwritten for [0..seq_len)
    // inside the loop, so no pre-zero is needed.)
    for (p, &token) in tokens.iter().enumerate().take(seq_len) {
        // RCD / 3SR: when an embedding override is active and this position is
        // still masked, use the override buffer instead of `wte[mask_token]`.
        // 3SR warm-start takes precedence — it pre-composes the RCD residual
        // (which serves as `h_pre_t`) with the prior step's solved state via
        // `warm_start_lerp`. When 3SR is inactive, the RCD residual is used.
        //
        // This is a single branch per position — does not touch the inner matmul loops.
        #[cfg(feature = "d2f_3sr_warm_start")]
        let emb = if bctx.tsr_active && token == config.mask_token {
            &bctx.tsr_warm_start_embeddings[p * n..(p + 1) * n]
        } else if bctx.rcd_active && token == config.mask_token {
            &bctx.rcd_residual_embeddings[p * n..(p + 1) * n]
        } else {
            &weights.wte[token * n..(token + 1) * n]
        };
        #[cfg(all(feature = "rcd_residual", not(feature = "d2f_3sr_warm_start")))]
        let emb = if bctx.rcd_active && token == config.mask_token {
            &bctx.rcd_residual_embeddings[p * n..(p + 1) * n]
        } else {
            &weights.wte[token * n..(token + 1) * n]
        };
        #[cfg(not(feature = "rcd_residual"))]
        let emb = &weights.wte[token * n..(token + 1) * n];
        simd::simd_add_into(&mut bctx.x, emb, &weights.wpe[p * n..(p + 1) * n]);
        rmsnorm(&mut bctx.x);
        bctx.xr_all[p * n..(p + 1) * n].copy_from_slice(&bctx.x);
        rmsnorm(&mut bctx.x);
        bctx.x_norm2_all[p * n..(p + 1) * n].copy_from_slice(&bctx.x);

        let layer = &weights.layers[0];
        // matmul overwrites all `kvd` rows of bctx.k / bctx.v, so no pre-zero needed.
        matmul(&mut bctx.k, &layer.attn_wk, &bctx.x, kvd, n);
        matmul(&mut bctx.v, &layer.attn_wv, &bctx.x, kvd, n);
        bctx.k_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&bctx.k);
        bctx.v_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&bctx.v);
    }

    // Phase B: Bidirectional attention for all positions
    let vocab = config.vocab_size;
    let n_heads = config.n_head;
    let logits_len = seq_len * vocab;
    let attn_len = seq_len * n_heads * seq_len;
    // (all_logits[..logits_len] and all_attn_weights[..attn_len] are fully
    // overwritten inside the loop, so no pre-zero is needed.)
    let layer = &weights.layers[0];

    for p in 0..seq_len {
        bctx.x
            .copy_from_slice(&bctx.x_norm2_all[p * n..(p + 1) * n]);

        // matmul overwrites all `n` rows of bctx.q, so no pre-zero needed.
        matmul(&mut bctx.q, &layer.attn_wq, &bctx.x, n, n);

        attention_forward_safe_into(
            &bctx.q,
            &bctx.k_cache,
            &bctx.v_cache,
            config.n_head,
            config.n_kv_head,
            hd,
            kvd,
            seq_len,
            scale,
            &mut bctx.attn_out_buf,
            &mut bctx.attn_weights_buf,
            &mut bctx.scores_buf,
        );

        // `matmul` overwrites all n rows of x_proj (output[r] = dot(...)), so no
        // pre-zero is needed — matches the sibling q/k/v/hidden/x_mlp buffers.
        matmul(&mut bctx.x_proj, &layer.attn_wo, &bctx.attn_out_buf, n, n);
        simd::simd_add_inplace(&mut bctx.x_proj, &bctx.xr_all[p * n..(p + 1) * n]);

        // MLP
        bctx.xr2.copy_from_slice(&bctx.x_proj);
        rmsnorm(&mut bctx.x_proj);
        // matmul_relu overwrites all `mlp_hidden` rows of bctx.hidden.
        matmul_relu(
            &mut bctx.hidden,
            &layer.mlp_w1,
            &bctx.x_proj,
            config.mlp_hidden,
            n,
        );
        // matmul overwrites all `n` rows of bctx.x_mlp.
        matmul(
            &mut bctx.x_mlp,
            &layer.mlp_w2,
            &bctx.hidden,
            n,
            config.mlp_hidden,
        );
        simd::simd_add_inplace(&mut bctx.x_mlp, &bctx.xr2);

        // matmul overwrites all `vocab_size` rows of bctx.logits.
        matmul(
            &mut bctx.logits,
            &weights.lm_head,
            &bctx.x_mlp,
            config.vocab_size,
            n,
        );
        bctx.all_logits[p * vocab..(p + 1) * vocab].copy_from_slice(&bctx.logits);
        bctx.all_attn_weights[p * n_heads * seq_len..(p + 1) * n_heads * seq_len]
            .copy_from_slice(&bctx.attn_weights_buf[..n_heads * seq_len]);
    }

    (logits_len, attn_len)
}

/// Safe bidirectional attention for one query position.
/// Returns (attn_output[n_embd], attn_weights[n_head * seq_len]).
///
/// Plan 402 (2026-07-06): moved from root `src/dllm.rs`. The `_into` variant
/// (in `crate::d2f_context`) is the zero-alloc hot-path kernel shared across
/// all 5 callers; this allocating wrapper is kept for API completeness.
#[allow(clippy::too_many_arguments)] // attention kernel: all args are distinct tensor dims/slices
pub fn attention_forward_safe(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    kv_dim: usize,
    seq_len: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_embd = n_head * head_dim;
    let mut attn_out = vec![0.0f32; n_embd];
    let mut all_weights = vec![0.0f32; n_head * seq_len];
    let mut scores = vec![0.0f32; seq_len];
    attention_forward_safe_into(
        q,
        k_all,
        v_all,
        n_head,
        n_kv_head,
        head_dim,
        kv_dim,
        seq_len,
        scale,
        &mut attn_out,
        &mut all_weights,
        &mut scores,
    );
    (attn_out, all_weights)
}

/// Block-causal forward pass for all positions.
///
/// Each position `p` attends to positions `[0..end_of_current_block]` where
/// `end_of_current_block = (p / causal_block_size + 1) * causal_block_size`.
/// This is the D2F (Discrete diffusion Forcing) attention pattern.
///
/// Returns nested `Vec<Vec<f32>>` (per-position logits + per-position per-head
/// attention weights padded to `seq_len`).
pub fn forward_block_causal_positions(
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    causal_block_size: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();
    let layer = &weights.layers[0];

    // Phase A: K/V for all positions
    let mut k_cache = vec![0.0f32; seq_len * kvd];
    let mut v_cache = vec![0.0f32; seq_len * kvd];
    let mut x_norm2_all = vec![0.0f32; seq_len * n];
    let mut xr_all = vec![0.0f32; seq_len * n];

    // Pre-allocate scratch buffers (reused across positions)
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

    // Phase B: Block-causal attention
    // Pre-allocate output buffers upfront (avoid per-position clone)
    let mut all_logits = vec![vec![0.0f32; config.vocab_size]; seq_len];
    let mut all_attn_weights = vec![vec![0.0f32; config.n_head * seq_len]; seq_len];

    // Pre-allocate Phase B scratch buffers (reused across positions)
    let mut q_buf = vec![0.0f32; n];
    let mut attn_out_buf = vec![0.0f32; n];
    let mut attn_w_buf = vec![0.0f32; config.n_head * seq_len];
    let mut scores_buf = vec![0.0f32; seq_len];
    let mut x_proj = vec![0.0f32; n];
    let mut hidden = vec![0.0f32; config.mlp_hidden];
    let mut x_mlp = vec![0.0f32; n];
    let mut xr2_buf = vec![0.0f32; n];

    for p in 0..seq_len {
        x_buf.copy_from_slice(&x_norm2_all[p * n..(p + 1) * n]);
        matmul(&mut q_buf, &layer.attn_wq, &x_buf, n, n);

        // Block-causal: attend to positions [0..end_of_current_block]
        let block_end = (p / causal_block_size + 1) * causal_block_size;
        let t_n = block_end.min(seq_len);

        attention_forward_safe_into(
            &q_buf,
            &k_cache,
            &v_cache,
            config.n_head,
            config.n_kv_head,
            hd,
            kvd,
            t_n,
            scale,
            &mut attn_out_buf,
            &mut attn_w_buf,
            &mut scores_buf,
        );

        // Pad attn_w to seq_len for consistent output (zero-fill then slice-copy per head)
        all_attn_weights[p].fill(0.0f32);
        for h in 0..config.n_head {
            all_attn_weights[p][h * seq_len..h * seq_len + t_n]
                .copy_from_slice(&attn_w_buf[h * t_n..h * t_n + t_n]);
        }

        matmul(&mut x_proj, &layer.attn_wo, &attn_out_buf, n, n);
        simd::simd_add_inplace(&mut x_proj, &xr_all[p * n..(p + 1) * n]);

        xr2_buf[..n].copy_from_slice(&x_proj[..n]);
        rmsnorm(&mut x_proj);
        matmul_relu(&mut hidden, &layer.mlp_w1, &x_proj, config.mlp_hidden, n);
        matmul(&mut x_mlp, &layer.mlp_w2, &hidden, n, config.mlp_hidden);
        simd::simd_add_inplace(&mut x_mlp[..n], &xr2_buf[..n]);

        matmul(
            &mut all_logits[p],
            &weights.lm_head,
            &x_mlp,
            config.vocab_size,
            n,
        );
    }

    (all_logits, all_attn_weights)
}

// ═══════════════════════════════════════════════════════════════
// Research 376 Phase 0 T0.2: Set-Causal comparison tests
// ═══════════════════════════════════════════════════════════════
//
// Plan 402 (2026-07-06): These 2 tests moved from root `src/dllm.rs` (they
// were the deferred "comparison" tests in Plan 401). They verify set-causal
// attention reduces to block-causal / bidirectional in the limit cases.
// Now that all 3 sibling forward functions live in katgpt-forward, the tests
// can live here too.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward_set_causal::forward_set_causal_positions;
    use katgpt_types::Rng;

    #[cfg(feature = "set_diffusion")]
    #[test]
    fn test_set_causal_matches_block_causal_when_block_ordered() {
        // GOAT G1: set-causal with position_order[p] = p / B must produce
        // bit-identical output to forward_block_causal_positions with
        // causal_block_size = B. This proves set-causal is a strict
        // generalization (no regression on the block-causal special case).
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let block_size = 4;

        // Block-causal reference
        let (logits_bc, attn_bc) =
            forward_block_causal_positions(&weights, &tokens, &config, block_size);

        // Set-causal with matching position_order: [0,0,0,0, 1,1,1,1]
        let position_order: Vec<usize> = tokens.iter().map(|&p| p / block_size).collect();
        let (logits_sc, attn_sc) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        // Logits must match within SIMD-vs-scalar exp tolerance.
        // (block-causal uses SIMD Cephes polynomial exp; set-causal uses scalar
        // f32::exp. The ~1 ULP difference accumulates through the value sum
        // and MLP, producing differences up to ~1e-4 on logit magnitudes ~10.)
        assert_eq!(logits_bc.len(), logits_sc.len(), "logits length mismatch");
        for q in 0..tokens.len() {
            assert_eq!(logits_bc[q].len(), logits_sc[q].len(), "vocab length mismatch");
            for v in 0..logits_bc[q].len() {
                let diff = (logits_bc[q][v] - logits_sc[q][v]).abs();
                let max_abs = logits_bc[q][v].abs().max(logits_sc[q][v].abs());
                let rel_tol = (max_abs * 1e-3).max(1e-5);
                assert!(
                    diff < rel_tol,
                    "Logit mismatch at q={q}, v={v}: bc={}, sc={}, diff={diff} (rel_tol={rel_tol})",
                    logits_bc[q][v],
                    logits_sc[q][v],
                );
            }
        }

        // Attention weights must match within exp tolerance.
        for q in 0..tokens.len() {
            for h in 0..config.n_head {
                for t in 0..tokens.len() {
                    let w_bc = attn_bc[q][h * tokens.len() + t];
                    let w_sc = attn_sc[q][h * tokens.len() + t];
                    let diff = (w_bc - w_sc).abs();
                    assert!(
                        diff < 1e-5,
                        "Attention weight mismatch at q={q}, h={h}, t={t}: \
                         bc={w_bc}, sc={w_sc}, diff={diff}",
                    );
                }
            }
        }
    }

    #[cfg(feature = "set_diffusion")]
    #[test]
    fn test_set_causal_mdlm_all_one_set_is_bidirectional() {
        // MDLM limit: position_order[p] = 0 for all p means all positions are
        // in the same set. Every position attends to every position
        // (fully bidirectional / standard softmax attention).
        //
        // forward_bidirectional_positions returns FLAT (Vec<f32>, Vec<f32>),
        // unlike forward_set_causal_positions's nested (Vec<Vec<f32>>, Vec<Vec<f32>>).
        // Index bidirectional as: logits[p * vocab + v], attn[p * (n_head*seq_len) + h*seq_len + t].
        let config = Config::micro_dllm();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let seq_len = tokens.len();

        let position_order = vec![0usize; seq_len];

        let (logits_sc, attn_sc) =
            forward_set_causal_positions(&weights, &tokens, &config, &position_order);

        // Compare against forward_bidirectional_positions (the existing
        // unconstrained attention implementation). They should match closely
        // (both compute full attention over all positions).
        let (logits_bi, attn_bi) =
            forward_bidirectional_positions(&weights, &tokens, &config);

        // Attention weights should match within SIMD-vs-scalar exp tolerance.
        // (bidirectional uses SIMD Cephes polynomial exp; set-causal uses scalar
        // f32::exp on eligible positions only — same math, different rounding.)
        for q in 0..seq_len {
            for h in 0..config.n_head {
                for t in 0..seq_len {
                    let w_sc = attn_sc[q][h * seq_len + t];
                    // bidirectional attn is flat: [q * (n_head*seq_len) + h*seq_len + t]
                    let w_bi = attn_bi[q * config.n_head * seq_len + h * seq_len + t];
                    let diff = (w_sc - w_bi).abs();
                    assert!(
                        diff < 1e-5,
                        "MDLM vs bidirectional mismatch at q={q}, h={h}, t={t}: \
                         sc={w_sc}, bi={w_bi}, diff={diff}",
                    );
                }
            }
        }

        // Logits should match within relative tolerance (sc is nested, bi is flat).
        // Accumulated exp differences can reach ~1e-4 on logit magnitudes ~10.
        for q in 0..seq_len {
            for v in 0..config.vocab_size {
                let l_sc = logits_sc[q][v];
                let l_bi = logits_bi[q * config.vocab_size + v];
                let diff = (l_sc - l_bi).abs();
                let max_abs = l_sc.abs().max(l_bi.abs());
                let rel_tol = (max_abs * 1e-3).max(1e-5);
                assert!(
                    diff < rel_tol,
                    "MDLM vs bidirectional logit mismatch at q={q}, v={v}: \
                     sc={l_sc}, bi={l_bi}, diff={diff} (rel_tol={rel_tol})",
                );
            }
        }
    }
}
