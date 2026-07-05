//! D2F Inference Substrate — Plan 398 (2026-07-05).
//!
//! Extracted from root `src/dllm.rs`. This module hosts the **inference-only**
//! substrate shared by the d2f cluster (`speculative/d2f.rs`,
//! `speculative/d2f_verifier.rs`, `speculative/diffusion_sampler.rs`):
//!
//! - `D2fContext` — pre-allocated flat buffers for zero-alloc D2F denoising
//! - `forward_block_causal_with` — block-causal attention forward kernel that
//!   writes into `D2fContext::logits_flat`
//! - `attention_forward_safe_into` — zero-alloc attention helper called by
//!   `forward_block_causal_with` (and re-exported back to root's training
//!   code via `pub use katgpt_forward::...`)
//! - `denoising_accuracy` — fraction of correctly recovered tokens
//!
//! ## Why this lives here
//!
//! Root's `src/dllm.rs` (4782 LOC) mixes training infrastructure
//! (`train_mini_dllm`, `train_mini_set_causal`, `evaluate_set_causal_nelbo`,
//! `generate_pattern_dataset`, `evaluate_accuracy`) with the inference
//! substrate above. The training code must stay in root (it's a research
//! concern, not a published-API concern, and per the modelless-first mandate
//! doesn't belong in a published leaf crate). The inference substrate, on
//! the other hand, blocks the entire d2f cluster from leaving root — so it
//! moves here.
//!
//! Root re-exports these via `pub use katgpt_forward::d2f_context::{...}`,
//! preserving every historical `katgpt_rs::dllm::D2fContext` /
//! `katgpt_rs::dllm::forward_block_causal_with` /
//! `katgpt_rs::dllm::denoising_accuracy` import path.
//!
//! ## DRY note
//!
//! `attention_forward_safe_into` has 5 callers in the codebase:
//! - 4 stay in root (`forward_bidirectional_positions_into`, `forward_save`,
//!   `forward_block_causal_positions`, `attention_forward_safe`)
//! - 1 moves with this file (`forward_block_causal_with`)
//!
//! The function is therefore `pub` here (workspace-internal — this crate is
//! `publish = false`) and root imports it via `use katgpt_forward::...`. This
//! preserves a single source of truth rather than duplicating the body.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use katgpt_core::simd;
use katgpt_core::types::{Config, kv_dim, matmul, matmul_relu, rmsnorm};
use katgpt_transformer::TransformerWeights;

// ═══════════════════════════════════════════════════════════════
// Zero-Alloc D2F Context + Forward
// ═══════════════════════════════════════════════════════════════

/// Pre-allocated buffers for zero-alloc D2F denoising.
///
/// Unlike `SpeculativeContext` (single-token autoregressive), D2F processes
/// all positions in a block simultaneously, so we use flat 2D buffers
/// indexed by `[p * dim..(p+1) * dim]`.
pub struct D2fContext {
    /// Flat KV cache: `[max_seq * kv_dim]`.
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    /// Normalized embeddings per position: `[max_seq * n_embd]`.
    pub x_norm: Vec<f32>,
    /// Residual (pre-norm) embeddings per position: `[max_seq * n_embd]`.
    pub xr: Vec<f32>,
    /// Flat logits: `[max_seq * vocab_size]`.
    pub logits_flat: Vec<f32>,
    /// Temp buffer for per-position embedding: `[n_embd]`.
    pub x_buf: Vec<f32>,
    /// Temp buffer for query: `[n_embd]`.
    pub q_buf: Vec<f32>,
    /// Temp buffer for key: `[kv_dim]`.
    pub k_buf: Vec<f32>,
    /// Temp buffer for value: `[kv_dim]`.
    pub v_buf: Vec<f32>,
    /// Temp buffer for attention projection: `[n_embd]`.
    pub x_proj_buf: Vec<f32>,
    /// Temp buffer for MLP hidden: `[mlp_hidden]`.
    pub hidden_buf: Vec<f32>,
    /// Temp buffer for MLP output: `[n_embd]`.
    pub x_mlp_buf: Vec<f32>,
    /// Temp buffer for single-position logits: `[vocab_size]`.
    pub logits_buf: Vec<f32>,
    /// Attention output buffer: `[n_head * head_dim]` (reused per position).
    pub attn_out_buf: Vec<f32>,
    /// Attention weights buffer: `[n_head * max_seq]` (reused per position).
    pub attn_weights_buf: Vec<f32>,
    /// Attention scores buffer: `[max_seq]` (reused per position).
    pub attn_scores_buf: Vec<f32>,
    /// Cached logits from previous denoising step: `[max_seq * vocab_size]`.
    /// Used by DPM-Solver++(2M) multistep extrapolation (Plan 078 T10.5).
    pub prev_logits_flat: Vec<f32>,
    /// Cached logits from two steps ago: `[max_seq * vocab_size]`.
    /// Second cache for multistep logit extrapolation (Plan 078 T10.5).
    pub prev_prev_logits_flat: Vec<f32>,
    /// Residual embeddings for RCD injection: `[max_seq * n_embd]`.
    /// Stores interpolated residual embeddings for masked positions.
    /// Written after token commitment, read during next step's input construction.
    #[cfg(feature = "rcd_residual")]
    pub residual_embeddings: Vec<f32>,
    /// Entropy weights per position: `[max_seq]`.
    /// α_i values computed from marginal distributions.
    #[cfg(feature = "rcd_residual")]
    pub entropy_weights: Vec<f32>,
    /// Softmax scratch buffer for RCD: `[vocab_size]`.
    #[cfg(feature = "rcd_residual")]
    pub rcd_softmax_scratch: Vec<f32>,
    // usize fields after all Vec<f32> fields to eliminate inter-field padding.
    /// Number of positions with committed KV cache entries.
    /// Positions `[0..committed_len)` are valid and won't be recomputed.
    pub committed_len: usize,
}

impl D2fContext {
    /// Create a new context with buffers sized for the given config.
    pub fn new(config: &Config) -> Self {
        let n = config.n_embd;
        let kvd = kv_dim(config);
        let max_seq = config.block_size;
        let vocab = config.vocab_size;
        let hidden = config.mlp_hidden;

        Self {
            k_cache: vec![0.0f32; max_seq * kvd],
            v_cache: vec![0.0f32; max_seq * kvd],
            x_norm: vec![0.0f32; max_seq * n],
            xr: vec![0.0f32; max_seq * n],
            logits_flat: vec![0.0f32; max_seq * vocab],
            x_buf: vec![0.0f32; n],
            q_buf: vec![0.0f32; n],
            k_buf: vec![0.0f32; kvd],
            v_buf: vec![0.0f32; kvd],
            x_proj_buf: vec![0.0f32; n],
            hidden_buf: vec![0.0f32; hidden],
            x_mlp_buf: vec![0.0f32; n],
            logits_buf: vec![0.0f32; vocab],
            attn_out_buf: vec![0.0f32; n],
            attn_weights_buf: vec![0.0f32; config.n_head * max_seq],
            attn_scores_buf: vec![0.0f32; max_seq],
            prev_logits_flat: vec![0.0f32; max_seq * vocab],
            prev_prev_logits_flat: vec![0.0f32; max_seq * vocab],
            #[cfg(feature = "rcd_residual")]
            residual_embeddings: vec![0.0f32; max_seq * n],
            #[cfg(feature = "rcd_residual")]
            entropy_weights: vec![0.0f32; max_seq],
            #[cfg(feature = "rcd_residual")]
            rcd_softmax_scratch: vec![0.0f32; vocab],
            committed_len: 0,
        }
    }

    /// Reset flat buffers for a new forward pass.
    ///
    /// Temp buffers (`x_buf`, `q_buf`, etc.) need not be reset since they are
    /// always fully written before being read.
    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.x_norm.fill(0.0);
        self.xr.fill(0.0);
        self.logits_flat.fill(0.0);
        self.committed_len = 0;
        self.prev_logits_flat.fill(0.0);
        self.prev_prev_logits_flat.fill(0.0);
        #[cfg(feature = "rcd_residual")]
        {
            self.residual_embeddings.fill(0.0);
            self.entropy_weights.fill(0.0);
        }
    }

    /// Commit KV cache entries for positions `[0..len)`.
    /// After calling this, subsequent forward passes will skip KV computation
    /// for these positions.
    pub fn commit(&mut self, len: usize) {
        self.committed_len = len;
    }
}

// ═══════════════════════════════════════════════════════════════
// Attention helper — shared by 5 root callers + 1 in-file caller
// ═══════════════════════════════════════════════════════════════

/// Zero-alloc fused attention head with GQA support.
///
/// Writes attention output (`[n_head * head_dim]`) into `attn_out`, attention
/// weights (`[n_head * seq_len]`) into `all_weights`, and reuses `scores`
/// (`[seq_len]`) as scratch. All buffers must be pre-sized by the caller.
///
/// `pub` (not `pub(crate)`) so root's `dllm.rs` training code can import it
/// via `use katgpt_forward::attention_forward_safe_into;` — preserves a
/// single source of truth across the 5 callers in the codebase.
pub fn attention_forward_safe_into(
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    kv_dim: usize,
    seq_len: usize,
    scale: f32,
    attn_out: &mut [f32],
    all_weights: &mut [f32],
    scores: &mut [f32],
) {
    debug_assert!(attn_out.len() >= n_head * head_dim);
    debug_assert!(all_weights.len() >= n_head * seq_len);
    debug_assert!(scores.len() >= seq_len);

    attn_out[..n_head * head_dim].fill(0.0);

    for h in 0..n_head {
        let kv_group = h * n_kv_head / n_head;
        let q_off = h * head_dim;
        let kv_off = kv_group * head_dim;

        // Compute scores (reuse buffer across heads)
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..seq_len {
            let dot = simd::simd_dot_f32(
                &q[q_off..q_off + head_dim],
                &k_all[t * kv_dim + kv_off..t * kv_dim + kv_off + head_dim],
                head_dim,
            );
            scores[t] = dot * scale;
            if scores[t] > max_score {
                max_score = scores[t];
            }
        }

        // Softmax (SIMD batch exp + sum)
        simd::simd_add_scalar_inplace(&mut scores[..seq_len], -max_score);
        simd::simd_exp_inplace(&mut scores[..seq_len]);
        let sum_exp = simd::simd_sum_f32(&scores[..seq_len]);
        let inv_sum = 1.0 / sum_exp;
        simd::simd_scale_inplace(&mut scores[..seq_len], inv_sum);
        all_weights[h * seq_len..h * seq_len + seq_len].copy_from_slice(&scores[..seq_len]);

        // Weighted value sum: accumulate per-position scaled value rows (SIMD-friendly)
        // Loop order: t outer → contiguous v_all row access, better cache locality.
        // Previous d-outer/t-inner order touched a different cache line per t for each d.
        for t in 0..seq_len {
            let s = scores[t];
            let v_row = &v_all[t * kv_dim + kv_off..t * kv_dim + kv_off + head_dim];
            simd::simd_fused_scale_acc(
                &mut attn_out[q_off..q_off + head_dim],
                v_row,
                s,
                head_dim,
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Block-causal forward (zero-alloc, writes into D2fContext)
// ═══════════════════════════════════════════════════════════════

/// Zero-alloc block-causal forward — writes logits into `ctx.logits_flat`.
///
/// Writes logits into `ctx.logits_flat[p * vocab..(p+1) * vocab]` instead of
/// returning `Vec<Vec<f32>>`. Attention weights are not computed since D2F
/// denoising only needs logits.
///
/// Returns `seq_len` (the actual number of positions processed).
pub fn forward_block_causal_with(
    ctx: &mut D2fContext,
    weights: &TransformerWeights,
    tokens: &[usize],
    config: &Config,
    causal_block_size: usize,
) -> usize {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let seq_len = tokens.len().min(config.block_size);
    let scale = 1.0 / (hd as f32).sqrt();
    let vocab = config.vocab_size;
    let layer = &weights.layers[0];

    let committed = ctx.committed_len;

    // Only clear logits for uncommitted positions
    for p in committed..seq_len {
        ctx.logits_flat[p * vocab..(p + 1) * vocab].fill(0.0);
    }

    // Phase A: Fill K/V cache, x_norm, xr for UNCOMMITTED positions only
    for (p, &token) in tokens.iter().enumerate().take(seq_len).skip(committed) {
        // Embedding = wte[token] + wpe[position]
        simd::simd_add_into(
            &mut ctx.x_buf,
            &weights.wte[token * n..(token + 1) * n],
            &weights.wpe[p * n..(p + 1) * n],
        );
        // First rmsnorm → residual (xr)
        rmsnorm(&mut ctx.x_buf);
        ctx.xr[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf);
        // Second rmsnorm → normalized embedding (x_norm)
        rmsnorm(&mut ctx.x_buf);
        ctx.x_norm[p * n..(p + 1) * n].copy_from_slice(&ctx.x_buf);

        // K, V projections
        matmul(&mut ctx.k_buf, &layer.attn_wk, &ctx.x_buf, kvd, n);
        matmul(&mut ctx.v_buf, &layer.attn_wv, &ctx.x_buf, kvd, n);
        ctx.k_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&ctx.k_buf);
        ctx.v_cache[p * kvd..(p + 1) * kvd].copy_from_slice(&ctx.v_buf);
    }

    // Phase B: Block-causal attention + MLP + logits for UNCOMMITTED positions only
    for p in committed..seq_len {
        // Load normalized embedding
        ctx.x_buf.copy_from_slice(&ctx.x_norm[p * n..(p + 1) * n]);

        // Query projection
        matmul(&mut ctx.q_buf, &layer.attn_wq, &ctx.x_buf, n, n);

        // Block-causal: attend to positions [0..end_of_current_block]
        let block_end = (p / causal_block_size + 1) * causal_block_size;
        let t_n = block_end.min(seq_len);

        // Zero-alloc attention using pre-allocated buffers in D2fContext
        attention_forward_safe_into(
            &ctx.q_buf,
            &ctx.k_cache,
            &ctx.v_cache,
            config.n_head,
            config.n_kv_head,
            hd,
            kvd,
            t_n,
            scale,
            &mut ctx.attn_out_buf,
            &mut ctx.attn_weights_buf,
            &mut ctx.attn_scores_buf,
        );

        // Attention output projection + residual connection
        matmul(&mut ctx.x_proj_buf, &layer.attn_wo, &ctx.attn_out_buf, n, n);
        simd::simd_add_inplace(&mut ctx.x_proj_buf, &ctx.xr[p * n..(p + 1) * n]);

        // Save residual before rmsnorm by reusing x_buf (no longer needed this iteration)
        ctx.x_buf.copy_from_slice(&ctx.x_proj_buf);

        // Post-attention rmsnorm
        rmsnorm(&mut ctx.x_proj_buf);

        // MLP: relu hidden → output projection + residual
        matmul_relu(
            &mut ctx.hidden_buf,
            &layer.mlp_w1,
            &ctx.x_proj_buf,
            config.mlp_hidden,
            n,
        );
        matmul(
            &mut ctx.x_mlp_buf,
            &layer.mlp_w2,
            &ctx.hidden_buf,
            n,
            config.mlp_hidden,
        );
        simd::simd_add_inplace(&mut ctx.x_mlp_buf, &ctx.x_buf[..n]);

        // Logits
        matmul(
            &mut ctx.logits_buf,
            &weights.lm_head,
            &ctx.x_mlp_buf,
            vocab,
            n,
        );
        ctx.logits_flat[p * vocab..(p + 1) * vocab].copy_from_slice(&ctx.logits_buf);
    }

    seq_len
}

// ═══════════════════════════════════════════════════════════════
// Accuracy metric
// ═══════════════════════════════════════════════════════════════

/// Measure denoising accuracy: fraction of correctly recovered tokens.
pub fn denoising_accuracy(predicted: &[usize], target: &[usize]) -> f32 {
    let len = predicted.len().min(target.len());
    if len == 0 {
        return 0.0;
    }
    let correct = (0..len).filter(|&i| predicted[i] == target[i]).count();
    correct as f32 / len as f32
}
