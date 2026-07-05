//! Forward-pass composition layer — extracted from root `src/transformer.rs`
//! (Plan 385, 2026-07-05).
//!
//! ## Why this lives here
//!
//! `forward`, `forward_base`, and `forward_coda` were the cognitive-primitive
//! composer in root. They appeared unmovable because `forward` was the
//! "linchpin" of root's transformer.rs — but a line-range audit reveals that
//! the trio's only `crate::*` references are `crate::types::*`
//! (= `pub use katgpt_core::types::*`) and `crate::pruners::*`
//! (= `pub use katgpt_pruners::*`). Every other call is to `katgpt_core::simd::*`
//! or to local helpers (`attention_head`, `standard_lm_head`, `clustered_lm_head`)
//! that are themselves katgpt-core-only. The cycle was real but trivially
//! breakable: move the trio + helpers to this crate, which already depends on
//! `katgpt-core` + `katgpt-pruners` + `katgpt-transformer`.
//!
//! This dissolves the cycle that blocked `dense_mesh/node_transformer.rs`
//! from moving to a leaf — `node_transformer.rs` now lives in this crate
//! alongside `forward` (see `dense_mesh_node_transformer.rs`).
//!
//! ## What stays in root
//!
//! Other forward variants (`forward_batched`, `forward_prefill`, `forward_paged`,
//! `forward_quantized`, `forward_raven`, `forward_looped`,
//! `forward_training_free_loop`, `generate*`) have genuine root deps
//! (`crate::sleep::*`, `crate::gdn2::*`, `crate::tf_loop` shim). They stay in
//! root and call into this crate via `use katgpt_forward::{forward_base, ...}`.
//!
//! ## Visibility
//!
//! `forward_base` and `forward_coda` are `pub` here (they were `fn` private
//! in root). They must be cross-crate visible so root's remaining forward
//! variants can call them. The unsafe `attention_head` is also `pub` for the
//! same reason. This is a wider API surface than the in-root era, but the
//! functions are well-documented as internal dispatchers / SIMD helpers.

use katgpt_core::types as types;
use katgpt_core::types::{Config, matmul, matmul_parallel};
// `rmsnorm` is only called on the `#[cfg(not(feature = "kog_cpu_fusion"))]` branch
// of forward_base / forward_coda. Import it conditionally to avoid the unused
// warning under `--all-features`.
#[cfg(not(feature = "kog_cpu_fusion"))]
use katgpt_core::types::rmsnorm;
use katgpt_transformer::{MultiLayerKVCache, TransformerWeights};

use crate::ForwardContext;

// Note: `katgpt_core::simd::*`, `katgpt_pruners::{should_skip_layer,
// cna_modulate, sparse_matmul_substrate}`, `katgpt_core::types::{LoraAdapter,
// DomainLatent}` and `types::{kv_dim, lora_apply, matmul_relu, rmsnorm,
// rmsnorm_with_gamma, sparse_matmul}` are referenced via their qualified
// paths below (preserved from root's path conventions). `types` is aliased
// to `katgpt_core::types` so paths like `types::kv_dim` resolve.

/// Fused attention head with GQA support: score → softmax → weighted value sum.
/// Avoids separate `softmax()` call and write-back of normalized scores.
///
/// GQA: each Q head (`q_head_offset / hd`) maps to a KV group (`kv_group_offset / hd`).
/// When `n_kv_head == n_head`, `kv_group_offset == q_head_offset` and `kv_dim == n_embd`
/// → identical to standard MHA (backward compatible).
///
/// SAFETY: caller must ensure all indices are in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub unsafe fn attention_head(
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
) {
    // Pass 1: compute Q·K scores into buffer (no per-element scalar max)
    for t in 0..t_n {
        let k_off = t * kv_dim + kv_group_offset;
        // SAFETY: q_head_offset + hd <= n_embd (head_dim * n_head), k_off + hd <= block_size * kv_dim
        let dot = unsafe {
            let q_slice = std::slice::from_raw_parts(q.as_ptr().add(q_head_offset), hd);
            let k_slice = std::slice::from_raw_parts(key_cache.as_ptr().add(k_off), hd);
            katgpt_core::simd::simd_dot_f32(q_slice, k_slice, hd)
        };
        unsafe {
            *scores_buf.get_unchecked_mut(t) = dot;
        }
    }
    // Fuse scale + find max: one SIMD scale pass + one SIMD max reduction
    // replaces N scalar (dot * scale) and N scalar max operations
    let scores_raw = unsafe { std::slice::from_raw_parts_mut(scores_buf.as_mut_ptr(), t_n) };
    katgpt_core::simd::simd_scale_inplace(scores_raw, scale);
    let max_score =
        katgpt_core::simd::simd_max_f32(unsafe { std::slice::from_raw_parts(scores_buf.as_ptr(), t_n) });

    // Pass 2: exp(scores - max) and accumulate sum
    // Shift scores by max using SIMD broadcast add, then SIMD exp
    let scores_slice = unsafe { std::slice::from_raw_parts_mut(scores_buf.as_mut_ptr(), t_n) };
    katgpt_core::simd::simd_add_scalar_inplace(scores_slice, -max_score);
    katgpt_core::simd::simd_exp_inplace(scores_slice);
    let sum: f32 = katgpt_core::simd::simd_sum_f32(scores_slice);

    // Pass 3: normalize + weighted value accumulation (no write-back of scores)
    // Pre-scale scores once using SIMD
    let inv_sum = 1.0 / sum;
    katgpt_core::simd::simd_scale_inplace(scores_slice, inv_sum);
    // Zero the output slice before accumulation
    attn_out[q_head_offset..q_head_offset + hd].fill(0.0);
    // Accumulate: t outer → contiguous value_cache row access
    for t in 0..t_n {
        let s = unsafe { *scores_buf.get_unchecked(t) };
        let v_row = unsafe {
            std::slice::from_raw_parts(value_cache.as_ptr().add(t * kv_dim + kv_group_offset), hd)
        };
        let out_slice =
            unsafe { std::slice::from_raw_parts_mut(attn_out.as_mut_ptr().add(q_head_offset), hd) };
        katgpt_core::simd::simd_fused_scale_acc(out_slice, v_row, s, hd);
    }
}

/// Causal decode: single token forward with optional LoRA adapter.
/// Backward-compatible wrapper that passes `None` for LoRA.
///
/// Wall Attention (Plan 173): when wall_config is Some and feature is enabled,
/// Wall gate projection + Q/K rescaling replaces RoPE rotation.
/// Attention kernels unchanged — they receive pre-rescaled Q/K.
#[inline(always)]
pub fn forward<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    cache.advance_pos(pos);
    #[cfg(feature = "coda_fusion")]
    {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_coda(ctx, weights, cache, token, pos, config, None)
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_coda(ctx, weights, cache, token, pos, config, None, None)
        }
    }
    #[cfg(not(feature = "coda_fusion"))]
    {
        #[cfg(not(feature = "domain_latent"))]
        {
            forward_base(ctx, weights, cache, token, pos, config, None)
        }
        #[cfg(feature = "domain_latent")]
        {
            forward_base(ctx, weights, cache, token, pos, config, None, None)
        }
    }
}
pub fn standard_lm_head(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    vocab_size: usize,
    n_embd: usize,
) {
    // matmul_parallel has an internal threshold (512 rows) — for small vocab
    // it falls back to serial automatically. For vocab_size >= 512 (e.g.
    // small_target vocab=4096), this parallelizes across rayon threads.
    matmul_parallel(logits, lm_head, hidden, vocab_size, n_embd);
}

/// Select top-K indices from scores (Plan 117 T25).
///
/// Uses partial selection sort: O(N + K log K).
/// Returns indices sorted by score descending (highest first).
///
/// **Note:** This function allocates internally and is intended for tests/benchmarks only.
/// For hot-path code, use [`select_topk_indices_into_buf`] which reuses pre-allocated buffers.
pub fn select_topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }

    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();

    // Partial sort to partition top K (unstable, O(N))
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort the top K by score descending (O(K log K))
    indexed[..k].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed[..k].iter().map(|(i, _)| *i).collect()
}

/// In-place variant of [`select_topk_indices`] that reuses pre-allocated buffers.
/// Writes top-K indices into `output_buf` (cleared and filled).
///
/// This is the preferred variant for hot-path code — no heap allocations.
pub fn select_topk_indices_into_buf(
    scores: &[f32],
    k: usize,
    indexed_buf: &mut Vec<(usize, f32)>,
    output_buf: &mut Vec<usize>,
) {
    let k = k.min(scores.len());
    if k == 0 {
        output_buf.clear();
        return;
    }

    // In-place indexed writes via resize + direct assignment: avoids `extend`'s
    // potential reallocation jitter and push-per-element overhead. Index
    // assignment is also more amenable to LLVM auto-vectorization. This runs
    // every decode step when clustered vocab is active (clustered_lm_head hot path).
    indexed_buf.resize(scores.len(), (0, 0.0));
    for (i, &s) in scores.iter().enumerate() {
        indexed_buf[i] = (i, s);
    }

    // total_cmp replaces partial_cmp().unwrap_or(Equal): eliminates the
    // per-element NaN branch (compiled to a predicted branch on x86-64),
    // giving LLVM a single instruction compare. Cluster scores are
    // simd_dot products, which never produce NaN for finite weights.
    indexed_buf.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    indexed_buf[..k].sort_by(|a, b| b.1.total_cmp(&a.1));

    // Direct index writes replace clear+extend: writes are contiguous and
    // skip Vec::push's length/capacity bookkeeping per element.
    output_buf.clear();
    output_buf.resize(k, 0);
    for (dst, (src, _)) in output_buf.iter_mut().zip(indexed_buf[..k].iter()) {
        *dst = *src;
    }
}

/// Two-stage clustered LM head for large vocabularies.
///
/// Stage 1: predict cluster ID(s) via classifier matmul + top-K selection.
/// Stage 2: compute exact logits only for tokens in the selected clusters.
///
/// When `topk=1`, behavior is identical to single-cluster argmax (backward compat).
/// When `topk >= num_clusters`, all clusters are selected (no pruning).
///
/// Only called when `vocab_size >= mtp_cluster_vocab_threshold` AND
/// cluster weights are available.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn clustered_lm_head(
    logits: &mut [f32],
    hidden: &[f32],
    lm_head: &[f32],
    cluster_classifier: &[f32],
    cluster_map: &[Vec<usize>],
    vocab_size: usize,
    n_embd: usize,
    topk: usize,
    cluster_scores_buf: &mut [f32],
    topk_indexed_buf: &mut Vec<(usize, f32)>,
    topk_output_buf: &mut Vec<usize>,
) {
    let num_clusters = cluster_map.len();

    // Stage 1: compute cluster scores (reuse pre-allocated buffer)
    let cluster_scores = &mut cluster_scores_buf[..num_clusters];
    for (c, score) in cluster_scores.iter_mut().enumerate() {
        let row_off = c * n_embd;
        *score = katgpt_core::simd::simd_dot_f32(
            &cluster_classifier[row_off..row_off + n_embd],
            &hidden[..n_embd],
            n_embd,
        );
    }

    // Select top-K clusters (Plan 117 T27: skip selection if topk >= num_clusters)
    let selected_clusters: &[usize] = if topk >= num_clusters {
        // Fill output_buf with all cluster indices
        topk_output_buf.clear();
        topk_output_buf.extend(0..num_clusters);
        topk_output_buf
    } else {
        select_topk_indices_into_buf(cluster_scores, topk, topk_indexed_buf, topk_output_buf);
        topk_output_buf
    };

    // Stage 2: fill all logits with -inf, then compute exact for selected clusters
    logits.fill(f32::NEG_INFINITY);

    // NOTE(078): Cluster tokens are non-contiguous (round-robin assignment), so
    // batched simd_matmul_rows cannot be used directly. Individual simd_dot_f32 calls
    // are optimal here — the function is inlined and dispatch overhead is negligible.
    for &cluster_idx in selected_clusters {
        let cluster_tokens = &cluster_map[cluster_idx];
        for &token_idx in cluster_tokens {
            if token_idx < vocab_size {
                let row_off = token_idx * n_embd;
                let dot = katgpt_core::simd::simd_dot_f32(
                    &lm_head[row_off..row_off + n_embd],
                    &hidden[..n_embd],
                    n_embd,
                );
                unsafe {
                    *logits.get_unchecked_mut(token_idx) = dot;
                }
            }
        }
    }
}

/// Create a round-robin cluster assignment for tokens.
///
/// Token `i` is assigned to cluster `i / cluster_size`.
/// Deterministic, no training needed — simple baseline.
pub fn cluster_map_round_robin(vocab_size: usize, cluster_size: usize) -> Vec<Vec<usize>> {
    let num_clusters = vocab_size.div_ceil(cluster_size);
    let mut map: Vec<Vec<usize>> = (0..num_clusters)
        .map(|_| Vec::with_capacity(cluster_size))
        .collect();
    for token_id in 0..vocab_size {
        let cluster_id = token_id / cluster_size;
        map[cluster_id].push(token_id);
    }
    map
}

/// Create cluster assignment from embedding similarity (K-means style).
///
/// Groups tokens with similar embeddings together for efficient LM head computation.
/// Current implementation: round-robin baseline.
/// TODO: implement actual K-means using embedding cosine similarity (Plan 056: riir-burner).
pub fn cluster_map_from_embeddings(
    _wte: &[f32],
    vocab_size: usize,
    _n_embd: usize,
    cluster_size: usize,
) -> Vec<Vec<usize>> {
    cluster_map_round_robin(vocab_size, cluster_size)
}
pub fn forward_base<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&katgpt_core::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&katgpt_core::types::DomainLatent>,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Wall Attention: reset prefix sums at sequence start (Plan 173).
    // Prefix sums accumulate per-layer, per-head across the sequence.
    // Must reset when pos=0 to avoid stale state from previous sequences.
    #[cfg(feature = "wall_attention")]
    if pos == 0 {
        ctx.wall_prefix.reset();
    }

    // MLS: reset accumulator at start of forward call (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    {
        ctx.mls_buf[..n].fill(0.0);
        ctx.mls_count = 0;
    }

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

    // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
    // Composes with Hydra: tier sets upper bound, Hydra skips within that bound.
    let max_layer = ctx
        .depth_tier
        .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
        let layer_cache = &mut cache.layers[layer_idx];

        // Hydra Adaptive Layer Budget: skip non-contributing layers (Research 148, Plan 165)
        // Modelless mode — zero overhead (single bool check on pre-computed plan).
        // When skipped, x passes through unchanged (z^l = z^{l-1}).
        #[cfg(feature = "hydra_budget")]
        if let Some(ref skip_plan) = ctx.hydra_skip_plan
            && katgpt_pruners::should_skip_layer(skip_plan, layer_idx)
        {
            // Skip this layer entirely — x passes through as-is.
            // Still need to copy x → hidden_state for snapshot consistency.
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
            continue;
        }

        // MLS: save pre-layer state for delta computation (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
        }

        // Pre-attention: RMSNorm → save residual
        #[cfg(feature = "kog_cpu_fusion")]
        types::rmsnorm_with_gamma(&mut ctx.x[..n], &layer_weights.attn_norm_gamma);
        #[cfg(not(feature = "kog_cpu_fusion"))]
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections from per-layer weights (GQA: K/V produce kv_dim outputs)
        #[cfg(feature = "kog_cpu_fusion")]
        if let Some(ref qkv_fused) = layer_weights.attn_qkv_fused {
            matmul(&mut ctx.q, &qkv_fused[..n * n], &ctx.x, n, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.k, &qkv_fused[n * n..(n + kvd) * n], &ctx.x, kvd, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.v, &qkv_fused[(n + kvd) * n..], &ctx.x, kvd, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
            }
        } else {
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
            }
        }
        #[cfg(not(feature = "kog_cpu_fusion"))]
        {
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
            }
            matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
            if let Some(lora) = lora {
                katgpt_core::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
            }
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            katgpt_core::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
            katgpt_core::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
        }

        // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
        // Replaces RoPE rotation when wall_config is Some. Factorized form means
        // attention kernels are unchanged — they receive pre-rescaled Q and K.
        // Gate is derived from K (key-projected variant, zero KV cache overhead).
        #[cfg(feature = "wall_attention")]
        if let Some(ref wall_cfg) = config.wall_config {
            let n_kv = config.n_kv_head;
            let hd = config.head_dim;
            for kv_h in 0..n_kv {
                let k_off = kv_h * hd;
                let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                let k_slice = &ctx.k[k_off..k_off + hd];
                // Compute gate from key, then update prefix sum in one call.
                ctx.wall_prefix.compute_gate_and_update(
                    layer_idx,
                    kv_h,
                    k_slice,
                    w_g,
                    wall_cfg.gate_bias,
                    wall_cfg.gate_max,
                );
            }
            ctx.wall_prefix
                .rescale_query(layer_idx, &mut ctx.q, &ctx.kv_group_lut, config.n_head);
            ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
        }

        // Store K,V in per-layer cache (kv_dim elements per position)
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // Multi-head attention with GQA: fused score → softmax → weighted value per head
        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h] as usize;
            unsafe {
                attention_head(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // Output projection + residual
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, n);
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.x, lora, &ctx.attn_out, &mut ctx.lora_buf);
        }
        katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr[..n]);

        // MLP: save residual → RMSNorm → MLP → residual
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);
        #[cfg(feature = "kog_cpu_fusion")]
        types::rmsnorm_with_gamma(&mut ctx.x[..n], &layer_weights.mlp_norm_gamma);
        #[cfg(not(feature = "kog_cpu_fusion"))]
        rmsnorm(&mut ctx.x);
        types::matmul_relu(
            &mut ctx.hidden,
            &layer_weights.mlp_w1,
            &ctx.x,
            config.mlp_hidden,
            n,
        );
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.hidden, lora, &ctx.x, &mut ctx.lora_buf);
        }
        // CNA: modulate discovered circuit neurons (Plan 087)
        #[cfg(feature = "cna_steering")]
        if let Some(ref modulator) = ctx.cna_modulator {
            katgpt_pruners::cna_modulate(&mut ctx.hidden, layer_idx, modulator);
        }
        // MLP w2: substrate dual-sparsity path (Plan 216)
        #[cfg(all(feature = "sparse_mlp", feature = "substrate_gate"))]
        {
            let alive = if let Some(ref substrate_mask) = ctx.substrate_mask {
                katgpt_pruners::sparse_matmul_substrate(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                    substrate_mask,
                    layer_idx,
                )
            } else {
                types::sparse_matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                    &mut ctx.active_indices,
                    &mut ctx.active_values,
                )
            };
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        // MLP w2: sparse-only path (no substrate)
        #[cfg(all(feature = "sparse_mlp", not(feature = "substrate_gate")))]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                matmul(
                    &mut ctx.x,
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    n,
                    config.mlp_hidden,
                );
            }
        }
        #[cfg(not(feature = "sparse_mlp"))]
        matmul(
            &mut ctx.x,
            &layer_weights.mlp_w2,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.x, lora, &ctx.hidden, &mut ctx.lora_buf);
        }
        katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);

        // MLS: accumulate layer delta (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        {
            if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
                katgpt_core::simd::simd_fused_sub_acc(
                    &mut ctx.mls_buf[..n],
                    &ctx.x[..n],
                    &ctx.hidden_state[..n],
                    n,
                );
                ctx.mls_count += 1;
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual (xr was saved after first rmsnorm)
            // Delta captures full layer contribution: attention + MLP residuals
            if block_idx < ctx.block_deltas.len() {
                katgpt_core::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // MLS: blend averaged layer deltas into final hidden state (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    if ctx.mls_count > 0 {
        let scale = 1.0 / ctx.mls_count as f32;
        katgpt_core::simd::simd_fused_decay_write(&mut ctx.x[..n], 1.0, &ctx.mls_buf[..n], scale);
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: clustered when vocab >= threshold AND cluster weights present
    if config.vocab_size >= config.mtp_cluster_vocab_threshold
        && let Some(classifier) = weights.mtp_cluster_classifier.as_ref()
        && let Some(cluster_map) = weights.mtp_cluster_map.as_ref()
    {
        clustered_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            classifier,
            cluster_map,
            config.vocab_size,
            n,
            config.mtp_cluster_topk,
            &mut ctx.cluster_scores_buf,
            &mut ctx.topk_indexed_buf,
            &mut ctx.topk_output_buf,
        );
    } else {
        standard_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            config.vocab_size,
            n,
        );
    }

    &mut ctx.logits
}
/// CODA-inspired fused forward pass (Research 67, Plan 103).
///
/// Algebraic reparameterization: fuse matmul+residual+rmsnorm+activation
/// into single-pass SIMD loops, eliminating intermediate buffer writes.
///
/// Key identity (CODA §3.2.1):
/// ```text
/// RMSNorm(x@W + z) * gamma @ W' = r * ((x@W + z) * gamma) @ W'
/// ```
///
/// This lets us delay the row-wise RMSNorm scale past the next GEMM,
/// fusing 3 separate operations into one SIMD loop per kernel.
///
/// # Buffer Write Savings (per layer)
///
/// Eliminated: out_proj write, residual add, xr2 copy, rmsnorm (pre-MLP),
/// gate_up write, activation pass, down_proj write, residual add = ~6 passes
///
/// Retained: 2× rmsnorm (pre-QKV), 1× xr copy = ~3 passes
///
/// # Feature Gate
///
/// Only compiled when `coda_fusion` feature is enabled. Falls back to
/// [`forward_base`] when LoRA is active (T10: future fused LoRA support).
#[cfg(feature = "coda_fusion")]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn forward_coda<'a>(
    ctx: &'a mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    lora: Option<&katgpt_core::types::LoraAdapter>,
    #[cfg(feature = "domain_latent")] domain_latent: Option<&katgpt_core::types::DomainLatent>,
) -> &'a mut [f32] {
    // NOTE(080): LoRA passthrough through CODA fused kernels.
    // LoRA is additive (scale * B @ (A @ input)), so it can't be fused into CODA's
    // bias parameter (which is a pre-computed vector). Instead, we compute LoRA
    // perturbations separately and add them to CODA kernel outputs, matching the
    // same projection points as forward_base: after QKV, after wo, after w1, after w2.

    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = types::kv_dim(config);
    let _n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off_emb = pos * n;
    katgpt_core::simd::simd_add_into(
        &mut ctx.x[..n],
        &weights.wte[tok_off..tok_off + n],
        &weights.wpe[pos_off_emb..pos_off_emb + n],
    );

    // Wall Attention: reset prefix sums at sequence start (Plan 173).
    #[cfg(feature = "wall_attention")]
    if pos == 0 {
        ctx.wall_prefix.reset();
    }

    // MLS: reset accumulator at start of forward call (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    {
        ctx.mls_buf[..n].fill(0.0);
        ctx.mls_count = 0;
    }

    // Loop-invariant values hoisted outside the layer loop
    let scale = ctx.attn_scale;
    let t_n = pos + 1;

    // Adaptive Depth Tier: cap layer count at inference time (Plan 284 T10).
    let max_layer = ctx
        .depth_tier
        .map_or(config.n_layer, |t| t.max_layers(config.n_layer));

    // 2. Layer loop with CODA-fused kernels
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate().take(max_layer) {
        let layer_cache = &mut cache.layers[layer_idx];

        // MLS: save pre-layer state for delta computation (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
            ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);
        }

        // Pre-attention: RMSNorm → save residual
        // Note: CODA fused kernels handle delayed RMS internally, no second rmsnorm needed
        #[cfg(feature = "kog_cpu_fusion")]
        types::rmsnorm_with_gamma(&mut ctx.x[..n], &layer_weights.attn_norm_gamma);
        #[cfg(not(feature = "kog_cpu_fusion"))]
        rmsnorm(&mut ctx.x);
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // QKV projections (same as baseline — attention needs separate Q, K, V)
        #[cfg(feature = "kog_cpu_fusion")]
        if let Some(ref qkv_fused) = layer_weights.attn_qkv_fused {
            matmul(&mut ctx.q, &qkv_fused[..n * n], &ctx.x, n, n);
            matmul(&mut ctx.k, &qkv_fused[n * n..(n + kvd) * n], &ctx.x, kvd, n);
            matmul(&mut ctx.v, &qkv_fused[(n + kvd) * n..], &ctx.x, kvd, n);
        } else {
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        }
        #[cfg(not(feature = "kog_cpu_fusion"))]
        {
            matmul(&mut ctx.q, &layer_weights.attn_wq, &ctx.x, n, n);
            matmul(&mut ctx.k, &layer_weights.attn_wk, &ctx.x, kvd, n);
            matmul(&mut ctx.v, &layer_weights.attn_wv, &ctx.x, kvd, n);
        }

        // LoRA perturbation for QKV projections (same as forward_base)
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.q, lora, &ctx.x, &mut ctx.lora_buf);
            katgpt_core::types::lora_apply(&mut ctx.k, lora, &ctx.x, &mut ctx.lora_buf);
            katgpt_core::types::lora_apply(&mut ctx.v, lora, &ctx.x, &mut ctx.lora_buf);
        }

        // Domain latent injection at mid-layer (Plan 038: Free Transformer adaptation)
        #[cfg(feature = "domain_latent")]
        if layer_idx == config.n_layer / 2
            && let Some(dl) = domain_latent
        {
            katgpt_core::simd::simd_add_inplace(&mut ctx.k[..kvd], &dl.embedding[..kvd]);
            katgpt_core::simd::simd_add_inplace(&mut ctx.v[..kvd], &dl.embedding[..kvd]);
        }

        // Wall Attention: gate projection + prefix sum update + Q/K rescale (Plan 173).
        // Same integration as forward_base — Wall is path-agnostic.
        #[cfg(feature = "wall_attention")]
        if let Some(ref wall_cfg) = config.wall_config {
            let n_kv = config.n_kv_head;
            let hd = config.head_dim;
            for kv_h in 0..n_kv {
                let k_off = kv_h * hd;
                let w_g = &layer_weights.attn_wg[k_off..k_off + hd];
                let k_slice = &ctx.k[k_off..k_off + hd];
                ctx.wall_prefix.compute_gate_and_update(
                    layer_idx,
                    kv_h,
                    k_slice,
                    w_g,
                    wall_cfg.gate_bias,
                    wall_cfg.gate_max,
                );
            }
            ctx.wall_prefix
                .rescale_query(layer_idx, &mut ctx.q, &ctx.kv_group_lut, config.n_head);
            ctx.wall_prefix.rescale_key(layer_idx, &mut ctx.k);
        }

        // Store K,V in per-layer cache (kv_dim elements per position)
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // Multi-head attention with GQA: fused score → softmax → weighted value per head
        for h in 0..config.n_head {
            let kv_group = ctx.kv_group_lut[h] as usize;
            unsafe {
                attention_head(
                    &ctx.q,
                    &layer_cache.key,
                    &layer_cache.value,
                    &mut ctx.attn_out,
                    &mut ctx.scores,
                    h * hd,
                    kv_group * hd,
                    kvd,
                    hd,
                    t_n,
                    scale,
                );
            }
        }

        // ── CODA FUSED KERNEL 1: out_proj + residual + partial_rms ──────
        // Replaces: matmul(x, wo, ao) + add(x, xr) + copy(xr2, x) + rmsnorm(x)
        //
        // D = wo @ attn_out + xr  → stored in ctx.xr2 (residual for down_proj)
        // O = D * gamma          → stored in ctx.x (input to MLP, gamma=identity)
        // partial_sums = Σ D[i]²  → for rstd computation
        katgpt_core::coda::simd_matmul_residual_partial_rms(
            &mut ctx.xr2[..n],          // output_d: D = matmul + residual
            &mut ctx.x[..n],            // output_o: O = D * gamma (gamma=identity)
            &mut ctx.coda_partial_sums, // partial RMS accumulation
            &layer_weights.attn_wo,     // weight
            &ctx.attn_out[..n],         // input
            &ctx.xr[..n],               // residual
            None,                       // gamma (None = identity for standard rmsnorm)
            None,                       // bias (no LoRA in fused path)
            n,                          // rows
            n,                          // cols
            n,                          // block_size (single block)
        );

        // LoRA perturbation for output projection: add to ctx.x (CODA output)
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.x[..n], lora, &ctx.attn_out[..n], &mut ctx.lora_buf);
        }

        // ── CODA AUXILIARY REDUCTION: compute rstd ─────────────────────
        // rstd = 1 / sqrt(mean(D²) + eps) — tiny reduction, O(1) for single block
        let rstd = katgpt_core::coda::compute_rstd(&ctx.coda_partial_sums, n, 1e-5);

        // ── CODA FUSED KERNEL 2: MLP matmul + delayed RMS + activation ─
        // Replaces: rmsnorm(x) + matmul_relu(hidden, w1, x)
        // hidden[i] = activation(dot(w1[i], O) * rstd)  — delayed RMS scale
        katgpt_core::coda::simd_matmul_rmsnorm_activation(
            &mut ctx.hidden,                         // output
            &layer_weights.mlp_w1,                   // weight
            &ctx.x[..n],                             // input (O from kernel 1)
            rstd,                                    // delayed RMS scale
            katgpt_core::coda::GateActivation::Relu, // matches baseline matmul_relu
            config.mlp_hidden,                       // rows
            n,                                       // cols
        );

        // LoRA perturbation for MLP up projection: add to hidden
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.hidden, lora, &ctx.x[..n], &mut ctx.lora_buf);
        }

        // CNA: modulate discovered circuit neurons (Plan 087)
        #[cfg(feature = "cna_steering")]
        if let Some(ref modulator) = ctx.cna_modulator {
            katgpt_pruners::cna_modulate(&mut ctx.hidden, layer_idx, modulator);
        }

        // ── CODA FUSED KERNEL 3: down_proj + residual ─────────────────
        // Replaces: matmul(x, w2, hidden) + add(x, xr2)
        // x[i] = dot(w2[i], hidden) + xr2[i]
        #[cfg(not(feature = "sparse_mlp"))]
        katgpt_core::coda::simd_matmul_residual(
            &mut ctx.x[..n],       // output
            &layer_weights.mlp_w2, // weight
            &ctx.hidden,           // input
            &ctx.xr2[..n],         // residual (D from kernel 1)
            n,                     // rows
            config.mlp_hidden,     // cols
        );

        // Sparse MLP: try sparse first, fall back to fused dense + residual
        #[cfg(feature = "sparse_mlp")]
        {
            let alive = types::sparse_matmul(
                &mut ctx.x,
                &layer_weights.mlp_w2,
                &ctx.hidden,
                n,
                config.mlp_hidden,
                &mut ctx.active_indices,
                &mut ctx.active_values,
            );
            if (alive as f32 / config.mlp_hidden as f32) > (1.0 - config.sparse_threshold) {
                // Too dense for sparse, use fused dense + residual
                katgpt_core::coda::simd_matmul_residual(
                    &mut ctx.x[..n],
                    &layer_weights.mlp_w2,
                    &ctx.hidden,
                    &ctx.xr2[..n],
                    n,
                    config.mlp_hidden,
                );
            } else {
                // Sparse succeeded, add residual manually
                katgpt_core::simd::simd_add_inplace(&mut ctx.x[..n], &ctx.xr2[..n]);
            }
        }

        // LoRA perturbation for MLP down projection (applies to both sparse and dense paths)
        if let Some(lora) = lora {
            katgpt_core::types::lora_apply(&mut ctx.x[..n], lora, &ctx.hidden, &mut ctx.lora_buf);
        }

        // MLS: accumulate layer delta (Plan 104)
        #[cfg(feature = "mls_aggregate")]
        {
            if config.mls_layers > 0 && layer_idx >= weights.layers.len() - config.mls_layers {
                katgpt_core::simd::simd_fused_sub_acc(
                    &mut ctx.mls_buf[..n],
                    &ctx.x[..n],
                    &ctx.hidden_state[..n],
                    n,
                );
                ctx.mls_count += 1;
            }
        }

        // Delta routing: accumulate per-sublayer deltas, route at block boundaries (Plan 097)
        #[cfg(feature = "delta_routing")]
        {
            let block_size = 4; // Default B=4
            let block_idx = layer_idx / block_size;
            let pos_in_block = layer_idx % block_size;

            // Compute delta: current x minus pre-layer residual
            if block_idx < ctx.block_deltas.len() {
                katgpt_core::simd::simd_fused_sub_acc(
                    &mut ctx.block_deltas[block_idx][..n],
                    &ctx.x[..n],
                    &ctx.xr[..n],
                    n,
                );
            }

            // At block boundary: route accumulated deltas from all completed blocks
            if pos_in_block == block_size - 1 && block_idx < ctx.block_deltas.len() {
                ctx.depth_route_blocks(
                    block_idx,
                    layer_idx,
                    &weights.delta_routing_query[layer_idx],
                    &weights.delta_routing_norm[layer_idx],
                    n,
                    weights,
                );
            }
        }
    }

    // MLS: blend averaged layer deltas into final hidden state (Plan 104)
    #[cfg(feature = "mls_aggregate")]
    if ctx.mls_count > 0 {
        let scale = 1.0 / ctx.mls_count as f32;
        katgpt_core::simd::simd_fused_decay_write(&mut ctx.x[..n], 1.0, &ctx.mls_buf[..n], scale);
    }

    // Snapshot hidden state (for Plan 009 compatibility)
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // LM Head: clustered when vocab >= threshold AND cluster weights present
    if config.vocab_size >= config.mtp_cluster_vocab_threshold
        && let Some(classifier) = weights.mtp_cluster_classifier.as_ref()
        && let Some(cluster_map) = weights.mtp_cluster_map.as_ref()
    {
        clustered_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            classifier,
            cluster_map,
            config.vocab_size,
            n,
            config.mtp_cluster_topk,
            &mut ctx.cluster_scores_buf,
            &mut ctx.topk_indexed_buf,
            &mut ctx.topk_output_buf,
        );
    } else {
        standard_lm_head(
            &mut ctx.logits,
            &ctx.x,
            &weights.lm_head,
            config.vocab_size,
            n,
        );
    }

    &mut ctx.logits
}
