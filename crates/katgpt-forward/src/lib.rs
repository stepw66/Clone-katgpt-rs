//! Forward-pass composition context — Issue 007 Phase F.
//!
//! `ForwardContext` is the topmost type in the inference DAG. It composes:
//!   - transformer substrate buffers (`x`, `q`, `k`, `v`, `attn_out`, …) +
//!     `WallPrefixState` (from `katgpt-transformer`), and
//!   - pruner handles `CnaModulator` / `SubstrateMask` / `HydraSkipPlan`
//!     (from `katgpt-pruners`, gated by `cna_steering` / `substrate_gate` /
//!     `hydra_budget`).
//!
//! It cannot live in `katgpt-transformer` (would force transformer → pruners,
//! but pruners already depends on transformer → cycle) nor in `katgpt-pruners`
//! (would invert the layering — a pruner crate shouldn't own forward-pass
//! buffers). This crate sits ABOVE both, breaking the composition-layer pin so
//! the 34 composition files (`dash_attn/forward.rs`, `gdn2/forward.rs`,
//! `hla/forward.rs`, `speculative/*`, `sleep/consolidation.rs`, …) can migrate
//! out of the root crate into their respective leaves (Phase F.4a–e).
//!
//! Fields are `pub`: `ForwardContext` is a pre-allocated scratch buffer accessed
//! directly by the forward-pass functions (e.g. `forward`, `forward_looped`,
//! `forward_batched`) that remain in the root crate. The `pub(crate)` visibility
//! of the in-root era was an artifact of same-crate access; once the type crosses
//! the crate boundary the forward-pass callers need `pub` access.

use katgpt_types::{Config, DepthTier, kv_dim};
// SIMD kernels (re-exported from katgpt_types as katgpt_types::simd, and by
// katgpt_core as katgpt_core::simd). We keep the katgpt_core::simd path used by
// the original delta-routing code so the move is byte-for-byte structural.
use katgpt_transformer::TransformerWeights;
#[cfg(feature = "wall_attention")]
use katgpt_transformer::WallPrefixState;

// RiM Reasoning Buffer Slot helpers (Plan 172) — `rim_extend_tokens` /
// `rim_readout_index` live in root's transformer.rs gated on `rim_slots`. They
// are intentionally NOT moved here: they are token-sequence helpers that don't
// touch ForwardContext at all, and `rim_slots` is forwarded to katgpt-core (not
// to this crate). Keep this crate focused on ForwardContext + its delta-routing
// helper only.

/// Pre-allocated buffers for zero-alloc forward passes.
/// Create once, reuse across calls.
pub struct ForwardContext {
    // ── u64-aligned fields first (Vec, usize, arrays) ──────────────
    // Grouped by alignment to eliminate inter-field padding.
    pub x: Vec<f32>,        // [n_embd] main activation
    pub xr: Vec<f32>,       // [n_embd] residual
    pub xr2: Vec<f32>,      // [n_embd] residual 2
    pub q: Vec<f32>,        // [n_embd] query
    pub k: Vec<f32>,        // [kv_dim] key (kv_dim = n_kv_head * head_dim)
    pub v: Vec<f32>,        // [kv_dim] value
    pub attn_out: Vec<f32>, // [n_embd] attention output
    pub scores: Vec<f32>,          // [block_size] attention scores (max possible)
    pub hidden: Vec<f32>,   // [mlp_hidden] MLP hidden
    pub logits: Vec<f32>,          // [vocab_size] output logits
    pub cdf: Vec<f32>,      // [vocab_size] pre-allocated CDF for sampling
    pub hidden_state: Vec<f32>,    // [n_embd] final hidden state (Plan 009 compat)
    /// LoRA intermediate buffer [lora_rank]. Pre-allocated, zero alloc in hot path.
    pub lora_buf: Vec<f32>,
    // CNA: contrastive neuron attribution runtime modulator (Plan 087)
    #[cfg(feature = "cna_steering")]
    pub cna_modulator: Option<katgpt_pruners::CnaModulator>,
    // Sparse MLP buffers (Plan 022: TwELL-inspired unstructured sparsity)
    #[cfg(feature = "sparse_mlp")]
    pub active_indices: Vec<usize>, // [mlp_hidden] pre-allocated index buffer
    #[cfg(feature = "sparse_mlp")]
    pub active_values: Vec<f32>, // [mlp_hidden] pre-allocated value buffer
    // SubstrateGate: per-sequence capability mask for dual sparsity (Plan 216)
    #[cfg(feature = "substrate_gate")]
    pub substrate_mask: Option<katgpt_pruners::SubstrateMask>,
    // Paged KV cache: pre-allocated flat buffers for attention computation
    pub paged_flat_key: Vec<f32>,   // [block_size * kv_dim]
    pub paged_flat_value: Vec<f32>, // [block_size * kv_dim]
    // Raven: pre-allocated query buffer for per-head slot attention
    pub raven_query_buf: Vec<f32>, // [max(kv_dim, 64)]
    // MTP Drafter: pre-allocated projection buffer [n_embd] for target activation conditioning (Plan 055)
    pub mtp_context_buf: Vec<f32>,
    // Quantized KV cache incremental dequant: tracks last dequantized position per layer (Plan 068).
    // When dequant_pos[layer] == pos - 1, only dequant the new position (O(1) vs O(pos)).
    // On mismatch (layer switch, reset, pos jump), rebuild all positions for that layer.
    pub dequant_pos: Vec<usize>, // [n_layer]
    // Delta routing: block delta accumulation buffers (Plan 097)
    #[cfg(feature = "delta_routing")]
    pub block_deltas: Vec<Vec<f32>>, // [n_blocks][n_embd] accumulated deltas per block
    #[cfg(feature = "delta_routing")]
    pub delta_routing_logits: Vec<f32>, // [max_sources] routing logits temp buffer
    // CODA fused kernels: partial RMS accumulation buffer (Plan 103)
    #[cfg(feature = "coda_fusion")]
    pub coda_partial_sums: Vec<f32>, // [1] single-block RMS sum of squares
    // MLS Multi-Layer Sum aggregation (Plan 104: Research 68)
    #[cfg(feature = "mls_aggregate")]
    pub mls_buf: Vec<f32>, // [n_embd] accumulator for last K layer residuals
    // Tiled attention: pre-allocated repacking buffers for forward_prefill (Plan 115)
    // Layout: [block_size × n_embd] (Q/out) or [block_size × kv_dim] (K/V)
    // Data is repacked from (position, head) → (head, position) for tiled_attention_batched
    #[cfg(feature = "tiled_attention")]
    pub tiled_q: Vec<f32>, // [block_size × n_embd] repacked queries per head
    #[cfg(feature = "tiled_attention")]
    pub tiled_k: Vec<f32>, // [block_size × kv_dim] repacked keys per kv group
    #[cfg(feature = "tiled_attention")]
    pub tiled_v: Vec<f32>, // [block_size × kv_dim] repacked values per kv group
    #[cfg(feature = "tiled_attention")]
    pub tiled_out: Vec<f32>, // [block_size × n_embd] tiled output before transpose
    // Clustered LM head scratch buffers (avoid per-forward-pass allocation)
    pub cluster_scores_buf: Vec<f32>, // [num_clusters] cluster scores for clustered LM head
    pub topk_indexed_buf: Vec<(usize, f32)>, // [num_clusters] indexed pairs for cluster top-K
    pub topk_output_buf: Vec<usize>,  // [topk] output indices buffer
    // Loop residual: saves h^(τ-1) for residual gating across weight-shared loops
    pub prev_h: Vec<f32>, // [n_embd]
    // Delta routing: pre-allocated source_refs index buffer (stores block indices, not slices)
    #[cfg(feature = "delta_routing")]
    pub delta_source_indices: Vec<usize>, // pre-allocated capacity for max sources
    // Delta routing: scratch buffer for SIMD scaling in depth_route (Issue 082)
    #[cfg(feature = "delta_routing")]
    pub delta_scaled_buf: Vec<f32>, // [n_embd] scratch for pre-scaled dot products
    // Training-free loop: pre-allocated buffers for window iteration (Issue 091)
    #[cfg(feature = "tf_loop")]
    pub tf_x_pre_window: Vec<f32>, // [n_embd] saved state before window
    #[cfg(feature = "tf_loop")]
    pub tf_x_anchor: Vec<f32>, // [n_embd] anchor state
    #[cfg(feature = "tf_loop")]
    pub tf_y_buf: Vec<f32>, // [n_embd] temp buffer for window output
    #[cfg(feature = "tf_loop")]
    pub tf_stash_x: Vec<f32>, // [n_embd] stash for KV cache write
    // GQA lookup: kv_group_lut[h] = h * n_kv_head / n_head (pre-computed once)
    pub kv_group_lut: [u8; 128], // fixed-size LUT for GQA head→kv_group mapping (up to 128 heads)
    pub _kv_group_lut_count: usize, // actual number of heads (n_head)
    #[cfg(feature = "mls_aggregate")]
    pub mls_count: usize, // How many layers accumulated
    // Hydra Adaptive Layer Budget: pre-computed skip plan (Research 148, Plan 165)
    // None = disabled (no profiles loaded). Some(plan) = modelless skip decisions.
    #[cfg(feature = "hydra_budget")]
    pub hydra_skip_plan: Option<katgpt_pruners::HydraSkipPlan>,
    // Adaptive Depth Tier: caps layer count at inference time (Plan 284 T10).
    // None = use all layers (backward compatible). Some(tier) = cap to tier.max_layers().
    pub depth_tier: Option<DepthTier>,
    // Wall Attention: per-head prefix sum state for diagonal forget gates (Plan 173).
    // Pre-allocated, zero alloc in hot path. Updated incrementally each token.
    #[cfg(feature = "wall_attention")]
    pub wall_prefix: WallPrefixState,
    // Batched forward output buffer (Issue 020, Path B). Grown on demand by
    // [`forward_batched`] to `n_tokens * vocab_size`. Reused across calls —
    // amortises per-token logits allocation when DenseMesh batches width-many
    // hidden-node forwards into one call.
    pub batch_logits: Vec<f32>,
    // ── f32 fields last (4-byte aligned, no padding before) ──────────
    /// Pre-computed attention scale: `1.0 / sqrt(head_dim)`. Constant per config.
    pub attn_scale: f32,
}

impl ForwardContext {
    pub fn new(config: &Config) -> Self {
        let kvd = kv_dim(config);
        let block_kv = config.block_size * kvd;
        Self {
            x: vec![0.0; config.n_embd],
            xr: vec![0.0; config.n_embd],
            xr2: vec![0.0; config.n_embd],
            q: vec![0.0; config.n_embd],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; config.n_embd],
            scores: vec![0.0; config.block_size],
            hidden: vec![0.0; config.mlp_hidden],
            logits: vec![0.0; config.vocab_size],
            cdf: vec![0.0; config.vocab_size],
            hidden_state: vec![0.0; config.n_embd],
            lora_buf: vec![0.0; config.lora_rank],
            #[cfg(feature = "cna_steering")]
            cna_modulator: None,
            #[cfg(feature = "sparse_mlp")]
            active_indices: vec![0; config.mlp_hidden],
            #[cfg(feature = "sparse_mlp")]
            active_values: vec![0.0; config.mlp_hidden],
            #[cfg(feature = "substrate_gate")]
            substrate_mask: None,
            paged_flat_key: vec![0.0; block_kv],
            paged_flat_value: vec![0.0; block_kv],
            raven_query_buf: vec![0.0; kvd.max(64)],
            mtp_context_buf: vec![0.0; config.n_embd],
            dequant_pos: vec![0; config.n_layer],
            #[cfg(feature = "delta_routing")]
            block_deltas: {
                let block_size = 4; // Default B=4
                let n_blocks = config.n_layer.div_ceil(block_size);
                (0..n_blocks).map(|_| vec![0.0; config.n_embd]).collect()
            },
            #[cfg(feature = "delta_routing")]
            delta_routing_logits: vec![0.0; config.n_layer + 1], // Max B+1 sources
            #[cfg(feature = "coda_fusion")]
            coda_partial_sums: vec![0.0; 1], // Single-block partial RMS (Plan 103)
            #[cfg(feature = "mls_aggregate")]
            mls_buf: vec![0.0; config.n_embd],
            #[cfg(feature = "tiled_attention")]
            tiled_q: vec![0.0; config.block_size * config.n_embd],
            #[cfg(feature = "tiled_attention")]
            tiled_k: vec![0.0; config.block_size * kvd],
            #[cfg(feature = "tiled_attention")]
            tiled_v: vec![0.0; config.block_size * kvd],
            #[cfg(feature = "tiled_attention")]
            tiled_out: vec![0.0; config.block_size * config.n_embd],
            cluster_scores_buf: vec![
                0.0;
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))
            ],
            topk_indexed_buf: vec![
                (0usize, 0.0f32);
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1))
            ],
            topk_output_buf: Vec::with_capacity(
                config.vocab_size.div_ceil(config.mtp_cluster_size.max(1)),
            ),
            prev_h: vec![0.0; config.n_embd],
            #[cfg(feature = "delta_routing")]
            delta_source_indices: {
                let block_size = 4; // Default B=4
                let n_blocks = config.n_layer.div_ceil(block_size);
                Vec::with_capacity(n_blocks + 1)
            },
            #[cfg(feature = "delta_routing")]
            delta_scaled_buf: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_x_pre_window: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_x_anchor: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_y_buf: vec![0.0f32; config.n_embd],
            #[cfg(feature = "tf_loop")]
            tf_stash_x: vec![0.0f32; config.n_embd],
            kv_group_lut: {
                let n_head = config.n_head;
                let n_kv_head = config.n_kv_head;
                assert!(
                    n_head <= 128,
                    "n_head ({n_head}) exceeds kv_group_lut capacity (128)"
                );
                let mut lut = [0u8; 128];
                for (h, slot) in lut.iter_mut().enumerate().take(n_head) {
                    *slot = (h * n_kv_head / n_head) as u8;
                }
                lut
            },
            _kv_group_lut_count: config.n_head,
            #[cfg(feature = "mls_aggregate")]
            mls_count: 0,
            #[cfg(feature = "hydra_budget")]
            hydra_skip_plan: None,
            depth_tier: None,
            #[cfg(feature = "wall_attention")]
            wall_prefix: WallPrefixState::new(config),
            batch_logits: Vec::new(),
            attn_scale: 1.0 / (config.head_dim as f32).sqrt(),
        }
    }

    /// Reset quantized KV cache incremental dequant state.
    /// Call when starting a new sequence or after cache reset.
    pub fn reset_dequant(&mut self) {
        self.dequant_pos.fill(0);
    }

    /// Backward-compat alias for [`reset_dequant`].
    #[cfg(feature = "turboquant")]
    pub fn reset_tq_dequant(&mut self) {
        self.reset_dequant();
    }

    /// Perform delta routing using pre-allocated index buffer (avoids Vec::new() per call).
    /// Collects block delta indices, then calls depth_route_with_deltas.
    #[cfg(feature = "delta_routing")]
    pub fn depth_route_blocks(
        &mut self,
        block_idx: usize,
        layer_idx: usize,
        query_weight: &[f32],
        norm_weight: &[f32],
        n_embd: usize,
        _weights: &TransformerWeights,
    ) {
        // Collect source indices (reuse pre-allocated buffer).
        // `extend` with a bounded range is faster than push-per-element and
        // removes the per-iteration branch (`prev_block < block_deltas.len()`).
        // When block_idx is in-range, all 0..=block_idx are valid; when out-of-range,
        // we cap at block_deltas.len() to avoid index-OOB in depth_route_with_indices.
        let limit = (block_idx + 1).min(self.block_deltas.len());
        self.delta_source_indices.clear();
        self.delta_source_indices.extend(0..limit);

        // Call depth_route directly using pre-gathered indices
        depth_route_with_indices(DepthRouteIndicesArgs {
            residual: &mut self.x[..n_embd],
            block_deltas: &self.block_deltas,
            source_indices: &self.delta_source_indices,
            query_weight,
            norm_weight,
            logits_buf: &mut self.delta_routing_logits,
            scaled_buf: &mut self.delta_scaled_buf,
            n_embd,
        });

        // Reset current block delta
        self.block_deltas[block_idx].fill(0.0);
        let _ = layer_idx; // suppress unused warning
    }
}

/// Delta routing variant that takes block deltas and indices directly.
/// Avoids allocating a `Vec<&[f32]>` for source_refs by indexing into `block_deltas`.
#[cfg(feature = "delta_routing")]
pub struct DepthRouteIndicesArgs<'a> {
    pub residual: &'a mut [f32],
    pub block_deltas: &'a [Vec<f32>],
    pub source_indices: &'a [usize],
    pub query_weight: &'a [f32],   // [n_embd] per-layer query
    pub norm_weight: &'a [f32],    // [n_embd] RMSNorm gamma
    pub logits_buf: &'a mut [f32], // [N] temp buffer
    pub scaled_buf: &'a mut [f32], // [n_embd] scratch for SIMD dot
    pub n_embd: usize,
}

#[cfg(feature = "delta_routing")]
pub fn depth_route_with_indices(args: DepthRouteIndicesArgs<'_>) {
    let DepthRouteIndicesArgs {
        residual,
        block_deltas,
        source_indices,
        query_weight,
        norm_weight,
        logits_buf,
        scaled_buf,
        n_embd,
    } = args;

    let n_sources = source_indices.len();
    if n_sources == 0 {
        return;
    }

    // 1. RMSNorm each source and compute dot product with query
    let eps = 1e-5f32;
    let mut max_logit = f32::NEG_INFINITY;

    for (i, &src_idx) in source_indices.iter().enumerate() {
        let src = &block_deltas[src_idx];
        // SIMD sum-of-squares for RMSNorm
        let sum_sq = katgpt_core::simd::simd_sum_sq(&src[..n_embd], n_embd);
        let rms = (sum_sq / n_embd as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Scale src * inv_rms * norm_weight into scratch via fused SIMD, then dot with query
        scaled_buf[..n_embd].copy_from_slice(&src[..n_embd]);
        katgpt_core::simd::simd_scale_mul_inplace(
            &mut scaled_buf[..n_embd],
            &norm_weight[..n_embd],
            inv_rms,
        );
        let logit = katgpt_core::simd::simd_dot_f32(&scaled_buf[..n_embd], query_weight, n_embd);

        logits_buf[i] = logit;
        // Branch-free max reduction: f32::max compiles to a single instruction
        // (vmaxss on x86-64 SSE, fmax on AArch64 NEON). Avoids predicted-branch
        // mispredicts when logits are similar (typical for well-normalized sources).
        max_logit = max_logit.max(logit);
    }

    // 2. Softmax (numerically stable, SIMD batch)
    katgpt_core::simd::simd_add_scalar_inplace(&mut logits_buf[..n_sources], -max_logit);
    katgpt_core::simd::simd_exp_inplace(&mut logits_buf[..n_sources]);
    let sum_exp = katgpt_core::simd::simd_sum_f32(&logits_buf[..n_sources]);
    let inv_sum = 1.0 / sum_exp;

    // 3. Weighted sum of sources, added to residual (additive routing).
    //    Fused into a single SIMD pass: residual[i] += src[i] * weight.
    //    Eliminates the scaled_buf copy + separate scale + add passes.
    for (i, &src_idx) in source_indices.iter().enumerate() {
        let src = &block_deltas[src_idx];
        let weight = logits_buf[i] * inv_sum;
        katgpt_core::simd::simd_fused_scale_acc(&mut residual[..n_embd], &src[..n_embd], weight, n_embd);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// DflashCtx impl — travels with ForwardContext (Issue 007 Phase F).
//
// Previously this impl lived in root's `src/speculative/dflash.rs`, where it
// was legal because ForwardContext was local to root. Now that ForwardContext
// lives in this crate, the orphan rule requires the impl to live here too
// (DflashCtx is defined in katgpt-speculative, TransformerWeights in
// katgpt-transformer — both foreign to root, so root can no longer provide the
// impl). The impl body is byte-for-byte identical to the original; only the
// `crate::types::matmul` path became `katgpt_types::matmul`.
// ─────────────────────────────────────────────────────────────────────────
impl katgpt_speculative::dflash::DflashCtx<TransformerWeights> for ForwardContext {
    #[inline]
    fn logits_slice(&self) -> &[f32] {
        &self.logits
    }

    fn apply_mtp_conditioning(
        &mut self,
        weights: &TransformerWeights,
        mtp_ctx: &[f32],
        n_embd: usize,
        vocab_size: usize,
    ) {
        let n = n_embd.min(mtp_ctx.len());
        for i in 0..n {
            // safety: i < n <= n_embd == hidden_state.len() and i < mtp_ctx.len()
            unsafe {
                *self.hidden_state.get_unchecked_mut(i) += *mtp_ctx.get_unchecked(i);
            }
        }
        katgpt_types::matmul(
            &mut self.logits,
            &weights.lm_head,
            &self.hidden_state,
            vocab_size,
            n_embd,
        );
    }
}

// HLA forward-pass composition (Issue 007 Phase F.4b, 2026-07-02).
// Moved from root `src/hla/forward.rs`. Lives here (not in katgpt-hla) because
// katgpt-core depends on katgpt-hla for substrate re-export, and this crate
// depends on katgpt-core — placing it in katgpt-hla would create a cycle.
pub mod hla_forward;
pub use hla_forward::{forward_ahla, forward_hla, generate_ahla_into, generate_hla_into};

// Forward-pass composition layer (Plan 385, 2026-07-05).
// `forward`, `forward_base`, `forward_coda`, `attention_head`, `standard_lm_head`,
// `clustered_lm_head`, `select_topk_indices*`, `cluster_map_*` moved from root
// `src/transformer.rs`. These were the "linchpin" that blocked
// `dense_mesh/node_transformer.rs` from leaving root — now dissolved.
//
// Root re-exports `forward` (and the helpers) so every historical call site at
// `katgpt_rs::transformer::forward` continues to resolve.
pub mod forward;
pub use forward::{
    attention_head, cluster_map_from_embeddings, cluster_map_round_robin,
    clustered_lm_head, forward, forward_base, select_topk_indices,
    select_topk_indices_into_buf, standard_lm_head,
};
#[cfg(feature = "coda_fusion")]
pub use forward::forward_coda;

// DenseMesh `node_transformer` — Plan 385 (2026-07-05).
// Moved from root `src/dense_mesh/node_transformer.rs`. The substrate (traits,
// types, topology) lives in katgpt-transformer; this file lives here because it
// consumes `crate::forward::forward` (which moved here in the same plan).
// Root re-exports `TransformerNode` so `katgpt_rs::dense_mesh::TransformerNode`
// continues to resolve.
#[cfg(feature = "dense_mesh")]
pub mod dense_mesh_node_transformer;
#[cfg(feature = "dense_mesh")]
pub use dense_mesh_node_transformer::TransformerNode;
