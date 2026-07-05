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

// Speculative-decoding composition context (Plan 393, 2026-07-05).
// `SpeculativeContext` moved from root `src/speculative/types.rs`. Lives here
// (not in katgpt-speculative) because it composes `ForwardContext` and
// katgpt-forward already depends on katgpt-speculative — placing it in the
// leaf would create a cycle. Root re-exports via
// `pub use katgpt_forward::SpeculativeContext;` so all historical
// `crate::speculative::types::SpeculativeContext` import paths resolve.
pub mod speculative_context;
pub use speculative_context::SpeculativeContext;

// Drafter LoRA training (Plan 117 Phase 1, Plan 394 2026-07-05).
// Moved from root `src/speculative/drafter_lora.rs`. Lives here because the
// training loop calls `forward()` (moved here in Plan 385) and uses
// `ForwardContext` (defined here). Root re-exports via
// `pub use katgpt_forward::drafter_lora;` so all historical
// `katgpt_rs::speculative::drafter_lora::*` paths resolve.
pub mod drafter_lora;
pub use drafter_lora::{
    DrafterForwardContext, DrafterLoraWeights, TrainingPair, generate_synthetic_pairs,
    generate_training_pairs_from_replays, load_drafter_lora, save_drafter_lora, train_drafter_lora,
};

// DFlare draft-prediction composition layer (Plan 394, 2026-07-05).
// Moved from root `src/speculative/dflash.rs`. The shared-core `_with`
// delegations live in `katgpt_speculative::dflash`; this file holds the
// katgpt-forward-specific `_with` entry points + the back-compat thin
// wrappers. Root re-exports via `pub use katgpt_forward::dflash;` so all
// historical `katgpt_rs::speculative::dflash::*` paths resolve.
pub mod dflash;
pub use dflash::{
    dflash_predict, dflash_predict_ar, dflash_predict_ar_with, dflash_predict_conditioned,
    dflash_predict_conditioned_with, dflash_predict_parallel, dflash_predict_with,
};
#[cfg(feature = "dflare_fusion")]
pub use dflash::{dflash_predict_ar_with_fusion, marginal_fusion_blend};
#[cfg(feature = "domino_lora")]
pub use dflash::dflash_predict_ar_with_domino;
#[cfg(feature = "dflare_kv_routing")]
pub use dflash::dflash_predict_conditioned_with_routing;

// Speculative verifier composition layer (Plan 394, 2026-07-05).
// Moved from root `src/speculative/verifier.rs`. Concrete verifier impls
// (`SimulatedVerifier`, `LeviathanVerifier`) live here because they compose
// the moved dflash + drafter_lora siblings + forward(). The
// `SpeculativeVerifier` trait lives in katgpt-speculative; this module
// re-exports it. Root re-exports via `pub use katgpt_forward::verifier;` so
// all historical `katgpt_rs::speculative::verifier::*` paths resolve.
pub mod verifier;
pub use verifier::{LeviathanVerifier, SimulatedVerifier, SpeculativeVerifier};

// Speculative step pipeline composition layer (Plan 394, 2026-07-05).
// Moved from root `src/speculative/step.rs`. Lives here because the pipeline
// composes the moved verifier + dflash siblings + forward(). The deprecated
// paged-KV variant (`speculative_step_rollback_paged`) stays root in
// `src/speculative/step_paged.rs` because `DDTreeBranchCache` consumes
// `forward_paged`, which has genuine root deps. Root re-exports via
// `pub use katgpt_forward::step;` so all historical
// `katgpt_rs::speculative::step::*` paths resolve. The internal
// `extract_ddtree_paths` helper is exported `pub(crate)` so the root-side
// `step_paged.rs` can call it via `katgpt_forward::step::extract_ddtree_paths`.
pub mod step;
pub use step::{speculative_step, speculative_step_verifier};
#[allow(deprecated)]
pub use step::{
    speculative_step_conditioned, speculative_step_conditioned_with,
    speculative_step_rollback, speculative_step_rollback_with,
};
#[cfg(feature = "selectivity_router")]
pub use step::{speculative_step_conditioned_with_router, speculative_step_rollback_with_router};
#[cfg(feature = "sr2am_configurator")]
pub use step::speculative_step_with_configurator;

// Speculative prefill scorers (Plan 394, 2026-07-05).
// Moved from root `src/speculative/prefill.rs`. Hosts the forward-coupled
// scorers (`AttentionScorer`, `BlockAttentionScorer`) + the substrate re-export
// shim from `katgpt_speculative::prefill`. The `block_select_entmax` function
// stays root (consumes `katgpt-attn`, which depends on katgpt-forward and
// would create a cycle). Root re-exports via `pub use katgpt_forward::prefill;`
// (chained through root's slim `prefill.rs`) so all historical
// `katgpt_rs::speculative::prefill::*` paths resolve.
pub mod prefill;
pub use prefill::{AttentionScorer, BlockAttentionScorer};

// Decision-Diffusion Tree feature-gated wrappers + integration tests
// (Plan 396, 2026-07-05). Moved from root `src/speculative/dd_tree.rs`.
// Hosts the two feature-gated production fns (`build_dd_tree_screened_with_schedule`,
// `build_dd_tree_gdsd`) plus the ~2380-LOC integration test module that
// exercises the full dd_tree + dflash_predict pipeline. The core dd-tree
// algorithm lives in `katgpt_speculative::dd_tree` (re-exported via glob inside
// this module). Root re-exports via `pub use katgpt_forward::dd_tree;` so all
// historical `katgpt_rs::speculative::dd_tree::*` paths resolve.
pub mod dd_tree;
#[cfg(feature = "gdsd_distill")]
pub use dd_tree::build_dd_tree_gdsd;
#[cfg(feature = "thinking_prune")]
pub use dd_tree::build_dd_tree_screened_with_schedule;

// D2F inference substrate (Plan 398, 2026-07-05). Extracted from root
// `src/dllm.rs` to dissolve the d2f cluster blocker. Hosts `D2fContext`,
// `forward_block_causal_with`, `attention_forward_safe_into`, and
// `denoising_accuracy`. Training code stays in root. Root re-exports via
// `pub use katgpt_forward::d2f_context::{...}` so all historical
// `katgpt_rs::dllm::{D2fContext, forward_block_causal_with, ...}` paths resolve.
//
// `attention_forward_safe_into` is `pub` (not `pub(crate)`) because root's
// `dllm.rs` training code re-imports it via `use katgpt_forward::...` to
// preserve a single source of truth across its 5 callers.
#[cfg(feature = "dllm")]
pub mod d2f_context;
#[cfg(feature = "dllm")]
pub use d2f_context::{
    D2fContext, attention_forward_safe_into, denoising_accuracy, forward_block_causal_with,
};

// ── Plan 399 (2026-07-05): D2F wrapper cluster ──
// `d2f.rs`, `d2f_verifier.rs`, `diffusion_sampler.rs` moved from root
// `src/speculative/`. Root's copies are now thin re-export shims. The 8
// training-dependent tests stayed in root (they call
// `crate::dllm::{train_mini_dllm, generate_pattern_dataset}` which is
// training code); the 39 inference-only tests moved with the files.
//
// Gating mirrors root: `d2f` is `dllm`-gated (base inference pipeline),
// `d2f_verifier` and `diffusion_sampler` are additionally `tri_mode`-gated.
#[cfg(feature = "dllm")]
pub mod d2f;

#[cfg(all(feature = "dllm", feature = "tri_mode"))]
pub mod d2f_verifier;
#[cfg(all(feature = "dllm", feature = "tri_mode"))]
pub use d2f_verifier::D2fDrafterVerifier;

#[cfg(all(feature = "dllm", feature = "tri_mode"))]
pub mod diffusion_sampler;
#[cfg(all(feature = "dllm", feature = "tri_mode"))]
pub use diffusion_sampler::{
    DiffusionSampler, SamplerDecision, SamplerFeatures, SamplerTrajectory, SamplerVariant,
    collect_trajectories,
};

// ── Plan 400 (2026-07-05): FlashAR cluster ──
// `flashar_anchor.rs` and `flashar_consensus.rs` moved from root
// `src/speculative/`. Root's copies are now thin re-export shims. All 10
// consensus tests + 6 of 8 anchor tests moved with the production files; the
// 2 training-coupled anchor tests stayed in root (they call
// `crate::dllm::{train_mini_dllm, generate_pattern_dataset}` which is training
// code).
//
// Gating mirrors root: both files consume `crate::d2f::*` (dllm-gated), so
// they require `dllm` plus their own tracking flag. Root's `flashar_anchor`
// pulls only `dllm`; root's `flashar_consensus` pulls `tri_mode` + `plasma_path`
// (which transitively pulls `dllm`).
#[cfg(all(feature = "dllm", feature = "flashar_anchor"))]
pub mod flashar_anchor;
#[cfg(all(feature = "dllm", feature = "flashar_anchor"))]
pub use flashar_anchor::{AnchorConfig, AnchorFillResult, anchor_then_fill};

#[cfg(all(feature = "dllm", feature = "flashar_consensus"))]
pub mod flashar_consensus;
#[cfg(all(feature = "dllm", feature = "flashar_consensus"))]
pub use flashar_consensus::{
    ConsensusConfig, ConsensusResult, DualPathResult, FlashARConsensusVerifier, MAX_DRAFT_WIDTH,
    ThermalPath, compute_ternary_consensus, dual_path_draft, route_thermal_paths,
};

// ── Plan 401 (2026-07-06): forward_set_causal_positions extraction ──
// `forward_set_causal_positions` + 5 of 7 Research 376 T0.2 tests moved from
// root `src/dllm.rs` (the function is pure inference — no gradients/backprop/
// loss, despite the old "Root-resident by design (Issue 033 §C)" comment).
// Root keeps a re-export shim at `crate::dllm::forward_set_causal_positions`
// so every historical caller continues to resolve, including the 2 comparison
// tests still in root that also need `forward_block_causal_positions` /
// `forward_bidirectional_positions` (deferred to Plan 402).
//
// Gating mirrors root: `set_diffusion` is the root feature, forwarded here as
// a tracking flag. The function is self-contained (no `dllm`-gated intra-crate
// deps), so only `set_diffusion` is required.
#[cfg(feature = "set_diffusion")]
pub mod forward_set_causal;
#[cfg(feature = "set_diffusion")]
pub use forward_set_causal::forward_set_causal_positions;

// ── Plan 402 (2026-07-06): forward-positions cluster ──
// (`BidirectionalContext`, `forward_bidirectional_positions`,
// `forward_bidirectional_positions_into`, `attention_forward_safe` allocating
// wrapper, `forward_block_causal_positions` moved from root `src/dllm.rs`).
// Gated `dllm` because the module depends on `attention_forward_safe_into`
// from `d2f_context` (which is `dllm`-gated). Root re-exports these so every
// historical `crate::dllm::*` import path continues to resolve.
//
// The struct fields are `pub` because root's `denoise_loop_rcd` /
// `denoise_loop_rcd_3sr` (which stay in root) write directly to the
// cfg-gated `rcd_residual_embeddings` / `tsr_warm_start_embeddings` buffers.
#[cfg(feature = "dllm")]
pub mod forward_positions;
#[cfg(feature = "dllm")]
pub use forward_positions::{
    attention_forward_safe, forward_bidirectional_positions, forward_bidirectional_positions_into,
    forward_block_causal_positions, BidirectionalContext,
};

// ── Plan 401 (2026-07-06): set_diffusion.rs relocation ──
// The full set-diffusion inference decoder + 23 PURE inference tests moved
// from root `src/speculative/set_diffusion.rs`. Root's copy is now a thin
// re-export shim (`pub use katgpt_forward::set_diffusion::*;`) plus the
// 6 TRAIN-only tests that call `crate::dllm::{train_mini_dllm,
// generate_pattern_dataset, evaluate_set_causal_nelbo, train_mini_set_causal}`
// (root-local training code).
//
// Gating mirrors root: the decoder + CpuSetCausalForward adapter are
// `set_diffusion`-gated (the adapter calls
// `crate::forward_set_causal::forward_set_causal_positions` which is itself
// `set_diffusion`-gated). The 3 gen-steps helpers (`order_to_gen_steps`,
// `block_causal_gen_steps`, `mdlm_gen_steps`) + the core decode loop +
// `SetDiffusionConfig`/`SetDiffusionResult`/`SetCausalForwardFn` trait are
// *unconditional* in the source — but the module is registered behind
// `set_diffusion` to mirror root's `#[cfg(feature = "set_diffusion")] pub mod set_diffusion;`
// gating in `src/speculative/mod.rs`. Root's re-export shim always provides
// the symbols; they're only reachable when the root feature is on.
#[cfg(feature = "set_diffusion")]
pub mod set_diffusion;
