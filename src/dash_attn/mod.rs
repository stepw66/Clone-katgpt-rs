//! DashAttention — Adaptive Sparse Hierarchical Attention via α-entmax routing.
//!
//! Feature gate: `dash_attn` (Plan 106, Research 68).
//! Replaces PFlash's fixed-budget top-k block selection with adaptive support
//! selection via α=1.5 entmax (α=1.5). Includes learned chunk summaries via head_cls.
//!
//! # Phase 12 absorption (Proposal 003, 2026-07-04)
//!
//! All DashAttention primitives now live in `katgpt-attn::dash_attn`:
//! - Clean core (`entmax`, `routing`, `chunk_summary`) — Phase 2.
//! - Composition layer (`forward_dash_attn_prefill` / `forward_dash_attn_decode`) —
//!   Issue 007 Phase F.4a.
//! - VortexFlow cluster (`vortex_flow`, `block_topk`, `channel_aware`,
//!   `entmax_router`, `kv_outer_prefill`, `msa_distill`, `value_energy`,
//!   `adaptive_k`, `meta_router`, `sat_analysis`) — Phase 12.
//!
//! The original "stays root" blocker (meta_router→pruners::bandit +
//! speculative::types, sat_analysis→cache_prune) dissolved once those landed in
//! their leaf crates (`katgpt-pruners`, `katgpt-core::traits`, `katgpt-kv`).
//!
//! This file is now a pure re-export shim. The token-level `tests.rs` stays
//! root (needs root transformer glue for `forward_dash_attn_decode_vortex`).

#[cfg(test)]
mod tests;

// Re-export the entire leaf module surface so `crate::dash_attn::*` paths
// continue to resolve for root consumers (transformer.rs, inference_router.rs,
// benchmark/, and the root-kept tests.rs).
pub use katgpt_attn::dash_attn::{
    chunk_summary, entmax, forward, routing,
    adaptive_k, block_topk, channel_aware, entmax_router, kv_outer_prefill,
    meta_router, msa_distill, sat_analysis, value_energy, vortex_flow,
};

// Flat symbol re-exports (preserved for back-compat with `katgpt_rs::dash_attn::Symbol` callers).
pub use katgpt_attn::dash_attn::chunk_summary::{ChunkSummaryCache, ChunkSummaryQuery};
pub use katgpt_attn::dash_attn::entmax::{entmax_1p5, entmax_gqa_aggregate, entmax_support};
pub use katgpt_attn::dash_attn::routing::{compute_routing_bias, score_blocks_entmax};
pub use katgpt_attn::dash_attn::forward::{forward_dash_attn_decode, forward_dash_attn_prefill};

#[cfg(feature = "msa_adaptive_k")]
pub use katgpt_attn::dash_attn::adaptive_k::{AdaptiveKConfig, AdaptiveKRouter};
#[cfg(feature = "msa_per_group")]
pub use katgpt_attn::dash_attn::block_topk::PerGroupTopKRouter;
#[cfg(feature = "vortex_flow")]
pub use katgpt_attn::dash_attn::block_topk::{BlockTopKCache, BlockTopKRouter};
#[cfg(feature = "vortex_flow")]
pub use katgpt_attn::dash_attn::channel_aware::{
    ChannelAwareCache, ChannelAwareRouter, RoutingChannelDiscovery, RoutingChannelMask,
    simd_dot_f32,
};
#[cfg(feature = "vortex_flow")]
pub use katgpt_attn::dash_attn::entmax_router::{EntmaxCache, EntmaxRouter};
#[cfg(feature = "msa_kv_outer")]
pub use katgpt_attn::dash_attn::kv_outer_prefill::{KvOuterIndex, KvOuterPrefill};
#[cfg(feature = "vortex_flow")]
pub use katgpt_attn::dash_attn::meta_router::{DynPolicy, DynRoutingCache, MetaRouter, compute_reward};
#[cfg(feature = "msa_sparse")]
pub use katgpt_attn::dash_attn::msa_distill::{MaxPoolBlockScorer, MaxStdDevBlockScorer, MsaBlockCache};
#[cfg(all(feature = "dash_attn", feature = "cache_prune"))]
pub use katgpt_attn::dash_attn::sat_analysis::{HeadSparsityInfo, head_sparsity_profile};
#[cfg(feature = "vortex_flow")]
pub use katgpt_attn::dash_attn::value_energy::{ValueEnergyCache, ValueEnergyRouter};
#[cfg(feature = "vortex_flow")]
pub use katgpt_attn::dash_attn::vortex_flow::{
    RoutingDecision, VortexFlow, VortexFlowConfig, VortexFlowExt, VortexRouter, VortexRouterCache,
    VortexScratch, build_vortex_router,
};
