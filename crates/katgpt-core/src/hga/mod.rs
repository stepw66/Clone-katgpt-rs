//! Hierarchical Global Attention (HGA) — chunk→group→token routing with
//! RoPE-aware summaries (Plan 397, Research 379).
//!
//! Distilled from Frank, Fedosov, Grinenko (BMW Group) 2026, *"Hierarchical
//! Global Attention: Drop-In Exact-Token Routing for Pretrained Long-Context
//! Transformers"* ([arxiv 2606.30709](https://arxiv.org/abs/2606.30709)).
//!
//! # What this is
//!
//! Three refinements of the shipped sparse-attention routing slot:
//!
//! 1. **`GroupSummaryCache`** — a sub-chunk group middle routing tier between
//!    DashAttention's chunk-level entmax routing and token-level attention.
//! 2. **`MixedRopeSummarizer`** — per-frequency-pair RoPE-aware chunk/group
//!    summary construction (high-freq rotate-then-average, low-freq
//!    average-then-rotate-at-mid).
//! 3. Consumes the generic `TieredKvStore` (`crate::tiered_kv`) for route-and-fetch.
//!
//! The HGA forward path (`forward_hga`) lives in `katgpt_attn::hga_forward`
//! because it needs `dash_attn::entmax_1p5` (which lives in katgpt-attn, not
//! katgpt-core — katgpt-core cannot import katgpt-attn without a circular dep).
//!
//! # AGENTS.md compliance
//!
//! - **sigmoid not softmax** — routing decisions (chunk selection, group selection)
//!   use dot-product scoring + top-K (deterministic), NOT softmax. Softmax is used
//!   only for the final output attention over fetched real-token K/V (standard SDPA,
//!   which is retrieval, not gating — consistent with the hippocampal_cache
//!   precedent in Plan 395).

pub mod group_summary;
pub mod summary;

#[cfg(test)]
mod tests;

pub use group_summary::GroupSummaryCache;
pub use summary::MixedRopeSummarizer;
