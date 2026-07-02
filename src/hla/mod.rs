//! Higher-order Linear Attention (HLA) — O(1) inference cache.
//!
//! Implements second-order HLA (symmetric + asymmetric AHLA) as an alternative
//! to standard KV-cache attention. Achieves O(1) per-token memory independent
//! of sequence length, replacing the growing KV cache with fixed-size prefix
//! sufficient statistics that capture higher-order query-key interactions.
//!
//! # Variants
//!
//! | Variant | State per head | Per-token cost | Best for |
//! |---------|---------------|---------------|----------|
//! | **Symmetric HLA** | O(d² + d·dv) | O(d² + d·dv) | Small hd, quality-critical |
//! | **AHLA** (asymmetric) | O(d·dv) | O(d·dv) | Large hd, perf-critical |
//!
//! # Usage
//!
//! ```ignore
//! use microgpt::hla::{forward_hla, MultiLayerHlaCache};
//!
//! let config = Config::micro();
//! let weights = TransformerWeights::random(&config);
//! let mut ctx = ForwardContext::new(&config);
//! let mut cache = MultiLayerHlaCache::new(&config);
//!
//! // Streaming inference — no context window limit
//! let logits = forward_hla(&mut ctx, &weights, &mut cache, token, pos, &config);
//! ```
//!
//! # Key Insight
//!
//! The second-order attention matrix QKᵀQKᵀᵀ = Q(KᵀK)Qᵀ depends only on
//! KᵀK (a d×d matrix), not the full N×N attention matrix. HLA maintains
//! running summaries (prefix sufficient statistics) of these moments.
//!
//! **Note:** HLA computes a different function than softmax attention.
//! Models must be trained with HLA from scratch for quality.
//!
//! # Substrate location
//!
//! The cache types ([`HlaLayerState`], [`MultiLayerHlaCache`],
//! [`MultiLayerAhlaCache`], Parallax variants, [`HlaVariant`]) and streaming
//! kernels ([`hla_state_update`], [`hla_readout`], [`ahla_step`], full-layer
//! helpers) live in **`katgpt-core::hla`** (Plan 008 Phase 1 Step 4,
//! 2026-06-28). This root module re-exports them for backward compatibility
//! and adds the **composition layer** ([`forward`] module) that wires the
//! kernels into a full transformer forward pass via `ForwardContext`.
//!
//! The split mirrors Plan 008 Step 2 (`katgpt-transformer`): substrate data
//! types + leaf kernels move to the publishable leaf; composition functions
//! that need engine-tier types (`ForwardContext` with pruner fields) stay in
//! the root.
//!
//! Reference: Zhang, Qin, Wang, Gu (2026). "Higher-order Linear Attention."
//! See `.research/28_Higher_order_Linear_Attention.md` for full derivation.

// Substrate re-export from katgpt-hla (Plan 008 Step 4, 2026-06-28; Issue 007
// Phase E Tier 2 #4). katgpt-core re-exports katgpt-hla as `katgpt_core::hla`.
// The types + kernels moved down so any crate can `cargo add katgpt-core`
// and get the HLA substrate without pulling the root engine.
pub use katgpt_core::hla::{kernel, types};

// Composition layer re-export (Issue 007 Phase F.4b, 2026-07-02):
// `forward_hla` / `forward_ahla` / `generate_*` moved from root
// `src/hla/forward.rs` into katgpt-forward (not katgpt-hla — katgpt-core
// depends on katgpt-hla for substrate re-export, so katgpt-hla can't depend
// on katgpt-forward without a cycle). Re-exported here so all historical
// `crate::hla::forward_hla` call sites resolve unchanged.
pub use katgpt_forward::{forward_ahla, forward_hla, generate_ahla_into, generate_hla_into};

// Re-export the substrate API at `crate::hla::*` for backward compatibility
// with all existing call sites (`crate::hla::MultiLayerHlaCache`, etc.).
pub use katgpt_core::hla::{
    ahla_denom, ahla_layer_step, ahla_step, hla_denom, hla_layer_readout, hla_layer_update,
    hla_readout, hla_readout_normalized, hla_state_update,
};
pub use katgpt_core::hla::{
    AhlaLayerState, AhlaQHeadState, HlaLayerState, HlaQHeadState, HlaVariant, MultiLayerAhlaCache,
    MultiLayerHlaCache, MultiLayerParallaxAhlaCache, ParallaxAhlaLayerState,
    ParallaxAhlaQHeadState,
};
