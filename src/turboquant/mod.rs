//! TurboQuant: Near-optimal vector quantization for KV cache compression.
//!
//! Based on "TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate"
//! (arXiv:2504.19874). Compresses KV cache from f32 to 2-4 bits per coordinate.
//!
//! Architecture:
//! - Random rotation → Beta-distributed coordinates
//! - Lloyd-Max codebook → optimal scalar quantizer
//! - Bit-packed storage → 8-16× compression

pub mod codebook;
pub mod forward;
pub mod kv_cache;
pub mod rotation;
pub mod types;

// ── SpectralQuant (Plan 078) ──────────────────────────────────
#[cfg(feature = "spectral_quant")]
pub mod nonuniform_quant;
#[cfg(feature = "spectral_quant")]
pub mod spectral;
#[cfg(feature = "spectral_quant")]
pub mod spectral_kv_cache;
#[cfg(feature = "spectral_quant")]
pub mod spectral_rotation;

pub use codebook::compute_codebook;
pub use kv_cache::TurboQuantKVCache;
pub use rotation::{generate_qjl_matrix, generate_rotation_matrix};
pub use types::{TurboQuantCodebook, TurboQuantKVCacheConfig, TurboQuantLayer};

// ── SpectralQuant re-exports (Plan 078) ───────────────────────
#[cfg(feature = "spectral_quant")]
pub use nonuniform_quant::{CompressedVector, NonUniformQuantizer};
#[cfg(feature = "spectral_quant")]
pub use spectral::{
    BitAllocator, LloydMaxQuantizer, calibrate_eigenbasis, cumulative_variance_thresholds,
    generate_selective_qjl_signs, marginal_gain, participation_ratio, spectral_gap, waterfill_bits,
};
#[cfg(feature = "spectral_quant")]
pub use spectral_kv_cache::SpectralQuantKVCache;
#[cfg(feature = "spectral_quant")]
pub use spectral_rotation::{RandomRotation, SpectralRotation};
#[cfg(feature = "spectral_quant")]
pub use types::{
    LloydMaxCodebook, SpectralQuantCalibration, SpectralQuantKVCacheConfig, SpectralQuantLayer,
    WaterfillAllocation,
};
