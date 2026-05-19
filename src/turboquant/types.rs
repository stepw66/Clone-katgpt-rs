//! Core types for TurboQuant KV cache compression.

/// Lloyd-Max codebook for scalar quantization of Beta-distributed values.
#[derive(Debug, Clone)]
pub struct TurboQuantCodebook {
    /// Centroid values (2^bits entries).
    pub centroids: Vec<f32>,
    /// Decision boundaries (2^bits - 1 entries).
    pub boundaries: Vec<f32>,
    /// MSE per coordinate at this (dim, bits) setting.
    pub mse_per_coord: f32,
    /// Dimension the codebook was computed for.
    pub dim: usize,
    /// Bits per coordinate.
    pub bits: u8,
}

/// Per-layer quantization state (rotation matrix + codebooks).
#[derive(Debug, Clone)]
pub struct TurboQuantLayer {
    /// Random rotation matrix (dim × dim), stored row-major.
    pub rotation: Vec<f32>,
    /// QJL projection matrix for residual estimation.
    pub qjl_matrix: Vec<f32>,
    /// Codebook for key cache.
    pub key_codebook: TurboQuantCodebook,
    /// Codebook for value cache.
    pub val_codebook: TurboQuantCodebook,
}

/// Configuration for TurboQuant KV cache.
#[derive(Debug, Clone)]
pub struct TurboQuantKVCacheConfig {
    /// Bits per key coordinate (default: 3).
    pub key_bits: u8,
    /// Bits per value coordinate (default: 3).
    pub val_bits: u8,
    /// Random seed for rotation matrix (deterministic).
    pub seed: u64,
    /// Number of layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_head).
    pub kv_dim: usize,
    /// Maximum sequence length (block_size).
    pub max_seq_len: usize,
}

// ── SpectralQuant types (Plan 078) ──────────────────────────────────

/// Lloyd-Max codebook for per-dimension quantization.
#[cfg(feature = "spectral_quant")]
#[derive(Debug, Clone)]
pub struct LloydMaxCodebook {
    /// Centroid values (2^n_bits entries).
    pub centroids: Vec<f32>,
    /// Bits per coordinate.
    pub n_bits: u8,
}

/// Result of offline calibration per (layer, head, kv_type).
/// Computed once, serialized with model weights.
#[cfg(feature = "spectral_quant")]
#[derive(Debug, Clone)]
pub struct SpectralQuantCalibration {
    /// Eigenbasis matrix V (d_h × d_h), row-major.
    /// Columns are eigenvectors sorted by eigenvalue descending.
    pub eigenvectors: Vec<f32>,
    /// Eigenvalues from covariance eigendecomposition, sorted descending.
    pub eigenvalues: Vec<f32>,
    /// Effective dimensionality: (Σλ_i)² / Σ(λ_i²).
    /// Typically 4–6 at d_h=128.
    pub d_eff: f32,
    /// Spectral gap: λ_{d_eff} / λ_{d_eff+1}.
    pub spectral_gap: Option<f32>,
    /// Min components for 95% cumulative variance.
    pub var_95: usize,
    /// Min components for 99% cumulative variance.
    pub var_99: usize,
    /// Number of calibration samples collected.
    pub n_samples: usize,
    /// Head dimension.
    pub head_dim: usize,
}

/// Water-fill bit allocation result.
#[cfg(feature = "spectral_quant")]
#[derive(Debug, Clone)]
pub struct WaterfillAllocation {
    /// Whether water-fill is enabled (v2 path).
    pub use_water_fill: bool,
    /// First d_eff eigenvalues (used for marginal gain computation).
    pub eigenvalues: Vec<f32>,
    /// Per-semantic-dim bit widths from water-fill.
    pub bits_per_dim: Vec<u8>,
    /// Effective dimensionality.
    pub d_eff: usize,
    /// Total bits allocated to semantic subspace.
    pub total_semantic_bits: usize,
    /// Per-dim minimum bits.
    pub min_bits: u8,
    /// Per-dim maximum bits (None = uncapped).
    pub max_bits: Option<u8>,
    /// Formula version tag for serialization.
    pub formula_version: u8,
}

/// Per-layer SpectralQuant state: calibration + fitted codebooks.
#[cfg(feature = "spectral_quant")]
#[derive(Debug, Clone)]
pub struct SpectralQuantLayer {
    /// Calibration data (eigenvectors, eigenvalues, d_eff).
    pub calibration: SpectralQuantCalibration,
    /// QJL sign matrix: (qjl_dim × d_eff) Rademacher ±1.
    pub qjl_signs: Vec<f32>,
    /// Effective dimensionality (integer ceiling of d_eff).
    pub d_eff: usize,
    /// Semantic regime bits per coordinate.
    pub b_high: u8,
    /// Tail regime bits per coordinate.
    pub b_low: u8,
    /// Per-dim semantic bits (v2 water-fill path, None for v1).
    pub semantic_bits_per_dim: Option<Vec<u8>>,
    /// Per-dim semantic codebooks (v2 water-fill path, None for v1).
    pub per_dim_semantic_codebooks: Option<Vec<LloydMaxCodebook>>,
    /// Shared semantic codebook (v1 uniform path, None for v2).
    pub semantic_codebook: Option<LloydMaxCodebook>,
    /// Tail regime codebook (shared across all tail dims).
    pub tail_codebook: LloydMaxCodebook,
}

/// Configuration for SpectralQuant KV cache.
#[cfg(feature = "spectral_quant")]
#[derive(Debug, Clone)]
pub struct SpectralQuantKVCacheConfig {
    /// Average bits per coordinate across all dimensions.
    pub avg_bits: f32,
    /// Minimum bits for tail dimensions.
    pub min_tail_bits: u8,
    /// Maximum bits per dimension.
    pub max_bits: u8,
    /// QJL projection dimension.
    pub qjl_dim: usize,
    /// Max Lloyd-Max iterations.
    pub lloyd_max_iter: usize,
    /// Number of calibration samples to collect.
    pub calibration_samples: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Whether to use water-fill allocation (v2).
    pub use_water_fill: bool,
    /// Water-fill minimum bits per dim.
    pub wf_min_bits: u8,
    /// Water-fill maximum bits per dim.
    pub wf_max_bits: u8,
    /// Number of layers.
    pub n_layers: usize,
    /// KV dimension (head_dim × n_kv_head).
    pub kv_dim: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}
