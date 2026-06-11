//! FidelityMatcher — Error-Matched KV Compression Level Selection
//!
//! Applies the paper's "error-matched classical solver" concept to KV cache
//! compression: for each sequence position, find the compression level that
//! produces the same perplexity as full attention (the "classical solver").
//!
//! This is Phase 3 of Plan 250. Stub implementation for now.

/// Compression level for KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CompressionLevel {
    /// No compression — full-precision KV cache.
    None = 0,
    /// 8-bit quantization.
    Bit8 = 1,
    /// 4-bit quantization.
    Bit4 = 2,
    /// 3-bit quantization.
    Bit3 = 3,
    /// 2-bit quantization.
    Bit2 = 4,
}

impl CompressionLevel {
    /// All levels from highest to lowest fidelity.
    pub const ALL: [CompressionLevel; 5] = [
        CompressionLevel::None,
        CompressionLevel::Bit8,
        CompressionLevel::Bit4,
        CompressionLevel::Bit3,
        CompressionLevel::Bit2,
    ];

    /// Bits per element.
    pub const fn bits(&self) -> u8 {
        match self {
            Self::None => 32,
            Self::Bit8 => 8,
            Self::Bit4 => 4,
            Self::Bit3 => 3,
            Self::Bit2 => 2,
        }
    }

    /// Compression ratio vs full precision.
    pub const fn compression_ratio(&self) -> f64 {
        32.0 / self.bits() as f64
    }
}

/// Error-matched compression level selector.
///
/// Given a target perplexity delta (how much quality loss is acceptable),
/// returns the highest compression level that stays within the delta
/// at each sequence position.
///
/// Stub — requires calibration data to function.
pub struct FidelityMatcher {
    /// Target perplexity delta (0.0 = lossless).
    target_delta: f64,
}

impl FidelityMatcher {
    /// Create with target perplexity delta.
    ///
    /// `target_delta` = acceptable increase in perplexity over full attention.
    /// 0.0 = lossless, 0.1 = ~10% perplexity increase acceptable.
    pub fn new(target_delta: f64) -> Self {
        Self { target_delta }
    }

    /// Get the error-matched compression level for a given position.
    ///
    /// Later positions can typically use higher compression because:
    /// 1. KV cache is larger → more compression savings
    /// 2. Earlier positions have less influence on current token
    /// 3. Attention weights decay for distant positions
    ///
    /// Stub: returns Bit4 for all positions until calibration is implemented.
    pub fn error_matched_level(&self, _pos: usize) -> CompressionLevel {
        // TODO: Implement calibration-based level selection
        // For now, use a simple heuristic: higher compression for later positions
        CompressionLevel::Bit4
    }

    /// Target perplexity delta.
    pub fn target_delta(&self) -> f64 {
        self.target_delta
    }
}

impl Default for FidelityMatcher {
    fn default() -> Self {
        Self::new(0.05) // 5% perplexity increase acceptable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_level_bits() {
        assert_eq!(CompressionLevel::None.bits(), 32);
        assert_eq!(CompressionLevel::Bit8.bits(), 8);
        assert_eq!(CompressionLevel::Bit4.bits(), 4);
        assert_eq!(CompressionLevel::Bit3.bits(), 3);
        assert_eq!(CompressionLevel::Bit2.bits(), 2);
    }

    #[test]
    fn test_compression_ratio() {
        assert_eq!(CompressionLevel::None.compression_ratio(), 1.0);
        assert_eq!(CompressionLevel::Bit8.compression_ratio(), 4.0);
        assert_eq!(CompressionLevel::Bit4.compression_ratio(), 8.0);
        assert_eq!(CompressionLevel::Bit3.compression_ratio(), 32.0 / 3.0);
        assert_eq!(CompressionLevel::Bit2.compression_ratio(), 16.0);
    }

    #[test]
    fn test_fidelity_matcher_default() {
        let fm = FidelityMatcher::default();
        assert_eq!(fm.target_delta(), 0.05);
    }

    #[test]
    fn test_fidelity_matcher_returns_level() {
        let fm = FidelityMatcher::new(0.1);
        let level = fm.error_matched_level(0);
        assert_eq!(level, CompressionLevel::Bit4);
    }
}
