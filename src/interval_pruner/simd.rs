//! SIMD-accelerated interval detection (placeholder).
//!
//! Currently delegates to scalar implementations in [`super::interval`].
//! Will be replaced with SIMD bit-scan when benchmarked as a bottleneck.

/// Placeholder: delegates to scalar [`crate::interval_pruner::interval::IntervalMask`].
#[cfg(feature = "interval_pruner")]
pub fn simd_is_interval_closed(mask: &[bool]) -> bool {
    use crate::interval_pruner::IntervalMask;
    IntervalMask::from_vec(mask.to_vec()).is_interval_closed()
}

/// Placeholder: delegates to scalar [`crate::interval_pruner::interval::IntervalMask`].
#[cfg(feature = "interval_pruner")]
pub fn simd_close_intervals(mask: &[bool], gap_threshold: usize) -> Vec<bool> {
    use crate::interval_pruner::IntervalMask;
    IntervalMask::from_vec(mask.to_vec())
        .close_intervals(gap_threshold)
        .as_slice()
        .to_vec()
}
