//! PeiraPruner — PEIRA alignment-modulated ScreeningPruner.
//!
//! Wraps any `ScreeningPruner` and modulates its relevance signal using PEIRA's
//! spectral alignment score. When PEIRA alignment is high (canonical structure
//! recovered), the inner relevance passes through. When alignment is low (early
//! training / poor convergence), relevance is attenuated — preventing untrusted
//! signals from contaminating the DDTree.
//!
//! # Formula
//!
//! ```text
//! relevance_peira(d, t, path) = inner_relevance × alignment^α
//! ```
//!
//! Where:
//! - `inner_relevance` = the wrapped pruner's relevance score
//! - `alignment` ∈ [0, 1] = PEIRA spectral alignment (from EMA covariance)
//! - `α` = modulation exponent (default: 0.5, sqrt-like soft gating)
//!
//! # Why alignment-modulated relevance?
//!
//! PEIRA's alignment score measures how well the cross-view covariance Σ recovers
//! canonical structure. Early in training, alignment is low — the predictor P* is
//! essentially random, so downstream signals are unreliable. By gating relevance
//! through alignment^α, we automatically suppress noise during warmup and
//! progressively trust the pruner as alignment converges.
//!
//! # Hot-path cost
//!
//! One multiplication + one `powf` call per token. The alignment score is computed
//! periodically (outside the hot-path) and cached. Total overhead: ~2ns/call.
//!
//! # Feature Gate
//!
//! All code behind `#[cfg(feature = "peira_distill")]`.

use super::types::ScreeningPruner;

// ── PeiraPruner ──────────────────────────────────────────────────

/// ScreeningPruner wrapper that modulates relevance using PEIRA spectral alignment.
///
/// # Design
///
/// - **Composable**: wraps any `ScreeningPruner` without modifying it
/// - **Zero-alloc hot-path**: one multiplication + one `powf`
/// - **Revertible**: set `alpha = 0.0` to disable (alignment^0 = 1.0)
/// - **Safe warmup**: low alignment → attenuated relevance → no garbage in DDTree
///
/// # Usage
///
/// ```rust,ignore
/// // Create with default settings
/// let pruner = PeiraPruner::new(BanditPruner::new(domain, Ucb1, 100));
///
/// // Update alignment periodically (e.g., after each training episode)
/// pruner.set_alignment(0.95);
///
/// // Use in DDTree — relevance is automatically modulated
/// let rel = pruner.relevance(depth, token, path);
/// ```
pub struct PeiraPruner<P: ScreeningPruner> {
    /// Inner pruner to wrap.
    inner: P,
    /// Modulation exponent α (default: 0.5).
    /// Higher = more aggressive gating. 0.0 = disabled.
    alpha: f32,
    /// Cached PEIRA alignment score ∈ [0, 1].
    /// Updated periodically via `set_alignment()`.
    alignment: f32,
}

impl<P: ScreeningPruner> PeiraPruner<P> {
    /// Create a new PeiraPruner wrapping an inner pruner.
    ///
    /// Starts with `alignment = 0.0` (fully attenuated) until `set_alignment()`
    /// is called. This is intentional — forces the caller to explicitly provide
    /// an alignment score before the pruner passes through any signal.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            alpha: 0.5,
            alignment: 0.0,
        }
    }

    /// Create with custom modulation exponent.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        assert!(
            alpha >= 0.0,
            "PeiraPruner alpha must be non-negative, got {alpha}"
        );
        self.alpha = alpha;
        self
    }

    /// Update the cached PEIRA alignment score.
    ///
    /// Call this periodically after computing alignment from `PeiraDistiller`
    /// or `peira_alignment_score()`. The value is cached for the hot-path.
    pub fn set_alignment(&mut self, alignment: f32) {
        self.alignment = alignment.clamp(0.0, 1.0);
    }

    /// Get the current cached alignment score.
    pub fn alignment(&self) -> f32 {
        self.alignment
    }

    /// Get the modulation exponent.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Compute the alignment modulation factor: `alignment^alpha`.
    ///
    /// - alignment=1.0 → factor=1.0 (full pass-through)
    /// - alignment=0.5, alpha=0.5 → factor=0.707
    /// - alignment=0.0 → factor=0.0 (full attenuation)
    #[inline]
    pub fn modulation_factor(&self) -> f32 {
        // alignment^alpha: use powf for generality
        // For alpha=0.5 this is sqrt(alignment) — a soft gate
        // For alpha=0.0 this is 1.0 — disabled
        if self.alpha == 0.0 {
            return 1.0;
        }
        if self.alignment <= 0.0 {
            return 0.0;
        }
        self.alignment.powf(self.alpha)
    }

    /// Access the inner pruner.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Mutable access to the inner pruner.
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }
}

impl<P: ScreeningPruner> ScreeningPruner for PeiraPruner<P> {
    #[inline]
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
        let inner = self.inner.relevance(depth, token_idx, parent_tokens);
        if inner <= 0.0 {
            return 0.0;
        }

        let factor = self.modulation_factor();
        (inner * factor).clamp(0.0, 1.0)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::types::NoScreeningPruner;

    #[test]
    fn zero_alignment_attenuates_everything() {
        let pruner = PeiraPruner::new(NoScreeningPruner);
        // Default alignment = 0.0 → factor = 0.0 → relevance = 0.0
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.0).abs() < 1e-6, "Expected 0.0, got {rel}");
    }

    #[test]
    fn perfect_alignment_passes_through() {
        let mut pruner = PeiraPruner::new(NoScreeningPruner);
        pruner.set_alignment(1.0);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6, "Expected 1.0, got {rel}");
    }

    #[test]
    fn half_alignment_with_alpha_half() {
        let mut pruner = PeiraPruner::new(NoScreeningPruner);
        pruner.set_alignment(0.5);
        // factor = 0.5^0.5 = sqrt(0.5) ≈ 0.7071
        let rel = pruner.relevance(0, 0, &[]);
        let expected = 0.5f32.powf(0.5);
        assert!(
            (rel - expected).abs() < 1e-5,
            "Expected {expected}, got {rel}"
        );
    }

    #[test]
    fn alpha_zero_disables_gating() {
        let mut pruner = PeiraPruner::new(NoScreeningPruner).with_alpha(0.0);
        pruner.set_alignment(0.0); // Even zero alignment...
        let factor = pruner.modulation_factor();
        assert!(
            (factor - 1.0).abs() < 1e-6,
            "alpha=0 should give factor=1.0"
        );
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 1.0).abs() < 1e-6, "Should pass through");
    }

    #[test]
    fn alpha_one_is_linear_gating() {
        let mut pruner = PeiraPruner::new(NoScreeningPruner).with_alpha(1.0);
        pruner.set_alignment(0.3);
        let rel = pruner.relevance(0, 0, &[]);
        assert!(
            (rel - 0.3).abs() < 1e-5,
            "alpha=1.0 should be linear: {rel}"
        );
    }

    #[test]
    fn composes_with_inner_pruner() {
        struct HalfPruner;
        impl ScreeningPruner for HalfPruner {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.5
            }
        }

        let mut pruner = PeiraPruner::new(HalfPruner);
        pruner.set_alignment(0.64);
        // factor = 0.64^0.5 = 0.8
        // relevance = 0.5 * 0.8 = 0.4
        let rel = pruner.relevance(0, 0, &[]);
        let expected = 0.5 * 0.64f32.powf(0.5);
        assert!(
            (rel - expected).abs() < 1e-5,
            "Expected {expected}, got {rel}"
        );
    }

    #[test]
    fn zero_inner_is_zero_regardless_of_alignment() {
        struct ZeroPruner;
        impl ScreeningPruner for ZeroPruner {
            fn relevance(&self, _: usize, _: usize, _: &[usize]) -> f32 {
                0.0
            }
        }

        let mut pruner = PeiraPruner::new(ZeroPruner);
        pruner.set_alignment(1.0);
        let rel = pruner.relevance(0, 0, &[]);
        assert!((rel - 0.0).abs() < 1e-6);
    }

    #[test]
    fn alignment_clamped_to_unit_interval() {
        let mut pruner = PeiraPruner::new(NoScreeningPruner);
        pruner.set_alignment(-0.5);
        assert!(
            (pruner.alignment() - 0.0).abs() < 1e-6,
            "Should clamp negative"
        );
        pruner.set_alignment(1.5);
        assert!((pruner.alignment() - 1.0).abs() < 1e-6, "Should clamp > 1");
    }

    #[test]
    fn modulation_factor_monotonic_in_alignment() {
        let mut pruner = PeiraPruner::new(NoScreeningPruner);
        let mut prev = 0.0f32;
        for a in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0] {
            pruner.set_alignment(a);
            let factor = pruner.modulation_factor();
            assert!(
                factor >= prev - 1e-6,
                "Not monotonic at alignment={a}: {factor} < {prev}"
            );
            prev = factor;
        }
    }

    #[test]
    fn higher_alpha_more_aggressive_gating() {
        let mut soft = PeiraPruner::new(NoScreeningPruner).with_alpha(0.25);
        let mut hard = PeiraPruner::new(NoScreeningPruner).with_alpha(2.0);

        soft.set_alignment(0.5);
        hard.set_alignment(0.5);

        let soft_factor = soft.modulation_factor();
        let hard_factor = hard.modulation_factor();

        // 0.5^0.25 ≈ 0.84 > 0.5^2.0 = 0.25
        assert!(
            soft_factor > hard_factor,
            "Soft alpha should give higher factor: {soft_factor} vs {hard_factor}"
        );
    }

    #[test]
    fn hot_path_overhead_vs_baseline() {
        use std::hint::black_box;
        use std::time::Instant;

        let iters = 1_000_000;

        // Baseline: NoScreeningPruner alone
        let baseline = NoScreeningPruner;
        let start = Instant::now();
        for i in 0..iters {
            black_box(baseline.relevance(black_box(0), black_box(i % 1000), black_box(&[])));
        }
        let baseline_time = start.elapsed();

        // PEIRA-wrapped: alignment=1.0 (full pass-through)
        let mut pruner = PeiraPruner::new(NoScreeningPruner);
        pruner.set_alignment(1.0);
        let start = Instant::now();
        for i in 0..iters {
            black_box(pruner.relevance(black_box(0), black_box(i % 1000), black_box(&[])));
        }
        let peira_time = start.elapsed();

        eprintln!("   Baseline: {baseline_time:?}  PEIRA: {peira_time:?}");

        let overhead_pct =
            (peira_time.as_nanos() as f64 / baseline_time.as_nanos() as f64 - 1.0) * 100.0;

        // Gate: PeiraPruner must add <100% overhead on relevance() hot-path
        // (powf is more expensive than the multiply in FlowPruner, but
        // this is still nanosecond-scale per token)
        assert!(
            overhead_pct < 100.0,
            "PeiraPruner overhead too high: {overhead_pct:.1}%"
        );
    }
}
