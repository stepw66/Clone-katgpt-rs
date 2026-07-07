//! `VarianceAdaptiveGuidance` (Plan 268 F4) — sigmoid-gated per-query
//! guidance weight `1/β`.
//!
//! # QGF Paper Insight
//!
//! The paper (Fig 20) shows the guidance weight `1/β` has a sweet spot:
//! too low → no improvement, too high → off-manifold exploitation.
//! The paper uses a *fixed* `1/β` tuned per-domain.
//!
//! # Our Extension
//!
//! We make `1/β` **adaptive per-query** using the critic's own variance
//! as the confidence signal. This is a novel extension the paper doesn't
//! explore, but is natural given our existing variance-probe infrastructure
//! (Thicket, Plan 267).
//!
//! ```text
//! confidence = 1.0 - normalized_variance(Q(s, ·))
//! guidance_weight = sigmoid(k · (confidence - threshold))
//! ```
//!
//! - **Low confidence** (noisy critic, novel state) → weight ≈ 0 →
//!   pure BC reference policy (safe fallback)
//! - **High confidence** (stable critic, familiar state) → weight ≈ 1 →
//!   strong Q-guidance (aggressive, high-quality)
//!
//! # Why Sigmoid, Not Softmax
//!
//! Per project rules (see `.contexts/optimization.md`):
//! - Sigmoid is per-query (independent of other queries)
//! - Softmax would couple queries (unnecessary, expensive)
//! - Sigmoid is one SIMD op; softmax requires a normalization pass
//!
//! # Stability
//!
//! Confidence is **EMA-smoothed** to avoid oscillation. Use the existing
//! PrudentBanker pattern (`alpha = 0.01`) — confidence changes slowly
//! because the critic's variance changes slowly.

/// Compute the adaptive guidance weight from a confidence value.
///
/// `guidance_weight = sigmoid(k · (confidence − threshold))`
///
/// Returns a value in `(0.0, 1.0)` suitable for use as `1/β` in QGF.
///
/// # Arguments
///
/// - `confidence`: critic confidence at the current state, in `[0.0, 1.0]`.
///   From `QGradientOracle::confidence()` or Thicket variance probe.
/// - `threshold`: the confidence level at which guidance is half-maximal.
///   Typical: `0.5` (guidance activates when critic is above average confidence).
/// - `steepness`: controls the transition sharpness. Typical: `4.0` to `8.0`.
///   Higher = sharper on/off; lower = smoother ramp.
///
/// # Properties
///
/// - At `confidence = threshold`, returns exactly `0.5`
/// - At `confidence << threshold`, returns ≈ `0.0` (pure BC reference)
/// - At `confidence >> threshold`, returns ≈ `1.0` (strong Q-guidance)
/// - Monotonically increasing in `confidence`
/// - Continuous (no discontinuity at the threshold)
///
/// # Example
///
/// ```
/// # use katgpt_core::qgf::adaptive_guidance_weight;
/// // Low confidence: ~5% guidance (safe)
/// let w_low = adaptive_guidance_weight(0.1, 0.5, 6.0);
/// assert!(w_low < 0.1);
///
/// // High confidence: ~95% guidance (aggressive)
/// let w_high = adaptive_guidance_weight(0.9, 0.5, 6.0);
/// assert!(w_high > 0.9);
///
/// // At threshold: exactly 50%
/// let w_mid = adaptive_guidance_weight(0.5, 0.5, 6.0);
/// assert!((w_mid - 0.5).abs() < 1e-6);
/// ```
#[inline]
pub fn adaptive_guidance_weight(confidence: f32, threshold: f32, steepness: f32) -> f32 {
    // Standard logistic sigmoid: 1 / (1 + exp(-x))
    // We use the numerically-stable branch: for x >= 0, use 1/(1+exp(-x));
    // for x < 0, use exp(x)/(1+exp(x)).
    let x = steepness * (confidence - threshold);
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Default configuration for adaptive guidance.
///
/// Tuned to be conservative — guidance activates only when the critic is
/// clearly confident. This matches QGF's finding that overly-aggressive
/// guidance pushes actions off-manifold.
pub const DEFAULT_THRESHOLD: f32 = 0.5;
pub const DEFAULT_STEEPNESS: f32 = 6.0;

/// Convenience: adaptive guidance weight with default config.
#[inline]
pub fn adaptive_guidance_weight_default(confidence: f32) -> f32 {
    adaptive_guidance_weight(confidence, DEFAULT_THRESHOLD, DEFAULT_STEEPNESS)
}

// ─────────────────────────────────────────────────────────────────────────
// Thicket (Plan 267) variance-probe integration — Plan 268 T7.
// ─────────────────────────────────────────────────────────────────────────
//
// Plan 268 T7: "Reuse Thicket (Plan 267) variance probe as the confidence
// signal." Originally deferred as "Phase 5 integration work", but per the
// modelless-first mandate the integration is a pure numeric bridge: a
// trait in katgpt-core (the lower layer), an impl in katgpt-pruners for
// its concrete `TvpSignal` (the upper layer that owns Thicket), and a
// free-function conversion `confidence = 1 − clamp(disagreement, 0, 1)`.
//
// No new deps. The katgpt-core primitive stays substrate-agnostic: it knows
// about "a normalized disagreement in [0,1]", not about decoding-space probes.

/// A normalized variance / disagreement signal that can drive the adaptive
/// guidance weight.
///
/// Implementors produce a scalar in `[0.0, 1.0]` where:
/// - `0.0` means the source agrees with itself (low variance, high confidence)
/// - `1.0` means maximally disagreeing (high variance, low confidence)
///
/// The canonical implementor is Thicket's `TvpSignal` (Plan 267, in
/// `katgpt-pruners`), whose `reasoning_disagreement` field is exactly this
/// signal. Any future variance probe (RV, BoM disagreement, ensemble KL)
/// can implement this trait and feed QGF's F4 adaptive weight without
/// touching the QGF primitive.
///
/// # Why a trait, not a concrete struct
///
/// katgpt-core cannot depend on katgpt-pruners (the dependency runs the
/// other way). A trait lets katgpt-pruners implement against the QGF
/// abstraction without forcing katgpt-core to import `TvpSignal`. This is
/// the same decoupling pattern already used by `QGradientOracle`.
pub trait QgfVarianceSignal {
    /// Normalized disagreement in `[0.0, 1.0]`. Higher = more variance =
    /// less confidence in the underlying critic / generator.
    ///
    /// Implementations SHOULD clamp to `[0, 1]` defensively — callers feed
    /// this directly into `confidence_from_disagreement`.
    fn normalized_disagreement(&self) -> f32;
}

/// Convert a normalized disagreement in `[0, 1]` to a critic confidence in
/// `[0, 1]`.
///
/// `confidence = 1.0 − clamp(disagreement, 0.0, 1.0)`
///
/// This is the bridge function between the Thicket variance-probe semantics
/// (high disagreement = bad) and the QGF adaptive-weight semantics (high
/// confidence = strong guidance). Pure, `#[inline]`, zero-alloc.
///
/// NaN is mapped to zero confidence (defensive — NaN must never silently
/// produce full-strength guidance).
///
/// # Plan 268 T7 / AGENTS.md bridge-pattern rule
///
/// Per the global `AGENTS.md` bridge rules: a raw → latent projection that
/// crosses the sync boundary must be zero-allocation, gateable, and must
/// not introduce sync dependency. This function is a one-scalar bridge; it
/// satisfies all three.
#[inline]
pub fn confidence_from_disagreement(disagreement: f32) -> f32 {
    if disagreement.is_nan() {
        return 0.0;
    }
    1.0 - disagreement.clamp(0.0, 1.0)
}

/// Adaptive guidance weight derived from a generic variance signal.
///
/// Equivalent to:
/// ```text
/// let confidence = confidence_from_disagreement(signal.normalized_disagreement());
/// adaptive_guidance_weight(confidence, threshold, steepness)
/// ```
///
/// This is the primary F4 entry point when wiring Thicket (Plan 267) into
/// QGF (Plan 268): the caller passes a `&TvpSignal` (or any other
/// `QgfVarianceSignal` implementor) and gets back a per-query `1/β`.
///
/// # Example
///
/// ```
/// # use katgpt_core::qgf::{adaptive_guidance_weight_from_signal, QgfVarianceSignal};
/// struct HighAgreementSignal;
/// impl QgfVarianceSignal for HighAgreementSignal {
///     fn normalized_disagreement(&self) -> f32 { 0.05 } // 95% confidence
/// }
/// let w = adaptive_guidance_weight_from_signal(&HighAgreementSignal, 0.5, 6.0);
/// assert!(w > 0.9, "high agreement → strong guidance, got {w}");
/// ```
#[inline]
pub fn adaptive_guidance_weight_from_signal<S: QgfVarianceSignal + ?Sized>(
    signal: &S,
    threshold: f32,
    steepness: f32,
) -> f32 {
    let confidence = confidence_from_disagreement(signal.normalized_disagreement());
    adaptive_guidance_weight(confidence, threshold, steepness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_confidence_near_zero() {
        let w = adaptive_guidance_weight(0.1, 0.5, 6.0);
        assert!(w < 0.1, "low confidence should give < 0.1 weight, got {w}");
    }

    #[test]
    fn test_high_confidence_near_one() {
        let w = adaptive_guidance_weight(0.9, 0.5, 6.0);
        assert!(w > 0.9, "high confidence should give > 0.9 weight, got {w}");
    }

    #[test]
    fn test_at_threshold_is_half() {
        let w = adaptive_guidance_weight(0.5, 0.5, 6.0);
        assert!(
            (w - 0.5).abs() < 1e-6,
            "at threshold should be 0.5, got {w}"
        );
    }

    #[test]
    fn test_monotonic_in_confidence() {
        // Guidance weight must be monotonically increasing in confidence.
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let conf = i as f32 / 100.0;
            let w = adaptive_guidance_weight(conf, 0.5, 6.0);
            assert!(
                w >= prev - 1e-6,
                "not monotonic at conf={conf}: prev={prev}, w={w}"
            );
            prev = w;
        }
    }

    #[test]
    fn test_continuous_at_threshold() {
        // No discontinuity — values approach 0.5 smoothly from both sides.
        let below = adaptive_guidance_weight(0.49, 0.5, 6.0);
        let at = adaptive_guidance_weight(0.5, 0.5, 6.0);
        let above = adaptive_guidance_weight(0.51, 0.5, 6.0);
        assert!(below < at);
        assert!(at < above);
        assert!((at - below).abs() < 0.05);
        assert!((above - at).abs() < 0.05);
    }

    #[test]
    fn test_steeper_is_sharper() {
        // Higher steepness → sharper transition.
        let gentle = adaptive_guidance_weight(0.55, 0.5, 2.0);
        let steep = adaptive_guidance_weight(0.55, 0.5, 20.0);
        assert!(
            steep > gentle,
            "steeper should activate faster: gentle={gentle}, steep={steep}"
        );
    }

    #[test]
    fn test_output_range() {
        // Output must always be in [0, 1] for any finite input.
        // Note: sigmoid asymptotically approaches (0, 1) but saturates at the
        // float limit for extreme inputs (e.g. confidence=3.28 with steepness=6
        // gives x=16.68, where exp(-16.68) underflows to 0 → sigmoid = 1.0 exactly).
        for i in -1000..=1000 {
            let conf = i as f32 / 100.0;
            let w = adaptive_guidance_weight(conf, 0.5, 6.0);
            assert!(
                (0.0..=1.0).contains(&w),
                "weight out of [0,1] for conf={conf}: w={w}"
            );
        }
    }

    // ── Plan 268 T7: Thicket variance-probe bridge tests ──────────────

    #[test]
    fn test_confidence_from_disagreement_endpoints() {
        // Zero disagreement → full confidence.
        assert_eq!(confidence_from_disagreement(0.0), 1.0);
        // Full disagreement → zero confidence.
        assert_eq!(confidence_from_disagreement(1.0), 0.0);
        // Half disagreement → half confidence.
        assert!((confidence_from_disagreement(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_confidence_from_disagreement_clamps_out_of_range() {
        // Negative disagreement clamps to 0 → confidence 1.0.
        assert_eq!(confidence_from_disagreement(-0.5), 1.0);
        // Disagreement > 1 clamps to 1 → confidence 0.0.
        assert_eq!(confidence_from_disagreement(1.5), 0.0);
        assert_eq!(confidence_from_disagreement(42.0), 0.0);
    }

    #[test]
    fn test_confidence_from_disagreement_nan_is_zero_confidence() {
        // NaN must never silently produce full-strength guidance — defensive.
        assert_eq!(confidence_from_disagreement(f32::NAN), 0.0);
    }

    #[test]
    fn test_confidence_from_disagreement_monotonic() {
        // Confidence is monotonically DECREASING in disagreement.
        let mut prev = f32::INFINITY;
        for i in 0..=100 {
            let d = i as f32 / 100.0;
            let c = confidence_from_disagreement(d);
            assert!(
                c <= prev + 1e-6,
                "not monotone-decreasing at d={d}: prev={prev}, c={c}"
            );
            prev = c;
        }
    }

    // A stand-in for Thicket's `TvpSignal` — the same shape katgpt-pruners
    // will implement against the real type.
    struct MockTvpLikeSignal {
        reasoning_disagreement: f32,
    }
    impl QgfVarianceSignal for MockTvpLikeSignal {
        fn normalized_disagreement(&self) -> f32 {
            self.reasoning_disagreement.clamp(0.0, 1.0)
        }
    }

    #[test]
    fn test_signal_high_agreement_gives_strong_guidance() {
        // Thicket semantics: low disagreement = probes agree = high confidence.
        let signal = MockTvpLikeSignal {
            reasoning_disagreement: 0.05,
        };
        let w = adaptive_guidance_weight_from_signal(&signal, 0.5, 6.0);
        assert!(w > 0.9, "low disagreement → strong guidance, got {w}");
    }

    #[test]
    fn test_signal_high_disagreement_gives_weak_guidance() {
        // High disagreement → low confidence → near-zero guidance.
        let signal = MockTvpLikeSignal {
            reasoning_disagreement: 0.95,
        };
        let w = adaptive_guidance_weight_from_signal(&signal, 0.5, 6.0);
        assert!(w < 0.1, "high disagreement → weak guidance, got {w}");
    }

    #[test]
    fn test_signal_matches_manual_composition() {
        // adaptive_guidance_weight_from_signal(s, τ, k)
        // == adaptive_guidance_weight(confidence_from_disagreement(s.d), τ, k)
        let signal = MockTvpLikeSignal {
            reasoning_disagreement: 0.3,
        };
        let via_signal = adaptive_guidance_weight_from_signal(&signal, 0.5, 6.0);
        let manual = adaptive_guidance_weight(
            confidence_from_disagreement(signal.reasoning_disagreement),
            0.5,
            6.0,
        );
        assert!((via_signal - manual).abs() < 1e-6);
    }

    #[test]
    fn test_signal_trait_object_works() {
        // The trait must be usable as `&dyn QgfVarianceSignal` (object-safe
        // erasure) so callers can hold a boxed collection of probe sources.
        let signal: &dyn QgfVarianceSignal = &MockTvpLikeSignal {
            reasoning_disagreement: 0.1,
        };
        let w = adaptive_guidance_weight_from_signal(signal, 0.5, 6.0);
        assert!(w > 0.5, "trait-object path must work, got {w}");
    }
}
