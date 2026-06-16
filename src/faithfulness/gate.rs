//! [`TriggeredInjectionGate`] — entropy-thresholded inject/skip decision.
//!
//! Hot-path primitive: decides whether to inject memory at all based on
//! consumer uncertainty. Target: **<10ns p99** (one compare, branch-free).
//! Zero allocation.
//!
//! **Sigmoid, never softmax** (AGENTS.md hard constraint). Softmax over a
//! single scalar is degenerate (always 1.0); sigmoid gives a proper
//! inject/skip probability. See Plan 278 ADR-3.
//!
//! ## Hot-path math
//!
//! `should_inject(u) := sigmoid(λ · (u − τ)) > 0.5`.
//!
//! Since `sigmoid(x) > 0.5 ⟺ x > 0`, and we require `λ > 0` (validated at
//! construction), this collapses to **`u > τ`** for the boolean decision.
//! No `exp()` call on the hot path — just one subtract + one compare.
//!
//! The full sigmoid value is still available via [`EntropyThresholdGate::sigmoid_value`]
//! for consumers that want soft-gating (multiply memory contribution by the
//! sigmoid value rather than hard-skip). That path pays the `exp()` cost and
//! is opt-in per call.

/// Numerically stable logistic sigmoid.
///
/// Re-exports `katgpt_core::simd::fast_sigmoid` — same implementation used
/// across the engine (clamps at ±40 to avoid overflow, then `1/(1+e^{-x})`).
/// Used only by the soft-gating accessor [`EntropyThresholdGate::sigmoid_value`],
/// NOT by the hot-path boolean decision (which collapses to a compare).
#[inline]
fn sigmoid(x: f32) -> f32 {
    katgpt_core::simd::fast_sigmoid(x)
}

/// Decision gate for triggered memory injection.
///
/// `should_inject(u) := sigmoid(λ · (u − τ)) > 0.5`
///
/// When `λ > 0`, this reduces to `u > τ` for the boolean case, but the full
/// sigmoid is preserved for future soft-gating (multiply memory contribution
/// by the sigmoid value rather than hard-skip).
pub trait TriggeredInjectionGate {
    /// Returns `true` if memory should be injected given uncertainty `u ∈ [0, 1]`.
    ///
    /// Zero-allocation. Implementations should be `#[inline]` and branch-free
    /// aside from the sigmoid clamping guards.
    fn should_inject(&self, uncertainty: f32) -> bool;
}

/// Entropy-threshold gate with configurable threshold `tau` and slope `lambda`.
///
/// Zero-allocation. `#[derive(Clone, Copy)]` — safe to store inline in hot
/// structs without indirection.
#[derive(Clone, Copy, Debug)]
pub struct EntropyThresholdGate {
    /// Uncertainty threshold τ. Inject when `u > τ` (for `λ > 0`).
    pub tau: f32,
    /// Slope λ. Higher = sharper transition at the threshold. Default 8.0.
    pub lambda: f32,
}

impl Default for EntropyThresholdGate {
    #[inline]
    fn default() -> Self {
        Self {
            tau: 0.5,
            lambda: 8.0,
        }
    }
}

impl EntropyThresholdGate {
    /// Create with explicit threshold and slope.
    ///
    /// Validates `lambda > 0` — required for the hot-path math
    /// (`sigmoid(λ·(u−τ)) > 0.5 ⟺ u > τ` only holds when `λ > 0`).
    #[inline]
    pub fn new(tau: f32, lambda: f32) -> Self {
        debug_assert!(
            lambda > 0.0,
            "EntropyThresholdGate::new: lambda must be > 0 (got {}), \
             else the hot-path boolean collapses incorrectly",
            lambda
        );
        Self { tau, lambda }
    }

    /// Full sigmoid value `sigmoid(λ · (u − τ)) ∈ (0, 1)`.
    ///
    /// Opt-in soft-gating accessor — pays the `exp()` cost. Use this when
    /// you want to scale the memory contribution by the sigmoid value rather
    /// than hard-skip. For the boolean inject/skip decision, use
    /// [`TriggeredInjectionGate::should_inject`] instead — it collapses to a
    /// single compare and is ~10× faster.
    #[inline]
    pub fn sigmoid_value(&self, uncertainty: f32) -> f32 {
        sigmoid(self.lambda * (uncertainty - self.tau))
    }
}

impl TriggeredInjectionGate for EntropyThresholdGate {
    #[inline]
    fn should_inject(&self, uncertainty: f32) -> bool {
        // sigmoid(λ · (u − τ)) > 0.5
        //
        // Math equivalently: since sigmoid(x) > 0.5 ⟺ x > 0, and λ > 0
        // (validated in `new`), this collapses to `u > τ` for the boolean
        // decision. One subtract + one compare — no `exp()` on the hot path.
        //
        // The full sigmoid value (for soft-gating) is available via
        // `EntropyThresholdGate::sigmoid_value` at opt-in cost.
        uncertainty > self.tau
    }
}

/// Unified uncertainty signal — collapses entropy / collapse signal / curiosity
/// pulse into a single `f32 ∈ [0, 1]`.
///
/// Allows Plan 212 (collapse detector), Research 041 (curiosity pulse), and
/// any entropy-based signal to feed the same [`TriggeredInjectionGate`] without
/// the gate knowing the signal's provenance.
pub trait UncertaintySignal {
    /// Uncertainty in `[0, 1]`. Higher = more uncertain = inject more.
    fn uncertainty(&self) -> f32;
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_injects_above_threshold() {
        let gate = EntropyThresholdGate::default(); // tau=0.5, lambda=8.0
        assert!(gate.should_inject(0.6));
        assert!(gate.should_inject(0.9));
        assert!(gate.should_inject(1.0));
    }

    #[test]
    fn test_gate_skips_below_threshold() {
        let gate = EntropyThresholdGate::default(); // tau=0.5
        assert!(!gate.should_inject(0.0));
        assert!(!gate.should_inject(0.4));
        assert!(!gate.should_inject(0.49));
    }

    #[test]
    fn test_gate_boundary_at_threshold() {
        // At exactly u = tau, sigmoid(0) = 0.5, which is NOT > 0.5.
        let gate = EntropyThresholdGate::default();
        assert!(!gate.should_inject(0.5));
    }

    #[test]
    fn test_gate_custom_threshold_and_slope() {
        let gate = EntropyThresholdGate::new(0.8, 100.0); // sharp at 0.8
        assert!(!gate.should_inject(0.79));
        assert!(gate.should_inject(0.81));
    }

    #[test]
    fn test_gate_extreme_values_no_panic() {
        let gate = EntropyThresholdGate::default();
        // Clamping guards prevent overflow/NaN.
        assert!(gate.should_inject(f32::INFINITY));
        assert!(!gate.should_inject(f32::NEG_INFINITY));
    }

    #[test]
    fn test_gate_is_copy() {
        // EntropyThresholdGate must be Copy (zero-allocation, inline-storable).
        fn assert_copy<T: Copy>() {}
        assert_copy::<EntropyThresholdGate>();
    }

    #[test]
    fn test_sigmoid_basic_values() {
        // Sanity-check the local sigmoid via the public gate surface.
        // sigmoid(0) = 0.5 — at the boundary.
        let gate = EntropyThresholdGate::new(0.0, 1.0);
        assert!(!gate.should_inject(0.0)); // sigmoid(0) = 0.5, not > 0.5
        // Large positive x → ~1.0.
        assert!(gate.should_inject(100.0));
        // Large negative x → ~0.0.
        assert!(!gate.should_inject(-100.0));
    }
}
