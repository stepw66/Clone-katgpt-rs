//! Activation/path-patching importance scores (paper Eq 10–11, Plan 358).
//!
//! Two normalized causal-importance scores sharing the same formula structure;
//! they differ only in *what* the caller patches to produce `m_patched`:
//!
//! - [`direct_effect_importance`] (Eq 10, receiver): the head's *own output* is
//!   swapped for its corrupted-run value (activation patching).
//! - [`indirect_effect_importance`] (Eq 11, sender): an upstream head's recorded
//!   activations are substituted at a downstream receiver's input (path patching).
//!
//! Both are `#[inline]`, allocation-free measurement functions. The caller owns
//! the patched forward pass that produces `m_patched`.

/// Normalized causal-importance score for a single head (or any patchable unit)
/// via activation patching (paper Eq 10).
///
/// `IE = (m(x) − m(x; O ← O(x'))) / (m(x) − m(x'))   ∈ [0, 1]`
///
/// - `m_clean`: readout on the clean input.
/// - `m_corrupt`: readout on the corrupted input (answer replaced by distractor).
/// - `m_patched`: readout when the head's output is replaced by its corrupted-run
///   value while all other components remain at their clean state.
///
/// `IE ≈ 0` → head is dispensable (safely convertible to a cheaper mechanism);
/// `IE ≈ 1` → head alone collapses the capability (load-bearing).
///
/// The caller is responsible for the "freeze downstream attention to clean
/// values" refinement (paper §4.1 / Appendix C.1) — i.e. the patched forward
/// pass must route the patched signal only through the residual stream + MLPs,
/// not through downstream attention that could compensate. This probe is the
/// *measurement*; the patched forward pass is supplied by the caller via a
/// closure.
#[inline]
pub fn direct_effect_importance(m_clean: f32, m_corrupt: f32, m_patched: f32) -> f32 {
    let denom = m_clean - m_corrupt;
    if denom.abs() < f32::EPSILON {
        // m_clean ≈ m_corrupt: the corruption itself doesn't move the readout,
        // so per-head necessity is undefined. Return 0 (treat as not load-bearing
        // for this capability; the capability isn't expressed in this input pair).
        return 0.0;
    }
    let ie = (m_clean - m_patched) / denom;
    // Numerical guard: IE should be in [0, 1] by construction when the patched
    // forward pass is a true substitution. Clamp to handle fp noise.
    ie.clamp(0.0, 1.0)
}

/// One-step-back indirect-effect (sender) score via path patching (paper Eq 11).
///
/// For an upstream head `u`, run the corrupted input and record the activations
/// it sends to a receiver head `r`. Then run an otherwise-clean forward pass
/// substituting only those recorded activations at `r`'s input. The normalized
/// drop in the readout is the indirect contribution of `u` through `r`:
///
/// `IE_send(u, r) = (m_clean − m_path_patched(u→r)) / (m_clean − m_corrupt)`
///
/// A head can be causally important without writing the signal directly — by
/// feeding a receiver. Iterating this (promoting senders to receivers, repeating)
/// traces the shallow circuit; paper notes long-context retrieval converges in
/// ~2 rounds.
///
/// `direct_effect_importance` and `indirect_effect_importance` share the same
/// formula structure; the difference is *what* is patched (head output vs the
/// pathway into a downstream receiver). Both callers supply `m_patched` from
/// their own forward-pass machinery.
#[inline]
pub fn indirect_effect_importance(m_clean: f32, m_corrupt: f32, m_path_patched: f32) -> f32 {
    // Same normalization as direct_effect_importance; the semantic difference
    // is in how m_path_patched is produced (path patching vs activation patching).
    direct_effect_importance(m_clean, m_corrupt, m_path_patched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispensable_head_has_zero_ie() {
        // m_patched = m_clean → head output irrelevant → IE = 0.
        let ie = direct_effect_importance(5.0, 1.0, 5.0);
        assert_eq!(ie, 0.0);
    }

    #[test]
    fn load_bearing_head_has_ie_one() {
        // m_patched = m_corrupt → patching alone collapses to corrupt readout → IE = 1.
        let ie = direct_effect_importance(5.0, 1.0, 1.0);
        assert!((ie - 1.0).abs() < 1e-6);
    }

    #[test]
    fn undefined_when_clean_equals_corrupt() {
        // m_clean = m_corrupt → denom ≈ 0 → IE = 0 (safe default).
        let ie = direct_effect_importance(3.0, 3.0, 2.0);
        assert_eq!(ie, 0.0);
    }

    #[test]
    fn intermediate_is_linear_interpolation() {
        // m_clean = 4, m_corrupt = 0, m_patched = 3 → IE = (4-3)/(4-0) = 0.25.
        let ie = direct_effect_importance(4.0, 0.0, 3.0);
        assert!((ie - 0.25).abs() < 1e-6);
    }

    #[test]
    fn indirect_shares_formula_with_direct() {
        // Same scalars → same result; semantics differ only in how m_patched
        // is produced by the caller.
        let direct = direct_effect_importance(6.0, 2.0, 5.0);
        let indirect = indirect_effect_importance(6.0, 2.0, 5.0);
        assert_eq!(direct, indirect);
        // (6-5)/(6-2) = 0.25
        assert!((direct - 0.25).abs() < 1e-6);
    }

    #[test]
    fn fp_noise_clamped_to_unit_interval() {
        // Slight overshoot beyond [0,1] from fp noise is clamped, not propagated.
        // m_patched slightly below m_corrupt → IE slightly > 1 → clamp to 1.
        let ie = direct_effect_importance(5.0, 1.0, 0.999);
        assert_eq!(ie, 1.0);
        // m_patched slightly above m_clean → IE slightly < 0 → clamp to 0.
        let ie = direct_effect_importance(5.0, 1.0, 5.001);
        assert_eq!(ie, 0.0);
    }
}
