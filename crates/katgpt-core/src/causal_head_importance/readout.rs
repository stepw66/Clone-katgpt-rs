//! Span-level logit-difference readout (paper Eq 9, Plan 358, Research 362).
//!
//! The readout `m(x)` aggregates per-position logit differences between the
//! correct answer token (`a+`) and a counterfactual answer token (`a-`) across
//! the answer span. Earlier (more informative) positions are weighted more
//! heavily via exponential decay. This is the capability-expression scalar that
//! the activation/path-patching scores in [`super::patching`] normalize against.

/// Span-level logit-difference readout with exponential decay (paper Eq 9).
///
/// `m(x) = (1/Z) Σ_{j∈A} λ^j · (z_j[a+_j] − z_j[a-_j])`, `Z = Σ λ^j`, `λ = 0.9`.
///
/// Aggregates per-position logit differences between correct (`a+`) and
/// counterfactual (`a-`) answer tokens across the answer span `A`, weighting
/// earlier (more informative) positions more heavily via exponential decay.
///
/// Why logit-diff not probability: approximately linear in the residual stream
/// and monotone in the underlying capability, avoiding softmax-saturation and
/// probability measurement-floor effects (paper §4.1, Zhang & Nanda 2024).
#[derive(Clone, Copy, Debug)]
pub struct SpanLogitDiffReadout {
    /// Exponential decay per answer position. Paper default: 0.9.
    pub lambda: f32,
}

impl Default for SpanLogitDiffReadout {
    fn default() -> Self {
        Self { lambda: 0.9 }
    }
}

impl SpanLogitDiffReadout {
    /// Compute the readout `m(x)` from per-position `(logit_correct, logit_counterfactual)` pairs.
    ///
    /// `per_position`: `[(z_j[a+_j], z_j[a-_j]); |A|]` in answer order (earliest first).
    /// Returns `m(x) ∈ ℝ`. Larger = stronger capability expression.
    #[inline]
    pub fn readout(&self, per_position: &[(f32, f32)]) -> f32 {
        // m(x) = (1/Z) Σ λ^j (z_j[a+] − z_j[a-_j]), Z = Σ λ^j
        let mut numer = 0.0f32;
        let mut denom = 0.0f32;
        let mut w = 1.0f32;
        for &(correct, counterfactual) in per_position {
            numer += w * (correct - counterfactual);
            denom += w;
            w *= self.lambda;
        }
        if denom > 0.0 { numer / denom } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_position_returns_difference() {
        // |A| = 1: m(x) = z[a+] − z[a-] (Z = 1, weight = 1).
        let r = SpanLogitDiffReadout::default();
        let m = r.readout(&[(2.0, 0.5)]);
        // numer = 1.0 * 1.5 = 1.5, denom = 1.0 → 1.5
        assert_eq!(m, 1.5);
    }

    #[test]
    fn two_positions_lambda_one_is_mean() {
        // λ = 1.0: equal weighting → mean of the differences.
        let r = SpanLogitDiffReadout { lambda: 1.0 };
        let m = r.readout(&[(1.0, -1.0), (3.0, 1.0)]);
        // diffs: 2.0, 2.0 → mean 2.0
        assert!((m - 2.0).abs() < 1e-6);
    }

    #[test]
    fn lambda_zero_is_first_position_only() {
        // λ = 0.0: only j=0 contributes (weight 1 at j=0, then w becomes 0 for j≥1).
        let r = SpanLogitDiffReadout { lambda: 0.0 };
        let m = r.readout(&[(5.0, 1.0), (100.0, 0.0)]);
        // first diff = 4.0, all others zeroed → 4.0
        assert!((m - 4.0).abs() < 1e-6);
    }

    #[test]
    fn empty_span_returns_zero() {
        // Empty span: Z = 0 → return 0.0 (no division by zero).
        let r = SpanLogitDiffReadout::default();
        assert_eq!(r.readout(&[]), 0.0);
    }

    #[test]
    fn decay_weights_earlier_positions_more() {
        // λ = 0.5, diffs [10.0, 0.0]: m = (1·10 + 0.5·0)/(1 + 0.5) = 10/1.5.
        let r = SpanLogitDiffReadout { lambda: 0.5 };
        let m = r.readout(&[(10.0, 0.0), (0.0, 0.0)]);
        assert!((m - (10.0_f32 / 1.5)).abs() < 1e-6);
    }
}
