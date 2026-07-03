//! ReMax Expected-Max-Over-m Aggregation — bonus-free exploration via objective
//! curvature.
//!
//! Distilled from Nishimori et al., *"Emergence of Exploration in Policy
//! Gradient Reinforcement Learning via Retrying"* (ICML 2026,
//! [arXiv:2606.00151](https://arxiv.org/abs/2606.00151)). Plan 374, Research
//! 373.
//!
//! # What this provides
//!
//! Two closed-form, O(K log K) operators parameterised by a continuous retry
//! count `m > 0`:
//!
//! - [`expected_max_over_m`] — the expected best-of-m value over m i.i.d. draws
//!   from a discrete policy (Proposition 3.2, Eq 4). This is the ReMax
//!   objective's inner expectation.
//! - [`expected_improvement`] — the Expected Improvement acquisition function
//!   (Proposition 4.3, Eq 10). Returns a scalar: how much one additional draw
//!   improves over the best of `m-1` others.
//!
//! Both are pure functions — deterministic, no RNG, no allocation beyond the
//! sort index buffer. The "exploration" lives in the *shape of the
//! objective*, not in sampling noise.
//!
//! # Continuous-m curvature control
//!
//! - `m = 1.0` → degenerates to the mean `E_{A~π}[q_A]` (standard RL).
//! - `m > 1.0` → flattens the objective near the optimum → sustains exploration.
//! - `m < 1.0` → sharpens the objective → accelerates convergence.
//! - `m → 0⁺` → degenerates to `min(q)` (extreme anti-exploration).
//! - `m → ∞` → degenerates to `max(q)` (pure exploitation).
//!
//! Empirical sweet spot on MinAtar: `m ∈ [1.2, 1.4]`.
//!
//! # No softmax
//!
//! Per the codebase constraint, this module never applies softmax internally.
//! The input `pi` is already a probability vector (from sigmoid gating or
//! normalisation). The operator transforms it via power-of-cumulative-mass,
//! not softmax.
//!
//! # Plan deviation note
//!
//! Plan 374 T1.2 specified `expected_improvement(R, ...) -> Vec<f32>`. The
//! paper's Eq 10 is a **scalar** formula — the `Vec<f32>` return type was
//! internally inconsistent with the specified formula. The per-action
//! `Q_plus` (used for the RePPO baseline `b_+(s)`) is a separate computation:
//! for each action `i`, evaluate [`expected_improvement`] with `R = q_i`. See
//! [`expected_improvement_per_action`]. This matches the paper's Appendix D
//! JAX code exactly.

use core::cmp::Ordering;

/// Lower clip for `(1 - C_j)` before exponentiation. Prevents `0^negative →
/// Inf` when `m < 1` and cumulative mass `C_j` reaches 1.0. The paper's JAX
/// code uses this exact value (App D, line 19).
const EPS: f32 = 1e-8;

/// Expected maximum over `m` i.i.d. draws from a discrete policy.
///
/// Given Q-values `q` for K actions and a policy `pi` (probabilities summing
/// to 1), computes the closed-form expected best-of-m value in O(K log K).
///
/// Source: Nishimori et al. ICML 2026, Proposition 3.2 (Eq 4).
///
/// # Arguments
///
/// - `pi` — policy probabilities, length K (need not be normalised; the
///   formula uses relative cumulative mass).
/// - `q` — Q-values, length K.
/// - `m` — continuous retry count, `m > 0`.
///
/// # Panics
///
/// Panics if `pi.len() != q.len()`, if `m <= 0.0`, or if the slices are empty.
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "remax_aggregation")]
/// # {
/// use katgpt_core::pruners::remax::expected_max_over_m;
///
/// // Two-armed bandit: q = [1.0, 0.0], pi = [0.5, 0.5], m = 2.
/// // Expected max of 2 draws = P(at least one draw hits arm 0) * 1.0
/// //                         = 1 - 0.5^2 = 0.75.
/// let val = expected_max_over_m(&[0.5, 0.5], &[1.0, 0.0], 2.0);
/// assert!((val - 0.75).abs() < 1e-6);
/// # }
/// ```
#[inline]
pub fn expected_max_over_m(pi: &[f32], q: &[f32], m: f32) -> f32 {
    let k = pi.len();
    assert!(k == q.len(), "pi and q must have the same length");
    assert!(k > 0, "pi and q must be non-empty");
    assert!(m > 0.0, "m must be positive (got {m})");

    if k == 1 {
        return q[0];
    }

    // Sort (q, pi) pairs by q descending.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap_or(Ordering::Equal));

    // q_sorted[0] is the maximum. Accumulate the telescoping sum.
    let mut result = q[idx[0]];
    let mut c = 0.0_f32; // cumulative policy mass of top-j actions
    for j in 0..k - 1 {
        c += pi[idx[j]];
        let tail = (1.0 - c).max(EPS);
        result += (q[idx[j + 1]] - q[idx[j]]) * tail.powf(m);
    }
    result
}

/// Expected Improvement: how much does one additional draw improve over the
/// best of `m-1` others?
///
/// Given a reference return `r`, policy `pi`, and Q-values `q`, returns the
/// scalar EI value. This is the `R_plus` term in RePPO (Algorithm 1, line 3).
///
/// Source: Nishimori et al. ICML 2026, Proposition 4.3 (Eq 10).
///
/// # Q-replacement
///
/// When computing EI for the return of a sampled action `a`, replace
/// `q[a]` with `r` *before* calling this function. This enforces
/// `v_a = (r - q_a)_+ = 0` by construction, preventing spurious
/// self-improvement from critic underestimation (paper §4.3).
///
/// # Arguments
///
/// - `r` — reference return (e.g. observed trajectory return, or an action's
///   Q-value for per-action EI).
/// - `pi` — policy probabilities, length K.
/// - `q` — Q-values, length K.
/// - `m` — continuous retry count, `m > 0`. The exponent used is `m - 1`.
///
/// # Panics
///
/// Same as [`expected_max_over_m`].
///
/// # Examples
///
/// ```
/// # #[cfg(feature = "remax_aggregation")]
/// # {
/// use katgpt_core::pruners::remax::expected_improvement;
///
/// // If r is below all Q-values, there is no improvement → EI = 0.
/// let ei = expected_improvement(-1.0, &[0.5, 0.5], &[1.0, 0.5], 2.0);
/// assert!(ei.abs() < 1e-6);
/// # }
/// ```
#[inline]
pub fn expected_improvement(r: f32, pi: &[f32], q: &[f32], m: f32) -> f32 {
    let k = pi.len();
    assert!(k == q.len(), "pi and q must have the same length");
    assert!(k > 0, "pi and q must be non-empty");
    assert!(m > 0.0, "m must be positive (got {m})");

    if k == 1 {
        return (r - q[0]).max(0.0);
    }

    let exponent = m - 1.0;

    // Sort by q descending. v = (r - q)_+ is then ascending.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap_or(Ordering::Equal));

    // v[j] = max(r - q_sorted[j], 0) — ascending because q is descending.
    let v0 = (r - q[idx[0]]).max(0.0);
    let mut result = v0;
    let mut c = 0.0_f32;
    let mut v_prev = v0;
    for j in 0..k - 1 {
        c += pi[idx[j]];
        let tail = (1.0 - c).max(EPS);
        let v_next = (r - q[idx[j + 1]]).max(0.0);
        result += (v_next - v_prev) * tail.powf(exponent);
        v_prev = v_next;
    }
    result
}

/// Per-action Expected Improvement: for each action `i`, computes
/// `EI_m(q_i; π, q)`.
///
/// This is the `Q_plus` term in RePPO (Algorithm 1, line 3), used to form the
/// action-independent baseline `b_+(s) = E_{a~π}[Q_plus(a)]`.
///
/// For Q-replacement: modify `q[action] ← R` before calling, then read
/// `out[action]` — it will reflect the replaced Q-value as the reference.
///
/// # Complexity
///
/// O(K²) after the initial O(K log K) sort. For K ≤ 128 this is ~16K float
/// operations — sub-microsecond on modern hardware.
///
/// # Arguments
///
/// - `pi` — policy probabilities, length K.
/// - `q` — Q-values, length K.
/// - `m` — continuous retry count, `m > 0`.
///
/// # Returns
///
/// `Vec<f32>` of length K, indexed by the **original** action order.
///
/// # Panics
///
/// Same as [`expected_max_over_m`].
pub fn expected_improvement_per_action(pi: &[f32], q: &[f32], m: f32) -> Vec<f32> {
    let k = pi.len();
    assert!(k == q.len(), "pi and q must have the same length");
    assert!(k > 0, "pi and q must be non-empty");
    assert!(m > 0.0, "m must be positive (got {m})");

    let mut out = vec![0.0_f32; k];
    expected_improvement_per_action_inplace(pi, q, m, &mut out);
    out
}

/// Zero-allocation variant of [`expected_improvement_per_action`].
///
/// Writes K values to `out`, indexed by the original action order. The sort
/// index buffer is still heap-allocated (O(K) `usize`); a fully stack-based
/// variant can be added if the G4 latency gate demands it.
///
/// # Panics
///
/// Panics if `out.len() != pi.len()`, plus the same assertions as
/// [`expected_max_over_m`].
pub fn expected_improvement_per_action_inplace(
    pi: &[f32],
    q: &[f32],
    m: f32,
    out: &mut [f32],
) {
    let k = pi.len();
    assert!(k == q.len(), "pi and q must have the same length");
    assert!(k > 0, "pi and q must be non-empty");
    assert!(k == out.len(), "out must have the same length as pi/q");
    assert!(m > 0.0, "m must be positive (got {m})");

    if k == 1 {
        out[0] = (q[0] - q[0]).max(0.0); // EI(q_0; q_0) = 0
        return;
    }

    let exponent = m - 1.0;

    // Sort by q descending.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_unstable_by(|&a, &b| q[b].partial_cmp(&q[a]).unwrap_or(Ordering::Equal));

    // Precompute weights w[j] = clip(1 - C[j], eps, 1)^(m-1) for j = 0..K-2.
    // These don't depend on the reference R, so we compute them once.
    let mut weights = vec![0.0_f32; k - 1];
    {
        let mut c = 0.0_f32;
        for j in 0..k - 1 {
            c += pi[idx[j]];
            weights[j] = (1.0 - c).max(EPS).powf(exponent);
        }
    }

    // For each original action i, compute EI with R = q[i].
    for i in 0..k {
        let r = q[i];
        // v[j] = max(r - q_sorted[j], 0) — ascending.
        let v0 = (r - q[idx[0]]).max(0.0);
        let mut ei = v0;
        let mut v_prev = v0;
        for j in 0..k - 1 {
            let v_next = (r - q[idx[j + 1]]).max(0.0);
            ei += (v_next - v_prev) * weights[j];
            v_prev = v_next;
        }
        out[i] = ei;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: dot product of two slices.
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }

    /// m = 1 should equal the mean E_{A~π}[q_A] = dot(pi, q).
    #[test]
    fn test_m1_equals_mean() {
        let pi = [0.2, 0.3, 0.1, 0.4];
        let q = [1.0, -0.5, 2.0, 0.3];
        let expected = dot(&pi, &q);
        let got = expected_max_over_m(&pi, &q, 1.0);
        assert!(
            (got - expected).abs() < 1e-5,
            "m=1 should give mean: expected {expected}, got {got}"
        );
    }

    /// m → 0⁺ should converge to min(q): (1-C)^0 = 1 telescopes to q_K.
    #[test]
    fn test_m0_converges_to_min() {
        let pi = [0.25, 0.25, 0.25, 0.25];
        let q = [3.0, 1.0, 2.0, 0.5];
        let min_q = q.iter().cloned().fold(f32::INFINITY, f32::min);
        let got = expected_max_over_m(&pi, &q, 1e-4);
        assert!(
            (got - min_q).abs() < 0.01,
            "m→0 should approach min(q)={min_q}, got {got}"
        );
    }

    /// m → ∞ should converge to max(q): (1-C)^∞ → 0, leaving only q_sorted[0].
    #[test]
    fn test_m_inf_converges_to_max() {
        let pi = [0.25, 0.25, 0.25, 0.25];
        let q = [3.0, 1.0, 2.0, 0.5];
        let max_q = q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let got = expected_max_over_m(&pi, &q, 50.0);
        assert!(
            (got - max_q).abs() < 1e-3,
            "m→∞ should approach max(q)={max_q}, got {got}"
        );
    }

    /// Deterministic two-armed bandit: μ=(0,1), p=π(a=1).
    /// J_m = 1 - (1-p)^m (paper §2.1).
    #[test]
    fn test_deterministic_bandit_closed_form() {
        let p = 0.3_f32;
        let pi = [1.0 - p, p]; // arm 0 (q=0) mass=1-p, arm 1 (q=1) mass=p
        let q = [0.0, 1.0];
        for m in [0.5_f32, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let expected = 1.0 - (1.0 - p).powf(m);
            let got = expected_max_over_m(&pi, &q, m);
            assert!(
                (got - expected).abs() < 1e-5,
                "m={m}: expected {expected}, got {got}"
            );
        }
    }

    /// If R < min(q), all v_i = 0, so EI = 0.
    #[test]
    fn test_ei_zero_when_r_below_all_q() {
        let pi = [0.3, 0.3, 0.4];
        let q = [1.0, 2.0, 0.5];
        let ei = expected_improvement(0.0, &pi, &q, 2.0);
        assert!(
            ei.abs() < 1e-6,
            "EI should be 0 when R < min(q), got {ei}"
        );
    }

    /// If R > max(q), EI is bounded and positive.
    #[test]
    fn test_ei_positive_when_r_above_all_q() {
        let pi = [0.3, 0.3, 0.4];
        let q = [1.0, 2.0, 0.5];
        let ei = expected_improvement(5.0, &pi, &q, 2.0);
        assert!(
            ei > 0.0,
            "EI should be positive when R > max(q), got {ei}"
        );
        // EI should be at most R - min(q) = 5.0 - 0.5 = 4.5 (best-case improvement).
        assert!(
            ei <= 5.0 - 0.5 + 1e-5,
            "EI should be bounded by R - min(q), got {ei}"
        );
    }

    /// Q-replacement: when q[action] is replaced by R, the v for that action
    /// becomes 0. The scalar EI should not increase (removing a positive term).
    #[test]
    fn test_q_replacement_reduces_ei() {
        let pi = [0.25, 0.25, 0.25, 0.25];
        let q = [3.0, 1.0, 2.0, 0.5];
        let r = 4.0_f32; // R above all q → all v_i > 0

        // Without replacement.
        let ei_no_replace = expected_improvement(r, &pi, &q, 2.0);

        // With replacement: q[0] ← R.
        let mut q_replaced = q;
        q_replaced[0] = r;
        let ei_replace = expected_improvement(r, &pi, &q_replaced, 2.0);

        assert!(
            ei_replace <= ei_no_replace + 1e-6,
            "Q-replacement should not increase EI: no_replace={ei_no_replace}, replace={ei_replace}"
        );
        // And specifically, v[0] = (R - R)_+ = 0 after replacement.
        // EI with replacement uses v = [0, R-q1, R-q2, R-q3] instead of [R-q0, R-q1, ...].
        // Since v[0] went from positive to 0, EI must decrease (the formula's
        // telescoping sum starts at v[0], which is now smaller).
        assert!(
            ei_replace < ei_no_replace,
            "Q-replacement should strictly decrease EI when R > q[action]"
        );
    }

    /// K = 1 edge case: expected_max returns q[0] directly.
    #[test]
    fn test_k1_max_returns_q_directly() {
        let val = expected_max_over_m(&[1.0], &[42.0], 3.0);
        assert_eq!(val, 42.0);
    }

    /// K = 1 edge case: EI returns (R - q[0])_+.
    #[test]
    fn test_k1_ei_returns_clamped_diff() {
        let ei_above = expected_improvement(5.0, &[1.0], &[3.0], 2.0);
        assert!((ei_above - 2.0).abs() < 1e-6);

        let ei_below = expected_improvement(1.0, &[1.0], &[3.0], 2.0);
        assert!(ei_below.abs() < 1e-6);
    }

    /// Numerical stability for m < 1 with near-degenerate pi (one arm ~1.0).
    /// Should not produce NaN or Inf.
    #[test]
    fn test_numerical_stability_m_below_1() {
        let pi = [0.999_999, 0.000_000_5, 0.000_000_5];
        let q = [1.0, 0.0, -1.0];
        for m in [0.1_f32, 0.3, 0.5, 0.9] {
            let val = expected_max_over_m(&pi, &q, m);
            assert!(val.is_finite(), "m={m}: got non-finite {val}");

            let ei = expected_improvement(2.0, &pi, &q, m);
            assert!(ei.is_finite(), "EI m={m}: got non-finite {ei}");
        }
    }

    /// Monotonicity: for fixed pi/q, expected_max is non-decreasing in m.
    /// More retries → higher (or equal) expected best.
    #[test]
    fn test_monotone_in_m() {
        let pi = [0.2, 0.3, 0.1, 0.4];
        let q = [1.0, -0.5, 2.0, 0.3];
        let ms = [0.3_f32, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        let vals: Vec<f32> = ms.iter().map(|&m| expected_max_over_m(&pi, &q, m)).collect();
        for i in 1..vals.len() {
            assert!(
                vals[i] >= vals[i - 1] - 1e-5,
                "Not monotone: m={:.1} → {:.6}, m={:.1} → {:.6}",
                ms[i - 1],
                vals[i - 1],
                ms[i],
                vals[i]
            );
        }
    }

    /// Per-action EI: each element should match the scalar EI with R = q_i.
    #[test]
    fn test_per_action_matches_scalar() {
        let pi = [0.2, 0.3, 0.1, 0.4];
        let q = [1.0, -0.5, 2.0, 0.3];
        let m = 2.0_f32;

        let per_action = expected_improvement_per_action(&pi, &q, m);
        assert_eq!(per_action.len(), q.len());

        for (i, &ei_i) in per_action.iter().enumerate() {
            let scalar = expected_improvement(q[i], &pi, &q, m);
            assert!(
                (ei_i - scalar).abs() < 1e-5,
                "per_action[{i}]={ei_i} != scalar EI(R=q[{i}]={})={scalar}",
                q[i]
            );
        }
    }

    /// Per-action EI: the baseline b_+(s) = E_{a~π}[Q_plus(a)] should be
    /// non-negative (it's a sum of non-negative EI values weighted by pi).
    #[test]
    fn test_baseline_non_negative() {
        let pi = [0.2, 0.3, 0.1, 0.4];
        let q = [1.0, -0.5, 2.0, 0.3];
        let m = 2.0_f32;

        let per_action = expected_improvement_per_action(&pi, &q, m);
        let baseline: f32 = pi.iter().zip(per_action.iter()).map(|(&p, &e)| p * e).sum();
        assert!(
            baseline >= -1e-6,
            "baseline should be non-negative, got {baseline}"
        );
    }

    /// Inplace variant should match the allocating variant bit-for-bit.
    #[test]
    fn test_inplace_matches_allocating() {
        let pi = [0.15, 0.35, 0.2, 0.3];
        let q = [0.7, -0.3, 1.5, 0.1];
        let m = 1.3_f32;

        let allocated = expected_improvement_per_action(&pi, &q, m);
        let mut inplace = vec![0.0; pi.len()];
        expected_improvement_per_action_inplace(&pi, &q, m, &mut inplace);

        for (i, (a, b)) in allocated.iter().zip(inplace.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "Mismatch at [{i}]: alloc={a}, inplace={b}");
        }
    }
}
