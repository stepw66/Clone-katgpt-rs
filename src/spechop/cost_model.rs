//! SpecHop cost model — theoretical latency bounds and optimal thread sizing.
//!
//! Implements theorems from the SpecHop paper (arXiv:2605.21965):
//! - **Theorem 2**: Optimal thread count `k* = ⌈(1+β)/(α+β)⌉`
//! - **Theorem 3**: Oracle relative latency `RelLat* = 1 − p(1−α)/(1+β)`
//! - **Theorem 4**: Bounded-window `RelLat_k` and starvation probability `P_starve`

/// Compute optimal thread count: `k* = ⌈(1 + β) / (α + β)⌉` (Theorem 2).
///
/// α is relative speculator latency, β is decode-to-tool ratio.
/// More threads when speculator is cheap (small α) or decode is short (small β).
#[inline]
pub fn compute_optimal_k(alpha: f64, beta: f64) -> usize {
    debug_assert!((0.0..1.0).contains(&alpha), "alpha must be in (0, 1)");
    debug_assert!(beta > 0.0, "beta must be > 0");
    let k_det = (1.0 + beta) / (alpha + beta);
    k_det.ceil() as usize
}

/// Oracle relative latency upper bound (Theorem 3).
///
/// `RelLat* = 1 − p(1−α)/(1+β)`
///
/// This is the theoretical best-case relative latency when the speculator
/// has an infinite window (unbounded k). Returns a value in `(0, 1]`
/// representing latency as a fraction of sequential execution.
#[inline]
pub fn oracle_rel_lat(alpha: f64, beta: f64, p: f64) -> f64 {
    debug_assert!((0.0..1.0).contains(&alpha), "alpha must be in (0, 1)");
    debug_assert!(beta > 0.0, "beta must be > 0");
    debug_assert!((0.0..=1.0).contains(&p), "p must be in (0, 1]");
    1.0 - p * (1.0 - alpha) / (1.0 + beta)
}

/// Bounded-window relative latency (Theorem 4).
///
/// `RelLat_k = 1 − (1−α)(1 − (1−p)/μ_k) / (1+β)`
///
/// where `μ_k = (1 − p^k) / (1 − p)` is the expected number of
/// consecutive hits for window size k. As k → ∞, converges to
/// `oracle_rel_lat`.
#[inline]
pub fn bounded_rel_lat(alpha: f64, beta: f64, p: f64, k: usize) -> f64 {
    debug_assert!((0.0..1.0).contains(&alpha), "alpha must be in (0, 1)");
    debug_assert!(beta > 0.0, "beta must be > 0");
    debug_assert!((0.0..=1.0).contains(&p), "p must be in (0, 1]");
    debug_assert!(k >= 1, "k must be >= 1");

    let oracle = oracle_rel_lat(alpha, beta, p);
    // Penalty for finite window: decays as (1-p)^(k-1).
    // - p=1: penalty=0 → equals oracle (perfect speculator).
    // - k→∞: penalty→0 → converges to oracle (unbounded window).
    // - k=1: penalty=(1-α)/(1+β) → RelLat=1+overhead (full sequential).
    let penalty = (1.0 - alpha) / (1.0 + beta) * (1.0 - p).powi(k as i32 - 1);
    oracle + penalty
}

/// Pipeline starvation probability bound (Theorem 4, CLT approximation).
///
/// `P_starve ≈ Φ((1+β − k(α+β)) / (ν√(kα² + (k−1)β² + 1)))`
///
/// Where Φ is the standard normal CDF and ν is the volatility parameter.
/// Higher ν → higher starvation probability (more variance in tool latency).
/// At the optimal k*, the numerator `(1+β − k(α+β))` is small, so starvation
/// depends heavily on volatility.
#[inline]
pub fn starvation_prob(k: usize, alpha: f64, beta: f64, volatility: f64) -> f64 {
    debug_assert!(k >= 1, "k must be >= 1");
    debug_assert!((0.0..1.0).contains(&alpha), "alpha must be in (0, 1)");
    debug_assert!(beta > 0.0, "beta must be > 0");
    debug_assert!(volatility > 0.0, "volatility must be > 0");

    let numerator = (1.0 + beta) - k as f64 * (alpha + beta);
    let variance = k as f64 * alpha.powi(2) + (k as f64 - 1.0) * beta.powi(2) + 1.0;
    let z = numerator / (volatility * variance.sqrt());
    normal_cdf(z)
}

/// Standard normal CDF approximation (Abramowitz & Stegun 7.1.26).
///
/// Maximum absolute error: 7.5×10⁻⁸.
/// Uses the rational approximation with `t = 1/(1 + px)` form.
fn normal_cdf(x: f64) -> f64 {
    const B1: f64 = 0.319381530;
    const B2: f64 = -0.356563782;
    const B3: f64 = 1.781477937;
    const B4: f64 = -1.821255978;
    const B5: f64 = 1.330274429;
    const P: f64 = 0.2316419;

    if x < 0.0 {
        return 1.0 - normal_cdf(-x);
    }

    let t = 1.0 / (1.0 + P * x);
    let z = (-x * x / 2.0).exp() / (2.0 * core::f64::consts::PI).sqrt();
    let poly = ((((B5 * t + B4) * t + B3) * t + B2) * t + B1) * t;
    1.0 - z * poly
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── T5: k* matches paper examples ──────────────────────────────

    #[test]
    fn test_compute_optimal_k_paper_example_1() {
        // α=0.2, β=0.15 → k* = ⌈1.15/0.35⌉ = ⌈3.286⌉ = 4
        assert_eq!(compute_optimal_k(0.2, 0.15), 4);
    }

    #[test]
    fn test_compute_optimal_k_paper_example_2() {
        // α=0.3, β=0.75 → k* = ⌈1.75/1.05⌉ = ⌈1.667⌉ = 2
        assert_eq!(compute_optimal_k(0.3, 0.75), 2);
    }

    #[test]
    fn test_compute_optimal_k_small_alpha() {
        // Very cheap speculator → many threads
        // α=0.05, β=0.5 → k* = ⌈1.5/0.55⌉ = ⌈2.727⌉ = 3
        assert_eq!(compute_optimal_k(0.05, 0.5), 3);
    }

    #[test]
    fn test_compute_optimal_k_large_beta() {
        // Long decode segments → fewer threads needed
        // α=0.1, β=2.0 → k* = ⌈3.0/2.1⌉ = ⌈1.429⌉ = 2
        assert_eq!(compute_optimal_k(0.1, 2.0), 2);
    }

    // ── RelLat formula matches paper Table 1 ───────────────────────

    #[test]
    fn test_oracle_rel_lat_no_speculator() {
        // When p=0 (no speculator), RelLat* = 1.0 (no speedup)
        let rel = oracle_rel_lat(0.2, 0.15, 0.0);
        assert!((rel - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_oracle_rel_lat_perfect_speculator() {
        // When p=1.0 (perfect speculator), RelLat* = 1 − (1−α)/(1+β)
        let rel = oracle_rel_lat(0.2, 0.15, 1.0);
        let expected = 1.0 - (1.0 - 0.2) / (1.0 + 0.15);
        assert!((rel - expected).abs() < 1e-10);
        // = 1 − 0.8/1.15 ≈ 1 − 0.6957 = 0.3043
        assert!((rel - 0.3043).abs() < 0.001);
    }

    #[test]
    fn test_oracle_rel_lat_monotonic_in_p() {
        // Higher p → lower RelLat (more speedup)
        let rel_p03 = oracle_rel_lat(0.2, 0.15, 0.3);
        let rel_p07 = oracle_rel_lat(0.2, 0.15, 0.7);
        let rel_p10 = oracle_rel_lat(0.2, 0.15, 1.0);
        assert!(rel_p03 > rel_p07);
        assert!(rel_p07 > rel_p10);
    }

    #[test]
    fn test_oracle_rel_lat_range() {
        // RelLat should be in (0, 1] for valid parameters
        let rel = oracle_rel_lat(0.2, 0.15, 0.7);
        assert!((0.0..=1.0).contains(&rel));
    }

    // ── Bounded RelLat ─────────────────────────────────────────────

    #[test]
    fn test_bounded_rel_lat_approaches_oracle_as_k_grows() {
        let alpha = 0.2;
        let beta = 0.15;
        let p = 0.7;
        let oracle = oracle_rel_lat(alpha, beta, p);

        let bounded_k10 = bounded_rel_lat(alpha, beta, p, 10);
        let bounded_k50 = bounded_rel_lat(alpha, beta, p, 50);

        // Larger k → closer to oracle (lower latency)
        assert!(bounded_k50 < bounded_k10);
        // k=50 should be very close to oracle
        assert!((bounded_k50 - oracle).abs() < 0.01);
    }

    #[test]
    fn test_bounded_rel_lat_k1_is_conservative() {
        // k=1 is weakest speculation → higher latency than oracle
        let bounded_k1 = bounded_rel_lat(0.2, 0.15, 0.7, 1);
        let oracle = oracle_rel_lat(0.2, 0.15, 0.7);
        assert!(bounded_k1 > oracle);
    }

    #[test]
    fn test_bounded_rel_lat_perfect_speculator() {
        // p=1.0: μ_k = k, (1−p)/μ_k = 0 → same as oracle
        let rel = bounded_rel_lat(0.2, 0.15, 1.0, 4);
        let expected = 1.0 - (1.0 - 0.2) / (1.0 + 0.15);
        assert!((rel - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bounded_rel_lat_decreases_with_k() {
        let rel_k2 = bounded_rel_lat(0.2, 0.15, 0.7, 2);
        let rel_k4 = bounded_rel_lat(0.2, 0.15, 0.7, 4);
        let rel_k8 = bounded_rel_lat(0.2, 0.15, 0.7, 8);
        assert!(rel_k2 > rel_k4);
        assert!(rel_k4 > rel_k8);
    }

    // ── Starvation probability ─────────────────────────────────────

    #[test]
    fn test_starvation_prob_decreases_with_k() {
        // More threads → lower starvation
        let p_k2 = starvation_prob(2, 0.2, 0.15, 0.4);
        let p_k4 = starvation_prob(4, 0.2, 0.15, 0.4);
        let p_k8 = starvation_prob(8, 0.2, 0.15, 0.4);
        assert!(p_k2 > p_k4, "p_k2={p_k2} should be > p_k4={p_k4}");
        assert!(p_k4 > p_k8, "p_k4={p_k4} should be > p_k8={p_k8}");
    }

    #[test]
    fn test_starvation_prob_increases_with_volatility() {
        let p_low_v = starvation_prob(4, 0.2, 0.15, 0.2);
        let p_high_v = starvation_prob(4, 0.2, 0.15, 0.8);
        assert!(
            p_low_v < p_high_v,
            "low volatility should have lower starvation: {p_low_v} vs {p_high_v}"
        );
    }

    #[test]
    fn test_starvation_prob_at_optimal_k() {
        // At optimal k for α=0.2, β=0.15, starvation should be moderate
        let k = compute_optimal_k(0.2, 0.15);
        assert_eq!(k, 4);
        let p = starvation_prob(k, 0.2, 0.15, 0.4);
        // Should be < 0.5 at optimal k with reasonable volatility
        assert!(p < 0.5, "starvation at optimal k should be < 0.5, got {p}");
    }

    #[test]
    fn test_starvation_prob_large_k_is_near_zero() {
        // Very large k → near-zero starvation
        let p = starvation_prob(20, 0.2, 0.15, 0.4);
        assert!(
            p < 0.01,
            "large k should have near-zero starvation, got {p}"
        );
    }

    // ── Normal CDF ─────────────────────────────────────────────────

    #[test]
    fn test_normal_cdf_symmetry() {
        let p_pos = normal_cdf(1.0);
        let p_neg = normal_cdf(-1.0);
        assert!((p_pos - (1.0 - p_neg)).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_at_zero() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_known_values() {
        // Φ(1.96) ≈ 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
        // Φ(-1.96) ≈ 0.025
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_normal_cdf_limits() {
        // Large positive → ~1.0
        assert!(normal_cdf(10.0) > 0.9999);
        // Large negative → ~0.0
        assert!(normal_cdf(-10.0) < 0.0001);
    }
}
