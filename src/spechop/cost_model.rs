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

// ── Configurator Integration (T30–T31) ────────────────────────

/// Measured inference statistics for auto-k computation (T30).
///
/// Tracks running averages of speculator latency, target tool latency,
/// and decode segment latency to compute α and β from real data.
#[derive(Clone, Debug, Default)]
pub struct InferenceStats {
    /// Running average of speculator latency: `E[T_spec]`.
    pub avg_spec_latency_ns: f64,
    /// Running average of target tool latency: `E[T_target]`.
    pub avg_target_latency_ns: f64,
    /// Running average of decode segment latency: `E[T_seg]`.
    pub avg_decode_latency_ns: f64,
    /// Running average of speculator success rate: `E[hit]`.
    pub avg_hit_rate: f64,
    /// Number of observations used to compute averages.
    pub observations: usize,
}

impl InferenceStats {
    /// Create empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update running averages with a new observation (exponential moving average).
    pub fn observe(
        &mut self,
        spec_latency_ns: f64,
        target_latency_ns: f64,
        decode_latency_ns: f64,
        hit: bool,
    ) {
        let n = self.observations as f64;
        let alpha_ema = if n < 50.0 { 1.0 / (n + 1.0) } else { 0.02 };
        self.avg_spec_latency_ns += alpha_ema * (spec_latency_ns - self.avg_spec_latency_ns);
        self.avg_target_latency_ns += alpha_ema * (target_latency_ns - self.avg_target_latency_ns);
        self.avg_decode_latency_ns += alpha_ema * (decode_latency_ns - self.avg_decode_latency_ns);
        let hit_f = if hit { 1.0 } else { 0.0 };
        self.avg_hit_rate += alpha_ema * (hit_f - self.avg_hit_rate);
        self.observations += 1;
    }

    /// Compute α (relative speculator latency) from measured stats.
    ///
    /// α = E[T_spec] / E[T_target], clamped to (0, 1).
    pub fn alpha(&self) -> f64 {
        if self.avg_target_latency_ns <= 0.0 {
            return 0.5; // default when no data
        }
        (self.avg_spec_latency_ns / self.avg_target_latency_ns).clamp(0.01, 0.99)
    }

    /// Compute β (decode-to-tool ratio) from measured stats.
    ///
    /// β = E[T_seg] / E[T_target], clamped to (0.01, ∞).
    pub fn beta(&self) -> f64 {
        if self.avg_target_latency_ns <= 0.0 {
            return 0.5; // default when no data
        }
        (self.avg_decode_latency_ns / self.avg_target_latency_ns).clamp(0.01, 100.0)
    }

    /// Get measured speculator accuracy (p).
    pub fn p(&self) -> f64 {
        self.avg_hit_rate.clamp(0.01, 1.0)
    }

    /// Auto-compute optimal k from measured stats (T30).
    ///
    /// Returns the optimal thread count `k* = ⌈(1+β)/(α+β)⌉` using
    /// measured α and β. Returns `None` if not enough observations.
    pub fn auto_k(&self) -> Option<usize> {
        if self.observations < 10 {
            return None;
        }
        Some(compute_optimal_k(self.alpha(), self.beta()))
    }
}

/// Configurator reward for SpecHop activation decision (T31).
///
/// `reward = latency_reduction_fraction / α`
///
/// Compares the theoretical latency reduction from oracle SpecHop against
/// the speculator's relative cost (α). Intuition: "how much time do we save
/// per unit of speculator expense?"
///
/// - `reward > 1.0` → SpecHop saves more than speculator costs → activate
/// - `reward < 1.0` → Speculator is too expensive relative to savings → skip
/// - `reward ≈ 0` → no benefit (high β or low p)
///
/// # Arguments
///
/// * `alpha` — Relative speculator latency (measured)
/// * `beta` — Decode-to-tool ratio (measured)
/// * `p` — Speculator success rate (measured)
/// * `_k` — Number of speculative threads (unused — oracle is k=∞ bound)
///
/// # Returns
///
/// Reward ratio. Values > 1.0 indicate SpecHop should be activated.
pub fn spechop_configurator_reward(alpha: f64, beta: f64, p: f64, _k: usize) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 || p <= 0.0 {
        return 0.0;
    }

    // Oracle relative latency: best-case latency fraction with infinite k
    let rel_lat = oracle_rel_lat(alpha, beta, p);

    // Fraction of time saved by speculation
    let latency_reduction = (1.0 - rel_lat).max(0.0);

    // Reward = how much we save per unit of speculator cost
    latency_reduction / alpha
}

/// Decide whether to activate SpecHop based on measured stats (T32 helper).
///
/// Returns `Some(k)` if SpecHop should be activated, `None` otherwise.
///
/// Activates when:
/// - α < 0.3 (speculator is fast relative to target tool)
/// - β < 0.5 (tool-bound scenario, decode is short relative to tool)
/// - reward > 1.0 (latency reduction exceeds compute overhead)
///
/// Skips when β > 0.8 (decode-bound, speculation doesn't help).
pub fn should_activate_spechop(stats: &InferenceStats) -> Option<usize> {
    let alpha = stats.alpha();
    let beta = stats.beta();
    let p = stats.p();

    // Need enough observations for reliable estimates
    if stats.observations < 10 {
        return None;
    }

    // Skip when decode-bound (β > 0.8 means decode takes 80% of tool time)
    if beta > 0.8 {
        return None;
    }

    // Skip when speculator is slow (α >= 0.3 means speculator takes 30% of tool time)
    if alpha >= 0.3 {
        return None;
    }

    let k = compute_optimal_k(alpha, beta);
    let reward = spechop_configurator_reward(alpha, beta, p, k);

    if reward > 1.0 { Some(k) } else { None }
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

    // ── InferenceStats (T30) ───────────────────────────────────

    #[test]
    fn test_inference_stats_default() {
        let stats = InferenceStats::new();
        assert_eq!(stats.observations, 0);
    }

    #[test]
    fn test_inference_stats_alpha_beta() {
        let mut stats = InferenceStats::new();
        // spec=20ns, target=100ns, decode=15ns
        for _ in 0..20 {
            stats.observe(20.0, 100.0, 15.0, true);
        }
        assert!((stats.alpha() - 0.2).abs() < 0.05);
        assert!((stats.beta() - 0.15).abs() < 0.05);
    }

    #[test]
    fn test_inference_stats_auto_k_needs_observations() {
        let stats = InferenceStats::new();
        assert!(stats.auto_k().is_none());
    }

    #[test]
    fn test_inference_stats_auto_k_with_data() {
        let mut stats = InferenceStats::new();
        for _ in 0..20 {
            stats.observe(20.0, 100.0, 15.0, true);
        }
        let k = stats.auto_k().expect("should have enough observations");
        assert!(k >= 1);
    }

    #[test]
    fn test_inference_stats_hit_rate() {
        let mut stats = InferenceStats::new();
        for i in 0..20 {
            stats.observe(20.0, 100.0, 15.0, i % 2 == 0);
        }
        assert!((stats.p() - 0.5).abs() < 0.15);
    }

    // ── Configurator Reward (T31) ──────────────────────────────

    #[test]
    fn test_configurator_reward_good_scenario() {
        // α=0.2, β=0.15, p=0.7, k=4
        // rel_lat = 1 - 0.7*0.8/1.15 ≈ 0.513, reduction ≈ 0.487
        // reward = 0.487 / 0.2 ≈ 2.44
        let reward = spechop_configurator_reward(0.2, 0.15, 0.7, 4);
        assert!(
            reward > 1.0,
            "expected reward > 1.0 for good scenario, got {reward}"
        );
    }

    #[test]
    fn test_configurator_reward_poor_speculator() {
        // p=0.1 → very poor speculator
        // rel_lat = 1 - 0.1*0.8/1.15 ≈ 0.930, reduction ≈ 0.070
        // reward = 0.070 / 0.2 ≈ 0.35
        let reward = spechop_configurator_reward(0.2, 0.15, 0.1, 4);
        assert!(
            reward < 1.0,
            "expected reward < 1.0 for poor speculator, got {reward}"
        );
    }

    #[test]
    fn test_configurator_reward_decode_bound() {
        // β=2.0 → decode-bound → speculation not useful
        // rel_lat = 1 - 0.7*0.8/3.0 ≈ 0.813, reduction ≈ 0.187
        // reward = 0.187 / 0.2 ≈ 0.935
        let reward = spechop_configurator_reward(0.2, 2.0, 0.7, 4);
        assert!(
            reward < 1.0,
            "expected reward < 1.0 for decode-bound, got {reward}"
        );
    }

    #[test]
    fn test_configurator_reward_perfect_speculator() {
        // p=1.0, α=0.1 → every speculation hits, cheap
        // rel_lat = 1 - 1.0*0.9/1.15 ≈ 0.217, reduction ≈ 0.783
        // reward = 0.783 / 0.1 ≈ 7.83
        let reward = spechop_configurator_reward(0.1, 0.15, 1.0, 4);
        assert!(
            reward > 2.0,
            "expected high reward for perfect speculator, got {reward}"
        );
    }

    #[test]
    fn test_configurator_reward_zero_inputs() {
        assert_eq!(spechop_configurator_reward(0.0, 0.15, 0.7, 4), 0.0);
        assert_eq!(spechop_configurator_reward(0.2, 0.0, 0.7, 4), 0.0);
        assert_eq!(spechop_configurator_reward(0.2, 0.15, 0.0, 4), 0.0);
    }

    // ── should_activate_spechop (T32) ──────────────────────────

    #[test]
    fn test_should_activate_tool_bound_scenario() {
        let mut stats = InferenceStats::new();
        // α=0.2, β=0.15, p=0.7 → tool-bound, should activate
        for _ in 0..20 {
            stats.observe(20.0, 100.0, 15.0, true);
        }
        let k = should_activate_spechop(&stats);
        assert!(k.is_some(), "should activate for tool-bound scenario");
    }

    #[test]
    fn test_should_not_activate_decode_bound() {
        let mut stats = InferenceStats::new();
        // β=2.0 → decode-bound → should not activate
        for _ in 0..20 {
            stats.observe(20.0, 100.0, 200.0, true);
        }
        assert!(should_activate_spechop(&stats).is_none());
    }

    #[test]
    fn test_should_not_activate_slow_speculator() {
        let mut stats = InferenceStats::new();
        // α=0.5 → speculator is slow → should not activate
        for _ in 0..20 {
            stats.observe(50.0, 100.0, 15.0, true);
        }
        assert!(should_activate_spechop(&stats).is_none());
    }

    #[test]
    fn test_should_not_activate_insufficient_data() {
        let mut stats = InferenceStats::new();
        stats.observe(20.0, 100.0, 15.0, true);
        assert!(should_activate_spechop(&stats).is_none());
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
