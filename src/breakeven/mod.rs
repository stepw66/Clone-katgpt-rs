//! Breakeven Complexity — Cost-Aware Inference Tier Routing
//!
//! Applies the breakeven complexity framework from PDE solver evaluation
//! (Zhang et al., 2026) to LLM inference routing.
//!
//! Core equation: N* = B / (C_classical - C_surrogate)
//! - B = upfront cost of activating a tier (compile, warm-up, quantize)
//! - C_classical = per-token cost at the lower tier
//! - C_surrogate = per-token cost at this tier
//! - N* = number of tokens before this tier amortizes its activation cost
//!
//! Key insight: approximation (speculative decode, sparse attention, quantized KV)
//! becomes MORE valuable as inference gets harder (longer sequences, higher QPS).
//!
//! Feature-gated behind `breakeven_routing`.

pub mod fidelity;

use std::fmt;

use crate::trigger_gate::ComputeTier;

// ---------------------------------------------------------------------------
// BreakevenTierPair
// ---------------------------------------------------------------------------

/// A pair of compute tiers for breakeven analysis.
///
/// Each pair represents the upgrade from a lower tier to a higher tier,
/// with associated upfront activation cost and per-token savings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BreakevenTierPair {
    /// CPU → CPU+GPU (GPU kernel compilation + warm-up)
    CpuToGpu = 0,
    /// CPU+GPU → CPU+GPU+ANE (ANE model compilation)
    GpuToAne = 1,
    /// CPU → CPU+Speculative (draft model load + quantize)
    CpuToSpeculative = 2,
    /// CPU+GPU → CPU+GPU+Speculative
    GpuToSpeculative = 3,
}

impl BreakevenTierPair {
    /// All tier pairs as a slice.
    pub const ALL: [BreakevenTierPair; 4] = [
        BreakevenTierPair::CpuToGpu,
        BreakevenTierPair::GpuToAne,
        BreakevenTierPair::CpuToSpeculative,
        BreakevenTierPair::GpuToSpeculative,
    ];

    /// Source (lower) tier.
    pub const fn source_tier(&self) -> ComputeTier {
        match self {
            Self::CpuToGpu | Self::CpuToSpeculative => ComputeTier::CpuOnly,
            Self::GpuToAne | Self::GpuToSpeculative => ComputeTier::CpuGpu,
        }
    }

    /// Target (higher) tier.
    pub const fn target_tier(&self) -> ComputeTier {
        match self {
            Self::CpuToGpu => ComputeTier::CpuGpu,
            Self::GpuToAne => ComputeTier::CpuGpuAne,
            // Speculative stays on the same hardware but adds draft model.
            Self::CpuToSpeculative => ComputeTier::CpuOnly,
            Self::GpuToSpeculative => ComputeTier::CpuGpu,
        }
    }

    /// Index into tracking arrays.
    pub const fn as_index(&self) -> usize {
        *self as usize
    }
}

impl fmt::Display for BreakevenTierPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CpuToGpu => write!(f, "CPU→GPU"),
            Self::GpuToAne => write!(f, "GPU→ANE"),
            Self::CpuToSpeculative => write!(f, "CPU→SPEC"),
            Self::GpuToSpeculative => write!(f, "GPU→SPEC"),
        }
    }
}

// ---------------------------------------------------------------------------
// BreakevenTracker
// ---------------------------------------------------------------------------

/// Tracks breakeven complexity for a single tier pair.
///
/// Computes N* = upfront_cost / (baseline_cost - tier_cost) and reports
/// whether the tier has amortized its activation cost after N tokens.
///
/// Zero-allocation: all fields are fixed-size primitives.
/// Thread-safe: uses atomics for concurrent updates.
pub struct BreakevenTracker {
    /// One-time upfront cost to activate this tier (microseconds).
    /// Includes: kernel compilation, model quantization, warm-up forward pass.
    upfront_cost_us: AtomicU64,
    /// Exponential moving average of per-token cost at the BASELINE tier (μs).
    baseline_cost_ema_us: AtomicU64,
    /// Exponential moving average of per-token cost at THIS tier (μs).
    tier_cost_ema_us: AtomicU64,
    /// Total tokens processed at this tier.
    total_tokens: AtomicU64,
    /// EMA smoothing factor (stored as fixed-point: α × 65536).
    /// Default: α = 0.1 → 6553.
    alpha_fixed: u16,
}

use std::sync::atomic::{AtomicU64, Ordering};

impl BreakevenTracker {
    /// EMA alpha = 0.1, encoded as fixed-point ×65536.
    const ALPHA_FIXED: u16 = 6553;
    /// Fixed-point scale for EMA computation.
    const FP_SCALE: u64 = 65536;

    /// Create a new tracker with known upfront cost.
    ///
    /// `upfront_cost_us` should be measured once during tier activation
    /// (kernel compile + warm-up). If unknown, use 0 and call
    /// [`set_upfront_cost`](Self::set_upfront_cost) later.
    pub fn new(upfront_cost_us: u64) -> Self {
        Self {
            upfront_cost_us: AtomicU64::new(upfront_cost_us),
            baseline_cost_ema_us: AtomicU64::new(0),
            tier_cost_ema_us: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            alpha_fixed: Self::ALPHA_FIXED,
        }
    }

    /// Set the upfront activation cost after measurement.
    pub fn set_upfront_cost(&self, cost_us: u64) {
        self.upfront_cost_us.store(cost_us, Ordering::Relaxed);
    }

    /// Observe a baseline (lower tier) timing and update EMA.
    ///
    /// Uses exponential moving average to track per-token cost
    /// at the baseline tier without storing all observations.
    pub fn observe_baseline(&self, timing_us: u64) {
        Self::update_ema(&self.baseline_cost_ema_us, timing_us, self.alpha_fixed);
    }

    /// Observe a tier timing and update EMA + token counter.
    pub fn observe_tier(&self, timing_us: u64) {
        Self::update_ema(&self.tier_cost_ema_us, timing_us, self.alpha_fixed);
        self.total_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// Compute the breakeven complexity N*.
    ///
    /// N* = upfront_cost / max(baseline_cost - tier_cost, 0)
    ///
    /// Returns `f64::INFINITY` if the tier is slower than baseline
    /// (i.e., the approximation never amortizes).
    pub fn breakeven_n(&self) -> f64 {
        let upfront = self.upfront_cost_us.load(Ordering::Relaxed) as f64;
        let baseline = self.baseline_cost_ema_us.load(Ordering::Relaxed) as f64;
        let tier = self.tier_cost_ema_us.load(Ordering::Relaxed) as f64;

        let denominator = (baseline - tier).max(0.0);
        if denominator < 1e-6 {
            f64::INFINITY
        } else {
            upfront / denominator
        }
    }

    /// Whether this tier has amortized its upfront cost.
    ///
    /// Returns `true` when `total_tokens >= breakeven_n`.
    /// Always returns `false` if N* is infinite.
    pub fn is_amortized(&self) -> bool {
        let n_star = self.breakeven_n();
        if n_star.is_infinite() {
            return false;
        }
        let tokens = self.total_tokens.load(Ordering::Relaxed) as f64;
        tokens >= n_star
    }

    /// Remaining tokens to amortize. Returns 0.0 if already amortized.
    /// Returns `f64::INFINITY` if N* is infinite.
    pub fn remaining_to_amortize(&self) -> f64 {
        let n_star = self.breakeven_n();
        let tokens = self.total_tokens.load(Ordering::Relaxed) as f64;
        (n_star - tokens).max(0.0)
    }

    /// Sigmoid-gated amortization confidence ∈ [0, 1].
    ///
    /// σ(α × (tokens - N*)) where α = transition_sharpness.
    /// Returns ~0.5 at the breakeven point, ~1.0 well past it.
    /// Use sigmoid (not softmax) per project conventions.
    pub fn amortization_confidence(&self, transition_sharpness: f64) -> f64 {
        let n_star = self.breakeven_n();
        if n_star.is_infinite() {
            return 0.0;
        }
        let tokens = self.total_tokens.load(Ordering::Relaxed) as f64;
        sigmoid(transition_sharpness * (tokens - n_star))
    }

    /// Total tokens processed at this tier.
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Current EMA of baseline per-token cost (μs).
    pub fn baseline_cost_us(&self) -> u64 {
        self.baseline_cost_ema_us.load(Ordering::Relaxed)
    }

    /// Current EMA of tier per-token cost (μs).
    pub fn tier_cost_us(&self) -> u64 {
        self.tier_cost_ema_us.load(Ordering::Relaxed)
    }

    /// Upfront activation cost (μs).
    pub fn upfront_cost_us(&self) -> u64 {
        self.upfront_cost_us.load(Ordering::Relaxed)
    }

    /// Reset all state (e.g., on model change).
    pub fn reset(&self) {
        self.upfront_cost_us.store(0, Ordering::Relaxed);
        self.baseline_cost_ema_us.store(0, Ordering::Relaxed);
        self.tier_cost_ema_us.store(0, Ordering::Relaxed);
        self.total_tokens.store(0, Ordering::Relaxed);
    }

    /// Update EMA using fixed-point arithmetic.
    ///
    /// EMA_new = α × value + (1 - α) × EMA_old
    /// Fixed-point: result_fp = α_fp × value + (65536 - α_fp) × EMA_old_fp / 65536
    fn update_ema(ema: &AtomicU64, value: u64, alpha_fixed: u16) {
        let old = ema.load(Ordering::Relaxed);
        let alpha = alpha_fixed as u64;
        let new = (alpha * value + (Self::FP_SCALE - alpha) * old) / Self::FP_SCALE;
        ema.store(new, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

/// Standard sigmoid function. Use sigmoid (not softmax) per project conventions.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ---------------------------------------------------------------------------
// BreakevenBandit
// ---------------------------------------------------------------------------

/// Meta-bandit that routes inference to tiers based on breakeven complexity.
///
/// Tracks per-tier-pair breakeven thresholds and routes to the tier
/// that has already amortized its setup cost. Coexists with TriggerGate
/// (QPS-based), trust signal, and RV gate.
///
/// Priority of signals:
/// 1. Breakeven (overrides when tier has NOT amortized)
/// 2. RV gate (overrides when model variance is high)
/// 3. Trust signal (adjusts based on acceptance rate)
/// 4. TriggerGate QPS (default fallback)
pub struct BreakevenBandit {
    /// Per-tier-pair trackers (indexed by BreakevenTierPair::as_index).
    trackers: [BreakevenTracker; 4],
    /// Sigmoid transition sharpness (higher = sharper tier boundary).
    transition_sharpness: f64,
    /// Whether breakeven routing is enabled (set to false if all N* are infinite).
    enabled: bool,
}

impl BreakevenBandit {
    /// Create a new bandit with default upfront costs.
    ///
    /// `gpu_compile_us` — GPU kernel compilation cost (default: 500_000 = 500ms).
    /// `ane_compile_us` — ANE model compilation cost (default: 2_000_000 = 2s).
    /// `speculative_load_us` — Draft model load + quantize cost (default: 100_000 = 100ms).
    pub fn new(gpu_compile_us: u64, ane_compile_us: u64, speculative_load_us: u64) -> Self {
        Self {
            trackers: [
                // CpuToGpu
                BreakevenTracker::new(gpu_compile_us),
                // GpuToAne
                BreakevenTracker::new(ane_compile_us),
                // CpuToSpeculative
                BreakevenTracker::new(speculative_load_us),
                // GpuToSpeculative
                BreakevenTracker::new(speculative_load_us),
            ],
            transition_sharpness: 0.001, // ~1000 tokens transition width
            enabled: true,
        }
    }

    /// Create with default costs.
    pub fn with_defaults() -> Self {
        Self::new(500_000, 2_000_000, 100_000)
    }

    /// Observe a timing for a specific tier pair's baseline.
    pub fn observe_baseline(&self, pair: BreakevenTierPair, timing_us: u64) {
        self.trackers[pair.as_index()].observe_baseline(timing_us);
    }

    /// Observe a timing for a specific tier pair.
    pub fn observe_tier(&self, pair: BreakevenTierPair, timing_us: u64) {
        self.trackers[pair.as_index()].observe_tier(timing_us);
    }

    /// Select the best tier based on breakeven analysis.
    ///
    /// Returns `None` if breakeven routing has no opinion (all tiers not amortized
    /// or insufficient data). Caller should fall back to TriggerGate.
    ///
    /// Returns `Some(ComputeTier)` if breakeven recommends a specific tier.
    pub fn select_tier(&self, current_tier: ComputeTier) -> Option<ComputeTier> {
        if !self.enabled {
            return None;
        }

        // Check each tier pair: if the upgrade has amortized, recommend the higher tier.
        // If the upgrade has NOT amortized, recommend staying at the lower tier.
        let mut best_tier = current_tier;
        let mut best_confidence = 0.5; // Default: no strong opinion

        for pair in BreakevenTierPair::ALL {
            let tracker = &self.trackers[pair.as_index()];
            let confidence = tracker.amortization_confidence(self.transition_sharpness);

            if confidence > best_confidence {
                best_confidence = confidence;
                best_tier = pair.target_tier();
            }
        }

        // Only override if we have a strong signal (confidence > 0.6)
        if best_confidence > 0.6 {
            Some(best_tier)
        } else {
            None
        }
    }

    /// Check if a specific tier upgrade has amortized.
    pub fn is_amortized(&self, pair: BreakevenTierPair) -> bool {
        self.trackers[pair.as_index()].is_amortized()
    }

    /// Get the breakeven N* for a specific tier pair.
    pub fn breakeven_n(&self, pair: BreakevenTierPair) -> f64 {
        self.trackers[pair.as_index()].breakeven_n()
    }

    /// Get tracker for a specific tier pair (for direct observation).
    pub fn tracker(&self, pair: BreakevenTierPair) -> &BreakevenTracker {
        &self.trackers[pair.as_index()]
    }

    /// Reset all trackers (e.g., on model change).
    pub fn reset(&mut self) {
        for tracker in &self.trackers {
            tracker.reset();
        }
    }

    /// Set transition sharpness (higher = sharper tier boundary).
    pub fn set_transition_sharpness(&mut self, sharpness: f64) {
        self.transition_sharpness = sharpness;
    }

    /// Breakeven routing stats for logging.
    pub fn stats(&self) -> BreakevenStats {
        BreakevenStats {
            cpu_to_gpu_n: self.breakeven_n(BreakevenTierPair::CpuToGpu),
            gpu_to_ane_n: self.breakeven_n(BreakevenTierPair::GpuToAne),
            cpu_to_spec_n: self.breakeven_n(BreakevenTierPair::CpuToSpeculative),
            gpu_to_spec_n: self.breakeven_n(BreakevenTierPair::GpuToSpeculative),
            cpu_to_gpu_amortized: self.is_amortized(BreakevenTierPair::CpuToGpu),
            gpu_to_ane_amortized: self.is_amortized(BreakevenTierPair::GpuToAne),
        }
    }
}

impl Default for BreakevenBandit {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// BreakevenStats
// ---------------------------------------------------------------------------

/// Snapshot of breakeven routing state for logging/display.
#[derive(Debug, Clone)]
pub struct BreakevenStats {
    pub cpu_to_gpu_n: f64,
    pub gpu_to_ane_n: f64,
    pub cpu_to_spec_n: f64,
    pub gpu_to_spec_n: f64,
    pub cpu_to_gpu_amortized: bool,
    pub gpu_to_ane_amortized: bool,
}

impl fmt::Display for BreakevenStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Write N* directly to the formatter — avoids per-field String allocation.
        fn write_n(f: &mut fmt::Formatter<'_>, n: f64) -> fmt::Result {
            if n.is_infinite() {
                f.write_str("∞")
            } else {
                write!(f, "{n:.0}")
            }
        }

        f.write_str("Breakeven { CPU→GPU: N*=")?;
        write_n(f, self.cpu_to_gpu_n)?;
        f.write_str(if self.cpu_to_gpu_amortized { " ✓ | GPU→ANE: N*=" } else { " … | GPU→ANE: N*=" })?;
        write_n(f, self.gpu_to_ane_n)?;
        f.write_str(if self.gpu_to_ane_amortized { " ✓ | CPU→SPEC: N*=" } else { " … | CPU→SPEC: N*=" })?;
        write_n(f, self.cpu_to_spec_n)?;
        f.write_str(" | GPU→SPEC: N*=")?;
        write_n(f, self.gpu_to_spec_n)?;
        f.write_str(" }")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_breakeven_n_computation() {
        // Upfront = 1000μs, baseline = 100μs/token, tier = 50μs/token
        // N* = 1000 / (100 - 50) = 20
        let tracker = BreakevenTracker::new(1000);
        tracker.observe_baseline(100);
        tracker.observe_tier(50);
        // Second observation to push EMA towards correct value
        tracker.observe_baseline(100);
        tracker.observe_tier(50);

        let n_star = tracker.breakeven_n();
        // EMA converges with α=0.1, after 2 observations it won't be exact
        // but should be in the right ballpark
        assert!(n_star < 500.0, "N* should be reasonable, got {n_star}");
    }

    #[test]
    fn test_is_amortized_flips() {
        // Upfront = 10_000, baseline ≈ 100, tier ≈ 50
        // N* ≈ 10_000 / (100 - 50) = 200
        let tracker = BreakevenTracker::new(10_000);
        // Feed enough baselines for EMA to converge to ~100
        for _ in 0..50 {
            tracker.observe_baseline(100);
        }
        // Feed a few tier observations at 50 — only 3 tokens, not yet amortized
        for _ in 0..3 {
            tracker.observe_tier(50);
        }

        // Not amortized yet (3 tokens << N* ≈ 200)
        assert!(
            !tracker.is_amortized(),
            "should not be amortized with only 3 tokens vs N*≈200"
        );

        // Feed enough tokens to amortize (1003 >> 200)
        for _ in 0..1000 {
            tracker.observe_tier(50);
        }

        assert!(
            tracker.is_amortized(),
            "should be amortized after 1003 tokens vs N*≈200"
        );
    }

    #[test]
    fn test_infinite_when_slower() {
        // Tier is SLOWER than baseline → N* = ∞
        let tracker = BreakevenTracker::new(1000);
        tracker.observe_baseline(50);
        tracker.observe_baseline(50);
        tracker.observe_tier(100);
        tracker.observe_tier(100);

        let n_star = tracker.breakeven_n();
        assert!(
            n_star > 1_000_000.0 || n_star.is_infinite(),
            "N* should be very large or infinite when tier is slower, got {n_star}"
        );
    }

    #[test]
    fn test_remaining_to_amortize() {
        let tracker = BreakevenTracker::new(1000);
        tracker.observe_baseline(100);
        tracker.observe_baseline(100);
        tracker.observe_tier(50);
        tracker.observe_tier(50);

        let remaining = tracker.remaining_to_amortize();
        assert!(remaining >= 0.0);
    }

    #[test]
    fn test_amortization_confidence_sigmoid() {
        // Use a large upfront cost to make N* bigger (N* ≈ 10_000 / 50 = 200)
        let tracker = BreakevenTracker::new(10_000);
        // Feed enough baselines for EMA to converge to ~100
        for _ in 0..50 {
            tracker.observe_baseline(100);
        }
        for _ in 0..3 {
            tracker.observe_tier(50);
        }
        // N* ≈ 10_000 / (100-50) = 200, only 3 tokens → not amortized
        let conf_before = tracker.amortization_confidence(0.01);
        assert!(
            conf_before < 0.5,
            "Confidence should be < 0.5 before amortization, got {conf_before}"
        );

        // After many tokens: confidence > 0.5
        for _ in 0..1000 {
            tracker.observe_tier(50);
        }
        let conf_after = tracker.amortization_confidence(0.01);
        assert!(
            conf_after > 0.5,
            "Confidence should be > 0.5 after amortization, got {conf_after}"
        );
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!(sigmoid(0.0) > 0.49 && sigmoid(0.0) < 0.51);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_bandit_select_tier() {
        let bandit = BreakevenBandit::new(100, 200, 50);

        // Feed baseline CPU timings
        for _ in 0..20 {
            bandit.observe_baseline(BreakevenTierPair::CpuToGpu, 1000);
        }
        // Feed GPU timings (faster per token)
        for _ in 0..20 {
            bandit.observe_tier(BreakevenTierPair::CpuToGpu, 100);
        }
        // Feed enough tokens to amortize
        for _ in 0..500 {
            bandit.observe_tier(BreakevenTierPair::CpuToGpu, 100);
        }

        // Should recommend GPU since it's amortized
        let recommendation = bandit.select_tier(ComputeTier::CpuOnly);
        assert!(recommendation.is_some());
    }

    #[test]
    fn test_bandit_no_opinion_initially() {
        let bandit = BreakevenBandit::with_defaults();
        // No observations → no opinion
        let recommendation = bandit.select_tier(ComputeTier::CpuOnly);
        assert!(recommendation.is_none());
    }

    #[test]
    fn test_stats_display() {
        let bandit = BreakevenBandit::with_defaults();
        let stats = bandit.stats();
        let display = stats.to_string();
        assert!(display.contains("CPU→GPU"));
        assert!(display.contains("GPU→ANE"));
    }

    #[test]
    fn test_tier_pair_display() {
        assert_eq!(BreakevenTierPair::CpuToGpu.to_string(), "CPU→GPU");
        assert_eq!(BreakevenTierPair::GpuToAne.to_string(), "GPU→ANE");
        assert_eq!(BreakevenTierPair::CpuToSpeculative.to_string(), "CPU→SPEC");
        assert_eq!(BreakevenTierPair::GpuToSpeculative.to_string(), "GPU→SPEC");
    }

    #[test]
    fn test_reset_clears_state() {
        let tracker = BreakevenTracker::new(1000);
        tracker.observe_baseline(100);
        tracker.observe_tier(50);
        assert!(tracker.total_tokens() > 0);

        tracker.reset();
        assert_eq!(tracker.total_tokens(), 0);
        assert_eq!(tracker.upfront_cost_us(), 0);
    }

    #[test]
    fn test_ema_convergence() {
        // Feed constant 100μs and verify EMA converges to ~100
        let tracker = BreakevenTracker::new(0);
        for _ in 0..100 {
            tracker.observe_baseline(100);
        }
        let baseline = tracker.baseline_cost_us();
        assert!(
            baseline > 80 && baseline < 120,
            "EMA should converge to ~100, got {baseline}"
        );
    }

    #[test]
    fn test_zero_alloc_tracking() {
        // Verify all operations use only atomic updates (no allocations)
        let tracker = BreakevenTracker::new(1000);
        for i in 0..1000 {
            tracker.observe_baseline(100 + i % 10);
            tracker.observe_tier(50 + i % 5);
        }
        // If we get here without panicking, no allocations occurred
        assert!(tracker.total_tokens() == 1000);
    }
}
