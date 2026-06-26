//! AcceptanceForecast — entropy-bounded acceptance-rate forecast for adaptive γ.
//!
//! Distills Bebop (arXiv:2606.12370 §3, §7.6): the speculative-decode
//! acceptance rate `α` is linearly bounded by the target model's next-token
//! entropy `H(p)`:
//!
//! ```text
//! α ≈ a − b · H(p),    H(p) = −Σ p_v ln p_v
//! ```
//!
//! The bound is proven (Propositions 1, 2, 4); the paper §7.6 flags adaptive-γ
//! from this forecast as unproven future work. This module implements the
//! primitive; Issue 023 is the GOAT gate that decides whether wiring it into
//! the decode loop is worth the overhead.
//!
//! # Design
//!
//! - **Two-parameter linear model** fitted once from warmup `(H, accept_rate)` pairs.
//! - **Zero-allocation** `observe_and_forecast`: entropy computed in 2 iterator
//!   passes (find-max + fused sum/S₂), no Vec/Box. The math identity used is
//!   `H = ln(Z) − S₂/Z` where `Z = Σ exp(xᵢ − x_max)` and
//!   `S₂ = Σ exp(xᵢ − x_max) · (xᵢ − x_max)` — derived from `−Σ pᵢ ln pᵢ`
//!   with `pᵢ = exp(xᵢ − x_max) / Z`.
//! - **Fits in L1**: 5 × f32 = 20 bytes.
//! - **Feature flag**: `adaptive_gamma_forecast` (default-OFF, GOAT-gated).
//!
//! # Entropy and softmax
//!
//! Entropy computation uses `softmax(logits)` as the mathematical definition of
//! a probability distribution from logits. This is not a *routing* decision —
//! per AGENTS.md the "use sigmoid not softmax" rule applies to routing and
//! latent-space projections, not to computing `H(p)` from logits. The forecast
//! itself is a raw linear estimate clamped to `[0, 1]` (the proven bound's
//! natural range), not a sigmoid projection.
//!
//! # References
//!
//! - Issue 023 (GOAT gate)
//! - Research 243 (distillation)
//! - Bebop §3 (entropy–acceptance bound), §7.6 (adaptive-γ future work)
//! - `src/attn_match/adaptive_cot.rs::entropy_from_logits` (allocating version
//!   used by the KV-compaction path — not reused here because this module must
//!   be zero-allocation).

/// EMA decay factor. α=0.1 → ~10-token smoothing window, matching
/// `AdaptiveTraceCompactor::DEFAULT_EMA_ALPHA`.
pub const DEFAULT_ALPHA_DECAY: f32 = 0.1;

/// Minimum forecast floor. Prevents division-by-zero in `adaptive_gamma`
/// when the fitted model predicts near-zero acceptance.
pub const ALPHA_FLOOR: f32 = 0.01;

/// Calibrated acceptance-rate forecast from target entropy.
///
/// `α ≈ a − b · H(p)`, proven linear bound (Bebop, arXiv:2606.12370 §3).
/// Fit `a, b` once from warmup via [`fit_from_warmup`](Self::fit_from_warmup);
/// the per-step forecast is O(1) after the O(vocab) entropy computation.
///
/// Field order: all `f32` (4B each, 4B aligned) — 20 bytes total, no padding.
#[derive(Clone, Copy, Debug)]
pub struct AcceptanceForecast {
    /// Fitted intercept. Empirically ~1.0 for RS+TV.
    pub a: f32,
    /// Fitted entropy slope (positive ⇒ higher entropy lowers acceptance).
    pub b: f32,
    /// EMA-smoothed target entropy `H(p_t)`.
    pub ema_entropy: f32,
    /// Last forecast `α` (cached for O(1) reads via
    /// [`forecast_alpha_current`](Self::forecast_alpha_current)).
    pub ema_alpha: f32,
    /// EMA decay factor in `(0, 1]`. Default `0.1`.
    pub alpha_decay: f32,
}

impl Default for AcceptanceForecast {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AcceptanceForecast {
    /// Sensible defaults: `a = 1.0`, `b = 0.0` (no entropy penalty until
    /// fitted), decay `0.1`. With these defaults the forecast always returns
    /// `1.0`, so callers that never fit behave as a no-op pass-through.
    #[inline]
    pub const fn new() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            ema_entropy: 0.0,
            ema_alpha: 1.0,
            alpha_decay: DEFAULT_ALPHA_DECAY,
        }
    }

    /// Custom intercept, slope, and decay.
    #[inline]
    pub const fn with_params(a: f32, b: f32, alpha_decay: f32) -> Self {
        Self {
            a,
            b,
            ema_entropy: 0.0,
            ema_alpha: a,
            alpha_decay,
        }
    }

    /// Fit `a, b` from warmup `(entropy, accept_rate)` samples via closed-form
    /// ordinary least squares on the model `α = a − b · H`.
    ///
    /// Internally fits `y = A + M·x` (standard OLS) and then sets `b = −M`
    /// so the public model reads `α = a − b · H`. Falls back to
    /// [`new()`](Self::new) on empty input, and to a slope-zero intercept
    /// (mean of `α` samples) on degenerate input (zero variance in `H`).
    ///
    /// O(n) — single folding pass over the samples.
    pub fn fit_from_warmup(samples: &[(f32, f32)]) -> Self {
        if samples.is_empty() {
            return Self::new();
        }
        let n = samples.len() as f32;
        // Fold: (Σx, Σy, Σxy, Σx²)
        let (sum_x, sum_y, sum_xy, sum_x2) = samples.iter().fold(
            (0.0f32, 0.0f32, 0.0f32, 0.0f32),
            |(sx, sy, sxy, sx2), &(x, y)| (sx + x, sy + y, sxy + x * y, sx2 + x * x),
        );
        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            // Degenerate: all entropies identical (e.g. warmup on a peaked
            // distribution). Slope is undefined; use the mean acceptance rate
            // as the intercept with zero slope.
            let mean_alpha = sum_y / n;
            return Self {
                a: mean_alpha.clamp(0.0, 1.0),
                b: 0.0,
                ema_entropy: 0.0,
                ema_alpha: mean_alpha.clamp(0.0, 1.0),
                alpha_decay: DEFAULT_ALPHA_DECAY,
            };
        }
        // Standard OLS slope for y = A + M·x.
        let m = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - m * sum_x) / n;
        // Our model: α = a − b · H  ⇒  b = −M.
        let b = -m;
        let clamped_a = a.clamp(0.0, 1.0);
        let b_nonneg = b.max(0.0); // entropy can only reduce acceptance
        Self {
            a: clamped_a,
            b: b_nonneg,
            ema_entropy: 0.0,
            ema_alpha: clamped_a,
            alpha_decay: DEFAULT_ALPHA_DECAY,
        }
    }

    /// Compute next-token entropy `H(p)` from `logits` in 2 zero-allocation
    /// iterator passes, update the EMA, and return the forecast acceptance
    /// rate `α = clamp(a − b · ema_entropy, 0, 1)`.
    ///
    /// # Zero-allocation proof
    ///
    /// Pass 1 scans `logits` to find the max (stability shift). Pass 2 fuses
    /// the normalizer `Z = Σ exp(xᵢ − x_max)` with the cross term
    /// `S₂ = Σ exp(xᵢ − x_max) · (xᵢ − x_max)`. Both passes use only
    /// stack-local accumulators — no `Vec`, no `Box`.
    ///
    /// Returns `1.0` (perfect acceptance) on empty input.
    #[inline]
    pub fn observe_and_forecast(&mut self, logits: &[f32]) -> f32 {
        let h = entropy_nats_zero_alloc(logits);
        self.ema_entropy =
            self.alpha_decay * h + (1.0 - self.alpha_decay) * self.ema_entropy;
        let alpha = (self.a - self.b * self.ema_entropy).clamp(ALPHA_FLOOR, 1.0);
        self.ema_alpha = alpha;
        alpha
    }

    /// Read the current EMA forecast `α` without a new observation.
    /// Returns the value cached by the last [`observe_and_forecast`](Self::observe_and_forecast).
    #[inline]
    pub const fn forecast_alpha_current(&self) -> f32 {
        self.ema_alpha
    }

    /// Adaptive draft length `γ` from the forecast acceptance rate.
    ///
    /// `γ = clamp(ceil(target_tokens / α), γ_min, γ_max)`
    ///
    /// When `α` is high (low entropy, predictable target) the drafter can
    /// safely aim for `target_tokens` accepted tokens. When `α` is low
    /// (high entropy) the formula *increases* `γ` to compensate — this is
    /// the paper §7.6 suggestion for batched-verification decode trees where
    /// the target forward cost is amortised over all drafted positions.
    /// Callers on non-batched verifiers should set `γ_max` tightly.
    #[inline]
    pub fn adaptive_gamma(
        &self,
        target_tokens: usize,
        forecast_alpha: f32,
        gamma_min: usize,
        gamma_max: usize,
    ) -> usize {
        let alpha = forecast_alpha.max(ALPHA_FLOOR);
        let gamma = ((target_tokens as f32) / alpha).ceil() as usize;
        gamma.clamp(gamma_min, gamma_max)
    }

    /// Reset the EMA state (e.g. at the start of a new trace).
    /// Keeps the fitted `(a, b)` — only the runtime state is cleared.
    #[inline]
    pub fn reset_ema(&mut self) {
        self.ema_entropy = 0.0;
        self.ema_alpha = self.a;
    }
}

/// Natural-log entropy `H(p) = −Σ pᵢ ln pᵢ` from raw logits, zero allocation.
///
/// 2-pass algorithm using the identity `H = ln(Z) − S₂/Z` where
/// `Z = Σ exp(xᵢ − x_max)` and `S₂ = Σ exp(xᵢ − x_max) · (xᵢ − x_max)`.
///
/// Derivation:
/// ```text
/// pᵢ = exp(xᵢ − x_max) / Z
/// ln(pᵢ) = (xᵢ − x_max) − ln(Z)
/// H = −Σ pᵢ · ln(pᵢ)
///   = −Σ pᵢ · (xᵢ − x_max) + ln(Z) · Σ pᵢ
///   = ln(Z) − (1/Z) · Σ exp(xᵢ − x_max) · (xᵢ − x_max)
///   = ln(Z) − S₂ / Z
/// ```
///
/// Returns `0.0` on empty input.
#[inline]
pub fn entropy_nats_zero_alloc(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    // Pass 1: max-shift for numerical stability.
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_logit {
            max_logit = l;
        }
    }
    if !max_logit.is_finite() {
        return 0.0;
    }
    // Pass 2: fuse Z = Σ exp(xᵢ − x_max) and S₂ = Σ exp(xᵢ − x_max) · (xᵢ − x_max).
    let mut z = 0.0f32;
    let mut s2 = 0.0f32;
    for &l in logits {
        let shifted = l - max_logit;
        let e = shifted.exp();
        z += e;
        s2 += e * shifted;
    }
    if z <= 0.0 || !z.is_finite() {
        return 0.0;
    }
    let ln_z = z.ln();
    let h = ln_z - s2 / z;
    // Clamp to [0, ∞) — H is non-negative; tiny float errors can push it
    // marginally below 0 on peaked distributions.
    h.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── entropy_nats_zero_alloc correctness ──────────────────────────────

    #[test]
    fn test_entropy_empty_returns_zero() {
        assert_eq!(entropy_nats_zero_alloc(&[]), 0.0);
    }

    #[test]
    fn test_entropy_uniform_two_tokens_is_ln2() {
        // Uniform over 2 tokens: H = ln(2) ≈ 0.6931 nats.
        let logits = [0.0f32, 0.0];
        let h = entropy_nats_zero_alloc(&logits);
        assert!(
            (h - 2.0f32.ln()).abs() < 1e-5,
            "uniform 2-token entropy should be ln(2) ≈ 0.6931, got {h}"
        );
    }

    #[test]
    fn test_entropy_uniform_n_tokens_is_ln_n() {
        for n in [3usize, 4, 8, 16, 32, 64, 128] {
            let logits = vec![0.0f32; n];
            let h = entropy_nats_zero_alloc(&logits);
            let expected = (n as f32).ln();
            assert!(
                (h - expected).abs() < 1e-4,
                "uniform {n}-token entropy should be ln({n}) ≈ {expected:.4}, got {h}"
            );
        }
    }

    #[test]
    fn test_entropy_peaked_is_near_zero() {
        // Very peaked: logit 20 vs 0 → near-deterministic.
        let logits = [20.0f32, 0.0, 0.0, 0.0];
        let h = entropy_nats_zero_alloc(&logits);
        assert!(h < 0.01, "peaked distribution entropy should be ≈ 0, got {h}");
    }

    #[test]
    fn test_entropy_matches_reference_implementation() {
        // Cross-check against a naive reference that allocates.
        fn reference(logits: &[f32]) -> f32 {
            if logits.is_empty() {
                return 0.0;
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            if z <= 0.0 {
                return 0.0;
            }
            let mut h = 0.0f32;
            for &e in &exps {
                let p = e / z;
                if p > 0.0 {
                    h -= p * p.ln();
                }
            }
            h.max(0.0)
        }
        let test_cases: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1.0, 2.0, 3.0, 0.5],
            vec![10.0, -2.0, 3.0, 0.0, 7.0],
            vec![-5.0, -5.0, -5.0],
            (0..256).map(|i| (i as f32) * 0.1 - 12.8).collect(),
        ];
        for (i, logits) in test_cases.iter().enumerate() {
            let zero_alloc = entropy_nats_zero_alloc(logits);
            let ref_val = reference(logits);
            assert!(
                (zero_alloc - ref_val).abs() < 1e-4,
                "case {i}: zero_alloc {zero_alloc:.6} ≠ reference {ref_val:.6} \
                 (Δ = {})",
                (zero_alloc - ref_val).abs()
            );
        }
    }

    #[test]
    fn test_entropy_shift_invariant() {
        // Entropy is invariant to constant shift of logits.
        let base = [1.0f32, 2.0, 3.0, 0.5];
        let shifted: Vec<f32> = base.iter().map(|&x| x + 100.0).collect();
        let neg: Vec<f32> = base.iter().map(|&x| x - 50.0).collect();
        let h_base = entropy_nats_zero_alloc(&base);
        let h_shifted = entropy_nats_zero_alloc(&shifted);
        let h_neg = entropy_nats_zero_alloc(&neg);
        assert!((h_base - h_shifted).abs() < 1e-4);
        assert!((h_base - h_neg).abs() < 1e-4);
    }

    // ── AcceptanceForecast EMA decay ─────────────────────────────────────

    #[test]
    fn test_new_defaults_to_alpha_one() {
        let f = AcceptanceForecast::new();
        assert_eq!(f.a, 1.0);
        assert_eq!(f.b, 0.0);
        assert_eq!(f.alpha_decay, 0.1);
        assert_eq!(f.forecast_alpha_current(), 1.0);
    }

    #[test]
    fn test_observe_and_forecast_constant_entropy_converges() {
        // With b = 0.5 and constant low entropy, forecast should converge
        // to a − b · H.
        let mut f = AcceptanceForecast::with_params(1.0, 0.5, 0.5);
        // Entropy 0 → peaked distribution.
        let peaked = [20.0f32, 0.0, 0.0];
        for _ in 0..100 {
            f.observe_and_forecast(&peaked);
        }
        // H ≈ 0 ⇒ α ≈ 1.0.
        assert!(
            (f.forecast_alpha_current() - 1.0).abs() < 1e-3,
            "converged to ≈ 1.0 at H≈0, got {}",
            f.forecast_alpha_current()
        );
    }

    #[test]
    fn test_ema_tracks_entropy_changes() {
        // Start peaked (H≈0), switch to uniform (H=ln2).
        // Forecast should drop from 1.0 toward a − b·ln(2).
        let mut f = AcceptanceForecast::with_params(1.0, 0.5, 0.5);
        let peaked = [20.0f32, 0.0, 0.0];
        let uniform = [0.0f32, 0.0];
        for _ in 0..50 {
            f.observe_and_forecast(&peaked);
        }
        let alpha_low_h = f.forecast_alpha_current();
        for _ in 0..50 {
            f.observe_and_forecast(&uniform);
        }
        let alpha_high_h = f.forecast_alpha_current();
        assert!(
            alpha_low_h > alpha_high_h,
            "low-H forecast ({alpha_low_h}) should exceed high-H forecast ({alpha_high_h})"
        );
        let expected = (1.0 - 0.5 * 2.0f32.ln()).clamp(ALPHA_FLOOR, 1.0);
        assert!(
            (alpha_high_h - expected).abs() < 1e-3,
            "converged to a − b·ln(2) = {expected:.4}, got {alpha_high_h:.4}"
        );
    }

    #[test]
    fn test_reset_ema_clears_state() {
        let mut f = AcceptanceForecast::with_params(1.0, 0.5, 0.5);
        let uniform = [0.0f32, 0.0];
        for _ in 0..20 {
            f.observe_and_forecast(&uniform);
        }
        assert!(f.ema_entropy > 0.1);
        f.reset_ema();
        assert_eq!(f.ema_entropy, 0.0);
        assert_eq!(f.forecast_alpha_current(), 1.0); // a − b·0 = a = 1.0
    }

    // ── fit_from_warmup linear regression ────────────────────────────────

    #[test]
    fn test_fit_from_warmup_empty_returns_defaults() {
        let f = AcceptanceForecast::fit_from_warmup(&[]);
        assert_eq!(f.a, 1.0);
        assert_eq!(f.b, 0.0);
    }

    #[test]
    fn test_fit_from_warmup_degenerate_identical_entropy() {
        // All same H → denom ≈ 0 → intercept = mean α, slope 0.
        let samples = vec![(1.0f32, 0.8f32), (1.0, 0.6), (1.0, 0.7)];
        let f = AcceptanceForecast::fit_from_warmup(&samples);
        assert!((f.a - 0.7).abs() < 1e-4, "mean α = 0.7, got a = {}", f.a);
        assert!(f.b.abs() < 1e-4, "degenerate fit should have b ≈ 0, got {}", f.b);
    }

    #[test]
    fn test_fit_from_warmup_recovers_known_line() {
        // Generate samples on the line α = 1.0 − 0.3 · H, with H in [0, 3)
        // so α stays in (0.1, 1.0] (no clamping artifacts).
        let samples: Vec<(f32, f32)> = (0..20)
            .map(|i| {
                let h = i as f32 * 0.15; // [0, 2.85)
                let alpha = 1.0 - 0.3 * h; // (0.145, 1.0]
                (h, alpha)
            })
            .collect();
        let f = AcceptanceForecast::fit_from_warmup(&samples);
        assert!(
            (f.a - 1.0).abs() < 1e-3,
            "intercept should be 1.0, got {}",
            f.a
        );
        assert!(
            (f.b - 0.3).abs() < 1e-3,
            "slope should be 0.3, got {}",
            f.b
        );
    }

    #[test]
    fn test_fit_from_warmup_clamps_intercept_and_forces_nonneg_slope() {
        // Samples where OLS would give a > 1 or b < 0: the constructor clamps.
        let samples = vec![(0.0f32, 1.5f32), (1.0, 1.8)]; // α increases with H
        let f = AcceptanceForecast::fit_from_warmup(&samples);
        assert!(f.a <= 1.0, "intercept clamped to ≤ 1.0, got {}", f.a);
        assert!(f.b >= 0.0, "slope forced ≥ 0, got {}", f.b);
    }

    // ── adaptive_gamma clamping ──────────────────────────────────────────

    #[test]
    fn test_adaptive_gamma_high_alpha_targets_lookahead() {
        let f = AcceptanceForecast::new();
        // α = 1.0, target = 8 → γ = 8.
        let gamma = f.adaptive_gamma(8, 1.0, 1, 16);
        assert_eq!(gamma, 8);
    }

    #[test]
    fn test_adaptive_gamma_low_alpha_increases() {
        let f = AcceptanceForecast::new();
        // α = 0.5, target = 8 → ceil(8/0.5) = 16.
        let gamma = f.adaptive_gamma(8, 0.5, 1, 16);
        assert_eq!(gamma, 16);
    }

    #[test]
    fn test_adaptive_gamma_clamps_to_max() {
        let f = AcceptanceForecast::new();
        // α = 0.1, target = 8 → ceil(8/0.1) = 80 → clamped to 16.
        let gamma = f.adaptive_gamma(8, 0.1, 1, 16);
        assert_eq!(gamma, 16);
    }

    #[test]
    fn test_adaptive_gamma_clamps_to_min() {
        let f = AcceptanceForecast::new();
        // α = 1.0, target = 0 → 0 → clamped to γ_min = 2.
        let gamma = f.adaptive_gamma(0, 1.0, 2, 16);
        assert_eq!(gamma, 2);
    }

    #[test]
    fn test_adaptive_gamma_zero_alpha_uses_floor() {
        let f = AcceptanceForecast::new();
        // α = 0.0 → floored to 0.01 → ceil(8/0.01) = 800 → clamped to 16.
        let gamma = f.adaptive_gamma(8, 0.0, 1, 16);
        assert_eq!(gamma, 16);
    }

    #[test]
    fn test_observe_empty_logits_returns_alpha_floor_at_least() {
        let mut f = AcceptanceForecast::with_params(1.0, 0.5, 0.1);
        let alpha = f.observe_and_forecast(&[]);
        // Empty logits → H = 0 → α = clamp(1.0 − 0.5·0, floor, 1) = 1.0.
        assert!(
            (alpha - 1.0).abs() < 1e-5,
            "empty logits should give α = a = 1.0, got {alpha}"
        );
    }
}
