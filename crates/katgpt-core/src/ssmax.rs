//! SSMax — length-aware log-N attention temperature.
//!
//! Distillation of Gollapudi et al., *Can Language Models Actually Retrieve
//! In-Context? Drowning in Documents at Million Token Scale*
//! (arXiv:2607.01538, 2026). The paper identifies **attention dilution** as
//! the primary bottleneck at million-token scale: as corpus size N grows, the
//! softmax denominator grows faster than the gold term's numerator, collapsing
//! the post-normalization mass on the gold document even when the pre-softmax
//! score stays high. The bound is:
//!
//! ```text
//! α_gold ≈ 1 / (1 + (N−1) · N^{−s·Δ})
//! ```
//!
//! where `Δ = s_{t⋆} − s̄_{distractor}` is the gold–distractor pre-softmax
//! logit gap. **SSMax** cancels the `(N−1)` growth by rescaling logits
//! multiplicatively by `s_L · log(N)`:
//!
//! ```text
//! s̃_{L,h,t} = s_L · log(N) · s_{L,h,t}
//! ```
//!
//! Whenever `s_L · Δ > 1`, the post-normalization gold weight is bounded away
//! from zero regardless of N.
//!
//! # The modelless reading (§3.5 unblock)
//!
//! The paper trains `s_L` in a shared parameter group. We do NOT train. We
//! derive `s_L` analytically from the desired behavior: pick `s_L` so that
//! `s_L · log(N) · Δ_typical ≈ log(N)`, i.e. `s_L ≈ 1/Δ_typical`. The default
//! `s_L = 1.0` ([`SsmaxMode::Fixed`]) is the truly modelless case (zero
//! training, zero new parameters); the [`SsmaxMode::Adaptive`] variant takes a
//! caller-managed rolling estimate of `Δ` and resolves `s_L = 1/Δ` clamped to
//! `[0.1, 10.0]`.
//!
//! # Composition
//!
//! SSMax operates at the **logit level** (before sigmoid/softmax). It composes
//! cleanly with:
//! - **sigmoid parallax** ([`crate::parallax_attn`]) — sigmoid's per-key bound
//!   means dilution is milder than softmax, but a length-adaptive sharpener
//!   still helps in the retrieval band when N grows into the thousands.
//! - **standard SDPA** ([`crate::attention`]) — for callers that need softmax.
//! - **sink-aware attention** — operates at the OUTPUT level (after the
//!   value-weighted sum), so SSMax and sink-aware don't interfere.
//!
//! SSMax does **NOT** apply to `funcattn` (Research 261 closed negative:
//! basis-mode structure has no `(n,n)` attention matrix, so dilution is
//! structurally absent). Do not wire it there.
//!
//! # Allocation discipline (G4)
//!
//! [`apply_ssmax_inplace`] is allocation-free by construction: a single
//! in-place multiply pass over the logits slice. No `Vec`, `Box`, or collecting
//! iterator appears. The chunked 8-wide loop pattern is there to help LLVM
//! auto-vectorize (per the AGENTS.md hot-loop rule).
//!
//! References:
//! - Plan 411 — open-primitive spec
//! - Research 392 — distillation, novelty gate, fusion analysis
//! - arXiv:2607.01538 — Gollapudi et al. (paper)
//! - arXiv:2501.19399 — Uszacorek et al., *Scalable-Softmax is Superior for
//!   Attention* (SSMax source paper, cited as [9] in 2607.01538)

// ──────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────

/// SSMax mode: the per-layer source of the temperature scalar `s_L`.
///
/// The multiplicative factor applied to pre-attention logits is
/// `s_L · log(N)` — see [`SsmaxMode::multiplier`].
///
/// # Modelless discipline
///
/// Both variants are modelless:
/// - [`Fixed`](SsmaxMode::Fixed) — a caller-chosen constant. The default
///   `s_l = 1.0` is the truly modelless case (zero training).
/// - [`Adaptive`](SsmaxMode::Adaptive) — derived from a caller-managed rolling
///   estimate of the gold-distractor logit gap Δ via the analytical formula
///   `s_L = 1/Δ_typical` clamped to `[0.1, 10.0]`. The estimator itself is
///   caller-owned (Plan 411 defers the built-in estimator to stretch S2); the
///   API here ships the contract, not the estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SsmaxMode {
    /// Fixed `s_L`. Truly modelless — zero training, zero new parameters.
    /// The default `s_l = 1.0` recovers the paper's SSMax with no tuning.
    Fixed {
        /// Per-layer temperature scalar `s_L`. Default `1.0`.
        s_l: f32,
    },
    /// Adaptive `s_L` derived from a caller-managed rolling estimate of the
    /// typical gold-distractor logit gap `Δ`. We resolve
    /// `s_L = 1/Δ_typical` (clamped to `[0.1, 10.0]`) so that
    /// `s_L · log(N) · Δ ≈ log(N)`, which cancels the `(N−1)` denominator
    /// growth in the dilution bound.
    Adaptive {
        /// Caller-managed rolling estimate of the gold-distractor logit gap.
        /// Should be positive (small values give large `s_L`, sharpening hard;
        /// large values give small `s_L`, sharpening mildly). Clamped on use.
        rolling_delta: f32,
    },
}

impl Default for SsmaxMode {
    /// Default: fixed `s_L = 1.0` (the truly modelless case).
    fn default() -> Self {
        Self::Fixed { s_l: 1.0 }
    }
}

impl SsmaxMode {
    /// Resolve the effective `s_L` scalar from the mode.
    ///
    /// - [`Fixed`](SsmaxMode::Fixed) → returns the stored `s_l` unchanged.
    /// - [`Adaptive`](SsmaxMode::Adaptive) → returns `1/rolling_delta` clamped
    ///   to `[0.1, 10.0]` (with `rolling_delta` floored at `1e-3` to avoid
    ///   division by zero).
    #[inline]
    pub fn resolve_s_l(&self) -> f32 {
        match self {
            SsmaxMode::Fixed { s_l } => *s_l,
            SsmaxMode::Adaptive { rolling_delta } => {
                (1.0_f32 / rolling_delta.max(1e-3)).clamp(0.1, 10.0)
            }
        }
    }

    /// The multiplicative factor applied to pre-attention logits: `s_L · log(N)`.
    ///
    /// `log_n` is `ln(N)` where N is the number of attended keys. The caller
    /// passes it in (rather than us computing `ln(N)` here) because the caller
    /// already knows N and we don't want to recompute `ln` in the hot loop.
    #[inline]
    pub fn multiplier(&self, log_n: f32) -> f32 {
        self.resolve_s_l() * log_n
    }
}

/// Pre-resolved SSMax configuration: bundles the resolved `s_L` scalar and the
/// precomputed `log(N)`.
///
/// Useful for storage in attention configs where `N` is known at construction
/// time and we want to cache the resolved values. For the hot-path apply, use
/// [`apply_ssmax_inplace`] with the [`SsmaxMode`] directly — the `log_n`
/// argument there is the only thing the caller needs to supply per call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SsmaxConfig {
    /// Resolved `s_L` (from [`SsmaxMode::resolve_s_l`]).
    pub s_l: f32,
    /// Precomputed `ln(N)` where N is the number of attended keys.
    pub log_n: f32,
}

impl SsmaxConfig {
    /// Construct from a mode + sequence length N.
    ///
    /// `log_n` is `ln(N)`, with `ln(1) = 0` and `ln(0) = 0` by convention
    /// (so SSMax is a no-op for sequences of length ≤ 1).
    #[inline]
    pub fn from_mode(mode: &SsmaxMode, n: usize) -> Self {
        let log_n = if n <= 1 { 0.0 } else { (n as f32).ln() };
        Self {
            s_l: mode.resolve_s_l(),
            log_n,
        }
    }

    /// The multiplicative factor applied to pre-attention logits: `s_L · log(N)`.
    #[inline]
    pub fn multiplier(&self) -> f32 {
        self.s_l * self.log_n
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Core function
// ──────────────────────────────────────────────────────────────────────────

/// Rescale pre-attention logits in place by `s_L · log(N)`.
///
/// This is the SSMax intervention. Each logit is multiplied by `s_L · log(N)`
/// where `s_L` is resolved from `mode` and `log_n = ln(N)` is passed in by the
/// caller (the caller knows N and we don't want to recompute `ln(N)` in the
/// hot loop).
///
/// # Arguments
///
/// - `logits` — `(n_heads, n_kv)` row-major f32, modified in place.
/// - `mode` — the per-layer source of `s_L`.
/// - `log_n` — `ln(n_kv)`, the natural log of the number of attended keys.
///
/// # Effect
///
/// When the pre-softmax gold-distractor gap is `Δ`, the post-normalization gold
/// weight becomes `α_gold ≈ 1 / (1 + (N−1) · N^{−s_L·Δ})`. SSMax cancels the
/// `(N−1)` growth whenever `s_L · Δ > 1`, recovering retrieval at large N.
///
/// # Composition
///
/// SSMax operates at the LOGIT level (before sigmoid/softmax). It composes
/// cleanly with sink-aware attention (which operates at the OUTPUT level, after
/// the value-weighted sum) — apply SSMax first, then the sink-aware gate.
///
/// SSMax does NOT apply to `funcattn` (Research 261 closed negative:
/// basis-mode structure has no `(n,n)` attention matrix, so dilution is
/// structurally absent). Do not wire it there.
///
/// # Allocation discipline (G4)
///
/// Allocation-free by construction: a single in-place multiply pass. No `Vec`,
/// `Box`, `String`, or collecting iterator appears.
///
/// # Example
///
/// ```ignore
/// use katgpt_core::ssmax::{SsmaxMode, apply_ssmax_inplace};
///
/// // N = 10_000 attended keys → log_n ≈ 9.21.
/// let log_n = (10_000_f32).ln();
/// let mut logits = vec![-1.0_f32, 0.5, 2.0, -0.3];
/// let mode = SsmaxMode::Fixed { s_l: 1.0 };
/// apply_ssmax_inplace(&mut logits, &mode, log_n);
/// // Each logit is now scaled by s_L · log(N) = 1.0 · 9.21 = 9.21.
/// assert!((logits[0] - (-1.0 * 9.21)).abs() < 1e-5);
/// ```
#[inline]
pub fn apply_ssmax_inplace(logits: &mut [f32], mode: &SsmaxMode, log_n: f32) {
    let mult = mode.multiplier(log_n);
    // Chunked 8-wide loop to help LLVM auto-vectorize (AGENTS.md hot-loop rule).
    for chunk in logits.chunks_exact_mut(8) {
        for x in chunk {
            *x *= mult;
        }
    }
    for x in logits.chunks_exact_mut(8).into_remainder() {
        *x *= mult;
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-6;

    #[test]
    fn fixed_mode_preserves_s_l() {
        assert_eq!(SsmaxMode::Fixed { s_l: 1.0 }.resolve_s_l(), 1.0);
        assert_eq!(SsmaxMode::Fixed { s_l: 0.5 }.resolve_s_l(), 0.5);
        assert_eq!(SsmaxMode::Fixed { s_l: 2.0 }.resolve_s_l(), 2.0);
    }

    #[test]
    fn adaptive_mode_clamps_tiny_delta() {
        // rolling_delta → 0 would make 1/delta → ∞; clamp to 10.0.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 1e-10,
        };
        assert!((m.resolve_s_l() - 10.0).abs() < TOL);
    }

    #[test]
    fn adaptive_mode_clamps_huge_delta() {
        // rolling_delta → ∞ would make 1/delta → 0; clamp to 0.1.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 1e10,
        };
        assert!((m.resolve_s_l() - 0.1).abs() < TOL);
    }

    #[test]
    fn adaptive_mode_resolves_inverse() {
        // The analytical contract: s_L = 1/Δ for in-range Δ.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 0.25,
        };
        assert!((m.resolve_s_l() - 4.0).abs() < TOL);

        let m = SsmaxMode::Adaptive {
            rolling_delta: 5.0,
        };
        assert!((m.resolve_s_l() - 0.2).abs() < TOL);
    }

    #[test]
    fn adaptive_mode_floors_zero_delta() {
        // rolling_delta = 0 would divide by zero; floor at 1e-3 → s_L = 1/1e-3
        // = 1000, then clamp to 10.0.
        let m = SsmaxMode::Adaptive {
            rolling_delta: 0.0,
        };
        assert!((m.resolve_s_l() - 10.0).abs() < TOL);

        // Negative delta should also floor (max(1e-3) sees the negative as
        // less than 1e-3 and replaces it).
        let m = SsmaxMode::Adaptive {
            rolling_delta: -1.0,
        };
        assert!((m.resolve_s_l() - 10.0).abs() < TOL);
    }

    #[test]
    fn multiplier_is_s_l_times_log_n() {
        let log_n = 9.21_f32; // ≈ ln(10_000)
        let m = SsmaxMode::Fixed { s_l: 2.0 };
        assert!((m.multiplier(log_n) - (2.0 * 9.21)).abs() < TOL);
    }

    #[test]
    fn apply_ssmax_scales_every_logit_by_multiplier() {
        let log_n = (10_000_f32).ln();
        let mode = SsmaxMode::Fixed { s_l: 1.0 };
        let mult = mode.multiplier(log_n);

        let original = [-1.0_f32, 0.5, 2.0, -0.3, 1.7, 0.0, -3.14, 0.001, 42.0];
        let mut logits = original;
        apply_ssmax_inplace(&mut logits, &mode, log_n);

        for (got, &orig) in logits.iter().zip(original.iter()) {
            let expected = orig * mult;
            assert!(
                (got - expected).abs() < TOL,
                "logit {orig} should scale to {expected}, got {got}"
            );
        }
    }

    #[test]
    fn apply_ssmax_empty_slice_is_noop() {
        let mut logits: [f32; 0] = [];
        apply_ssmax_inplace(&mut logits, &SsmaxMode::Fixed { s_l: 1.0 }, 9.21);
        // No panic, no-op.
    }

    #[test]
    fn apply_ssmax_identity_at_multiplier_one() {
        // When s_L · log_n = 1.0, logits are unchanged (identity scale).
        // This is the small-N no-regression guarantee (G5): at s_L = 1.0,
        // log_n = 1.0 (N = e ≈ 2.718) the multiplier is exactly 1.0.
        // (Note: log_n = 0.0 / N ≤ 1 gives multiplier 0.0, NOT identity —
        //  SSMax zeroes the logits at N ≤ 1, which is the documented
        //  small-N safety net, not a no-op.)
        let original = [-1.0_f32, 0.5, 2.0, -0.3, 1.7];
        let mut logits = original;
        apply_ssmax_inplace(&mut logits, &SsmaxMode::Fixed { s_l: 1.0 }, 1.0);
        for (got, &orig) in logits.iter().zip(original.iter()) {
            assert!((got - orig).abs() < TOL, "logit {orig} changed to {got}");
        }
    }

    #[test]
    fn apply_ssmax_simd_and_remainder_paths_agree_with_naive() {
        // Test multiple sizes to exercise both the 8-wide chunk path and the
        // remainder path.
        let log_n = 9.21_f32;
        let mode = SsmaxMode::Fixed { s_l: 0.7 };
        let mult = mode.multiplier(log_n);

        for &size in &[0_usize, 1, 7, 8, 9, 15, 16, 17, 32, 33, 100] {
            let original: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 5.0).collect();
            let mut logits = original.clone();
            apply_ssmax_inplace(&mut logits, &mode, log_n);

            // Compare against the naive scalar loop.
            for (got, &orig) in logits.iter().zip(original.iter()) {
                let expected = orig * mult;
                assert!(
                    (got - expected).abs() < TOL,
                    "size {size}: logit {orig} should scale to {expected}, got {got}"
                );
            }
        }
    }

    #[test]
    fn ssmax_config_from_mode_caches_log_n() {
        let mode = SsmaxMode::Fixed { s_l: 2.0 };
        let cfg = SsmaxConfig::from_mode(&mode, 10_000);
        assert!((cfg.s_l - 2.0).abs() < TOL);
        assert!((cfg.log_n - (10_000_f32).ln()).abs() < TOL);
        assert!((cfg.multiplier() - (2.0 * (10_000_f32).ln())).abs() < TOL);
    }

    #[test]
    fn ssmax_config_from_mode_handles_small_n() {
        // N ≤ 1 → log_n = 0 → multiplier = 0 (no-op / safety net).
        let cfg = SsmaxConfig::from_mode(&SsmaxMode::Fixed { s_l: 1.0 }, 0);
        assert_eq!(cfg.log_n, 0.0);
        assert_eq!(cfg.multiplier(), 0.0);

        let cfg = SsmaxConfig::from_mode(&SsmaxMode::Fixed { s_l: 1.0 }, 1);
        assert_eq!(cfg.log_n, 0.0);
        assert_eq!(cfg.multiplier(), 0.0);
    }

    #[test]
    fn default_mode_is_fixed_one() {
        // The truly-modelless default: s_L = 1.0.
        assert_eq!(SsmaxMode::default(), SsmaxMode::Fixed { s_l: 1.0 });
    }
}
