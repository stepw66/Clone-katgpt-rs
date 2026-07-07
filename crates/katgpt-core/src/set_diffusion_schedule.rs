//! Position-Offset Reveal-Time Schedule for Set Diffusion (Research 376 Phase 1 T1.1).
//!
//! Implements Eq. 7 from Arriola & Kuleshov, "Set Diffusion: Interpolating
//! Token Orderings Between AR and Diffusion" (arXiv:2607.01775, ICML 2026).
//! Distilled in `riir-train/.research/376_Set_Diffusion_Flexible_Token_Sets.md`.
//!
//! # Core mechanism
//!
//! Each token position ℓ ∈ [0, L) gets a CDF for its "reveal time" R_ℓ ∈ [0, 1]:
//!
//! ```text
//! α^τ_ℓ = 0                        if τ ≤ a_ℓ            (not yet revealed)
//!       = ((τ - a_ℓ) / w)^k        if a_ℓ < τ < a_ℓ + w  (active interval)
//!       = 1                        if τ ≥ a_ℓ + w         (definitely revealed)
//! ```
//!
//! Where:
//! - **w** = active generation interval width ∈ (0, 1]. Controls left-to-right bias.
//! - **k** = shape parameter within the interval ∈ (0, ∞). k<1 front-loads reveal
//!   times toward interval start; k>1 back-loads toward interval end; k=1 is uniform.
//! - **a_ℓ = ℓ / (L-1) · (1-w)** = evenly spaced offset per position (0-indexed).
//!
//! # Interpolation property
//!
//! - **w → 0**: intervals shrink to points → strict left-to-right → **AR**.
//! - **w = 1**: all offsets collapse to 0, intervals fully overlap → **order-agnostic
//!   diffusion (MDLM)**.
//! - **Intermediate w**: tokens reveal in overlapping windows → **SW-SetDLM**
//!   (sliding-window set diffusion language model).
//!
//! # Sampling
//!
//! To sample a generation ordering σ (a permutation of [0, L)):
//! 1. For each position ℓ, sample R_ℓ via inverse-CDF: R_ℓ = a_ℓ + w · U^(1/k),
//!    where U ~ Uniform(0, 1).
//! 2. Sort positions by reveal time ascending; ties broken by position index
//!    (deterministic left-to-right preference on ties).
//!
//! # Expected inference budget (Eq. 52)
//!
//! C̄ = L · w · k / (k + 1) — the expected number of tokens eligible per
//! inference step. Matches: AR (w→0) → C̄→0 (singleton sets, many steps);
//! diffusion (w=1, k→∞) → C̄=L (all tokens, one step).
//!
//! # Consumers
//!
//! - **T1.2** (training loop): `sample_order` provides σ; `order_to_gen_steps`
//!   converts σ to the `position_order` buffer consumed by the set-causal
//!   attention kernel (`riir-gpu/src/kernels/attention_score_set_causal.wgsl`,
//!   shipped Plan 312 T2.2) and the CPU reference (`riir_engine::forward_set_causal`,
//!   shipped Research 376 T0.3).
//! - **T1.3** (A/B battle): `expected_budget` matches inference budget between
//!   block-causal and set-causal schedules.
//!
//! TL;DR: Eq. 7 CDF + inverse-CDF sampler for set-diffusion position-offset schedules.
//!
//! # Canonical source (DRY consolidation, 2026-07-04)
//!
//! This module is the **single source of truth** for `PositionOffsetSchedule`.
//! Consumers (`katgpt-rs`, `riir-train`) re-export from here rather than
//! maintaining their own copies. The RNG-agnostic core primitive is
//! [`PositionOffsetSchedule::sample_order_with`], which takes a closure
//! yielding `Uniform(0, 1)` floats — this lets each consumer use its preferred
//! RNG (`katgpt_types::Rng`, `fastrand::Rng`, etc.) without forcing a hard
//! dependency. A `fastrand::Rng` convenience wrapper is provided because
//! katgpt-core already depends on `fastrand`.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════
// PositionOffsetSchedule
// ═══════════════════════════════════════════════════════════════════

/// Position-dependent reveal-time schedule (Eq. 7).
///
/// See module docs for the full math. The two knobs:
/// - `w` ∈ (0, 1]: interval width. w→0 is AR; w=1 is order-agnostic diffusion.
/// - `k` ∈ (0, ∞): intra-interval shape. k<1 front-loads; k=1 uniform; k>1 back-loads.
///
/// `Default` is the SW-SetDLM paper-aligned setting: w=0.5, k=1.0 (the config
/// that won the real-text battle in Plan 312 Phase 2 by 0.71 NLL over D2F).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct PositionOffsetSchedule {
    /// Active generation interval width ∈ (0, 1].
    pub w: f32,
    /// Shape parameter within the interval ∈ (0, ∞).
    pub k: f32,
}

impl Default for PositionOffsetSchedule {
    fn default() -> Self {
        Self { w: 0.5, k: 1.0 }
    }
}

/// Minimum positive value for `w` and `k` to avoid degenerate math (div-by-zero,
/// INFINITY exponents). Anything below this is clamped on construction.
const MIN_POSITIVE: f32 = 1e-6;

impl PositionOffsetSchedule {
    /// Create a schedule with linear intra-interval shape (k=1).
    ///
    /// `w` is clamped to `[MIN_POSITIVE, 1.0]` to stay in the valid range.
    #[inline]
    pub fn new(w: f32) -> Self {
        Self {
            w: w.clamp(MIN_POSITIVE, 1.0),
            k: 1.0,
        }
    }

    /// Create a schedule with custom shape parameter.
    ///
    /// Both `w` and `k` are clamped to their valid ranges.
    #[inline]
    pub fn shaped(w: f32, k: f32) -> Self {
        Self {
            w: w.clamp(MIN_POSITIVE, 1.0),
            k: k.clamp(MIN_POSITIVE, f32::INFINITY),
        }
    }

    /// The AR endpoint schedule (w minimal → near-deterministic left-to-right).
    ///
    /// Note: this produces near-AR orderings, not exact AR. For exact AR
    /// (guaranteed [0, 1, ..., L-1]), use [`ar_order`]. The schedule is useful
    /// when you want the AR-like endpoint of the continuous w axis with a
    /// small stochastic perturbation.
    #[inline]
    pub fn ar() -> Self {
        Self {
            w: MIN_POSITIVE,
            k: 1.0,
        }
    }

    /// The order-agnostic diffusion endpoint (w=1, k=1) — uniform random orderings.
    #[inline]
    pub fn diffusion() -> Self {
        Self { w: 1.0, k: 1.0 }
    }

    // ── Core math (Eq. 7) ──────────────────────────────────────────

    /// Interval offset for position `ell` in a sequence of length `l`.
    ///
    /// `a_ℓ = ell / (L-1) · (1-w)` for `L > 1`; `0.0` for `L <= 1`.
    ///
    /// Returns the left endpoint of position ℓ's active reveal-time window.
    /// The window spans `[offset(ell), offset(ell) + w]`.
    #[inline]
    pub fn offset(&self, ell: usize, l: usize) -> f32 {
        if l <= 1 {
            0.0
        } else {
            (ell as f32 / (l - 1) as f32) * (1.0 - self.w)
        }
    }

    /// Check if position `ell` is eligible for unmasking at ordering time `tau`.
    ///
    /// Eligible if `tau` falls within the position's active generation interval
    /// [a_ℓ, a_ℓ + w] and the position has a non-zero reveal rate.
    ///
    /// At the boundary (tau >= a_ℓ + w), the position is always eligible
    /// (it must eventually be decoded).
    #[inline]
    pub fn is_eligible(&self, ell: usize, l: usize, tau: f32) -> bool {
        let a = self.offset(ell, l);
        // Position is eligible if we're past its interval start.
        // At tau >= a + w, it's past due — always eligible.
        tau >= a
    }

    /// Returns the set of positions eligible for unmasking at ordering time `tau`.
    ///
    /// A position is eligible if its active generation interval has started
    /// (tau >= a_ℓ). Positions whose intervals haven't started yet are
    /// blocked — they can't be committed even if confidence is high.
    pub fn eligible_positions(&self, l: usize, tau: f32) -> Vec<bool> {
        (0..l).map(|ell| self.is_eligible(ell, l, tau)).collect()
    }

    /// The CDF α^τ_ℓ = P(R_ℓ ≤ τ) at time `tau` for position `ell` (Eq. 7).
    ///
    /// Returns a value in `[0, 1]`. Monotonically non-decreasing in `tau`.
    #[inline]
    pub fn alpha(&self, tau: f32, ell: usize, l: usize) -> f32 {
        let a = self.offset(ell, l);
        if tau <= a {
            0.0
        } else if tau >= a + self.w {
            1.0
        } else {
            ((tau - a) / self.w).powf(self.k)
        }
    }

    /// Inverse-CDF: map a uniform `u ∈ [0, 1]` to a reveal time R_ℓ.
    ///
    /// `R_ℓ = a_ℓ + w · u^(1/k)`. This is the sampling primitive — call with
    /// `u ~ Uniform(0, 1)` to draw from the position-ℓ reveal-time distribution.
    #[inline]
    pub fn reveal_time_from_uniform(&self, u: f32, ell: usize, l: usize) -> f32 {
        let a = self.offset(ell, l);
        // u clamped to [0, 1] to handle fastrand::Rng::f32() which returns [0, 1).
        let u = u.clamp(0.0, 1.0);
        a + self.w * u.powf(1.0 / self.k)
    }

    /// Expected inference prediction budget C̄ (Eq. 52).
    ///
    /// `C̄ = L · w · k / (k + 1)` — the expected number of tokens eligible per
    /// inference step under this schedule.
    #[inline]
    pub fn expected_budget(&self, l: usize) -> f32 {
        l as f32 * self.w * self.k / (self.k + 1.0)
    }

    /// Find the width `w` that matches a target inference budget B for length L.
    ///
    /// Solves `L · w · k / (k+1) = B` for w (closed form):
    /// `w = B · (k+1) / (L · k)`, clamped to `[MIN_POSITIVE, 1.0]`.
    ///
    /// Used by T1.3 to match the inference budget between block-causal
    /// (budget = block_size) and set-causal schedules.
    ///
    /// Returns a new schedule with this instance's `k` and the solved `w`.
    #[inline]
    pub fn with_matched_budget(&self, target_budget: f32, l: usize) -> Self {
        let w = if self.k > MIN_POSITIVE && l > 0 {
            (target_budget * (self.k + 1.0) / (l as f32 * self.k)).clamp(MIN_POSITIVE, 1.0)
        } else {
            MIN_POSITIVE
        };
        Self { w, k: self.k }
    }

    // ── Sampling ───────────────────────────────────────────────────
    //
    // The core primitive is `sample_order_with` — RNG-agnostic via a closure
    // that yields `Uniform(0, 1)` floats. This lets katgpt-rs (which uses
    // `katgpt_types::Rng`) and riir-train (which uses `fastrand::Rng`) share
    // the same code without a hard RNG dependency. The `fastrand::Rng`
    // convenience wrappers below delegate to the core primitive.

    /// Sample a generation ordering σ — a permutation of `[0, L)`.
    ///
    /// **RNG-agnostic core primitive.** Pass a closure that yields
    /// `Uniform(0, 1)` floats — e.g. `|| rng.uniform()` for `katgpt_types::Rng`,
    /// or `|| rng.f32()` for `fastrand::Rng`.
    ///
    /// Each position gets an independent reveal time drawn via inverse-CDF;
    /// sorting ascending gives the order. Ties (measure-zero for continuous
    /// distributions, but possible with extreme k) are broken by position
    /// index (smaller index first) for deterministic left-to-right preference.
    ///
    /// Returns `vec![]` for `l == 0`, `vec![0]` for `l == 1`.
    pub fn sample_order_with(&self, l: usize, mut uniform: impl FnMut() -> f32) -> Vec<usize> {
        if l == 0 {
            return Vec::new();
        }
        if l == 1 {
            return vec![0];
        }
        // Draw reveal times, sort by (reveal_time, position) for stable tie-break.
        let mut indexed: Vec<(f32, usize)> = (0..l)
            .map(|ell| (self.reveal_time_from_uniform(uniform(), ell, l), ell))
            .collect();
        indexed.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        indexed.into_iter().map(|(_, idx)| idx).collect()
    }

    /// Convenience wrapper: sample an ordering using `fastrand::Rng`.
    ///
    /// Delegates to [`sample_order_with`](Self::sample_order_with) with
    /// `|| rng.f32()`. Provided because katgpt-core already depends on
    /// `fastrand` and riir-train uses it as its primary RNG.
    pub fn sample_order(&self, l: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
        self.sample_order_with(l, || rng.f32())
    }

    /// Sample an ordering AND return the per-position reveal times (RNG-agnostic).
    ///
    /// **Core primitive** — pass a closure yielding `Uniform(0, 1)` floats.
    /// Returns `(order, reveal_times)` where `order` is the permutation and
    /// `reveal_times[i]` is the reveal time of position `i` (indexed by original
    /// position, not by order rank). Useful for debugging and for constructing
    /// weighted masks in T1.2.
    pub fn sample_order_with_reveal_times_with(
        &self,
        l: usize,
        mut uniform: impl FnMut() -> f32,
    ) -> (Vec<usize>, Vec<f32>) {
        if l == 0 {
            return (Vec::new(), Vec::new());
        }
        if l == 1 {
            return (
                vec![0],
                vec![self.reveal_time_from_uniform(uniform(), 0, 1)],
            );
        }
        let reveal_times: Vec<f32> = (0..l)
            .map(|ell| self.reveal_time_from_uniform(uniform(), ell, l))
            .collect();
        let mut indexed: Vec<(f32, usize)> = reveal_times.iter().copied().zip(0..l).collect();
        indexed.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let order: Vec<usize> = indexed.into_iter().map(|(_, idx)| idx).collect();
        (order, reveal_times)
    }

    /// Convenience wrapper: sample ordering + reveal times using `fastrand::Rng`.
    pub fn sample_order_with_reveal_times(
        &self,
        l: usize,
        rng: &mut fastrand::Rng,
    ) -> (Vec<usize>, Vec<f32>) {
        self.sample_order_with_reveal_times_with(l, || rng.f32())
    }
}

// ═══════════════════════════════════════════════════════════════════
// Static orderings (no schedule needed)
// ═══════════════════════════════════════════════════════════════════

/// Exact AR ordering: `[0, 1, ..., L-1]` (left-to-right, singleton sets).
pub fn ar_order(l: usize) -> Vec<usize> {
    (0..l).collect()
}

/// Uniform random ordering (MDLM / order-agnostic diffusion).
///
/// Fisher-Yates shuffle of `[0, L)`. **RNG-agnostic core primitive** —
/// pass a closure that yields `Uniform(0, n)` for an exclusive upper bound `n`.
pub fn uniform_order_with(l: usize, mut uniform_bounded: impl FnMut(u32) -> u32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..l).collect();
    // Fisher-Yates: for i from L-1 down to 1, swap with random j in [0, i].
    for i in (1..l).rev() {
        let j = uniform_bounded((i + 1) as u32) as usize;
        order.swap(i, j);
    }
    order
}

/// Convenience wrapper: uniform random ordering using `fastrand::Rng`.
pub fn uniform_order(l: usize, rng: &mut fastrand::Rng) -> Vec<usize> {
    uniform_order_with(l, |n| rng.u32(0..n))
}

// ═══════════════════════════════════════════════════════════════════
// Gen-step conversion (for the set-causal attention kernel)
// ═══════════════════════════════════════════════════════════════════

/// Convert an ordering σ to per-position generation steps (the kernel input).
///
/// Given `order = [σ_0, σ_1, ..., σ_{L-1}]` (σ_0 generated first), returns
/// `gen_step[p] = rank of position p in the ordering`. For SW-SetDLM with
/// singleton sets, this is a permutation of `[0, L)` as `u32`.
///
/// The returned buffer is what `riir_engine::forward_set_causal` and the
/// `attention_score_set_causal.wgsl` kernel consume as `position_order`.
pub fn order_to_gen_steps(order: &[usize]) -> Vec<u32> {
    if order.is_empty() {
        return Vec::new();
    }
    let l = order.len();
    let mut gen_steps = vec![0u32; l];
    for (step, &pos) in order.iter().enumerate() {
        debug_assert!(
            pos < l,
            "order contains position {pos} >= length {l} (not a valid permutation)"
        );
        gen_steps[pos] = step as u32;
    }
    gen_steps
}

/// Block-causal gen-steps: contiguous blocks of `block_size` share a step.
///
/// Positions `[0, block_size)` → step 0, `[block_size, 2·block_size)` → step 1, etc.
/// The final block may be shorter. This is what D2F/block-diffusion uses.
pub fn block_causal_gen_steps(l: usize, block_size: usize) -> Vec<u32> {
    assert!(block_size > 0, "block_size must be > 0");
    (0..l).map(|p| (p / block_size) as u32).collect()
}

/// MDLM gen-steps: all positions share step 0 (fully bidirectional).
pub fn mdlm_gen_steps(l: usize) -> Vec<u32> {
    vec![0u32; l]
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: verify a slice is a permutation of [0, n) ──
    fn assert_permutation(order: &[usize], n: usize) {
        assert_eq!(order.len(), n, "order length mismatch");
        let mut seen = vec![false; n];
        for &p in order {
            assert!(p < n, "position {p} out of range [0, {n})");
            assert!(!seen[p], "position {p} appears twice");
            seen[p] = true;
        }
    }

    // ── Helper: verify gen_steps is the inverse of order ──
    fn assert_gen_steps_inverse(order: &[usize], gen_steps: &[u32]) {
        assert_eq!(order.len(), gen_steps.len());
        for (step, &pos) in order.iter().enumerate() {
            assert_eq!(
                gen_steps[pos], step as u32,
                "gen_step[{pos}] should be {step}"
            );
        }
    }

    // ── Construction & validation ──

    #[test]
    fn test_new_clamps_w() {
        let s = PositionOffsetSchedule::new(2.0);
        assert!((s.w - 1.0).abs() < 1e-6, "w should clamp to 1.0");
        let s = PositionOffsetSchedule::new(-1.0);
        assert!(s.w > 0.0, "w should clamp to positive");
        assert!((s.k - 1.0).abs() < 1e-6, "new() should set k=1");
    }

    #[test]
    fn test_shaped_clamps_both() {
        let s = PositionOffsetSchedule::shaped(0.5, 0.0);
        assert!((s.w - 0.5).abs() < 1e-6);
        assert!(s.k > 0.0, "k should clamp to positive");
    }

    #[test]
    fn test_ar_endpoint() {
        let s = PositionOffsetSchedule::ar();
        assert!(s.w < 1e-3, "ar() should have tiny w");
    }

    #[test]
    fn test_diffusion_endpoint() {
        let s = PositionOffsetSchedule::diffusion();
        assert!((s.w - 1.0).abs() < 1e-6);
        assert!((s.k - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_default_is_paper_winner() {
        // Plan 312 Phase 2: w=0.5 won the real-text battle by 0.71 NLL.
        let s = PositionOffsetSchedule::default();
        assert!((s.w - 0.5).abs() < 1e-6);
        assert!((s.k - 1.0).abs() < 1e-6);
    }

    // ── Offset ──

    #[test]
    fn test_offset_zero_length() {
        let s = PositionOffsetSchedule::new(0.5);
        assert_eq!(s.offset(0, 0), 0.0);
        assert_eq!(s.offset(0, 1), 0.0);
    }

    #[test]
    fn test_offset_endpoints() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        // First position: offset 0.
        assert!((s.offset(0, l) - 0.0).abs() < 1e-6);
        // Last position: offset (1-w) = 0.5.
        assert!((s.offset(l - 1, l) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_offset_diffusion_all_zero() {
        let s = PositionOffsetSchedule::diffusion();
        let l = 8;
        for ell in 0..l {
            assert!(
                (s.offset(ell, l) - 0.0).abs() < 1e-6,
                "diffusion offset should be 0"
            );
        }
    }

    #[test]
    fn test_offset_monotone_increasing() {
        let s = PositionOffsetSchedule::new(0.3);
        let l = 16;
        for ell in 1..l {
            assert!(
                s.offset(ell, l) >= s.offset(ell - 1, l),
                "offset should be monotone non-decreasing"
            );
        }
    }

    // ── CDF (alpha) ──

    #[test]
    fn test_alpha_below_interval_is_zero() {
        let s = PositionOffsetSchedule::new(0.5);
        let a = s.offset(4, 8);
        assert_eq!(s.alpha(a - 0.01, 4, 8), 0.0);
        assert_eq!(s.alpha(a, 4, 8), 0.0);
    }

    #[test]
    fn test_alpha_above_interval_is_one() {
        let s = PositionOffsetSchedule::new(0.5);
        let a = s.offset(4, 8);
        assert_eq!(s.alpha(a + s.w, 4, 8), 1.0);
        assert_eq!(s.alpha(a + s.w + 0.01, 4, 8), 1.0);
    }

    #[test]
    fn test_alpha_midpoint_linear_is_half() {
        // k=1 → linear CDF. Midpoint of interval should give alpha=0.5.
        let s = PositionOffsetSchedule::new(0.5);
        let a = s.offset(4, 8);
        let mid = a + s.w * 0.5;
        assert!((s.alpha(mid, 4, 8) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_alpha_monotone_non_decreasing() {
        let s = PositionOffsetSchedule::shaped(0.4, 0.7);
        let l = 8;
        for ell in 0..l {
            let a = s.offset(ell, l);
            let mut prev = 0.0f32;
            let n_steps = 100;
            for i in 0..=n_steps {
                let tau = a + s.w * (i as f32 / n_steps as f32);
                let alpha = s.alpha(tau, ell, l);
                assert!(
                    alpha >= prev - 1e-6,
                    "alpha not monotone at ell={ell}, tau={tau}"
                );
                prev = alpha;
            }
        }
    }

    #[test]
    fn test_alpha_in_zero_one_range() {
        let s = PositionOffsetSchedule::shaped(0.6, 2.0);
        for ell in 0..8 {
            for i in 0..100 {
                let tau = i as f32 / 100.0;
                let a = s.alpha(tau, ell, 8);
                assert!((0.0..=1.0).contains(&a), "alpha out of [0,1]: {a}");
            }
        }
    }

    // ── Inverse CDF (reveal_time_from_uniform) ──

    #[test]
    fn test_reveal_time_endpoints() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        for ell in 0..l {
            let a = s.offset(ell, l);
            // u=0 → reveal time = a (interval start).
            assert!((s.reveal_time_from_uniform(0.0, ell, l) - a).abs() < 1e-6);
            // u=1 → reveal time = a + w (interval end).
            assert!((s.reveal_time_from_uniform(1.0, ell, l) - (a + s.w)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_reveal_time_in_interval() {
        let s = PositionOffsetSchedule::shaped(0.4, 0.5);
        let l = 16;
        for ell in 0..l {
            let a = s.offset(ell, l);
            for _ in 0..100 {
                let mut rng = fastrand::Rng::new();
                let r = s.reveal_time_from_uniform(rng.f32(), ell, l);
                assert!(
                    r >= a - 1e-6 && r <= a + s.w + 1e-6,
                    "reveal time {r} out of interval"
                );
            }
        }
    }

    #[test]
    fn test_inverse_cdf_roundtrip() {
        // alpha(reveal_time_from_uniform(u)) should recover u for k=1 (linear).
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        for ell in 0..l {
            for &u in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                let r = s.reveal_time_from_uniform(u, ell, l);
                let recovered = s.alpha(r, ell, l);
                assert!(
                    (recovered - u).abs() < 1e-4,
                    "roundtrip failed: u={u}, recovered={recovered}"
                );
            }
        }
    }

    // ── Expected budget ──

    #[test]
    fn test_budget_formula() {
        let s = PositionOffsetSchedule::shaped(0.5, 2.0);
        let l = 10;
        // C̄ = L · w · k / (k+1) = 10 · 0.5 · 2 / 3 = 10/3 ≈ 3.333
        let expected = 10.0 * 0.5 * 2.0 / 3.0;
        assert!((s.expected_budget(l) - expected).abs() < 1e-5);
    }

    #[test]
    fn test_budget_ar_tends_to_zero() {
        // AR: w → 0, k=1. C̄ = L · w · 0.5 → 0.
        let s = PositionOffsetSchedule::ar();
        assert!(s.expected_budget(16) < 0.01, "AR budget should tend to 0");
    }

    #[test]
    fn test_budget_diffusion_is_l_half() {
        // Diffusion: w=1, k=1. C̄ = L · 1 · 0.5 = L/2.
        let s = PositionOffsetSchedule::diffusion();
        assert!((s.expected_budget(16) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_with_matched_budget() {
        // Target budget B=4 for L=16, k=1.
        // w = B · (k+1) / (L · k) = 4 · 2 / (16 · 1) = 0.5.
        let base = PositionOffsetSchedule::new(0.3); // original w doesn't matter
        let matched = base.with_matched_budget(4.0, 16);
        assert!(
            (matched.w - 0.5).abs() < 1e-5,
            "w should be 0.5, got {}",
            matched.w
        );
        assert!((matched.k - 1.0).abs() < 1e-5, "k should be preserved");
        // Verify: expected_budget should now be ≈ 4.
        assert!((matched.expected_budget(16) - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_with_matched_budget_clamps_to_one() {
        // If target budget > L·k/(k+1) (= L/2 for k=1), w should clamp to 1.
        let base = PositionOffsetSchedule::new(0.5);
        let matched = base.with_matched_budget(100.0, 16); // way over max
        assert!((matched.w - 1.0).abs() < 1e-6, "w should clamp to 1.0");
    }

    // ── sample_order ──

    #[test]
    fn test_sample_order_empty() {
        let s = PositionOffsetSchedule::new(0.5);
        let mut rng = fastrand::Rng::with_seed(42);
        assert!(s.sample_order(0, &mut rng).is_empty());
    }

    #[test]
    fn test_sample_order_single() {
        let s = PositionOffsetSchedule::new(0.5);
        let mut rng = fastrand::Rng::with_seed(42);
        assert_eq!(s.sample_order(1, &mut rng), vec![0]);
    }

    #[test]
    fn test_sample_order_is_permutation() {
        let s = PositionOffsetSchedule::new(0.5);
        let mut rng = fastrand::Rng::with_seed(42);
        for l in [2, 4, 8, 16, 32] {
            let order = s.sample_order(l, &mut rng);
            assert_permutation(&order, l);
        }
    }

    #[test]
    fn test_sample_order_deterministic_with_seed() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 16;
        let order1 = s.sample_order(l, &mut fastrand::Rng::with_seed(123));
        let order2 = s.sample_order(l, &mut fastrand::Rng::with_seed(123));
        assert_eq!(order1, order2, "same seed should give same order");
    }

    #[test]
    fn test_sample_order_different_seeds_differ() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 32;
        let order1 = s.sample_order(l, &mut fastrand::Rng::with_seed(1));
        let order2 = s.sample_order(l, &mut fastrand::Rng::with_seed(2));
        // Overwhelmingly likely to differ for l=32.
        assert_ne!(
            order1, order2,
            "different seeds should (almost certainly) differ"
        );
    }

    #[test]
    fn test_ar_schedule_near_left_to_right() {
        // w very small → near-AR ordering. The schedule with w=1e-6 should
        // almost always produce [0, 1, ..., L-1] because intervals barely overlap.
        let s = PositionOffsetSchedule::ar();
        let l = 16;
        let mut near_ar_count = 0;
        for seed in 0..100 {
            let order = s.sample_order(l, &mut fastrand::Rng::with_seed(seed));
            if order == (0..l).collect::<Vec<_>>() {
                near_ar_count += 1;
            }
        }
        // With w ≈ 1e-6 and L=16, overlap is negligible → expect ≥95% exact AR.
        assert!(
            near_ar_count >= 95,
            "AR schedule should produce near-AR orderings: {near_ar_count}/100"
        );
    }

    #[test]
    fn test_diffusion_schedule_is_uniform() {
        // w=1 → all offsets are 0, so reveal times are i.i.d. → uniform random order.
        // Check: position 0 is first roughly 1/L of the time.
        let s = PositionOffsetSchedule::diffusion();
        let l = 8;
        let mut first_is_zero = 0;
        let n_trials = 1000;
        for seed in 0..n_trials {
            let order = s.sample_order(l, &mut fastrand::Rng::with_seed(seed as u64));
            if order[0] == 0 {
                first_is_zero += 1;
            }
        }
        // Expected: 1000/8 = 125. Allow [80, 170] for randomness.
        assert!(
            (80..=170).contains(&first_is_zero),
            "diffusion should be ~uniform: position 0 first {first_is_zero}/{n_trials} (expected ~125)"
        );
    }

    #[test]
    fn test_sample_order_with_reveal_times() {
        let s = PositionOffsetSchedule::new(0.5);
        let l = 8;
        let (order, reveal_times) =
            s.sample_order_with_reveal_times(l, &mut fastrand::Rng::with_seed(42));
        assert_permutation(&order, l);
        assert_eq!(reveal_times.len(), l);
        // Reveal times should be sorted when read in order.
        for i in 1..l {
            let prev = reveal_times[order[i - 1]];
            let curr = reveal_times[order[i]];
            assert!(
                prev <= curr + 1e-6,
                "reveal times not sorted in order: {prev} > {curr}"
            );
        }
    }

    // ── Static orderings ──

    #[test]
    fn test_ar_order() {
        assert_eq!(ar_order(0), Vec::<usize>::new());
        assert_eq!(ar_order(1), vec![0]);
        assert_eq!(ar_order(5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_uniform_order_is_permutation() {
        let mut rng = fastrand::Rng::with_seed(42);
        for l in [2, 4, 8, 16] {
            let order = uniform_order(l, &mut rng);
            assert_permutation(&order, l);
        }
    }

    #[test]
    fn test_uniform_order_deterministic() {
        let l = 16;
        let o1 = uniform_order(l, &mut fastrand::Rng::with_seed(7));
        let o2 = uniform_order(l, &mut fastrand::Rng::with_seed(7));
        assert_eq!(o1, o2);
    }

    // ── Gen-step conversion ──

    #[test]
    fn test_order_to_gen_steps_identity() {
        // AR order: gen_steps should be [0, 1, 2, ..., L-1].
        let order = ar_order(8);
        let gs = order_to_gen_steps(&order);
        assert_eq!(gs, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_order_to_gen_steps_reversed() {
        // Reversed order [L-1, ..., 1, 0]: position 0 is last → gen_step[0] = L-1.
        let order: Vec<usize> = (0..8).rev().collect();
        let gs = order_to_gen_steps(&order);
        assert_eq!(gs[0], 7);
        assert_eq!(gs[7], 0);
    }

    #[test]
    fn test_order_to_gen_steps_inverse_property() {
        let s = PositionOffsetSchedule::new(0.5);
        let order = s.sample_order(16, &mut fastrand::Rng::with_seed(42));
        let gs = order_to_gen_steps(&order);
        assert_gen_steps_inverse(&order, &gs);
    }

    #[test]
    fn test_order_to_gen_steps_empty() {
        assert!(order_to_gen_steps(&[]).is_empty());
    }

    #[test]
    fn test_block_causal_gen_steps() {
        // L=8, block_size=4: positions 0-3 → step 0, positions 4-7 → step 1.
        let gs = block_causal_gen_steps(8, 4);
        assert_eq!(gs, vec![0, 0, 0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_block_causal_gen_steps_partial_last_block() {
        // L=10, block_size=4: [0,0,0,0, 1,1,1,1, 2,2].
        let gs = block_causal_gen_steps(10, 4);
        assert_eq!(gs, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn test_mdlm_gen_steps() {
        let gs = mdlm_gen_steps(8);
        assert_eq!(gs, vec![0u32; 8]);
    }

    // ── Cross-validation against the PoC reference ──

    #[test]
    fn test_matches_poc_reference_budget() {
        // The PoC at riir-ai/crates/riir-poc/src/set_diffusion_poc.rs implements
        // the same formula. Verify our production version matches at the paper-
        // winning config (w=0.5, k=1, L=16).
        let s = PositionOffsetSchedule::new(0.5);
        let l = 16;
        // PoC: expected_budget = L · w · k / (k+1) = 16 · 0.5 · 1 / 2 = 4.0.
        assert!((s.expected_budget(l) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_matches_poc_reference_offset() {
        // PoC offset for ell=4, L=8, w=0.5: (4/7) · 0.5 = 2/7 ≈ 0.2857.
        let s = PositionOffsetSchedule::new(0.5);
        let expected = (4.0f32 / 7.0) * 0.5;
        assert!((s.offset(4, 8) - expected).abs() < 1e-6);
    }

    // ── Sanity: schedule interpolates between AR and diffusion ──

    #[test]
    fn test_schedule_interpolation_property() {
        // As w increases from ~0 to 1, the average "left-to-right-ness" should
        // decrease monotonically. We measure this as the Kendall tau distance
        // from the identity (AR) ordering — more inversions = less AR-like.
        let l = 16;
        let n_trials = 200;

        let avg_inversions = |w: f32| -> f32 {
            let s = PositionOffsetSchedule::new(w);
            let mut total = 0usize;
            for seed in 0..n_trials {
                let order = s.sample_order(l, &mut fastrand::Rng::with_seed(seed as u64));
                total += count_inversions(&order);
            }
            total as f32 / n_trials as f32
        };

        let inv_ar = avg_inversions(0.01);
        let inv_mid = avg_inversions(0.5);
        let inv_diff = avg_inversions(1.0);

        // AR-like should have few inversions; diffusion should have ~L(L-1)/4.
        let max_inv = (l * (l - 1) / 2) as f32;
        assert!(
            inv_ar < max_inv * 0.1,
            "AR-like (w=0.01) should have <10% max inversions, got {inv_ar:.1}/{max_inv:.0}"
        );
        assert!(
            inv_diff > max_inv * 0.35,
            "diffusion (w=1.0) should have >35% max inversions, got {inv_diff:.1}/{max_inv:.0}"
        );
        assert!(
            inv_mid > inv_ar && inv_mid < inv_diff,
            "intermediate w should interpolate: inv_ar={inv_ar:.1}, inv_mid={inv_mid:.1}, inv_diff={inv_diff:.1}"
        );
    }

    /// Count the number of inversions in a permutation (pairs (i,j) with i<j but order[i]>order[j]).
    fn count_inversions(order: &[usize]) -> usize {
        let n = order.len();
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if order[i] > order[j] {
                    count += 1;
                }
            }
        }
        count
    }
}
