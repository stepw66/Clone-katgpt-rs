//! QmcHalter — sample-efficiency-aware halting for QMC rollout budgets
//! (Plan 367 Fusion E, Research 367 §2.3 + Research 205 §1 union bound).
//!
//! The sample-efficiency-aware analog of [`GainCostLoopHalter`] (Plan 304).
//! Where the gain/cost halter asks "is the marginal refinement worth the
//! drift cost?", QmcHalter asks "have I drawn enough QMC rollouts to cover
//! the target event?".
//!
//! ## The union-bound ceiling
//!
//! For k rollouts each hitting a target event with marginal probability p,
//! the union bound (R205 §1, Eq. 30–32) caps the probability of *at least
//! one* success:
//!
//! ```text
//! P(at least one success) ≤ min(1, k · p)        — the union-bound ceiling
//! ```
//!
//! i.i.d. rollouts achieve only `1 − (1−p)^k`, which is strictly below the
//! ceiling (except at the 0/1 endpoints). QMC's controlled correlation
//! pushes the actual probability **toward** the ceiling — that's the whole
//! point of QuasiMoTTo (Plan 367). So a halter that trusts the QMC
//! saturation can stop as soon as the ceiling reaches the caller's target
//! coverage, rather than waiting for the i.i.d.-derived `k ≥
//! log(1−target)/log(1−p)` budget.
//!
//! ## Decision surface
//!
//! The halter is a pure config struct (stateless — the caller tracks the
//! running k and n_hits). [`QmcHalter::evaluate`] takes the current batch
//! observables and returns a [`QmcHaltDecision`]:
//!
//! - [`QmcHaltDecision::RefusedFloor`] — k < k_min (protect minimum batch).
//! - [`QmcHaltDecision::Halt`] — target met, hit observed, ceiling saturated,
//!   or k_max cap reached.
//! - [`QmcHaltDecision::Continue`] — ceiling below target and no hit yet;
//!   drawing more raises `k · p` toward the target.
//!
//! ## Latent vs Raw
//!
//! The halter inputs (`p`, `n_hits`) are local latent observables — callers
//! compute them from their own LM-state / point-set. The decision is a
//! deterministic raw scalar (halt/continue) safe to sync/replay, mirroring
//! the gain/cost halter contract.
//!
//! ## Zero-allocation contract
//!
//! [`QmcHalter::evaluate`] does no allocation. The caller-provided point
//! set (when using [`count_hits_1d`]) is borrowed; the hit count itself is
//! a `usize`.
//!
//! [`GainCostLoopHalter`]: crate::gain_cost_halt::GainCostLoopHalter

// ─────────────────────────────────────────────────────────────────────────────
// Decision types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a single QmcHalter evaluation.
///
/// Returned by [`QmcHalter::evaluate`]. Small (≤ 16 bytes), `Copy`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QmcHaltDecision {
    /// k < k_min — refuse to halt (protect minimum batch). Caller MUST draw
    /// at least `k_min` points before any halt is permitted.
    RefusedFloor,
    /// Continue drawing — union-bound ceiling is below `target_coverage`
    /// and no hit has been observed yet.
    Continue {
        /// Current union-bound ceiling `min(1, k·p)`.
        ceiling: f32,
        /// Gap to target: `(target_coverage − ceiling).max(0.0)`. Zero once
        /// the ceiling reaches the target (caller can use this as a
        /// progress signal).
        gap: f32,
    },
    /// Halt — stop drawing more rollouts. See [`QmcHaltReason`] for why.
    Halt {
        /// Why the halter fired.
        reason: QmcHaltReason,
        /// Final union-bound ceiling `min(1, k·p)`.
        ceiling: f32,
        /// Empirical coverage from the actual point set:
        /// - `1.0` if at least one of the k points hit the event.
        /// - `ceiling` if the ceiling reached target without a hit (QMC is
        ///   trusted to saturate; the gap is the residual QMC variance).
        coverage: f32,
    },
}

/// Why [`QmcHalter::evaluate`] returned [`QmcHaltDecision::Halt`].
///
/// `#[repr(u8)]` keeps the payload at 1 byte so the whole
/// [`QmcHaltDecision`] stays well under a cache word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum QmcHaltReason {
    /// At least one of the k QMC points hit the target event — early
    /// termination. The caller's objective (pass@k, "at least one success")
    /// is satisfied.
    HitObserved = 0,
    /// Union-bound ceiling `min(1, k·p)` reached `target_coverage`.
    /// Budget is sufficient per the union bound; QMC's controlled
    /// correlation is trusted to saturate it. Drawing more cannot raise the
    /// ceiling above `target_coverage` (it's already there).
    TargetMet = 1,
    /// Ceiling saturated at 1.0 (`k·p ≥ 1`) without a hit. More rollouts
    /// provably cannot improve the bound — the event geometry does not
    /// align with this QMC point set. The caller should perturb the QMC
    /// seed or relax the event.
    CeilingSaturated = 2,
    /// k_max safety cap reached. Halt regardless of coverage — protects
    /// against pathological small-p loops that never saturate.
    KMaxReached = 3,
}

// ─────────────────────────────────────────────────────────────────────────────
// QmcHalter config
// ─────────────────────────────────────────────────────────────────────────────

/// Sample-efficiency-aware halter for QMC rollout budgets.
///
/// Configured with a target coverage, a minimum batch floor, and a maximum
/// batch cap. The caller invokes [`Self::evaluate`] after each batch draw
/// with the current `(k, p, n_hits)` observables.
///
/// # Defaults
///
/// `target_coverage = 0.95`, `k_min = 1`, `k_max = 64`. The target matches
/// the conventional pass@k confidence threshold; k_max caps the worst-case
/// budget at the largest LatticeQmc batch size used in Plan 367 benches.
///
/// # Example
///
/// ```no_run
/// # use katgpt_core::speculative::qmc_halter::{QmcHalter, count_hits_1d};
/// let halter = QmcHalter::default();
/// let points = [0.05, 0.15, 0.25, 0.35]; // 4 QMC points
/// let p = 0.10;                          // per-rollout success probability
/// let n_hits = count_hits_1d(&points, p); // → 1 (0.05 < 0.10)
/// let decision = halter.evaluate(points.len(), p, n_hits);
/// // → Halt { reason: HitObserved, coverage: 1.0, ... }
/// ```
#[derive(Clone, Debug)]
pub struct QmcHalter {
    /// Target "at least one success" probability. Halt when the union-bound
    /// ceiling `min(1, k·p)` reaches this AND no hit has been observed yet,
    /// OR halt immediately on the first hit. Default `0.95`.
    pub(crate) target_coverage: f32,
    /// Minimum batch size before any halt is permitted. Default `1`.
    pub(crate) k_min: usize,
    /// Maximum batch size (safety cap). Halt at this k regardless of
    /// coverage. Default `64`.
    pub(crate) k_max: usize,
}

impl QmcHalter {
    /// Construct a halter with explicit config.
    ///
    /// - `target_coverage` — desired `P(at least one success)`. Clamped to
    ///   `(0.0, 1.0]`. A value > 1.0 is clamped to 1.0; a value ≤ 0.0 is
    ///   clamped to the smallest positive f32 (otherwise the halter would
    ///   fire immediately with `TargetMet` on k=0).
    /// - `k_min` — minimum batch before halting (≥ 1 enforced).
    /// - `k_max` — maximum batch cap. Clamped to `k_min.max(1)` so the cap
    ///   is never below the floor.
    #[inline]
    pub fn new(target_coverage: f32, k_min: usize, k_max: usize) -> Self {
        let target_coverage = if target_coverage.is_finite() && target_coverage > 0.0 {
            target_coverage.min(1.0)
        } else {
            f32::MIN_POSITIVE
        };
        let k_min = k_min.max(1);
        let k_max = k_max.max(k_min);
        Self {
            target_coverage,
            k_min,
            k_max,
        }
    }

    /// Target coverage accessor (for diagnostics / serialization).
    #[inline]
    pub fn target_coverage(&self) -> f32 {
        self.target_coverage
    }

    /// k_min accessor.
    #[inline]
    pub fn k_min(&self) -> usize {
        self.k_min
    }

    /// k_max accessor.
    #[inline]
    pub fn k_max(&self) -> usize {
        self.k_max
    }

    /// Decide whether to draw more QMC rollouts.
    ///
    /// # Arguments
    /// - `k` — number of QMC points drawn so far (current batch size).
    /// - `p` — per-rollout success probability (marginal, from the LM).
    ///   NaN-safe: a NaN p is treated as `0.0` (ceiling = 0, no halt on
    ///   TargetMet, halt only if k_max reached).
    /// - `n_hits` — number of the k points that fell in the success region
    ///   (caller computes; use [`count_hits_1d`] for the common 1D case).
    ///
    /// # Evaluation order
    /// 1. **k_min floor** — if `k < k_min`, return [`RefusedFloor`].
    /// 2. **k_max cap** — if `k >= k_max`, return [`Halt`] with [`KMaxReached`].
    /// 3. **Hit observed** — if `n_hits > 0`, return [`Halt`] with
    ///    [`HitObserved`] (early termination — pass@k objective met).
    /// 4. **Ceiling saturated** — if `ceiling ≥ 1.0`, return [`Halt`] with
    ///    [`CeilingSaturated`] (k·p ≥ 1 but no hit; event geometry mismatch).
    /// 5. **Target met** — if `ceiling ≥ target_coverage`, return [`Halt`]
    ///    with [`TargetMet`] (budget sufficient per union bound).
    /// 6. Otherwise [`Continue`] with the current ceiling and gap.
    ///
    /// [`RefusedFloor`]: QmcHaltDecision::RefusedFloor
    /// [`Halt`]: QmcHaltDecision::Halt
    /// [`HitObserved`]: QmcHaltReason::HitObserved
    /// [`CeilingSaturated`]: QmcHaltReason::CeilingSaturated
    /// [`TargetMet`]: QmcHaltReason::TargetMet
    /// [`Continue`]: QmcHaltDecision::Continue
    #[inline]
    pub fn evaluate(&self, k: usize, p: f32, n_hits: usize) -> QmcHaltDecision {
        // 1. k_min floor — refuse to halt below minimum batch.
        if k < self.k_min {
            return QmcHaltDecision::RefusedFloor;
        }

        // 2. k_max cap — safety halt.
        if k >= self.k_max {
            let ceiling = union_bound_ceiling(k, p);
            return QmcHaltDecision::Halt {
                reason: QmcHaltReason::KMaxReached,
                ceiling,
                coverage: if n_hits > 0 { 1.0 } else { ceiling },
            };
        }

        // NaN-safe ceiling: NaN p → 0.0 (no halt on TargetMet, continue).
        let ceiling = union_bound_ceiling(k, p);

        // 3. Hit observed — early termination (pass@k objective met).
        if n_hits > 0 {
            return QmcHaltDecision::Halt {
                reason: QmcHaltReason::HitObserved,
                ceiling,
                coverage: 1.0,
            };
        }

        // 4. Ceiling saturated — k·p ≥ 1 but no hit. Event geometry mismatch;
        //    more rollouts cannot raise the bound above 1.0.
        if ceiling >= 1.0 {
            return QmcHaltDecision::Halt {
                reason: QmcHaltReason::CeilingSaturated,
                ceiling: 1.0,
                coverage: ceiling,
            };
        }

        // 5. Target met — budget sufficient per union bound.
        if ceiling >= self.target_coverage {
            return QmcHaltDecision::Halt {
                reason: QmcHaltReason::TargetMet,
                ceiling,
                coverage: ceiling,
            };
        }

        // 6. Continue — ceiling below target, no hit yet.
        let gap = (self.target_coverage - ceiling).max(0.0);
        QmcHaltDecision::Continue { ceiling, gap }
    }
}

impl Default for QmcHalter {
    /// Default config: `target_coverage = 0.95`, `k_min = 1`, `k_max = 64`.
    ///
    /// These mirror the conventional pass@k confidence threshold (95%) and
    /// the largest LatticeQmc batch used in Plan 367 benches (k=64).
    #[inline]
    fn default() -> Self {
        Self::new(0.95, 1, 64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure modelless helpers (R205 §1 union bound + i.i.d. baseline)
// ─────────────────────────────────────────────────────────────────────────────

/// Union-bound ceiling on `P(at least one success)` for k rollouts of
/// marginal probability p.
///
/// `min(1, k·p)` — the theoretical maximum achievable coverage (R205 §1,
/// Eq. 30–32). QMC's controlled correlation pushes the actual probability
/// toward this ceiling; i.i.d. achieves only `1 − (1−p)^k`, which is
/// strictly below (except at the 0/1 endpoints).
///
/// NaN-safe: a NaN `p` produces a NaN-free `0.0` ceiling (NaN comparisons
/// are false, so `k·p ≥ 1.0` fails and we return `k·p`, but we then clamp
/// negative / NaN to 0). This prevents the halter from spuriously firing
/// `TargetMet` on corrupt input.
///
/// # Example
///
/// ```
/// # use katgpt_core::speculative::qmc_halter::union_bound_ceiling;
/// assert_eq!(union_bound_ceiling(4, 0.1), 0.4);
/// assert_eq!(union_bound_ceiling(20, 0.1), 1.0); // saturated
/// assert_eq!(union_bound_ceiling(0, 0.5), 0.0);
/// ```
#[inline]
pub fn union_bound_ceiling(k: usize, p: f32) -> f32 {
    let raw = k as f32 * p;
    // Clamp negative (p < 0) or NaN to 0 before the min. A NaN p would
    // otherwise propagate through `min` and confuse the halter's `>=`
    // checks (NaN comparisons are false → never fires TargetMet, but the
    // returned ceiling would be NaN, leaking into diagnostics).
    let clamped = if raw.is_finite() && raw > 0.0 {
        raw
    } else {
        0.0
    };
    clamped.min(1.0)
}

/// i.i.d. baseline: `P(at least one success)` for k independent rollouts.
///
/// `1 − (1−p)^k`. This is what i.i.d. sampling achieves — strictly below
/// [`union_bound_ceiling`] except at the endpoints. The gap between the two
/// is exactly what QMC recovers via controlled correlation.
///
/// NaN-safe: returns `0.0` for NaN p (consistent with [`union_bound_ceiling`]).
///
/// # Example
///
/// ```
/// # use katgpt_core::speculative::qmc_halter::iid_at_least_one;
/// // k=1, p=0.5 → 0.5
/// assert!((iid_at_least_one(1, 0.5) - 0.5).abs() < 1e-6);
/// // k=2, p=0.5 → 0.75
/// assert!((iid_at_least_one(2, 0.5) - 0.75).abs() < 1e-6);
/// ```
#[inline]
pub fn iid_at_least_one(k: usize, p: f32) -> f32 {
    if !p.is_finite() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    1.0 - (1.0 - p).powi(k as i32)
}

/// Count QMC points falling in the 1D success region `[0, p)`.
///
/// This is the empirical coverage signal for [`QmcHalter::evaluate`]. Each
/// point `u_i` represents one rollout whose marginal probability of success
/// is `p` (by the QMC marginal-exactness contract). A point in `[0, p)`
/// means that rollout succeeded.
///
/// For multi-dimensional success regions (e.g. a dot-product + sigmoid gate
/// onto a learned direction vector), the caller computes their own `n_hits`
/// and passes it directly to [`QmcHalter::evaluate`] — this helper is only
/// for the common 1D inverse-CDF case.
///
/// NaN-safe: NaN points are never `< p` (NaN comparisons are false), so
/// they are counted as misses. This is the correct direction — a corrupt
/// uniform should not count as a success.
///
/// # Example
///
/// ```
/// # use katgpt_core::speculative::qmc_halter::count_hits_1d;
/// let points = [0.05, 0.15, 0.25, 0.35];
/// assert_eq!(count_hits_1d(&points, 0.10), 1); // only 0.05 < 0.10
/// assert_eq!(count_hits_1d(&points, 0.50), 4); // all four < 0.50
/// ```
#[inline]
pub fn count_hits_1d(points: &[f32], p: f32) -> usize {
    points.iter().filter(|&&u| u < p).count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── union_bound_ceiling ────────────────────────────────────────────────

    #[test]
    fn test_ceiling_basic() {
        assert_eq!(union_bound_ceiling(4, 0.1), 0.4);
        assert_eq!(union_bound_ceiling(1, 0.5), 0.5);
        assert_eq!(union_bound_ceiling(10, 0.05), 0.5);
    }

    #[test]
    fn test_ceiling_saturated() {
        assert_eq!(union_bound_ceiling(20, 0.1), 1.0); // 20 * 0.1 = 2.0 → clamp
        assert_eq!(union_bound_ceiling(100, 0.5), 1.0);
        assert_eq!(union_bound_ceiling(1, 1.0), 1.0);
    }

    #[test]
    fn test_ceiling_zero_k() {
        assert_eq!(union_bound_ceiling(0, 0.5), 0.0);
    }

    #[test]
    fn test_ceiling_negative_p_clamped() {
        assert_eq!(union_bound_ceiling(4, -0.5), 0.0);
    }

    #[test]
    fn test_ceiling_nan_p_is_zero() {
        let c = union_bound_ceiling(4, f32::NAN);
        assert_eq!(c, 0.0);
        assert!(!c.is_nan(), "ceiling must be NaN-free");
    }

    // ── iid_at_least_one ───────────────────────────────────────────────────

    #[test]
    fn test_iid_known_values() {
        assert!((iid_at_least_one(1, 0.5) - 0.5).abs() < 1e-6);
        assert!((iid_at_least_one(2, 0.5) - 0.75).abs() < 1e-6);
        assert!((iid_at_least_one(3, 0.5) - 0.875).abs() < 1e-6);
        assert!((iid_at_least_one(10, 0.1) - 0.65132156).abs() < 1e-5);
    }

    #[test]
    fn test_iid_zero_p() {
        assert_eq!(iid_at_least_one(10, 0.0), 0.0);
    }

    #[test]
    fn test_iid_full_p() {
        assert_eq!(iid_at_least_one(1, 1.0), 1.0);
        assert_eq!(iid_at_least_one(10, 1.0), 1.0);
    }

    #[test]
    fn test_iid_nan_p_is_zero() {
        assert_eq!(iid_at_least_one(10, f32::NAN), 0.0);
    }

    #[test]
    fn test_iid_below_ceiling() {
        // i.i.d. is strictly below the union-bound ceiling except endpoints.
        for &(k, p) in &[(4usize, 0.1f32), (8, 0.05), (16, 0.02), (2, 0.3)] {
            let iid = iid_at_least_one(k, p);
            let ceil = union_bound_ceiling(k, p);
            assert!(
                iid <= ceil + 1e-6,
                "iid {iid} should be <= ceiling {ceil} for k={k}, p={p}"
            );
            // And strictly below when not saturated:
            if ceil < 1.0 {
                assert!(
                    iid < ceil,
                    "iid {iid} should be strictly < ceiling {ceil} for k={k}, p={p}"
                );
            }
        }
    }

    // ── count_hits_1d ──────────────────────────────────────────────────────

    #[test]
    fn test_count_hits_basic() {
        let points = [0.05, 0.15, 0.25, 0.35];
        assert_eq!(count_hits_1d(&points, 0.10), 1);
        assert_eq!(count_hits_1d(&points, 0.20), 2);
        assert_eq!(count_hits_1d(&points, 0.50), 4);
        assert_eq!(count_hits_1d(&points, 0.01), 0);
    }

    #[test]
    fn test_count_hits_empty() {
        let points: [f32; 0] = [];
        assert_eq!(count_hits_1d(&points, 0.5), 0);
    }

    #[test]
    fn test_count_hits_nan_points_excluded() {
        let points = [0.05, f32::NAN, 0.15, f32::NAN];
        assert_eq!(count_hits_1d(&points, 0.50), 2); // NaNs excluded
    }

    #[test]
    fn test_count_hits_nan_p_zero_hits() {
        // NaN p: u < NaN is always false → 0 hits.
        let points = [0.05, 0.15, 0.25];
        assert_eq!(count_hits_1d(&points, f32::NAN), 0);
    }

    // ── QmcHalter::evaluate — RefusedFloor ─────────────────────────────────

    #[test]
    fn test_refused_floor_when_k_below_k_min() {
        let halter = QmcHalter::new(0.95, 4, 64);
        // k=3 < k_min=4 → RefusedFloor regardless of p / n_hits.
        assert_eq!(halter.evaluate(3, 0.5, 1), QmcHaltDecision::RefusedFloor);
    }

    #[test]
    fn test_refused_floor_zero_k() {
        let halter = QmcHalter::default();
        assert_eq!(halter.evaluate(0, 0.5, 0), QmcHaltDecision::RefusedFloor);
    }

    // ── QmcHalter::evaluate — HitObserved ──────────────────────────────────

    #[test]
    fn test_halt_on_hit_observed() {
        let halter = QmcHalter::default();
        // k=4, p=0.5, n_hits=1 → Halt { HitObserved, coverage: 1.0 }.
        let d = halter.evaluate(4, 0.5, 1);
        match d {
            QmcHaltDecision::Halt {
                reason,
                ceiling,
                coverage,
            } => {
                assert_eq!(reason, QmcHaltReason::HitObserved);
                assert!((ceiling - 1.0).abs() < 1e-6, "ceiling = {ceiling}"); // 4 * 0.5 = 2 → 1.0
                assert!((coverage - 1.0).abs() < 1e-6, "coverage = {coverage}");
            }
            _ => panic!("expected Halt, got {d:?}"),
        }
    }

    #[test]
    fn test_halt_on_hit_early_termination() {
        // Even with very small p, if a hit is observed → Halt (pass@k met).
        let halter = QmcHalter::default();
        let d = halter.evaluate(2, 0.01, 1);
        match d {
            QmcHaltDecision::Halt { reason, .. } => {
                assert_eq!(reason, QmcHaltReason::HitObserved);
            }
            _ => panic!("expected Halt on hit, got {d:?}"),
        }
    }

    // ── QmcHalter::evaluate — KMaxReached ──────────────────────────────────

    #[test]
    fn test_halt_kmax_reached() {
        let halter = QmcHalter::new(0.99, 1, 8);
        // k=8 >= k_max=8 → KMaxReached, regardless of p.
        let d = halter.evaluate(8, 0.01, 0);
        match d {
            QmcHaltDecision::Halt {
                reason, ceiling, ..
            } => {
                assert_eq!(reason, QmcHaltReason::KMaxReached);
                assert!((ceiling - 0.08).abs() < 1e-6, "ceiling = {ceiling}");
            }
            _ => panic!("expected Halt KMaxReached, got {d:?}"),
        }
    }

    #[test]
    fn test_kmax_takes_precedence_over_kmin() {
        // If k_min == k_max, k == k_min triggers KMax (not RefusedFloor).
        let halter = QmcHalter::new(0.95, 4, 4);
        let d = halter.evaluate(4, 0.5, 0);
        assert!(
            matches!(d, QmcHaltDecision::Halt { reason, .. } if reason == QmcHaltReason::KMaxReached),
            "expected KMaxReached, got {d:?}"
        );
    }

    // ── QmcHalter::evaluate — CeilingSaturated ────────────────────────────

    #[test]
    fn test_ceiling_saturated_no_hit() {
        // k * p >= 1 but no hit → CeilingSaturated.
        let halter = QmcHalter::default();
        let d = halter.evaluate(20, 0.1, 0); // 20 * 0.1 = 2.0 → saturated, 0 hits
        match d {
            QmcHaltDecision::Halt {
                reason, ceiling, ..
            } => {
                assert_eq!(reason, QmcHaltReason::CeilingSaturated);
                assert!((ceiling - 1.0).abs() < 1e-6, "ceiling = {ceiling}");
            }
            _ => panic!("expected CeilingSaturated, got {d:?}"),
        }
    }

    // ── QmcHalter::evaluate — TargetMet ────────────────────────────────────

    #[test]
    fn test_target_met_when_ceiling_reaches_target() {
        // target = 0.95, k * p = 0.95 exactly → TargetMet.
        let halter = QmcHalter::new(0.95, 1, 64);
        let d = halter.evaluate(19, 0.05, 0); // 19 * 0.05 = 0.95 → exactly target
        match d {
            QmcHaltDecision::Halt {
                reason, ceiling, ..
            } => {
                assert_eq!(reason, QmcHaltReason::TargetMet);
                assert!((ceiling - 0.95).abs() < 1e-6, "ceiling = {ceiling}");
            }
            _ => panic!("expected TargetMet, got {d:?}"),
        }
    }

    #[test]
    fn test_target_met_above_target() {
        // ceiling = 0.96 > target 0.95 → TargetMet (no hit).
        let halter = QmcHalter::new(0.95, 1, 64);
        let d = halter.evaluate(24, 0.04, 0); // 24 * 0.04 = 0.96
        match d {
            QmcHaltDecision::Halt {
                reason, ceiling, ..
            } => {
                assert_eq!(reason, QmcHaltReason::TargetMet);
                assert!((ceiling - 0.96).abs() < 1e-6);
            }
            _ => panic!("expected TargetMet, got {d:?}"),
        }
    }

    // ── QmcHalter::evaluate — Continue ─────────────────────────────────────

    #[test]
    fn test_continue_when_ceiling_below_target() {
        // target = 0.95, k * p = 0.4 → Continue, gap = 0.55.
        let halter = QmcHalter::default();
        let d = halter.evaluate(4, 0.1, 0); // ceiling = 0.4
        match d {
            QmcHaltDecision::Continue { ceiling, gap } => {
                assert!((ceiling - 0.4).abs() < 1e-6, "ceiling = {ceiling}");
                assert!((gap - 0.55).abs() < 1e-6, "gap = {gap}");
            }
            _ => panic!("expected Continue, got {d:?}"),
        }
    }

    #[test]
    fn test_continue_reports_correct_gap() {
        // Edge: target = 0.95, ceiling = 0.9 (just below) → Continue, gap = 0.05.
        // Verifies the gap field is the simple `target - ceiling` difference.
        let halter = QmcHalter::new(0.95, 1, 64);
        let d = halter.evaluate(18, 0.05, 0); // 18 * 0.05 = 0.9
        match d {
            QmcHaltDecision::Continue { ceiling, gap } => {
                assert!((ceiling - 0.9).abs() < 1e-6, "ceiling = {ceiling}");
                assert!((gap - 0.05).abs() < 1e-6, "gap = {gap}");
            }
            _ => panic!("expected Continue, got {d:?}"),
        }
    }

    // ── HitObserved precedence ─────────────────────────────────────────────

    #[test]
    fn test_hit_observed_beats_target_met() {
        // Both HitObserved and TargetMet conditions hold → HitObserved wins
        // (it's checked first and is the stronger signal — actual success).
        let halter = QmcHalter::default();
        let d = halter.evaluate(20, 0.1, 1); // ceiling saturated AND hit
        match d {
            QmcHaltDecision::Halt { reason, .. } => {
                assert_eq!(reason, QmcHaltReason::HitObserved);
            }
            _ => panic!("expected Halt HitObserved, got {d:?}"),
        }
    }

    // ── NaN safety ─────────────────────────────────────────────────────────

    #[test]
    fn test_evaluate_nan_p_continues_or_kmax() {
        // NaN p → ceiling = 0 → Continue (unless k_max reached).
        let halter = QmcHalter::new(0.95, 1, 64);
        let d = halter.evaluate(4, f32::NAN, 0);
        match d {
            QmcHaltDecision::Continue { ceiling, gap } => {
                assert!(!ceiling.is_nan(), "ceiling must be NaN-free, got {ceiling}");
                assert_eq!(ceiling, 0.0);
                assert!((gap - 0.95).abs() < 1e-6, "gap = {gap}");
            }
            _ => panic!("expected Continue on NaN p, got {d:?}"),
        }
    }

    // ── Default config ─────────────────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let h = QmcHalter::default();
        assert!((h.target_coverage() - 0.95).abs() < 1e-6);
        assert_eq!(h.k_min(), 1);
        assert_eq!(h.k_max(), 64);
    }

    #[test]
    fn test_new_clamps_target_above_one() {
        let h = QmcHalter::new(2.0, 1, 10);
        assert!((h.target_coverage() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_new_clamps_target_zero_or_negative() {
        let h = QmcHalter::new(0.0, 1, 10);
        assert!(h.target_coverage() > 0.0);
        assert!(h.target_coverage() <= 1.0);
    }

    #[test]
    fn test_new_clamps_kmin_at_least_one() {
        let h = QmcHalter::new(0.95, 0, 10);
        assert_eq!(h.k_min(), 1);
    }

    #[test]
    fn test_new_ensures_kmax_at_least_kmin() {
        let h = QmcHalter::new(0.95, 8, 4); // k_max < k_min
        assert!(h.k_max() >= h.k_min());
        assert_eq!(h.k_max(), 8);
    }

    // ── QMC advantage demonstration ────────────────────────────────────────

    #[test]
    fn test_qmc_advantage_ceiling_above_iid() {
        // For a range of (k, p), demonstrate that the union-bound ceiling
        // exceeds the i.i.d. baseline. This is the theoretical justification
        // for QmcHalter: QMC can saturate the ceiling, i.i.d. cannot.
        let cases = [
            (4usize, 0.1f32),
            (8, 0.05),
            (10, 0.05),
            (16, 0.02),
            (20, 0.01),
            (3, 0.2),
        ];
        for &(k, p) in &cases {
            let ceil = union_bound_ceiling(k, p);
            let iid = iid_at_least_one(k, p);
            // Skip saturated cases (both clamp at 1.0).
            if ceil < 1.0 {
                assert!(
                    ceil > iid,
                    "QMC advantage should hold: ceiling {ceil} > iid {iid} for k={k}, p={p}"
                );
                let advantage = ceil - iid;
                // The advantage should be meaningful (not rounding noise).
                assert!(
                    advantage > 1e-4,
                    "QMC advantage {advantage} too small for k={k}, p={p}"
                );
            }
        }
    }

    #[test]
    fn test_qmc_saves_rollouts_vs_iid_budget() {
        // To achieve target = 0.95 with p = 0.1:
        // - i.i.d. needs k >= log(0.05) / log(0.9) ≈ 28.4 → k = 29
        // - QMC (union-bound) needs k * 0.1 >= 0.95 → k >= 9.5 → k = 10
        // So QMC uses ~3× fewer rollouts. This is the sample-efficiency claim.
        let target = 0.95f32;
        let p = 0.1f32;

        // i.i.d. budget:
        let iid_k = (1..100)
            .find(|&k| iid_at_least_one(k, p) >= target)
            .expect("i.i.d. should reach target within k=100");
        // QMC budget (union-bound):
        let qmc_k = (1..100)
            .find(|&k| union_bound_ceiling(k, p) >= target)
            .expect("QMC should reach target within k=100");

        assert!(qmc_k < iid_k, "QMC k={qmc_k} should be < i.i.d. k={iid_k}");
        // Sanity: i.i.d. budget for p=0.1, target=0.95 is ~29.
        assert_eq!(iid_k, 29, "i.i.d. budget for p=0.1, target=0.95");
        assert_eq!(qmc_k, 10, "QMC budget for p=0.1, target=0.95");
    }

    // ── End-to-end: halter drives an adaptive QMC draw loop ────────────────

    #[test]
    fn test_end_to_end_halt_after_target_budget() {
        // Simulate a caller drawing QMC points in batches and checking the
        // halter after each batch. p = 0.05, target = 0.5. No hits observed
        // (worst case). The halter should fire TargetMet at k=10 (10*0.05 =
        // 0.5 = target, below saturation).
        let halter = QmcHalter::new(0.5, 1, 64);
        let p = 0.05f32;
        let mut k = 0;
        let mut final_decision = QmcHaltDecision::RefusedFloor;
        while k < 100 {
            k += 1;
            final_decision = halter.evaluate(k, p, 0); // no hits
            if !matches!(final_decision, QmcHaltDecision::Continue { .. }) {
                break;
            }
        }
        match final_decision {
            QmcHaltDecision::Halt {
                reason, ceiling, ..
            } => {
                assert_eq!(reason, QmcHaltReason::TargetMet);
                assert_eq!(k, 10, "should halt at k=10 (k*p = 0.5 >= 0.5 target)");
                assert!((ceiling - 0.5).abs() < 1e-6);
            }
            _ => panic!("expected Halt, got {final_decision:?} at k={k}"),
        }
    }

    #[test]
    fn test_end_to_end_halt_on_ceiling_saturation() {
        // Companion to the above: target = 0.95, p = 0.1. At k=10 the ceiling
        // hits 1.0 (saturated) before reaching the target via the TargetMet
        // branch (which would need ceiling < 1.0). The halter should fire
        // CeilingSaturated at k=10.
        let halter = QmcHalter::new(0.95, 1, 64);
        let p = 0.1f32;
        let mut k = 0;
        let mut final_decision = QmcHaltDecision::RefusedFloor;
        while k < 100 {
            k += 1;
            final_decision = halter.evaluate(k, p, 0); // no hits
            if !matches!(final_decision, QmcHaltDecision::Continue { .. }) {
                break;
            }
        }
        match final_decision {
            QmcHaltDecision::Halt {
                reason, ceiling, ..
            } => {
                // k=10 → ceiling = min(1, 1.0) = 1.0 → CeilingSaturated
                // (step 4 fires before step 5 TargetMet).
                assert_eq!(reason, QmcHaltReason::CeilingSaturated);
                assert_eq!(k, 10, "should halt at k=10 (ceiling saturated at 1.0)");
                assert!((ceiling - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected Halt, got {final_decision:?} at k={k}"),
        }
    }

    #[test]
    fn test_end_to_end_halt_on_first_hit() {
        // Simulate a draw loop where the third rollout hits. The halter
        // should fire HitObserved at k=3.
        let halter = QmcHalter::default();
        let p = 0.1f32;
        let hit_at_k = 3; // pretend the 3rd point is in [0, p)
        let mut final_decision = QmcHaltDecision::RefusedFloor;
        for k in 1..=64 {
            let n_hits = if k >= hit_at_k { 1 } else { 0 };
            final_decision = halter.evaluate(k, p, n_hits);
            if !matches!(final_decision, QmcHaltDecision::Continue { .. }) {
                assert_eq!(k, hit_at_k, "should halt at first hit k={hit_at_k}");
                break;
            }
        }
        match final_decision {
            QmcHaltDecision::Halt {
                reason, coverage, ..
            } => {
                assert_eq!(reason, QmcHaltReason::HitObserved);
                assert!((coverage - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected Halt HitObserved, got {final_decision:?}"),
        }
    }

    // ── Copy + Debug ───────────────────────────────────────────────────────

    #[test]
    fn test_decision_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<QmcHaltDecision>();
        assert_copy::<QmcHaltReason>();
    }

    #[test]
    fn test_decision_debug_formats() {
        let d = QmcHaltDecision::Continue {
            ceiling: 0.5,
            gap: 0.45,
        };
        let s = format!("{d:?}");
        assert!(s.contains("Continue"));
        assert!(s.contains("0.5"));
    }

    // ── Latency gate (run with --release --ignored) ────────────────────────

    #[test]
    #[ignore]
    fn bench_evaluate_latency() {
        let halter = QmcHalter::default();
        let p = 0.1f32;
        // Warm up
        for _ in 0..1000 {
            let _ = halter.evaluate(8, p, 0);
        }
        // Measure — vary k and n_hits each iteration so the compiler cannot
        // constant-fold the decision (the loop-invariant version measured
        // 0.00 ns/call, which is a DCE artifact, not a real measurement).
        const N: u32 = 1_000_000;
        let start = std::time::Instant::now();
        let mut sink = 0u64;
        for i in 0..N {
            let k = ((i as usize) & 63) + 1; // k in [1, 64]
            let n_hits = if (i & 7) == 0 { 1 } else { 0 }; // occasional hit
            let d = halter.evaluate(k, p, n_hits);
            // Sink depends on the decision output → prevents elision.
            sink = sink.wrapping_add(i as u64);
            match d {
                QmcHaltDecision::Continue { ceiling, gap } => {
                    sink = sink.wrapping_add(ceiling.to_bits() as u64);
                    sink = sink.wrapping_add(gap.to_bits() as u64);
                }
                QmcHaltDecision::Halt {
                    ceiling, coverage, ..
                } => {
                    sink = sink.wrapping_add(ceiling.to_bits() as u64);
                    sink = sink.wrapping_add(coverage.to_bits() as u64);
                }
                QmcHaltDecision::RefusedFloor => {}
            }
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / N as f64;
        eprintln!("bench_evaluate_latency: {ns_per_call:.2} ns/call (sink={sink}, budget=50 ns)");
        // Budget: 50 ns (the decision is a few float ops + branches).
        assert!(
            ns_per_call < 50.0,
            "evaluate latency {ns_per_call:.2} ns exceeds 50 ns budget"
        );
    }

    #[test]
    #[ignore]
    fn bench_union_bound_ceiling_latency() {
        // Warm up
        for _ in 0..1000 {
            let _ = union_bound_ceiling(8, 0.1);
        }
        const N: u32 = 1_000_000;
        let start = std::time::Instant::now();
        let mut sink = 0u64;
        for i in 0..N {
            let c = union_bound_ceiling(((i as usize) & 31) + 1, 0.05);
            sink = sink.wrapping_add(c.to_bits() as u64);
        }
        let elapsed = start.elapsed();
        let ns_per_call = elapsed.as_nanos() as f64 / N as f64;
        eprintln!(
            "bench_union_bound_ceiling_latency: {ns_per_call:.2} ns/call (sink={sink}, budget=10 ns)"
        );
        assert!(
            ns_per_call < 10.0,
            "union_bound_ceiling latency {ns_per_call:.2} ns exceeds 10 ns budget"
        );
    }
}
