//! Head substitution gate — cheap-proxy IoU + cached FaithfulnessProbe veto.
//!
//! Implements the cadence pattern (Plan 287 SinkAware) for deciding when a
//! [`FuncAttn`](crate::funcattn)-style surrogate should substitute for a real
//! attention head on the current forward pass:
//!
//! 1. **Cheap proxy (per-call)**: IoU between the real head's attention row
//!    and the surrogate. Below `tau_iou`, reject immediately.
//! 2. **Expensive veto (cached, audit cadence)**: worst-case behavioral delta
//!    from the cached [`FaithfulnessProfile`]. Above `tau_behavior`, reject.
//!
//! Both checks must pass for the gate to fire. The expensive probe is **not**
//! re-run per token — it is cached per-head and refreshed on an audit cadence
//! (Plan 287 SinkAware pattern). The hot path is a pure decision over those
//! cached measurements: no allocation, no I/O.

use crate::faithfulness::types::FaithfulnessProfile;

/// Conservative worst-case behavioral delta across the FaithfulnessProbe
/// intervention suite.
///
/// The probe runs five interventions: Empty, Shuffle, Corrupt, Irrelevant,
/// Filler. Three of them (`Shuffle`/`Corrupt` aggregated as
/// `shuffle_or_corrupt_delta`, plus `irrelevant_delta` and `filler_delta`)
/// measure how much the consumer's behavior changes when the memory is
/// disrupted. The `empty_delta` is excluded: it measures the *graceful
/// absence* baseline (a faithful consumer returns to baseline behavior — small
/// is good, not bad).
///
/// For substitution decisions we want the conservative worst case: if **any**
/// disruption produces a large behavioral delta, the head is causally
/// load-bearing and substituting it is risky. The gate vetoes substitution
/// when this worst-case delta exceeds `tau_behavior`.
///
/// `D` matches [`FaithfulnessProfile`]'s type parameter (typically `f32`).
#[inline]
pub fn worst_case_behavior_delta<D>(profile: &FaithfulnessProfile<D>) -> D
where
    D: PartialOrd + Copy + Default,
{
    // `shuffle_or_corrupt_delta` is already max(Shuffle, Corrupt). Take the
    // max of that with irrelevant and filler. Branch-free via fold-style
    // chained `max` — lowers to a sequence of `maxss` on x86-64.
    let mut m = profile.shuffle_or_corrupt_delta;
    if profile.irrelevant_delta > m {
        m = profile.irrelevant_delta;
    }
    if profile.filler_delta > m {
        m = profile.filler_delta;
    }
    m
}

/// Gate that decides whether to substitute a real attention head with a
/// [`FuncAttn`](crate::funcattn)-style surrogate during a forward pass.
///
/// Combines the source paper's IoU gate (cheap proxy, paper §3 Fig 5b
/// `r > 0.9`) with [`FaithfulnessProbe`](crate::faithfulness) (expensive
/// validation, cached at audit cadence per Plan 287 SinkAware pattern).
///
/// # Modelless discipline
///
/// The gate performs **zero training, zero backprop**. Surrogates arrive as
/// pre-constructed FuncAttn-compatible callables owned by the caller; the gate
/// only decides when to apply them. The cached [`FaithfulnessProfile`] is a
/// pure measurement, not a learned quantity.
///
/// # What the gate does NOT hold
///
/// Per Plan 353's revision note, the gate does **not** hold the surrogate
/// itself. The caller owns the FuncAttn instance. This keeps the gate a pure
/// decision function and avoids duplicating FuncAttn's existing primitive
/// surface (the original plan proposed a redundant `ProgramSynthesizedHead`
/// primitive; that was dropped after re-review identified FuncAttn as the
/// existing primitive).
///
/// # Field genericity
///
/// The cached profiles are generic over the delta metric `D` to mirror
/// [`FaithfulnessProfile<D>`]. The default hot-path specialization is `f32`.
pub struct HeadSubstitutionGate<D = f32>
where
    D: PartialOrd + Copy + Default,
{
    /// IoU threshold below which substitution is rejected outright.
    /// Paper default ~0.4 (paper §3 reports 25–40% of GPT-2 heads are
    /// programmable; the IoU gate at ~0.4 selects that population).
    tau_iou: f32,
    /// Behavioral-tolerance threshold. A head whose worst-case
    /// FaithfulnessProbe delta exceeds this is vetoed (the head is causally
    /// load-bearing). Paper §3 reports perplexity deltas ≤ ~16% for safe
    /// substitutions — set this in the same order of magnitude.
    tau_behavior: D,
    /// Cached [`FaithfulnessProfile`] per head, indexed by head id `h`.
    /// Refreshed at audit cadence (Plan 287 SinkAware), NOT per-token.
    cached_faithfulness: Vec<FaithfulnessProfile<D>>,
}

impl<D> HeadSubstitutionGate<D>
where
    D: PartialOrd + Copy + Default,
{
    /// Construct a gate with the given thresholds and cached profiles.
    ///
    /// `cached_faithfulness[h]` must be populated for every head id `h` that
    /// will be queried via [`should_substitute`](Self::should_substitute).
    /// Heads beyond the cache length always return `false` (defensive —
    /// un-profiled heads are not substituted).
    #[inline]
    pub fn new(
        tau_iou: f32,
        tau_behavior: D,
        cached_faithfulness: Vec<FaithfulnessProfile<D>>,
    ) -> Self {
        Self {
            tau_iou,
            tau_behavior,
            cached_faithfulness,
        }
    }

    /// Construct an empty gate (no cached profiles — `should_substitute`
    /// always returns `false`). Useful for callers that want the IoU-only
    /// fast-reject path without committing to a full faithfulness audit.
    #[inline]
    pub fn empty(tau_iou: f32, tau_behavior: D) -> Self {
        Self {
            tau_iou,
            tau_behavior,
            cached_faithfulness: Vec::new(),
        }
    }

    /// Replace the cached faithfulness profiles. Call this on the audit
    /// cadence (Plan 287 SinkAware) — NOT per-token.
    ///
    /// Allocates only on this refresh path; the hot path
    /// ([`should_substitute`](Self::should_substitute)) is alloc-free.
    #[inline]
    pub fn refresh_cache(&mut self, cached_faithfulness: Vec<FaithfulnessProfile<D>>) {
        self.cached_faithfulness = cached_faithfulness;
    }

    /// Update the cached profile for a single head `h`. Grows the cache if
    /// needed (audit-cadence path only).
    pub fn update_head(&mut self, h: usize, profile: FaithfulnessProfile<D>) {
        if h >= self.cached_faithfulness.len() {
            self.cached_faithfulness
                .resize(h + 1, FaithfulnessProfile::default_at());
        }
        self.cached_faithfulness[h] = profile;
    }

    /// IoU threshold accessor (for tests / diagnostics).
    #[inline]
    pub fn tau_iou(&self) -> f32 {
        self.tau_iou
    }

    /// Behavioral-tolerance threshold accessor (for tests / diagnostics).
    #[inline]
    pub fn tau_behavior(&self) -> D {
        self.tau_behavior
    }

    /// Number of heads currently in the cached profile.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.cached_faithfulness.len()
    }

    /// Read-only access to a head's cached profile. Returns `None` if `h` is
    /// beyond the cache (not yet profiled).
    #[inline]
    pub fn cached_profile(&self, h: usize) -> Option<&FaithfulnessProfile<D>> {
        self.cached_faithfulness.get(h)
    }

    /// Hot-path decision: should head `h` be replaced by its surrogate on
    /// this forward pass?
    ///
    /// Pure decision over cached measurements — no I/O, no allocation. The
    /// actual surrogate callable is supplied separately by the caller
    /// (typically a [`FuncAttn`](crate::funcattn) instance).
    ///
    /// # Decision rule
    ///
    /// 1. `head_iou < tau_iou` → reject (cheap-proxy gate fails).
    /// 2. Head `h` is beyond the cached profile length → reject (un-profiled).
    /// 3. `worst_case_behavior_delta(cached[h]) > tau_behavior` → reject
    ///    (faithfulness veto: the head is causally load-bearing).
    /// 4. Otherwise → accept.
    ///
    /// All four branches are branch-predictor-friendly: the common production
    /// case is step 1 (the cheap proxy rejects most heads before the cached
    /// profile is even consulted).
    #[inline]
    pub fn should_substitute(&self, h: usize, head_iou: f32) -> bool {
        // Step 1 — cheap proxy. The common case; placed first so the branch
        // predictor learns "usually false here".
        if head_iou < self.tau_iou {
            return false;
        }
        // Step 2 — bounds check. `.get()` returns `Option` without panicking;
        // the `match` keeps the hot path branch-free on the accept arm.
        let profile = match self.cached_faithfulness.get(h) {
            None => return false,
            Some(p) => p,
        };
        // Step 3 — faithfulness veto. Worst-case delta must be tolerable.
        worst_case_behavior_delta(profile) <= self.tau_behavior
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Default-`D` helper trait — lets `HeadSubstitutionGate::default_at` and
// `update_head` produce an all-zero profile without forcing the call site to
// spell out `FaithfulnessProfile { empty_delta: D::default(), ... }`.
// We don't put `Default` on `FaithfulnessProfile` itself because the existing
// `types.rs` intentionally omits it (D's bound is `PartialOrd + Copy`, not
// `Default`); adding `Default` as a super-trait there would be a wider change
// than Plan 353 warrants.
// ──────────────────────────────────────────────────────────────────────────

impl<D> FaithfulnessProfile<D>
where
    D: PartialOrd + Copy + Default,
{
    /// Construct an all-default profile (every delta = `D::default()`).
    /// Used by [`HeadSubstitutionGate::update_head`] to backfill un-profiled
    /// head slots; semantically equivalent to "no behavior change observed
    /// under any intervention" (i.e. a non-causally-load-bearing head).
    #[inline]
    pub fn default_at() -> Self {
        Self {
            empty_delta: D::default(),
            shuffle_or_corrupt_delta: D::default(),
            irrelevant_delta: D::default(),
            filler_delta: D::default(),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Unit tests — G1 correctness for the gate (Plan 353 T2.1)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(empty: f32, shuf: f32, irrel: f32, fill: f32) -> FaithfulnessProfile<f32> {
        FaithfulnessProfile {
            empty_delta: empty,
            shuffle_or_corrupt_delta: shuf,
            irrelevant_delta: irrel,
            filler_delta: fill,
        }
    }

    /// Worst-case delta picks the max of shuffle/corrupt, irrelevant, filler
    /// (NOT empty — empty is the graceful-absence baseline, not a disruption).
    #[test]
    fn worst_case_picks_max_of_disruptive_interventions() {
        // shuffle is the max.
        let p = profile(0.0, 5.0, 3.0, 2.0);
        assert_eq!(worst_case_behavior_delta(&p), 5.0);
        // irrelevant is the max.
        let p = profile(0.0, 1.0, 7.0, 2.0);
        assert_eq!(worst_case_behavior_delta(&p), 7.0);
        // filler is the max.
        let p = profile(0.0, 1.0, 2.0, 9.0);
        assert_eq!(worst_case_behavior_delta(&p), 9.0);
        // empty is large but NOT counted.
        let p = profile(99.0, 1.0, 2.0, 3.0);
        assert_eq!(worst_case_behavior_delta(&p), 3.0);
    }

    /// G1 (Plan 353 T2.1): identity surrogate (IoU=1.0, all-zero deltas)
    /// is accepted.
    #[test]
    fn g1_identity_surrogate_accepted() {
        let zero = profile(0.0, 0.0, 0.0, 0.0);
        let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![zero]);
        assert!(gate.should_substitute(0, 1.0));
    }

    /// G1: disjoint surrogate (IoU=0.0) is rejected regardless of
    /// faithfulness profile.
    #[test]
    fn g1_disjoint_surrogate_rejected_regardless_of_faithfulness() {
        // Even a fully-zero profile (worst-case delta = 0) cannot rescue a
        // disjoint surrogate.
        let zero = profile(0.0, 0.0, 0.0, 0.0);
        let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![zero]);
        assert!(!gate.should_substitute(0, 0.0));
    }

    /// G1: partial-overlap surrogate at known IoU is accepted iff
    /// `tau_iou <= iou AND worst_case_delta <= tau_behavior`.
    #[test]
    fn g1_partial_overlap_boundary_conditions() {
        let small_delta = profile(0.0, 0.1, 0.1, 0.1); // worst = 0.1
        let large_delta = profile(0.0, 0.5, 0.5, 0.5); // worst = 0.5

        // tau_iou = 0.4, tau_behavior = 0.16
        // IoU = 0.4 → exactly at the boundary; strict `<` means 0.4 is accepted.
        let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![small_delta, large_delta]);
        assert!(
            gate.should_substitute(0, 0.4),
            "boundary IoU accepted for head 0"
        );
        // Same IoU, but head 1 has a large delta → vetoed.
        assert!(!gate.should_substitute(1, 0.4), "large-delta head vetoed");
        // Just below threshold → rejected.
        assert!(!gate.should_substitute(0, 0.399), "below tau_iou rejected");
    }

    /// G1: high IoU but high behavior delta → rejected (faithfulness veto).
    #[test]
    fn g1_high_iou_high_delta_rejected_by_faithfulness_veto() {
        let load_bearing = profile(0.0, 0.9, 0.9, 0.9); // worst = 0.9
        let gate = HeadSubstitutionGate::new(0.4, 0.16, vec![load_bearing]);
        // IoU = 1.0 passes the cheap proxy, but the faithfulness veto fires.
        assert!(!gate.should_substitute(0, 1.0));
    }

    /// Un-profiled head (beyond cache) → rejected defensively.
    #[test]
    fn unprofiled_head_rejected() {
        let gate: HeadSubstitutionGate<f32> = HeadSubstitutionGate::empty(0.4, 0.16);
        assert!(!gate.should_substitute(0, 1.0));
    }

    /// `update_head` grows the cache and backfills un-profiled slots with
    /// the all-default profile.
    #[test]
    fn update_head_grows_cache() {
        let mut gate: HeadSubstitutionGate<f32> = HeadSubstitutionGate::empty(0.4, 0.16);
        assert_eq!(gate.num_heads(), 0);
        // Update head 2 → cache grows to length 3 with default profiles for 0, 1.
        gate.update_head(2, profile(0.0, 0.1, 0.1, 0.1));
        assert_eq!(gate.num_heads(), 3);
        // Heads 0 and 1 were backfilled with default profiles (worst-case = 0),
        // so they pass the faithfulness veto at IoU = 1.0.
        assert!(gate.should_substitute(0, 1.0));
        assert!(gate.should_substitute(1, 1.0));
        // Head 2 has the small delta we set → also accepted.
        assert!(gate.should_substitute(2, 1.0));
    }

    /// `refresh_cache` swaps the whole cache (audit-cadence path).
    #[test]
    fn refresh_cache_swaps_whole_cache() {
        let mut gate = HeadSubstitutionGate::new(
            0.4,
            0.16,
            vec![profile(0.0, 0.9, 0.9, 0.9)], // head 0 vetoed
        );
        assert!(!gate.should_substitute(0, 1.0));
        gate.refresh_cache(vec![profile(0.0, 0.0, 0.0, 0.0)]); // head 0 cleared
        assert!(gate.should_substitute(0, 1.0));
    }

    /// Accessors return the configured thresholds.
    #[test]
    fn accessors_return_configured_thresholds() {
        let gate = HeadSubstitutionGate::new(0.42, 0.17, vec![]);
        assert_eq!(gate.tau_iou(), 0.42);
        assert_eq!(gate.tau_behavior(), 0.17);
        assert_eq!(gate.num_heads(), 0);
    }
}
