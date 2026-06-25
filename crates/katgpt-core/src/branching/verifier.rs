//! `VerifierGate` — reward + curiosity + centroid-quarantine write gate
//! (Plan 329 T1.5).
//!
//! Composes with CLR's `should_write_memory(r_k, S_LP)` (Plan 284): CLR is the
//! upstream two-sided reward gate (`r_k > τ_reliable ∧ S_LP > τ_curiosity`);
//! the `VerifierGate` adds a branch-centroid quarantine check downstream.
//!
//! # Decision rule
//!
//! ```text
//! if reward <= tau_write         → Reject   (reward too low)
//! if curiosity <= tau_curiosity  → Reject   (curiosity too low)
//! if branch_centroid_sim < quarantine_centroid_thresh
//!                                → Quarantine (off-centroid, possible contamination)
//! else                           → Write    (all gates pass)
//! ```
//!
//! `Quarantine` is a soft-reject: the write is held for later review (e.g., a
//! human-in-the-loop or a slower offline verifier). It does NOT enter the
//! branch's episodic store until promoted.
//!
//! # Hot path
//!
//! `should_write` is a pure function of three `f32` scalars — zero allocation,
//! three comparisons, early returns. Suitable for per-tick per-NPC calls.

/// Default reward threshold for write admission (CLR-aligned).
pub const DEFAULT_TAU_WRITE: f32 = 0.5;

/// Default curiosity threshold for write admission (CLR `S_LP`-aligned).
pub const DEFAULT_TAU_CURIOSITY: f32 = 0.3;

/// Default branch-centroid similarity threshold below which a write is
/// quarantined as a potential contamination source.
pub const DEFAULT_QUARANTINE_CENTROID_THRESH: f32 = 0.5;

/// Verifier-gated write decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WriteDecision {
    /// All gates pass; write the entry to the branch's episodic store.
    Write = 0,
    /// Off-centroid; hold the write for review before admitting.
    Quarantine = 1,
    /// Reward or curiosity too low; reject outright.
    Reject = 2,
}

impl WriteDecision {
    /// True if the decision admits the write immediately (`Write`).
    #[inline]
    #[must_use]
    pub const fn is_write(self) -> bool {
        matches!(self, Self::Write)
    }

    /// True if the write should be held for review (`Quarantine`).
    #[inline]
    #[must_use]
    pub const fn is_quarantine(self) -> bool {
        matches!(self, Self::Quarantine)
    }

    /// True if the write is rejected outright (`Reject`).
    #[inline]
    #[must_use]
    pub const fn is_reject(self) -> bool {
        matches!(self, Self::Reject)
    }

    /// True if the write is NOT admitted to the active store (`Quarantine` or
    /// `Reject`). The inverse of [`is_write`](Self::is_write).
    #[inline]
    #[must_use]
    pub const fn is_blocked(self) -> bool {
        !self.is_write()
    }
}

/// Reward + curiosity + centroid-quarantine write gate.
///
/// Construct with [`VerifierGate::default`] for CLR-aligned thresholds, or
/// [`VerifierGate::new`] for custom thresholds.
#[derive(Clone, Copy, Debug)]
pub struct VerifierGate {
    /// Reward `r ∈ [0,1]` must be strictly greater than this to admit.
    pub tau_write: f32,
    /// Curiosity `S_LP ∈ [0,1]` must be strictly greater than this to admit.
    pub tau_curiosity: f32,
    /// Branch-centroid similarity below this → quarantine (off-centroid).
    pub quarantine_centroid_thresh: f32,
}

impl Default for VerifierGate {
    #[inline]
    fn default() -> Self {
        Self {
            tau_write: DEFAULT_TAU_WRITE,
            tau_curiosity: DEFAULT_TAU_CURIOSITY,
            quarantine_centroid_thresh: DEFAULT_QUARANTINE_CENTROID_THRESH,
        }
    }
}

impl VerifierGate {
    /// Construct with custom thresholds.
    #[inline]
    #[must_use]
    pub const fn new(
        tau_write: f32,
        tau_curiosity: f32,
        quarantine_centroid_thresh: f32,
    ) -> Self {
        Self {
            tau_write,
            tau_curiosity,
            quarantine_centroid_thresh,
        }
    }

    /// Decide whether to write, quarantine, or reject.
    ///
    /// - `reward`: verifier score `r ∈ [0,1]` (CLR `r_k`).
    /// - `curiosity`: learning-potential score `S_LP ∈ [0,1]` (CLR curiosity).
    /// - `branch_centroid_sim`: cosine similarity between the write's embedding
    ///   and the branch's centroid (computed by the caller; the branch's
    ///   centroid is the mean of its episodic embeddings). Values near 1.0 mean
    ///   the write is on-distribution for this branch; near 0.0 means it's
    ///   off-distribution and may contaminate the branch.
    ///
    /// Returns [`WriteDecision`]. Zero alloc, three comparisons, early returns.
    #[inline]
    #[must_use]
    pub fn should_write(
        &self,
        reward: f32,
        curiosity: f32,
        branch_centroid_sim: f32,
    ) -> WriteDecision {
        // Gate 1: reward (CLR upstream — `r_k > τ_reliable`).
        if reward <= self.tau_write {
            return WriteDecision::Reject;
        }
        // Gate 2: curiosity (CLR upstream — `S_LP > τ_curiosity`).
        if curiosity <= self.tau_curiosity {
            return WriteDecision::Reject;
        }
        // Gate 3: branch-centroid quarantine (RIZZ downstream addition).
        if branch_centroid_sim < self.quarantine_centroid_thresh {
            return WriteDecision::Quarantine;
        }
        WriteDecision::Write
    }

    /// Convenience: compose with CLR's `should_write_memory(r_k, S_LP)` as the
    /// upstream gate. Returns `Reject` if CLR rejects, otherwise delegates to
    /// [`should_write`](Self::should_write) for the centroid check.
    ///
    /// This is the canonical composition pattern from Research 310 §2.2:
    /// CLR is the upstream reward+curiosity gate; `VerifierGate` adds the
    /// branch-centroid quarantine downstream.
    ///
    /// - `clr_allows`: output of CLR's `should_write_memory(r_k, S_LP)`.
    /// - `branch_centroid_sim`: cosine similarity to the branch centroid.
    #[inline]
    #[must_use]
    pub fn should_write_composed(
        &self,
        clr_allows: bool,
        branch_centroid_sim: f32,
    ) -> WriteDecision {
        if !clr_allows {
            return WriteDecision::Reject;
        }
        if branch_centroid_sim < self.quarantine_centroid_thresh {
            return WriteDecision::Quarantine;
        }
        WriteDecision::Write
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_when_all_gates_pass() {
        let gate = VerifierGate::default();
        let d = gate.should_write(0.8, 0.5, 0.9);
        assert_eq!(d, WriteDecision::Write);
        assert!(d.is_write());
        assert!(!d.is_blocked());
    }

    #[test]
    fn reject_when_reward_too_low() {
        let gate = VerifierGate::default();
        let d = gate.should_write(0.3, 0.9, 0.9);
        assert_eq!(d, WriteDecision::Reject);
        assert!(d.is_reject());
    }

    #[test]
    fn reject_when_curiosity_too_low() {
        let gate = VerifierGate::default();
        let d = gate.should_write(0.9, 0.1, 0.9);
        assert_eq!(d, WriteDecision::Reject);
    }

    #[test]
    fn quarantine_when_off_centroid() {
        let gate = VerifierGate::default();
        let d = gate.should_write(0.9, 0.9, 0.3);
        assert_eq!(d, WriteDecision::Quarantine);
        assert!(d.is_quarantine());
        assert!(d.is_blocked()); // blocked but not rejected
    }

    #[test]
    fn boundary_reward_strict_inequality() {
        let gate = VerifierGate::default();
        // reward == tau_write → Reject (strict inequality).
        assert_eq!(gate.should_write(0.5, 0.9, 0.9), WriteDecision::Reject);
        // reward just above → passes reward gate.
        assert_eq!(gate.should_write(0.501, 0.9, 0.9), WriteDecision::Write);
    }

    #[test]
    fn boundary_curiosity_strict_inequality() {
        let gate = VerifierGate::default();
        assert_eq!(gate.should_write(0.9, 0.3, 0.9), WriteDecision::Reject);
        assert_eq!(gate.should_write(0.9, 0.301, 0.9), WriteDecision::Write);
    }

    #[test]
    fn boundary_centroid_strict_inequality() {
        let gate = VerifierGate::default();
        // centroid == threshold → Write (NOT quarantined; the quarantine check
        // uses strict less-than).
        assert_eq!(gate.should_write(0.9, 0.9, 0.5), WriteDecision::Write);
        // centroid just below → Quarantine.
        assert_eq!(gate.should_write(0.9, 0.9, 0.499), WriteDecision::Quarantine);
    }

    #[test]
    fn custom_thresholds() {
        let gate = VerifierGate::new(0.7, 0.4, 0.6);
        // Default thresholds would Write; custom rejects.
        assert_eq!(gate.should_write(0.65, 0.5, 0.9), WriteDecision::Reject);
        assert_eq!(gate.should_write(0.8, 0.35, 0.9), WriteDecision::Reject);
        assert_eq!(gate.should_write(0.8, 0.5, 0.55), WriteDecision::Quarantine);
        assert_eq!(gate.should_write(0.8, 0.5, 0.65), WriteDecision::Write);
    }

    #[test]
    fn composed_rejects_when_clr_rejects() {
        let gate = VerifierGate::default();
        assert_eq!(gate.should_write_composed(false, 0.99), WriteDecision::Reject);
    }

    #[test]
    fn composed_quarantines_when_clr_accepts_but_off_centroid() {
        let gate = VerifierGate::default();
        assert_eq!(
            gate.should_write_composed(true, 0.3),
            WriteDecision::Quarantine
        );
    }

    #[test]
    fn composed_writes_when_clr_accepts_and_on_centroid() {
        let gate = VerifierGate::default();
        assert_eq!(gate.should_write_composed(true, 0.9), WriteDecision::Write);
    }

    #[test]
    fn write_decision_predicates() {
        assert!(WriteDecision::Write.is_write());
        assert!(!WriteDecision::Write.is_quarantine());
        assert!(!WriteDecision::Write.is_reject());
        assert!(!WriteDecision::Write.is_blocked());

        assert!(!WriteDecision::Quarantine.is_write());
        assert!(WriteDecision::Quarantine.is_quarantine());
        assert!(WriteDecision::Quarantine.is_blocked());

        assert!(WriteDecision::Reject.is_reject());
        assert!(WriteDecision::Reject.is_blocked());
    }
}
