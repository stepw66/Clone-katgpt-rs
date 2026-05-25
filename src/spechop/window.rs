//! Speculative window manager for SpecHop pipeline.
//!
//! Manages up to k concurrent speculative threads in FIFO order.
//! The pipeline pushes new speculations, then verifies the oldest
//! pending thread when the target tool returns. Verification either:
//! - **Commits** the branch (observation matched) → continue
//! - **Rolls back** the branch (observation differed) → discard downstream work

use crate::spechop::types::{HopObservation, HopState, SpecOutcome};
use crate::spechop::verifier::ObservationVerifier;

/// Window of speculative threads, bounded to capacity `k`.
///
/// Threads are ordered by insertion time. The pipeline verifies the
/// earliest (oldest) pending thread first — FIFO commit/rollback.
///
/// When a thread is committed, subsequent speculation can continue building
/// on top of it. When rolled back, the caller typically invokes
/// `rollback_all()` to discard all downstream speculative work.
pub struct SpecWindow {
    /// All threads in insertion order. Committed threads stay in the window
    /// until explicitly drained; pending threads are at the tail.
    threads: Vec<HopObservation>,
    /// Maximum number of concurrent speculative threads (k).
    max_threads: usize,
    /// Index of the first pending (unverified) thread.
    /// All threads before this index are committed.
    pending_start: usize,
}

impl SpecWindow {
    /// Create a new window with the given maximum thread capacity.
    ///
    /// `max_threads` corresponds to k from the cost model
    /// (`SpecHopConfig::effective_k()`).
    pub fn new(max_threads: usize) -> Self {
        assert!(
            max_threads >= 1,
            "max_threads must be >= 1, got {max_threads}"
        );
        Self {
            threads: Vec::with_capacity(max_threads),
            max_threads,
            pending_start: 0,
        }
    }

    /// Maximum number of speculative threads (k).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.max_threads
    }

    /// Current number of threads in the window (both committed and pending).
    #[inline]
    pub fn len(&self) -> usize {
        self.threads.len()
    }

    /// Whether the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.threads.is_empty()
    }

    /// Number of pending (unverified) threads.
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.threads.len().saturating_sub(self.pending_start)
    }

    /// Number of committed threads.
    #[inline]
    pub fn committed_count(&self) -> usize {
        self.pending_start
    }

    /// Remaining capacity for new speculative threads.
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.max_threads.saturating_sub(self.threads.len())
    }

    /// Whether the window is at capacity (cannot push more threads).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.threads.len() >= self.max_threads
    }

    /// Push a new speculative thread into the window.
    ///
    /// # Panics
    ///
    /// Panics if the window is at capacity (`len() >= max_threads`).
    pub fn push_thread(&mut self, observation: HopObservation) {
        assert!(
            !self.is_full(),
            "SpecWindow at capacity ({}/{}): cannot push more threads",
            self.threads.len(),
            self.max_threads
        );
        self.threads.push(observation);
    }

    /// Verify the earliest pending thread against the target observation.
    ///
    /// Uses the provided verifier to compare `o_target` against the
    /// speculative observation. On match → `Commit`, on mismatch → `Rollback`.
    ///
    /// Returns `None` if there are no pending threads to verify.
    ///
    /// # State transitions
    ///
    /// - `Commit`: thread transitions to `HopState::Committed`, `pending_start` advances
    /// - `Rollback`: thread transitions to `HopState::RolledBack` (caller should `rollback_all()`)
    pub fn verify_earliest(
        &mut self,
        verifier: &dyn ObservationVerifier,
        o_target: &str,
    ) -> Option<SpecOutcome> {
        let pending = &mut self.threads[self.pending_start..];

        // Find the first Speculating thread
        let idx = pending
            .iter()
            .position(|t| t.state == HopState::Speculating)?;
        let thread = &mut pending[idx];

        let o_spec = thread.o_spec.as_deref().unwrap_or("");
        let matched = verifier.verify(o_target, o_spec);

        let outcome = if matched {
            thread.commit(o_target);
            // Advance past this committed thread
            self.pending_start += idx + 1;
            // Also skip any already-committed threads that might follow
            self.advance_past_committed();
            SpecOutcome::Commit
        } else {
            thread.rollback();
            SpecOutcome::Rollback
        };

        Some(outcome)
    }

    /// Advance `pending_start` past any already-committed threads.
    fn advance_past_committed(&mut self) {
        while self.pending_start < self.threads.len()
            && self.threads[self.pending_start].state == HopState::Committed
        {
            self.pending_start += 1;
        }
    }

    /// Rollback all pending speculative work, keeping only committed threads.
    ///
    /// Truncates the thread list to `pending_start`, discarding all
    /// unverified or rolled-back threads. This is the recovery path
    /// when verification fails — the pipeline resets to the last
    /// known-good state.
    pub fn rollback_all(&mut self) {
        self.threads.truncate(self.pending_start);
    }

    /// Drain all committed threads, returning them and resetting the window.
    ///
    /// Useful for collecting results after a sequence of successful commits.
    /// After draining, the window retains only pending threads (if any).
    pub fn drain_committed(&mut self) -> Vec<HopObservation> {
        if self.pending_start == 0 {
            return Vec::new();
        }

        let committed: Vec<HopObservation> = self.threads.drain(0..self.pending_start).collect();
        self.pending_start = 0;
        committed
    }

    /// Get a reference to the earliest pending thread, if any.
    pub fn earliest_pending(&self) -> Option<&HopObservation> {
        self.threads
            .get(self.pending_start)
            .filter(|t| t.state == HopState::Speculating)
    }

    /// Get a reference to the most recently pushed thread, if any.
    pub fn latest(&self) -> Option<&HopObservation> {
        self.threads.last()
    }

    /// Reset the window entirely, clearing all threads and state.
    pub fn reset(&mut self) {
        self.threads.clear();
        self.pending_start = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spechop::verifier::RuleBasedVerifier;

    fn make_verifier() -> RuleBasedVerifier {
        RuleBasedVerifier::default()
    }

    fn make_window(k: usize) -> SpecWindow {
        SpecWindow::new(k)
    }

    // ── T15: SpecWindow struct ─────────────────────────────────

    #[test]
    fn test_window_new() {
        let w = make_window(4);
        assert_eq!(w.capacity(), 4);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
        assert_eq!(w.pending_count(), 0);
        assert_eq!(w.remaining_capacity(), 4);
        assert!(!w.is_full());
    }

    #[test]
    #[should_panic(expected = "max_threads must be >= 1")]
    fn test_window_new_zero_capacity_panics() {
        let _ = SpecWindow::new(0);
    }

    // ── T16: push_thread ───────────────────────────────────────

    #[test]
    fn test_push_thread_within_capacity() {
        let mut w = make_window(3);
        w.push_thread(HopObservation::speculating("a1", "s1"));
        w.push_thread(HopObservation::speculating("a2", "s2"));
        assert_eq!(w.len(), 2);
        assert_eq!(w.remaining_capacity(), 1);
    }

    #[test]
    fn test_push_thread_fills_capacity() {
        let mut w = make_window(2);
        w.push_thread(HopObservation::speculating("a1", "s1"));
        w.push_thread(HopObservation::speculating("a2", "s2"));
        assert!(w.is_full());
        assert_eq!(w.remaining_capacity(), 0);
    }

    #[test]
    #[should_panic(expected = "SpecWindow at capacity")]
    fn test_push_thread_over_capacity_panics() {
        let mut w = make_window(2);
        w.push_thread(HopObservation::speculating("a1", "s1"));
        w.push_thread(HopObservation::speculating("a2", "s2"));
        w.push_thread(HopObservation::speculating("a3", "s3")); // should panic
    }

    // ── T17: verify_earliest ───────────────────────────────────

    #[test]
    fn test_verify_earliest_commit() {
        let mut w = make_window(4);
        w.push_thread(HopObservation::speculating("a1", "result 42"));
        w.push_thread(HopObservation::speculating("a2", "result 99"));

        let v = make_verifier();
        let outcome = w.verify_earliest(&v, "result 42");
        assert_eq!(outcome, Some(SpecOutcome::Commit));

        // First thread is committed, second is still pending
        assert_eq!(w.committed_count(), 1); // pending_start advanced past thread 0
        assert_eq!(w.pending_count(), 1);
    }

    #[test]
    fn test_verify_earliest_rollback() {
        let mut w = make_window(4);
        w.push_thread(HopObservation::speculating("a1", "speculation"));
        w.push_thread(HopObservation::speculating("a2", "speculation"));

        let v = make_verifier();
        let outcome = w.verify_earliest(&v, "completely different answer");
        assert_eq!(outcome, Some(SpecOutcome::Rollback));

        // Thread 0 is rolled back, thread 1 is still pending
        assert_eq!(w.threads[0].state, HopState::RolledBack);
    }

    #[test]
    fn test_verify_earliest_empty_returns_none() {
        let mut w = make_window(4);
        let outcome = w.verify_earliest(&make_verifier(), "target");
        assert_eq!(outcome, None);
    }

    // ── T19: Window capacity enforcement ───────────────────────

    #[test]
    fn test_capacity_enforcement() {
        let mut w = make_window(3);
        w.push_thread(HopObservation::speculating("a1", "s1"));
        w.push_thread(HopObservation::speculating("a2", "s2"));
        w.push_thread(HopObservation::speculating("a3", "s3"));
        assert!(w.is_full());
        assert_eq!(w.remaining_capacity(), 0);
    }

    // ── T19: Commit shifts window ──────────────────────────────

    #[test]
    fn test_commit_shifts_window() {
        let mut w = make_window(4);
        w.push_thread(HopObservation::speculating("a1", "answer 42"));
        w.push_thread(HopObservation::speculating("a2", "answer 99"));
        w.push_thread(HopObservation::speculating("a3", "answer 7"));

        // Commit first → window should track 1 committed, 2 pending
        let v = make_verifier();
        let outcome = w.verify_earliest(&v, "answer 42");
        assert_eq!(outcome, Some(SpecOutcome::Commit));
        assert_eq!(w.committed_count(), 1);
        assert_eq!(w.pending_count(), 2);

        // After commit, can push more (capacity freed after drain)
        // But threads are still in the window, so capacity is still limited
        assert_eq!(w.remaining_capacity(), 1);

        // Commit second
        let outcome = w.verify_earliest(&v, "answer 99");
        assert_eq!(outcome, Some(SpecOutcome::Commit));
        assert_eq!(w.committed_count(), 2);
        assert_eq!(w.pending_count(), 1);
    }

    // ── T19: Rollback clears downstream ────────────────────────

    #[test]
    fn test_rollback_clears_downstream() {
        let mut w = make_window(4);
        w.push_thread(HopObservation::speculating("a1", "spec"));
        w.push_thread(HopObservation::speculating("a2", "spec2"));
        w.push_thread(HopObservation::speculating("a3", "spec3"));

        // Rollback first
        let v = make_verifier();
        let outcome = w.verify_earliest(&v, "totally different");
        assert_eq!(outcome, Some(SpecOutcome::Rollback));

        // rollback_all clears everything
        w.rollback_all();
        assert!(w.is_empty());
        assert_eq!(w.pending_count(), 0);
    }

    // ── T19: Sequential commits advance state ──────────────────

    #[test]
    fn test_sequential_commits_advance_state() {
        let mut w = make_window(4);
        let v = make_verifier();

        // Push 4 threads
        w.push_thread(HopObservation::speculating("a1", "r1"));
        w.push_thread(HopObservation::speculating("a2", "r2"));
        w.push_thread(HopObservation::speculating("a3", "r3"));
        w.push_thread(HopObservation::speculating("a4", "r4"));

        // Commit all 4 sequentially
        assert_eq!(w.verify_earliest(&v, "r1"), Some(SpecOutcome::Commit));
        assert_eq!(w.verify_earliest(&v, "r2"), Some(SpecOutcome::Commit));
        assert_eq!(w.verify_earliest(&v, "r3"), Some(SpecOutcome::Commit));
        assert_eq!(w.verify_earliest(&v, "r4"), Some(SpecOutcome::Commit));

        // All committed, no pending
        assert_eq!(w.committed_count(), 4);
        assert_eq!(w.pending_count(), 0);

        // Drain committed
        let committed = w.drain_committed();
        assert_eq!(committed.len(), 4);
        assert!(w.is_empty());
    }

    #[test]
    fn test_commit_then_rollback_then_resume() {
        let mut w = make_window(4);
        let v = make_verifier();

        w.push_thread(HopObservation::speculating("a1", "match"));
        w.push_thread(HopObservation::speculating("a2", "wrong"));
        w.push_thread(HopObservation::speculating("a3", "spec3"));

        // Commit first
        assert_eq!(w.verify_earliest(&v, "match"), Some(SpecOutcome::Commit));
        assert_eq!(w.committed_count(), 1);

        // Rollback second
        assert_eq!(
            w.verify_earliest(&v, "totally different"),
            Some(SpecOutcome::Rollback)
        );

        // rollback_all keeps committed, discards rest
        w.rollback_all();
        assert_eq!(w.len(), 1); // only the committed one remains
        assert_eq!(w.pending_count(), 0);

        // Can push new threads
        w.push_thread(HopObservation::speculating("a4", "new spec"));
        assert_eq!(w.pending_count(), 1);
    }

    // ── Drain and reset ────────────────────────────────────────

    #[test]
    fn test_drain_committed_empty() {
        let mut w = make_window(4);
        assert!(w.drain_committed().is_empty());
    }

    #[test]
    fn test_reset_clears_everything() {
        let mut w = make_window(4);
        w.push_thread(HopObservation::speculating("a1", "s1"));
        w.push_thread(HopObservation::speculating("a2", "s2"));
        w.verify_earliest(&make_verifier(), "s1");

        w.reset();
        assert!(w.is_empty());
        assert_eq!(w.pending_count(), 0);
        assert_eq!(w.committed_count(), 0);
        assert_eq!(w.remaining_capacity(), 4);
    }

    // ── Earliest pending and latest ────────────────────────────

    #[test]
    fn test_earliest_pending() {
        let mut w = make_window(4);
        assert!(w.earliest_pending().is_none());

        w.push_thread(HopObservation::speculating("a1", "s1"));
        assert_eq!(w.earliest_pending().unwrap().action, "a1");
    }

    #[test]
    fn test_latest() {
        let mut w = make_window(4);
        assert!(w.latest().is_none());

        w.push_thread(HopObservation::speculating("a1", "s1"));
        w.push_thread(HopObservation::speculating("a2", "s2"));
        assert_eq!(w.latest().unwrap().action, "a2");
    }
}
