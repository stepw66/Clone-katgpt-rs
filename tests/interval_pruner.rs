//! Integration tests for IntervalPruner module (Plan 252 Phase 1, T6-T7).

#[cfg(feature = "interval_pruner")]
mod tests {
    use katgpt_core::ConstraintPruner;
    use katgpt_rs::interval_pruner::{IntervalMask, IntervalPruner};

    // ── Test 1: contiguous valid range is interval-closed ──

    #[test]
    fn test_contiguous_valid_is_interval_closed() {
        let mask = IntervalMask::from_vec(vec![false, true, true, true, false]);
        assert!(mask.is_interval_closed());
    }

    // ── Test 2: Swiss cheese pattern is NOT interval-closed ──

    #[test]
    fn test_swiss_cheese_not_interval_closed() {
        let mask = IntervalMask::from_vec(vec![true, false, true, false, true]);
        assert!(!mask.is_interval_closed());
        assert_eq!(mask.gap_count(), 2);
    }

    // ── Test 3: close_intervals merges small gaps ──

    #[test]
    fn test_close_intervals_merges_small_gaps() {
        let mask = IntervalMask::from_vec(vec![true, false, false, true]);
        let closed = mask.close_intervals(2);
        assert_eq!(closed.as_slice(), &[true, true, true, true]);
    }

    // ── Test 4: close_intervals does NOT merge large gaps ──

    #[test]
    fn test_close_intervals_preserves_large_gaps() {
        let mask = IntervalMask::from_vec(vec![true, false, false, false, true]);
        let closed = mask.close_intervals(1);
        assert_eq!(closed.as_slice(), &[true, false, false, false, true]);
    }

    // ── Test 5: valid_intervals returns correct ranges ──

    #[test]
    fn test_valid_intervals_correct_ranges() {
        let mask = IntervalMask::from_vec(vec![false, true, true, false, true, false]);
        let intervals = mask.valid_intervals();
        assert_eq!(intervals, vec![(1, 3), (4, 5)]);
    }

    // ── Test 6: IntervalPruner wrapping NoPruner: all tokens valid ──

    #[test]
    fn test_nopruner_all_tokens_valid() {
        let pruner = IntervalPruner::new(katgpt_core::NoPruner, 1);
        let candidates: Vec<usize> = (0..5).collect();
        let mut results = vec![false; 5];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        assert!(
            results.iter().all(|&v| v),
            "NoPruner should accept all tokens"
        );
    }

    // ── Test 7: DDTree scenario: scattered rejects get filled ──
    //
    // Inner pruner rejects tokens 3, 7 (but accepts 0,1,2,4,5,6,8,9).
    // After interval closure with gap_threshold=2: gaps of size 1 (pos 3)
    // and size 1 (pos 7) are both ≤ 2 → filled.
    // Result: tokens 0-9 all accepted.

    /// Simulated DDTree pruner that rejects specific tokens.
    struct SelectivePruner {
        rejected: Vec<usize>,
    }

    impl ConstraintPruner for SelectivePruner {
        fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
            !self.rejected.contains(&token_idx)
        }
    }

    #[test]
    fn test_ddtree_scattered_rejects_filled() {
        let inner = SelectivePruner {
            rejected: vec![3, 7],
        };
        let pruner = IntervalPruner::new(inner, 2);

        let candidates: Vec<usize> = (0..10).collect();
        let mut results = vec![false; 10];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        // All 10 tokens should be accepted after interval closure.
        assert!(
            results.iter().all(|&v| v),
            "scattered rejects should be filled: {:?}",
            results
        );
    }

    // ── Test 8: gap_threshold=0 means no filling ──

    #[test]
    fn test_zero_threshold_no_filling() {
        let inner = SelectivePruner {
            rejected: vec![3, 7],
        };
        let pruner = IntervalPruner::new(inner, 0);

        let candidates: Vec<usize> = (0..10).collect();
        let mut results = vec![false; 10];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        // Token 3 and 7 should remain rejected.
        assert!(!results[3], "token 3 should stay rejected with threshold 0");
        assert!(!results[7], "token 7 should stay rejected with threshold 0");
        assert!(
            results[0] && results[9],
            "non-rejected tokens should be valid"
        );
    }

    // ── Test 9: large gap not filled ──

    #[test]
    fn test_large_gap_not_filled() {
        // Reject tokens 0..5, accept 5..10 → single contiguous range, no gap.
        // Now test: accept 0, reject 1..4, accept 5..9 → gap size 4.
        let inner = SelectivePruner {
            rejected: vec![1, 2, 3, 4],
        };
        let pruner = IntervalPruner::new(inner, 2); // threshold 2 < gap 4

        let candidates: Vec<usize> = (0..10).collect();
        let mut results = vec![false; 10];
        pruner.batch_is_valid(0, &candidates, &[], &mut results);

        // Tokens 1..4 should remain rejected (gap=4 > threshold=2).
        assert!(results[0], "token 0 accepted");
        assert!(!results[2], "token 2 should stay rejected (gap too large)");
        assert!(results[5], "token 5 accepted");
    }
}
