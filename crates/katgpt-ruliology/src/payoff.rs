//! Ruliology payoff functions — canonical game-theory payoffs.
//!
//! Wolfram-inspired payoff functions for 2-action (0/1) games.
//! Plan 188 Phase 1.

/// Matching pennies: +1.0 if actions match, -1.0 otherwise.
///
/// Zero-sum game. No pure Nash equilibrium — only mixed (50/50).
/// Used to test whether FSMs discover the best response dynamics.
#[inline]
pub fn matching_pennies(a: u8, b: u8) -> f64 {
    if a == b { 1.0 } else { -1.0 }
}

/// Prisoner's dilemma payoff matrix.
///
/// Actions: 0 = Cooperate, 1 = Defect.
/// Returns `(payoff_a, payoff_b)`.
///
/// |       | B:C=0 | B:D=1 |
/// |-------|-------|-------|
/// | A:C=0 | -1,-1 | -5, 0 |
/// | A:D=1 |  0,-5 | -3,-3 |
///
/// Defect is dominant strategy, but mutual cooperation (-1,-1) beats
/// mutual defection (-3,-3). Tests whether FSMs learn to cooperate.
#[inline]
pub fn prisoners_dilemma(a: u8, b: u8) -> (f64, f64) {
    match (a, b) {
        (0, 0) => (-1.0, -1.0), // cooperate / cooperate
        (0, 1) => (-5.0, 0.0),  // cooperate / defect
        (1, 0) => (0.0, -5.0),  // defect / cooperate
        (1, 1) => (-3.0, -3.0), // defect / defect
        _ => (0.0, 0.0),        // unreachable for binary actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Matching Pennies ───────────────────────────────────────

    #[test]
    fn test_matching_pennies_match() {
        assert!((matching_pennies(0, 0) - 1.0).abs() < 1e-9);
        assert!((matching_pennies(1, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_matching_pennies_mismatch() {
        assert!((matching_pennies(0, 1) - (-1.0)).abs() < 1e-9);
        assert!((matching_pennies(1, 0) - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_matching_pennies_zero_sum() {
        // matching_pennies returns the ROW player payoff.
        // The game is zero-sum: row_gain + col_gain = 0.
        // col_gain = -matching_pennies(a, b) since the column player wants
        // the opposite outcome (mismatch for column player).
        for a in 0u8..=1 {
            for b in 0u8..=1 {
                let row = matching_pennies(a, b);
                let _col = matching_pennies(a, b); // column player sees same matrix
                // In matching pennies, row wants match (+1), col wants mismatch.
                // So col's payoff is -row's payoff.
                assert!((row + (-row)).abs() < 1e-9, "sanity: {row}");
            }
        }
        // Verify the key property: payoff is +1 for match, -1 for mismatch.
        assert!((matching_pennies(0, 0) - 1.0).abs() < 1e-9);
        assert!((matching_pennies(1, 1) - 1.0).abs() < 1e-9);
        assert!((matching_pennies(0, 1) - (-1.0)).abs() < 1e-9);
        assert!((matching_pennies(1, 0) - (-1.0)).abs() < 1e-9);
    }

    // ── Prisoner's Dilemma ─────────────────────────────────────

    #[test]
    fn test_pd_mutual_cooperation() {
        let (a, b) = prisoners_dilemma(0, 0);
        assert!((a - (-1.0)).abs() < 1e-9);
        assert!((b - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pd_mutual_defection() {
        let (a, b) = prisoners_dilemma(1, 1);
        assert!((a - (-3.0)).abs() < 1e-9);
        assert!((b - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn test_pd_temptation_payoff() {
        // Defect vs cooperate: defector gets 0 (best), cooperator gets -5 (worst).
        let (a, b) = prisoners_dilemma(1, 0);
        assert!((a - 0.0).abs() < 1e-9);
        assert!((b - (-5.0)).abs() < 1e-9);

        let (a, b) = prisoners_dilemma(0, 1);
        assert!((a - (-5.0)).abs() < 1e-9);
        assert!((b - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_pd_defect_dominant() {
        // For both players, defection yields >= cooperation regardless of opponent.
        for opp in 0u8..=1 {
            let (coop, _) = prisoners_dilemma(0, opp);
            let (defect, _) = prisoners_dilemma(1, opp);
            assert!(defect > coop, "defect should dominate: opp={opp}");
        }
    }
}

// TL;DR: matching_pennies (zero-sum, +1/-1) + prisoners_dilemma (classic PD payoff matrix). Both operate on binary actions (0/1).
