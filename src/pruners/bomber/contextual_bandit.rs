//! Contextual bandit for HLPlayer (Issue 371 Option 1 — T6).
//!
//! Linear per-arm Q model: `Q(a, s) = θ_a^T · φ(s)` where `θ_a ∈ R^d` for each
//! of the 7 bomber arms and `φ(s)` is a board-state context vector.
//!
//! ## Why this is modelless
//!
//! The update rule is **online LMS** (least-mean-squares) gradient descent on a
//! fixed-size linear model — equivalent in spirit to an adaptive filter (LMS /
//! NLMS / RLS family), which is a deterministic signal-processing primitive,
//! not a trained neural network. There is:
//!
//! - **No offline training** — weights start at zero and are updated online
//!   from observed per-tick rewards during the game loop.
//! - **No backprop** — the gradient is `error · φ`, a direct outer product.
//! - **No gradient descent on network weights** — `θ_a` is a flat `R^d` vector
//!   (d=7), not the parameter tensor of a learned model.
//! - **No softmax** — scoring uses sigmoid (per global rule) for the bounded
//!   `[0,1]` Q used in the centered blend; exploration is ε-greedy.
//!
//! This satisfies the katgpt-rs modelless mandate: the only "weight mutation"
//! is online bandit learning on a linear model, which is the same class of
//! primitive as LMS adaptive filtering.
//!
//! ## Context vector (d = 7)
//!
//! 1. `in_blast_zone` (0.0 or 1.0) — current danger.
//! 2. `blast_proximity` (sigmoid-normalized Manhattan distance to nearest bomb).
//! 3. `opponent_pressure` (sigmoid-normalized inverse distance to nearest opponent).
//! 4. `wall_density_3x3` (wall count / 9) — movement constraint.
//! 5. `powerup_proximity` (sigmoid-normalized inverse distance to nearest powerup).
//! 6. `bomb_pressure` (sigmoid-normalized active bomb count).
//! 7. `bias` (1.0) — unconditional offset per arm.
//!
//! ## Update rule
//!
//! For observed reward `r` after taking action `a` in context `φ`:
//! ```text
//! error = Q_raw(a, s) - r
//! θ_a  -= α · error · φ
//! ```
//! where `α` is the learning rate (default 0.01).
//!
//! ## Scoring integration
//!
//! The n-armed centered blend is `(Q - 0.5) * 2.0`, which assumes Q ∈ [0, 1].
//! Raw linear Q can exceed this range, so the scoring path sigmoids the raw Q:
//! `Q_score = sigmoid(Q_raw)` ∈ (0, 1). This keeps the bandit contribution in
//! `(-1, 1)` and follows the global "sigmoid not softmax" rule.

use super::blend_context::{CONTEXT_DIM, sigmoid};
use super::players::ACTION_COUNT;
#[cfg(test)]
use super::blend_context::compute_phi;
#[cfg(test)]
use super::players::KnownBomb;

/// Default learning rate for online LMS updates.
pub const DEFAULT_LEARNING_RATE: f32 = 0.01;

/// Bias term index in the context vector (last element).
const BIAS_IDX: usize = CONTEXT_DIM - 1;

/// Linear contextual bandit — per-arm `θ_a ∈ R^d` modelless Q learner.
///
/// See module docs for the modelless justification and the update rule.
#[derive(Clone, Debug)]
pub struct ContextualBandit {
    /// Per-arm weight vectors, stored flat: `theta[arm * CONTEXT_DIM + feat]`.
    /// All zeros at cold start → `Q_raw = 0` → `sigmoid(0) = 0.5` (neutral).
    theta: Vec<f32>,
    /// Per-arm visit counts (for cold-start gating and `compress_cycle`).
    visits: [u32; ACTION_COUNT],
    /// Total updates across all arms.
    total_pulls: u32,
    /// Learning rate α.
    learning_rate: f32,
}

impl Default for ContextualBandit {
    fn default() -> Self {
        Self::new(DEFAULT_LEARNING_RATE)
    }
}

impl ContextualBandit {
    /// Create a contextual bandit with all-θ = 0 (cold start).
    pub fn new(learning_rate: f32) -> Self {
        Self {
            theta: vec![0.0; ACTION_COUNT * CONTEXT_DIM],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            learning_rate,
        }
    }

    /// Raw linear Q: `θ_a^T · φ(s)`. Used internally for training.
    #[inline]
    pub fn predict_raw(&self, arm: usize, phi: &[f32; CONTEXT_DIM]) -> f32 {
        debug_assert!(arm < ACTION_COUNT, "arm {arm} out of range");
        let base = arm * CONTEXT_DIM;
        let mut sum = 0.0f32;
        let mut i = 0;
        while i < CONTEXT_DIM {
            sum += self.theta[base + i] * phi[i];
            i += 1;
        }
        sum
    }

    /// Sigmoid-bounded Q ∈ (0, 1) for scoring integration with the centered
    /// blend `(Q - 0.5) * 2.0`.
    ///
    /// At cold start (θ = 0): `Q_raw = 0` → `sigmoid(0) = 0.5` → bandit_term = 0
    /// (neutral, same as an unvisited n-armed arm).
    #[inline]
    pub fn predict(&self, arm: usize, phi: &[f32; CONTEXT_DIM]) -> f32 {
        sigmoid(self.predict_raw(arm, phi))
    }

    /// Online LMS update: `θ_a -= α · (Q_raw - r) · φ`.
    ///
    /// This is the only weight mutation path. It is deterministic, online, and
    /// modelless (see module docs).
    #[inline]
    pub fn update(&mut self, arm: usize, phi: &[f32; CONTEXT_DIM], reward: f32) {
        debug_assert!(arm < ACTION_COUNT, "arm {arm} out of range");
        let q_raw = self.predict_raw(arm, phi);
        let error = q_raw - reward;
        let alpha = self.learning_rate;
        let base = arm * CONTEXT_DIM;
        let mut i = 0;
        while i < CONTEXT_DIM {
            self.theta[base + i] -= alpha * error * phi[i];
            i += 1;
        }
        self.visits[arm] = self.visits[arm].saturating_add(1);
        self.total_pulls = self.total_pulls.saturating_add(1);
    }

    /// Per-arm visit count (cold-start gate, mirrors `HLPlayer::arm_visits`).
    #[inline]
    pub fn visits(&self, arm: usize) -> u32 {
        self.visits[arm]
    }

    /// Total pulls across all arms.
    #[inline]
    pub fn total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Representative per-arm Q (the bias weight) for `compress_cycle` /
    /// `compress_report` diagnostics. The bias term captures the unconditional
    /// expected reward for that arm across all observed contexts.
    ///
    /// Returns the raw bias weight (not sigmoided) so the existing
    /// `Q < 0.1` compress threshold operates on the same scale as the n-armed
    /// bandit's average-Q.
    #[inline]
    pub fn arm_q(&self, arm: usize) -> f32 {
        self.theta[arm * CONTEXT_DIM + BIAS_IDX]
    }

    /// Whether this arm has been pulled at least once (cold-start gate).
    #[inline]
    pub fn is_cold(&self, arm: usize) -> bool {
        self.visits[arm] == 0
    }

    /// Diagnostic: format the per-arm bias weights as a compact string.
    /// Mirrors the n-armed `compress_report` format.
    pub fn q_report(&self) -> String {
        use super::players::ALL_ACTIONS;
        (0..ACTION_COUNT)
            .map(|i| format!("{}:{:.2}", ALL_ACTIONS[i], self.arm_q(i)))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

// ── Unit tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn phi_safe() -> [f32; CONTEXT_DIM] {
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    }

    fn phi_danger() -> [f32; CONTEXT_DIM] {
        [1.0, 0.9, 0.8, 0.7, 0.1, 0.9, 1.0]
    }

    #[test]
    fn predict_returns_neutral_at_cold_start() {
        // All θ = 0 → Q_raw = 0 → sigmoid(0) = 0.5 (neutral for centered blend).
        let cb = ContextualBandit::default();
        let phi = phi_safe();
        for arm in 0..ACTION_COUNT {
            let q = cb.predict(arm, &phi);
            assert!(
                (q - 0.5).abs() < 1e-6,
                "arm {arm}: cold-start Q should be 0.5 (neutral), got {q}"
            );
        }
    }

    #[test]
    fn positive_reward_increases_q() {
        let mut cb = ContextualBandit::new(0.1);
        let phi = phi_safe();
        let arm = 0; // Up
        let q_before = cb.predict(arm, &phi);
        // Positive reward (e.g. survived): error = 0 - 1.0 = -1.0 → θ += α·φ
        cb.update(arm, &phi, 1.0);
        let q_after = cb.predict(arm, &phi);
        assert!(
            q_after > q_before,
            "positive reward should increase Q: {q_before} -> {q_after}"
        );
    }

    #[test]
    fn negative_reward_decreases_q() {
        let mut cb = ContextualBandit::new(0.1);
        let phi = phi_safe();
        let arm = 1; // Down
        let q_before = cb.predict(arm, &phi);
        // Negative reward (e.g. died): error = 0 - (-1.0) = 1.0 → θ -= α·φ
        cb.update(arm, &phi, -1.0);
        let q_after = cb.predict(arm, &phi);
        assert!(
            q_after < q_before,
            "negative reward should decrease Q: {q_before} -> {q_after}"
        );
    }

    #[test]
    fn different_contexts_produce_different_q_after_updates() {
        // THE CORE TEST: the contextual fix. After training arm 0 (Up) with
        // positive reward in a safe context, the SAME arm should have a
        // DIFFERENT Q in a dangerous context (because θ · φ_safe ≠ θ · φ_danger).
        let mut cb = ContextualBandit::new(0.1);
        let arm = 0; // Up
        let phi_safe = phi_safe();
        let phi_danger = phi_danger();

        // Before any update: both contexts give Q = 0.5 (θ = 0).
        let q_safe_cold = cb.predict(arm, &phi_safe);
        let q_danger_cold = cb.predict(arm, &phi_danger);
        assert!((q_safe_cold - q_danger_cold).abs() < 1e-6);

        // Train: "Up" always gets positive reward in the safe context.
        for _ in 0..50 {
            cb.update(arm, &phi_safe, 1.0);
        }
        // Also train: "Up" gets negative reward in the danger context.
        for _ in 0..50 {
            cb.update(arm, &phi_danger, -1.0);
        }

        let q_safe = cb.predict(arm, &phi_safe);
        let q_danger = cb.predict(arm, &phi_danger);

        // The same arm (Up) must now have DIFFERENT Q in safe vs danger contexts.
        // This is the whole point of the contextual bandit — the n-armed bandit
        // cannot do this (it has one Q per arm regardless of context).
        let diff = (q_safe - q_danger).abs();
        assert!(
            diff > 0.1,
            "same arm should differ across contexts after training: \
             q_safe={q_safe:.4}, q_danger={q_danger:.4}, diff={diff:.4}"
        );
    }

    #[test]
    fn updates_only_affect_target_arm() {
        let mut cb = ContextualBandit::new(0.1);
        let phi = phi_safe();
        cb.update(0, &phi, 1.0); // Train arm 0
        let q0 = cb.predict(0, &phi);
        let q1 = cb.predict(1, &phi); // Arm 1 untouched
        assert!(
            (q1 - 0.5).abs() < 1e-6,
            "untouched arm should stay at cold-start 0.5, got {q1}; trained arm got {q0}"
        );
    }

    #[test]
    fn visits_increment_on_update() {
        let mut cb = ContextualBandit::default();
        let phi = phi_safe();
        assert_eq!(cb.visits(0), 0);
        assert!(cb.is_cold(0));
        cb.update(0, &phi, 0.5);
        assert_eq!(cb.visits(0), 1);
        assert!(!cb.is_cold(0));
        cb.update(0, &phi, 0.5);
        assert_eq!(cb.visits(0), 2);
        assert_eq!(cb.total_pulls(), 2);
        // Other arms still cold.
        assert_eq!(cb.visits(1), 0);
    }

    #[test]
    fn arm_q_returns_bias_weight() {
        let mut cb = ContextualBandit::new(0.1);
        let phi = phi_safe();
        // After positive reward updates, the bias weight (last feature, =1.0)
        // should increase.
        let bias_before = cb.arm_q(0);
        assert!(bias_before.abs() < 1e-6);
        for _ in 0..10 {
            cb.update(0, &phi, 1.0);
        }
        let bias_after = cb.arm_q(0);
        assert!(
            bias_after > bias_before,
            "bias weight should increase after positive rewards: {bias_before} -> {bias_after}"
        );
    }

    #[test]
    fn sigmoid_basics() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.9999);
        assert!(sigmoid(-10.0) < 0.0001);
        assert!(sigmoid(1.0) > 0.5);
        assert!(sigmoid(-1.0) < 0.5);
    }

    #[test]
    fn q_report_is_non_empty() {
        let cb = ContextualBandit::default();
        let report = cb.q_report();
        assert!(report.contains("↑:0.00"));
        assert!(report.contains("💣:0.00"));
    }

    #[test]
    fn compute_phi_distinguishes_safe_from_dangerous() {
        // The core property the contextual bandit depends on: compute_phi
        // must produce DIFFERENT vectors for safe vs dangerous board states,
        // so the linear model can learn context-dependent Q.
        use super::super::{ARENA_H, ARENA_W, ArenaGrid, Cell, GridPos};

        let cells = vec![vec![Cell::Floor; ARENA_W]; ARENA_H];
        let grid = ArenaGrid {
            cells,
            width: ARENA_W,
            height: ARENA_H,
        };

        let pos = GridPos { x: 5, y: 5 };
        let powerups: [(i32, i32); 0] = [];

        // Safe state: no bombs, no opponent.
        let phi_safe = compute_phi(pos, &grid, &[], &powerups, None);

        // Dangerous state: bomb adjacent, opponent nearby.
        let bombs: [KnownBomb; 1] = [((5, 4), 2, 4)]; // bomb 1 cell above
        let phi_danger = compute_phi(pos, &grid, &bombs, &powerups, Some((6, 5)));

        // The in_blast_zone feature (index 0) must differ.
        assert!(
            phi_danger[0] > phi_safe[0],
            "in_blast_zone should be higher in danger state: safe={}, danger={}",
            phi_safe[0],
            phi_danger[0]
        );

        // blast_proximity (index 1) must be higher when bombs are near.
        assert!(
            phi_danger[1] > phi_safe[1],
            "blast_proximity should be higher with bomb nearby: safe={:.3}, danger={:.3}",
            phi_safe[1],
            phi_danger[1]
        );

        // opponent_pressure (index 2) must be higher with opponent nearby.
        assert!(
            phi_danger[2] > phi_safe[2],
            "opponent_pressure should be higher with opponent nearby: safe={:.3}, danger={:.3}",
            phi_safe[2],
            phi_danger[2]
        );

        // The vectors should not be identical.
        let diff: f32 = (0..CONTEXT_DIM)
            .map(|i| (phi_safe[i] - phi_danger[i]).powi(2))
            .sum();
        assert!(
            diff > 0.01,
            "safe and danger phi should differ significantly: squared diff = {diff:.6}"
        );
    }

    #[test]
    fn compute_phi_bias_is_always_one() {
        use super::super::{ARENA_H, ARENA_W, ArenaGrid, Cell, GridPos};
        let cells = vec![vec![Cell::Floor; ARENA_W]; ARENA_H];
        let grid = ArenaGrid {
            cells,
            width: ARENA_W,
            height: ARENA_H,
        };
        let pos = GridPos { x: 1, y: 1 };
        let phi = compute_phi(pos, &grid, &[], &[], None);
        assert!((phi[BIAS_IDX] - 1.0).abs() < 1e-6, "bias term must be 1.0");
    }
}
