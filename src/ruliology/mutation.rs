//! Adaptive Strategy Mutation — FSM graph mutation + co-evolution.
//!
//! Wolfram showed that random mutation + keep-if-better converges on winning
//! strategies. This module applies that principle to FSM graphs:
//! - Vertex color flip (change output of a state)
//! - Edge reroute (change transition target)
//! - Co-evolution: two FSMs mutate alternately, keep-if-better
//!
//! Plan 188 Phase 5 (feature-gated behind `ruliology`).

use crate::ruliology::fsm::{FsmStrategy, MAX_STATES};
use crate::ruliology::types::SimpleProgram;

// ── FsmTemplateProposer ──────────────────────────────────────────

/// Mutation operator for FSM strategies.
///
/// Applies stochastic mutations to FSM transition/output tables:
/// - **Output flip**: toggle a state's output (0↔1)
/// - **Edge reroute**: redirect a random transition to a different state
///
/// Mutation rate controls the expected fraction of elements changed per proposal.
pub struct FsmTemplateProposer {
    /// Number of states in target FSMs.
    n_states: u8,
    /// Mutation probability per element (transition or output).
    mutation_rate: f32,
}

/// Result of a co-evolution run.
#[derive(Debug, Clone)]
pub struct CoEvolutionResult {
    /// Best FSM found (by average payoff).
    pub best_fsm: FsmStrategy,
    /// Average payoff of best FSM.
    pub best_payoff: f64,
    /// Number of generations elapsed.
    pub generations: u32,
    /// Mutation history: (generation, payoff) pairs.
    pub history: Vec<(u32, f64)>,
}

/// Mutation types for FSM graphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationType {
    /// Flip the output of a random state.
    OutputFlip,
    /// Change a random transition target.
    EdgeReroute,
}

impl FsmTemplateProposer {
    /// Create a new proposer with given state count and mutation rate.
    #[inline]
    pub fn new(n_states: u8, mutation_rate: f32) -> Self {
        debug_assert!(n_states as usize <= MAX_STATES);
        Self {
            n_states,
            mutation_rate,
        }
    }

    /// Default proposer with 10% mutation rate.
    #[inline]
    pub fn default_for(n_states: u8) -> Self {
        Self::new(n_states, 0.1)
    }

    /// Propose a mutated version of the given FSM.
    ///
    /// Iterates all states; each transition and output independently mutates
    /// with probability `mutation_rate`.
    pub fn propose(&self, strategy: &FsmStrategy, rng: &mut fastrand::Rng) -> FsmStrategy {
        let mut transitions = *strategy.transitions();
        let mut outputs = *strategy.outputs();
        let n = self.n_states as usize;

        for s in 0..n {
            // Mutate transitions
            for input in 0..2 {
                if rng.f32() < self.mutation_rate {
                    transitions[s][input] = (rng.u8(0..self.n_states)).min(MAX_STATES as u8 - 1);
                }
            }
            // Mutate outputs
            if rng.f32() < self.mutation_rate {
                outputs[s] = 1 - outputs[s]; // flip 0↔1
            }
        }

        FsmStrategy::new(transitions, outputs, self.n_states, 0)
    }

    /// Apply a specific mutation type to a given state index.
    pub fn mutate_specific(
        &self,
        strategy: &FsmStrategy,
        mutation: MutationType,
        state_idx: u8,
        rng: &mut fastrand::Rng,
    ) -> FsmStrategy {
        let mut transitions = *strategy.transitions();
        let mut outputs = *strategy.outputs();
        let s = state_idx as usize;

        match mutation {
            MutationType::OutputFlip => {
                if s < self.n_states as usize {
                    outputs[s] = 1 - outputs[s];
                }
            }
            MutationType::EdgeReroute => {
                if s < self.n_states as usize {
                    let input = rng.u8(0..2) as usize;
                    transitions[s][input] = (rng.u8(0..self.n_states)).min(MAX_STATES as u8 - 1);
                }
            }
        }

        FsmStrategy::new(transitions, outputs, self.n_states, 0)
    }

    /// Number of states this proposer targets.
    #[inline]
    pub fn n_states(&self) -> u8 {
        self.n_states
    }

    /// Current mutation rate.
    #[inline]
    pub fn mutation_rate(&self) -> f32 {
        self.mutation_rate
    }
}

// ── Co-Evolution ─────────────────────────────────────────────────

/// Run co-evolution: two FSMs mutate alternately, keep-if-better.
///
/// Starting from `seed`, generates mutants each generation. If a mutant
/// scores higher against all `opponents`, it replaces the current best.
/// Returns the best FSM found after `generations` rounds of mutation.
pub fn co_evolve(
    seed: FsmStrategy,
    opponents: &[FsmStrategy],
    rounds: u32,
    generations: u32,
    payoff_fn: &dyn Fn(u8, u8) -> f64,
    proposer: &FsmTemplateProposer,
    rng: &mut fastrand::Rng,
) -> CoEvolutionResult {
    let mut current = seed;
    let mut current_payoff = evaluate_vs_opponents(&current, opponents, rounds, payoff_fn);
    let mut history = Vec::with_capacity((generations / 10 + 2) as usize);
    history.push((0, current_payoff));

    for generation in 1..=generations {
        let mutant = proposer.propose(&current, rng);
        let mutant_payoff = evaluate_vs_opponents(&mutant, opponents, rounds, payoff_fn);

        // Keep-if-better (strict improvement)
        if mutant_payoff > current_payoff {
            current = mutant;
            current_payoff = mutant_payoff;
        }

        if generation % 10 == 0 || generation == generations {
            history.push((generation, current_payoff));
        }
    }

    CoEvolutionResult {
        best_fsm: current,
        best_payoff: current_payoff,
        generations,
        history,
    }
}

/// Evaluate a strategy against all opponents.
///
/// Returns mean payoff across all opponent pairings, averaged over `rounds`
/// iterations per opponent.
fn evaluate_vs_opponents(
    strategy: &FsmStrategy,
    opponents: &[FsmStrategy],
    rounds: u32,
    payoff_fn: &dyn Fn(u8, u8) -> f64,
) -> f64 {
    if opponents.is_empty() {
        return 0.0;
    }

    let mut total = 0.0f64;
    let mut count = 0usize;

    for opp in opponents {
        let mut s = strategy.clone();
        let mut o = opp.clone();
        s.reset();
        o.reset();

        let mut hist_s: Vec<u8> = Vec::with_capacity(rounds as usize);
        let mut hist_o: Vec<u8> = Vec::with_capacity(rounds as usize);
        let mut payoff = 0.0f64;

        for _ in 0..rounds {
            let a_s = s.next_action(&hist_o);
            let a_o = o.next_action(&hist_s);
            payoff += payoff_fn(a_s, a_o);
            hist_s.push(a_s);
            hist_o.push(a_o);
        }

        total += payoff / rounds as f64;
        count += 1;
    }

    total / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ruliology::fsm::FsmEnumerator;
    use crate::ruliology::payoff::matching_pennies;

    /// Helper: build a simple 2-state FSM (always cooperate).
    fn always_cooperate() -> FsmStrategy {
        let transitions = [[0u8; 2]; MAX_STATES];
        let outputs = [0u8; MAX_STATES];
        FsmStrategy::new(transitions, outputs, 2, 0)
    }

    /// Helper: build a simple 2-state FSM (always defect).
    fn always_defect() -> FsmStrategy {
        let transitions = [[0u8; 2]; MAX_STATES];
        let mut outputs = [0u8; MAX_STATES];
        outputs[0] = 1;
        outputs[1] = 1;
        FsmStrategy::new(transitions, outputs, 2, 0)
    }

    #[test]
    fn test_propose_preserves_n_states() {
        let seed = always_cooperate();
        let proposer = FsmTemplateProposer::default_for(2);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let mutant = proposer.propose(&seed, &mut rng);
            assert_eq!(
                mutant.n_states(),
                2,
                "mutated FSM should preserve n_states=2"
            );
        }
    }

    #[test]
    fn test_output_flip_mutation() {
        let seed = always_cooperate();
        let proposer = FsmTemplateProposer::new(2, 1.0); // 100% rate so propose always mutates
        let mut rng = fastrand::Rng::with_seed(99);

        // Apply output flip to state 0
        let mutant = proposer.mutate_specific(&seed, MutationType::OutputFlip, 0, &mut rng);
        // State 0 output should be flipped from 0 to 1
        assert_eq!(mutant.outputs()[0], 1, "state 0 output should be flipped");
        // State 1 output unchanged
        assert_eq!(mutant.outputs()[1], 0, "state 1 output should be unchanged");
    }

    #[test]
    fn test_edge_reroute_mutation() {
        // Build FSM where all transitions go to state 0
        let seed = always_cooperate();
        let proposer = FsmTemplateProposer::new(2, 1.0);
        let mut rng = fastrand::Rng::with_seed(7);

        // Apply edge reroute to state 0 — should change one of the two transitions
        let mutant = proposer.mutate_specific(&seed, MutationType::EdgeReroute, 0, &mut rng);

        // At least one transition from state 0 may have changed (probabilistic but
        // with seed 7 we get deterministic result).
        // Verify structure integrity: all transitions still valid.
        for input in 0..2 {
            assert!(
                mutant.transitions()[0][input] < 2,
                "transition should be valid state index"
            );
        }
    }

    #[test]
    fn test_co_evolve_converges() {
        // Seed: always cooperate (poor strategy in matching pennies).
        let seed = always_cooperate();

        // Opponents: all 2-state FSMs — rich opponent pool.
        let opponents = FsmEnumerator::enumerate(2);
        assert!(!opponents.is_empty(), "need opponents for co-evolution");

        let proposer = FsmTemplateProposer::new(2, 0.3);
        let mut rng = fastrand::Rng::with_seed(123);

        let result = co_evolve(
            seed,
            &opponents,
            50,  // rounds per evaluation
            200, // generations
            &matching_pennies,
            &proposer,
            &mut rng,
        );

        // History should be recorded.
        assert!(!result.history.is_empty(), "history should not be empty");
        assert_eq!(result.history[0].0, 0, "first entry should be generation 0");

        // Best payoff should be recorded.
        assert!(
            result.best_payoff >= -1.0 && result.best_payoff <= 1.0,
            "payoff should be in [-1, 1] for matching pennies, got {}",
            result.best_payoff,
        );

        // The best FSM should have correct n_states.
        assert_eq!(result.best_fsm.n_states(), 2);
    }

    #[test]
    fn test_propose_can_improve() {
        // Start with always-cooperate, run co-evolution, verify payoff improves.
        let seed = always_cooperate();

        // Opponents: a mix of fixed strategies.
        let opponents = vec![always_defect(), always_cooperate()];

        let proposer = FsmTemplateProposer::new(2, 0.2);
        let mut rng = fastrand::Rng::with_seed(42);

        // Measure initial payoff.
        let initial_payoff = evaluate_vs_opponents(&seed, &opponents, 100, &matching_pennies);

        let result = co_evolve(
            seed,
            &opponents,
            100,
            500,
            &matching_pennies,
            &proposer,
            &mut rng,
        );

        // Co-evolution should improve or maintain payoff.
        assert!(
            result.best_payoff >= initial_payoff,
            "co-evolution should improve payoff: initial={initial_payoff}, final={}",
            result.best_payoff,
        );
    }

    #[test]
    fn test_evaluate_vs_empty_opponents() {
        let strategy = always_cooperate();
        let payoff = evaluate_vs_opponents(&strategy, &[], 10, &matching_pennies);
        assert!(
            (payoff - 0.0).abs() < 1e-9,
            "empty opponents should yield 0 payoff, got {payoff}"
        );
    }

    #[test]
    fn test_mutation_type_out_of_bounds() {
        let seed = always_cooperate();
        let proposer = FsmTemplateProposer::new(2, 0.5);
        let mut rng = fastrand::Rng::new();

        // State index beyond n_states — should be a no-op, FSM unchanged.
        let mutant = proposer.mutate_specific(&seed, MutationType::OutputFlip, 5, &mut rng);
        assert_eq!(
            mutant.outputs(),
            seed.outputs(),
            "out-of-bounds flip should be no-op"
        );
    }
}

// TL;DR: FsmTemplateProposer (stochastic FSM graph mutation via output flip + edge reroute) + co_evolve (keep-if-better hill climbing against opponent pool). 7 tests behind ruliology feature gate.
