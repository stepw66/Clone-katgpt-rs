//! `TransitionUnitTest<S>` — auto-generated forward-model regression tests.
//!
//! Paper §4.1: the CWM is verified by auto-generated transition unit tests
//! produced from observed trajectories. Each test is a triple
//! `(pre-state, action, expected-post-state)` plus an optional
//! `expected-legal-actions` check. A kernel "passes" if every observed
//! transition is reproduced exactly.
//!
//! This is the G1 verifiability gate of the GOAT proof: a correctly-induced
//! kernel MUST pass 100% of the tests derived from the trajectory it was
//! induced from, and MUST fail with a precise diff when any transition is
//! mutated.
//!
//! # Latent boundary
//!
//! Test inputs (`pre`, `expected_post`) are raw game-state snapshots — they
//! cross the sync boundary as-is. `expected_legal_actions` is also raw (the
//! legal action set is deterministic for a given state). Nothing latent here.

use crate::induced_cwm::InducedCwmKernel;

/// One observed `(pre, action, expected_post)` triple, plus optional
/// legal-action check.
///
/// Construct by hand or via
/// [`make_transition_tests_from_trajectory`](make_transition_tests_from_trajectory),
/// which walks an observed trajectory and emits one test per step.
#[derive(Clone, Debug)]
pub struct TransitionUnitTest<S: crate::traits::GameState> {
    /// State before the action was applied.
    pub pre: S,
    /// Action applied at `pre`.
    pub action: S::Action,
    /// Player who took the action.
    pub player_id: u8,
    /// Expected state after applying `action` at `pre`.
    pub expected_post: S,
    /// If `Some`, also check that the legal-action set at `pre` for
    /// `player_id` equals this set (as a set — order-independent). If `None`,
    /// skip the legal-action check.
    pub expected_legal_actions: Option<Vec<S::Action>>,
}

/// Failure diagnostic from [`verify_transition`].
///
/// Carries the precise diff between expected and actual — not just a bool.
/// Paper §4.1 stresses that mutation testing must localise the failure.
#[derive(Clone, Debug)]
pub enum TransitionTestFailure<A> {
    /// `kernel.advance(pre, &action, player_id)` did not equal `expected_post`.
    StateMismatch {
        /// What `verify_transition` computed by calling `kernel.advance`.
        actual_post_debug: String,
        /// What the test expected.
        expected_post_debug: String,
        /// The action that triggered the mismatch (for log context).
        action_debug: String,
    },
    /// Legal-action set at `pre` for `player_id` did not equal expected.
    ///
    /// Compared as sets (order-independent). Length-equal but order-different
    /// sets PASS — only set-membership differences fail.
    LegalActionsMismatch {
        /// What `verify_transition` computed via `kernel.available_actions`.
        actual: Vec<A>,
        /// What the test expected.
        expected: Vec<A>,
    },
}

/// Verify a single transition test against an induced kernel.
///
/// Runs `test.pre.advance(&test.action, test.player_id)` and (if
/// `expected_legal_actions` is `Some`) `test.pre.available_actions(test.player_id)`.
/// Returns `Ok(())` iff both checks match.
///
/// # Design note (deviation from Plan 296 T1.6)
///
/// The plan called for `verify_transition(kernel: &K, test: ...)`. We drop the
/// `kernel` parameter because it is redundant: `test.pre: K` already IS the
/// kernel-instance being advanced (per the codebase's `GameState` convention —
/// the state's `impl GameState` defines the rules; there is no separate
/// "rules object"). The `InducedCwmKernel` bound on `S` enforces that the
/// state type is an LLM-induced kernel.
///
/// This matches how `mcts_search(state: &S, ...)` works upstream — there is no
/// separate kernel parameter there either.
///
/// # Type bounds
///
/// `S` must implement `PartialEq + Debug` so we can compare pre/expected-post
/// states and format the diff on mismatch. Most game-state structs already
/// implement both (they're small snapshots).
pub fn verify_transition<S>(
    test: &TransitionUnitTest<S>,
) -> Result<(), TransitionTestFailure<S::Action>>
where
    S: InducedCwmKernel + PartialEq + std::fmt::Debug,
    S::Action: PartialEq + std::fmt::Debug,
{
    // Legal-action check (run first — cheaper than state clone, and a
    // legal-action mismatch is more informative than a downstream state mismatch).
    if let Some(expected) = &test.expected_legal_actions {
        let actual = test.pre.available_actions(test.player_id);
        if !set_eq(&actual, expected) {
            return Err(TransitionTestFailure::LegalActionsMismatch {
                actual,
                expected: expected.clone(),
            });
        }
    }

    // State transition check.
    let actual_post = test.pre.advance(&test.action, test.player_id);
    if actual_post != test.expected_post {
        return Err(TransitionTestFailure::StateMismatch {
            actual_post_debug: format!("{:?}", actual_post),
            expected_post_debug: format!("{:?}", test.expected_post),
            action_debug: format!("{:?}", test.action),
        });
    }
    Ok(())
}

/// Walk an observed `(state, action, next_state)` trajectory and emit one
/// `TransitionUnitTest` per step.
///
/// `trajectory` is an iterator yielding `(pre, action, player_id, post)`.
/// `expected_legal_actions` is left `None` — the trajectory walker does not
/// observe legal-action sets, only transitions. Integrators that want the
/// legal-action check should set the field manually after construction.
///
/// Allocates one `TransitionUnitTest` per step. Cold-tier only — induction
/// events are rare.
pub fn make_transition_tests_from_trajectory<S, I>(trajectory: I) -> Vec<TransitionUnitTest<S>>
where
    S: crate::traits::GameState,
    I: IntoIterator<Item = (S, S::Action, u8, S)>,
{
    trajectory
        .into_iter()
        .map(
            |(pre, action, player_id, expected_post)| TransitionUnitTest {
                pre,
                action,
                player_id,
                expected_post,
                expected_legal_actions: None,
            },
        )
        .collect()
}

/// Order-independent equality check on two slices.
///
/// `actual.len() == expected.len()` AND every element of `actual` appears in
/// `expected` (counted, so duplicates are respected). O(n²) — fine for the
/// small action spaces this is called on (typically ≤ 32 actions).
fn set_eq<T: PartialEq>(actual: &[T], expected: &[T]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    actual.iter().all(|a| {
        actual.iter().filter(|x| x == &a).count() == expected.iter().filter(|x| x == &a).count()
    })
}
