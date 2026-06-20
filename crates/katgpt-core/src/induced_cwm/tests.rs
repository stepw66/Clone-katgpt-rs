//! Unit tests for the induced CWM kernel primitive (Plan 296 Phase 1 T1.8).
//!
//! These tests exercise the G1 (verifiability) and G4 (commitment integrity)
//! gates of the GOAT proof on a mock domain — a tiny "counter game" where
//! the state is a `u32` counter advanced by a per-kernel `step_size`.
//!
//! Phase 2 will add the G2 (play strength) and G3 (latency) gates on a mock
//! Leduc-poker-style domain under `induced_cwm_ismcts`.

use crate::induced_cwm::{
    BeliefInferenceFn, CwmCommitment, InducedCwmKernel, TransitionTestFailure, TransitionUnitTest,
    make_transition_tests_from_trajectory, verify_transition,
};
use crate::traits::GameState;

// ── Mock domain: counter game (Scenario A — kernel IS state type) ─────────
//
// The codebase's `GameState` trait conflates rules and state: the state's
// `impl GameState` IS the rules (see how `mcts_search(state: &S, ...)` works
// — there's no separate kernel object). For `InducedCwmKernel`, this means
// the kernel type IS the state type, and `canonical_bytes` describes the
// rule schema + parameters carried in the state value.
//
// For our mock:
// - `step_size` is a rule parameter (carried in every state value).
// - `counter` is the per-position state.
// - Two states with different `step_size` are considered DIFFERENT kernels
//   (different canonical_bytes), even though they share a type.
// - `advance` uses `self.step_size`, so a state with step_size=3 advances
//   the counter by 3 on `Inc`.
//
// This design lets us test "wrong kernel" naturally: a test generated from
// step_size=3 (expected_post has counter+3) will fail verification against a
// pre-state with step_size=4 (which would produce counter+4).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MockCounterState {
    pub counter: u32,
    pub step_size: u32,
    pub tick: u32,
}

impl MockCounterState {
    pub(crate) fn new(counter: u32, step_size: u32) -> Self {
        Self { counter, step_size, tick: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MockCounterAction {
    Inc,
    Dec,
    NoOp,
}

impl GameState for MockCounterState {
    type Action = MockCounterAction;

    fn available_actions(&self, _player_id: u8) -> Vec<Self::Action> {
        // Legal set is state-independent: {Inc, Dec, NoOp}.
        vec![MockCounterAction::Inc, MockCounterAction::Dec, MockCounterAction::NoOp]
    }

    fn advance(&self, action: &Self::Action, _player_id: u8) -> Self {
        let delta: i64 = match action {
            MockCounterAction::Inc => self.step_size as i64,
            MockCounterAction::Dec => -(self.step_size as i64),
            MockCounterAction::NoOp => 0,
        };
        Self {
            counter: (self.counter as i64).wrapping_add(delta) as u32,
            step_size: self.step_size,
            tick: self.tick + 1,
        }
    }

    fn is_terminal(&self) -> bool {
        false
    }

    fn reward(&self, _player_id: u8) -> f32 {
        0.0
    }

    fn tick(&self) -> u32 {
        self.tick
    }
}

impl InducedCwmKernel for MockCounterState {
    fn canonical_bytes(&self) -> Vec<u8> {
        // Schema tag + step_size. NOT counter/tick — those are per-position,
        // not part of the rule definition. Two states with same step_size
        // produce identical canonical_bytes (G4 determinism).
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(b"mock_cwm_v1");
        bytes.extend_from_slice(&self.step_size.to_le_bytes());
        bytes
    }
}

// ── Mock belief fn ────────────────────────────────────────────────────────

/// Mock belief fn that deterministically enumerates a small hidden-state
/// support. `n` samples = `min(n, support_size)` samples; identical seed →
/// identical output.
pub(crate) struct MockEnumBelief {
    pub support_size: usize,
}

impl BeliefInferenceFn<MockCounterState> for MockEnumBelief {
    type Sample = u32;

    fn sample(
        &self,
        _obs_history: &[MockCounterAction],
        _action_history: &[MockCounterAction],
        _player_id: u8,
        n: usize,
        seed: u64,
    ) -> Vec<Self::Sample> {
        let count = n.min(self.support_size);
        // Deterministic given seed: derive each sample from seed + index.
        (0..count).map(|i| seed.wrapping_add(i as u64) as u32).collect()
    }
}

// ── G4: canonical_bytes / BLAKE3 determinism ──────────────────────────────

#[test]
fn canonical_bytes_determinism() {
    // Same step_size → same BLAKE3, regardless of counter/tick value.
    let k_at_counter_10 = MockCounterState::new(10, 7);
    let k_at_counter_999 = MockCounterState::new(999, 7);
    assert_eq!(
        k_at_counter_10.commitment(),
        k_at_counter_999.commitment(),
        "BLAKE3 must depend only on step_size, not counter/tick"
    );

    // Re-construction is also stable.
    let expected = k_at_counter_10.commitment();
    for _ in 0..10 {
        let k2 = MockCounterState::new(0, 7);
        assert_eq!(k2.commitment(), expected, "BLAKE3 must be deterministic");
    }
}

#[test]
fn different_step_size_produces_different_commitment() {
    let k1 = MockCounterState::new(0, 1);
    let k2 = MockCounterState::new(0, 2);
    assert_ne!(
        k1.commitment(),
        k2.commitment(),
        "different step_size → different canonical bytes → different BLAKE3"
    );
}

#[test]
fn commitment_is_blake3_of_canonical_bytes() {
    // The default `commitment()` impl must equal `blake3::hash(canonical_bytes())`.
    let k = MockCounterState::new(0, 42);
    let expected = *blake3::hash(&k.canonical_bytes()).as_bytes();
    assert_eq!(k.commitment(), expected);
}

// ── CwmCommitment roundtrip ───────────────────────────────────────────────

#[test]
fn cwm_commitment_from_kernel_matches_manual() {
    let k = MockCounterState::new(0, 5);
    let c = CwmCommitment::from_kernel(&k, 1, 100);
    assert_eq!(c.blake3, k.commitment());
    assert_eq!(c.version, 1);
    assert_eq!(c.created_at_tick, 100);
}

#[test]
fn cwm_commitment_matches_kernel() {
    let k = MockCounterState::new(0, 5);
    let c = CwmCommitment::from_kernel(&k, 1, 100);
    assert!(c.matches_kernel(&k), "commitment must match its source kernel");
    // Different step_size → different kernel → no match.
    let k_other = MockCounterState::new(0, 6);
    assert!(!c.matches_kernel(&k_other));
}

#[test]
fn cwm_commitment_version_does_not_affect_blake3() {
    // Two commitments with identical kernel but different versions share blake3.
    let k = MockCounterState::new(0, 5);
    let c1 = CwmCommitment::from_kernel(&k, 1, 100);
    let c2 = CwmCommitment::from_kernel(&k, 999, 200);
    assert_eq!(c1.blake3, c2.blake3);
    assert_ne!(c1.version, c2.version);
    assert_ne!(c1.created_at_tick, c2.created_at_tick);
}

#[test]
fn cwm_commitment_serde_roundtrip() {
    let k = MockCounterState::new(0, 5);
    let c = CwmCommitment::from_kernel(&k, 7, 1234);
    let json = serde_json::to_string(&c).expect("serialize");
    let back: CwmCommitment = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, c);
}

#[test]
fn cwm_commitment_hash_and_eq() {
    // Required for use as a HashMap key in audit layers (riir-ai Plan 326).
    let k = MockCounterState::new(0, 5);
    let c1 = CwmCommitment::from_kernel(&k, 1, 100);
    let c2 = CwmCommitment::from_kernel(&k, 1, 100);
    assert_eq!(c1, c2);
    let mut set = std::collections::HashSet::new();
    set.insert(c1);
    assert!(set.contains(&c2));
}

// ── G1: transition tests ──────────────────────────────────────────────────

#[test]
fn verify_transition_passes_for_correct_kernel() {
    // A test generated by the correct kernel (step_size=3): Inc adds 3.
    let pre = MockCounterState::new(10, 3);
    let expected_post = pre.advance(&MockCounterAction::Inc, 0);
    assert_eq!(expected_post.counter, 13);

    let test = TransitionUnitTest {
        pre,
        action: MockCounterAction::Inc,
        player_id: 0,
        expected_post,
        expected_legal_actions: None,
    };
    assert!(verify_transition(&test).is_ok());
}

#[test]
fn verify_transition_detects_wrong_step_size() {
    // Test generated by step_size=3 (expected_post.counter = 10 + 3 = 13).
    // But pre-state has step_size=4, so pre.advance(Inc) gives 10 + 4 = 14.
    // The mismatch must be caught.
    let pre_correct = MockCounterState::new(10, 3);
    let expected_post = pre_correct.advance(&MockCounterAction::Inc, 0); // counter=13, step_size=3

    let pre_wrong = MockCounterState::new(10, 4); // different step_size
    let test = TransitionUnitTest {
        pre: pre_wrong,
        action: MockCounterAction::Inc,
        player_id: 0,
        expected_post,
        expected_legal_actions: None,
    };
    match verify_transition(&test) {
        Err(TransitionTestFailure::StateMismatch {
            actual_post_debug,
            expected_post_debug,
            ..
        }) => {
            // actual_post should have counter=14, step_size=4.
            // expected_post should have counter=13, step_size=3.
            assert!(actual_post_debug.contains("counter: 14"), "actual: {}", actual_post_debug);
            assert!(expected_post_debug.contains("counter: 13"), "expected: {}", expected_post_debug);
        }
        other => panic!("expected StateMismatch, got {:?}", other),
    }
}

#[test]
fn verify_transition_legal_actions_mismatch_path() {
    // The kernel reports {Inc, Dec, NoOp}; the test expects only {Inc, Dec}.
    let pre = MockCounterState::new(0, 1);
    let test = TransitionUnitTest {
        pre,
        action: MockCounterAction::NoOp,
        player_id: 0,
        expected_post: pre, // will match (NoOp is a no-op)
        expected_legal_actions: Some(vec![MockCounterAction::Inc, MockCounterAction::Dec]),
    };
    match verify_transition(&test) {
        Err(TransitionTestFailure::LegalActionsMismatch { actual, expected }) => {
            assert_eq!(actual.len(), 3, "kernel reports 3 actions");
            assert_eq!(expected.len(), 2, "test expected 2");
        }
        other => panic!("expected LegalActionsMismatch, got {:?}", other),
    }
}

#[test]
fn verify_transition_passes_with_matching_legal_actions() {
    let pre = MockCounterState::new(0, 1);
    let test = TransitionUnitTest {
        pre,
        action: MockCounterAction::NoOp,
        player_id: 0,
        expected_post: pre.advance(&MockCounterAction::NoOp, 0), // NoOp → same counter
        expected_legal_actions: Some(vec![
            MockCounterAction::Inc,
            MockCounterAction::Dec,
            MockCounterAction::NoOp,
        ]),
    };
    assert!(verify_transition(&test).is_ok());
}

#[test]
fn verify_transition_state_mismatch_path() {
    // Force a state mismatch: pre and expected_post differ.
    let pre = MockCounterState::new(10, 1);
    let wrong_post = MockCounterState::new(999, 1); // counter=999, won't match
    let test = TransitionUnitTest {
        pre,
        action: MockCounterAction::Inc,
        player_id: 0,
        expected_post: wrong_post,
        expected_legal_actions: None,
    };
    match verify_transition(&test) {
        Err(TransitionTestFailure::StateMismatch { .. }) => {}
        other => panic!("expected StateMismatch, got {:?}", other),
    }
}

// ── Trajectory helper ─────────────────────────────────────────────────────

#[test]
fn make_transition_tests_from_trajectory_emits_one_per_step() {
    let s0 = MockCounterState::new(0, 1);
    let steps: [(MockCounterState, MockCounterAction, u8, MockCounterState); 3] = [
        (s0, MockCounterAction::Inc, 0, s0.advance(&MockCounterAction::Inc, 0)),
        (s0, MockCounterAction::Dec, 0, s0.advance(&MockCounterAction::Dec, 0)),
        (s0, MockCounterAction::NoOp, 0, s0.advance(&MockCounterAction::NoOp, 0)),
    ];
    let tests = make_transition_tests_from_trajectory(steps);
    assert_eq!(tests.len(), 3);
    assert!(tests.iter().all(|t| t.expected_legal_actions.is_none()));
}

// ── Belief sampler ────────────────────────────────────────────────────────

#[test]
fn belief_sampler_returns_requested_count() {
    let b = MockEnumBelief { support_size: 5 };
    let samples = b.sample(&[], &[], 0, 3, 42);
    assert_eq!(samples.len(), 3, "n=3 with support_size=5 → 3 samples");
}

#[test]
fn belief_sampler_caps_at_support_size() {
    let b = MockEnumBelief { support_size: 2 };
    let samples = b.sample(&[], &[], 0, 10, 42);
    assert_eq!(samples.len(), 2, "n=10 with support_size=2 → 2 samples");
}

#[test]
fn belief_sampler_is_deterministic_given_seed() {
    let b = MockEnumBelief { support_size: 5 };
    let s1 = b.sample(&[], &[], 0, 3, 42);
    let s2 = b.sample(&[], &[], 0, 3, 42);
    assert_eq!(s1, s2);
}
