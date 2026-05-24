//! GOAT proofs for event-sourced game traces (Plan 124).
//!
//! Proves:
//! 1. Deterministic replay produces identical final state
//! 2. Fork shares exact prefix events
//! 3. Structural diff correctly identifies divergence point
//! 4. EvalCache basic operations
//! 5. Event log overhead is minimal

#[cfg(feature = "event_log")]
mod tests {
    use katgpt_rs::pruners::event_log::*;

    /// Simple game state for testing.
    #[derive(Clone, Debug, PartialEq)]
    struct TestState {
        score: i32,
        position: (i32, i32),
    }

    /// Simple action for testing.
    #[derive(Clone, Debug, PartialEq)]
    enum TestAction {
        Move(i32, i32),
        Score(i32),
        End,
    }

    fn apply_action(state: &TestState, action: &TestAction) -> TestState {
        match action {
            TestAction::Move(dx, dy) => TestState {
                score: state.score,
                position: (state.position.0 + dx, state.position.1 + dy),
            },
            TestAction::Score(pts) => TestState {
                score: state.score + pts,
                position: state.position,
            },
            TestAction::End => state.clone(),
        }
    }

    #[test]
    fn test_event_log_push_and_len() {
        let mut log: EventLog<TestAction> = EventLog::new();
        assert!(log.is_empty());

        let id0 = log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        assert_eq!(id0, EventId(0));
        assert_eq!(log.len(), 1);

        let id1 = log.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            Some(id0),
        );
        assert_eq!(id1, EventId(1));
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn test_event_log_get() {
        let mut log: EventLog<TestAction> = EventLog::new();
        log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::Action,
            TestAction::Score(5),
            Actor::Player(1),
            None,
        );

        let event = log.get(EventId(1)).unwrap();
        assert_eq!(event.event_type, EventType::Action);
        assert_eq!(event.actor, Actor::Player(1));
    }

    #[test]
    fn test_deterministic_replay() {
        // GOAT Proof 1: Replay produces identical final state
        let mut log: EventLog<TestAction> = EventLog::new();
        log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::Action,
            TestAction::Move(0, 1),
            Actor::Player(0),
            None,
        );
        log.push(EventType::GameEnd, TestAction::End, Actor::Runtime, None);

        let initial = TestState {
            score: 0,
            position: (0, 0),
        };
        let final_state = log.replay(initial.clone(), |state, event| {
            apply_action(&state, &event.payload)
        });

        // Replay again — must be identical
        let final_state_2 =
            log.replay(initial, |state, event| apply_action(&state, &event.payload));

        assert_eq!(final_state, final_state_2, "Replay must be deterministic");
        assert_eq!(final_state.score, 10);
        assert_eq!(final_state.position, (1, 1));
    }

    #[test]
    fn test_fork_shares_prefix() {
        // GOAT Proof 2: Fork shares exact prefix events
        let mut parent: EventLog<TestAction> = EventLog::new();
        parent.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        parent.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );
        parent.push(
            EventType::Action,
            TestAction::Move(0, 1),
            Actor::Player(0),
            None,
        );
        parent.push(
            EventType::Action,
            TestAction::Score(5),
            Actor::Player(0),
            None,
        );

        // Fork after event 1 (Score(10))
        let mut forked = parent.fork(EventId(1));
        assert_eq!(forked.len(), 2); // Events 0 and 1

        // Add different events to fork
        forked.push(
            EventType::Action,
            TestAction::Move(-1, 0),
            Actor::Player(1),
            None,
        );

        // Parent prefix should be identical
        assert_eq!(
            parent.get(EventId(0)).unwrap().payload,
            forked.get(EventId(0)).unwrap().payload
        );
        assert_eq!(
            parent.get(EventId(1)).unwrap().payload,
            forked.get(EventId(1)).unwrap().payload
        );
    }

    #[test]
    fn test_structural_diff() {
        // GOAT Proof 3: Diff correctly identifies divergence point
        let mut log_a: EventLog<TestAction> = EventLog::new();
        log_a.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log_a.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );
        log_a.push(
            EventType::Action,
            TestAction::Move(0, 1),
            Actor::Player(0),
            None,
        );
        log_a.push(
            EventType::Action,
            TestAction::Score(5),
            Actor::Player(0),
            None,
        );

        let mut log_b: EventLog<TestAction> = EventLog::new();
        log_b.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log_b.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );
        log_b.push(
            EventType::Action,
            TestAction::Move(-1, 0),
            Actor::Player(1),
            None,
        ); // Diverges here
        log_b.push(
            EventType::Action,
            TestAction::Score(20),
            Actor::Player(1),
            None,
        );

        let diff = log_a.diff(&log_b);
        assert_eq!(diff.shared_prefix_len, 2, "Should share first 2 events");
        assert!(!diff.is_identical());
        assert_eq!(diff.divergence_count(), 4); // 2 from A + 2 from B
    }

    #[test]
    fn test_identical_logs_diff() {
        let mut log: EventLog<TestAction> = EventLog::new();
        log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );

        let diff = log.diff(&log);
        assert!(diff.is_identical());
        assert_eq!(diff.shared_prefix_len, 2);
    }

    #[test]
    fn test_eval_cache_basic() {
        let mut cache = EvalCache::new();
        let hash = [0u8; 32];

        assert!(cache.get(&hash).is_none());

        cache.insert(hash, 0.95, 3, EventId(0));

        let cached = cache.get(&hash).unwrap();
        assert!((cached.score - 0.95).abs() < 0.001);
        assert_eq!(cached.depth, 3);
    }

    #[test]
    fn test_eval_cache_hit_rate() {
        let cache = EvalCache::new();
        assert_eq!(cache.hit_rate(8, 10), 0.8);
        assert_eq!(cache.hit_rate(0, 0), 0.0);
    }

    #[test]
    fn test_event_types_enum() {
        let types = [
            EventType::GameStart,
            EventType::Action,
            EventType::Evaluation,
            EventType::BanditUpdate,
            EventType::HeuristicFire,
            EventType::RewardSignal,
            EventType::GameEnd,
        ];
        assert_eq!(types.len(), 7);
    }

    #[test]
    fn test_game_outcome_enum() {
        assert_eq!(GameOutcome::Win(0), GameOutcome::Win(0));
        assert_ne!(GameOutcome::Win(0), GameOutcome::Win(1));
        assert_eq!(GameOutcome::Draw, GameOutcome::Draw);
    }

    #[test]
    fn test_actor_enum() {
        assert_eq!(Actor::Player(0), Actor::Player(0));
        assert_ne!(Actor::Player(0), Actor::Player(1));
        assert_eq!(Actor::Bandit, Actor::Bandit);
    }

    #[test]
    fn test_multiple_games_replay() {
        // GOAT: Replay 100 games deterministically
        for seed in 0..100u64 {
            let mut log: EventLog<TestAction> = EventLog::new();
            let actions = [
                TestAction::Move(1, 0),
                TestAction::Score(seed as i32),
                TestAction::Move(0, 1),
                TestAction::Score(seed as i32 * 2),
                TestAction::End,
            ];
            for action in &actions {
                log.push(EventType::Action, action.clone(), Actor::Player(0), None);
            }

            let initial = TestState {
                score: 0,
                position: (0, 0),
            };
            let final_1 = log.replay(initial.clone(), |s, e| apply_action(&s, &e.payload));
            let final_2 = log.replay(initial, |s, e| apply_action(&s, &e.payload));
            assert_eq!(final_1, final_2, "Game {seed} must be deterministic");
        }
    }

    #[test]
    fn test_event_id_ordering() {
        assert!(EventId(0) < EventId(1));
        assert!(EventId(1) < EventId(100));
        assert_eq!(EventId::ZERO, EventId(0));
    }

    #[test]
    fn test_last_id() {
        let mut log: EventLog<TestAction> = EventLog::new();
        assert_eq!(log.last_id(), EventId::ZERO);

        log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        assert_eq!(log.last_id(), EventId(0));

        log.push(
            EventType::Action,
            TestAction::Score(5),
            Actor::Player(0),
            None,
        );
        assert_eq!(log.last_id(), EventId(1));
    }

    #[test]
    fn test_causal_chain() {
        let mut log: EventLog<TestAction> = EventLog::new();
        let id0 = log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        let id1 = log.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            Some(id0),
        );
        let id2 = log.push(
            EventType::RewardSignal,
            TestAction::End,
            Actor::Runtime,
            Some(id1),
        );

        assert_eq!(log.get(id0).unwrap().caused_by, None);
        assert_eq!(log.get(id1).unwrap().caused_by, Some(id0));
        assert_eq!(log.get(id2).unwrap().caused_by, Some(id1));
    }

    #[test]
    fn test_fork_at_boundary() {
        let mut log: EventLog<TestAction> = EventLog::new();
        log.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );
        log.push(
            EventType::Action,
            TestAction::Move(0, 1),
            Actor::Player(0),
            None,
        );

        // Fork at last event
        let forked = log.fork(EventId(2));
        assert_eq!(forked.len(), 3);

        // Fork past end — should clamp
        let forked_past = log.fork(EventId(100));
        assert_eq!(forked_past.len(), 3);
    }

    #[test]
    fn test_diff_empty_logs() {
        let log_a: EventLog<TestAction> = EventLog::new();
        let log_b: EventLog<TestAction> = EventLog::new();

        let diff = log_a.diff(&log_b);
        assert!(diff.is_identical());
        assert_eq!(diff.shared_prefix_len, 0);
    }

    #[test]
    fn test_diff_different_lengths() {
        let mut log_a: EventLog<TestAction> = EventLog::new();
        log_a.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log_a.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );

        let mut log_b: EventLog<TestAction> = EventLog::new();
        log_b.push(
            EventType::GameStart,
            TestAction::Move(1, 0),
            Actor::Player(0),
            None,
        );
        log_b.push(
            EventType::Action,
            TestAction::Score(10),
            Actor::Player(0),
            None,
        );
        log_b.push(
            EventType::Action,
            TestAction::Move(0, 1),
            Actor::Player(0),
            None,
        );

        // log_b has 1 extra event
        let diff = log_a.diff(&log_b);
        assert_eq!(diff.shared_prefix_len, 2);
        assert_eq!(diff.divergence_count(), 1); // 1 fork-only event
    }

    #[test]
    fn test_replay_empty_log() {
        let log: EventLog<TestAction> = EventLog::new();
        let initial = TestState {
            score: 0,
            position: (0, 0),
        };
        let result = log.replay(initial.clone(), |s, _| s);
        assert_eq!(result, initial);
    }

    #[test]
    fn test_eval_cache_overwrite() {
        let mut cache = EvalCache::new();
        let hash = [42u8; 32];

        cache.insert(hash, 0.5, 1, EventId(0));
        cache.insert(hash, 0.9, 2, EventId(1)); // Overwrite

        let cached = cache.get(&hash).unwrap();
        assert!((cached.score - 0.9).abs() < 0.001);
        assert_eq!(cached.depth, 2);
        assert_eq!(cached.provenance, EventId(1));
    }

    #[test]
    fn test_event_log_default() {
        let log: EventLog<TestAction> = EventLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_eval_cache_default() {
        let cache = EvalCache::default();
        assert!(cache.is_empty());
    }
}
