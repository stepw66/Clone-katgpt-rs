//! Plan 390 Phase 1 T1.5 — `mcts_state_action_cache` basic unit tests.
//!
//! Covers the cache semantics that require a real `InferenceActionSpace`
//! implementation (the pure-cache round-trip / clear / key-distinction tests
//! live inline in the module under `#[cfg(test)]`). This file adds:
//! - A deterministic synthetic action space (3 actions, 4-step transition graph).
//! - Search runs with cache reuse (first run populates; second run hits).
//! - The determinism-contract doc-test.

use katgpt_core::mcts_state_action_cache::{
    InferenceAction, InferenceActionSpace, SearchScratch, StateActionCache,
    mcts_search_with_state_action_cache,
};

// ── Synthetic deterministic action space ─────────────────────────────────
//
// A 4-step deterministic transition graph with 3 actions. Each step `t` has
// a state value `v ∈ {0..=3}`; actions advance `v` by different deterministic
// deltas:
//   action 0 (config=0, strat=0): v += 0  (no-op — keeps the state, still terminal-bounded)
//   action 1 (config=1, strat=0): v += 1  (advances one step)
//   action 2 (config=2, strat=0): v += 2  (advances two steps — reaches terminal faster)
//
// Reward = terminal `v / 3.0` (so action 2 reaches the max faster, but the
// reward is the same at terminal). This makes the graph deterministic AND
// gives a clear "best action" signal for the search to find.
//
// The DeterministicTransition contract holds by construction: `apply` is a
// pure function of `(state, action)` — no RNG, no mutable shared state.

#[derive(Clone, Debug, PartialEq, Eq)]
struct ToyState {
    v: u8,
}

const MAX_V: u8 = 3;

const ACTIONS: [InferenceAction; 3] = [
    InferenceAction::new(0, 0),
    InferenceAction::new(1, 0),
    InferenceAction::new(2, 0),
];

struct ToySpace;

impl InferenceActionSpace<ToyState> for ToySpace {
    fn actions_at(&self, state: &ToyState) -> &[InferenceAction] {
        if state.v >= MAX_V { &[] } else { &ACTIONS }
    }

    fn apply(&self, state: &ToyState, action: InferenceAction) -> ToyState {
        // Deterministic: delta = config_id (0, 1, or 2), clamped to MAX_V.
        let delta = (action.config_id as u8).min(MAX_V - state.v);
        ToyState { v: state.v + delta }
    }

    fn reward(&self, state: &ToyState) -> Option<f32> {
        if state.v >= MAX_V {
            Some(state.v as f32 / MAX_V as f32)
        } else {
            None
        }
    }

    fn is_terminal(&self, state: &ToyState) -> bool {
        state.v >= MAX_V
    }

    fn state_hash(&self, state: &ToyState) -> blake3::Hash {
        blake3::hash(&[state.v])
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn cache_round_trip_via_space() {
    // Insert a transition observed through the space, then retrieve it.
    let cache: StateActionCache<f32> = StateActionCache::new();
    let space = ToySpace;
    let s = ToyState { v: 0 };
    let a = InferenceAction::new(2, 0);
    let next = space.apply(&s, a);
    let next_hash = space.state_hash(&next);
    cache.insert(space.state_hash(&s), a, next_hash, 1.0);
    let got = cache
        .get(space.state_hash(&s), a)
        .expect("inserted entry must be retrievable");
    assert_eq!(got.0, next_hash);
    assert!((got.1 - 1.0).abs() < 1e-6);
}

#[test]
fn same_state_different_action_different_entries_via_space() {
    // The headline novelty vs state-only caching: one state, two actions,
    // two distinct (next_state, reward) entries.
    let cache: StateActionCache<f32> = StateActionCache::new();
    let space = ToySpace;
    let s = ToyState { v: 0 };
    let a1 = InferenceAction::new(1, 0);
    let a2 = InferenceAction::new(2, 0);
    let n1 = space.apply(&s, a1);
    let n2 = space.apply(&s, a2);
    cache.insert(space.state_hash(&s), a1, space.state_hash(&n1), 0.5);
    cache.insert(space.state_hash(&s), a2, space.state_hash(&n2), 0.9);
    assert_eq!(cache.len(), 2, "two actions at one state = two entries");
    let g1 = cache.get(space.state_hash(&s), a1).expect("a1 must hit");
    let g2 = cache.get(space.state_hash(&s), a2).expect("a2 must hit");
    assert_ne!(
        g1.0, g2.0,
        "different actions must lead to different next-state hashes"
    );
}

#[test]
fn different_state_same_action_different_entries_via_space() {
    let cache: StateActionCache<f32> = StateActionCache::new();
    let space = ToySpace;
    let s1 = ToyState { v: 0 };
    let s2 = ToyState { v: 1 };
    let a = InferenceAction::new(1, 0);
    cache.insert(
        space.state_hash(&s1),
        a,
        space.state_hash(&space.apply(&s1, a)),
        0.3,
    );
    cache.insert(
        space.state_hash(&s2),
        a,
        space.state_hash(&space.apply(&s2, a)),
        0.6,
    );
    assert_eq!(cache.len(), 2, "one action at two states = two entries");
}

#[test]
fn cache_clear_empties_then_repopulates() {
    let cache: StateActionCache<f32> = StateActionCache::new();
    let space = ToySpace;
    let s = ToyState { v: 0 };
    let a = InferenceAction::new(1, 0);
    cache.insert(
        space.state_hash(&s),
        a,
        space.state_hash(&space.apply(&s, a)),
        1.0,
    );
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 1);
    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert!(
        cache.get(space.state_hash(&s), a).is_none(),
        "cleared entry must not be retrievable"
    );
    // Re-populate and verify it returns again.
    cache.insert(
        space.state_hash(&s),
        a,
        space.state_hash(&space.apply(&s, a)),
        0.42,
    );
    let got = cache
        .get(space.state_hash(&s), a)
        .expect("re-inserted must hit");
    assert!((got.1 - 0.42).abs() < 1e-6);
}

#[test]
fn inference_action_is_4_bytes() {
    assert_eq!(
        std::mem::size_of::<InferenceAction>(),
        4,
        "InferenceAction must be 4 bytes (#[repr(C)] u16 + u8 + 1 pad)"
    );
}

// ── Search integration tests ────────────────────────────────────────────

#[test]
fn search_finds_an_action_on_fresh_cache() {
    // First search: cache is empty, so the FIRST visit to each (state, action)
    // pair is a miss. Within a single run the search may revisit pairs (via
    // different tree paths or the rollout), so hits can be > 0 even on a
    // fresh cache — that's intra-search reuse, the intended behavior.
    let space = ToySpace;
    let root = ToyState { v: 0 };
    let cache: StateActionCache<f32> = StateActionCache::new();
    let mut scratch = SearchScratch::default();
    let result = mcts_search_with_state_action_cache(&space, &root, 50, &cache, &mut scratch);
    let action = result.best_action.expect("search must return an action");
    // The action must be one of the three available.
    assert!(
        ACTIONS.contains(&action),
        "returned action {action:?} must be in the available set"
    );
    assert!(
        result.cache_misses > 0,
        "fresh cache must have at least one miss (the first visit of each pair)"
    );
    assert!(!cache.is_empty(), "search must have populated the cache");
}

#[test]
fn second_search_hits_cache() {
    // Run a search to populate the cache, then run again. The second run
    // should observe MORE cache hits than the first (the cache is pre-
    // populated from run 1, so pairs visited in run 1 that are revisited in
    // run 2 are immediate hits rather than fresh computations).
    let space = ToySpace;
    let root = ToyState { v: 0 };
    let cache: StateActionCache<f32> = StateActionCache::new();
    let mut scratch = SearchScratch::default();

    // First run: populates.
    let r1 = mcts_search_with_state_action_cache(&space, &root, 50, &cache, &mut scratch);
    let entries_after_first = cache.len();
    assert!(entries_after_first > 0, "first run must populate cache");

    // Second run: should have fewer misses (more of the graph is cached).
    let r2 = mcts_search_with_state_action_cache(&space, &root, 50, &cache, &mut scratch);
    assert!(
        r2.cache_misses <= r1.cache_misses,
        "second run should have fewer-or-equal misses (run1={}, run2={})",
        r1.cache_misses,
        r2.cache_misses
    );
    assert!(
        r2.cache_hits > 0,
        "second run must observe cache hits (got {})",
        r2.cache_hits
    );
}

#[test]
fn search_is_deterministic_across_runs() {
    // The DeterministicTransition contract + first-available-action rollout
    // means the search is deterministic: two runs on fresh caches with the
    // same budget must produce the same cache contents (same len) and the
    // same best action (or at least both valid).
    let space = ToySpace;
    let root = ToyState { v: 0 };

    let cache1: StateActionCache<f32> = StateActionCache::new();
    let mut scratch1 = SearchScratch::default();
    let r1 = mcts_search_with_state_action_cache(&space, &root, 30, &cache1, &mut scratch1);

    let cache2: StateActionCache<f32> = StateActionCache::new();
    let mut scratch2 = SearchScratch::default();
    let r2 = mcts_search_with_state_action_cache(&space, &root, 30, &cache2, &mut scratch2);

    assert_eq!(
        cache1.len(),
        cache2.len(),
        "deterministic search must produce identical cache sizes"
    );
    assert_eq!(
        r1.best_action, r2.best_action,
        "deterministic search must pick the same action"
    );
    assert_eq!(r1.cache_hits, r2.cache_hits);
    assert_eq!(r1.cache_misses, r2.cache_misses);
}

#[test]
fn search_returns_none_for_terminal_root() {
    // A root with no actions (terminal) returns best_action = None.
    let space = ToySpace;
    let root = ToyState { v: MAX_V };
    let cache: StateActionCache<f32> = StateActionCache::new();
    let mut scratch = SearchScratch::default();
    let result = mcts_search_with_state_action_cache(&space, &root, 10, &cache, &mut scratch);
    assert!(
        result.best_action.is_none(),
        "terminal root must return None"
    );
    assert_eq!(result.tree_size, 0);
    assert_eq!(result.cache_hits, 0);
    assert_eq!(result.cache_misses, 0);
}

// ── Phase 2 T2.2 — Determinism re-check (debug-only) ────────────────────

#[test]
fn verify_determinism_returns_zero_on_deterministic_space() {
    // Populate the cache by running a search, then audit every (state, action)
    // pair the search could have visited. On a deterministic space, re-applying
    // the action must reproduce the cached next-state hash exactly.
    let space = ToySpace;
    let root = ToyState { v: 0 };
    let cache: StateActionCache<f32> = StateActionCache::new();
    let mut scratch = SearchScratch::default();
    mcts_search_with_state_action_cache(&space, &root, 50, &cache, &mut scratch);

    // Audit all reachable (state, action) pairs on this 4-step graph.
    // The graph has states v ∈ {0,1,2,3} and actions {0,1,2} (3 actions when
    // non-terminal). Provide every non-terminal (state, action) pair.
    let mut samples: Vec<(ToyState, InferenceAction)> = Vec::new();
    for v in 0..MAX_V {
        for &a in &ACTIONS {
            samples.push((ToyState { v }, a));
        }
    }

    #[cfg(debug_assertions)]
    {
        let mismatches = cache.verify_determinism(&space, &samples);
        assert_eq!(
            mismatches, 0,
            "deterministic space must have 0 determinism mismatches"
        );
    }
}

#[test]
fn verify_determinism_skips_uncached_pairs() {
    // If we audit a (state, action) pair that was never cached, verify_determinism
    // must skip it (a cache miss is not a determinism violation).
    let space = ToySpace;
    let cache: StateActionCache<f32> = StateActionCache::new();
    // Insert one known transition.
    let s = ToyState { v: 0 };
    let a = InferenceAction::new(1, 0);
    let next = space.apply(&s, a);
    cache.insert(space.state_hash(&s), a, space.state_hash(&next), 0.5);

    // Audit includes the cached pair PLUS an uncached pair.
    let samples = vec![
        (s.clone(), a),
        (ToyState { v: 2 }, InferenceAction::new(2, 0)), // uncached
    ];

    #[cfg(debug_assertions)]
    {
        let mismatches = cache.verify_determinism(&space, &samples);
        // 0 mismatches: the cached pair is deterministic, the uncached pair is skipped.
        assert_eq!(mismatches, 0);
    }
}

// ── Phase 2 T2.3 — Cache invalidation semantics (clear / re-populate) ──

#[test]
fn cache_invalidation_clear_then_lookup_returns_none_then_some() {
    // Full invalidation lifecycle: populate → verify hit → clear → verify miss
    // → re-populate → verify hit again.
    let space = ToySpace;
    let cache: StateActionCache<f32> = StateActionCache::new();
    let s = ToyState { v: 1 };
    let a = InferenceAction::new(1, 0);
    let next_hash = space.state_hash(&space.apply(&s, a));

    // Populate.
    cache.insert(space.state_hash(&s), a, next_hash, 0.8);
    assert!(
        cache.get(space.state_hash(&s), a).is_some(),
        "after insert: must hit"
    );

    // Clear.
    cache.clear();
    assert!(
        cache.get(space.state_hash(&s), a).is_none(),
        "after clear: must miss"
    );

    // Re-populate.
    cache.insert(space.state_hash(&s), a, next_hash, 0.9);
    let got = cache
        .get(space.state_hash(&s), a)
        .expect("after re-insert: must hit");
    assert!(
        (got.1 - 0.9).abs() < 1e-6,
        "re-inserted reward must be the new value"
    );
    assert_eq!(got.0, next_hash, "re-inserted next-state hash must match");
}
