//! Plan 390 Phase 1 T1.6 — `mcts_state_action_cache` basic example.
//!
//! Demonstrates a synthetic 3-action space over a 4-step deterministic
//! transition graph, runs the search, prints cache hit/miss statistics, and
//! re-runs with the SAME cache populated to show that the second run observes
//! cache hits (the UnMaskFork budget win).
//!
//! Run with:
//! ```sh
//! cargo run --example mcts_state_action_cache_basic --features mcts_state_action_cache --release
//! ```

use katgpt_core::mcts_state_action_cache::{
    InferenceAction, InferenceActionSpace, SearchScratch, StateActionCache,
    mcts_search_with_state_action_cache,
};

// ── Synthetic deterministic action space ─────────────────────────────────
//
// 3 "inference configurations" (the action axis) over a 4-step transition
// graph. Each state is a single `u8` progress value `v ∈ {0..=3}`:
//   action 0: v += 0  (no-op — stalls)
//   action 1: v += 1  (one step)
//   action 2: v += 2  (two steps — fastest path to terminal)
//
// Terminal reward = v / 3.0. The optimal first action is action 2 (reaches
// terminal in 2 steps). The DeterministicTransition contract holds: `apply`
// is a pure function of (state, action).

#[derive(Clone, Debug)]
struct DemoState {
    v: u8,
}

const MAX_V: u8 = 3;

const ACTIONS: [InferenceAction; 3] = [
    InferenceAction::new(0, 0),
    InferenceAction::new(1, 0),
    InferenceAction::new(2, 0),
];

struct DemoSpace;

impl InferenceActionSpace<DemoState> for DemoSpace {
    fn actions_at(&self, state: &DemoState) -> &[InferenceAction] {
        if state.v >= MAX_V { &[] } else { &ACTIONS }
    }

    fn apply(&self, state: &DemoState, action: InferenceAction) -> DemoState {
        let delta = (action.config_id as u8).min(MAX_V - state.v);
        DemoState { v: state.v + delta }
    }

    fn reward(&self, state: &DemoState) -> Option<f32> {
        if state.v >= MAX_V {
            Some(state.v as f32 / MAX_V as f32)
        } else {
            None
        }
    }

    fn is_terminal(&self, state: &DemoState) -> bool {
        state.v >= MAX_V
    }

    fn state_hash(&self, state: &DemoState) -> blake3::Hash {
        blake3::hash(&[state.v])
    }
}

fn main() {
    let space = DemoSpace;
    let root = DemoState { v: 0 };
    let cache: StateActionCache<f32> = StateActionCache::new();
    let mut scratch = SearchScratch::default();

    println!("=== Plan 390: State-Action Pair Cache for MCTS (UnMaskFork) ===\n");
    println!("Domain: 4-step deterministic transition graph (v: 0 → 3)");
    println!("Actions: 3 inference configurations (config_id 0, 1, 2)");
    println!("Reward: terminal v / 3.0\n");

    // ── Run 1: fresh cache (all misses) ──
    let budget = 100;
    let r1 = mcts_search_with_state_action_cache(&space, &root, budget, &cache, &mut scratch);
    println!("--- Run 1 (fresh cache, budget={budget}) ---");
    println!("  best action:    {:?}", r1.best_action);
    println!("  cache hits:     {}", r1.cache_hits);
    println!("  cache misses:   {}", r1.cache_misses);
    println!("  tree size:      {}", r1.tree_size);
    println!("  cache entries:  {}", cache.len());
    let total1 = r1.cache_hits + r1.cache_misses;
    let hit_rate1 = if total1 > 0 {
        r1.cache_hits as f64 / total1 as f64 * 100.0
    } else {
        0.0
    };
    println!("  hit rate:       {hit_rate1:.1}%\n");

    // ── Run 2: same cache (should see hits) ──
    let r2 = mcts_search_with_state_action_cache(&space, &root, budget, &cache, &mut scratch);
    println!("--- Run 2 (populated cache, budget={budget}) ---");
    println!("  best action:    {:?}", r2.best_action);
    println!("  cache hits:     {}", r2.cache_hits);
    println!("  cache misses:   {}", r2.cache_misses);
    println!("  tree size:      {}", r2.tree_size);
    let total2 = r2.cache_hits + r2.cache_misses;
    let hit_rate2 = if total2 > 0 {
        r2.cache_hits as f64 / total2 as f64 * 100.0
    } else {
        0.0
    };
    println!("  hit rate:       {hit_rate2:.1}%\n");

    // ── Summary ──
    println!("=== Summary ===");
    println!(
        "Run 2 hit rate ({:.1}%) > Run 1 hit rate ({:.1}%): {}",
        hit_rate2,
        hit_rate1,
        if hit_rate2 > hit_rate1 {
            "YES — cache reuse is working"
        } else {
            "NO — investigate"
        }
    );
    println!("\nThe cache converts deterministic-transition revisits into");
    println!("zero-NFE hits. This is the UnMaskFork budget-expansion primitive.");
}
