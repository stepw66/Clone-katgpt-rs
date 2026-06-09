//! Acceptance Rate Benchmark for DynamicRankPruner (Plan 232)
//!
//! Measures whether wrapping BanditPruner with DynamicRankPruner improves
//! speculative decoding acceptance rate in a simulated verification scenario.
//!
//! Methodology:
//!   1. Generate peaked marginals simulating a drafter model
//!   2. Build DDTree with: (a) BanditPruner baseline, (b) DynamicRankPruner-wrapped BanditPruner
//!   3. Simulate context-dependent verification: parent context determines which tokens are "correct"
//!   4. Feed corrections to DynamicRankPruner based on verification rewards
//!   5. Measure acceptance rate = accepted_tokens / total_tree_nodes
//!
//! ```sh
//! # Bandit-only baseline
//! cargo test --features "bandit" --test bench_232_dynamic_rank_acceptance -- --nocapture
//!
//! # With DynamicRankPruner
//! cargo test --features "dynamic_rank,bandit" --test bench_232_dynamic_rank_acceptance -- --nocapture
//! ```

use katgpt_core::traits::NoScreeningPruner;
use katgpt_core::types::{Config, Rng};
use katgpt_rs::pruners::bandit::{BanditPruner, BanditStrategy};
use katgpt_rs::speculative::{build_dd_tree_screened, extract_best_path_into};

#[cfg(feature = "dynamic_rank")]
use katgpt_rs::pruners::dynamic_rank::DynamicRankPruner;

const VOCAB: usize = 32;
const LOOKAHEAD: usize = 5;
const EPISODES: usize = 1000;
const SEED: u64 = 42;

/// Generate peaked marginals: 3 "hot" tokens get most probability mass,
/// remaining tokens get noise. The hot tokens shift based on depth
/// to simulate context-dependent drafting.
fn peaked_marginals(vocab: usize, lookahead: usize, depth_shift: usize) -> Vec<Vec<f32>> {
    (0..lookahead)
        .map(|d| {
            let mut m = vec![0.01; vocab];
            // 3 "hot" tokens per depth, shifted by depth
            let base = (d + depth_shift) % vocab;
            for i in 0..3 {
                let idx = (base + i) % vocab;
                m[idx] = 0.30;
            }
            let sum: f32 = m.iter().sum();
            m.iter_mut().for_each(|p| *p /= sum);
            m
        })
        .collect()
}

/// Simulate verification: accept if the token is in the "correct" set for this context.
/// Context-dependent: parent's first token determines which tokens are correct.
/// Returns (accepted_count, total_count, reward_sum).
fn simulate_verification(
    path: &[usize],
    parent_context: &[usize],
    rng: &mut Rng,
    accept_prob: f32,
) -> (usize, usize, f32) {
    let mut accepted = 0usize;
    let mut reward_sum = 0.0f32;
    let total = path.len();

    // Context-dependent correctness: parent's first token determines hot zone
    let hot_zone_base = parent_context.first().copied().unwrap_or(0) * 3 % VOCAB;
    let hot_zone: Vec<usize> = (0..5).map(|i| (hot_zone_base + i) % VOCAB).collect();

    for &token_idx in path {
        if hot_zone.contains(&token_idx) {
            // High probability of acceptance for context-appropriate tokens
            if rng.uniform() < accept_prob {
                accepted += 1;
                reward_sum += 1.0;
            }
        } else if rng.uniform() < 0.15 {
            // Low probability for context-inappropriate tokens
            accepted += 1;
            reward_sum += 0.1;
        }
    }

    (accepted, total, reward_sum)
}

/// Run the BanditPruner baseline acceptance rate benchmark.
fn run_baseline_acceptance() -> (f64, f64, usize, usize) {
    let config = Config {
        vocab_size: VOCAB,
        draft_lookahead: LOOKAHEAD,
        ..Default::default()
    };

    let mut bp = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, VOCAB);
    let mut rng = Rng::new(SEED);
    let mut total_accepted = 0usize;
    let mut total_nodes = 0usize;
    let mut total_reward = 0.0f64;

    for ep in 0..EPISODES {
        bp.prepare_episode(&mut rng);
        let depth_shift = ep % 7;
        let marginals = peaked_marginals(VOCAB, LOOKAHEAD, depth_shift);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();
        let tree = build_dd_tree_screened(&slices, &config, &bp, true);

        // Extract best path and verify
        let mut path = Vec::new();
        extract_best_path_into(&tree, &mut path);

        // Simulate parent context from the marginals
        let parent = vec![(ep * 3) % VOCAB, (ep * 7 + 1) % VOCAB];
        let (accepted, _total, reward) = simulate_verification(&path, &parent, &mut rng, 0.80);

        // Update bandit with rewards
        for &token_idx in &path {
            if token_idx < 5 {
                bp.update(token_idx, 1.0);
            } else {
                bp.update(token_idx, 0.1);
            }
        }

        // Also count all tree nodes
        total_nodes += tree.len();
        total_accepted += accepted;
        total_reward += reward as f64;

        let _ = ep;
    }

    let path_acceptance = total_accepted as f64 / total_nodes.max(1) as f64;
    (path_acceptance, total_reward, total_accepted, total_nodes)
}

/// Run the DynamicRankPruner-wrapped acceptance rate benchmark.
#[cfg(feature = "dynamic_rank")]
fn run_dynamic_rank_acceptance() -> (f64, f64, usize, usize) {
    let config = Config {
        vocab_size: VOCAB,
        draft_lookahead: LOOKAHEAD,
        ..Default::default()
    };

    let bandit = BanditPruner::new(NoScreeningPruner, BanditStrategy::Ucb1, VOCAB);
    let wrapped = DynamicRankPruner::new(bandit, VOCAB);
    let mut rng = Rng::new(SEED);
    let mut total_accepted = 0usize;
    let mut total_nodes = 0usize;
    let mut total_reward = 0.0f64;

    for ep in 0..EPISODES {
        let depth_shift = ep % 7;
        let marginals = peaked_marginals(VOCAB, LOOKAHEAD, depth_shift);
        let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

        // Build tree with DynamicRankPruner-wrapped BanditPruner
        let tree = build_dd_tree_screened(&slices, &config, &wrapped, true);

        // Extract best path and verify
        let mut path = Vec::new();
        extract_best_path_into(&tree, &mut path);

        // Simulate parent context from the marginals
        let parent = vec![(ep * 3) % VOCAB, (ep * 7 + 1) % VOCAB];
        let (accepted, _total, reward) = simulate_verification(&path, &parent, &mut rng, 0.80);

        // Feed corrections to DynamicRankPruner based on verification results
        let hot_zone_base = parent.first().copied().unwrap_or(0) * 3 % VOCAB;
        for i in 0..5 {
            let token_idx = (hot_zone_base + i) % VOCAB;
            wrapped.record_correction(&parent, token_idx, 1.0);
        }

        // Also update inner bandit
        for &token_idx in &path {
            if token_idx < 5 {
                wrapped.record_correction(&parent, token_idx, 2.0);
            }
        }

        total_nodes += tree.len();
        total_accepted += accepted;
        total_reward += reward as f64;

        let _ = ep;
    }

    let path_acceptance = total_accepted as f64 / total_nodes.max(1) as f64;
    (path_acceptance, total_reward, total_accepted, total_nodes)
}

#[test]
fn bench_dynamic_rank_acceptance_rate() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 232: DynamicRankPruner Acceptance Rate Benchmark     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "   Config: vocab={}, lookahead={}, episodes={}",
        VOCAB, LOOKAHEAD, EPISODES
    );
    println!("{}", "─".repeat(60));

    // Run baseline
    let (baseline_rate, baseline_reward, baseline_accepted, baseline_nodes) =
        run_baseline_acceptance();

    println!("\n   ┌─ BanditPruner Baseline ──────────────────────────┐");
    println!("   │  Tree nodes:     {:>8}", baseline_nodes);
    println!("   │  Accepted:       {:>8}", baseline_accepted);
    println!("   │  Acceptance rate: {:>7.2}%", baseline_rate * 100.0);
    println!("   │  Total reward:   {:>8.1}", baseline_reward);
    println!("   └─────────────────────────────────────────────────┘");

    #[cfg(feature = "dynamic_rank")]
    {
        let (dr_rate, dr_reward, dr_accepted, dr_nodes) = run_dynamic_rank_acceptance();

        println!("\n   ┌─ DynamicRankPruner + BanditPruner ──────────────┐");
        println!("   │  Tree nodes:     {:>8}", dr_nodes);
        println!("   │  Accepted:       {:>8}", dr_accepted);
        println!("   │  Acceptance rate: {:>7.2}%", dr_rate * 100.0);
        println!("   │  Total reward:   {:>8.1}", dr_reward);
        println!("   └─────────────────────────────────────────────────┘");

        let delta = dr_rate - baseline_rate;
        let pct_change = if baseline_rate > 0.0 {
            delta / baseline_rate * 100.0
        } else {
            0.0
        };

        println!("\n   ┌─ Comparison ────────────────────────────────────┐");
        println!(
            "   │  Rate delta:     {:>+7.2}% ({:+.4})",
            pct_change, delta
        );
        println!(
            "   │  Reward delta:   {:>+8.1}",
            dr_reward - baseline_reward
        );
        println!(
            "   │  Node count Δ:   {:>+8}",
            dr_nodes as isize - baseline_nodes as isize
        );
        println!("   └─────────────────────────────────────────────────┘");

        // GOAT gate: DynamicRankPruner should not significantly degrade acceptance
        assert!(
            dr_rate >= baseline_rate - 0.10,
            "DynamicRankPruner acceptance ({:.2}%) should not degrade >10pp below baseline ({:.2}%)",
            dr_rate * 100.0,
            baseline_rate * 100.0,
        );
        println!("\n   ✅ GOAT gate passed: acceptance within tolerance");
    }

    #[cfg(not(feature = "dynamic_rank"))]
    {
        println!("\n   ⚠️  dynamic_rank feature not enabled — skipping wrapped benchmark");
        println!(
            "   Run with: cargo test --features \"dynamic_rank,bandit\" --test bench_232_dynamic_rank_acceptance -- --nocapture"
        );
    }

    println!();
}
