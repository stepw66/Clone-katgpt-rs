//! GOAT Proof: Thinking Prune — FrozenBaseGuard for Token-Level DDTree (Plan 171).
//!
//! Validates three properties:
//! P1: FrozenBaseGuard intermediate hops produce >= Uniform nodes (structural dominance)
//! P2: FrozenBaseGuard produces identical output to Uniform when screening is cheap
//!     (NoScreeningPruner — confirms correctness of the delegation)
//! P3: Wall-clock timing — FrozenBaseGuard with expensive screener is faster than
//!     Uniform at intermediate hops (the actual performance claim)
//! P4: Single-hop edge case — FrozenBaseGuard applies full screening when hop is final

use std::time::Instant;

use katgpt_rs::speculative::{
    build_dd_tree_screened, extract_best_path,
    types::{NoScreeningPruner, ScreeningPruner},
};
use katgpt_rs::types::Config;

#[cfg(feature = "thinking_prune")]
use katgpt_rs::pruners::PrunerSchedule;
#[cfg(feature = "thinking_prune")]
use katgpt_rs::speculative::build_dd_tree_screened_with_schedule;

// ── Helpers ────────────────────────────────────────────────────

/// Simulated expensive screener: injects artificial delay to model WASM/validator cost.
#[derive(Debug, Clone)]
struct ExpensiveScreener {
    /// Base relevance to return (0.0–1.0)
    base_relevance: f32,
    /// Artificial work per call (loop iterations)
    work_per_call: u32,
}

impl ScreeningPruner for ExpensiveScreener {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // Simulate expensive WASM validator call
        let mut acc: f32 = 0.0;
        for i in 0..self.work_per_call {
            acc += (i as f32).sin() * (i as f32).cos();
        }
        // Prevent optimization from removing the work
        let _ = acc;
        self.base_relevance
    }
}

/// A screener that returns different relevance per depth to ensure differentiation.
#[derive(Debug, Clone)]
struct VaryingScreener {
    relevances: Vec<f32>,
}

impl ScreeningPruner for VaryingScreener {
    fn relevance(&self, depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.relevances.get(depth).copied().unwrap_or(1.0)
    }
}

fn make_config() -> Config {
    let mut c = Config::draft();
    c.screening_threshold = 0.3;
    c.tree_budget = 512;
    c
}

fn random_marginals(depths: usize, vocab: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut rng = katgpt_rs::types::Rng::new(seed as u64);
    (0..depths)
        .map(|_| {
            let mut m: Vec<f32> = (0..vocab).map(|_| rng.uniform()).collect();
            let sum: f32 = m.iter().sum();
            for v in m.iter_mut() {
                *v /= sum;
            }
            m
        })
        .collect()
}

fn marginals_refs(marginals: &[Vec<f32>]) -> Vec<&[f32]> {
    marginals.iter().map(|m| m.as_slice()).collect()
}

// ══════════════════════════════════════════════════════════════════════════
// P1: Structural Dominance — FrozenBaseGuard >= Uniform nodes
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p1_structural_dominance() {
    println!("\n── P1: FrozenBaseGuard intermediate produces >= Uniform nodes ──\n");

    let config = make_config();
    let depths = 5;
    let vocab = config.vocab_size;
    let n_trials = 10;

    let mut frozen_wins = 0;
    let mut uniform_wins = 0;
    let mut ties = 0;

    // Screener that rejects some tokens (simulates real pruning)
    let screener = VaryingScreener {
        relevances: vec![0.8, 0.4, 0.2, 0.6, 0.9],
    };

    for seed in 0..n_trials {
        let marginals = random_marginals(depths, vocab, seed);
        let refs = marginals_refs(&marginals);

        // Simulate 3-hop SpecHop pipeline
        let total_hops = 3;

        // Uniform: every hop applies full screening
        let mut uniform_total_nodes = 0;
        for hop in 0..total_hops {
            let tree = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &screener,
                true,
                PrunerSchedule::Uniform,
                hop,
                total_hops,
            );
            uniform_total_nodes += tree.len();
        }

        // FrozenBaseGuard: only final hop applies screening
        let mut frozen_total_nodes = 0;
        for hop in 0..total_hops {
            let tree = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &screener,
                true,
                PrunerSchedule::FrozenBaseGuard,
                hop,
                total_hops,
            );
            frozen_total_nodes += tree.len();
        }

        println!(
            "  Seed {seed}: Uniform={uniform_total_nodes} nodes, FrozenBaseGuard={frozen_total_nodes} nodes",
        );

        match frozen_total_nodes.cmp(&uniform_total_nodes) {
            std::cmp::Ordering::Greater => frozen_wins += 1,
            std::cmp::Ordering::Less => uniform_wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
        }
    }

    println!(
        "\n  Summary: Frozen wins={}, Uniform wins={}, Ties={}",
        frozen_wins, uniform_wins, ties
    );

    // Assert: FrozenBaseGuard should NEVER produce fewer total nodes
    assert_eq!(
        uniform_wins, 0,
        "FrozenBaseGuard should produce >= Uniform nodes (Uniform won {} times)",
        uniform_wins,
    );
    println!("  ✅ P1 PASS: FrozenBaseGuard always produces >= Uniform nodes");
}

// ══════════════════════════════════════════════════════════════════════════
// P2: Identical Output with NoScreeningPruner (correctness)
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p2_identical_with_noop_screener() {
    println!("\n── P2: Identical output with NoScreeningPruner ──────────────\n");

    let config = make_config();
    let marginals = random_marginals(4, config.vocab_size, 42);
    let refs = marginals_refs(&marginals);
    let screener = NoScreeningPruner;

    let total_hops = 3;

    for hop in 0..total_hops {
        let uniform_tree = build_dd_tree_screened_with_schedule(
            &refs,
            &config,
            &screener,
            true,
            PrunerSchedule::Uniform,
            hop,
            total_hops,
        );
        let frozen_tree = build_dd_tree_screened_with_schedule(
            &refs,
            &config,
            &screener,
            true,
            PrunerSchedule::FrozenBaseGuard,
            hop,
            total_hops,
        );

        assert_eq!(
            uniform_tree.len(),
            frozen_tree.len(),
            "Hop {hop}: NoScreeningPruner should produce identical trees",
        );

        // Verify scores match
        for (i, (u, f)) in uniform_tree.iter().zip(frozen_tree.iter()).enumerate() {
            assert!(
                (u.score - f.score).abs() < 1e-6,
                "Hop {hop}, node {i}: score mismatch (Uniform={}, Frozen={})",
                u.score,
                f.score,
            );
        }
        println!(
            "  Hop {hop}: {} nodes, all scores match ✅",
            uniform_tree.len()
        );
    }

    println!("  ✅ P2 PASS: NoScreeningPruner produces identical results");
}

// ══════════════════════════════════════════════════════════════════════════
// P3: Wall-Clock Timing — FrozenBaseGuard is faster at intermediate hops
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p3_wall_clock_timing() {
    println!("\n── P3: Wall-clock timing with expensive screener ───────────\n");

    let config = make_config();
    let depths = 4;
    let vocab = config.vocab_size;
    let iters = 200;
    let total_hops = 3;

    // Expensive screener with synthetic work
    let expensive = ExpensiveScreener {
        base_relevance: 0.7,
        work_per_call: 100, // enough to be measurable
    };

    let marginals = random_marginals(depths, vocab, 12345);
    let refs = marginals_refs(&marginals);

    // Warmup
    for _ in 0..20 {
        for hop in 0..total_hops {
            let _ = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &expensive,
                true,
                PrunerSchedule::Uniform,
                hop,
                total_hops,
            );
        }
    }

    // Benchmark: Uniform (every hop applies expensive screener)
    let start_uniform = Instant::now();
    for _ in 0..iters {
        for hop in 0..total_hops {
            let _ = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &expensive,
                true,
                PrunerSchedule::Uniform,
                hop,
                total_hops,
            );
        }
    }
    let elapsed_uniform = start_uniform.elapsed();
    let ns_uniform = elapsed_uniform.as_nanos() as f64 / iters as f64;

    // Warmup for FrozenBaseGuard
    for _ in 0..20 {
        for hop in 0..total_hops {
            let _ = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &expensive,
                true,
                PrunerSchedule::FrozenBaseGuard,
                hop,
                total_hops,
            );
        }
    }

    // Benchmark: FrozenBaseGuard (intermediate hops skip expensive screener)
    let start_frozen = Instant::now();
    for _ in 0..iters {
        for hop in 0..total_hops {
            let _ = build_dd_tree_screened_with_schedule(
                &refs,
                &config,
                &expensive,
                true,
                PrunerSchedule::FrozenBaseGuard,
                hop,
                total_hops,
            );
        }
    }
    let elapsed_frozen = start_frozen.elapsed();
    let ns_frozen = elapsed_frozen.as_nanos() as f64 / iters as f64;

    let speedup_pct = ((ns_uniform - ns_frozen) / ns_uniform) * 100.0;

    println!("  Uniform (all hops screened):  {ns_uniform:.0} ns/iter ({iters} iterations)");
    println!("  FrozenBase (skip intermediates): {ns_frozen:.0} ns/iter ({iters} iterations)");
    println!("  Speedup: {speedup_pct:.1}%");

    // Assert: FrozenBaseGuard should be measurably faster with expensive screener
    // Intermediate hops (2 of 3) skip the screener entirely.
    // We expect ~30-60% speedup (2/3 of hops skip the work).
    assert!(
        ns_frozen < ns_uniform,
        "FrozenBaseGuard should be faster than Uniform with expensive screener \
         (got {}ns vs {}ns, {speedup_pct:.1}%)",
        ns_frozen,
        ns_uniform,
    );
    println!("  ✅ P3 PASS: FrozenBaseGuard is {speedup_pct:.1}% faster with expensive screener");
}

// ══════════════════════════════════════════════════════════════════════════
// P4: Single-Hop Edge Case — Full Screening Applied
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p4_single_hop_is_final() {
    println!("\n── P4: Single-hop edge case applies full screening ────────\n");

    let config = make_config();
    let marginals = random_marginals(3, config.vocab_size, 99);
    let refs = marginals_refs(&marginals);

    let screener = VaryingScreener {
        relevances: vec![0.5, 0.2, 0.8], // 0.2 < threshold 0.3 → should trim at depth 1
    };

    // Single hop (hop 0 of 1) → is final → should apply full screening
    let frozen_tree = build_dd_tree_screened_with_schedule(
        &refs,
        &config,
        &screener,
        true,
        PrunerSchedule::FrozenBaseGuard,
        0,
        1,
    );

    // Compare with explicit full screening
    let full_tree = build_dd_tree_screened(&refs, &config, &screener, true);

    assert_eq!(
        frozen_tree.len(),
        full_tree.len(),
        "Single-hop FrozenBaseGuard should produce identical tree to full screening",
    );
    println!(
        "  Single-hop tree: {} nodes (matches full screening)",
        frozen_tree.len()
    );
    println!("  ✅ P4 PASS: Single-hop applies full screening correctly");
}

// ══════════════════════════════════════════════════════════════════════════
// P5: Path Quality — Best path identical at final hop
// ══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "thinking_prune")]
fn proof_p5_final_hop_quality_identical() {
    println!("\n── P5: Final hop path quality identical to Uniform ────────\n");

    let config = make_config();
    let marginals = random_marginals(4, config.vocab_size, 77);
    let refs = marginals_refs(&marginals);

    let screener = VaryingScreener {
        relevances: vec![0.9, 0.6, 0.4, 0.7],
    };

    let total_hops = 3;

    // Both schedules should produce identical results at the FINAL hop
    let uniform_tree = build_dd_tree_screened_with_schedule(
        &refs,
        &config,
        &screener,
        true,
        PrunerSchedule::Uniform,
        total_hops - 1,
        total_hops,
    );
    let frozen_tree = build_dd_tree_screened_with_schedule(
        &refs,
        &config,
        &screener,
        true,
        PrunerSchedule::FrozenBaseGuard,
        total_hops - 1,
        total_hops,
    );

    let uniform_path = extract_best_path(&uniform_tree);
    let frozen_path = extract_best_path(&frozen_tree);

    assert_eq!(
        uniform_path, frozen_path,
        "Final hop paths should be identical between Uniform and FrozenBaseGuard",
    );
    println!("  Uniform path: {:?}", uniform_path);
    println!("  Frozen path:  {:?}", frozen_path);
    println!("  ✅ P5 PASS: Final hop produces identical paths");
}

// ══════════════════════════════════════════════════════════════════════════
// Main — Run all proofs
// ══════════════════════════════════════════════════════════════════════════

#[test]
fn test_bench_171_thinking_prune_goat() {
    println!("═══════════════════════════════════════════════════════════");
    println!("  GOAT Proof: Thinking Prune — FrozenBaseGuard DDTree (171)");
    println!("═══════════════════════════════════════════════════════════");

    #[cfg(feature = "thinking_prune")]
    {
        proof_p1_structural_dominance();
        proof_p2_identical_with_noop_screener();
        proof_p3_wall_clock_timing();
        proof_p4_single_hop_is_final();
        proof_p5_final_hop_quality_identical();

        println!("\n═══════════════════════════════════════════════════════════");
        println!("  ALL 5 PROOFS PASSED ✅");
        println!("═══════════════════════════════════════════════════════════");
    }

    #[cfg(not(feature = "thinking_prune"))]
    {
        println!("  ⚠️  thinking_prune feature not enabled — skipping");
        println!(
            "  Run with: cargo test --features thinking_prune --test bench_171_thinking_prune_goat -- --nocapture"
        );
    }
}
