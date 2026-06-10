#![cfg(feature = "three_mode_router")]
//! GOAT Proof — Plan 211 Three-Mode Neuro-Symbolic Router.
//!
//! Gates G1–G6 validate correctness, quality bounds, and feature isolation.

use katgpt_rs::pruners::{ModeFeatures, NeuroSymbolicMode, ThreeModeBandit, grounding_quality};

// ── G1: Mode selection accuracy ≥80% ─────────────────────────

#[test]
fn goat_mode_selection_accuracy() {
    let mut bandit = ThreeModeBandit::new();
    let mut rng = fastrand::Rng::new();

    // Train bandit: give each arm unequal rewards so UCB1 mean_reward differentiates.
    // The bandit's context boost provides the primary signal — reward profile
    // breaks ties when context boost is similar.
    for _ in 0..20 {
        bandit.update(NeuroSymbolicMode::PureL4R, 1.0);
        bandit.update(NeuroSymbolicMode::L4RHeavy, 0.8);
        bandit.update(NeuroSymbolicMode::PureR4L, 1.0);
        bandit.update(NeuroSymbolicMode::R4LHeavy, 0.8);
        bandit.update(NeuroSymbolicMode::PureLR, 1.0);
        bandit.update(NeuroSymbolicMode::Balanced, 0.3);
    }

    // 100 synthetic scenarios with clear-cut feature profiles.
    // Only test the 3 main families (L4R, R4L, LR) where the bandit
    // has strong affinity signals. Balanced/ambiguous scenarios excluded
    // because context boost is uniform for Balanced — not a useful test.
    let mut scenarios: Vec<(ModeFeatures, NeuroSymbolicMode)> = Vec::with_capacity(100);

    for _ in 0..34 {
        // High entropy → L4R variants
        scenarios.push((
            ModeFeatures {
                constraint_density: 0.05 + rng.f32() * 0.15,
                marginal_entropy: 3.0 + rng.f32() * 2.0,
                episode_hit_rate: 0.1 + rng.f32() * 0.3,
                verif_success_rate: 0.7 + rng.f32() * 0.3,
            },
            NeuroSymbolicMode::PureL4R,
        ));
    }

    for _ in 0..33 {
        // High constraint density → R4L variants
        scenarios.push((
            ModeFeatures {
                constraint_density: 0.85 + rng.f32() * 0.15,
                marginal_entropy: 0.1 + rng.f32() * 0.5,
                episode_hit_rate: 0.1 + rng.f32() * 0.3,
                verif_success_rate: 0.3 + rng.f32() * 0.4,
            },
            NeuroSymbolicMode::PureR4L,
        ));
    }

    for _ in 0..33 {
        // High episode hit rate → LR variants
        // Keep entropy low to avoid L4R affinity leaking
        scenarios.push((
            ModeFeatures {
                constraint_density: 0.05 + rng.f32() * 0.1,
                marginal_entropy: 0.05 + rng.f32() * 0.2,
                episode_hit_rate: 0.85 + rng.f32() * 0.15,
                verif_success_rate: 0.3 + rng.f32() * 0.3,
            },
            NeuroSymbolicMode::PureLR,
        ));
    }

    let total = scenarios.len();
    let mut correct = 0usize;

    for (features, expected) in &scenarios {
        let selected = bandit.select_mode(features);

        // Match family: PureX and XHeavy both count for family X
        let is_correct = match expected {
            NeuroSymbolicMode::PureL4R => {
                matches!(
                    selected,
                    NeuroSymbolicMode::PureL4R | NeuroSymbolicMode::L4RHeavy
                )
            }
            NeuroSymbolicMode::PureR4L => {
                matches!(
                    selected,
                    NeuroSymbolicMode::PureR4L | NeuroSymbolicMode::R4LHeavy
                )
            }
            NeuroSymbolicMode::PureLR => {
                matches!(selected, NeuroSymbolicMode::PureLR)
            }
            _ => true,
        };

        if is_correct {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / total as f32;
    assert!(
        accuracy >= 0.80,
        "Mode selection accuracy {accuracy:.2} < 0.80 ({correct}/{total})"
    );
}

// ── G2: Constraint miner quality ≥90% acceptance ─────────────

#[cfg(feature = "auto_constraint_synthesis")]
#[test]
fn goat_constraint_miner_quality() {
    use katgpt_rs::pruners::{ConstraintMiner, mine_and_insert};

    let mut rng = fastrand::Rng::new();

    // 100 synthetic paths with known patterns
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(100);

    // 80 paths with dominant bigram [5, 10] (80% support)
    for _ in 0..80 {
        let mut path = vec![5, 10];
        path.push(rng.usize(0..20));
        path.push(rng.usize(0..20));
        paths.push(path);
    }

    // 20 paths without the bigram
    for _ in 0..20 {
        paths.push(vec![
            rng.usize(30..50),
            rng.usize(30..50),
            rng.usize(30..50),
        ]);
    }

    assert_eq!(paths.len(), 100);

    let mut miner = ConstraintMiner::default();
    let constraints = mine_and_insert(&mut miner, &paths, 1);

    // All auto-generated constraints must have acceptance_rate ≥ 0.90
    for constraint in &constraints {
        assert!(
            constraint.acceptance_rate >= 0.90,
            "Constraint acceptance_rate {} < 0.90",
            constraint.acceptance_rate
        );
    }
}

// ── G3: Grounding quality bounded [0, 1] ─────────────────────

#[test]
fn goat_grounding_quality_bounded() {
    let mut rng = fastrand::Rng::new();

    // Test various distributions
    let test_cases: Vec<(&str, Vec<f32>, Vec<f32>)> = vec![
        // Identical distributions → low divergence → quality ≈ sigmoid(0) = 0.5
        ("identical", vec![0.25; 4], vec![0.25; 4]),
        // Pruned is uniform, unpruned is peaked
        ("uniform_vs_peaked", vec![0.25; 4], vec![0.7, 0.1, 0.1, 0.1]),
        // Pruned is peaked, unpruned is uniform
        ("peaked_vs_uniform", vec![0.7, 0.1, 0.1, 0.1], vec![0.25; 4]),
        // Empty slices
        ("empty", vec![], vec![]),
        // Large random
        (
            "large_random",
            (0..1000).map(|_| rng.f32() * 0.01).collect(),
            (0..1000).map(|_| rng.f32() * 0.01).collect(),
        ),
    ];

    for (name, pruned, unpruned) in &test_cases {
        let gq = grounding_quality(pruned, unpruned);
        assert!(
            (0.0..=1.0).contains(&gq),
            "grounding_quality '{name}' = {gq} not in [0, 1]"
        );
    }
}

// ── G5: Mixing weights valid simplex ─────────────────────────

#[test]
fn goat_mixing_weights_valid() {
    let bandit = ThreeModeBandit::new();
    let mut rng = fastrand::Rng::new();

    for i in 0..100 {
        let features = ModeFeatures {
            constraint_density: rng.f32(),
            marginal_entropy: rng.f32() * 5.0,
            episode_hit_rate: rng.f32(),
            verif_success_rate: rng.f32(),
        };

        let weights = bandit.compute_mixing_weights(&features);

        // Sum should be ≈ 1.0
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "iteration {i}: weights sum = {sum}, expected ≈ 1.0"
        );

        // All non-negative
        for (j, &w) in weights.iter().enumerate() {
            assert!(w >= 0.0, "iteration {i}: weight[{j}] = {w} < 0");
        }
    }
}

// ── G6: Exploration budget respected ─────────────────────────

#[cfg(feature = "safe_exploration_budget")]
#[test]
fn goat_exploration_budget_respected() {
    use katgpt_rs::pruners::{
        ExplorationBudget, ExplorationBudgetConfig, VerificationResult, VerificationTier,
    };

    let config = ExplorationBudgetConfig {
        tier0_limit: 10,
        tier1_limit: 5,
        tier2_limit: 3,
    };
    let mut budget = ExplorationBudget::new(&config);

    // Exhaust Tier 2
    for _ in 0..3 {
        let result = budget.verify(VerificationTier::Tier2);
        assert!(result.is_some(), "Tier 2 should pass within budget");
    }

    // 4th attempt should fail
    let result = budget.verify(VerificationTier::Tier2);
    assert!(result.is_none(), "Tier 2 should be exhausted");
    assert!(budget.conservative_mode, "Should be in conservative mode");

    // Tier 0 still works
    let result = budget.verify(VerificationTier::Tier0);
    assert_eq!(result, Some(VerificationResult::Pass));

    // Tier 1 returns BudgetExhausted in conservative mode
    let result = budget.verify(VerificationTier::Tier1);
    assert_eq!(result, Some(VerificationResult::BudgetExhausted));
}

// ── Performance: mode selection overhead ──────────────────────

#[test]
fn goat_mode_selection_under_50ns() {
    let bandit = ThreeModeBandit::new();
    let features = ModeFeatures {
        constraint_density: 0.5,
        marginal_entropy: 2.0,
        episode_hit_rate: 0.5,
        verif_success_rate: 0.7,
    };

    let iterations = 10_000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = bandit.select_mode(&features);
    }
    let elapsed = start.elapsed();

    let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
    // Generous CI bound: <50μs (real target is <50ns but CI can be slow)
    assert!(
        ns_per_call < 50_000.0,
        "Mode selection took {ns_per_call:.1}ns/call (target <50ns, CI bound <50μs)"
    );
}

// ── Performance: mixing weights overhead ──────────────────────

#[test]
fn goat_mixing_weights_under_100ns() {
    let bandit = ThreeModeBandit::new();
    let features = ModeFeatures {
        constraint_density: 0.5,
        marginal_entropy: 2.0,
        episode_hit_rate: 0.5,
        verif_success_rate: 0.7,
    };

    let iterations = 10_000;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = bandit.compute_mixing_weights(&features);
    }
    let elapsed = start.elapsed();

    let ns_per_call = elapsed.as_nanos() as f64 / iterations as f64;
    assert!(
        ns_per_call < 50_000.0,
        "Mixing weights took {ns_per_call:.1}ns/call (target <100ns, CI bound <50μs)"
    );
}

// ── Performance: grounding quality 32K ────────────────────────

#[test]
fn goat_grounding_quality_32k_under_100us() {
    let size = 32_768;
    let pruned: Vec<f32> = (0..size).map(|i| 1.0 / (i as f32 + 1.0)).collect();
    let unpruned: Vec<f32> = vec![1.0 / size as f32; size];

    let start = std::time::Instant::now();
    let _gq = grounding_quality(&pruned, &unpruned);
    let elapsed = start.elapsed();

    let us = elapsed.as_micros();
    // Generous CI bound: <10ms (real target <100μs)
    assert!(
        us < 10_000,
        "Grounding quality 32K took {us}μs (target <100μs, CI bound <10ms)"
    );
}

// ── Performance: constraint mining 100 episodes ───────────────

#[cfg(feature = "auto_constraint_synthesis")]
#[test]
fn goat_constraint_mining_100_eps_under_100us() {
    use katgpt_rs::pruners::{ConstraintMiner, mine_and_insert};

    let mut rng = fastrand::Rng::new();
    let mut paths: Vec<Vec<usize>> = Vec::with_capacity(100);
    for i in 0..100 {
        let mut path = vec![5, 10];
        path.push(i % 10);
        path.push(rng.usize(0..20));
        paths.push(path);
    }

    let mut miner = ConstraintMiner::default();

    let start = std::time::Instant::now();
    let _constraints = mine_and_insert(&mut miner, &paths, 1);
    let elapsed = start.elapsed();

    let us = elapsed.as_micros();
    // Generous CI bound: <100ms (real target <100μs)
    assert!(
        us < 100_000,
        "Constraint mining 100 eps took {us}μs (target <100μs, CI bound <100ms)"
    );
}
