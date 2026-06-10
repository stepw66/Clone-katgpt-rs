#![cfg(feature = "three_mode_router")]
//! Three-Mode Neuro-Symbolic Router Demo — Plan 211.
//!
//! Demonstrates mode selection, mixing weights, grounding quality,
//! and exploration budget usage.
//!
//! Run: `cargo run --features three_mode_router --example three_mode_router_demo`

use katgpt_rs::pruners::{
    ModeFeatures, NeuroSymbolicMode, ThreeModeBandit, compute_mode_features, grounding_quality,
};

fn main() {
    println!("=== Plan 211: Three-Mode Neuro-Symbolic Router Demo ===\n");

    let mut bandit = ThreeModeBandit::new();

    // ── Scenario 1: High entropy (learning-heavy) ─────────────
    let high_entropy = ModeFeatures {
        constraint_density: 0.1,
        marginal_entropy: 3.5,
        episode_hit_rate: 0.3,
        verif_success_rate: 0.8,
    };
    demo_scenario(&bandit, "High Entropy (learning)", &high_entropy);
    bandit.update(NeuroSymbolicMode::PureL4R, 1.0);

    // ── Scenario 2: High constraints (reasoning-heavy) ────────
    let high_constraints = ModeFeatures {
        constraint_density: 0.9,
        marginal_entropy: 0.5,
        episode_hit_rate: 0.4,
        verif_success_rate: 0.5,
    };
    demo_scenario(&bandit, "High Constraints (reasoning)", &high_constraints);
    bandit.update(NeuroSymbolicMode::PureR4L, 1.0);

    // ── Scenario 3: High episode hits (balanced) ──────────────
    let high_hits = ModeFeatures {
        constraint_density: 0.3,
        marginal_entropy: 1.0,
        episode_hit_rate: 0.9,
        verif_success_rate: 0.6,
    };
    demo_scenario(&bandit, "High Episode Hits", &high_hits);
    bandit.update(NeuroSymbolicMode::PureLR, 1.0);

    // ── Scenario 4: Low everything (default) ──────────────────
    let low_all = ModeFeatures {
        constraint_density: 0.1,
        marginal_entropy: 0.3,
        episode_hit_rate: 0.2,
        verif_success_rate: 0.3,
    };
    demo_scenario(&bandit, "Low Everything", &low_all);

    // ── Grounding Quality ─────────────────────────────────────
    println!("--- Grounding Quality ---");

    let pruned = vec![0.5, 0.3, 0.15, 0.05];
    let unpruned = vec![0.25, 0.25, 0.25, 0.25];
    let gq = grounding_quality(&pruned, &unpruned);
    println!(
        "  Pruned {:?} vs Uniform {:?} → quality = {gq:.4}",
        pruned, unpruned
    );

    let identical = vec![0.25; 4];
    let gq_ident = grounding_quality(&identical, &identical);
    println!("  Identical distributions → quality = {gq_ident:.4} (≈ sigmoid(0) = 0.5)");

    // ── Compute Mode Features from raw state ──────────────────
    println!("\n--- Compute Mode Features ---");
    let token_probs = vec![0.4, 0.3, 0.2, 0.1];
    let features = compute_mode_features(5, 10, &token_probs, 70, 100, 85, 100);
    println!("  Active rules: 5/10, episode hits: 70/100, verif: 85/100");
    println!(
        "  → constraint_density={:.2}, marginal_entropy={:.2}",
        features.constraint_density, features.marginal_entropy
    );
    println!(
        "    episode_hit_rate={:.2}, verif_success_rate={:.2}",
        features.episode_hit_rate, features.verif_success_rate
    );

    // ── Exploration Budget (if feature enabled) ───────────────
    #[cfg(feature = "safe_exploration_budget")]
    {
        use katgpt_rs::pruners::{ExplorationBudget, ExplorationBudgetConfig, VerificationTier};

        println!("\n--- Exploration Budget ---");
        let config = ExplorationBudgetConfig {
            tier0_limit: u32::MAX,
            tier1_limit: 100,
            tier2_limit: 5,
        };
        let mut budget = ExplorationBudget::new(&config);
        println!(
            "  Budget: Tier0={}, Tier1={}, Tier2={}",
            budget.tier0_remaining, budget.tier1_remaining, budget.tier2_remaining
        );

        // Simulate some verifications
        for i in 0..7 {
            let result = budget.verify(VerificationTier::Tier2);
            match result {
                Some(_) => println!("  Tier 2 verify #{}: PASS", i + 1),
                None => println!("  Tier 2 verify #{}: EXHAUSTED (conservative)", i + 1),
            }
        }
        println!("  Conservative mode: {}", budget.conservative_mode);
    }

    #[cfg(not(feature = "safe_exploration_budget"))]
    {
        println!("\n--- Exploration Budget ---");
        println!("  (enable 'safe_exploration_budget' feature to see budget demo)");
    }

    println!("\n=== Demo Complete ===");
}

fn demo_scenario(bandit: &ThreeModeBandit, name: &str, features: &ModeFeatures) {
    let mode = bandit.select_mode(features);
    let weights = bandit.compute_mixing_weights(features);

    println!("--- {name} ---");
    println!(
        "  Features: density={:.2}, entropy={:.2}, hit_rate={:.2}, verif={:.2}",
        features.constraint_density,
        features.marginal_entropy,
        features.episode_hit_rate,
        features.verif_success_rate,
    );
    println!("  Selected mode: {mode:?}");
    println!(
        "  Mixing weights: L4R={:.3}, R4L={:.3}, LR={:.3}",
        weights[0], weights[1], weights[2]
    );
    println!();
}
