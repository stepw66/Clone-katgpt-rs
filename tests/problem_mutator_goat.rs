//! Plan 191 T2.6: GOAT proof that mutated configs produce ≥1.5× arm diversity.
//!
//! Run two 4-arm bandit sessions:
//! 1. **Fixed config**: Same `GameConfig` every round → generalist arm dominates.
//! 2. **Mutated config**: `EvolutionArena` produces diverse configs each round →
//!    different arms become optimal in different rounds → more active arms.
//!
//! Config mutation shifts which arm is optimal per round. Specialist arms win
//! on their favored configs, so the bandit explores more arms instead of
//! converging to a single generalist.
//!
//! ```sh
//! cargo test --features "problem_mutator" --test problem_mutator_goat -- --nocapture
//! ```

#![cfg(feature = "problem_mutator")]

use katgpt_core::GameConfig;
use katgpt_rs::pruners::problem_mutator::{BomberConfigMutator, EvolutionArena};
use katgpt_rs::types::Rng;

const N_ARMS: usize = 4;
const N_ROUNDS: usize = 100;
/// Number of independent trials to average over (reduces RNG variance).
const N_TRIALS: usize = 10;
/// Threshold for counting an arm as "active" (>10% selection rate).
const ACTIVE_THRESHOLD: f32 = 0.10;

/// Arm profiles: (survival_affinity, kill_affinity, time_pressure_affinity, opponent_affinity)
///
/// Arm 0: generalist — mild bonus on all dimensions
/// Arm 1: survival specialist — strong on survival-weighted configs, weak on kill-weighted
/// Arm 2: kill specialist — strong on kill-weighted configs, weak on survival-weighted
/// Arm 3: time_pressure specialist — strong when max_steps is tight, weak otherwise
const ARM_PROFILES: [(f32, f32, f32, f32); N_ARMS] = [
    (0.05, 0.05, 0.05, 0.05),  // generalist
    (0.30, -0.20, 0.10, 0.00), // survival specialist
    (-0.20, 0.30, 0.00, 0.10), // kill specialist
    (0.00, 0.00, 0.35, 0.05),  // time_pressure specialist
];

/// Compute expected reward for an arm given a game config.
///
/// The reward depends on how the arm's specialization aligns with config parameters:
/// - `survival_weight`: high → favors arm 1 (survival specialist)
/// - `kill_weight`: high → favors arm 2 (kill specialist)
/// - `max_steps` tight (< default): favors arm 3 (time_pressure specialist)
/// - Arm 0 (generalist) gets mild bonus everywhere but never dominates
fn arm_reward(arm: usize, config: &GameConfig, rng: &mut Rng) -> f32 {
    let (survival_aff, kill_aff, time_aff, opp_aff) = ARM_PROFILES[arm];

    let mut reward = 0.5; // baseline

    // Survival-weight dimension: deviation from 0.5 signals survival-heavy config
    let survival_signal = (config.survival_weight - 0.5) * 2.0; // [-1, 1] range
    reward += survival_signal * survival_aff;

    // Kill-weight dimension: deviation from 0.5 signals kill-heavy config
    let kill_signal = (config.kill_weight - 0.5) * 2.0; // [-1, 1] range
    reward += kill_signal * kill_aff;

    // Time pressure dimension: fewer steps = more pressure
    // Default max_steps = 200, so (200 - max_steps) / 200 ∈ [0, 1]
    let time_signal = (200.0 - config.max_steps as f32) / 200.0;
    reward += time_signal * time_aff;

    // Opponent dimension: more opponents = stronger signal
    let opp_signal = (config.opponent_count as f32 - 1.0).max(0.0) / 3.0; // [0, ~1]
    reward += opp_signal * opp_aff;

    // Add noise so the bandit must learn from repeated sampling
    let noise = (rng.uniform() - 0.5) * 0.15;
    (reward + noise).clamp(0.0, 1.0)
}

/// Thompson Sampling arm selection using Beta posterior.
///
/// α = Q·n + 1, β = (1-Q)·n + 1 (Laplace smoothing).
/// Unvisited arms get uniform sample.
fn select_arm_thompson(visits: &[u32], q_values: &[f32], rng: &mut Rng) -> usize {
    let mut best_arm = 0;
    let mut best_sample = f32::NEG_INFINITY;

    for i in 0..visits.len() {
        let sample = if visits[i] == 0 {
            rng.uniform()
        } else {
            let n = visits[i] as f32;
            let q = q_values[i].clamp(0.0, 1.0);
            let alpha = q * n + 1.0;
            let beta = (1.0 - q) * n + 1.0;
            sample_beta(alpha, beta, rng)
        };

        if sample > best_sample {
            best_sample = sample;
            best_arm = i;
        }
    }

    best_arm
}

/// Beta distribution sampling via Johnk's algorithm (matches bandit.rs).
fn sample_beta(alpha: f32, beta: f32, rng: &mut Rng) -> f32 {
    if alpha <= 1.0 && beta <= 1.0 {
        for _ in 0..100 {
            let u = rng.uniform().powf(1.0 / alpha);
            let v = rng.uniform().powf(1.0 / beta);
            if u + v <= 1.0 {
                return if u + v > 0.0 { u / (u + v) } else { 0.5 };
            }
        }
        return 0.5;
    }

    let mean = alpha / (alpha + beta);
    let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
    let std_dev = variance.sqrt().max(1e-6);
    let u1 = rng.uniform().max(1e-10);
    let u2 = rng.uniform();
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    (mean + std_dev * z).clamp(0.0, 1.0)
}

/// Count arms with selection rate above threshold.
fn count_active_arms(visits: &[u32], total_pulls: u32, threshold: f32) -> usize {
    if total_pulls == 0 {
        return 0;
    }
    let total = total_pulls as f32;
    visits
        .iter()
        .filter(|&&v| v as f32 / total > threshold)
        .count()
}

/// Run one bandit session with fixed config.
fn run_fixed_config(seed: u64) -> usize {
    let mut rng = Rng::new(seed);
    let config = GameConfig::default();
    let mut visits = vec![0u32; N_ARMS];
    let mut q_values = vec![0.0f32; N_ARMS];

    for _ in 0..N_ROUNDS {
        let arm = select_arm_thompson(&visits, &q_values, &mut rng);
        let reward = arm_reward(arm, &config, &mut rng);

        visits[arm] += 1;
        let n = visits[arm] as f32;
        q_values[arm] += (reward - q_values[arm]) / n;
    }

    let total: u32 = visits.iter().sum();
    count_active_arms(&visits, total, ACTIVE_THRESHOLD)
}

/// Run one bandit session with mutated configs from EvolutionArena.
fn run_mutated_config(seed: u64) -> usize {
    let mut rng = Rng::new(seed);
    let mut arena = EvolutionArena::new(
        GameConfig::default(),
        Box::new(BomberConfigMutator),
        N_ROUNDS as u32,
    );
    let mut visits = vec![0u32; N_ARMS];
    let mut q_values = vec![0.0f32; N_ARMS];

    for _ in 0..N_ROUNDS {
        let config = arena.next_config();
        let arm = select_arm_thompson(&visits, &q_values, &mut rng);
        let reward = arm_reward(arm, &config, &mut rng);

        visits[arm] += 1;
        let n = visits[arm] as f32;
        q_values[arm] += (reward - q_values[arm]) / n;
    }

    let total: u32 = visits.iter().sum();
    count_active_arms(&visits, total, ACTIVE_THRESHOLD)
}

#[test]
fn test_problem_mutator_goat_diversity() {
    let mut fixed_active_sum = 0usize;
    let mut mutated_active_sum = 0usize;
    let mut fixed_details = Vec::new();
    let mut mutated_details = Vec::new();

    for trial in 0..N_TRIALS {
        let seed = 42 + trial as u64;
        let fixed = run_fixed_config(seed);
        let mutated = run_mutated_config(seed);
        fixed_active_sum += fixed;
        mutated_active_sum += mutated;
        fixed_details.push(fixed);
        mutated_details.push(mutated);
    }

    let avg_fixed = fixed_active_sum as f64 / N_TRIALS as f64;
    let avg_mutated = mutated_active_sum as f64 / N_TRIALS as f64;

    // ── Diagnostics ────────────────────────────────────────────
    println!("\n{}", "═".repeat(70));
    println!("Plan 191 T2.6 — GOAT Proof: Mutated Config Arm Diversity");
    println!("   {N_TRIALS} trials × {N_ROUNDS} rounds × {N_ARMS} arms (Thompson Sampling)");
    println!("{}", "═".repeat(70));

    println!("\n── Fixed Config Path ────────────────────────────────────────");
    println!("   Active arms per trial: {fixed_details:?}");
    println!("   Average active arms:   {avg_fixed:.2}");

    println!("\n── Mutated Config Path ──────────────────────────────────────");
    println!("   Active arms per trial: {mutated_details:?}");
    println!("   Average active arms:   {avg_mutated:.2}");

    let ratio = if avg_fixed > 0.0 {
        avg_mutated / avg_fixed
    } else {
        f64::INFINITY
    };
    println!("\n   Diversity ratio: {ratio:.2}× (target ≥ 1.5×)");
    println!("{}", "═".repeat(70));

    // ── Assertion ──────────────────────────────────────────────
    assert!(
        avg_mutated >= avg_fixed * 1.5,
        "GOAT FAIL: mutation should produce ≥1.5× arm diversity. \
         mutated={avg_mutated:.2}, fixed={avg_fixed:.2}, ratio={ratio:.2}×"
    );

    println!(
        "\n✅ GOAT Proof PASSED: mutated configs produce ≥1.5× arm diversity vs fixed config."
    );
}

#[test]
fn test_problem_mutator_config_diversity() {
    let mut arena = EvolutionArena::new(
        GameConfig::default(),
        Box::new(BomberConfigMutator),
        N_ROUNDS as u32,
    );

    let configs: Vec<GameConfig> = (0..N_ROUNDS).map(|_| arena.next_config()).collect();

    // Count distinct configs by key fields
    let distinct: std::collections::HashSet<_> = configs
        .iter()
        .map(|c| {
            (
                c.grid_size,
                c.opponent_count,
                c.max_steps,
                c.survival_weight.to_bits(),
            )
        })
        .collect();

    // ── Diagnostics ────────────────────────────────────────────
    println!("\n{}", "═".repeat(70));
    println!("Plan 191 T2.6 — Config Diversity Audit");
    println!(
        "   {N_ROUNDS} configs generated, {} distinct",
        distinct.len()
    );
    println!("{}", "═".repeat(70));

    // Group by config for visibility
    let mut grid_sizes: Vec<u32> = configs.iter().map(|c| c.grid_size).collect();
    grid_sizes.sort();
    grid_sizes.dedup();
    println!("   Grid sizes seen: {grid_sizes:?}");

    let mut max_steps: Vec<u32> = configs.iter().map(|c| c.max_steps).collect();
    max_steps.sort();
    max_steps.dedup();
    println!("   Max steps seen: {max_steps:?}");

    let mut survival_weights: Vec<String> = configs
        .iter()
        .map(|c| format!("{:.2}", c.survival_weight))
        .collect();
    survival_weights.sort();
    survival_weights.dedup();
    println!("   Survival weights seen: {survival_weights:?}");

    // ── Assertions ─────────────────────────────────────────────

    // 1. Multiple distinct configs (not just the same one repeated)
    assert!(
        distinct.len() >= 3,
        "EvolutionArena should produce ≥3 distinct configs over {N_ROUNDS} rounds, got {}",
        distinct.len()
    );

    // 2. Config variation is not trivial — at least 2 different max_steps
    assert!(
        max_steps.len() >= 2,
        "Should see ≥2 different max_steps, got {}: {max_steps:?}",
        max_steps.len()
    );

    // 3. Config variation is not trivial — at least 2 different survival_weights
    assert!(
        survival_weights.len() >= 2,
        "Should see ≥2 different survival_weights, got {}: {survival_weights:?}",
        survival_weights.len()
    );

    println!("\n✅ Config diversity audit PASSED: EvolutionArena produces diverse configs.");
}

// TL;DR: GOAT proof — EvolutionArena with BomberConfigMutator produces ≥1.5× arm diversity
// vs fixed config in 4-arm bandit simulation. Config mutation shifts optimal arm per round,
// forcing the bandit to explore specialist arms instead of converging to a single generalist.
// Second test directly validates EvolutionArena produces diverse configs (grid, steps, weights).
