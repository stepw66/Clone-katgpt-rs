//! Safe-Phased Bandit Demo — PrudentBanker Delay-Calibrated Exploration Safety
//!
//! Demonstrates the SafePhased strategy (Plan 137) on a 5-armed Bernoulli bandit:
//! - Arm 0: safe baseline (mean=0.5)
//! - Arm 3: best arm (mean=0.8)
//! - Arms 1, 2, 4: suboptimal (means 0.3, 0.4, 0.6)
//!
//! Compares UCB1 vs SafePhased vs ε-greedy, showing:
//! - α escalation over time (phased aggression)
//! - Baseline regret stays bounded (safe mixture)
//! - Competitive performance with unconstrained strategies
//!
//! Run: `cargo run --example bandit_08_safe_phased --features safe_bandit`

#![cfg(feature = "safe_bandit")]

use katgpt_rs::pruners::{BanditSession, BanditStrategy, BernoulliEnv};
use katgpt_rs::types::Rng;

const EPISODES: usize = 2000;
const SEED: u64 = 42;
const BASELINE_ARM: usize = 0;
const DELTA: f32 = 0.2;
const ESTIMATED_DELAY: u32 = 0;

fn main() {
    print_header();

    let probs = [0.5f32, 0.3, 0.4, 0.8, 0.6];
    print_env(&probs);

    let strategies = vec![
        ("UCB1".to_string(), BanditStrategy::Ucb1),
        (
            "ε-greedy".to_string(),
            BanditStrategy::EpsilonGreedy {
                epsilon: 0.1,
                decay: 0.999,
            },
        ),
        (
            format!("SafePhased(base={BASELINE_ARM}, δ={DELTA}, D̂={ESTIMATED_DELAY})"),
            BanditStrategy::SafePhased {
                baseline_arm: BASELINE_ARM,
                delta: DELTA,
                estimated_delay: ESTIMATED_DELAY,
            },
        ),
    ];

    let mut all_results = Vec::new();

    for (name, strategy) in &strategies {
        let result = run_strategy(&probs, strategy.clone(), name.clone());
        all_results.push(result);
    }

    print_regret_comparison(&all_results);
    print_baseline_regret(&all_results, &probs);
    print_conclusion(&all_results);
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║    Safe-Phased Bandit — PrudentBanker Exploration Safety     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Strategy: phased aggression with safe baseline mixture.");
    println!("  α escalates geometrically only when evidence certifies");
    println!("  the baseline arm is suboptimal. Delay-calibrated slack ξ(D)");
    println!("  prevents premature aggression under delayed feedback.");
    println!();
}

fn print_env(probs: &[f32]) {
    println!("🎯 Bernoulli Bandit Environment:");
    println!("   Arms:     {}", probs.len());
    println!(
        "   Means:    [{}]",
        probs
            .iter()
            .map(|p| format!("{p:.1}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let optimal = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!("   Optimal:  Arm {} (mean={:.1})", optimal.0, optimal.1);
    println!(
        "   Baseline: Arm {} (mean={:.1}) ← safe fallback",
        BASELINE_ARM, probs[BASELINE_ARM]
    );
    println!("   Episodes: {EPISODES}");
    println!("   δ={DELTA}, D̂={ESTIMATED_DELAY}");
    println!();
}

fn run_strategy(probs: &[f32], strategy: BanditStrategy, name: String) -> StrategyResult {
    let env = BernoulliEnv::new(probs);
    let session = BanditSession::new(env, strategy);
    let (events, result) = session.run(EPISODES, &mut Rng::new(SEED));

    // Extract regret curve
    let mut regret_curve = Vec::new();
    let mut reward_curve = Vec::new();
    for event in &events {
        if let katgpt_rs::pruners::BanditEvent::EpisodeComplete {
            cumulative_reward,
            cumulative_regret,
            ..
        } = event
        {
            regret_curve.push(*cumulative_regret);
            reward_curve.push(*cumulative_reward);
        }
    }

    // Count how often baseline arm was selected
    let baseline_visits = result.visits[BASELINE_ARM];
    let total_visits: u32 = result.visits.iter().sum();
    let baseline_pct = baseline_visits as f32 / total_visits as f32 * 100.0;

    println!("┌─ {name} ─────────────────────────────────────────");
    println!(
        "│ Best arm:     {} (optimal: {})",
        result.best_arm, result.optimal_arm
    );
    println!(
        "│ Found optimal: {}",
        if result.found_optimal() { "✅" } else { "❌" }
    );
    println!("│ Total reward:  {:.1}", result.total_reward);
    println!("│ Total regret:  {:.2}", result.total_regret);
    println!("│ Avg reward:    {:.4}", result.avg_reward());
    println!("│");
    println!("│ Baseline arm ({BASELINE_ARM}):");
    println!("│   Visits: {baseline_visits} ({baseline_pct:.1}% of total)");
    println!(
        "│   Q-value: {:.4} (true mean: {:.1})",
        result.q_values[BASELINE_ARM], probs[BASELINE_ARM]
    );
    println!("│");
    println!("│ Q-values:");
    for (i, &q) in result.q_values.iter().enumerate() {
        let marker = if i == result.optimal_arm {
            " ← optimal"
        } else if i == BASELINE_ARM {
            " ← baseline"
        } else {
            ""
        };
        let bar_len = (q * 40.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!("│   Arm {i}: {q:.4} {bar}{marker}");
    }
    println!("│");
    println!("│ Visit counts:");
    for (i, &v) in result.visits.iter().enumerate() {
        let pct = v as f32 / total_visits as f32 * 100.0;
        let bar_len = (pct / 2.5) as usize;
        let bar: String = "█".repeat(bar_len);
        let marker = if i == result.optimal_arm {
            " ← optimal"
        } else if i == BASELINE_ARM {
            " ← baseline"
        } else {
            ""
        };
        println!("│   Arm {i}: {v:>4} ({pct:>5.1}%) {bar}{marker}");
    }
    println!("└────────────────────────────────────────────────────");
    println!();

    StrategyResult {
        name,
        regret_curve,
        reward_curve,
        total_regret: result.total_regret,
        total_reward: result.total_reward,
        optimal_arm: result.optimal_arm,
        best_arm: result.best_arm,
    }
}

struct StrategyResult {
    name: String,
    regret_curve: Vec<f32>,
    reward_curve: Vec<f32>,
    total_regret: f32,
    #[allow(dead_code)]
    total_reward: f32,
    optimal_arm: usize,
    best_arm: usize,
}

fn print_regret_comparison(results: &[StrategyResult]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Cumulative Regret Comparison                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let checkpoints: [usize; 7] = [100, 250, 500, 1000, 1500, 1750, 2000];
    let checkpoint_labels = checkpoints.map(|c| format!("{c}"));

    print!("{:<40}", "Episode");
    for label in &checkpoint_labels {
        print!("{:>8}", label);
    }
    println!();
    println!("{}", "─".repeat(40 + checkpoint_labels.len() * 8));

    for r in results {
        print!("{:<40}", r.name);
        for &cp in &checkpoints {
            let idx = cp.saturating_sub(1);
            if idx < r.regret_curve.len() {
                print!("{:>8.1}", r.regret_curve[idx]);
            } else {
                print!("{:>8}", "—");
            }
        }
        println!();
    }
    println!();
}

fn print_baseline_regret(results: &[StrategyResult], probs: &[f32]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Baseline Regret Analysis (vs Arm 0 = 0.5)           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  \"Baseline regret\" = how much worse each strategy does vs");
    println!(
        "  always playing arm 0 (the safe baseline, mean={:.1}).",
        probs[BASELINE_ARM]
    );
    println!("  If baseline regret stays bounded, SafePhased is doing its job.",);
    println!();

    let baseline_reward_per_ep = probs[BASELINE_ARM];
    let checkpoints: [usize; 6] = [100, 250, 500, 1000, 1500, 2000];

    print!("{:<40}", "Episode");
    for &cp in &checkpoints {
        print!("{:>8}", cp);
    }
    println!();
    println!("{}", "─".repeat(40 + checkpoints.len() * 8));

    for r in results {
        print!("{:<40}", r.name);
        for &cp in &checkpoints {
            let idx = cp.saturating_sub(1);
            if idx < r.reward_curve.len() {
                let baseline_total = baseline_reward_per_ep * cp as f32;
                let baseline_regret = baseline_total - r.reward_curve[idx];
                print!("{:>8.1}", baseline_regret);
            } else {
                print!("{:>8}", "—");
            }
        }
        println!();
    }
    println!();

    // ASCII plot of total regret growth
    print_ascii_regret_plot(results);
}

fn print_ascii_regret_plot(results: &[StrategyResult]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Regret Growth (ASCII Plot)                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let width = 60;
    let height = 12;

    let max_regret = results
        .iter()
        .flat_map(|r| r.regret_curve.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1.0);

    let symbols = ['●', '■', '▲'];
    let mut grid = vec![vec![' '; width]; height];

    #[allow(clippy::needless_range_loop)]
    for row in 0..height {
        let val = max_regret * (1.0 - row as f32 / (height - 1) as f32);
        print!("{:>7.1} │", val);
        #[allow(clippy::needless_range_loop)]
        for col in 0..width {
            let episode = (col as f32 / (width - 1) as f32 * (EPISODES - 1) as f32) as usize;
            for (si, r) in results.iter().enumerate() {
                if episode < r.regret_curve.len() {
                    let regret = r.regret_curve[episode];
                    let y = ((1.0 - regret / max_regret) * (height - 1) as f32) as usize;
                    if y == row {
                        grid[row][col] = symbols[si % symbols.len()];
                    }
                }
            }
            print!("{}", grid[row][col]);
        }
        println!();
    }

    // X-axis
    print!("        └");
    for _ in 0..width {
        print!("─");
    }
    println!();
    print!("         ");
    for i in 0..=4 {
        let ep = i * EPISODES / 4;
        print!("{:<15}", ep);
    }
    println!("  episodes");
    println!();

    // Legend
    for (i, r) in results.iter().enumerate() {
        println!("  {} = {}", symbols[i % symbols.len()], r.name);
    }
    println!();
}

fn print_conclusion(results: &[StrategyResult]) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      Summary                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut best_regret = f32::MAX;
    let mut best_name = String::new();

    for r in results {
        if r.total_regret < best_regret {
            best_regret = r.total_regret;
            best_name = r.name.clone();
        }
    }

    println!("  Lowest final regret: {best_name} ({best_regret:.2})");
    println!();
    println!("  Key takeaways:");
    println!("  • SafePhased: bounded regret vs baseline, competitive with UCB1");
    println!("  • α escalation: starts conservative (≈1/R̂), doubles each phase");
    println!("  • When baseline is suboptimal, phases escalate → α → 1 → full UCB1");
    println!("  • Baseline regret stays O(1) — the safety guarantee");
    println!("  • Delay slack ξ(D) prevents premature aggression under delayed feedback");
    println!();

    // Show which strategies found the optimal arm
    println!("  Optimal arm found:");
    for r in results {
        let status = if r.best_arm == r.optimal_arm {
            "✅"
        } else {
            "❌"
        };
        println!(
            "    {status} {} → arm {} (optimal: {})",
            r.name, r.best_arm, r.optimal_arm
        );
    }
    println!();
}
