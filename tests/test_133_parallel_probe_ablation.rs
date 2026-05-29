#![cfg(feature = "parallel_probe")]

use katgpt_rs::speculative::parallel_probe::{
    ParallelProbeConfig, ParallelProbeController, ProbeDecision,
};

// ── Config builders ──────────────────────────────────────────

fn full_config() -> ParallelProbeConfig {
    ParallelProbeConfig {
        probe_interval: 3,
        stability_patience: 2,
        prune_patience: 3,
        warmup_steps: 3,
        min_active_branches: 2,
        prune_vote_ratio: 0.5,
    }
}

fn no_prune_config() -> ParallelProbeConfig {
    let mut c = full_config();
    c.prune_patience = usize::MAX;
    c
}

fn no_consensus_config() -> ParallelProbeConfig {
    let mut c = full_config();
    c.stability_patience = usize::MAX;
    c
}

fn no_warmup_config() -> ParallelProbeConfig {
    let mut c = full_config();
    c.warmup_steps = 0;
    c
}

// ── Simulation data ──────────────────────────────────────────

/// Simulate a diverging-then-converging answer pattern across 4 branches.
/// Steps 0–4: branches disagree. Step 5+: all agree on "final_answer".
fn simulate_convergence(step: usize) -> Vec<Option<String>> {
    const N: usize = 4;
    if step < 5 {
        // Diverging phase: each branch gives a different modulo answer
        vec![
            Some(format!("ans_{}", step % 3)),
            Some(format!("ans_{}", (step + 1) % 3)),
            Some(format!("ans_{}", (step + 2) % 3)),
            Some(format!("ans_{}", step % 3)),
        ]
    } else {
        // Converged phase: all agree
        vec![Some("final_answer".to_string()); N]
    }
}

// ── Ablation runner ──────────────────────────────────────────

struct AblationResult {
    name: &'static str,
    steps_to_stop: Option<usize>,
    active_at_end: usize,
    reached_consensus: bool,
}

fn run_ablation(name: &'static str, config: ParallelProbeConfig) -> AblationResult {
    let mut ctrl = ParallelProbeController::new(4, config);
    let mut steps = 0usize;
    let mut reached_consensus = false;

    for step in 0..20 {
        let answers = simulate_convergence(step);
        let decision = ctrl.probe(&answers);
        steps = step + 1;

        match decision {
            ProbeDecision::Stop { .. } | ProbeDecision::StopAndPrune { .. } => {
                reached_consensus = true;
                break;
            }
            _ => {}
        }
    }

    AblationResult {
        name,
        steps_to_stop: if reached_consensus { Some(steps) } else { None },
        active_at_end: ctrl.active_count(),
        reached_consensus,
    }
}

// ── Test ─────────────────────────────────────────────────────

#[test]
fn ablation_parallel_probe_components() {
    let configs: Vec<(&str, ParallelProbeConfig)> = vec![
        ("Full System", full_config()),
        ("No Pruning", no_prune_config()),
        ("No Consensus", no_consensus_config()),
        ("No Warmup", no_warmup_config()),
    ];

    println!();
    println!("{}", "=".repeat(60));
    println!("Parallel-Probe Ablation Study (Plan 133 T4)");
    println!("{}", "=".repeat(60));
    println!(
        "{:<15} {:>12} {:>10} {:>12}",
        "Config", "Steps", "Active", "Consensus"
    );
    println!("{:-<15} {:->12} {:->10} {:->12}", "", "", "", "");

    let mut results = Vec::new();
    for (name, config) in configs {
        let r = run_ablation(name, config);
        let steps_str = r
            .steps_to_stop
            .map(|s| s.to_string())
            .unwrap_or_else(|| "> 20".to_string());
        println!(
            "{:<15} {:>12} {:>10} {:>12}",
            r.name, steps_str, r.active_at_end, r.reached_consensus
        );
        results.push(r);
    }

    println!("{}", "=".repeat(60));

    // Assertions

    let full = &results[0];
    let no_prune = &results[1];
    let no_cons = &results[2];
    let no_warmup = &results[3];

    // 1. Full system must reach consensus
    assert!(full.reached_consensus, "Full system should reach consensus");

    // 2. No-consensus (stability_patience = MAX) should NOT early-stop
    assert!(
        !no_cons.reached_consensus,
        "No-consensus config should not stop early"
    );

    // 3. Full system should prune at least as aggressively as no-prune variant
    assert!(
        full.active_at_end <= no_prune.active_at_end,
        "Full system should prune at least as much as no-prune ({} <= {})",
        full.active_at_end,
        no_prune.active_at_end
    );

    // 4. No-warmup should prune at least as aggressively as full
    //    (pruning starts from step 0, so divergent branches get cut sooner)
    assert!(
        no_warmup.active_at_end <= full.active_at_end,
        "No-warmup should prune at least as much as full system ({} <= {})",
        no_warmup.active_at_end,
        full.active_at_end
    );

    // 5. No-warmup still reaches consensus because answers converge by step 5
    assert!(
        no_warmup.reached_consensus,
        "No-warmup should still reach consensus"
    );

    println!("\nAblation GOAT proof PASS ✅");
}
