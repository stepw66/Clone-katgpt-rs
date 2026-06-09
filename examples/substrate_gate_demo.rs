//! SubstrateGate demo — capability-routed inference (Plan 216).
//!
//! Demonstrates:
//! 1. Creating substrate masks for different capabilities
//! 2. Router selecting masks based on input context
//! 3. Dual sparsity: ReLU ∩ substrate mask intersection
//! 4. Branch scoring with recovery

#![cfg(feature = "substrate_gate")]

use katgpt_rs::pruners::{SubstrateMask, substrate_branch_score};

fn main() {
    println!("=== SubstrateGate Demo (Plan 216) ===\n");

    // 1. Create masks for different capabilities
    let mut python_mask = SubstrateMask::new(
        4,    // layers
        1024, // mlp_hidden
        "python_stdlib".to_string(),
        "demo_model".to_string(),
    );
    python_mask.set(0, 42);
    python_mask.set(0, 100);
    python_mask.set(1, 200);
    python_mask.set(2, 500);
    python_mask.set_recovery_score(0.85);

    let mut async_mask = SubstrateMask::new(
        4,
        1024,
        "async_patterns".to_string(),
        "demo_model".to_string(),
    );
    async_mask.set(0, 42);
    async_mask.set(0, 150);
    async_mask.set(1, 300);
    async_mask.set_recovery_score(0.72);

    println!(
        "Python mask: {} active channels, recovery={:.2}",
        python_mask.active_count(),
        python_mask.recovery_score(),
    );
    println!(
        "Async mask: {} active channels, recovery={:.2}",
        async_mask.active_count(),
        async_mask.recovery_score(),
    );

    // 2. Mask intersection (dual sparsity)
    let intersection = python_mask.intersect(&async_mask);
    println!(
        "\nIntersection: {} active channels",
        intersection.active_count()
    );

    // 3. Branch scoring
    let score_python = substrate_branch_score(-1.2, 0.85, 1.0);
    let score_async = substrate_branch_score(-1.5, 0.72, 1.0);
    println!("\nBranch scores:");
    println!("  Python stdlib: {:.4}", score_python);
    println!("  Async patterns: {:.4}", score_async);
    println!(
        "  Best: {}",
        if score_python > score_async {
            "Python"
        } else {
            "Async"
        }
    );

    println!("\n=== Demo Complete ===");
}
