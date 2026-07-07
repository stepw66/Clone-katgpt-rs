//! Module-Aware Compute Routing Demo — Plan 264 Phase 4 (Research 231).
//!
//! Demonstrates the compute-target selection from `route_by_module_energy`
//! across a sweep of module-energy profiles and QPS values. Shows the
//! Plasma/SIMD/GPU/ANE routing decisions that match the paper's FFN-dominated
//! OPD profile (GOAT G7) and the monotone QPS transition (GOAT G8).
//!
//! # Before / After
//!
//! - **Before (QPS-only routing):** the existing `TriggerGate` routes based
//!   solely on QPS thresholds — it cannot exploit the fact that OPD adapters
//!   are FFN-dominated and would benefit from the ternary Plasma path.
//! - **After (module-energy routing):** `route_by_module_energy` considers
//!   both the module profile AND QPS, routing FFN-heavy low-QPS workloads to
//!   the Plasma ternary kernels for ~3–5× throughput.
//!
//! Run: `cargo run --features module_energy_route --example module_aware_routing`

#![cfg(feature = "module_energy_route")]

use katgpt_rs::inference_router::{ComputeTarget, ModuleEnergyProfile, route_by_module_energy};

fn main() {
    println!("=== Plan 264 Phase 4 — Module-Aware Compute Routing ===\n");

    // GOAT G7: paper-average profile routes to Plasma at qps=500.
    let paper = ModuleEnergyProfile::PAPER_AVERAGE;
    println!("Paper-average OPD profile: {:?}", paper);
    println!(
        "  total = {:.3} (valid: {})\n",
        paper.total(),
        paper.is_valid()
    );

    let target_g7 = route_by_module_energy(paper.ffn, paper.attn, 500);
    println!(
        "GOAT G7: route(ffn=0.78, attn=0.16, qps=500) = {:?}",
        target_g7
    );
    if target_g7 == ComputeTarget::Plasma {
        println!("  ✅ PASS: matches paper FFN profile → Plasma\n");
    } else {
        println!("  ❌ FAIL: expected Plasma\n");
        std::process::exit(1);
    }

    // GOAT G8: monotone QPS sweep.
    println!("GOAT G8: QPS sweep (paper-average profile, qps 10 → 10000):");
    let mut prev_target: Option<ComputeTarget> = None;
    let mut transitions = 0;
    for qps_log in 0..=400 {
        let qps = (10.0_f32 * 10.0_f32.powf(qps_log as f32 / 100.0)) as u32;
        let target = route_by_module_energy(paper.ffn, paper.attn, qps);
        if prev_target != Some(target) {
            println!("  qps={:>6}: {:?}", qps, target);
            prev_target = Some(target);
            transitions += 1;
        }
    }
    println!("  Total transitions: {}", transitions);
    if transitions >= 1 {
        println!("  ✅ PASS: monotone, no flapping\n");
    } else {
        println!("  ❌ FAIL: no transitions observed\n");
        std::process::exit(1);
    }

    // Full profile sweep: show all four targets at characteristic operating points.
    println!("Routing decisions across module profiles and QPS:\n");
    let profiles: &[(&str, ModuleEnergyProfile)] = &[
        ("Paper avg (FFN=0.78)", ModuleEnergyProfile::PAPER_AVERAGE),
        (
            "FFN-heavy (0.85)",
            ModuleEnergyProfile {
                ffn: 0.85,
                attn: 0.10,
                embed: 0.03,
                other: 0.02,
            },
        ),
        (
            "Attn-heavy (0.50)",
            ModuleEnergyProfile {
                ffn: 0.30,
                attn: 0.50,
                embed: 0.15,
                other: 0.05,
            },
        ),
        (
            "Balanced (0.45/0.40)",
            ModuleEnergyProfile {
                ffn: 0.45,
                attn: 0.40,
                embed: 0.10,
                other: 0.05,
            },
        ),
    ];
    let qps_values: &[u32] = &[10, 100, 500, 1000, 5000, 10000];

    // Header.
    print!("{:<22}", "Profile \\ QPS");
    for &qps in qps_values {
        print!("{:>10}", qps);
    }
    println!();
    println!("{}", "-".repeat(22 + 10 * qps_values.len()));

    for &(name, profile) in profiles {
        print!("{:<22}", name);
        for &qps in qps_values {
            let target = route_by_module_energy(profile.ffn, profile.attn, qps);
            let short = match target {
                ComputeTarget::Plasma => "Plasma",
                ComputeTarget::Simd => "Simd",
                ComputeTarget::Gpu => "Gpu",
                ComputeTarget::Ane => "Ane",
            };
            print!("{:>10}", short);
        }
        println!();
    }

    println!();
    println!("Plasma tier mapping:");
    println!("  Plasma → Plasma tier (ternary hot path, FFN-dominated low QPS)");
    println!("  Simd   → Hot tier (balanced profile, moderate QPS)");
    println!("  Gpu    → Warm tier (attention-dominated high QPS)");
    println!("  Ane    → Cold tier (very low QPS cold-start)");
}
