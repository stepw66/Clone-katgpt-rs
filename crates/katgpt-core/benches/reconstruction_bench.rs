//! Benchmark: OctreeCTC Reconstructive Navigation (Plan 248).
//!
//! Measures per-cycle latency for 3-step reconstruction (scalar vs SIMD).
//! GOAT Gate: <200ns per 3-step reconstruction cycle.
//!
//! Measures:
//!   - Scalar `reconstruct()` — baseline
//!   - SIMD `reconstruct_simd()` — optimized evolve_hla (proven win)
//!   - Per-step breakdown: expand → route → accumulate → evolve_hla
//!   - Full SIMD step (expand_simd + route_simd + evolve_hla_simd)
//!
//! Note: expand_simd/route_simd are scaling-optimized for larger module counts.
//! At 6 modules × 8-dim HLA, scalar expand/route is faster due to SIMD setup
//! overhead. evolve_hla_simd wins at any size.

use katgpt_core::sense::brain::NpcBrain;
use katgpt_core::sense::octree::{KgEmbedding, SenseOctreeBuilder};
use katgpt_core::sense::reconstruction::{ReconstructionConfig, ReconstructionState};
use katgpt_core::types::SenseKind;

const ITERS: usize = 10_000;

fn make_brain_with_6_modules() -> NpcBrain {
    let builder = SenseOctreeBuilder::new(3);
    let kinds = [
        SenseKind::CommonSense,
        SenseKind::FighterSense,
        SenseKind::GameTheorySense,
        SenseKind::SpatialSense,
        SenseKind::SocialSense,
        SenseKind::SkillSense,
    ];
    let modules: Vec<_> = kinds
        .iter()
        .enumerate()
        .map(|(i, &kind)| {
            let emb = KgEmbedding {
                entity_hash: kind as u64,
                relation_hash: kind as u64,
                embedding: [0.5; 8],
                sign: true,
                confidence: 1.0,
            };
            let m = builder.build(kind, &[emb]);
            // Vary confidence per module
            let mut m = m;
            m.confidence = 0.3 + 0.1 * i as f32;
            m.commit();
            m
        })
        .collect();

    let mut brain = NpcBrain::compose(modules);
    brain.hla_state = [0.3, 0.7, 0.1, 0.5, 0.4, 0.2, 0.6, 0.8];
    brain
}

fn bench_reconstruct_scalar(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Warmup
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct(brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct(brain);
        std::hint::black_box(&state);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ITERS as f64
}

fn bench_reconstruct_simd(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Warmup
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_simd(brain);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let _ = state.reconstruct_simd(brain);
        std::hint::black_box(&state);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ITERS as f64
}

fn bench_step_scalar(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Warmup
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla();
        std::hint::black_box(&state);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ITERS as f64
}

fn bench_step_simd(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Warmup
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla_simd();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand(brain);
        let selected = state.route(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla_simd();
        std::hint::black_box(&state);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ITERS as f64
}

/// Full SIMD path: expand_simd + route_simd + accumulate + evolve_hla_simd
fn bench_step_full_simd(brain: &NpcBrain, config: ReconstructionConfig) -> f64 {
    // Warmup
    for _ in 0..100 {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand_simd(brain);
        let selected = state.route_simd(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla_simd();
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERS {
        let mut state = ReconstructionState::with_config(brain.hla_state, config);
        let activations = state.expand_simd(brain);
        let selected = state.route_simd(&activations);
        state.accumulate(&selected, &activations);
        state.evolve_hla_simd();
        std::hint::black_box(&state);
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ITERS as f64
}

fn main() {
    println!("=== Plan 248: OctreeCTC Reconstruction Benchmark ===\n");

    let brain = make_brain_with_6_modules();
    let config = ReconstructionConfig::default(); // 3 steps

    // Report SIMD level
    let level = katgpt_core::simd::simd_level();
    println!("SIMD level: {level:?}");

    println!(
        "Config: max_steps={}, lr={}",
        config.max_steps, config.hla_learning_rate
    );
    println!("Modules: {}", brain.modules.len());
    println!("Iterations: {ITERS}\n");

    // Full cycle benchmarks
    let scalar_ns = bench_reconstruct_scalar(&brain, config);
    let simd_ns = bench_reconstruct_simd(&brain, config);

    println!("--- Full 3-Step Cycle ---");
    println!("Scalar:  {scalar_ns:>8.1} ns/cycle");
    println!("SIMD:    {simd_ns:>8.1} ns/cycle");
    let speedup = scalar_ns / simd_ns;
    println!("Speedup: {speedup:.2}x");

    let goat_pass = simd_ns < 200.0;
    println!(
        "GOAT (<200ns): {}",
        if goat_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // Per-step benchmarks
    println!("\n--- Per-Step Breakdown (expand+route+accumulate+evolve) ---");
    let step_scalar_ns = bench_step_scalar(&brain, config);
    let step_simd_ns = bench_step_simd(&brain, config);
    let step_full_simd_ns = bench_step_full_simd(&brain, config);

    println!("Scalar step:       {step_scalar_ns:>8.1} ns");
    println!("SIMD evolve only:  {step_simd_ns:>8.1} ns");
    println!("SIMD full path:    {step_full_simd_ns:>8.1} ns");
    let step_speedup = step_scalar_ns / step_full_simd_ns;
    println!("Full SIMD speedup: {step_speedup:.2}x");

    // Correctness check: SIMD produces same results as scalar
    let mut state_scalar = ReconstructionState::with_config(brain.hla_state, config);
    let _ = state_scalar.reconstruct(&brain);

    let mut state_simd = ReconstructionState::with_config(brain.hla_state, config);
    let _ = state_simd.reconstruct_simd(&brain);

    let mut max_diff = 0.0f32;
    for i in 0..8 {
        let diff = (state_scalar.hla()[i] - state_simd.hla()[i]).abs();
        max_diff = max_diff.max(diff);
    }
    println!("\n--- Correctness ---");
    println!("Max HLA diff (scalar vs SIMD): {max_diff:.6e}");
    assert!(
        max_diff < 1e-4,
        "SIMD and scalar should produce similar results, diff={max_diff}"
    );
    println!("Numerical equivalence: PASS ✅");

    // Correctness check: full SIMD step vs scalar step
    let mut state_a = ReconstructionState::with_config(brain.hla_state, config);
    let act_scalar = state_a.expand(&brain);
    let sel_scalar = state_a.route(&act_scalar);
    state_a.accumulate(&sel_scalar, &act_scalar);
    state_a.evolve_hla();

    let mut state_b = ReconstructionState::with_config(brain.hla_state, config);
    let act_simd = state_b.expand_simd(&brain);
    let sel_simd = state_b.route_simd(&act_simd);
    state_b.accumulate(&sel_simd, &act_simd);
    state_b.evolve_hla_simd();

    let mut max_step_diff = 0.0f32;
    for i in 0..8 {
        let diff = (state_a.hla()[i] - state_b.hla()[i]).abs();
        max_step_diff = max_step_diff.max(diff);
    }
    println!("Max step diff (scalar vs full SIMD): {max_step_diff:.6e}");
    assert!(
        max_step_diff < 1e-4,
        "Full SIMD step should match scalar, diff={max_step_diff}"
    );
    println!("Step equivalence: PASS ✅");
}
