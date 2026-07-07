//! Adaptive Rank vs Fixed Rank Demo — Plan 264 Phase 3 (Research 231).
//!
//! Demonstrates the LoRA rank savings from spectral-concentration adaptive
//! rank selection. On a synthetic workload of 100 adapters with OPD-shaped
//! eigenvalue spectra, the adaptive rank (mapped from spectral concentration
//! via a sigmoid) uses ~50% less total rank than a fixed max-rank policy,
//! while still allocating enough rank to concentrated adapters.
//!
//! # Before / After
//!
//! - **Before (fixed rank):** every adapter gets `max_rank = 64`.
//! - **After (adaptive rank):** each adapter's rank is derived from its
//!   spectral concentration via `adaptive_rank(c, 4, 64)`. Concentrated
//!   spectra (high c) still get up to 64; diffuse spectra (low c) get as
//!   few as 4.
//!
//! Run: `cargo run --features spectral_rank --example adaptive_rank_vs_fixed`

#![cfg(feature = "spectral_rank")]

use katgpt_rs::spectral_concentration::{
    adaptive_rank, cot_budget_from_concentration, spectral_concentration,
};

const N_ADAPTERS: usize = 100;
const SPECTRUM_LEN: usize = 128;
const RANK_K: usize = 16;
const MIN_RANK: usize = 4;
const MAX_RANK: usize = 64;

fn main() {
    println!("=== Plan 264 Phase 3 — Adaptive Rank vs Fixed Rank ===\n");
    println!(
        "Synthetic workload: {} adapters, spectrum length {}",
        N_ADAPTERS, SPECTRUM_LEN
    );
    println!(
        "Rank range: [{}, {}], concentration measured at k={}\n",
        MIN_RANK, MAX_RANK, RANK_K
    );

    let mut total_fixed = 0_usize;
    let mut total_adaptive = 0_usize;
    let mut total_cot_fixed = 0_usize;
    let mut total_cot_adaptive = 0_usize;
    let mut concentrations: Vec<f32> = Vec::with_capacity(N_ADAPTERS);

    for i in 0..N_ADAPTERS {
        // Generate a synthetic OPD-shaped spectrum with varying alpha.
        // Alpha in [0.4, 0.6] produces paper-shaped concentration bands.
        let alpha = 0.40 + 0.20 * ((i as f32) / (N_ADAPTERS as f32));
        let spectrum = synthetic_spectrum(SPECTRUM_LEN, alpha, i as u64);
        let c = spectral_concentration(&spectrum, RANK_K);
        concentrations.push(c);

        // Fixed rank: everyone gets MAX_RANK.
        total_fixed += MAX_RANK;

        // Adaptive rank: sigmoid-mapped from concentration.
        let r = adaptive_rank(c, MIN_RANK, MAX_RANK);
        total_adaptive += r;

        // Fixed CoT budget: base=8, everyone gets base + 16 = 24.
        total_cot_fixed += 8 + 16;

        // Adaptive CoT: concentrated spectra earn more chain-of-thought.
        let cot = cot_budget_from_concentration(c, 8, 16);
        total_cot_adaptive += cot;
    }

    let avg_fixed = total_fixed as f32 / N_ADAPTERS as f32;
    let avg_adaptive = total_adaptive as f32 / N_ADAPTERS as f32;
    let rank_reduction = 1.0 - avg_adaptive / avg_fixed;

    let avg_cot_fixed = total_cot_fixed as f32 / N_ADAPTERS as f32;
    let avg_cot_adaptive = total_cot_adaptive as f32 / N_ADAPTERS as f32;

    // Concentration stats.
    let c_min = concentrations.iter().cloned().fold(f32::INFINITY, f32::min);
    let c_max = concentrations
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let c_mean = concentrations.iter().sum::<f32>() / N_ADAPTERS as f32;

    println!("Spectral concentration stats:");
    println!("  min:    {:.4}", c_min);
    println!("  max:    {:.4}", c_max);
    println!("  mean:   {:.4}", c_mean);
    println!();

    println!("Rank allocation:");
    println!("  Fixed:    avg = {:.1} (total {})", avg_fixed, total_fixed);
    println!(
        "  Adaptive: avg = {:.1} (total {})",
        avg_adaptive, total_adaptive
    );
    println!(
        "  Reduction: {:.1}% (savings: {} rank units)",
        rank_reduction * 100.0,
        total_fixed - total_adaptive
    );
    println!();

    println!("CoT budget allocation:");
    println!(
        "  Fixed:    avg = {:.1} (total {})",
        avg_cot_fixed, total_cot_fixed
    );
    println!(
        "  Adaptive: avg = {:.1} (total {})",
        avg_cot_adaptive, total_cot_adaptive
    );
    println!();

    if rank_reduction >= 0.30 {
        println!("✅ GOAT G6 PASS: adaptive rank reduces avg rank by ≥30%");
    } else {
        println!(
            "❌ GOAT G6 FAIL: rank reduction {:.1}% < 30%",
            rank_reduction * 100.0
        );
        std::process::exit(1);
    }
}

/// Synthetic OPD-shaped eigenvalue spectrum: power-law decay with jitter.
fn synthetic_spectrum(n: usize, alpha: f32, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    let mut eigs = Vec::with_capacity(n);
    for i in 0..n {
        // xorshift64 step.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let jitter = 0.02 * ((state >> 11) as f32 / (1u64 << 52) as f32);
        let base = 1.0 / (i as f32 + 1.0).powf(alpha);
        eigs.push((base + jitter).max(1e-6));
    }
    eigs
}
