#![cfg(feature = "dash_attn")]
//! GOAT Gate Scaffold — Entropy-Calibrated Chunk Summary (Issue 044, Research 399)
//!
//! This benchmark demonstrates that the HiLS Prop 3.1 entropy bias `b'_c`
//! changes routing decisions when chunks have varying concentration profiles.
//! It is the synthetic proof that the mechanism has a routing effect, and the
//! scaffold for the riir-train GOAT gate (NIAH-style accuracy at fixed budget).
//!
//! **Why this exists:** The entropy bias is dormant at zero-init (constant
//! across same-size chunks → no ranking change). A meaningful GOAT gate
//! requires riir-train-provided learned `head_cls` (non-uniform softmax →
//! entropy discriminates concentrated vs. spread chunks). This benchmark
//! simulates that scenario with synthetic non-uniform entropy profiles to
//! prove the routing mechanism works.
//!
//! **GOAT gate criteria (when trained weights land):**
//! 1. NIAH-style needle-in-haystack at fixed budget (32K, 128K).
//! 2. Before/after: entropy-blind (`&[]`) vs entropy-aware (`b'_c`), same
//!    trained `head_cls`.
//! 3. Metric: chunk-selection accuracy (top-k contains needle?).
//! 4. Pass: entropy-aware improves accuracy ≥X% at fixed budget, OR same
//!    accuracy at lower budget.
//!
//! Run: `cargo test --features dash_attn --test bench_044_entropy_gate_scaffold -- --nocapture`

use katgpt_rs::dash_attn::{score_blocks_entmax, score_blocks_entmax_with_entropy};
use katgpt_rs::types::DashAttnConfig;

// ── Helpers ───────────────────────────────────────────────────

/// Deterministic pseudo-random vector generator (index-based seed).
fn make_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x =
                ((i.wrapping_mul(2654435761)).wrapping_add(seed.wrapping_mul(40503))) as f32;
            (x * 0.0001).sin() * 0.5 + 0.5
        })
        .collect()
}

/// Two chunks are "routing-different" if their active-index sets differ.
fn active_sets_differ(
    a: &katgpt_rs::dash_attn::routing::RoutingResult,
    b: &katgpt_rs::dash_attn::routing::RoutingResult,
) -> bool {
    let set_a: std::collections::HashSet<usize> = a.active_indices.iter().copied().collect();
    let set_b: std::collections::HashSet<usize> = b.active_indices.iter().copied().collect();
    set_a != set_b
}

// ── Test 1: Non-uniform entropy changes routing at realistic scale ──
//
// This is the core mechanism proof: with 64 chunks (realistic prefill scale)
// and a non-uniform entropy profile (some chunks concentrated, some spread),
// entropy-aware routing selects different chunks than entropy-blind routing.

#[test]
fn bench_entropy_changes_routing_at_scale() {
    let config = DashAttnConfig::default();
    let n_chunks = 64;
    let dim = 32;

    // Generate 64 chunks with deterministic summaries.
    let summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    // Simulate a learned-query entropy profile: alternate between
    // concentrated chunks (low entropy ≈ 0, one token dominates) and spread
    // chunks (high entropy ≈ ln(64) ≈ 4.16, uniform attention).
    let ln_chunk_size = (64.0f32).ln();
    let entropy_profile: Vec<f32> = (0..n_chunks)
        .map(|c| if c % 2 == 0 { 0.1 } else { ln_chunk_size })
        .collect();

    let mut n_changed = 0usize;
    for seed in 0..50 {
        let query = make_vec(dim, 10_000 + seed);

        // Entropy-blind routing (the pre-Issue-044 behavior).
        let blind = score_blocks_entmax(&query, &summaries, &config);
        // Entropy-aware routing (the Issue 044 mechanism).
        let aware =
            score_blocks_entmax_with_entropy(&query, &summaries, &entropy_profile, &config);

        if active_sets_differ(&blind, &aware) {
            n_changed += 1;
        }
    }

    println!(
        "┌──────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ Entropy Gate Scaffold: routing decisions changed by b'_c     │"
    );
    println!(
        "├──────────────────────────────────────────────────────────────┤"
    );
    println!(
        "│ Chunks: {n_chunks}, Queries: 50, Alternating entropy profile   │"
    );
    println!(
        "│ Routing changed: {n_changed}/50 queries                          │"
    );
    println!(
        "└──────────────────────────────────────────────────────────────┘"
    );

    // The entropy bias MUST change routing for at least some queries when
    // the entropy profile is non-uniform. If it never changes, the mechanism
    // is broken.
    assert!(
        n_changed > 0,
        "entropy-aware routing should differ from entropy-blind for at least 1/50 queries \
         with a non-uniform entropy profile"
    );
}

// ── Test 2: Uniform entropy does NOT change ranking ──────────────
//
// The dormant-at-uniform guarantee: when all chunks have the same entropy
// (e.g., zero-init where b'_c = ln(chunk_size) for all), the routing must be
// bit-identical to the entropy-blind path. This is the backward-compat proof.

#[test]
fn bench_uniform_entropy_preserves_ranking() {
    let config = DashAttnConfig::default();
    let n_chunks = 64;
    let dim = 32;

    let summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    // Uniform entropy (simulates zero-init: b'_c = ln(chunk_size) for all).
    let uniform_entropy = vec![(64.0f32).ln(); n_chunks];

    let mut n_changed = 0usize;
    for seed in 0..50 {
        let query = make_vec(dim, 20_000 + seed);
        let blind = score_blocks_entmax(&query, &summaries, &config);
        let aware =
            score_blocks_entmax_with_entropy(&query, &summaries, &uniform_entropy, &config);
        if active_sets_differ(&blind, &aware) {
            n_changed += 1;
        }
    }

    println!(
        "┌──────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ Dormant-at-uniform guarantee: uniform b'_c = no change      │"
    );
    println!(
        "├──────────────────────────────────────────────────────────────┤"
    );
    println!(
        "│ Chunks: {n_chunks}, Queries: 50, Uniform entropy = ln(64)      │"
    );
    println!(
        "│ Routing changed: {n_changed}/50 queries (MUST be 0)             │"
    );
    println!(
        "└──────────────────────────────────────────────────────────────┘"
    );

    // Uniform entropy (constant across chunks) MUST NOT change routing —
    // this is the dormant-at-zero-init guarantee.
    assert_eq!(
        n_changed, 0,
        "uniform entropy (constant across chunks) must not change routing ranking"
    );
}

// ── Test 3: Entropy magnitude effect on support size ─────────────
//
// The HiLS Prop 3.1 prediction: high-entropy chunks (spread attention) should
// be MORE likely to appear in the active support (they represent more
// LogSumExp mass), while low-entropy chunks (concentrated) contribute less.
// This test verifies the directionality: adding positive entropy to a chunk
// increases its probability of being selected.

#[test]
fn bench_entropy_boosts_high_entropy_chunks() {
    let config = DashAttnConfig::default();
    let n_chunks = 32;
    let dim = 16;

    let summaries: Vec<Vec<f32>> = (0..n_chunks).map(|c| make_vec(dim, c)).collect();

    // Give the first half high entropy, second half low entropy.
    let entropy_profile: Vec<f32> = (0..n_chunks)
        .map(|c| if c < n_chunks / 2 { 4.0 } else { 0.0 })
        .collect();

    let mut high_entropy_selected = 0usize;
    let mut low_entropy_selected = 0usize;

    for seed in 0..100 {
        let query = make_vec(dim, 30_000 + seed);
        let result =
            score_blocks_entmax_with_entropy(&query, &summaries, &entropy_profile, &config);
        for &idx in &result.active_indices {
            if idx < n_chunks / 2 {
                high_entropy_selected += 1;
            } else {
                low_entropy_selected += 1;
            }
        }
    }

    println!(
        "┌──────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ Directionality: high-entropy chunks selected more often     │"
    );
    println!(
        "├──────────────────────────────────────────────────────────────┤"
    );
    println!(
        "│ Chunks: {n_chunks} (half high-entropy, half low), Queries: 100  │"
    );
    println!(
        "│ High-entropy selections: {high_entropy_selected:<5}                │"
    );
    println!(
        "│ Low-entropy selections:  {low_entropy_selected:<5}                │"
    );
    println!(
        "└──────────────────────────────────────────────────────────────┘"
    );

    // High-entropy chunks should be selected at least as often as low-entropy
    // chunks (the entropy bias boosts their logit, making them more likely
    // to appear in the entmax support).
    assert!(
        high_entropy_selected >= low_entropy_selected,
        "high-entropy chunks should be selected ≥ low-entropy chunks \
         (got high={high_entropy_selected}, low={low_entropy_selected})"
    );
}
