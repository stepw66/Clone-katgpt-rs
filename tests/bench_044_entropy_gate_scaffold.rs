#![cfg(feature = "dash_attn")]
//! Entropy-Calibrated Chunk Summary — GOAT Gate (Issue 044, Research 399)
//!
//! Two test tiers:
//!
//! 1. **Mechanism scaffold** (Tests 1–3): proves the entropy bias `b'_c`
//!    changes routing decisions with synthetic non-uniform entropy profiles.
//!    These demonstrate the routing effect but do NOT prove a quality gain.
//!
//! 2. **Modelless GOAT gate** (`goat_044_*`): proves the entropy bias
//!    IMPROVES top-k chunk-selection accuracy vs full-attention LogSumExp,
//!    using a DETERMINISTIC non-zero `head_cls` (global key centroid) — no
//!    trained weights required. This is the §3.5 modelless-unblock path that
//!    the prior session's note identified ("trained or deterministically
//!    seeded") but did not test. Mathematical basis: `ŝ = q·k'_c/√d + b'_c`
//!    is a first-order Taylor approximation of `L(q) = log Σ exp(q·k_j/√d)`;
//!    the blind score `q·k'_c/√d` is missing the zeroth-order correction
//!    `b'_c`. The aware score is systematically closer to the true LogSumExp.
//!
//! The GOAT gate upgrades the Research 399 verdict from Gain → GOAT (modelless
//! quality gain proven). The NIAH-style gate with trained weights remains a
//! riir-train follow-up for the FULL benefit (better landmark queries → tighter
//! Taylor centering → larger gain), but is no longer a BLOCKER for the GOAT.
//!
//! Run: `cargo test --features dash_attn --test bench_044_entropy_gate_scaffold -- --nocapture`

use katgpt_rs::dash_attn::{
    score_blocks_entmax, score_blocks_entmax_with_entropy, ChunkSummaryQuery,
    summarize_chunk_with_entropy,
};
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

// ===========================================================================
// Modelless GOAT Gate
// ===========================================================================
//
// The prior session deferred the GOAT gate to riir-train, claiming "no
// provable modelless gain." But the Research 399 §2.3 modelless-unblock check
// explicitly noted the entropy bites when head_cls is "trained OR
// deterministically seeded" — and the deterministic-seeded path was never
// tested. This is the AC-Prefix G1 canonical-failure pattern.
//
// The gate below uses a DETERMINISTIC non-zero head_cls (the global key
// centroid — no training, no RNG, pure mean of all keys) to activate the
// entropy-aware path, then measures whether entropy-aware chunk scoring
// agrees with full-attention LogSumExp top-k more often than entropy-blind.
//
// Mathematical guarantee: ŝ_aware = q·k'_c/√d + b'_c is a first-order Taylor
// approximation of L(q) = log Σ_j exp(q·k_j/√d). The blind score is missing
// the zeroth-order correction b'_c, so it has a systematic bias of ~b'_c.
// Removing this bias (aware) must improve top-k agreement with L(q) when
// b'_c varies across chunks.

/// Return the indices of the `k` largest values in `scores` (descending).
fn topk_indices(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

/// Count how many elements of `a` appear in `b`.
fn intersection_size(a: &[usize], b: &[usize]) -> usize {
    let set: std::collections::HashSet<usize> = b.iter().copied().collect();
    a.iter().filter(|&&x| set.contains(&x)).count()
}

/// Compute log Σ_j exp(logits_j) with max-subtraction for stability.
fn logsumexp(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|&x| (x - max_val).exp()).sum::<f32>();
    max_val + sum.ln()
}

#[test]
fn goat_044_entropy_bias_improves_topk_vs_full_attention() {
    let hd = 32usize;
    let chunk_size = 32usize;
    let n_chunks = 48usize;
    let n_queries = 300usize;
    let top_k = 8usize;
    let scale = 1.0 / (hd as f32).sqrt();

    // ── Generate chunks with varied key distributions ──
    // Each chunk has a per-chunk center direction and spread, producing varied
    // entropy profiles when viewed from any non-trivial q_cls.
    let chunks: Vec<Vec<f32>> = (0..n_chunks)
        .map(|c| {
            // Vary the spread per chunk so entropy profiles differ.
            let spread = 0.15 + 0.85 * (((c * 7) as f32 * 0.13).sin() * 0.5 + 0.5);
            let center_seed = c.wrapping_mul(2654435761);
            let mut keys = Vec::with_capacity(chunk_size * hd);
            for j in 0..chunk_size {
                let token_seed = c.wrapping_mul(1000).wrapping_add(j).wrapping_mul(40503);
                for d in 0..hd {
                    let center = ((center_seed.wrapping_add(d)) as f32 * 0.001).sin() * 0.5;
                    let noise = ((token_seed.wrapping_add(d * 17)) as f32 * 0.001).sin();
                    keys.push(center + spread * noise);
                }
            }
            keys
        })
        .collect();

    // ── Deterministic q_cls = global key centroid ──
    // This is the maximum-likelihood center of the key distribution — the
    // natural modelless choice that minimizes the expected Taylor remainder.
    let mut centroid = vec![0.0f32; hd];
    for chunk in &chunks {
        for j in 0..chunk_size {
            for d in 0..hd {
                centroid[d] += chunk[j * hd + d];
            }
        }
    }
    let total = (n_chunks * chunk_size) as f32;
    for c in &mut centroid {
        *c /= total;
    }

    // ── Build ChunkSummaryQuery with the centroid ──
    let mut q_cls = ChunkSummaryQuery::new(1, hd);
    q_cls.head_cls.copy_from_slice(&centroid);
    q_cls.recompute_zero_init_cache();
    assert!(
        !q_cls.is_zero_init(),
        "centroid must be non-zero for the test to be meaningful"
    );

    // ── Summarize all chunks with q_cls → (k'_c, b'_c) ──
    let (summaries, entropy_biases): (Vec<Vec<f32>>, Vec<f32>) = (0..n_chunks)
        .map(|c| summarize_chunk_with_entropy(&q_cls, &chunks[c], chunk_size, 0, hd))
        .unzip();

    // Precondition: entropy must vary across chunks for a meaningful gate.
    let min_entropy = entropy_biases.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_entropy = entropy_biases.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let entropy_spread = max_entropy - min_entropy;
    assert!(
        entropy_spread > 0.01,
        "entropy must vary across chunks (spread={entropy_spread:.4})"
    );

    // ── For each query: ground-truth top-k vs blind top-k vs aware top-k ──
    let mut blind_hits = 0usize;
    let mut aware_hits = 0usize;
    let total_possible = n_queries * top_k;

    for seed in 0..n_queries {
        // Queries near the centroid (realistic: q and k share the same space).
        let query: Vec<f32> = (0..hd)
            .map(|d| {
                let noise = ((seed.wrapping_mul(hd).wrapping_add(d).wrapping_mul(2654435761))
                    as f32
                    * 0.001)
                    .sin();
                centroid[d] + noise * 0.5
            })
            .collect();

        // Ground truth: L(q) = logsumexp(q·k_j/√d) per chunk.
        let mut true_scores = vec![0.0f32; n_chunks];
        for c in 0..n_chunks {
            let chunk = &chunks[c];
            let mut logits = vec![0.0f32; chunk_size];
            for j in 0..chunk_size {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += query[d] * chunk[j * hd + d];
                }
                logits[j] = dot * scale;
            }
            true_scores[c] = logsumexp(&logits);
        }
        let true_topk = topk_indices(&true_scores, top_k);

        // Blind scores: q·k'_c/√d (Prop 3.1 without entropy correction).
        let blind_scores: Vec<f32> = summaries
            .iter()
            .map(|s| {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += query[d] * s[d];
                }
                dot * scale
            })
            .collect();

        // Aware scores: q·k'_c/√d + b'_c (full Prop 3.1).
        let aware_scores: Vec<f32> = summaries
            .iter()
            .zip(&entropy_biases)
            .map(|(s, b)| {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += query[d] * s[d];
                }
                dot * scale + b
            })
            .collect();

        let blind_topk = topk_indices(&blind_scores, top_k);
        let aware_topk = topk_indices(&aware_scores, top_k);

        blind_hits += intersection_size(&blind_topk, &true_topk);
        aware_hits += intersection_size(&aware_topk, &true_topk);
    }

    let blind_acc = blind_hits as f32 / total_possible as f32;
    let aware_acc = aware_hits as f32 / total_possible as f32;
    let delta = aware_acc - blind_acc;
    let chance = top_k as f32 / n_chunks as f32;

    println!(
        "┌──────────────────────────────────────────────────────────────────┐"
    );
    println!(
        "│ GOAT Gate (modelless): entropy bias vs full-attention LogSumExp  │"
    );
    println!(
        "├──────────────────────────────────────────────────────────────────┤"
    );
    println!("│ q_cls = global key centroid (deterministic, zero training)      │");
    println!("│ Chunks: {n_chunks}, chunk_size: {chunk_size}, queries: {n_queries}, top-k: {top_k}     ");
    println!("│ D={hd}, entropy range: [{min_entropy:.3}, {max_entropy:.3}] (spread {entropy_spread:.3})  ");
    println!("│ Chance baseline: {chance:.4}                                       ");
    println!("│ Blind accuracy:  {blind_acc:.4}  ({blind_hits}/{total_possible})                    ");
    println!("│ Aware accuracy:  {aware_acc:.4}  ({aware_hits}/{total_possible})                    ");
    println!("│ Delta: {delta:+.4} ({:+.1}%)                                        ", delta * 100.0);
    println!(
        "└──────────────────────────────────────────────────────────────────┘"
    );

    // GOAT gate G1 (quality): aware must STRICTLY beat blind. The Prop 3.1
    // entropy bias removes the systematic ~b'_c underestimation, so top-k
    // agreement with the true LogSumExp ranking must improve.
    assert!(
        aware_acc > blind_acc,
        "GOAT GATE G1 FAILED: entropy-aware top-k accuracy ({aware_acc:.4}) must exceed \
         entropy-blind ({blind_acc:.4}). The Prop 3.1 entropy bias did not improve \
         chunk selection for the centroid q_cls."
    );
}

#[test]
fn goat_044_zero_init_preserves_full_attention_ranking_correlation() {
    // G3 (no-regression) complement: at zero-init, the entropy is constant
    // (ln(chunk_size)) so adding it must NOT change the ranking vs blind.
    // This is the dormant-at-zero-init guarantee expressed as top-k agreement:
    // blind and aware top-k must be IDENTICAL when entropy is uniform.
    let hd = 32usize;
    let chunk_size = 32usize;
    let n_chunks = 48usize;
    let top_k = 8usize;
    let scale = 1.0 / (hd as f32).sqrt();

    let chunks: Vec<Vec<f32>> = (0..n_chunks)
        .map(|c| {
            let spread = 0.15 + 0.85 * (((c * 7) as f32 * 0.13).sin() * 0.5 + 0.5);
            let center_seed = c.wrapping_mul(2654435761);
            let mut keys = Vec::with_capacity(chunk_size * hd);
            for j in 0..chunk_size {
                let token_seed = c.wrapping_mul(1000).wrapping_add(j).wrapping_mul(40503);
                for d in 0..hd {
                    let center = ((center_seed.wrapping_add(d)) as f32 * 0.001).sin() * 0.5;
                    let noise = ((token_seed.wrapping_add(d * 17)) as f32 * 0.001).sin();
                    keys.push(center + spread * noise);
                }
            }
            keys
        })
        .collect();

    // Zero-init query → mean-pool summaries, constant entropy.
    let q_zero = ChunkSummaryQuery::new(1, hd);
    assert!(q_zero.is_zero_init());

    let (summaries, entropy_biases): (Vec<Vec<f32>>, Vec<f32>) = (0..n_chunks)
        .map(|c| summarize_chunk_with_entropy(&q_zero, &chunks[c], chunk_size, 0, hd))
        .unzip();

    // All entropy biases must be ln(chunk_size) at zero-init.
    let expected_entropy = (chunk_size as f32).ln();
    for (c, &b) in entropy_biases.iter().enumerate() {
        assert!(
            (b - expected_entropy).abs() < 1e-5,
            "zero-init entropy for chunk {c} should be ln({chunk_size})={expected_entropy:.5}, got {b:.5}"
        );
    }

    // For any query, blind and aware top-k must be identical (constant entropy
    // → constant additive shift → ranking preserved).
    let mut mismatches = 0usize;
    for seed in 0usize..100 {
        let query: Vec<f32> = (0..hd)
            .map(|d| {
                ((seed.wrapping_mul(hd).wrapping_add(d).wrapping_mul(2654435761)) as f32
                    * 0.001)
                    .sin()
            })
            .collect();

        let blind: Vec<f32> = summaries
            .iter()
            .map(|s| {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += query[d] * s[d];
                }
                dot * scale
            })
            .collect();
        let aware: Vec<f32> = summaries
            .iter()
            .zip(&entropy_biases)
            .map(|(s, b)| {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += query[d] * s[d];
                }
                dot * scale + b
            })
            .collect();

        let blind_topk = topk_indices(&blind, top_k);
        let aware_topk = topk_indices(&aware, top_k);
        if blind_topk != aware_topk {
            mismatches += 1;
        }
    }

    assert_eq!(
        mismatches, 0,
        "zero-init (constant entropy) must not change ranking: {mismatches}/100 mismatches"
    );
}
