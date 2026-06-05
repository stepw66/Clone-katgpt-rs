//! Plan 193 T15: GOAT proof for SpeculativeGenerator trait.
//!
//! Validates that the `SpeculativeGenerator` trait path produces identical results
//! to the standard `build_dd_tree_screened` path, that pruning works correctly,
//! and that trait dispatch adds negligible overhead (≤5%).
//!
//! ```sh
//! cargo test --features "speculative_generator" --test speculative_generator_goat -- --nocapture
//! ```

#![cfg(feature = "speculative_generator")]

use katgpt_core::{Config, ConstraintPruner, NoPruner};
use katgpt_rs::speculative::{
    MarginalTokenGenerator, NoScreeningPruner, TokenConstraintPruner, build_dd_tree_screened,
    build_dd_tree_speculative, extract_best_path,
};

/// Generate uniform-ish marginals for testing.
///
/// Each depth gets `n_tokens` entries with a slight linear bias so the best
/// path is deterministic (token index 0 is always highest).
fn make_test_marginals(depths: usize, n_tokens: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(depths);
    for _d in 0..depths {
        let mut row = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            // Higher index = lower prob, with depth-based variation
            let v = 1.0 / ((t + 1) as f32);
            row.push(v);
        }
        out.push(row);
    }
    out
}

/// Marginals where every 3rd token has probability 0.0 (invalid).
fn make_sparse_marginals(depths: usize, n_tokens: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(depths);
    for _d in 0..depths {
        let mut row = Vec::with_capacity(n_tokens);
        for t in 0..n_tokens {
            let v = if t % 3 == 0 {
                0.0
            } else {
                1.0 / ((t + 1) as f32)
            };
            row.push(v);
        }
        out.push(row);
    }
    out
}

// ── Test 1: Equivalence ─────────────────────────────────────────────

#[test]
fn test_speculative_generator_goat_equivalence() {
    let config = Config::draft();
    let mut rng = fastrand::Rng::new();

    let marginals = make_test_marginals(5, 100);
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Standard path: build_dd_tree_screened with NoScreeningPruner
    let tree_standard = build_dd_tree_screened(&slices, &config, &NoScreeningPruner, false);

    // SpeculativeGenerator path: MarginalTokenGenerator + TokenConstraintPruner<NoPruner>
    let mut generator = MarginalTokenGenerator { top_k: 100 };
    let pruner = TokenConstraintPruner::new(NoPruner);
    let tree_spec = build_dd_tree_speculative(&mut generator, &pruner, &slices, &config, &mut rng);

    // Same node count
    assert_eq!(
        tree_standard.len(),
        tree_spec.len(),
        "node count mismatch: standard={}, speculative={}",
        tree_standard.len(),
        tree_spec.len(),
    );

    // Same best path
    let path_standard = extract_best_path(&tree_standard);
    let path_spec = extract_best_path(&tree_spec);
    assert_eq!(
        path_standard, path_spec,
        "best path differs: standard={:?}, speculative={:?}",
        path_standard, path_spec,
    );

    // Same token indices and scores (float tolerance)
    for (i, (a, b)) in tree_standard.iter().zip(tree_spec.iter()).enumerate() {
        assert_eq!(
            a.token_idx, b.token_idx,
            "token mismatch at node {i}: standard={}, speculative={}",
            a.token_idx, b.token_idx,
        );
        assert_eq!(a.depth, b.depth, "depth mismatch at node {i}",);
        assert!(
            (a.score - b.score).abs() < 1e-3,
            "score mismatch at node {i}: standard={:.6}, speculative={:.6}",
            a.score,
            b.score,
        );
    }

    println!(
        "\n✅ GOAT Equivalence: {} nodes, best path {:?}",
        tree_spec.len(),
        path_spec
    );
}

// ── Test 2: Pruning Effectiveness ───────────────────────────────────

/// Pruner that rejects tokens whose index is divisible by 3.
struct FilterMod3Pruner;

impl ConstraintPruner for FilterMod3Pruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        token_idx % 3 != 0
    }
}

#[test]
fn test_speculative_generator_goat_pruning_effectiveness() {
    let config = Config::draft();
    let mut rng = fastrand::Rng::new();

    let marginals = make_sparse_marginals(5, 100);
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Unfiltered: NoPruner → all non-zero marginals included
    let mut gen_unfiltered = MarginalTokenGenerator { top_k: 100 };
    let pruner_unfiltered = TokenConstraintPruner::new(NoPruner);
    let tree_unfiltered = build_dd_tree_speculative(
        &mut gen_unfiltered,
        &pruner_unfiltered,
        &slices,
        &config,
        &mut rng,
    );

    // Filtered: FilterMod3Pruner rejects tokens with idx % 3 == 0
    let mut gen_filtered = MarginalTokenGenerator { top_k: 100 };
    let pruner_filtered = TokenConstraintPruner::new(FilterMod3Pruner);
    let tree_filtered = build_dd_tree_speculative(
        &mut gen_filtered,
        &pruner_filtered,
        &slices,
        &config,
        &mut rng,
    );

    // Filtered tree should have fewer or equal nodes (pruned branches)
    assert!(
        tree_filtered.len() <= tree_unfiltered.len(),
        "pruned tree ({} nodes) should have ≤ unfiltered ({} nodes)",
        tree_filtered.len(),
        tree_unfiltered.len(),
    );

    // All nodes in the filtered tree must have valid token indices (not % 3 == 0)
    for node in &tree_filtered {
        assert_ne!(
            node.token_idx % 3,
            0,
            "node at depth {} has pruned token_idx={}",
            node.depth,
            node.token_idx,
        );
    }

    println!(
        "\n✅ GOAT Pruning: unfiltered={} nodes, filtered={} nodes ({:.1}% reduction)",
        tree_unfiltered.len(),
        tree_filtered.len(),
        (1.0 - tree_filtered.len() as f64 / tree_unfiltered.len() as f64) * 100.0,
    );
}

// ── Test 3: Overhead ────────────────────────────────────────────────

#[test]
fn test_speculative_generator_goat_overhead() {
    let config = Config::draft();
    let n_iters = 100;

    let marginals = make_test_marginals(5, 100);
    let slices: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Warm up (avoid cold-start bias)
    {
        let mut rng = fastrand::Rng::new();
        let mut gen_warmup = MarginalTokenGenerator { top_k: 100 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        for _ in 0..5 {
            let _ = build_dd_tree_speculative(&mut gen_warmup, &pruner, &slices, &config, &mut rng);
        }
    }
    {
        for _ in 0..5 {
            let _ = build_dd_tree_screened(&slices, &config, &NoScreeningPruner, false);
        }
    }

    // Measure standard path
    let start_standard = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = build_dd_tree_screened(&slices, &config, &NoScreeningPruner, false);
    }
    let elapsed_standard = start_standard.elapsed();

    // Measure speculative path
    let start_spec = std::time::Instant::now();
    for _ in 0..n_iters {
        let mut rng = fastrand::Rng::new();
        let mut gen_iter = MarginalTokenGenerator { top_k: 100 };
        let pruner = TokenConstraintPruner::new(NoPruner);
        let _ = build_dd_tree_speculative(&mut gen_iter, &pruner, &slices, &config, &mut rng);
    }
    let elapsed_spec = start_spec.elapsed();

    let overhead_pct = (elapsed_spec.as_secs_f64() - elapsed_standard.as_secs_f64())
        / elapsed_standard.as_secs_f64()
        * 100.0;

    println!("\n── GOAT Overhead ({n_iters} iterations) ──");
    println!("   Standard:  {:?}", elapsed_standard);
    println!("   Speculative: {:?}", elapsed_spec);
    println!("   Overhead: {overhead_pct:.1}%");

    // Assert ≤ 5% overhead
    assert!(
        overhead_pct <= 5.0,
        "speculative path overhead {overhead_pct:.1}% exceeds 5% threshold \
         (standard={elapsed_standard:?}, spec={elapsed_spec:?})",
    );

    println!("\n✅ GOAT Overhead: {overhead_pct:.1}% ≤ 5%");
}

// TL;DR: GOAT proof for SpeculativeGenerator — equivalence (identical trees),
// pruning effectiveness (fewer nodes, valid tokens), overhead ≤5%. Feature-gated
// behind `speculative_generator`.
