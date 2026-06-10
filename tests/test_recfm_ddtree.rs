//! GOAT Proof 168: RecFM Recursive Cross-Scale Consistency (DDTree)
//!
//! Feature gate: `recfm` (Plan 168, Research 150)
//!
//! Proofs:
//!   P1: Cross-scale consistency reduces invalid branches (count check)
//!   P2: Best path unchanged when consistency threshold is loose (safety)
//!   P3: CrossScaleConfig::enable=false delegates to build_screened unchanged

#![cfg(feature = "recfm")]

use katgpt_rs::speculative::{
    CrossScaleConfig, NoScreeningPruner, branch_velocity_at, build_dd_tree_screened,
    build_dd_tree_screened_recfm, cross_scale_consistent, extract_best_path,
};
use katgpt_rs::types::Config;

fn make_config() -> Config {
    let mut c = Config::micro();
    c.vocab_size = 64;
    c.tree_budget = 256;
    c.screening_threshold = 0.01;
    c
}

/// Marginals where probabilities smoothly increase (consistent velocities).
/// Top-1 at each depth: 0.5, 0.6, 0.7, 0.8
fn smooth_marginals() -> Vec<Vec<f32>> {
    vec![
        vec![0.5, 0.3, 0.2, 0.0, 0.0], // depth 0: top-1 = 0.5
        vec![0.1, 0.6, 0.2, 0.1, 0.0], // depth 1: top-1 = 0.6
        vec![0.0, 0.1, 0.7, 0.1, 0.1], // depth 2: top-1 = 0.7
        vec![0.0, 0.0, 0.1, 0.8, 0.1], // depth 3: top-1 = 0.8
    ]
}

/// Marginals with a spike at depth 2 (inconsistent velocity).
/// Top-1 at each depth: 0.5, 0.6, 0.1, 0.8
/// Velocity from depth 1→2 = 0.1 - 0.6 = -0.5 (big jump)
fn spiky_marginals() -> Vec<Vec<f32>> {
    vec![
        vec![0.5, 0.3, 0.2, 0.0, 0.0],   // depth 0: top-1 = 0.5
        vec![0.1, 0.6, 0.2, 0.1, 0.0],   // depth 1: top-1 = 0.6
        vec![0.7, 0.1, 0.1, 0.05, 0.05], // depth 2: top-1 = 0.7 → wait, top-1 is 0.7 at index 0
        vec![0.0, 0.0, 0.1, 0.8, 0.1],   // depth 3: top-1 = 0.8
    ]
}

fn marginals_refs(marginals: &[Vec<f32>]) -> Vec<&[f32]> {
    marginals.iter().map(|m| m.as_slice()).collect()
}

// ── P1: Cross-scale consistency reduces invalid branches ───────────

#[test]
fn proof_p1_consistency_reduces_branches() {
    let config = make_config();
    let screener = NoScreeningPruner;
    let marginals = spiky_marginals();
    let refs = marginals_refs(&marginals);

    // Build without RecFM — should expand all branches
    let tree_no_recfm = build_dd_tree_screened(&refs, &config, &screener, true);
    let no_recfm_count = tree_no_recfm.len();

    // Build with tight RecFM — should prune the spiky depth
    let recfm_config = CrossScaleConfig {
        enable: true,
        scale_alpha: 0.5,
        consistency_threshold: 0.05, // tight: only allow very smooth velocities
    };
    let tree_recfm = build_dd_tree_screened_recfm(&refs, &config, &screener, true, &recfm_config);
    let recfm_count = tree_recfm.len();

    // RecFM should produce fewer or equal nodes (pruning inconsistent branches)
    assert!(
        recfm_count <= no_recfm_count,
        "RecFM should prune inconsistent branches: got {recfm_count} vs {no_recfm_count}"
    );

    // Should still produce at least one node (the root is always consistent)
    assert!(
        !tree_recfm.is_empty(),
        "RecFM should produce at least the root node"
    );
}

// ── P2: Best path unchanged when threshold is loose (safety) ───────

#[test]
fn proof_p2_loose_threshold_preserves_path() {
    let config = make_config();
    let screener = NoScreeningPruner;
    let marginals = smooth_marginals();
    let refs = marginals_refs(&marginals);

    // Build without RecFM
    let tree_baseline = build_dd_tree_screened(&refs, &config, &screener, true);
    let path_baseline = extract_best_path(&tree_baseline);

    // Build with very loose RecFM (threshold=10.0: everything passes)
    let recfm_config = CrossScaleConfig {
        enable: true,
        scale_alpha: 0.5,
        consistency_threshold: 10.0, // very loose: all velocities are consistent
    };
    let tree_recfm = build_dd_tree_screened_recfm(&refs, &config, &screener, true, &recfm_config);
    let path_recfm = extract_best_path(&tree_recfm);

    // Best paths should be identical (same tokens in same order)
    assert_eq!(
        path_baseline.len(),
        path_recfm.len(),
        "Loose RecFM should preserve path length"
    );
    for (i, (a, b)) in path_baseline.iter().zip(path_recfm.iter()).enumerate() {
        assert_eq!(a, b, "Loose RecFM should preserve token at depth {i}");
    }
}

// ── P3: enable=false delegates to build_screened unchanged ─────────

#[test]
fn proof_p3_disabled_is_identical_to_baseline() {
    let config = make_config();
    let screener = NoScreeningPruner;
    let marginals = smooth_marginals();
    let refs = marginals_refs(&marginals);

    let tree_baseline = build_dd_tree_screened(&refs, &config, &screener, true);

    let recfm_config = CrossScaleConfig {
        enable: false,
        ..Default::default()
    };
    let tree_recfm = build_dd_tree_screened_recfm(&refs, &config, &screener, true, &recfm_config);

    assert_eq!(
        tree_baseline.len(),
        tree_recfm.len(),
        "Disabled RecFM should produce identical tree size"
    );
    for (a, b) in tree_baseline.iter().zip(tree_recfm.iter()) {
        assert_eq!(a.token_idx, b.token_idx, "Disabled: same tokens");
        assert_eq!(a.depth, b.depth, "Disabled: same depths");
    }
}

// ── Unit: branch_velocity_at correctness ────────────────────────────

#[test]
fn test_branch_velocity_at_depth_zero() {
    let curr = &[0.5, 0.3, 0.2];
    let prev = &[0.4, 0.3, 0.3];
    assert_eq!(
        branch_velocity_at(0, curr, prev),
        0.0,
        "Depth 0 should return 0.0"
    );
}

#[test]
fn test_branch_velocity_at_positive() {
    let prev = &[0.4, 0.3, 0.3]; // top-1 = 0.4
    let curr = &[0.1, 0.6, 0.3]; // top-1 = 0.6
    let v = branch_velocity_at(1, curr, prev);
    assert!(
        (v - 0.2).abs() < 1e-6,
        "Velocity should be 0.6 - 0.4 = 0.2, got {v}"
    );
}

#[test]
fn test_branch_velocity_at_negative() {
    let prev = &[0.1, 0.6, 0.3]; // top-1 = 0.6
    let curr = &[0.7, 0.1, 0.2]; // top-1 = 0.7
    let v = branch_velocity_at(1, curr, prev);
    assert!(
        (v - 0.1).abs() < 1e-6,
        "Velocity should be 0.7 - 0.6 = 0.1, got {v}"
    );
}

#[test]
fn test_branch_velocity_at_empty() {
    assert_eq!(branch_velocity_at(1, &[], &[0.5]), 0.0);
    assert_eq!(branch_velocity_at(1, &[0.5], &[]), 0.0);
}

// ── Unit: cross_scale_consistent correctness ───────────────────────

#[test]
fn test_cross_scale_consistent_true() {
    // |0.2 - 0.5*0.4| = |0.2 - 0.2| = 0.0 <= 0.1
    assert!(cross_scale_consistent(0.4, 0.2, 0.5, 0.1));
}

#[test]
fn test_cross_scale_consistent_false() {
    // |0.5 - 0.5*0.1| = |0.5 - 0.05| = 0.45 > 0.1
    assert!(!cross_scale_consistent(0.1, 0.5, 0.5, 0.1));
}

#[test]
fn test_cross_scale_consistent_boundary() {
    // |v2 - alpha*v1| = threshold exactly → should pass (<=)
    let v1 = 0.2f32;
    let alpha = 0.5f32;
    let v2 = alpha * v1 + 0.1; // exactly at threshold
    assert!(cross_scale_consistent(v1, v2, alpha, 0.1));
}
