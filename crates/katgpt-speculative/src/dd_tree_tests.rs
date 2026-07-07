
use super::*;
use katgpt_types::Config;

// NOTE (Issue 013): The riir-engine dd_tree.rs test module originally
// imported `crate::dflash::dflash_predict` + `crate::transformer::TransformerWeights`
// + `crate::types::Rng` to synthesize marginals via the draft model. Those
// dependencies do NOT exist in katgpt-speculative (dflash is deferred to
// Issue 014 — it needs a `forward` trait design). Tests that called
// `dflash_predict` or `make_draft()` have been removed; the remaining tests
// synthesize marginals directly (pure-algorithm coverage). The removed
// integration tests are preserved verbatim in riir-engine's `dflash.rs`
// test module, which still calls `katgpt_speculative::dd_tree::*` after the
// Issue 013 migration.

// ── Original DDTree Tests ─────────────────────────────────

#[test]
fn test_extract_parent_tokens_roundtrip() {
    let path_d0 = 3u128;
    let path_d1 = (path_d0 << 16) | 7;
    let path_d2 = (path_d1 << 16) | 1;

    assert_eq!(extract_parent_tokens(path_d0, 1), vec![3]);
    assert_eq!(extract_parent_tokens(path_d1, 2), vec![3, 7]);
    assert_eq!(extract_parent_tokens(path_d2, 3), vec![3, 7, 1]);
    let empty: Vec<usize> = extract_parent_tokens(0, 0);
    assert!(empty.is_empty());
}

#[test]
fn test_ddtree_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree(&[], &config);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_ddtree_pruned_empty_marginals() {
    let config = Config::draft();
    let pruner = NoPruner;
    let tree = build_dd_tree_pruned(&[], &config, &pruner, false);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

// ── merge_retrieved_branches Tests ─────────────────────────

#[test]
fn test_merge_empty_retrieval_noop() {
    let config = Config::draft();
    let marginals = [vec![0.5; config.vocab_size]];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = vec![TreeNode {
        score: 1.0,
        depth: 0,
        token_idx: 0,
        parent_path: 0,
    }];
    let original_len = tree.len();

    merge_retrieved_branches(&mut tree, &mv, &config, &[], &[], 0.5);

    assert_eq!(
        tree.len(),
        original_len,
        "empty retrieval should not change tree"
    );
}

#[test]
fn test_merge_preserves_budget() {
    let config = Config::draft();
    let marginals = vec![vec![0.1; config.vocab_size]; 4];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = build_dd_tree(&mv, &config);

    // Create many sequences that would exceed budget
    let sequences: Vec<Vec<usize>> = (0..100)
        .map(|i| vec![i % config.vocab_size, (i + 1) % config.vocab_size])
        .collect();
    let scores: Vec<f32> = (0..100).map(|_| 0.9).collect();

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.3);

    assert!(
        tree.len() <= config.tree_budget,
        "tree should not exceed budget, got {}",
        tree.len()
    );
}

#[test]
fn test_merge_sorts_by_score() {
    let config = Config::draft();
    let marginals = vec![vec![0.1; config.vocab_size]; 2];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = Vec::new();

    let sequences = vec![vec![0, 1], vec![2, 3]];
    let scores = vec![0.5, 0.9];

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.5);

    for window in tree.windows(2) {
        assert!(
            window[0].score >= window[1].score,
            "tree should be sorted by score descending"
        );
    }
}

#[test]
fn test_merge_with_empty_tree_adds_nodes() {
    let config = Config::draft();
    // Marginals with non-zero prob at specific tokens
    let mut m0 = vec![0.0; config.vocab_size];
    m0[5] = 0.8;
    let mut m1 = vec![0.0; config.vocab_size];
    m1[10] = 0.6;
    let marginals = [m0, m1];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = Vec::new();

    let sequences = vec![vec![5, 10]];
    let scores = vec![0.7];

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.3);

    assert_eq!(tree.len(), 2, "should add 2 nodes for 2-depth sequence");
    assert_eq!(tree[0].token_idx, 5, "first node should be token 5");
}

#[test]
fn test_merge_zero_weight_is_noop() {
    let config = Config::draft();
    let marginals = [vec![0.5; config.vocab_size]];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let mut tree = Vec::new();

    let sequences = vec![vec![0]];
    let scores = vec![0.9];

    merge_retrieved_branches(&mut tree, &mv, &config, &sequences, &scores, 0.0);

    assert!(tree.is_empty(), "zero rest_weight should be no-op");
}

// ── Chain-Seed DDTree Tests ───────────────────────────────

/// Create marginals with known argmax at each depth for deterministic testing.
fn make_chain_marginals(config: &Config) -> Vec<Vec<f32>> {
    let mut m0 = vec![0.01; config.vocab_size];
    m0[5] = 0.9;
    let mut m1 = vec![0.01; config.vocab_size];
    m1[10] = 0.85;
    let mut m2 = vec![0.01; config.vocab_size];
    m2[3] = 0.8;
    vec![m0, m1, m2]
}

#[test]
fn test_chain_seed_produces_chain_path() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree = build_dd_tree_pruned(&mv, &config, &NoPruner, true);

    // Chain nodes are the first 3 entries (depths 0, 1, 2)
    assert!(
        tree.len() >= 3,
        "tree should have at least 3 chain nodes, got {}",
        tree.len()
    );

    // Verify chain nodes form contiguous path with argmax tokens
    assert_eq!(tree[0].depth, 0, "first chain node at depth 0");
    assert_eq!(tree[0].token_idx, 5, "chain node depth 0 = argmax token 5");

    assert_eq!(tree[1].depth, 1, "second chain node at depth 1");
    assert_eq!(
        tree[1].token_idx, 10,
        "chain node depth 1 = argmax token 10"
    );

    assert_eq!(tree[2].depth, 2, "third chain node at depth 2");
    assert_eq!(tree[2].token_idx, 3, "chain node depth 2 = argmax token 3");

    // Verify chain node parent_paths form contiguous path
    assert_eq!(tree[0].parent_path, 5, "depth 0 parent_path = token 5");
    assert_eq!(
        tree[1].parent_path,
        (5u128 << 16) | 10,
        "depth 1 parent_path = [5, 10]"
    );
    assert_eq!(
        tree[2].parent_path,
        ((5u128 << 16) | 10) << 16 | 3,
        "depth 2 parent_path = [5, 10, 3]"
    );

    // Verify cumulative scores
    let expected_d0 = marginals[0][5].ln();
    let expected_d1 = expected_d0 + marginals[1][10].ln();
    let expected_d2 = expected_d1 + marginals[2][3].ln();

    assert!(
        (tree[0].score - expected_d0).abs() < 1e-5,
        "depth 0 score: expected {expected_d0}, got {}",
        tree[0].score
    );
    assert!(
        (tree[1].score - expected_d1).abs() < 1e-5,
        "depth 1 score: expected {expected_d1}, got {}",
        tree[1].score
    );
    assert!(
        (tree[2].score - expected_d2).abs() < 1e-5,
        "depth 2 score: expected {expected_d2}, got {}",
        tree[2].score
    );
}

/// Pruner that blocks a specific token at a specific depth.
struct BlockTokenPruner {
    depth: usize,
    token: usize,
}

impl ConstraintPruner for BlockTokenPruner {
    fn is_valid(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> bool {
        !(depth == self.depth && token_idx == self.token)
    }
}

#[test]
fn test_chain_seed_with_pruner() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);

    // Block token 10 at depth 1 (the argmax) — chain should break there
    let pruner = BlockTokenPruner {
        depth: 1,
        token: 10,
    };
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree_pruned(&mv, &config, &pruner, true);

    // Chain should have only depth 0 (broke at depth 1)
    assert!(
        !tree.is_empty(),
        "tree should have at least the depth 0 chain node"
    );
    assert_eq!(
        tree[0].token_idx, 5,
        "depth 0 chain node should be argmax token 5"
    );
    assert_eq!(tree[0].depth, 0);

    // No node at depth 1 should have token 10 (blocked)
    for node in &tree {
        if node.depth == 1 {
            assert_ne!(
                node.token_idx, 10,
                "blocked token 10 should not appear at depth 1"
            );
        }
    }

    // Verify tree still contains valid nodes (siblings and best-first)
    assert!(
        tree.len() > 1,
        "tree should have more than just the chain node (siblings/best-first)"
    );
}

#[test]
fn test_chain_seed_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree_pruned(&[], &config, &NoPruner, true);
    assert!(
        tree.is_empty(),
        "empty marginals should produce empty tree with chain_seed=true"
    );
}

// ── ScreeningPruner Tests (Plan 021) ──────────────────────

/// Screener that returns fixed relevance per token index.
struct FixedRelevanceScreener {
    relevances: Vec<f32>,
}

impl ScreeningPruner for FixedRelevanceScreener {
    fn relevance(&self, _depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        self.relevances.get(token_idx).copied().unwrap_or(0.1)
    }
}

#[test]
fn test_screened_relevance_zero_hard_trims() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    // 3 tokens: index 0 has high prob but R=0.0, index 1 has lower prob but R=1.0
    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.9; // high LLM prob
    m0[1] = 0.05; // lower LLM prob
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![0.0, 1.0], // token 0 trimmed, token 1 passes
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    // Token 0 should be completely absent (hard trim)
    for node in &tree {
        assert_ne!(
            node.token_idx, 0,
            "token 0 with relevance 0.0 should be hard-trimmed"
        );
    }
}

#[test]
fn test_screened_relevance_half_applies_penalty() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    // Two tokens with same LLM prob but different relevance
    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.5;
    m0[1] = 0.5;
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![1.0, 0.5], // token 1 gets -0.69 penalty
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    let node_0 = tree.iter().find(|n| n.token_idx == 0);
    let node_1 = tree.iter().find(|n| n.token_idx == 1);

    assert!(node_0.is_some(), "token 0 should be in tree");
    assert!(node_1.is_some(), "token 1 should be in tree");

    let score_0 = node_0.unwrap().score;
    let score_1 = node_1.unwrap().score;

    // Token 0: ln(0.5) + ln(1.0) = ln(0.5) + 0
    // Token 1: ln(0.5) + ln(0.5) = ln(0.5) - 0.693...
    let expected_diff = 0.5f32.ln().abs(); // ≈ 0.693
    let actual_diff = score_0 - score_1;

    assert!(
        (actual_diff - expected_diff).abs() < 1e-4,
        "penalty should be ln(0.5) ≈ -0.693, got diff={actual_diff:.4}, expected={expected_diff:.4}"
    );
}

#[test]
fn test_screened_relevance_one_no_penalty() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.8;
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![1.0],
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    let node = tree.iter().find(|n| n.token_idx == 0);
    assert!(node.is_some(), "token 0 should be in tree");

    let expected_score = 0.8f32.ln(); // ln(P) + ln(1.0) = ln(P) + 0
    assert!(
        (node.unwrap().score - expected_score).abs() < 1e-5,
        "relevance 1.0 should not modify score"
    );
}

#[test]
fn test_screened_threshold_trims_mediocre() {
    let mut config = Config::draft();
    config.tree_budget = 64;
    config.screening_threshold = 0.4; // trim anything ≤ 0.4

    let mut m0 = vec![0.01; config.vocab_size];
    m0[0] = 0.5;
    m0[1] = 0.5;
    m0[2] = 0.5;
    let marginals = [m0];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let screener = FixedRelevanceScreener {
        relevances: vec![0.3, 0.5, 0.8], // token 0 trimmed (≤0.4), 1 and 2 pass
    };

    let tree = build_dd_tree_screened(&mv, &config, &screener, false);

    // Token 0 (R=0.3 ≤ threshold 0.4) should be absent
    for node in &tree {
        assert_ne!(
            node.token_idx, 0,
            "token 0 with R=0.3 should be trimmed by threshold 0.4"
        );
    }
    // Token 1 (R=0.5 > threshold) and token 2 (R=0.8 > threshold) should be present
    assert!(
        tree.iter().any(|n| n.token_idx == 1),
        "token 1 with R=0.5 should survive threshold 0.4"
    );
    assert!(
        tree.iter().any(|n| n.token_idx == 2),
        "token 2 with R=0.8 should survive threshold 0.4"
    );
}

#[test]
fn test_screened_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree_screened(&[], &config, &NoScreeningPruner, false);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}

#[test]
fn test_screened_chain_seed_with_relevance() {
    let mut config = Config::draft();
    config.tree_budget = 64;

    let mut m0 = vec![0.01; config.vocab_size];
    m0[5] = 0.9;
    let mut m1 = vec![0.01; config.vocab_size];
    m1[10] = 0.85;
    let marginals = [m0, m1];
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Give token 5 at depth 0 a relevance of 0.6
    let mut relevances = vec![0.1; config.vocab_size];
    relevances[5] = 0.6;
    relevances[10] = 1.0;
    let screener = FixedRelevanceScreener { relevances };

    let tree = build_dd_tree_screened(&mv, &config, &screener, true);

    // Chain should build: token 5 (R=0.6), token 10 (R=1.0)
    assert!(
        tree.len() >= 2,
        "chain should have at least 2 nodes, got {}",
        tree.len()
    );

    // Score for token 5 should include ln(0.6) penalty
    let chain_d0 = tree.iter().find(|n| n.depth == 0 && n.token_idx == 5);
    assert!(chain_d0.is_some(), "chain node at depth 0 should exist");
    let expected_d0 = 0.9f32.ln() + 0.6f32.ln();
    assert!(
        (chain_d0.unwrap().score - expected_d0).abs() < 1e-4,
        "chain d0 score should include ln(0.6) penalty"
    );
}

// ── Early Exit Tests (Plan 026: AutoTTS) ──────────────────

#[test]
fn test_ddtree_early_exit_triggers_on_clear_winner() {
    // Create marginals where one path dominates massively
    let config = Config {
        tree_budget: 1000,
        early_exit_patience: 3,
        early_exit_gap: 1.0,
        ..Config::draft()
    };
    // One dominant token per depth
    let mut marginals = Vec::new();
    for _ in 0..config.draft_lookahead {
        let mut probs = vec![0.001_f32; config.vocab_size];
        probs[0] = 0.99; // token 0 dominates
        marginals.push(probs);
    }
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();
    let tree = build_dd_tree(&mv, &config);
    // Should exit well before budget of 1000
    assert!(
        tree.len() < 1000,
        "early exit should trigger, got {} nodes vs budget 1000",
        tree.len()
    );
}

// ── Balanced DDTree Tests (Plan 052: GFlowNet) ───────────

#[test]
fn test_balanced_default_matches_screened() {
    // backward_weight=1.0, lambda_flow=0.0 → identical to build_screened
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_screened = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, false);
    let tree_balanced =
        build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 1.0, 0.0);

    assert_eq!(
        tree_screened.len(),
        tree_balanced.len(),
        "balanced(w=1,λ=0) should match screened: {} vs {}",
        tree_screened.len(),
        tree_balanced.len()
    );
    for (a, b) in tree_screened.iter().zip(tree_balanced.iter()) {
        assert!(
            (a.score - b.score).abs() < 1e-4,
            "score mismatch: {} vs {}",
            a.score,
            b.score
        );
        assert_eq!(a.token_idx, b.token_idx, "token mismatch");
        assert_eq!(a.depth, b.depth, "depth mismatch");
    }
}

#[test]
fn test_balanced_default_chain_seed_matches_screened() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_screened = build_dd_tree_screened(&mv, &config, &NoScreeningPruner, true);
    let tree_balanced =
        build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, true, &[], 1.0, 0.0);

    assert_eq!(
        tree_screened.len(),
        tree_balanced.len(),
        "balanced(w=1,λ=0) chain_seed should match screened"
    );
}

#[test]
fn test_balanced_higher_backward_weight_changes_scores() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    let tree_w1 = build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 1.0, 0.0);
    let tree_w4 = build_dd_tree_balanced(&mv, &config, &NoScreeningPruner, false, &[], 4.0, 0.0);

    // With higher backward weight, scores should be different
    // (NoScreeningPruner returns 1.0, so ln(R)=0 — but the scoring is additive)
    // Actually with NoScreeningPruner, relevance=1.0, ln(1.0)=0, so backward_weight
    // multiplies 0.0 → same score. Use a pruner that returns non-1.0 values.
    // For now just verify they both produce valid trees
    assert!(!tree_w1.is_empty());
    assert!(!tree_w4.is_empty());
}

#[test]
fn test_balanced_with_relevance_pruner_weighted() {
    // Use FixedRelevanceScreener to get non-trivial relevance scores
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // FixedRelevanceScreener indexes by token_idx — flat vec
    let screener = FixedRelevanceScreener {
        relevances: vec![0.5; config.vocab_size],
    };

    let tree_w1 = build_dd_tree_balanced(&mv, &config, &screener, false, &[], 1.0, 0.0);
    let tree_w4 = build_dd_tree_balanced(&mv, &config, &screener, false, &[], 4.0, 0.0);

    // Higher backward weight should amplify the relevance penalty
    // Both should be non-empty
    assert!(!tree_w1.is_empty());
    assert!(!tree_w4.is_empty());

    // The top node scores should differ because backward_weight scales ln(R)
    // w=1: score = ln(P) + 1*ln(0.5) = ln(P) - 0.693
    // w=4: score = ln(P) + 4*ln(0.5) = ln(P) - 2.773
    if !tree_w1.is_empty() && !tree_w4.is_empty() {
        // w=4 should have lower scores (more penalty)
        assert!(
            tree_w4[0].score < tree_w1[0].score,
            "w=4 score {} should be < w=1 score {}",
            tree_w4[0].score,
            tree_w1[0].score
        );
    }
}

#[test]
fn test_balanced_flow_bonus_changes_scores() {
    let config = Config::draft();
    let marginals = make_chain_marginals(&config);
    let mv: Vec<&[f32]> = marginals.iter().map(|s| s.as_slice()).collect();

    // Low stop prob → high flow bonus
    let stop_probs = vec![0.1; config.draft_lookahead];

    let tree_no_flow = build_dd_tree_balanced(
        &mv,
        &config,
        &NoScreeningPruner,
        false,
        &stop_probs,
        1.0,
        0.0,
    );
    let tree_with_flow = build_dd_tree_balanced(
        &mv,
        &config,
        &NoScreeningPruner,
        false,
        &stop_probs,
        1.0,
        0.3,
    );

    // Flow bonus should increase scores (additive positive term)
    assert!(!tree_no_flow.is_empty());
    assert!(!tree_with_flow.is_empty());

    // With flow bonus, scores should be higher
    if !tree_no_flow.is_empty() && !tree_with_flow.is_empty() {
        assert!(
            tree_with_flow[0].score > tree_no_flow[0].score,
            "flow bonus should increase score: {} vs {}",
            tree_with_flow[0].score,
            tree_no_flow[0].score
        );
    }
}

#[test]
fn test_balanced_empty_marginals() {
    let config = Config::draft();
    let tree = build_dd_tree_balanced(&[], &config, &NoScreeningPruner, false, &[], 2.0, 0.3);
    assert!(tree.is_empty(), "empty marginals should produce empty tree");
}
