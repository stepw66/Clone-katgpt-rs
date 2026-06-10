//! DDTree Path → Logical Rule Extraction Demo (Plan 209, T2.6)
//!
//! Demonstrates extracting FOL-like logical rules from DDTree paths:
//! - Build a sample TreeNode tree manually
//! - Extract top-K rules via RuleExtractor
//! - Print each rule with conditions, action, score, support
//! - Show deduplication merging similar paths
//!
//! Run: `cargo run --features rule_extraction --example rule_extraction_demo`

#[cfg(feature = "rule_extraction")]
use katgpt_rs::pruners::{ExtractedRule, RuleExtractor, TreeNode, deduplicate_rules};

// ── Tree construction ──────────────────────────────────────────

#[cfg(feature = "rule_extraction")]
fn build_demo_tree() -> TreeNode {
    // Leaf A1: high-score path Root→A→A1
    let a1 = TreeNode {
        depth: 2,
        token_idx: 5,
        score: 0.9,
        children: vec![],
    };
    // Leaf A2: below threshold
    let a2 = TreeNode {
        depth: 2,
        token_idx: 7,
        score: 0.4,
        children: vec![],
    };
    // Child A
    let child_a = TreeNode {
        depth: 1,
        token_idx: 3,
        score: 0.8,
        children: vec![a1, a2],
    };

    // Leaf B1: similar action (token 5) to A1 — should deduplicate
    let b1 = TreeNode {
        depth: 2,
        token_idx: 5,
        score: 0.85,
        children: vec![],
    };
    // Child B
    let child_b = TreeNode {
        depth: 1,
        token_idx: 4,
        score: 0.7,
        children: vec![b1],
    };

    // Leaf C1: distinct path
    let c1 = TreeNode {
        depth: 2,
        token_idx: 8,
        score: 0.6,
        children: vec![],
    };
    // Child C
    let child_c = TreeNode {
        depth: 1,
        token_idx: 6,
        score: 0.5,
        children: vec![c1],
    };

    // Root
    TreeNode {
        depth: 0,
        token_idx: 1,
        score: 1.0,
        children: vec![child_a, child_b, child_c],
    }
}

#[cfg(feature = "rule_extraction")]
fn print_tree(node: &TreeNode, indent: &str) {
    let branch = if indent.is_empty() { "" } else { "├── " };
    println!(
        "{indent}{branch}depth={} tok={} score={:.2}",
        node.depth, node.token_idx, node.score
    );
    for child in &node.children {
        print_tree(child, &format!("{indent}    "));
    }
}

#[cfg(feature = "rule_extraction")]
fn print_rule(idx: usize, rule: &ExtractedRule) {
    let conds: Vec<String> = rule
        .conditions
        .iter()
        .map(|(d, t)| format!("(d={},t={})", d, t))
        .collect();
    println!(
        "  Rule {}: {} → action {} | score={:.4} support={}",
        idx + 1,
        conds.join(" ∧ "),
        format_args!("(d={},t={})", rule.action.0, rule.action.1),
        rule.score,
        rule.support,
    );
}

// ── Main ───────────────────────────────────────────────────────

#[cfg(feature = "rule_extraction")]
fn main() {
    println!("=== DDTree Rule Extraction Demo ===\n");

    // 1. Build sample tree
    println!("--- Sample Tree ---");
    let tree = build_demo_tree();
    print_tree(&tree, "");
    println!();

    // 2. Extract rules (top-3, min_score=0.3)
    println!("--- Extraction (top_k=3, min_score=0.3) ---");
    let extractor = RuleExtractor::new(3, 0.3);
    let mut rules = extractor.extract(std::slice::from_ref(&tree));
    println!("Extracted {} rules:\n", rules.len());

    for (i, rule) in rules.iter().enumerate() {
        print_rule(i, rule);
    }
    println!();

    // 3. Deduplicate with Hamming threshold=0 (exact condition match)
    println!("--- Deduplication (hamming_threshold=0) ---");
    println!("Rules before dedup: {}", rules.len());
    deduplicate_rules(&mut rules, 0);
    println!("Rules after dedup:  {}", rules.len());
    println!();

    for (i, rule) in rules.iter().enumerate() {
        print_rule(i, rule);
    }
    println!();

    // 4. Show deduplication with relaxed threshold
    println!("--- Relaxed Dedup (hamming_threshold=1) ---");
    let mut rules2 = extractor.extract(std::slice::from_ref(&tree));
    deduplicate_rules(&mut rules2, 1);
    println!("After relaxed dedup: {} rules\n", rules2.len());

    for (i, rule) in rules2.iter().enumerate() {
        print_rule(i, rule);
    }

    println!("\n=== Done ===");
}

#[cfg(not(feature = "rule_extraction"))]
fn main() {
    eprintln!("This example requires the `rule_extraction` feature.");
    eprintln!("Run: cargo run --example rule_extraction_demo --features rule_extraction");
}
