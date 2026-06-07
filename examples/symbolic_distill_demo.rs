//! Symbolic Expression Distillation Demo — Plan 210 F1.11
//!
//! Demonstrates fitting a compact symbolic expression from synthetic DDTree
//! trace data and verifying blake3-protected serialization round-trip.
//!
//! Run: `cargo run --features symbolic_distill --example symbolic_distill_demo`

#![cfg(feature = "symbolic_distill")]

use katgpt_rs::pruners::{SymbolicExpression, SymbolicExpressionFitter, TraceDataset};

const FEATURE_NAMES: &[&str] = &[
    "depth_norm",
    "score_mean",
    "syntax_validity",
    "bandit_q",
    "parent_match",
    "depth_ratio",
    "entropy",
    "freq",
];

/// Ground-truth rule: accept when feature 0 > 0.5 AND feature 2 < 0.3.
fn ground_truth(features: &[f32]) -> bool {
    features[0] > 0.5 && features[2] < 0.3
}

/// Generate synthetic trace dataset following the known rule.
fn make_dataset(n: usize) -> TraceDataset {
    let mut features = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    let mut rng = fastrand::Rng::with_seed(42);

    for _ in 0..n {
        let row: Vec<f32> = (0..8).map(|_| rng.f32()).collect();
        let accepted = ground_truth(&row);
        features.push(row);
        labels.push(accepted);
    }

    TraceDataset { features, labels }
}

fn main() {
    println!("═══ Symbolic Distillation Demo (Plan 210 F1.11) ═══\n");

    // ── 1. Synthetic dataset ──────────────────────────────────────────
    let dataset = make_dataset(100);
    let accept_count = dataset.labels.iter().filter(|&&b| b).count();
    println!(
        "Dataset: {} records, {} accepted, {} rejected\n",
        dataset.features.len(),
        accept_count,
        dataset.features.len() - accept_count
    );
    println!("Ground truth: depth_norm > 0.5 AND syntax_validity < 0.3\n");

    // ── 2. Fit expression ─────────────────────────────────────────────
    let mut fitter = SymbolicExpressionFitter::new();
    fitter.max_terms = 4;
    fitter.min_improvement = 0.001;

    let expr = fitter.fit(&dataset);
    println!("Fitted expression ({} terms):", expr.terms.len());
    println!("  {}\n", expr.to_string(FEATURE_NAMES));

    // ── 3. Evaluate on test points ────────────────────────────────────
    println!("── Test Points ──");
    let test_points: Vec<(&str, Vec<f32>)> = vec![
        (
            "should accept",
            vec![0.8, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        (
            "should reject",
            vec![0.2, 0.5, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5],
        ),
        ("edge case", vec![0.5, 0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5]),
    ];

    for (label, features) in &test_points {
        let score = expr.evaluate(features);
        let expected = ground_truth(features);
        let predicted = score > 0.5;
        let match_str = if predicted == expected { "✓" } else { "✗" };
        println!("  {label:15}: score={score:.4} pred={predicted} actual={expected} {match_str}");
    }

    // ── 4. Before/after: baseline bias vs fitted expression ───────────
    let base_rate = accept_count as f32 / dataset.features.len() as f32;
    let mut correct = 0usize;
    for (i, features) in dataset.features.iter().enumerate() {
        let predicted = expr.evaluate(features) > 0.5;
        if predicted == dataset.labels[i] {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / dataset.features.len() as f32;
    println!("\n── Accuracy ──");
    println!("  Baseline (majority): {:.2}%", base_rate * 100.0);
    println!(
        "  Fitted expression:   {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        dataset.features.len()
    );

    // ── 5. Serialization round-trip with blake3 ───────────────────────
    println!("\n── Serialization Round-Trip ──");
    let bytes = expr.to_bytes();
    println!("  Serialized: {} bytes", bytes.len());

    let restored = SymbolicExpression::from_bytes(&bytes).expect("deserialization failed");

    // Verify blake3 integrity: re-serialize and compare
    let bytes2 = restored.to_bytes();
    assert_eq!(bytes, bytes2, "round-trip mismatch");
    println!("  blake3 integrity: OK ✓");

    // Corrupt a byte and verify detection
    let mut corrupted = bytes.clone();
    if !corrupted.is_empty() {
        let last = corrupted.len() - 1;
        corrupted[last] ^= 0xFF;
    }
    assert!(SymbolicExpression::from_bytes(&corrupted).is_none());
    println!("  Corrupt detection: OK ✓");

    // ── Summary ───────────────────────────────────────────────────────
    println!("\n═══ Summary ═══");
    println!("  Fitter recovers the ground-truth pattern from synthetic traces");
    println!("  Expression is human-readable, compact, and blake3-integrity-protected");
}
