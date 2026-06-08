//! DecisionTrace Demo — interpretable decision traces from DDTree exploration (Plan 209 T4.3).
//!
//! Demonstrates building and rendering decision traces showing applied rules,
//! rejected alternatives, and sigmoid-bounded confidence scores.
//!
//! Run: `cargo run --features decision_trace --example decision_trace_demo`

#[cfg(feature = "decision_trace")]
use katgpt_rs::pruners::{DecisionTraceBuilder, ExtractedRule};

// ── Helpers ──────────────────────────────────────────────────────

/// Small Rust-token vocabulary for human-readable output.
#[cfg(feature = "decision_trace")]
fn vocab() -> Vec<String> {
    vec![
        "fn".into(),
        "Result".into(),
        "match".into(),
        "if".into(),
        "Ok".into(),
        "Err".into(),
        "let".into(),
    ]
}

/// Shorthand to build a rule with conditions.
#[cfg(feature = "decision_trace")]
fn rule(
    conditions: Vec<(usize, usize)>,
    action: (usize, usize),
    score: f32,
    support: u32,
) -> ExtractedRule {
    ExtractedRule::new(conditions, action, score, support)
}

#[cfg(feature = "decision_trace")]
fn separator(title: &str) {
    println!("\n── {title} ──\n");
}

// ── Main ─────────────────────────────────────────────────────────

#[cfg(feature = "decision_trace")]
fn main() {
    let vocab = vocab();
    println!("=== DecisionTrace Demo (Plan 209 T4.3) ===");

    // ── Scenario 1: Normal trace with applied rules and rejected alternatives ──
    separator("Scenario 1: Normal Trace");

    let trace = DecisionTraceBuilder::new()
        .applied(rule(vec![(0, 0), (1, 1)], (2, 2), 0.85, 5)) // fn → Result → match
        .applied(rule(vec![(0, 0), (1, 1)], (2, 3), 0.72, 3)) // fn → Result → if
        .rejected(rule(vec![(0, 0), (1, 1)], (2, 5), 0.30, 1)) // fn → Result → Err (too low)
        .build();

    println!("{}", trace.to_string(&vocab));
    println!(
        "\n  rules_applied: {}, alternatives_rejected: {}, confidence: {:.4}",
        trace.rules_applied.len(),
        trace.alternatives_rejected.len(),
        trace.confidence,
    );

    // ── Scenario 2: Empty trace (miss path — no rules matched) ──────────────
    separator("Scenario 2: Empty Trace (Miss Path)");

    let empty = DecisionTraceBuilder::new().build();
    println!("{}", empty.to_string(&vocab));
    assert_eq!(
        empty.confidence, 0.0,
        "Empty trace should have zero confidence"
    );
    println!("  ✓ Confidence is 0.0 as expected");

    // ── Scenario 3: High-confidence single-rule trace ───────────────────────
    separator("Scenario 3: High-Confidence Trace");

    let high = DecisionTraceBuilder::new()
        .applied(rule(vec![(0, 6), (1, 0)], (2, 4), 4.8, 12)) // let → fn → Ok
        .build();

    println!("{}", high.to_string(&vocab));
    println!("  sigmoid({:.1}) = {:.4}", 4.8f32, high.confidence);
    assert!(
        high.confidence > 0.99,
        "High score should give confidence near 1.0"
    );

    // ── Scenario 4: Low-confidence trace with alternatives only ─────────────
    separator("Scenario 4: Low-Confidence Trace");

    let low = DecisionTraceBuilder::new()
        .applied(rule(vec![(0, 0)], (1, 2), -3.5, 2)) // fn → match (weak)
        .rejected(rule(vec![(0, 0)], (1, 3), -4.2, 1)) // fn → if (weaker)
        .build();

    println!("{}", low.to_string(&vocab));
    println!("  sigmoid({:.1}) = {:.4}", -3.5f32, low.confidence);
    assert!(
        low.confidence < 0.05,
        "Negative score should give low confidence"
    );

    // ── Scenario 5: Alternatives-only trace (no rules applied) ──────────────
    separator("Scenario 5: Alternatives Only (No Rules Applied)");

    let alt_only = DecisionTraceBuilder::new()
        .rejected(rule(vec![(0, 0), (1, 1)], (2, 2), 0.40, 1))
        .rejected(rule(vec![(0, 0), (1, 1)], (2, 3), 0.25, 1))
        .build();

    println!("{}", alt_only.to_string(&vocab));
    assert_eq!(alt_only.confidence, 0.0);

    // ── TL;DR ─────────────────────────────────────────────────────
    separator("TL;DR");
    println!("Demonstrated 5 scenarios:");
    println!("  1. Normal trace with 2 applied + 1 rejected");
    println!("  2. Empty trace (miss path, confidence = 0.0)");
    println!("  3. High-confidence trace (sigmoid(4.8) ≈ 1.0)");
    println!("  4. Low-confidence trace (sigmoid(-3.5) ≈ 0.03)");
    println!("  5. Alternatives-only trace (no rules applied)");
    println!("\nConfidence is sigmoid-bounded ∈ [0, 1], never softmax.");
}

#[cfg(not(feature = "decision_trace"))]
fn main() {
    eprintln!("This example requires the `decision_trace` feature.");
    eprintln!("Run: cargo run --example decision_trace_demo --features decision_trace");
}
