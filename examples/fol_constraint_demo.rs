//! FOL Constraint Demo — Prompt→Constraint Extraction Pipeline (Plan 209, T6.1).
//!
//! Demonstrates the modelless FOL constraint extraction pipeline:
//! 1. Build a vocabulary of Rust tokens
//! 2. Extract `FolConstraint`s from natural-language prompts
//! 3. Wrap a `NoPruner` in `FolPruner` and show `is_valid()` behavior
//!
//! Run: `cargo run --features fol_constraints --example fol_constraint_demo`

#![cfg(feature = "fol_constraints")]

use katgpt_rs::pruners::{FolConstraint, FolPruner, extract_fol_constraints};
use katgpt_rs::speculative::{ConstraintPruner, NoPruner};

// ── Vocabulary ─────────────────────────────────────────────────────

/// 20 common Rust tokens used for constraint resolution.
fn build_vocab() -> Vec<String> {
    [
        "fn", "async", "pub", "struct", "enum", "match", "if", "else", "Result", "Option",
        "unsafe", "impl", "trait", "where", "return", "let", "mut", "ref", "self", "i32",
    ]
    .map(String::from)
    .to_vec()
}

// ── Helpers ────────────────────────────────────────────────────────

fn print_constraints(constraints: &[FolConstraint], vocab: &[String]) {
    if constraints.is_empty() {
        println!("    (no constraints extracted)");
        return;
    }
    for (i, c) in constraints.iter().enumerate() {
        let allowed: Vec<&str> = c.allowed.iter().map(|&idx| vocab[idx].as_str()).collect();
        let disallowed: Vec<&str> = c
            .disallowed
            .iter()
            .map(|&idx| vocab[idx].as_str())
            .collect();
        println!(
            "    [{i}] depth {:?} | allowed: {allowed:?} | disallowed: {disallowed:?} | conf: {:.3}",
            c.depth_range, c.confidence
        );
    }
}

// ── Main ───────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   FOL Constraint Demo — Prompt→Constraint Extraction       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let vocab = build_vocab();
    println!("Vocabulary ({} tokens): {:?}\n", vocab.len(), vocab);

    let prompts = [
        "Write an async function that returns a Result type",
        "Create a public struct with no unsafe code",
        "Implement a trait with match arms",
        "", // empty — miss path
    ];

    // ── Section 1: Constraint Extraction ──────────────────────────
    println!("{}", "═".repeat(62));
    println!("  Section 1: Extract Constraints from Prompts");
    println!("{}", "═".repeat(62));
    println!();

    let mut all_constraints: Vec<Vec<FolConstraint>> = Vec::new();

    for prompt in &prompts {
        let display = if prompt.is_empty() { "(empty)" } else { prompt };
        println!("  Prompt: \"{display}\"");

        let constraints = extract_fol_constraints(prompt, &vocab);
        println!("  Extracted {} constraint(s):", constraints.len());
        print_constraints(&constraints, &vocab);
        println!();

        all_constraints.push(constraints);
    }

    // ── Section 2: FolPruner is_valid() Behavior ──────────────────
    println!("{}", "═".repeat(62));
    println!("  Section 2: FolPruner — is_valid() Behavior");
    println!("{}", "═".repeat(62));
    println!();

    // Use prompt 0 ("async function ... Result") for live pruner demo
    let prompt = prompts[0];
    let pruner_constraints = extract_fol_constraints(prompt, &vocab);
    let pruner = FolPruner::new(NoPruner, pruner_constraints.clone());

    // Token indices in our vocab: async=1, fn=0, Result=8, unsafe=10
    let test_cases = [
        (0, "fn"),
        (1, "async"),
        (8, "Result"),
        (10, "unsafe"),
        (5, "match"),
    ];

    println!("  FolPruner from prompt: \"{prompt}\"");
    println!("  Constraints loaded: {}", pruner_constraints.len());
    println!();
    println!("  {:<10} {:<12} {:<10}", "Token", "Index", "is_valid");
    println!("  {}", "-".repeat(32));

    for (idx, name) in &test_cases {
        let valid = pruner.is_valid(0, *idx, &[]);
        println!(
            "  {:<10} {:<12} {}",
            name,
            idx,
            if valid { "true" } else { "false" }
        );
    }

    // ── Section 3: Empty Prompt — Miss Path ───────────────────────
    println!();
    println!("{}", "═".repeat(62));
    println!("  Section 3: Empty Prompt — Zero-Cost Miss Path");
    println!("{}", "═".repeat(62));
    println!();

    let empty_constraints = extract_fol_constraints("", &vocab);
    let empty_pruner = FolPruner::new(NoPruner, empty_constraints.clone());
    println!("  Constraints: {}", empty_constraints.len());
    println!(
        "  NoPruner.is_valid(0, 0, &[]) = {} (delegated to inner)",
        empty_pruner.is_valid(0, 0, &[])
    );

    // ── Summary ────────────────────────────────────────────────────
    println!();
    println!("{}", "═".repeat(62));
    println!("  Summary");
    println!("{}", "═".repeat(62));
    println!();
    println!("  ✓ extract_fol_constraints resolves prompt keywords to token indices");
    println!("  ✓ FolPruner wraps any ConstraintPruner with extracted constraints");
    println!("  ✓ Empty prompt → zero constraints → delegates to inner (miss path)");
    println!("  ✓ Negation patterns (\"no unsafe\") produce disallowed tokens");
    println!();
}

// TL;DR: Demonstrates the FOL constraint extraction pipeline — prompt keyword→token index resolution, FolPruner wrapping NoPruner with extracted constraints, and zero-cost miss path on empty prompts.
