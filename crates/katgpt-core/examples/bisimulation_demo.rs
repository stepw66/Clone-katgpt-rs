//! Bisimulation Operator Inference demo (Plan 324 T6.5).
//!
//! Builds a small Towers-of-Hanoi-style transition graph, refines it into a
//! bisimulation quotient, infers the abstract operator schema, plans a path,
//! and prints everything.
//!
//! # Run
//!
//! ```sh
//! cargo run --example bisimulation_demo \
//!     --features bisimulation_operator_inference --release
//! ```

#![cfg(feature = "bisimulation_operator_inference")]

use katgpt_core::bisimulation::{
    OperatorLabel, StateId, TransitionGraphBuilder, infer_operators, partition_refine, plan,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 324 — Bisimulation Operator Inference Demo            ║");
    println!("║  Source: arXiv:2602.19260 \"The Price Is Not Right\"          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── Build a small transition graph ──────────────────────────────────
    //
    // Synthetic " Towers of Hanoi"-like graph with 6 states and 3 operator
    // types. This is NOT a full Hanoi state space (that would be 27 states
    // for 3 disks × 3 pegs) — it's a small fixture that exercises all three
    // operator labels and produces a non-trivial quotient.
    //
    // State meanings (illustrative):
    //   s0: initial — both disks on peg A
    //   s1: small disk picked up from peg A
    //   s2: small disk placed on peg B (peg B was empty)
    //   s3: large disk picked up from peg A
    //   s4: large disk placed on peg C (peg C was empty)
    //   s5: goal — small disk placed on top of large disk on peg C

    let mut builder = TransitionGraphBuilder::new();
    builder.push_transition(StateId(0), StateId(1), OperatorLabel::PickTop);
    builder.push_transition(StateId(1), StateId(2), OperatorLabel::PlaceOnEmpty);
    builder.push_transition(StateId(2), StateId(3), OperatorLabel::PickTop);
    builder.push_transition(StateId(3), StateId(4), OperatorLabel::PlaceOnEmpty);
    builder.push_transition(StateId(4), StateId(5), OperatorLabel::PlaceOn);
    // Add a "reset" edge to make the graph cyclic (more interesting quotient).
    builder.push_transition(StateId(5), StateId(0), OperatorLabel::PickTop);

    let graph = builder.build();

    println!("── Input Transition Graph ─────────────────────────────────────");
    println!("States: {}", graph.n_states());
    println!("Edges:  {}", graph.n_edges());
    for e in graph.edges() {
        println!("  {:>3} --{:?}--> {:>3}", e.from, e.op, e.to);
    }
    println!();

    // ── Refine into a bisimulation quotient ─────────────────────────────
    let quotient = partition_refine(&graph);

    println!("── Bisimulation Quotient ──────────────────────────────────────");
    println!("Classes: {}", quotient.n_classes);
    println!("State → Class mapping:");
    for state in graph.states() {
        println!("  {:>3} → class {}", state, quotient.class_of(*state));
    }
    println!();
    println!("Quotient edges ({}):", quotient.quotient_edges.len());
    for e in &quotient.quotient_edges {
        println!("  class {:>2} --{:?}--> class {:>2}", e.from, e.op, e.to);
    }
    println!();
    println!("BLAKE3 commitment: {}", hex(&quotient.blake3));
    println!();

    // ── Infer operator schema ───────────────────────────────────────────
    let schema = infer_operators(&quotient);

    println!("── Inferred Operator Schema ───────────────────────────────────");
    println!("Operators: {}", schema.n_operators());
    for op in &schema.operators {
        println!("  {:?}:", op.label);
        println!("    preconditions: {:?}", op.preconditions);
        println!("    effects:       {:?}", op.effects);
    }
    println!();
    println!("Schema BLAKE3: {}", hex(&schema.blake3));
    println!();

    // ── Plan a path from class(0) to class(5) ───────────────────────────
    let start_class = quotient.class_of(StateId(0));
    let goal_class = quotient.class_of(StateId(5));

    println!(
        "── Plan: class({}) → class({}) ──────────────────────────────",
        start_class, goal_class
    );
    match plan(&schema, &quotient, start_class, goal_class) {
        Some(sequence) => {
            println!("Found plan ({} steps):", sequence.len());
            for (i, op) in sequence.iter().enumerate() {
                println!("  step {}: {:?}", i, op);
            }

            // Replay to verify.
            match schema.replay_plan(&quotient, start_class, &sequence, goal_class) {
                Ok(final_class) => {
                    println!();
                    println!("✅ Replay succeeded — landed on class {}", final_class);
                }
                Err(step) => {
                    println!();
                    println!("❌ Replay failed at step {}", step);
                }
            }
        }
        None => {
            println!(
                "❌ No path exists from class {} to class {}",
                start_class, goal_class
            );
        }
    }
    println!();

    // ── Summary ─────────────────────────────────────────────────────────
    println!("── Summary ────────────────────────────────────────────────────");
    println!(
        "Input:   {} states, {} edges",
        graph.n_states(),
        graph.n_edges()
    );
    println!(
        "Quotient: {} classes, {} edges (compaction: {:.1}×)",
        quotient.n_classes,
        quotient.quotient_edges.len(),
        graph.n_states() as f64 / quotient.n_classes as f64,
    );
    println!("Schema:  {} operators", schema.n_operators());
    println!();
    println!("This is the deterministic half of the NSM pipeline (arXiv:2508.21501).");
    println!("The heavier-weight half (ASP-based PDDL domain inference + diffusion");
    println!("skill training) is the CWM path (Plan 296) / riir-train respectively.");
}

/// Format a byte slice as a lowercase hex string.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
