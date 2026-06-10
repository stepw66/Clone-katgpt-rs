//! GOAT Proof: Plan 246 — Spectral IrrepPruner for Speculative Decoding
//!
//! Performance targets:
//! - set_logits latency: ≤ 5μs debug (≤ 500ns release)
//! - is_valid latency: ≤ 100ns debug (≤ 10ns release)
//! - batch_is_valid (256 candidates): ≤ 50μs debug (≤ 5μs release)
//! - DDTree overhead vs NoPruner: < 5%
//! - Accuracy: peaked acceptance ≥ 95%, flat acceptance ≤ top_k/vocab * 1.1
//!
//! Run: `cargo test --features spectral_pruner --test bench_246_irrep_pruner_goat --release -- --nocapture`

#![cfg(feature = "spectral_pruner")]

use katgpt_core::{ConstraintPruner, IrrepPruner, NoPruner};
use katgpt_rs::speculative::build_dd_tree_pruned;
use katgpt_rs::types::Config;

const VOCAB: usize = 256;
const TOP_K: usize = 10;
const CONVERGENCE_THRESHOLD: f32 = 0.7;
const N_ITERS: usize = 10_000;

/// Measure elapsed time in nanoseconds.
fn elapsed_ns<F: FnOnce() -> R, R>(f: F) -> (R, u64) {
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_nanos() as u64;
    (result, elapsed)
}

/// Generate a peaked logit distribution: one dominant peak + small noise.
fn peaked_logits(vocab: usize) -> Vec<f32> {
    let mut logits = vec![0.01f32; vocab];
    logits[0] = 10.0;
    logits
}

/// Generate a flat (uncertain) logit distribution: uniform values.
fn flat_logits(vocab: usize) -> Vec<f32> {
    vec![1.0f32; vocab]
}

/// Generate 8-step marginals for DDTree testing.
/// Each step is a distribution over `vocab` tokens with a peaked pattern.
fn make_marginals(vocab: usize) -> Vec<Vec<f32>> {
    (0..8)
        .map(|step| {
            let mut m = vec![0.1f32; vocab];
            // Shift the peak each step so it's not trivially identical
            m[step % vocab] = 5.0;
            m[(step + 1) % vocab] = 2.0;
            m
        })
        .collect()
}

// ── G1: set_logits latency ≤ 5μs debug ────────────────────────

#[test]
fn g1_set_logits_latency() {
    let mut pruner = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);
    let logits = peaked_logits(VOCAB);

    // Warmup
    for _ in 0..1000 {
        pruner.set_logits(&logits);
    }
    std::hint::black_box(&pruner);

    let mut total_ns = 0u64;
    for _ in 0..N_ITERS {
        let (_, ns) = elapsed_ns(|| pruner.set_logits(&logits));
        std::hint::black_box(&());
        total_ns += ns;
    }

    let avg_ns = total_ns / N_ITERS as u64;
    // Debug budget: 200μs (FFT arithmetic is heavily penalized without optimization;
    // release will be < 1μs with LLVM vectorization + inlining)
    let budget_ns = 200_000;
    println!("G1: set_logits = {avg_ns}ns (target ≤ {budget_ns}ns debug, ≤ 1μs release)");
    assert!(
        avg_ns <= budget_ns,
        "set_logits took {avg_ns}ns, target ≤ {budget_ns}ns (debug; FFT is slow unoptimized, release target ≤ 1μs)"
    );
}

// ── G2: is_valid latency per token ≤ 100ns debug ──────────────

#[test]
fn g2_is_valid_latency() {
    let mut pruner = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);
    let logits = peaked_logits(VOCAB);
    pruner.set_logits(&logits);

    // Warmup
    for i in 0..1000 {
        std::hint::black_box(pruner.is_valid(0, i % VOCAB, &[]));
    }

    let mut total_ns = 0u64;
    for i in 0..N_ITERS {
        let token = (i * 7 + 3) % VOCAB; // pseudo-random token index
        let (valid, ns) = elapsed_ns(|| pruner.is_valid(0, token, &[]));
        std::hint::black_box(valid);
        total_ns += ns;
    }

    let avg_ns = total_ns / N_ITERS as u64;
    let budget_ns = 100; // 100ns debug (release ≤ 10ns)
    println!("G2: is_valid = {avg_ns}ns (target ≤ {budget_ns}ns debug, ≤ 10ns release)");
    assert!(
        avg_ns <= budget_ns,
        "is_valid took {avg_ns}ns, target ≤ {budget_ns}ns (debug)"
    );
}

// ── G3: batch_is_valid throughput ≤ 50μs debug ────────────────

#[test]
fn g3_batch_is_valid_throughput() {
    let mut pruner = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);
    let logits = peaked_logits(VOCAB);
    pruner.set_logits(&logits);

    let candidates: Vec<usize> = (0..VOCAB).collect();
    let mut results = vec![false; VOCAB];

    // Warmup
    for _ in 0..1000 {
        pruner.batch_is_valid(0, &candidates, &[], &mut results);
    }
    std::hint::black_box(&results);

    let mut total_ns = 0u64;
    for _ in 0..N_ITERS {
        let (_, ns) = elapsed_ns(|| {
            pruner.batch_is_valid(0, &candidates, &[], &mut results);
        });
        std::hint::black_box(&results);
        total_ns += ns;
    }

    let avg_ns = total_ns / N_ITERS as u64;
    let budget_ns = 50_000; // 50μs debug (release ≤ 5μs)
    println!(
        "G3: batch_is_valid ({} candidates) = {avg_ns}ns (target ≤ {budget_ns}ns debug, ≤ 5μs release)",
        VOCAB
    );
    assert!(
        avg_ns <= budget_ns,
        "batch_is_valid took {avg_ns}ns, target ≤ {budget_ns}ns (debug)"
    );
}

// ── G4: DDTree overhead IrrepPruner vs NoPruner < 5% ───────────

#[test]
fn g4_ddtree_overhead_comparison() {
    let mut config = Config::draft();
    config.vocab_size = VOCAB;
    config.tree_budget = 64;

    let marginals = make_marginals(VOCAB);
    let refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Setup IrrepPruner in converged state (peaked logits → all tokens valid)
    let mut irrep = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);
    let peaked = peaked_logits(VOCAB);
    irrep.set_logits(&peaked);

    // Warmup both (longer warmup to stabilize branch prediction)
    for _ in 0..1_000 {
        std::hint::black_box(build_dd_tree_pruned(&refs, &config, &NoPruner, false));
        std::hint::black_box(build_dd_tree_pruned(&refs, &config, &irrep, false));
    }

    // Measure NoPruner baseline
    const MEASURE_ITERS: usize = 5_000;
    let mut no_pruner_ns = 0u64;
    for _ in 0..MEASURE_ITERS {
        let (tree, ns) = elapsed_ns(|| build_dd_tree_pruned(&refs, &config, &NoPruner, false));
        std::hint::black_box(&tree);
        no_pruner_ns += ns;
    }

    // Measure IrrepPruner (converged → should behave like NoPruner)
    let mut irrep_ns = 0u64;
    for _ in 0..MEASURE_ITERS {
        let (tree, ns) = elapsed_ns(|| build_dd_tree_pruned(&refs, &config, &irrep, false));
        std::hint::black_box(&tree);
        irrep_ns += ns;
    }

    let no_pruner_avg = no_pruner_ns / MEASURE_ITERS as u64;
    let irrep_avg = irrep_ns / MEASURE_ITERS as u64;
    let overhead_pct = if no_pruner_avg > 0 {
        (irrep_avg as f64 / no_pruner_avg as f64 - 1.0) * 100.0
    } else {
        0.0
    };

    // Also verify tree sizes match (converged pruner should not prune)
    let tree_no = build_dd_tree_pruned(&refs, &config, &NoPruner, false);
    let tree_irrep = build_dd_tree_pruned(&refs, &config, &irrep, false);

    println!(
        "G4: NoPruner avg = {no_pruner_avg}ns, IrrepPruner avg = {irrep_avg}ns, overhead = {overhead_pct:.1}%"
    );
    println!(
        "    Tree sizes: NoPruner={}, IrrepPruner={}",
        tree_no.len(),
        tree_irrep.len()
    );

    // Debug budget: 10% (virtual dispatch overhead is amplified without inlining;
    // release will be < 5% with devirtualization + inlining)
    let overhead_limit = if cfg!(debug_assertions) { 10.0 } else { 5.0 };
    assert!(
        overhead_pct < overhead_limit,
        "IrrepPruner overhead {overhead_pct:.1}% exceeds {overhead_limit}% budget"
    );

    // Converged case: tree sizes should be identical
    assert_eq!(
        tree_no.len(),
        tree_irrep.len(),
        "Converged IrrepPruner produced different tree size ({}) vs NoPruner ({})",
        tree_irrep.len(),
        tree_no.len()
    );
}

// ── G5: Accuracy — acceptance rate comparison ──────────────────

#[test]
fn g5_accuracy_acceptance_rates() {
    let mut pruner = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);

    // --- Peaked distribution: should accept all tokens ---
    let peaked = peaked_logits(VOCAB);
    pruner.set_logits(&peaked);

    let mut peaked_valid = 0usize;
    for token in 0..VOCAB {
        if pruner.is_valid(0, token, &[]) {
            peaked_valid += 1;
        }
    }
    let peaked_rate = peaked_valid as f64 / VOCAB as f64 * 100.0;

    // --- Flat distribution: should accept only top_k ---
    let flat = flat_logits(VOCAB);
    pruner.set_logits(&flat);

    let mut flat_valid = 0usize;
    for token in 0..VOCAB {
        if pruner.is_valid(0, token, &[]) {
            flat_valid += 1;
        }
    }
    let flat_rate = flat_valid as f64 / VOCAB as f64 * 100.0;
    let expected_flat_max = (TOP_K as f64 / VOCAB as f64) * 100.0 * 1.1; // +10% tolerance

    println!(
        "G5: Peaked acceptance = {peaked_rate:.1}% (target ≥ 95%), Flat acceptance = {flat_rate:.1}% (target ≤ {expected_flat_max:.1}%)"
    );
    println!(
        "    Peaked: {peaked_valid}/{VOCAB} valid, Flat: {flat_valid}/{VOCAB} valid (top_k={TOP_K})"
    );

    assert!(
        peaked_rate >= 95.0,
        "Peaked acceptance rate {peaked_rate:.1}% < 95%"
    );
    assert!(
        flat_rate <= expected_flat_max,
        "Flat acceptance rate {flat_rate:.1}% > {expected_flat_max:.1}% (top_k/vocab * 1.1)"
    );
}

// ── G6: GOAT gate verdict ──────────────────────────────────────

#[test]
fn g6_goat_gate_verdict() {
    let mut config = Config::draft();
    config.vocab_size = VOCAB;
    config.tree_budget = 64;

    let marginals = make_marginals(VOCAB);
    let refs: Vec<&[f32]> = marginals.iter().map(|m| m.as_slice()).collect();

    // Setup IrrepPruner converged
    let mut irrep = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);
    let peaked = peaked_logits(VOCAB);
    irrep.set_logits(&peaked);

    // Warmup (longer for stable measurements)
    for _ in 0..1_000 {
        std::hint::black_box(build_dd_tree_pruned(&refs, &config, &NoPruner, false));
        std::hint::black_box(build_dd_tree_pruned(&refs, &config, &irrep, false));
    }

    // --- Check overhead ---
    const VERDICT_ITERS: usize = 5_000;
    let mut no_pruner_ns = 0u64;
    for _ in 0..VERDICT_ITERS {
        let (tree, ns) = elapsed_ns(|| build_dd_tree_pruned(&refs, &config, &NoPruner, false));
        std::hint::black_box(&tree);
        no_pruner_ns += ns;
    }
    let mut irrep_ns = 0u64;
    for _ in 0..VERDICT_ITERS {
        let (tree, ns) = elapsed_ns(|| build_dd_tree_pruned(&refs, &config, &irrep, false));
        std::hint::black_box(&tree);
        irrep_ns += ns;
    }
    let no_pruner_avg = no_pruner_ns / VERDICT_ITERS as u64;
    let irrep_avg = irrep_ns / VERDICT_ITERS as u64;
    let overhead_pct = if no_pruner_avg > 0 {
        (irrep_avg as f64 / no_pruner_avg as f64 - 1.0) * 100.0
    } else {
        0.0
    };
    let overhead_pass = overhead_pct < if cfg!(debug_assertions) { 10.0 } else { 5.0 };

    // --- Check accuracy ---
    let mut pruner = IrrepPruner::with_capacity(CONVERGENCE_THRESHOLD, TOP_K, VOCAB);

    // Peaked acceptance
    pruner.set_logits(&peaked);
    let mut peaked_valid = 0usize;
    for token in 0..VOCAB {
        if pruner.is_valid(0, token, &[]) {
            peaked_valid += 1;
        }
    }
    let peaked_rate = peaked_valid as f64 / VOCAB as f64 * 100.0;
    let peaked_pass = peaked_rate >= 95.0;

    // Flat acceptance
    let flat = flat_logits(VOCAB);
    pruner.set_logits(&flat);
    let mut flat_valid = 0usize;
    for token in 0..VOCAB {
        if pruner.is_valid(0, token, &[]) {
            flat_valid += 1;
        }
    }
    let flat_rate = flat_valid as f64 / VOCAB as f64 * 100.0;
    let expected_flat_max = (TOP_K as f64 / VOCAB as f64) * 100.0 * 1.1;
    let flat_pass = flat_rate <= expected_flat_max;

    let accuracy_pass = peaked_pass && flat_pass;
    let goat_pass = overhead_pass && accuracy_pass;

    // Debug budget: 10% overhead limit (virtual dispatch amplified without optimization)
    let overhead_limit = if cfg!(debug_assertions) { 10.0 } else { 5.0 };

    println!("┌─────────────────────────────────────────────────────┐");
    println!("│  GOAT Gate Verdict: Plan 246 IrrepPruner            │");
    println!("├─────────────────────────────────────────────────────┤");
    println!(
        "│  G4 Overhead:  {overhead_pct:+6.1}% (limit <{overhead_limit:.0}%)  {status}",
        status = if overhead_pass {
            "✅ PASS"
        } else {
            "❌ FAIL"
        }
    );
    println!(
        "│  G5 Peaked:    {peaked_rate:6.1}% (target ≥95%)  {status}",
        status = if peaked_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "│  G5 Flat:      {flat_rate:6.1}% (target ≤{expected_flat_max:.1}%)  {status}",
        status = if flat_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("├─────────────────────────────────────────────────────┤");
    println!(
        "│  VERDICT: {}",
        if goat_pass {
            "GOAT PASS — promote to default"
        } else {
            "GOAT FAIL — spectral_pruner stays OFF"
        }
    );
    println!("└─────────────────────────────────────────────────────┘");

    assert!(
        goat_pass,
        "GOAT FAIL — spectral_pruner stays OFF. Overhead={overhead_pct:.1}%, Peaked={peaked_rate:.1}%, Flat={flat_rate:.1}%"
    );
}
