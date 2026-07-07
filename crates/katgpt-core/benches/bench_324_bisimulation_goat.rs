//! Plan 324 Phase 4 T4.4/T4.5 — Bisimulation GOAT gate latency bench.
//!
//! Measures:
//! - **G4** `partition_refine` latency on N ∈ {64, 256, 1024, 4096}
//!   random-transition graphs. Target: ≤ 1 ms for N=1024 on Apple Silicon
//!   arm64 release build.
//! - **G4 support** `infer_operators` latency on the resulting quotients.
//! - **G5 support** `class_of` lookup throughput (target ≥ 100M lookups/sec).
//!
//! Convention: `std::time::Instant` + `harness = false` (mirrors
//! `salience_tri_gate_bench.rs`, no Criterion dev-dep).
//!
//! Run:
//! ```bash
//! cargo run --release --bench bench_324_bisimulation_goat \
//!     --features bisimulation_operator_inference
//! ```

#![cfg(feature = "bisimulation_operator_inference")]

use katgpt_core::bisimulation::{
    BisimulationQuotient, OperatorSchema, TransitionGraphBuilder, infer_operators, partition_refine,
};
use std::time::{Duration, Instant};

// ─── Config ────────────────────────────────────────────────────────────────

/// Graph sizes to sweep for the G4 latency gate.
const SIZES: &[usize] = &[64, 256, 1024, 4096];

/// Average out-degree for random graphs. Sparse enough to be realistic
/// (Hanoi-style state spaces have small branching factors), dense enough to
/// stress the signature computation.
const AVG_DEGREE: usize = 3;

/// Warmup iterations.
const WARMUP: usize = 10;

/// Number of timed runs to take the median over.
const TIMED_RUNS: usize = 20;

// ─── Deterministic LCG (matches the crate convention) ─────────────────────

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// Build a random transition graph with `n_states` states and ~`avg_degree`
/// outgoing edges per state. Operator labels cycle through the 3 named
/// variants (PickTop, PlaceOn, PlaceOnEmpty) to produce a realistic label
/// distribution.
fn build_random_graph(n_states: usize, avg_degree: usize, seed: u64) -> TransitionGraphBuilder {
    let mut rng = Lcg::new(seed);
    let mut builder = TransitionGraphBuilder::with_capacity(n_states, n_states * avg_degree);
    let labels = [
        katgpt_core::bisimulation::OperatorLabel::PickTop,
        katgpt_core::bisimulation::OperatorLabel::PlaceOn,
        katgpt_core::bisimulation::OperatorLabel::PlaceOnEmpty,
    ];
    for from in 0..n_states as u32 {
        let n_edges = 1 + (rng.next() as usize % (avg_degree * 2));
        for _ in 0..n_edges {
            let to = (rng.next() % n_states as u64) as u32;
            let op = labels[(rng.next() as usize) % labels.len()];
            builder.push_transition(
                katgpt_core::bisimulation::StateId(from),
                katgpt_core::bisimulation::StateId(to),
                op,
            );
        }
    }
    builder
}

// ─── Timing helpers ────────────────────────────────────────────────────────

/// Measure `partition_refine` latency: median of `TIMED_RUNS` runs after
/// `WARMUP` warmup iterations. Returns per-run duration.
fn bench_partition_refine(n_states: usize, seed: u64) -> Duration {
    // Build the graph once (graph construction is NOT what we're measuring).
    let graph = build_random_graph(n_states, AVG_DEGREE, seed).build();

    // Warmup.
    for _ in 0..WARMUP {
        let _ = partition_refine(&graph);
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let q = partition_refine(&graph);
        samples.push(t0.elapsed());
        // Prevent the compiler from eliding the call.
        if q.blake3 == [0xff; 32] {
            std::process::abort();
        }
    }
    samples.sort();
    samples[TIMED_RUNS / 2]
}

/// Measure `infer_operators` latency on a pre-computed quotient.
fn bench_infer_operators(n_states: usize, seed: u64) -> (Duration, BisimulationQuotient) {
    let graph = build_random_graph(n_states, AVG_DEGREE, seed).build();
    let quotient = partition_refine(&graph);

    for _ in 0..WARMUP {
        let _ = infer_operators(&quotient);
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let t0 = Instant::now();
        let schema = infer_operators(&quotient);
        samples.push(t0.elapsed());
        if schema.blake3 == [0xff; 32] {
            std::process::abort();
        }
    }
    samples.sort();
    (samples[TIMED_RUNS / 2], quotient)
}

/// Measure `class_of` lookup throughput. Target: ≥ 100M lookups/sec.
fn bench_class_of_throughput(quotient: &BisimulationQuotient) -> f64 {
    let n = quotient.n_states();
    if n == 0 {
        return f64::INFINITY;
    }
    const BATCH: usize = 100_000;
    const OUTER: usize = 50;

    // Warmup.
    let mut sink: u64 = 0;
    for i in 0..WARMUP * BATCH {
        let state = katgpt_core::bisimulation::StateId((i % n) as u32);
        let class = quotient.class_of(state);
        sink = sink.wrapping_add(class.0 as u64);
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(OUTER);
    for i in 0..OUTER {
        let t0 = Instant::now();
        for j in 0..BATCH {
            let idx = (i * BATCH + j) % n;
            let class = quotient.class_of(katgpt_core::bisimulation::StateId(idx as u32));
            sink = sink.wrapping_add(class.0 as u64);
        }
        samples.push(t0.elapsed());
    }
    if sink == u64::MAX {
        std::process::abort();
    }
    samples.sort();
    let median_batch = samples[OUTER / 2];
    let per_lookup_ns = median_batch.as_nanos() as f64 / (BATCH as f64);
    1e9 / per_lookup_ns // lookups/sec
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Plan 324 — Bisimulation Operator Inference GOAT Gate       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "Graph config: avg out-degree = {}, {} timed runs (median), seed = 42",
        AVG_DEGREE, TIMED_RUNS
    );
    println!();

    // ── G4: partition_refine latency ────────────────────────────────────
    println!("── G4: partition_refine latency ────────────────────────────────");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>8}  {:>8}",
        "N_states", "refine_time", "infer_time", "n_class", "n_ops"
    );
    println!("{}", "-".repeat(60));

    let mut g4_1024_passes = false;
    for &n in SIZES {
        let refine_dur = bench_partition_refine(n, 42);
        let (infer_dur, quotient) = bench_infer_operators(n, 42);
        let n_classes = quotient.n_classes;
        let schema: OperatorSchema = infer_operators(&quotient);
        let n_ops = schema.n_operators();

        let refine_str = format_duration(refine_dur);
        let infer_str = format_duration(infer_dur);

        if n == 1024 && refine_dur.as_millis() <= 1 {
            g4_1024_passes = true;
        }

        println!(
            "{:>8}  {:>12}  {:>12}  {:>8}  {:>8}",
            n, refine_str, infer_str, n_classes, n_ops
        );
    }
    println!();
    let g4_status = if g4_1024_passes {
        "✅ PASS"
    } else {
        "❌ FAIL"
    };
    println!("G4 (partition_refine ≤ 1ms @ N=1024): {}", g4_status);
    println!();

    // ── G5: class_of throughput ─────────────────────────────────────────
    println!("── G5: class_of lookup throughput ──────────────────────────────");
    let (_, quotient_1024) = bench_infer_operators(1024, 42);
    let throughput = bench_class_of_throughput(&quotient_1024);
    let g5_status = if throughput >= 1e8 {
        "✅ PASS"
    } else {
        "❌ FAIL"
    };
    println!(
        "class_of throughput @ N=1024: {:.2}M lookups/sec (target ≥ 100M)  {}",
        throughput / 1e6,
        g5_status
    );
    println!();

    // ── Summary ─────────────────────────────────────────────────────────
    println!("── Summary ────────────────────────────────────────────────────");
    println!("G4 latency:        {}", g4_status);
    println!("G5 class_of tput:  {}", g5_status);
    if !g4_1024_passes || throughput < 1e8 {
        std::process::exit(1);
    }
}

fn format_duration(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{}ns", ns)
    } else if ns < 1_000_000 {
        format!("{:.1}µs", ns as f64 / 1_000.0)
    } else {
        format!("{:.2}ms", ns as f64 / 1_000_000.0)
    }
}
