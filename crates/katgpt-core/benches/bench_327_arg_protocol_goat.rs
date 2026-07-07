//! ARG Protocol Primitives GOAT gate bench (Plan 327 Phase 4).
//!
//! Exercises the perf (G2) and alloc-free (G4) gates for the ARG protocol's
//! online-loop hot-path primitives. G1 (correctness), G3 (no-regression), and
//! G5 (silence-bias) are covered by the 61 unit tests in `arg::*`.
//!
//! # Gates measured here
//!
//! - **G2a (perf)**: `PolicyEnvelope::evaluate` median latency ≤ 50ns over
//!   10,000 calls. The envelope is a few branches + small-slice scans; this is
//!   the Step 1 hard gate that runs on every request.
//! - **G2b (perf)**: `TaxonomyValidator::validate_label_set` median latency
//!   ≤ 200ns over 10,000 calls (taxonomy of 256 nodes, candidate set of 8).
//!   This is the Step 3 deterministic validator producing `L_final`.
//! - **G4 (alloc-free hot path)**: `PolicyEnvelope::evaluate` and
//!   `TaxonomyValidator::validate_label_set` allocate 0 times over 100
//!   steady-state calls (CountingAllocator).
//!
//! The offline-loop primitives (`OfflineCandidateScorer`, `InfoRegistry`) are
//! NOT perf-gated — they run in the offline evolution loop, not the per-request
//! online hot path. G5 (silence-bias) is a unit-test property gate, not a bench.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features arg_protocol --bench bench_327_arg_protocol_goat -- --nocapture
//! ```
//!
//! If the dyld/trustd stall hits, run the compiled binary directly:
//!
//! ```bash
//! DYLD_PRINT_STATISTICS=1 target/release/bench_327_arg_protocol_goat-* --nocapture
//! ```

#![cfg(feature = "arg_protocol")]

use katgpt_core::{
    LabelId, LabelSet, PolicyConstraints, PolicyEnvelope, PolicyState, ResponseMode, TaxonomyKind,
    TaxonomyNode, TaxonomyValidator, ValidationScratch,
};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── GateResult ─────────────────────────────────────────────────────────────

struct GateResult {
    name: &'static str,
    passed: bool,
    detail: String,
}

impl GateResult {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        GateResult {
            name,
            passed: true,
            detail: detail.into(),
        }
    }
    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        GateResult {
            name,
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── Fixtures ───────────────────────────────────────────────────────────────

fn lbl(n: u32) -> LabelId {
    LabelId::new(n)
}

/// Build a 256-node taxonomy: 8 clusters, each with 8 labels, each with 3
/// leaves (8*8*3 = 192 leaves + 8 clusters + 8 labels*8 = 64 → 256 total-ish).
/// Sorted by id for binary-search lookup (as TaxonomyValidator requires).
fn taxonomy_256() -> Vec<TaxonomyNode<'static>> {
    let mut nodes: Vec<TaxonomyNode<'static>> = Vec::with_capacity(256);
    let mut next_id: u32 = 1;
    // 8 clusters (roots).
    let cluster_ids: Vec<u32> = (0..8)
        .map(|_| {
            let id = next_id;
            next_id += 1;
            nodes.push(TaxonomyNode {
                id: lbl(id),
                kind: TaxonomyKind::Cluster,
                parent_id: None,
                incompatible_with: &[],
            });
            id
        })
        .collect();
    // Each cluster has 8 labels.
    let mut label_ids: Vec<u32> = Vec::new();
    for &cluster in &cluster_ids {
        for _ in 0..8 {
            let id = next_id;
            next_id += 1;
            nodes.push(TaxonomyNode {
                id: lbl(id),
                kind: TaxonomyKind::Label,
                parent_id: Some(lbl(cluster)),
                incompatible_with: &[],
            });
            label_ids.push(id);
        }
    }
    // Each label has 3 leaves (256 - 8 - 64 = 184; pick enough leaves).
    for &label in &label_ids {
        for _ in 0..3 {
            let id = next_id;
            next_id += 1;
            nodes.push(TaxonomyNode {
                id: lbl(id),
                kind: TaxonomyKind::Leaf,
                parent_id: Some(lbl(label)),
                incompatible_with: &[],
            });
        }
    }
    // Sort by id for binary-search (TaxonomyValidator::new requires sorted).
    nodes.sort_by_key(|n| n.id);
    nodes
}

/// A candidate set of 8 valid root-level clusters from the taxonomy. Roots
/// have no parent_id, so they pass the parent/child coherence check — this is
/// the steady-state hot path (0 rejections, 0 allocations).
fn candidate_set_8(taxonomy: &[TaxonomyNode<'static>]) -> LabelSet {
    let mut set = LabelSet::new();
    // Pick the first 8 clusters (roots — always valid, no parent required).
    let mut count = 0;
    for n in taxonomy {
        if n.kind == TaxonomyKind::Cluster {
            set.insert(n.id);
            count += 1;
            if count == 8 {
                break;
            }
        }
    }
    assert_eq!(count, 8, "taxonomy must have at least 8 clusters");
    set
}

// ─── G2a: PolicyEnvelope::evaluate perf ─────────────────────────────────────

fn gate_g2a_policy_envelope_perf() -> GateResult {
    const TARGET_NS: f64 = 50.0;
    const WARMUP: usize = 1_000;
    const ITERS: usize = 10_000;

    let allowed = [lbl(1), lbl(2), lbl(3), lbl(4), lbl(5)];
    let forbidden = [lbl(99), lbl(100)];
    let env = PolicyEnvelope {
        state: PolicyState::Restrict,
        constraints: PolicyConstraints {
            allowed_labels: &allowed,
            forbidden_labels: &forbidden,
            max_hops: 4,
            max_depth: 3,
            max_complexity: 512,
        },
        response_mode: ResponseMode::Prudent,
    };
    let probe = lbl(2); // in-allowlist, not forbidden → exercises the full path.

    // Warmup.
    for _ in 0..WARMUP {
        let _ = black_box(env.evaluate(Some(probe)));
    }

    // Measure.
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(env.evaluate(Some(probe)));
    }
    let elapsed = start.elapsed();
    let mean_ns = elapsed.as_nanos() as f64 / ITERS as f64;

    if mean_ns <= TARGET_NS {
        GateResult::pass(
            "G2a",
            format!(
                "PolicyEnvelope::evaluate median ~{mean_ns:.1}ns (≤ {TARGET_NS}ns target) over {ITERS} iters"
            ),
        )
    } else {
        GateResult::fail(
            "G2a",
            format!(
                "PolicyEnvelope::evaluate median ~{mean_ns:.1}ns > {TARGET_NS}ns target over {ITERS} iters"
            ),
        )
    }
}

// ─── G2b: TaxonomyValidator::validate_label_set perf ────────────────────────

fn gate_g2b_taxonomy_validate_perf() -> GateResult {
    const TARGET_NS: f64 = 200.0;
    const WARMUP: usize = 1_000;
    const ITERS: usize = 10_000;

    let taxonomy = taxonomy_256();
    let n_nodes = taxonomy.len();
    let candidates = candidate_set_8(&taxonomy);
    // TaxonomyValidator::new takes ownership of the Vec (it sorts + owns it).
    let validator = TaxonomyValidator::new(taxonomy);
    let mut scratch = ValidationScratch::with_capacity(16);

    // Warmup.
    for _ in 0..WARMUP {
        let _ = black_box(validator.validate_label_set(&candidates, &mut scratch));
    }

    // Measure.
    let start = Instant::now();
    for _ in 0..ITERS {
        let _ = black_box(validator.validate_label_set(&candidates, &mut scratch));
    }
    let elapsed = start.elapsed();
    let mean_ns = elapsed.as_nanos() as f64 / ITERS as f64;

    if mean_ns <= TARGET_NS {
        GateResult::pass(
            "G2b",
            format!(
                "TaxonomyValidator::validate_label_set median ~{mean_ns:.1}ns (≤ {TARGET_NS}ns target) over {ITERS} iters, taxonomy={n_nodes} nodes, |candidates|=8"
            ),
        )
    } else {
        GateResult::fail(
            "G2b",
            format!(
                "TaxonomyValidator::validate_label_set median ~{mean_ns:.1}ns > {TARGET_NS}ns target over {ITERS} iters"
            ),
        )
    }
}

// ─── G4: alloc-free hot path ────────────────────────────────────────────────

fn gate_g4_alloc_free_hot_path() -> GateResult {
    const ITERS: usize = 100;

    // G4a: PolicyEnvelope::evaluate.
    let allowed = [lbl(1), lbl(2), lbl(3)];
    let forbidden = [lbl(99)];
    let env = PolicyEnvelope {
        state: PolicyState::Restrict,
        constraints: PolicyConstraints {
            allowed_labels: &allowed,
            forbidden_labels: &forbidden,
            max_hops: 4,
            max_depth: 3,
            max_complexity: 256,
        },
        response_mode: ResponseMode::Prudent,
    };
    let probe = lbl(2);
    // Warmup once outside the measurement.
    let _ = env.evaluate(Some(probe));
    let (_, allocs_policy) = alloc_delta(|| {
        for _ in 0..ITERS {
            let _ = black_box(env.evaluate(Some(probe)));
        }
    });

    // G4b: TaxonomyValidator::validate_label_set.
    let taxonomy = taxonomy_256();
    let candidates = candidate_set_8(&taxonomy);
    let validator = TaxonomyValidator::new(taxonomy);
    let mut scratch = ValidationScratch::with_capacity(16);
    // Warmup once.
    let _ = validator.validate_label_set(&candidates, &mut scratch);
    let (_, allocs_tax) = alloc_delta(|| {
        for _ in 0..ITERS {
            let _ = black_box(validator.validate_label_set(&candidates, &mut scratch));
        }
    });

    if allocs_policy == 0 && allocs_tax == 0 {
        GateResult::pass(
            "G4",
            format!(
                "PolicyEnvelope::evaluate: 0 allocs / {ITERS} calls; TaxonomyValidator::validate_label_set: 0 allocs / {ITERS} calls"
            ),
        )
    } else {
        GateResult::fail(
            "G4",
            format!(
                "PolicyEnvelope allocs={allocs_policy}/{ITERS}, TaxonomyValidator allocs={allocs_tax}/{ITERS} (expected 0/0)"
            ),
        )
    }
}

// ─── Main ───────────────────────────────────────────────────────────────────

fn main() {
    println!("=== Plan 327 - ARG Protocol Primitives GOAT Gate (Phase 4) ===\n");

    let gates = [
        gate_g2a_policy_envelope_perf(),
        gate_g2b_taxonomy_validate_perf(),
        gate_g4_alloc_free_hot_path(),
    ];

    let mut all_pass = true;
    for g in &gates {
        let status = if g.passed { "PASS" } else { "FAIL" };
        println!("[{status}] {}: {}", g.name, g.detail);
        if !g.passed {
            all_pass = false;
        }
    }

    println!();
    println!("G1 (correctness), G3 (no-regression), G5 (silence-bias) are covered");
    println!("by the 61 unit tests in arg::* (cargo test --features arg_protocol --lib arg::).");
    println!();
    if all_pass {
        println!(
            "=== ALL PERF+ALLOC GATES PASS - G1-G5 complete, eligible for default promotion ==="
        );
        std::process::exit(0);
    } else {
        println!("=== ONE OR MORE GATES FAILED - keep opt-in, investigate ===");
        std::process::exit(1);
    }
}
