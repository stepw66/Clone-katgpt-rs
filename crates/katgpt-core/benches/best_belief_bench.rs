//! best_belief latency benchmark (Plan 336 Phase 2 T2.2 — G2 perf gate).
//!
//! Reports median latency for:
//! - `best_belief_score` across a representative `(S, F)` grid (target ≤ 100 ns).
//! - `select_best_belief` on 4 and 8 candidates (target ≤ 500 ns).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=target_plan336 \
//!   cargo bench -p katgpt-core --features best_belief --bench best_belief_bench \
//!     -- --warm-up-time 0.5 --measurement-time 1.5 --sample-size 30
//! ```

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::best_belief::{best_belief_score, select_best_belief};

fn bench_score(c: &mut Criterion) {
    let cases: &[(u32, u32)] = &[
        (0, 0),
        (1, 1),
        (10, 1),
        (100, 90),
        (50, 50),
        (1000, 1000),
    ];
    let mut g = c.benchmark_group("best_belief_score");
    for &(s, f) in cases {
        g.bench_with_input(
            BenchmarkId::from_parameter(format!("S{s}_F{f}")),
            &(s, f),
            |b, &(s, f)| {
                b.iter(|| {
                    black_box(best_belief_score(
                        black_box(s),
                        black_box(f),
                        black_box(0.05),
                    ))
                })
            },
        );
    }
    g.finish();
}

fn bench_select(c: &mut Criterion) {
    let candidates_8: Vec<(u32, u32)> = vec![
        (10, 1),
        (5, 2),
        (100, 90),
        (3, 0),
        (20, 15),
        (8, 8),
        (50, 49),
        (1, 10),
    ];
    let candidates_4 = &candidates_8[..4];
    c.bench_function("select_4_candidates", |b| {
        b.iter(|| {
            black_box(select_best_belief(
                black_box(candidates_4),
                black_box(0.05),
                black_box(Some(0)),
            ))
        })
    });
    c.bench_function("select_8_candidates", |b| {
        b.iter(|| {
            black_box(select_best_belief(
                black_box(&candidates_8),
                black_box(0.05),
                black_box(Some(0)),
            ))
        })
    });
}

criterion_group!(benches, bench_score, bench_select);
criterion_main!(benches);
