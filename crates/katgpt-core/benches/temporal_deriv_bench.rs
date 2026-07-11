//! Temporal Derivative Kernel microbenchmark (Plan 277 T1.10).
//!
//! Validates the GOAT gates:
//! - `observe` N=8 scalar/SIMD path: target <10ns/call on Apple Silicon arm64.
//! - 1000-NPC batch (1000 × N=8 kernels in a Vec): target <10µs total via
//!   rayon chunked iteration.
//!
//! # Run
//!
//! ```bash
//! cargo bench --bench temporal_deriv_bench --features temporal_deriv
//! ```
//!
//! # Feature gate
//!
//! Requires `temporal_deriv` (Plan 277 Phase 1).

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use katgpt_core::{TemporalDerivativeKernel, sigmoid_surprise_gate};
use rayon::prelude::*;

fn bench_observe(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_deriv/observe");

    for n in [1usize, 8, 16] {
        match n {
            1 => {
                let mut k: TemporalDerivativeKernel<1> = TemporalDerivativeKernel::new(0.3, 0.03);
                let signal = [0.5f32; 1];
                group.bench_function("n=1", |b| {
                    b.iter(|| {
                        let _ = k.observe(black_box(&signal));
                    });
                });
            }
            8 => {
                let mut k: TemporalDerivativeKernel<8> = TemporalDerivativeKernel::new(0.3, 0.03);
                let signal = [0.5f32; 8];
                group.bench_function("n=8", |b| {
                    b.iter(|| {
                        let _ = k.observe(black_box(&signal));
                    });
                });
            }
            16 => {
                let mut k: TemporalDerivativeKernel<16> = TemporalDerivativeKernel::new(0.3, 0.03);
                let signal = [0.5f32; 16];
                group.bench_function("n=16", |b| {
                    b.iter(|| {
                        let _ = k.observe(black_box(&signal));
                    });
                });
            }
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn bench_surprise_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_deriv/surprise_norm");

    let k: TemporalDerivativeKernel<8> = TemporalDerivativeKernel::new(0.3, 0.03);
    group.bench_function("n=8", |b| {
        b.iter(|| {
            black_box(k.surprise_norm());
        });
    });
    group.finish();
}

fn bench_sigmoid_surprise_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_deriv/sigmoid_surprise_gate");
    let deriv = [0.1f32; 8];
    group.bench_function("n=8_beta=4", |b| {
        b.iter(|| {
            black_box(sigmoid_surprise_gate(black_box(&deriv), 4.0));
        });
    });
    group.finish();
}

fn bench_batch_1000_npcs(c: &mut Criterion) {
    // 1000-NPC batch target: <10µs total via rayon chunked iteration.
    // Each NPC owns a kernel of N=8; we step all 1000 in one batch.
    use std::sync::Mutex;

    let mut group = c.benchmark_group("temporal_deriv/batch_1000_npcs");
    let n_npcs: usize = 1000;
    let kernels: Mutex<Vec<TemporalDerivativeKernel<8>>> = Mutex::new(
        (0..n_npcs)
            .map(|_| TemporalDerivativeKernel::new(0.3, 0.03))
            .collect(),
    );
    let signal = [0.5f32; 8];

    group.bench_function("rayon_par_iter_n=8", |b| {
        b.iter(|| {
            let mut kernels = kernels.lock().unwrap();
            kernels.par_iter_mut().for_each(|k| {
                let _ = k.observe(black_box(&signal));
            });
        });
    });

    group.bench_function("serial_iter_n=8", |b| {
        b.iter(|| {
            let mut kernels = kernels.lock().unwrap();
            for k in kernels.iter_mut() {
                let _ = k.observe(black_box(&signal));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_observe,
    bench_surprise_norm,
    bench_sigmoid_surprise_gate,
    bench_batch_1000_npcs,
);
criterion_main!(benches);
