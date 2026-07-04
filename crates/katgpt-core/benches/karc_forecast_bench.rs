//! KARC forecast hot-path benchmark — GOAT gate G2 (Plan 308).
//!
//! Target: `forecast_into` ≤ 500 ns/call at D=8, M=8, K=4 (d_h = 256 features,
//! the HLA-shaped config). This is the per-NPC per-tick forecast cost.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::{FourierBasis, KarcForecaster};

/// Fit a forecaster on a deterministic trajectory and return it + a seed delay state.
/// Returns the seed as a `Vec<f32>` because `[f32; K*D]` is not expressible in
/// stable Rust when K, D are const generics (`generic_const_exprs` is unstable).
fn make_fitted_forecaster<const D: usize, const M: usize, const K: usize>(
    basis: FourierBasis<M>,
    n_train: usize,
) -> (KarcForecaster<FourierBasis<M>, D, M, K>, Vec<f32>) {
    let traj: Vec<f32> = (0..n_train)
        .flat_map(|i| {
            let t = i as f32 * 0.05;
            let mut row = [0.0f32; 32];
            let n = D.min(32);
            for (d, row_d) in row.iter_mut().enumerate().take(n) {
                let freq = 0.3 + 0.2 * d as f32;
                *row_d = (freq * t).sin() + 0.5 * ((freq + 1.0) * t).cos();
            }
            row[..D].to_vec()
        })
        .collect();
    let n_total = traj.len() / D;
    let mut f = KarcForecaster::with_capacity(basis, n_total);
    let kd = K * D;
    for t in (K - 1)..(n_total - 1) {
        let mut delay = vec![0.0f32; kd];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        let mut target = vec![0.0f32; D];
        for d in 0..D {
            target[d] = traj[(t + 1) * D + d];
        }
        f.accumulate_pair(&delay, target.as_slice().try_into().unwrap());
    }
    f.fit_ridge(1e-6).expect("fit_ridge");
    // Seed delay state = last K observations.
    let mut seed = vec![0.0f32; kd];
    for lag in 0..K {
        let idx = (n_total - 1) - lag;
        for d in 0..D {
            seed[lag * D + d] = traj[idx * D + d];
        }
    }
    (f, seed)
}

fn bench_forecast(c: &mut Criterion) {
    let mut group = c.benchmark_group("karc_forecast_into");
    group.throughput(Throughput::Elements(1));

    // G2 target config: D=8, M=8, K=4 → d_h = 256.
    let (mut f_hla, seed_hla) = make_fitted_forecaster::<8, 8, 4>(FourierBasis::new(4.0), 500);
    let mut out_hla = [0.0f32; 8];
    group.bench_with_input(BenchmarkId::new("D8_M8_K4_dh256", "hla"), &seed_hla, |b, seed| {
        b.iter(|| {
            let ok = f_hla.forecast_into(black_box(seed), black_box(&mut out_hla));
            black_box(ok);
        });
    });

    // Smaller config for scaling reference: D=3, M=8, K=4 → d_h = 96 (double-scroll).
    let (mut f_ds, seed_ds) = make_fitted_forecaster::<3, 8, 4>(FourierBasis::new(4.0), 500);
    let mut out_ds = [0.0f32; 3];
    group.bench_with_input(BenchmarkId::new("D3_M8_K4_dh96", "double_scroll"), &seed_ds, |b, seed| {
        b.iter(|| {
            let ok = f_ds.forecast_into(black_box(seed), black_box(&mut out_ds));
            black_box(ok);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_forecast);
criterion_main!(benches);
