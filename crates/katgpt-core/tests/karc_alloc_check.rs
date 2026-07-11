//! KARC zero-allocation test — GOAT gate G3 (Plan 308).
//!
//! `forecast_into` must not allocate heap memory after the forecaster is fit
//! (the feature buffer is stack-local `[f32; K·D·M]`). We verify this with a
//! manual `GlobalAlloc` counter — no `cargo-allocations` dependency required
//! (per the plan brief, which explicitly allows this fallback).

use katgpt_core::{FourierBasis, KarcForecaster};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

#[test]
fn g3_forecast_into_zero_alloc_after_warmup() {
    const D: usize = 8; // HLA-shaped config per Plan T1.11
    const K: usize = 4;
    const M: usize = 8;
    // d_h = K·D·M = 256 features.

    // Build a deterministic synthetic trajectory and fit. Multi-frequency,
    // incommensurate components so basis-expanded features span the full space.
    let traj: Vec<f32> = (0..1000)
        .flat_map(|i| {
            let t = i as f32 * 0.073;
            vec![
                (0.7 * t).sin() + 0.3 * (2.1 * t).cos(),
                (1.3 * t).cos() + 0.2 * (0.6 * t).sin(),
                (1.9 * t).sin() + 0.4 * (3.3 * t).cos(),
                (0.4 * t).cos() + 0.5 * (2.7 * t).sin(),
                (1.1 * t).sin() + 0.2 * (4.1 * t).cos(),
                (0.9 * t).cos() + 0.3 * (1.7 * t).sin(),
                (2.3 * t).sin() + 0.1 * (0.3 * t).cos(),
                (1.5 * t).cos() + 0.4 * (3.9 * t).sin(),
            ]
        })
        .collect();

    let n_total = traj.len() / D;
    let mut forecaster: KarcForecaster<FourierBasis<M>, D, M, K> =
        KarcForecaster::with_capacity(FourierBasis::new(4.0), n_total);
    for t in (K - 1)..(n_total - 1) {
        let mut delay = [0.0f32; K * D];
        for lag in 0..K {
            let idx = t - lag;
            for d in 0..D {
                delay[lag * D + d] = traj[idx * D + d];
            }
        }
        let mut target = [0.0f32; D];
        for d in 0..D {
            target[d] = traj[(t + 1) * D + d];
        }
        forecaster.accumulate_pair(&delay, &target);
    }
    forecaster.fit_ridge(1e-4).expect("fit_ridge");
    assert!(forecaster.is_fitted());

    // Warmup: a few forecasts to settle any lazy allocations (e.g. from the
    // SIMD dispatcher's Once init, stdout buffers, etc.).
    let mut delay_seed = [0.0f32; K * D];
    for lag in 0..K {
        let idx = (n_total - 1) - lag;
        for d in 0..D {
            delay_seed[lag * D + d] = traj[idx * D + d];
        }
    }
    let mut out = [0.0f32; D];
    for _ in 0..10 {
        let _ = forecaster.forecast_into(&delay_seed, &mut out);
    }

    // Measure: snapshot alloc/dealloc counts, run N forecasts, expect zero delta.
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const N_CALLS: usize = 1000;
    let mut total: f32 = 0.0;
    for _ in 0..N_CALLS {
        let ok = forecaster.forecast_into(&delay_seed, &mut out);
        assert!(ok, "forecast_into returned false after fit");
        total += out[0]; // sink — prevent the loop being optimised away
    }

    let alloc_after = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_after = DEALLOC_COUNT.load(Ordering::Relaxed);
    let alloc_delta = alloc_after - alloc_before;
    let dealloc_delta = dealloc_after - dealloc_before;

    // The sink must be observable so the compiler doesn't drop the loop.
    std::hint::black_box(total);

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: forecast_into allocated {} times in {} calls",
        alloc_delta, N_CALLS
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: forecast_into deallocated {} times in {} calls",
        dealloc_delta, N_CALLS
    );
}
