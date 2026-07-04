//! Heterogeneous Velocity-Field Ensemble zero-allocation test — GOAT gate G3
//! (Plan 376 Phase 4).
//!
//! `HeterogeneousEnsemble::eval_into` and `fit_into` must not allocate heap
//! memory after the ensemble + scratch are constructed. The transport step
//! (project to spectral + reconstruct at D) is the new hot-path concern — it
//! must not allocate when fields have different k values.
//!
//! We verify with the same CountingAllocator pattern as the homogeneous
//! `tests/velocity_field_ensemble_alloc_check.rs`: snapshot ALLOC_COUNT before
//! and after the hot loop, assert delta is 0. Single test function for
//! serial execution within the test binary.

#![cfg(feature = "velocity_field_ensemble_heterogeneous")]

use katgpt_core::cross_resolution::CrossResolutionBases;
use katgpt_core::velocity_field_ensemble::{
    HeterogeneousEnsemble, HeterogeneousEntry, HeterogeneousFitScratch,
    HeterogeneousVelocityField,
};
use std::sync::atomic::Ordering;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

const D: usize = 8;
const P: usize = 3;
const INPUT_DIM: usize = 8;
const N_PAIRS: usize = 50;

/// Linear field with runtime native_dim: `b(x) = W · x` where W is
/// `native_dim × INPUT_DIM` row-major. Distinct per field for non-singular Gram.
struct LinearNativeField {
    w: Vec<f32>,
    native_dim: usize,
    id: u64,
}

impl HeterogeneousVelocityField for LinearNativeField {
    fn eval_native_into(&self, x: &[f32], out_native: &mut [f32]) {
        debug_assert_eq!(x.len(), INPUT_DIM);
        debug_assert_eq!(out_native.len(), self.native_dim);
        for (r, out_slot) in out_native.iter_mut().enumerate().take(self.native_dim) {
            let row = &self.w[r * INPUT_DIM..(r + 1) * INPUT_DIM];
            let mut acc = 0.0f32;
            for c in 0..INPUT_DIM {
                acc += row[c] * x[c];
            }
            *out_slot = acc;
        }
    }
    fn native_dim(&self) -> usize {
        self.native_dim
    }
    fn field_id(&self) -> u64 {
        self.id
    }
}

/// Build "pad-to-D" bases: identity phi_src (k = d_src) + psi_dst that places
/// the first d_src coords of spectral into the first d_src coords of d_dst,
/// zero-padding the rest. NOT orthonormal at d_dst > d_src but sufficient for
/// the alloc check (we don't verify transport correctness here — only that it
/// doesn't allocate).
fn pad_bases(d_src: usize, d_dst: usize) -> CrossResolutionBases {
    assert!(d_src <= d_dst);
    let k = d_src;
    let mut phi_src = vec![0.0f32; d_src * k];
    for r in 0..d_src {
        phi_src[r * k + r] = 1.0;
    }
    let mut psi_dst = vec![0.0f32; d_dst * k];
    for r in 0..d_src {
        psi_dst[r * k + r] = 1.0;
    }
    CrossResolutionBases::new(phi_src, psi_dst, d_src, d_dst, k).unwrap()
}

fn build_field(id: u64, native_dim: usize, scale: f32) -> LinearNativeField {
    let mut w = vec![0.0f32; native_dim * INPUT_DIM];
    for r in 0..native_dim {
        w[r * INPUT_DIM + r] = scale;
    }
    LinearNativeField { w, native_dim, id }
}

#[test]
fn g3_eval_and_fit_zero_alloc_after_warmup() {
    // Three fields with different native dims (4, 6, 8) — exercises the
    // per-field transport with varying k values.
    let f0 = build_field(0, 4, 0.5);
    let f1 = build_field(1, 6, 0.3);
    let f2 = build_field(2, 8, 0.2);

    let b0 = pad_bases(4, D);
    let b1 = pad_bases(6, D);
    let b2 = pad_bases(8, D);

    let entries = [
        HeterogeneousEntry::new(Box::new(f0), b0),
        HeterogeneousEntry::new(Box::new(f1), b1),
        HeterogeneousEntry::new(Box::new(f2), b2),
    ];
    let mut ens = HeterogeneousEnsemble::<P, D>::new(entries);
    let mut scratch = HeterogeneousFitScratch::<P, D>::new(&ens.entries);

    // Build N_PAIRS synthetic pairs.
    let xs: Vec<Vec<f32>> = (0..N_PAIRS)
        .map(|i| {
            (0..INPUT_DIM)
                .map(|k| ((i + k) as f32) * 0.01)
                .collect()
        })
        .collect();
    let ys: Vec<Vec<f32>> = (0..N_PAIRS)
        .map(|i| {
            (0..D)
                .map(|k| ((i * 2 + k) as f32) * 0.005)
                .collect()
        })
        .collect();
    let x_refs: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
    let y_refs: Vec<&[f32]> = ys.iter().map(|v| v.as_slice()).collect();

    // Warmup fit + eval.
    ens.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);
    let mut out = [0.0f32; D];
    for _ in 0..10 {
        ens.eval_into(&xs[0], &mut out, &mut scratch);
    }

    // ── Part 1: eval_into zero-alloc ────────────────────────────────────────
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const N_CALLS: usize = 1000;
    let mut total: f32 = 0.0;
    for i in 0..N_CALLS {
        let x = &xs[i % N_PAIRS];
        ens.eval_into(x, &mut out, &mut scratch);
        total += out[0];
    }
    std::hint::black_box(total);

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: HeterogeneousEnsemble::eval_into allocated {} times in {} calls",
        alloc_delta, N_CALLS
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: HeterogeneousEnsemble::eval_into deallocated {} times in {} calls",
        dealloc_delta, N_CALLS
    );

    // ── Part 2: fit_into zero-alloc ─────────────────────────────────────────
    let alloc_before = ALLOC_COUNT.load(Ordering::Relaxed);
    let dealloc_before = DEALLOC_COUNT.load(Ordering::Relaxed);

    const N_FITS: usize = 10;
    for _ in 0..N_FITS {
        ens.fit_into(&x_refs, &y_refs, 1e-4, &mut scratch);
    }

    let alloc_delta = ALLOC_COUNT.load(Ordering::Relaxed) - alloc_before;
    let dealloc_delta = DEALLOC_COUNT.load(Ordering::Relaxed) - dealloc_before;

    assert_eq!(
        alloc_delta, 0,
        "G3 FAIL: HeterogeneousEnsemble::fit_into allocated {} times in {} calls",
        alloc_delta, N_FITS
    );
    assert_eq!(
        dealloc_delta, 0,
        "G3 FAIL: HeterogeneousEnsemble::fit_into deallocated {} times in {} calls",
        dealloc_delta, N_FITS
    );
}
