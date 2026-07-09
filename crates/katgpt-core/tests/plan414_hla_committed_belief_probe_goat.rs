//! Plan 414 — HLA Committed-Belief π-Sensitivity Probe G4+G5 GOAT gate.
//!
//! G4: `committed_blend_pi_sensitivity` allocates ZERO times over 1000 calls
//!     (fixed `[f32; N]`, `[f32; D]`, `[f32; 8]` arrays — no heap).
//! G5: p50 latency < 5µs at N=3, D=32, k_draws=8 (generous diagnostic budget).
//!
//! Separate binary from the lib unit tests because the `CountingAllocator`
//! global (`#[global_allocator]`) cannot live inside a `#[cfg(test)] mod` in
//! the lib — it would conflict with the test harness. Follows the
//! `linking_fold_alloc_check.rs` pattern.

#![cfg(feature = "hla_committed_belief_probe")]

use katgpt_core::committed_field_blend::pi_sensitivity::*;
use katgpt_core::committed_field_blend::{ArchetypeFieldSource, CommittedFieldBlend};
use std::sync::atomic::Ordering;
use std::time::Instant;

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

// ─── Test field impl (mirrors the lib's LinearField, which is private) ──────

struct LinearField {
    scale: f32,
    commitment: [u8; 32],
}

impl LinearField {
    fn new(scale: f32, id: u8) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"LinearField");
        hasher.update(&[id]);
        hasher.update(&scale.to_le_bytes());
        Self {
            scale,
            commitment: *hasher.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for LinearField {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for j in 0..32 {
            dz_scratch[j] = self.scale * z[j];
        }
        &mut dz_scratch[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        self.scale.abs()
    }
}

fn make_blend(pi: [f32; 3]) -> CommittedFieldBlend<3, 32> {
    let mut blend = CommittedFieldBlend::uncommitted();
    blend.pi = pi;
    blend.tau = 1.0;
    blend
}

// ─── G4: zero-alloc hot path ────────────────────────────────────────────────

#[test]
fn g4_zero_alloc() {
    let f0 = LinearField::new(2.0, 0);
    let f1 = LinearField::new(5.0, 1);
    let f2 = LinearField::new(1.0, 2);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];
    let blend = make_blend([1.0, -1.0, 0.5]);
    let z = [0.5f32; 32];

    let mut rng = fastrand::Rng::with_seed(123);

    // Warmup (1 call — any one-time init).
    let mut warmup_rng = fastrand::Rng::with_seed(0);
    let _ = committed_blend_pi_sensitivity::<3, 32, 32>(&blend, &fields, &z, 0.01, 8, &mut warmup_rng);

    let before = ALLOC_COUNT.load(Ordering::SeqCst);
    for _ in 0..1000 {
        let _ = committed_blend_pi_sensitivity::<3, 32, 32>(
            &blend, &fields, &z, 0.01, 8, &mut rng,
        );
    }
    let after = ALLOC_COUNT.load(Ordering::SeqCst);
    let allocs = after - before;
    assert_eq!(
        allocs, 0,
        "G4 FAIL: {allocs} allocs in 1000 probe calls — expected 0"
    );
}

// ─── G5: latency < 5µs p50 (release mode only — debug builds are unoptimized) ──

#[test]
fn g5_latency_under_5us_p50() {
    let f0 = LinearField::new(2.0, 0);
    let f1 = LinearField::new(5.0, 1);
    let f2 = LinearField::new(1.0, 2);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];
    let blend = make_blend([1.0, -1.0, 0.5]);
    let z = [0.5f32; 32];
    let mut rng = fastrand::Rng::with_seed(999);

    const N_SAMPLES: usize = 1000;
    let mut latencies_ns = [0u64; N_SAMPLES];

    for latency in latencies_ns.iter_mut() {
        let start = Instant::now();
        let _ = committed_blend_pi_sensitivity::<3, 32, 32>(
            &blend, &fields, &z, 0.01, 8, &mut rng,
        );
        *latency = start.elapsed().as_nanos() as u64;
    }

    latencies_ns.sort_unstable();
    let p50_ns = latencies_ns[N_SAMPLES / 2];
    let p50_us = p50_ns as f64 / 1000.0;
    let target_us = 5.0;

    eprintln!(
        "G5 latency: p50 = {p50_us:.3}µs (target < {target_us}µs), p99 = {:.3}µs",
        latencies_ns[N_SAMPLES * 99 / 100] as f64 / 1000.0
    );

    // G5 assertion is release-only — debug builds are unoptimized and 4-5x
    // slower. The probe is a diagnostic, not hot-path; even 20µs debug is fine.
    #[cfg(not(debug_assertions))]
    {
        assert!(
            p50_us < target_us,
            "G5 FAIL (release): p50 = {p50_us:.3}µs >= {target_us}µs"
        );
    }
}
