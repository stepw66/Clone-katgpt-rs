//! Latent Field Steering — G4 crowd-scale performance gate (Plan 309 T2.4).
//!
//! ## Hypothesis (Research 290 §1.4)
//!
//! Applying a global steering field to a crowd of 5000 8-dim NPC latent states
//! completes in <1ms — fast enough for a 20Hz game tick (50ms budget). The
//! element-wise SAXPY auto-vectorizes at d=8.
//!
//! ## Setup
//!
//! - 5000 NPC latent states, 8-dim each (flattened 40000-element buffer).
//! - Global field, α=0.3.
//! - Time 1000 iterations, report p50/p95/p99.
//!
//! ## Gate
//!
//! - **PASS:** p50 < 1ms (1000µs) in release build.
//! - **KILL:** p50 > 5ms — too slow for real-time game tick.
//!
//! ## Run
//!
//! ```bash
//! cargo test --features latent_field_steering --release \
//!   --test latent_steering_g4_crowd_perf -- --nocapture
//! ```

#![cfg(feature = "latent_field_steering")]

use katgpt_core::latent_steering::{
    FieldSupport, LatentField, LatentSteeringVector, apply_field_to_crowd,
};
use std::time::Instant;

const D: usize = 8;
const N_NPCS: usize = 5000;
const N_ITERS: usize = 1000;
/// Gate: median per-apply latency < 1ms (1000µs).
const GATE_US: f64 = 1000.0;

struct Rng {
    s: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { s: seed.max(1) }
    }
    fn next_f32(&mut self) -> f32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        let bits = (self.s >> 11) as u32;
        bits as f32 / u32::MAX as f32 * 2.0 - 1.0
    }
}

#[test]
fn g4_crowd_scale_perf() {
    let mut rng = Rng::new(0xC40D);

    // ── Random unit steering direction ─────────────────────────────────
    let mut dir: Vec<f32> = (0..D).map(|_| rng.next_f32()).collect();
    let norm = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut dir {
        *x /= norm;
    }
    let steering = LatentSteeringVector::new(dir, 0.3, 1e-4).unwrap();
    let field = LatentField {
        steering,
        support: FieldSupport::Global,
    };

    // ── Build crowd ────────────────────────────────────────────────────
    let mut states: Vec<f32> = (0..N_NPCS * D).map(|_| rng.next_f32()).collect();
    let positions = vec![None; N_NPCS];
    let zones = vec![None; N_NPCS];

    // ── Warmup ─────────────────────────────────────────────────────────
    for _ in 0..10 {
        apply_field_to_crowd(&mut states, D, &positions, &zones, &field);
    }

    // ── Measure: time each apply individually for p50/p95/p99 ──────────
    let mut latencies_us: Vec<f64> = Vec::with_capacity(N_ITERS);
    for _ in 0..N_ITERS {
        let t0 = Instant::now();
        apply_field_to_crowd(&mut states, D, &positions, &zones, &field);
        let elapsed = t0.elapsed();
        latencies_us.push(elapsed.as_nanos() as f64 / 1000.0);
    }
    std::hint::black_box(&states);

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies_us[N_ITERS / 2];
    let p95 = latencies_us[(N_ITERS as f64 * 0.95) as usize];
    let p99 = latencies_us[(N_ITERS as f64 * 0.99) as usize];

    println!(
        "G4 crowd perf ({N_NPCS} NPCs × {D}d, global field): p50={p50:.1}µs, p95={p95:.1}µs, \
         p99={p99:.1}µs (gate p50 < {GATE_US:.0}µs)"
    );

    assert!(
        p50 < GATE_US,
        "G4 FAIL: p50={p50:.1}µs ≥ {GATE_US:.0}µs — too slow for 20Hz tick"
    );
    println!("G4 PASS: p50={p50:.1}µs < {GATE_US:.0}µs.");
}
