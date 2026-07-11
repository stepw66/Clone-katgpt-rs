//! Plan 303 T4.2 — Salience Tri-Gate batched throughput demo.
//!
//! Runs `decide_batch` with `N = 10_000` activations and prints wall-clock
//! throughput in decisions/sec.
//!
//! **No game semantics** — synthetic inputs. The point is the throughput
//! number, not the decision distribution.
//!
//! Note: a single-shot `Instant` measurement includes setup noise and is
//! **not** a GOAT-gate-quality benchmark. The authoritative latency /
//! throughput numbers come from the Criterion bench in Plan 303 Phase 2
//! (T2.2, deferred). This example is a quick smoke-test of the batched API
//! shape, not a perf claim.
//!
//! Run with:
//! ```text
//! cargo run --example salience_tri_gate_batch --features salience_tri_gate --release
//! ```

use std::time::Instant;

use katgpt_core::salience::{SalienceDecision, SalienceTriGate};

const D: usize = 8;
const D_SPEAK: [f32; D] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const D_DELEGATE: [f32; D] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Minimal deterministic LCG — same convention as `gate::tests` and the
/// `salience_tri_gate_basic` example.
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
    fn next_f32(&mut self) -> f32 {
        // Uniform in [0, 1). `next()` returns the top 31 bits — divide by 2^31
        // (not `u32::MAX` ≈ 2^32) so the range is actually [0, 1).
        (self.next() as f32) / ((1u64 << 31) as f32)
    }
}

fn main() {
    let gate: SalienceTriGate<u32, D> = SalienceTriGate::new(
        D_SPEAK, D_DELEGATE, 0.3,  // w_z
        0.2,  // w_c
        2.0,  // beta_speak
        2.0,  // beta_delegate
        0.5,  // tau_speak
        0.5,  // tau_delegate
        0.15, // floor_speak
        0.4,  // ceil_delegate
    );

    let n: usize = 10_000;
    let mut rng = Lcg::new(0xBA7C_CAFE);

    // Allocate the input/output buffers up-front. Allocation cost is excluded
    // from the timed region — only the `decide_batch` call itself is measured.
    let mut activations = vec![[0f32; D]; n];
    let mut z = vec![0f32; n];
    let mut c = vec![0f32; n];
    let payloads: Vec<u32> = (0..n as u32).collect();
    let mut out: Vec<SalienceDecision<u32>> = vec![SalienceDecision::Silent; n];

    // Fill inputs (not timed).
    for i in 0..n {
        for v in activations[i].iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        z[i] = rng.next_f32();
        c[i] = rng.next_f32();
    }

    // ── Timed region ────────────────────────────────────────────────────
    let t0 = Instant::now();
    gate.decide_batch(&activations, &z, &c, &payloads, 0, &mut out);
    let elapsed = t0.elapsed();
    // ── End timed region ────────────────────────────────────────────────

    let secs = elapsed.as_secs_f64();
    let us = secs * 1e6;
    let decisions_per_sec = (n as f64) / secs;
    let millions = decisions_per_sec / 1e6;

    // Sanity: the batch did real work (not all the same variant by accident).
    let silent = out
        .iter()
        .filter(|d| matches!(d, SalienceDecision::Silent))
        .count();
    let speak = out
        .iter()
        .filter(|d| matches!(d, SalienceDecision::Speak))
        .count();
    let delegate = out
        .iter()
        .filter(|d| matches!(d, SalienceDecision::Delegate(_)))
        .count();
    assert_eq!(silent + speak + delegate, n);

    println!("Batched decide (N={n}, D={D}): {us:.2} μs → {millions:.1}M decisions/sec");
    println!("  (variant split: Silent={silent} Speak={speak} Delegate={delegate})");
    println!(
        "  note: single-shot timing — for GOAT-gate numbers see Plan 303 Phase 2 (T2.2) Criterion bench"
    );
}
