//! Plan 303 T4.1 — Salience Tri-Gate basic demo.
//!
//! Constructs a gate with hand-tuned direction vectors, runs 100 deterministic
//! pseudo-random activations, and prints the decision distribution
//! (`Silent` / `Speak` / `Delegate`).
//!
//! **No game semantics** — this crate is game-agnostic. The activation `a`,
//! zone-attention `z`, and curiosity `c` here are just synthetic scalars
//! drawn from a fixed-seed LCG; the point is to show the API shape and the
//! three-way decision surface.
//!
//! Run with:
//! ```text
//! cargo run --example salience_tri_gate_basic --features salience_tri_gate
//! ```

use katgpt_rs::salience::{SalienceDecision, SalienceTriGate};

/// Activation dimension for this demo. The gate is generic over `D` — pick
/// whatever fits your latent space.
const D: usize = 8;

/// "Speak" direction — projection of `a` onto this axis drives `salience`.
/// Hand-tuned unit vector so the math is easy to reason about.
const D_SPEAK: [f32; D] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// "Delegate" direction — projection of `a` onto this axis drives
/// `delegate_dot`. Orthogonal to `D_SPEAK` so the two sigmoids are
/// independently controllable in this demo.
const D_DELEGATE: [f32; D] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

/// Minimal deterministic LCG (Numerical Recipes constants). Used in place of
/// `rand` so the demo output is bit-reproducible across runs and platforms —
/// same convention as `gate::tests`.
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
        // Uniform in [0, 1). `next()` returns the top 31 bits, so divide by
        // 2^31 (not `u32::MAX` ≈ 2^32) — otherwise the range is [0, 0.5) and
        // `* 2.0 - 1.0` only spans [-1, 0), never positive.
        (self.next() as f32) / ((1u64 << 31) as f32)
    }
}

fn main() {
    // Gate parameters tuned so a meaningful fraction of the synthetic
    // activations lands in each of the three buckets. `floor_speak` is low
    // enough that low-salience samples still survive past the anti-babble
    // gate; `ceil_delegate` is low enough that high-`delegate_dot` samples
    // actually delegate rather than speaking inline.
    let gate: SalienceTriGate<u32, D> = SalienceTriGate::new(
        D_SPEAK, D_DELEGATE, 0.3,  // w_z — zone-attention weight
        0.2,  // w_c — curiosity weight
        2.0,  // beta_speak
        2.0,  // beta_delegate
        0.5,  // tau_speak
        0.5,  // tau_delegate
        0.15, // floor_speak — below this speak score → Silent
        0.4,  // ceil_delegate — above this delegate score → Delegate
    );

    // Fixed seed so the printed distribution is stable run-to-run.
    let mut rng = Lcg::new(0xCAFE_BABE_1234);
    let n = 100usize;

    let mut silent = 0u32;
    let mut speak = 0u32;
    let mut delegate = 0u32;

    let mut a = [0f32; D];
    for _ in 0..n {
        // Span [-1, 1] per axis — covers both direction projections with both
        // signs, so all three branches can fire.
        for v in a.iter_mut() {
            *v = rng.next_f32() * 2.0 - 1.0;
        }
        let z = rng.next_f32();
        let c = rng.next_f32();
        match gate.decide(&a, z, c, 0, 0) {
            SalienceDecision::Silent => silent += 1,
            SalienceDecision::Speak => speak += 1,
            SalienceDecision::Delegate(_) => delegate += 1,
        }
    }

    println!("Salience Tri-Gate (D={D}) over {n} activations:");
    println!("  Silent:   {silent}");
    println!("  Speak:    {speak}");
    println!("  Delegate: {delegate}");
    assert_eq!(
        silent + speak + delegate,
        n as u32,
        "internal: decision counts must sum to n"
    );
}
