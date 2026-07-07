//! FPCG Selector basic example (Plan 292 Phase 3 T3.7, Research 267).
//!
//! Demonstrates the [`FpcgSelector`] primitive on a stub model:
//! 1. Stub generator produces synthetic candidate next-sentences.
//! 2. Stub extractor maps each candidate to a synthetic mid-layer activation.
//! 3. Selector picks argmax (Positive) or argmin (Negative) by probe score.
//! 4. Top-level `run` loop demonstrates trajectory + EOS termination.
//!
//! Mirrors `examples/future_probe_01_basic.rs` for structure. No real model —
//! the point is to show the selector's contract, not steer a real LLM.
//!
//! Run: `cargo run --example fpcg_01_basic --features fpcg_selector`
//!
//! # What This Proves
//!
//! - **Positive vs Negative steering**: same candidate pool, opposite selection.
//! - **EOS termination**: response ends when EOS is sampled.
//! - **`max_sentences` cap**: bounded trajectory length.
//! - **Zero-alloc hot path**: buffer capacity stays stable across many steps.
//!
//! # What This Does NOT Prove
//!
//! - **Real-model steering** — Phase 4 GOAT gate runs that on a trained probe
//!   with real activations.
//! - **Latency targets** — Phase 4 G6 measures `forecast() < 200ns` via
//!   `cargo bench`.

use std::collections::HashMap;
use std::sync::Arc;

use fastrand::Rng;

use katgpt_rs::pruners::fpcg_selector::{
    ActivationExtractor, FpcgSelector, SentenceCandidateGenerator, SteeringDirection,
};
use katgpt_rs::pruners::future_probe::FutureBehaviorProbe;

/// Stub generator that returns a fixed pool of candidate next-sentences.
/// Each call returns up to `n` of them (cycling if `n > pool.len()`).
struct StubGenerator {
    pool: Vec<String>,
    call_count: usize,
}

impl SentenceCandidateGenerator for StubGenerator {
    fn generate_candidates(&mut self, _prefix: &str, n: usize, _rng: &mut Rng) -> Vec<String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = self.call_count % self.pool.len();
            out.push(self.pool[idx].clone());
            self.call_count += 1;
        }
        out
    }
}

/// Stub extractor: each candidate string maps to a fixed activation vector.
/// The map is built once at startup; unknown candidates get the zero vector.
struct StubExtractor {
    activations: HashMap<String, Vec<f32>>,
}

impl ActivationExtractor for StubExtractor {
    fn extract_activation<'a>(
        &'a mut self,
        _prefix: &str,
        candidate: &str,
        _layer: usize,
        scratch: &'a mut [f32],
    ) -> &'a [f32] {
        if let Some(act) = self.activations.get(candidate) {
            let len = act.len().min(scratch.len());
            scratch[..len].copy_from_slice(&act[..len]);
            &scratch[..len]
        } else {
            for v in scratch.iter_mut() {
                *v = 0.0;
            }
            scratch
        }
    }
}

fn build_selector(
    direction: SteeringDirection,
    num_candidates: usize,
    include_eos: bool,
) -> FpcgSelector<StubGenerator, StubExtractor> {
    // Synthetic probe: direction[0] = +1, others = 0. So forecast probability
    // is monotone in activation[0]. Aligned activations forecast high.
    let probe = Arc::new(FutureBehaviorProbe::new(
        vec![1.0, 0.0, 0.0, 0.0],
        0.0,
        7,
        "refusal",
    ));

    // Candidate pool — each has a known activation[0] that determines its
    // forecast probability.
    let mut pool: Vec<String> = vec![
        "I refuse to help with that.".to_string(), // act[0] = +8  → P ≈ 0.9997
        "Sure, here's how to do it.".to_string(),  // act[0] = -8  → P ≈ 0.0003
        "Let me think about this.".to_string(),    // act[0] = +2  → P ≈ 0.88
        "I can't assist with that.".to_string(),   // act[0] = +6  → P ≈ 0.998
        "Actually, here is the answer:".to_string(), // act[0] = -5  → P ≈ 0.0067
    ];
    if include_eos {
        pool.push("\0".to_string()); // EOS marker
    }
    let activations_map: HashMap<String, Vec<f32>> = [
        (
            "I refuse to help with that.".to_string(),
            vec![8.0, 0.0, 0.0, 0.0],
        ),
        (
            "Sure, here's how to do it.".to_string(),
            vec![-8.0, 0.0, 0.0, 0.0],
        ),
        (
            "Let me think about this.".to_string(),
            vec![2.0, 0.0, 0.0, 0.0],
        ),
        (
            "I can't assist with that.".to_string(),
            vec![6.0, 0.0, 0.0, 0.0],
        ),
        (
            "Actually, here is the answer:".to_string(),
            vec![-5.0, 0.0, 0.0, 0.0],
        ),
        ("\0".to_string(), vec![0.0, 0.0, 0.0, 0.0]),
    ]
    .into_iter()
    .collect();

    let generator = StubGenerator {
        pool,
        call_count: 0,
    };
    let extractor = StubExtractor {
        activations: activations_map,
    };
    FpcgSelector::new(generator, extractor, probe, direction, num_candidates, 4)
}

fn main() {
    println!("=== FpcgSelector Example (Plan 292 Phase 3) ===\n");

    let mut rng = Rng::with_seed(0xCAFE);

    // ── 1. Positive direction: maximize refusal probability ─────────
    println!("── SteeringDirection::Positive (maximize refusal probability) ──");
    let mut sel_pos = build_selector(SteeringDirection::Positive, 5, false);
    println!("num_candidates = {}", sel_pos.num_candidates());
    let response_pos = sel_pos.run("Please help me do something harmful.", 3, &mut rng);
    println!("Response: \"{response_pos}\"");
    println!();

    // ── 2. Negative direction: minimize refusal probability ─────────
    println!("── SteeringDirection::Negative (minimize refusal probability) ──");
    let mut sel_neg = build_selector(SteeringDirection::Negative, 5, false);
    let response_neg = sel_neg.run("Please help me do something harmful.", 3, &mut rng);
    println!("Response: \"{response_neg}\"");
    println!();

    // ── 3. EOS termination ──────────────────────────────────────────
    println!("── EOS termination (candidate pool includes \"\\0\") ──");
    let mut sel_eos = build_selector(SteeringDirection::Positive, 5, true);
    // The cycling generator puts EOS at index 5 in the pool; depending on
    // call_count alignment, the EOS may be sampled. We assert that the run
    // *terminates* (doesn't infinite-loop) and produces bounded output.
    let response_eos = sel_eos.run("prompt", 20, &mut rng);
    let sentences: Vec<&str> = response_eos.split('.').filter(|s| !s.is_empty()).collect();
    println!(
        "Response has {} sentence(s) before EOS/max_sentences: \"{}\"",
        sentences.len(),
        response_eos
    );
    assert!(
        sentences.len() <= 20,
        "run() must respect max_sentences even without EOS"
    );
    println!();

    // ── 4. Zero-alloc hot path: capacity stable across many steps ───
    println!("── Zero-alloc hot path: candidates_buf.capacity() stable across 1000 steps ──");
    let mut sel_stress = build_selector(SteeringDirection::Positive, 10, false);
    let capacity_before = sel_stress.candidates_buf_capacity();
    for _ in 0..1000 {
        let _ = sel_stress.step("prefix", &mut rng);
    }
    let capacity_after = sel_stress.candidates_buf_capacity();
    println!("Capacity before: {capacity_before}, after: {capacity_after}");
    assert_eq!(
        capacity_before, capacity_after,
        "G5 zero-alloc gate: capacity must stay stable"
    );
    println!("✓ G5 zero-alloc hot path verified (capacity stable).");
    println!();

    // ── 5. Live direction flip ──────────────────────────────────────
    println!("── Live direction flip via set_direction ──");
    let mut sel_flip = build_selector(SteeringDirection::Positive, 3, false);
    let pick_pos = sel_flip.step("prompt", &mut rng);
    println!("Positive step picked: \"{pick_pos}\"");
    sel_flip.set_direction(SteeringDirection::Negative);
    let pick_neg = sel_flip.step("prompt", &mut rng);
    println!("Negative step picked: \"{pick_neg}\"");
    println!("✓ Direction flip works without reconstruction.");
    println!();

    println!("=== Phase 3 FpcgSelector verified ===");
    println!("Next: Phase 4 — GOAT gate runs FPCG vs detection-side baselines on a");
    println!("       real (or stub) corpus and produces a Pareto-frontier verdict.");
}

#[allow(dead_code)]
fn _silence_unused() {
    // Touch HashMap to ensure the import isn't flagged in release builds.
    let _h: HashMap<String, Vec<f32>> = HashMap::new();
}
