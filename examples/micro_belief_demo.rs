//! Plan 276 T4.2: MicroRecurrentBeliefState demo.
//!
//! Minimal end-to-end lifecycle of a per-entity belief kernel:
//!   1. construct a `LeakyIntegrator` (Family C — the promotable, battle-tested kernel),
//!   2. advance it 1000 ticks on a deterministic synthetic input stream,
//!   3. bridge the latent belief vector to 3 bounded raw scalars (the syncable output),
//!   4. snapshot the kernel weights (BLAKE3-committed freeze/thaw artifact).
//!
//! Run with:
//!   cargo run --release --example micro_belief_demo --features micro_belief
//!
//! The attractor family (Family A) and latent-thought family (Family B) are also
//! demonstrated briefly — they remain opt-in experiments (G1.4 latency + G2.1 coherence
//! both failed; see `.benchmarks/276_micro_belief_goat.md`).

use katgpt_core::micro_belief::{
    AttractorKernel, LatentThoughtKernel, LeakyIntegrator, MicroRecurrentBeliefState,
    MicroRecurrentKernelSnapshot, RecurrenceFamily, project_to_scalars,
};

/// Deterministic synthetic input: smooth bounded signal in [-1, 1].
/// Same formula as `micro_belief::tests::deterministic_input` so the run is reproducible.
fn deterministic_input(step: usize, dim: usize) -> Vec<f32> {
    let s = step as f32;
    (0..dim)
        .map(|i| {
            let f = i as f32;
            (f * 0.1 + s * 0.01).sin() * 0.5 * (s * 0.003).cos()
        })
        .collect()
}

fn main() {
    println!("=== Plan 276: MicroRecurrentBeliefState demo ===\n");

    let dim = 16usize;
    let steps = 1000usize;

    // ── Family C (LeakyIntegrator) — the promotable kernel ────────────────
    // Defaults match ReconstructionConfig (lr=0.1, max_delta=0.2) and are
    // byte-identical to ReconstructionState::evolve_hla via the shared
    // leaky_core::leaky_step primitive.
    let leaky = LeakyIntegrator::hla_default(dim);
    let mut state_c = vec![0.0f32; dim];
    for t in 0..steps {
        let x = deterministic_input(t, dim);
        leaky.step(&mut state_c, &x);
    }
    println!("Family C (LeakyIntegrator) after {steps} steps:");
    println!("  family     = {:?}", leaky.family());
    println!("  state[..5] = {:?}", &state_c[..5]);
    println!("  |state|    = {:.4}", norm(&state_c));
    assert_eq!(leaky.family(), RecurrenceFamily::DeltaRule);

    // ── Bridge: latent belief → 3 bounded raw scalars (syncable) ──────────
    // direction_k = unit vector along dim k → out[k] = sigmoid(state[k]).
    let directions = identity_directions(3, dim);
    let mut scalars = [0.0f32; 3];
    project_to_scalars(&state_c, &directions, dim, &mut scalars);
    println!("\nBridge (3 synced scalars, sigmoid-projected):");
    println!("  scalars   = {scalars:?}");
    for v in &scalars {
        assert!(*v > 0.0 && *v < 1.0, "bridge output must be in (0,1)");
    }

    // ── Family A (Attractor) — opt-in experiment ──────────────────────────
    let attractor = AttractorKernel::from_seed(42, dim);
    let mut state_a = vec![0.0f32; dim];
    for t in 0..steps {
        let x = deterministic_input(t, dim);
        attractor.step(&mut state_a, &x);
    }
    println!("\nFamily A (AttractorKernel, seed=42) after {steps} steps:");
    println!("  state[..5] = {:?}", &state_a[..5]);
    for &v in &state_a {
        assert!(v > -1.0001 && v < 1.0001, "attractor state must stay in (-1,1)");
    }

    // ── Family B (LatentThought, K=3) — opt-in experiment ─────────────────
    let latent_thought = LatentThoughtKernel::from_seed(42, dim, 3);
    let mut state_b = vec![0.0f32; dim];
    for t in 0..steps {
        let x = deterministic_input(t, dim);
        latent_thought.step(&mut state_b, &x);
    }
    println!("\nFamily B (LatentThoughtKernel, K=3, seed=42) after {steps} steps:");
    println!("  state[..5] = {:?}", &state_b[..5]);
    assert_eq!(latent_thought.family(), RecurrenceFamily::LatentThought);

    // ── Snapshot: freeze/thaw artifact (BLAKE3-committed) ──────────────────
    // `from_kernel` takes the serialised weights blob (caller-owned). version is
    // caller-managed (would be incremented by the hot-swap layer on each swap).
    let snapshot = MicroRecurrentKernelSnapshot::from_kernel(
        &attractor,
        attractor.to_snapshot_blob(),
        1,
    );
    println!("\nSnapshot (freeze/thaw, BLAKE3-committed):");
    println!("  family  = {:?}", snapshot.family);
    println!("  dim     = {}", snapshot.dim);
    println!("  version = {}", snapshot.version);
    println!("  blake3  = {}", hex_short(&snapshot.blake3));
    assert!(snapshot.verify(), "snapshot MUST verify (BLAKE3 match)");

    // G1.6 sanity: K=1 LatentThought is bit-identical to the attractor with the same seed.
    let lt_k1 = LatentThoughtKernel::from_seed(42, dim, 1);
    let mut s_a2 = vec![0.0f32; dim];
    let mut s_b2 = vec![0.0f32; dim];
    for t in 0..100 {
        let x = deterministic_input(t, dim);
        attractor.step(&mut s_a2, &x);
        lt_k1.step(&mut s_b2, &x);
    }
    assert_eq!(s_a2, s_b2, "G1.6: K=1 LatentThought must equal Attractor");
    println!("\nG1.6 verified: LatentThought(K=1) bit-identical to Attractor (same seed).");

    println!("\nAll lifecycle steps OK. See .benchmarks/276_micro_belief_goat.md for the GOAT verdict.");
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Flattened identity direction matrix: K rows of `dim`, row k = unit vector along dim k.
fn identity_directions(k: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; k * dim];
    for row in 0..k {
        out[row * dim + row] = 1.0;
    }
    out
}

fn hex_short(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(16);
    for b in &bytes[..8] {
        s.push_str(&format!("{b:02x}"));
    }
    s.push('…');
    s
}
