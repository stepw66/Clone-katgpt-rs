//! Example: CommittedFieldBlend — Re-commit on Major Event (Plan 321 T4.2).
//!
//! Demonstrates the **re-commit lifecycle** of `CommittedFieldBlend`. Unlike
//! `PersonalityWeightedComposition` (Plan 297), which *drifts* continuously,
//! `CommittedFieldBlend` is **frozen** for the entity's lifetime and changes
//! only via an explicit `commit()` call on a major personality event.
//!
//! ## Scenario
//!
//! A single NPC starts as **cautious** — its initial trajectory summary dots
//! weakly against the aggressive/social directions and strongly against the
//! cautious direction, so the committed `pi` favors the cautious field.
//!
//! A "predator encounter" is the major personality event: the host feeds the
//! blend a *new* trajectory summary (recent behavior dominated by flight,
//! which dots strongly against the aggressive direction). The host calls
//! `commit()` again with `version = 2`, which:
//!
//! 1. Recomputes `pi` from the new summary.
//! 2. Increments `version` (`version` IS part of the BLAKE3 input here,
//!    unlike `PersonalitySnapshot`).
//! 3. Produces a **new** BLAKE3 commitment — a distinct audit event.
//!
//! After re-commit, the NPC's dynamics change: the cautious field's gate
//! weakens and the aggressive field's gate strengthens (or vice versa,
//! depending on the new summary). The old commitment no longer verifies;
//! the new one does.
//!
//! ## What the example proves
//!
//! - **Re-commit is a distinct event.** `version` is part of the BLAKE3 input,
//!   so `blake3(v=1) ≠ blake3(v=2)` even if `pi` were identical (it isn't).
//! - **Old commitment fails verification** after re-commit (the stored hash
//!   changed; an observer that cached the v=1 hash detects the swap).
//! - **Tamper detection still works** — flipping a `pi` byte after commit
//!   makes `verify_commitment` return `false`.
//! - **Lipschitz bound tracks the new personality.** The committed safety
//!   bound is a closed-form function of the new `pi` and the frozen field
//!   Lipschitz constants.
//!
//! ## Run
//!
//! ```sh
//! cargo run --example committed_blend_02_recommit_on_event \
//!     --features committed_field_blend --release
//! ```

// Indexed loops are intentional in this demo: the per-axis band-fill loops
// (`for j in 0..11`, `for j in 22..32`) need the index for the band boundary
// semantics, and the inner kernels are clearer in indexed form.
#![allow(clippy::needless_range_loop)]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, TriArchetypeBlend};

// Reuse the three-archetype scenario from `committed_blend_01_three_archetypes`.
// We inline minimal copies here to keep this example standalone.

struct LinearField {
    scale: f32,
    axis: usize,
    commitment: [u8; 32],
}

impl LinearField {
    fn new(scale: f32, axis: usize) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"linear");
        h.update(&scale.to_le_bytes());
        h.update(&(axis as u32).to_le_bytes());
        Self {
            scale,
            axis,
            commitment: *h.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for LinearField {
    fn evolve<'a>(&self, z: &[f32], dz: &'a mut [f32]) -> &'a mut [f32] {
        for k in 0..32 {
            dz[k] = 0.0;
        }
        dz[self.axis] = self.scale * z[self.axis];
        &mut dz[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        self.scale.abs()
    }
}

struct ConstantField {
    bias: [f32; 32],
    commitment: [u8; 32],
}

impl ConstantField {
    fn new(bias: [f32; 32]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"constant");
        for b in &bias {
            h.update(&b.to_le_bytes());
        }
        Self {
            bias,
            commitment: *h.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for ConstantField {
    fn evolve<'a>(&self, _z: &[f32], dz: &'a mut [f32]) -> &'a mut [f32] {
        dz[..32].copy_from_slice(&self.bias);
        &mut dz[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        0.0
    }
}

struct RotationField {
    i: usize,
    j: usize,
    cos_a: f32,
    sin_a: f32,
    commitment: [u8; 32],
}

impl RotationField {
    fn new(i: usize, j: usize, angle_rad: f32) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(b"rotation");
        h.update(&(i as u32).to_le_bytes());
        h.update(&(j as u32).to_le_bytes());
        h.update(&angle_rad.to_le_bytes());
        Self {
            i,
            j,
            cos_a: angle_rad.cos(),
            sin_a: angle_rad.sin(),
            commitment: *h.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for RotationField {
    fn evolve<'a>(&self, z: &[f32], dz: &'a mut [f32]) -> &'a mut [f32] {
        for k in 0..32 {
            dz[k] = 0.0;
        }
        let zi = z[self.i];
        let zj = z[self.j];
        dz[self.i] = self.cos_a * zi - self.sin_a * zj - zi;
        dz[self.j] = self.sin_a * zi + self.cos_a * zj - zj;
        &mut dz[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        1.0
    }
}

fn make_direction_vectors() -> [[f32; 32]; 3] {
    let mut d0 = [0.0f32; 32];
    let mut d1 = [0.0f32; 32];
    let mut d2 = [0.0f32; 32];
    for j in 0..11 {
        d0[j] = 1.0;
    }
    for j in 11..22 {
        d1[j] = 1.0;
    }
    for j in 22..32 {
        d2[j] = 1.0;
    }
    [d0, d1, d2]
}

fn hex_short(bytes: &[u8; 32]) -> String {
    bytes[..8].iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("=== CommittedFieldBlend — Re-commit on Major Event ===");
    println!("=== Plan 321 T4.2 — version-bumped personality swap ===\n");

    // Three archetype fields — same family as the 01 example.
    let aggressive = LinearField::new(0.8, 0);
    let cautious = RotationField::new(8, 16, 0.3);
    let mut social_bias = [0.0f32; 32];
    for j in 0..32 {
        social_bias[j] = 0.02 + 0.003 * (j as f32);
    }
    let social = ConstantField::new(social_bias);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&aggressive, &cautious, &social];

    let dirs = make_direction_vectors();

    // ─── v=1: initial commit — "cautious" personality ──────────────────────
    //
    // Initial trajectory summary: weak on aggressive axis, strong on cautious
    // axis, weak on social axis.
    let mut initial_summary = [0.0f32; 32];
    for j in 0..32 {
        // DC 0.05 everywhere, plus an extra +0.3 on the cautious band [11..22).
        initial_summary[j] = 0.05 + if (11..22).contains(&j) { 0.3 } else { 0.0 };
    }

    let mut blend = TriArchetypeBlend::uncommitted();
    let hash_v1 = blend.commit(&initial_summary, &dirs, &fields, 1);

    println!("── v=1: initial commit (cautious personality) ──");
    println!(
        "  pi        = [{:+.4}, {:+.4}, {:+.4}]",
        blend.pi[0], blend.pi[1], blend.pi[2]
    );
    println!("  version   = {}", blend.version);
    println!("  blake3    = {}", hex_short(&blend.blake3));
    println!("  L_pi      = {:.4}", blend.lipschitz_bound(&fields));
    assert!(
        blend.verify_commitment(&fields),
        "v=1 commitment must verify"
    );

    // Snapshot the v=1 hash — an observer (sync layer, audit log) would cache
    // this for later comparison.
    let cached_hash_v1 = hash_v1;

    // ─── Major personality event: predator encounter ───────────────────────
    //
    // The host observes a NEW recent trajectory: the NPC fled. Flight
    // dominates the aggressive direction (axes 0..11) negatively — the NPC
    // retreated. The new summary dots strongly NEGATIVE on aggressive and
    // POSITIVE on social (it ran toward its pack).
    let mut post_encounter_summary = [0.0f32; 32];
    for j in 0..32 {
        post_encounter_summary[j] = 0.05;
    }
    for j in 0..11 {
        post_encounter_summary[j] -= 0.4; // strong negative on aggressive axis
    }
    for j in 22..32 {
        post_encounter_summary[j] += 0.4; // strong positive on social axis
    }

    // ─── v=2: re-commit on the major event ─────────────────────────────────
    //
    // Same `commit()` call, but with `version = 2`. This produces:
    //   - a new `pi` (from the new summary)
    //   - a new `blake3` (because both pi and version are part of the input)
    //   - a new audit event that the sync layer can record
    let pi_before = blend.pi;
    let hash_v2 = blend.commit(&post_encounter_summary, &dirs, &fields, 2);

    println!("\n── v=2: re-commit after predator encounter ──");
    println!(
        "  pi (old)  = [{:+.4}, {:+.4}, {:+.4}]",
        pi_before[0], pi_before[1], pi_before[2]
    );
    println!(
        "  pi (new)  = [{:+.4}, {:+.4}, {:+.4}]",
        blend.pi[0], blend.pi[1], blend.pi[2]
    );
    println!("  version   = {}", blend.version);
    println!("  blake3    = {}", hex_short(&blend.blake3));
    println!("  L_pi      = {:.4}", blend.lipschitz_bound(&fields));

    // ─── Assertions: re-commit semantics ───────────────────────────────────

    // (1) version IS part of the BLAKE3 input → distinct audit events.
    assert_ne!(
        hash_v1, hash_v2,
        "v=1 and v=2 must produce distinct BLAKE3 commitments"
    );
    assert_ne!(
        cached_hash_v1, blend.blake3,
        "the cached v=1 hash must NOT match the live v=2 hash"
    );
    println!("\n✓ v=1 and v=2 produce distinct BLAKE3 commitments (version IS part of the hash).");

    // (2) The new commitment verifies.
    assert!(
        blend.verify_commitment(&fields),
        "v=2 commitment must verify"
    );
    println!("✓ v=2 commitment verifies against the frozen fields.");

    // (3) The committed personality actually changed: aggressive gate dropped,
    // social gate rose. This is the behavioral consequence of re-commit.
    let g = |pi_k: f32| katgpt_core::personality_sigmoid(pi_k / blend.tau);
    let g_aggr_old = g(pi_before[0]);
    let g_aggr_new = g(blend.pi[0]);
    let g_social_old = g(pi_before[2]);
    let g_social_new = g(blend.pi[2]);
    println!(
        "\n  aggressive gate: {:.4} → {:.4}  (Δ = {:+.4})",
        g_aggr_old,
        g_aggr_new,
        g_aggr_new - g_aggr_old
    );
    println!(
        "  social gate:     {:.4} → {:.4}  (Δ = {:+.4})",
        g_social_old,
        g_social_new,
        g_social_new - g_social_old
    );
    assert!(
        g_aggr_new < g_aggr_old,
        "aggressive gate should drop after fleeing"
    );
    assert!(
        g_social_new > g_social_old,
        "social gate should rise after running to the pack"
    );
    println!("✓ personality changed in the expected direction.");

    // ─── Anti-tamper: detect a forged pi after commit ──────────────────────
    //
    // An attacker (or a buggy sync layer) flips a bit of pi[1]. The stored
    // hash no longer matches the recomputed one — verify_commitment catches
    // it. This is the thaw-time anti-tamper check.
    let pre_tamper_hash = blend.blake3;
    let pi1_original = blend.pi[1];
    let mut bytes = blend.pi[1].to_le_bytes();
    bytes[0] ^= 0x01;
    blend.pi[1] = f32::from_le_bytes(bytes);

    println!("\n── Anti-tamper check (flip 1 bit of pi[1]) ──");
    println!("  pi[1]: {:+.4} → {:+.4}", pi1_original, blend.pi[1]);
    assert!(
        !blend.verify_commitment(&fields),
        "tampered pi must fail verify_commitment"
    );
    println!("✓ verify_commitment returns false after pi tamper.");
    println!(
        "  (stored blake3 {} no longer matches recomputed hash)",
        hex_short(&pre_tamper_hash)
    );

    // Restore pi[1] so the commitment verifies again — demonstrating recovery.
    blend.pi[1] = pi1_original;
    assert!(
        blend.verify_commitment(&fields),
        "restored pi must verify again"
    );
    println!("✓ after restoring pi[1], verify_commitment passes again.");

    // ─── Observer view: cached-hash comparison detects the swap ────────────
    //
    // An observer that cached hash_v1 (e.g. the chain quorum layer, an audit
    // log) sees that the live hash differs from its cache → it knows a
    // re-commit happened. This is the sync-boundary audit semantics: the K
    // raw pi scalars + the new version cross the wire as a commitment event.
    println!("\n── Observer view (sync layer / audit log) ──");
    println!("  cached v=1 hash: {}", hex_short(&cached_hash_v1));
    println!("  live   v=2 hash: {}", hex_short(&blend.blake3));
    assert_ne!(
        cached_hash_v1, blend.blake3,
        "observer detects the personality swap via hash mismatch"
    );
    println!("✓ observer detects the re-commit via hash mismatch.");
    println!("  → the host would emit a SyncBlock with the new (pi, version, blake3)");
    println!("    as a commitment event (riir-chain R003 LatCal recipe, deferred).");

    println!("\n=== Demo complete ===");
}
