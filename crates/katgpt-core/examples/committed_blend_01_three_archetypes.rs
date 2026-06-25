//! Example: CommittedFieldBlend — Three Archetypes × 100 Entities (Plan 321 T4.1).
//!
//! Demonstrates the **sampling-invariance** property (FAME Proposition 3) of
//! `CommittedFieldBlend` end-to-end with a population of 100 entities, each
//! committing a per-entity blend of three archetype operator fields.
//!
//! ## Scenario
//!
//! Three archetype fields (the production Entity Cognition Stack case, K=3,
//! D=32) model analogues of NPC behavior archetypes:
//!
//! - **Aggressive** — pushes state along a fixed "advance" direction (linear
//!   field with positive scale). Lipschitz bound = `|scale|`.
//! - **Cautious**  — rotates state within a 2D subspace (constant-magnitude
//!   rotation field). Lipschitz bound = `1.0`.
//! - **Social**    — pushes every axis by a constant bias (constant field).
//!   Lipschitz bound = `0.0` (output is input-independent → zero Lipschitz).
//!
//! Each of 100 entities has its own hidden "true personality" expressed as a
//! periodic latent trajectory. From this trajectory we derive a summary
//! (per-axis mean over an integer number of full periods — a sampling-invariant
//! statistic). The committed blend weights `pi` are computed once from that
//! summary via sigmoid projection onto three direction vectors, then frozen.
//!
//! ## What the example proves
//!
//! **Fog-of-war sampling invariance.** For each entity we build two summaries
//! of the *same* underlying trajectory:
//!
//! - **Dense** — all 1000 trajectory steps (10 full periods).
//! - **Sparse** — every 10th step (100 steps = 10 full periods), simulating
//!   observation gaps from fog-of-war, network desync, or snapshot thaw.
//!
//! Because the trajectory is periodic and both summaries span an integer number
//! of full periods, both converge to the same per-axis DC component. The
//! committed `pi` vectors are therefore bit-close (within float accumulation
//! noise), and the resulting blended dynamics diverge by `< 1e-3` over a
//! 100-step rollout from identical initial state.
//!
//! This is the defining property: the entity's *committed personality* is
//! invariant under observation gaps.
//!
//! ## Run
//!
//! ```sh
//! cargo run --example committed_blend_01_three_archetypes \
//!     --features committed_field_blend --release
//! ```

// Indexed loops are intentional in this demo: the band-filling patterns
// (`for j in 11..22`, `for j in 22..32`) need the index for the band boundary
// semantics, and the inner per-axis kernels are clearer in indexed form.
#![allow(clippy::needless_range_loop)]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, TriArchetypeBlend};

/// Trajectory period (steps). Must divide the rollout length so dense and
/// sparse summaries both span an integer number of full periods — this is the
/// mathematical precondition for the mean to be a sampling-invariant statistic
/// (FAME Prop. 3).
const PERIOD: usize = 100;

/// Trajectory length (steps). `1000 / PERIOD = 10` full periods in both the
/// dense (every step) and sparse (every 10th step) summaries.
const ROLLEN: usize = 1000;

/// Sparse sampling stride (simulates fog-of-war observation gaps).
const SPARSE_STRIDE: usize = 10;

/// Entity population size.
const N_ENTITIES: usize = 100;

/// Per-entity rollout length for the post-commit dynamics check.
const DYN_STEPS: usize = 100;

/// Dynamics step size (forward-Euler `z += DT · f_pi(z)`).
const DT: f32 = 0.01;

/// Tolerance for "two summaries produce the same `pi`".
const PI_TOL: f32 = 1e-3;

/// Tolerance for "two blends produce the same trajectory".
const TRAJ_TOL: f32 = 1e-3;

// ─── Archetype fields ──────────────────────────────────────────────────────

/// **Aggressive** archetype: linear push along a fixed axis.
/// `f(z) = scale · z[axis] · e_axis`. Lipschitz bound = `|scale|`.
struct AggressiveField {
    scale: f32,
    axis: usize,
    commitment: [u8; 32],
}

impl AggressiveField {
    fn new(scale: f32, axis: usize) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aggressive");
        hasher.update(&scale.to_le_bytes());
        hasher.update(&(axis as u32).to_le_bytes());
        Self {
            scale,
            axis,
            commitment: *hasher.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for AggressiveField {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for j in 0..32 {
            dz_scratch[j] = 0.0;
        }
        // Push along `axis` proportional to the current `axis` component.
        dz_scratch[self.axis] = self.scale * z[self.axis];
        &mut dz_scratch[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        self.scale.abs()
    }
}

/// **Cautious** archetype: constant-magnitude rotation in a 2D subspace.
/// Lipschitz bound = `1.0` (rotation is an isometry).
struct CautiousField {
    i: usize,
    j: usize,
    cos_a: f32,
    sin_a: f32,
    commitment: [u8; 32],
}

impl CautiousField {
    fn new(i: usize, j: usize, angle_rad: f32) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cautious");
        hasher.update(&(i as u32).to_le_bytes());
        hasher.update(&(j as u32).to_le_bytes());
        hasher.update(&angle_rad.to_le_bytes());
        Self {
            i,
            j,
            cos_a: angle_rad.cos(),
            sin_a: angle_rad.sin(),
            commitment: *hasher.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for CautiousField {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for k in 0..32 {
            dz_scratch[k] = 0.0;
        }
        // f(z) = (cos·z_i − sin·z_j − z_i,  sin·z_i + cos·z_j − z_j) on axes (i,j).
        // This is a small-angle rotation minus identity → a damping-like field.
        let zi = z[self.i];
        let zj = z[self.j];
        dz_scratch[self.i] = self.cos_a * zi - self.sin_a * zj - zi;
        dz_scratch[self.j] = self.sin_a * zi + self.cos_a * zj - zj;
        &mut dz_scratch[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        1.0
    }
}

/// **Social** archetype: constant bias along every axis (input-independent).
/// Lipschitz bound = `0.0` (output does not depend on `z`).
struct SocialField {
    bias: [f32; 32],
    commitment: [u8; 32],
}

impl SocialField {
    fn new(bias: [f32; 32]) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"social");
        for b in &bias {
            hasher.update(&b.to_le_bytes());
        }
        Self {
            bias,
            commitment: *hasher.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for SocialField {
    fn evolve<'a>(&self, _z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        dz_scratch[..32].copy_from_slice(&self.bias);
        &mut dz_scratch[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
    fn lipschitz_bound(&self) -> f32 {
        0.0
    }
}

// ─── Direction vectors (one per archetype) ─────────────────────────────────
//
// These are the three "personality axes" each entity projects its trajectory
// summary onto. They are FROZEN library constants — the K=3 archetype field
// library is the freeze/thaw substrate (riir-train trains it once offline;
// katgpt-rs only consumes the frozen result).

fn make_direction_vectors() -> [[f32; 32]; 3] {
    let mut d0 = [0.0f32; 32]; // aggressive axis: first 11 dims
    let mut d1 = [0.0f32; 32]; // cautious axis: middle 11 dims
    let mut d2 = [0.0f32; 32]; // social axis: last 10 dims
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

// ─── Trajectory generation ─────────────────────────────────────────────────
//
// Each entity's hidden "true personality" is encoded as a periodic latent
// trajectory. The per-entity phase shift `phi` and per-axis amplitude `amp[j]`
// differ across entities (deterministic PRNG seed = entity index), so each
// entity commits a *distinct* blend. But all trajectories share the same
// period (PERIOD) and DC component — so dense vs sparse summaries converge
// to the same per-entity `pi`.

fn make_trajectory(entity_idx: usize) -> Vec<[f32; 32]> {
    // Deterministic LCG seeded by entity index — no Rand crate dependency,
    // fully reproducible across runs.
    let mut state = 0x9E37_79B9u32.wrapping_add((entity_idx as u32).wrapping_mul(0x85EB_CA77));
    let mut next_f32 = || {
        // xorshift32 → (0, 1)
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32) / (u32::MAX as f32)
    };

    let phi = next_f32() * core::f32::consts::TAU;
    let dc = 0.05 + 0.10 * next_f32(); // per-entity DC in [0.05, 0.15)
    let mut amp = [0.0f32; 32];
    for j in 0..32 {
        amp[j] = 0.02 + 0.06 * next_f32(); // per-axis amplitude in [0.02, 0.08)
    }

    let mut traj = Vec::with_capacity(ROLLEN);
    for t in 0..ROLLEN {
        let mut s = [0.0f32; 32];
        for j in 0..32 {
            let theta =
                core::f32::consts::TAU * (t as f32) / (PERIOD as f32) + phi + (j as f32) * 0.2;
            s[j] = dc + amp[j] * theta.sin();
        }
        traj.push(s);
    }
    traj
}

/// Per-axis mean of a trajectory slice. This is the host-supplied "summary"
/// passed to `commit()`. For a periodic signal over an integer number of full
/// periods, the mean equals the DC component — a sampling-invariant statistic.
fn mean_summary(traj: &[[f32; 32]]) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    for s in traj {
        for j in 0..32 {
            out[j] += s[j];
        }
    }
    let n = traj.len() as f32;
    for v in out.iter_mut() {
        *v /= n;
    }
    out
}

/// Per-axis mean of the sparse subset (every `stride`-th step).
fn mean_summary_sparse(traj: &[[f32; 32]], stride: usize) -> [f32; 32] {
    let mut out = [0.0f32; 32];
    let mut count = 0usize;
    for s in traj.iter().step_by(stride) {
        for j in 0..32 {
            out[j] += s[j];
        }
        count += 1;
    }
    let n = count as f32;
    for v in out.iter_mut() {
        *v /= n;
    }
    out
}

// ─── Per-entity sampling-invariance check ──────────────────────────────────

struct EntityReport {
    idx: usize,
    pi_dense: [f32; 3],
    pi_sparse: [f32; 3],
    max_pi_diff: f32,
    max_traj_diff: f32,
    blake3: [u8; 32],
    lipschitz: f32,
}

/// Run one entity through the full commit + fog-of-war invariance check.
fn run_entity(
    idx: usize,
    dirs: &[[f32; 32]; 3],
    fields: &[&dyn ArchetypeFieldSource<32>; 3],
) -> EntityReport {
    let traj = make_trajectory(idx);

    let dense = mean_summary(&traj);
    let sparse = mean_summary_sparse(&traj, SPARSE_STRIDE);

    let mut blend_dense = TriArchetypeBlend::uncommitted();
    let mut blend_sparse = TriArchetypeBlend::uncommitted();
    blend_dense.commit(&dense, dirs, fields, 1);
    blend_sparse.commit(&sparse, dirs, fields, 1);

    let mut max_pi_diff = 0.0f32;
    for k in 0..3 {
        max_pi_diff = max_pi_diff.max((blend_dense.pi[k] - blend_sparse.pi[k]).abs());
    }

    // Forward-Euler rollout from identical initial state under each blend.
    let z0 = [0.5f32; 32];
    let mut state_dense = z0;
    let mut state_sparse = z0;
    let mut scratch = [0.0f32; 32];
    let mut dz = [0.0f32; 32];
    for _ in 0..DYN_STEPS {
        blend_dense.apply_blended(fields, &state_dense, &mut scratch, &mut dz);
        for j in 0..32 {
            state_dense[j] += DT * dz[j];
        }
        blend_sparse.apply_blended(fields, &state_sparse, &mut scratch, &mut dz);
        for j in 0..32 {
            state_sparse[j] += DT * dz[j];
        }
    }
    let mut max_traj_diff = 0.0f32;
    for j in 0..32 {
        max_traj_diff = max_traj_diff.max((state_dense[j] - state_sparse[j]).abs());
    }

    // Lipschitz safety bound — closed-form per-entity quantity (Phase 3 T3.1).
    let lipschitz = blend_dense.lipschitz_bound(fields);

    EntityReport {
        idx,
        pi_dense: blend_dense.pi,
        pi_sparse: blend_sparse.pi,
        max_pi_diff,
        max_traj_diff,
        blake3: blend_dense.blake3,
        lipschitz,
    }
}

fn hex_short(bytes: &[u8; 32]) -> String {
    bytes[..8].iter().map(|b| format!("{:02x}", b)).collect()
}

fn main() {
    println!("=== CommittedFieldBlend — Three Archetypes × 100 Entities ===");
    println!("=== Plan 321 T4.1 — sampling invariance under fog-of-war ===\n");

    println!(
        "Config: K=3 archetypes, D=32 state, N={} entities, rollout={} steps ({} periods), sparse stride={}",
        N_ENTITIES,
        ROLLEN,
        ROLLEN / PERIOD,
        SPARSE_STRIDE
    );
    println!("Tolerances: pi_diff < {PI_TOL:.e}, traj_diff < {TRAJ_TOL:.e}\n");

    // Construct the three archetype fields (frozen for the run).
    let aggressive = AggressiveField::new(0.8, 0);
    let cautious = CautiousField::new(8, 16, 0.3);
    let social_bias = {
        let mut b = [0.0f32; 32];
        for j in 0..32 {
            b[j] = 0.02 + 0.003 * (j as f32);
        }
        b
    };
    let social = SocialField::new(social_bias);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&aggressive, &cautious, &social];

    let dirs = make_direction_vectors();

    // Run all N_ENTITIES entities.
    let mut reports = Vec::with_capacity(N_ENTITIES);
    let mut max_pi_diff_all = 0.0f32;
    let mut max_traj_diff_all = 0.0f32;
    let mut n_pi_pass = 0usize;
    let mut n_traj_pass = 0usize;
    for idx in 0..N_ENTITIES {
        let r = run_entity(idx, &dirs, &fields);
        if r.max_pi_diff < PI_TOL {
            n_pi_pass += 1;
        }
        if r.max_traj_diff < TRAJ_TOL {
            n_traj_pass += 1;
        }
        max_pi_diff_all = max_pi_diff_all.max(r.max_pi_diff);
        max_traj_diff_all = max_traj_diff_all.max(r.max_traj_diff);
        reports.push(r);
    }

    // Show three sample entities (first, middle, last).
    let sample_indices = [0usize, N_ENTITIES / 2, N_ENTITIES - 1];
    println!("── Sample entities (first / middle / last) ──");
    for &i in &sample_indices {
        let r = &reports[i];
        println!("entity {:3}:", r.idx);
        println!(
            "    pi_dense  = [{:+.4}, {:+.4}, {:+.4}]",
            r.pi_dense[0], r.pi_dense[1], r.pi_dense[2]
        );
        println!(
            "    pi_sparse = [{:+.4}, {:+.4}, {:+.4}]",
            r.pi_sparse[0], r.pi_sparse[1], r.pi_sparse[2]
        );
        println!(
            "    max |Δpi|     = {:.2e}   max |Δtraj| = {:.2e}   L_pi = {:.4}",
            r.max_pi_diff, r.max_traj_diff, r.lipschitz
        );
        println!("    blake3        = {}", hex_short(&r.blake3));
    }

    // Population verdict.
    println!("\n── Population verdict ({N_ENTITIES} entities) ──");
    println!(
        "  sampling invariance on pi:     {n_pi_pass}/{N_ENTITIES} pass ({:.1}%)",
        100.0 * (n_pi_pass as f32) / (N_ENTITIES as f32)
    );
    println!(
        "  trajectory invariance:         {n_traj_pass}/{N_ENTITIES} pass ({:.1}%)",
        100.0 * (n_traj_pass as f32) / (N_ENTITIES as f32)
    );
    println!("  worst-case max |Δpi|     over population: {max_pi_diff_all:.2e}");
    println!("  worst-case max |Δtraj|   over population: {max_traj_diff_all:.2e}");

    // Sanity asserts (also catch a regression in the primitive).
    assert!(
        n_pi_pass == N_ENTITIES,
        "all entities must pass the pi invariance gate, only {n_pi_pass}/{N_ENTITIES} did"
    );
    assert!(
        n_traj_pass == N_ENTITIES,
        "all entities must pass the trajectory invariance gate, only {n_traj_pass}/{N_ENTITIES} did"
    );

    // Population diversity sanity: entities should NOT all commit identical pi
    // (they have distinct hidden trajectories → distinct personalities). If
    // they did, the test would be vacuous.
    let mut distinct_pi = std::collections::HashSet::new();
    for r in &reports {
        // Quantize to 4 decimal places for the distinctness check.
        let key = (
            (r.pi_dense[0] * 1e4).round() as i64,
            (r.pi_dense[1] * 1e4).round() as i64,
            (r.pi_dense[2] * 1e4).round() as i64,
        );
        distinct_pi.insert(key);
    }
    println!(
        "\n  distinct committed pi vectors: {}/{}",
        distinct_pi.len(),
        N_ENTITIES
    );
    assert!(
        distinct_pi.len() >= 50,
        "expected ≥50 distinct personalities across 100 entities, got {} — \
         test would be vacuous otherwise",
        distinct_pi.len()
    );

    println!("\n✓ FAME Proposition 3 holds: every entity's committed personality");
    println!("  is invariant under fog-of-war observation gaps (dense vs sparse");
    println!("  summaries of the same underlying trajectory → identical pi and");
    println!("  identical post-commit dynamics).");
    println!("\n=== Demo complete ===");
}
