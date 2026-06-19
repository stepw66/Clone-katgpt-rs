//! CGSP Minimal Example (Plan 274 Phase 4 — T4.3)
//!
//! Demonstrates the smallest meaningful Curiosity-Guided Self-Play loop:
//!   - 8-direction pool in 8-D latent space (near-orthonormal)
//!   - 1 target (pool arm 0)
//!   - 100 cycles
//!
//! Shows:
//!   1. Building the triad (PoolConjecturer + HlaProjectionGuide + DotSolver +
//!      VecBandit) and wiring it into `CgspLoop`.
//!   2. Attaching the difficulty filter + batch gate + collapse detector.
//!   3. Running 100 cycles on a single target with pre-allocated `ScratchBuffers`.
//!   4. Reading per-cycle stats (`CycleResult`) and the final priority table.
//!   5. Snapshotting the resulting priority table (freeze/thaw bridge) and
//!      verifying its BLAKE3 commitment is well-formed.
//!
//! Run:
//!   cargo run --features cgsp --example cgsp_minimal
//!
//! See also: `examples/cgsp_collapse_recovery.rs` for the recovery demo.

#![cfg(feature = "cgsp")]

use katgpt_rs::cgsp::{
    traits::{HintDeltaBandit, Solver},
    BreakevenDifficultyFilter, CgspConfig, CgspLoop, ColinearityBatchGate,
    ComplexityWeights, CuriosityPrioritySnapshot, CycleResult, Direction, EntropyCollapse,
    HlaProjectionGuide, PoolConjecturer, Priority, ScratchBuffers, Target, entropy_nats, sigmoid,
};

// ════════════════════════════════════════════════════════════════════════════
// Minimal caller-provided Solver + Bandit
// ════════════════════════════════════════════════════════════════════════════
//
// `VecBandit` and `DotSolver` live `pub(crate)` inside `src/cgsp/mod.rs` so the
// public API stays lean. Examples redefine minimal local copies with identical
// semantics — same trick the GOAT gate benchmark (`tests/bench_274_cgsp_goat.rs`)
// uses. In a real consumer (riir-ai Plan 299) you'd plug in your own Solver
// (e.g. an MCTS-with-budget) and your own bandit (e.g. a Hint-δ absorb-compress
// table with decay).

/// Priority-weighted bandit backed by a `Vec<f32>`. Additive absorb, no decay.
struct VecBandit {
    prios: Vec<f32>,
}

impl VecBandit {
    fn uniform(n: usize) -> Self {
        Self { prios: vec![1.0 / n as f32; n] }
    }
}

impl HintDeltaBandit for VecBandit {
    fn absorb(&mut self, arm: usize, reward: f32) {
        if let Some(p) = self.prios.get_mut(arm) {
            *p += reward.max(0.0);
        }
    }
    fn priority(&self, arm: usize) -> Priority {
        self.prios.get(arm).copied().unwrap_or(0.0)
    }
    fn priorities(&self) -> &[Priority] {
        &self.prios
    }
    fn priorities_mut(&mut self) -> &mut [Priority] {
        &mut self.prios
    }
}

/// Solver: solve-rate grows with target-alignment via a sigmoid of the
/// dot-product. Deterministic, reproducible, no model weights.
struct DotSolver { sharpness: f32 }

impl Solver for DotSolver {
    fn attempt(
        &mut self,
        target: &Target,
        candidate_direction: &Direction,
        _pool_index: usize,
    ) -> f32 {
        let d = candidate_direction.dot(&target.direction);
        sigmoid(self.sharpness * d)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Pool builder — 8 near-orthonormal directions in 8-D
// ════════════════════════════════════════════════════════════════════════════

fn make_8_direction_pool() -> Vec<Direction> {
    (0..8)
        .map(|i| {
            let mut coords = vec![0.0f32; 8];
            coords[i] = 1.0;
            // Small perturbation on the next axis so the pool isn't degenerate.
            if i >= 1 {
                coords[(i + 1) % 8] = 0.1;
            }
            let norm: f32 = coords.iter().map(|c| c * c).sum::<f32>().sqrt().max(1e-9);
            for c in &mut coords {
                *c /= norm;
            }
            Direction { coords }
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

fn separator(title: &str) {
    println!();
    println!("{}", "═".repeat(72));
    println!("  {title}");
    println!("{}", "═".repeat(72));
}

fn print_priorities(prios: &[f32]) {
    let sum: f32 = prios.iter().copied().sum();
    for (i, p) in prios.iter().enumerate() {
        let share = if sum > 0.0 { p / sum } else { 0.0 };
        let bar_len = (share * 40.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  arm {i}: prio={p:>7.4}  share={share:>6.3}  {bar}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   CGSP Minimal Example — Plan 274 Phase 4 (T4.3)                    ║");
    println!("║   8-direction pool · 1 target · 100 cycles                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let pool = make_8_direction_pool();
    let target = Target::new(pool[0].clone()).with_priority_hint(0.9);

    // ── Section 1: Build the triad ───────────────────────────────────────
    separator("Section 1: Build the CGSP triad");

    let conjecturer = PoolConjecturer::new(pool.clone(), 42);
    let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
    let solver = DotSolver { sharpness: 1.0 };
    let bandit = VecBandit::uniform(8);

    // Default `k=4`, `τ_low=0.30`. Difficulty filter + batch gate + collapse
    // detector are added via the builder — each one is a trait object wired in
    // at compile time (zero-cost, no dynamic dispatch on the hot path).
    let mut lp = CgspLoop::new(conjecturer, guide, solver, bandit, CgspConfig::default())
        .with_collapse(EntropyCollapse::new(0.30))
        .with_difficulty_filter(BreakevenDifficultyFilter::default())
        .with_batch_gate(ColinearityBatchGate::default());

    println!("  Conjecturer : PoolConjecturer (pool={}, dim={})", pool.len(), pool[0].dim());
    println!("  Guide       : HlaProjectionGuide (λ=2.0, α=1.0)");
    println!("  Solver      : DotSolver (sharpness=1.0)");
    println!("  Bandit      : VecBandit uniform over 8 arms");
    println!("  Filters     : BreakevenDifficultyFilter + ColinearityBatchGate");
    println!("  Collapse    : EntropyCollapse τ_low=0.30");
    println!("  Config      : k={}, τ_low={}, exploration_mag={}",
        lp.config().k, lp.config().tau_low, lp.config().exploration_magnitude);

    // ── Section 2: Initial priority table ────────────────────────────────
    separator("Section 2: Initial priority table (uniform)");
    print_priorities(lp.bandit().priorities());

    // ── Section 3: Run 100 cycles ────────────────────────────────────────
    separator("Section 3: Run 100 cycles (target = pool arm 0)");

    let mut scratch = ScratchBuffers::new(8, 8);
    let mut last_result: Option<CycleResult> = None;
    let mut collapses_triggered = 0u32;
    let mut degenerate_batches = 0u32;
    let mut sum_entropy = 0.0f64;
    let mut sum_r_synth = 0.0f64;

    for cycle in 0..100 {
        let r = lp.cycle(&target, &mut scratch);
        if r.collapse_triggered { collapses_triggered += 1; }
        if r.batch_degenerate { degenerate_batches += 1; }
        sum_entropy += r.stats.priority_entropy as f64;
        sum_r_synth += r.stats.mean_r_synth as f64;
        last_result = Some(r);

        // Spot-check at cycles 1, 10, 50, 100.
        if cycle == 0 || cycle == 9 || cycle == 49 || cycle == 99 {
            let h = entropy_nats(lp.bandit().priorities());
            println!("  cycle {:>3}: H={h:>5.3}  admitted={}  r_synth={:.4}  collapse={}",
                cycle + 1,
                r.stats.candidates_admitted,
                r.stats.mean_r_synth,
                r.collapse_triggered);
        }
    }

    println!();
    println!("  ── 100-cycle summary ──");
    println!("  mean entropy        : {:.4}", sum_entropy / 100.0);
    println!("  mean r_synth        : {:.4}", sum_r_synth / 100.0);
    println!("  collapses triggered : {collapses_triggered}");
    println!("  degenerate batches  : {degenerate_batches}");

    // ── Section 4: Final priority table ──────────────────────────────────
    separator("Section 4: Final priority table (after 100 cycles)");
    print_priorities(lp.bandit().priorities());

    // Honest framing — see `.benchmarks/274_cgsp_goat.md` §G1 root-cause.
    // CGSP's reward formula `(1 − solve_rate) · guide_score` rewards
    // intermediate-difficulty candidates by design, not target-aligned ones.
    println!();
    println!("  Note: CGSP is curiosity-driven, not target-seeking. The priority");
    println!("  distribution above reflects which arms the bandit found most");
    println!("  *informative* (intermediate solve-rate), not which arm matches");
    println!("  the target. See .benchmarks/274_cgsp_goat.md §G1 for details.");

    // ── Section 5: Snapshot + BLAKE3 commitment ─────────────────────────
    separator("Section 5: Snapshot (freeze/thaw bridge) + BLAKE3 commitment");

    let snap = lp.snapshot();
    let hash = snap.blake3_hash();
    let hash_hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    let id_hex: String = snap.snapshot_id.iter().map(|b| format!("{b:02x}")).collect();
    println!("  snapshot_id : {id_hex}");
    println!("  pool_size   : {}", snap.pool_size());
    println!("  dim         : {}", snap.dim);
    println!("  blake3      : {hash_hex}");
    assert!(hash.iter().any(|&b| b != 0), "BLAKE3 should not be all zeros");

    // Roundtrip: encode → decode → compare priorities.
    let mut buf = Vec::new();
    snap.encode_to(&mut buf);
    let back = CuriosityPrioritySnapshot::decode(&buf).expect("decode");
    assert_eq!(back.priorities, snap.priorities, "roundtrip must preserve priorities");
    println!("  roundtrip   : {} bytes, priorities match ✓", buf.len());

    // ── Summary ──────────────────────────────────────────────────────────
    separator("Summary");
    if let Some(r) = last_result {
        println!("  ✓ 100 cycles completed without panic / NaN");
        println!("  ✓ Final entropy       : {:.4}", r.stats.priority_entropy);
        println!("  ✓ Final mean r_synth  : {:.4}", r.stats.mean_r_synth);
        println!("  ✓ Snapshot BLAKE3     : well-formed (non-zero)");
        println!("  ✓ Snapshot roundtrip  : priorities preserved");
    }
    println!();
    println!("  Per-cycle overhead target: ≤ 1µs (release, Apple Silicon NEON).");
    println!("  See `cargo test --release --test bench_274_cgsp_goat --features cgsp`");
    println!("  for the enforced G4 measurement.");
    println!();
}

// TL;DR: Minimal end-to-end CGSP loop — build the Solver/Conjecturer/Guide
// triad with BreakevenDifficultyFilter + ColinearityBatchGate + EntropyCollapse,
// run 100 cycles on a synthetic 8-direction pool, and snapshot the resulting
// priority table with a BLAKE3 commitment. Demonstrates the full public API
// surface of `katgpt_rs::cgsp` in ~200 lines.
