//! CGSP Collapse Recovery Example (Plan 274 Phase 4 — T4.4)
//!
//! Demonstrates CGSP's defining property: **collapse recovery**.
//!
//! After artificially forcing the priority table into a one-hot state
//! (single arm has all the mass, entropy ≈ 0), the `EntropyCollapse` detector
//! injects exploration on the next cycle. Priority entropy climbs back above
//! `τ_low` within a handful of cycles — typically 1.
//!
//! This is the asymmetric proof from GOAT gate G2 (see `.benchmarks/274_cgsp_goat.md`):
//!   - With `EntropyCollapse`    : recovery in ~1 cycle
//!   - Without (baseline)        : stays collapsed for 200+ cycles
//!
//! Run:
//!   cargo run --features cgsp --example cgsp_collapse_recovery

#![cfg(feature = "cgsp")]

use katgpt_rs::cgsp::{
    traits::{CollapseSignal, HintDeltaBandit, Solver},
    BreakevenDifficultyFilter, CgspConfig, CgspLoop, ColinearityBatchGate,
    ComplexityWeights, CycleResult, Direction, EntropyCollapse, HlaProjectionGuide,
    NoOpBatchGate, NoOpDifficultyFilter, PoolConjecturer, Priority, ScratchBuffers, Target,
    entropy_nats, sigmoid,
};

// ════════════════════════════════════════════════════════════════════════════
// Minimal caller-provided Solver + Bandit (same as cgsp_minimal.rs)
// ════════════════════════════════════════════════════════════════════════════

struct VecBandit { prios: Vec<f32> }

impl VecBandit {
    fn uniform(n: usize) -> Self { Self { prios: vec![1.0 / n as f32; n] } }
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
    fn priorities(&self) -> &[Priority] { &self.prios }
    fn priorities_mut(&mut self) -> &mut [Priority] { &mut self.prios }
}

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

/// Collapse detector that NEVER fires — used by the baseline arm to disable
/// exploration injection. Mirrors `NeverCollapse` in the GOAT benchmark.
#[derive(Default)]
struct NeverCollapse;

impl CollapseSignal for NeverCollapse {
    fn check_collapse(&mut self, _p: &[Priority], _r: &CycleResult) -> bool { false }
    fn inject_exploration(&mut self, _p: &mut [Priority], _m: f32) {}
}

// ════════════════════════════════════════════════════════════════════════════
// Pool builder — 8 near-orthonormal directions in 8-D
// ════════════════════════════════════════════════════════════════════════════

fn make_8_direction_pool() -> Vec<Direction> {
    (0..8)
        .map(|i| {
            let mut coords = vec![0.0f32; 8];
            coords[i] = 1.0;
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

fn print_priorities(label: &str, prios: &[f32]) {
    let _ = label;
    let sum: f32 = prios.iter().copied().sum();
    for (i, p) in prios.iter().enumerate() {
        let share = if sum > 0.0 { p / sum } else { 0.0 };
        let bar_len = (share * 40.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!("  arm {i}: prio={p:>7.4}  share={share:>6.3}  {bar}");
    }
}

/// Force the priority table into a one-hot state on `collapsed_arm`.
fn force_collapse<C, G, S, B, Col, Df, Qg>(lp: &mut CgspLoop<C, G, S, B, Col, Df, Qg>, collapsed_arm: usize)
where
    C: katgpt_rs::cgsp::traits::CuriosityConjecturer,
    G: katgpt_rs::cgsp::traits::QualityGuide,
    S: Solver,
    B: HintDeltaBandit,
    Col: CollapseSignal,
    Df: katgpt_rs::cgsp::traits::DifficultyFilter,
    Qg: katgpt_rs::cgsp::traits::BatchQualityGate,
{
    for (i, p) in lp.bandit_mut().priorities_mut().iter_mut().enumerate() {
        *p = if i == collapsed_arm { 1.0 } else { 0.0 };
    }
}

/// Run cycles until entropy climbs back above `tau_low`, or `cap` cycles elapse.
/// Returns the number of cycles needed (1-indexed).
fn count_cycles_to_recover<C, G, S, B, Col, Df, Qg>(
    lp: &mut CgspLoop<C, G, S, B, Col, Df, Qg>,
    target: &Target,
    scratch: &mut ScratchBuffers,
    tau_low: f32,
    cap: usize,
) -> (usize, bool)
where
    C: katgpt_rs::cgsp::traits::CuriosityConjecturer,
    G: katgpt_rs::cgsp::traits::QualityGuide,
    S: Solver,
    B: HintDeltaBandit,
    Col: CollapseSignal,
    Df: katgpt_rs::cgsp::traits::DifficultyFilter,
    Qg: katgpt_rs::cgsp::traits::BatchQualityGate,
{
    for c in 0..cap {
        let _ = lp.cycle(target, scratch);
        let h = entropy_nats(lp.bandit().priorities());
        if h >= tau_low {
            return (c + 1, true);
        }
    }
    (cap, false)
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   CGSP Collapse Recovery — Plan 274 Phase 4 (T4.4)                  ║");
    println!("║   Force one-hot priority table → measure cycles to recover          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    const TAU_LOW: f32 = 0.30;
    const COLLAPSED_ARM: usize = 3;
    const RECOVERY_CAP: usize = 200;

    let pool = make_8_direction_pool();
    let target = Target::new(pool[0].clone());

    // ── Section 1: CGSP with collapse-aware ──────────────────────────────
    separator("Section 1: CGSP (with EntropyCollapse) — force one-hot, recover");

    let cgsp_loop = {
        let conj = PoolConjecturer::new(pool.clone(), 42);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        CgspLoop::new(conj, guide, solver, bandit, CgspConfig {
            tau_low: TAU_LOW,
            ..CgspConfig::default()
        })
        .with_collapse(EntropyCollapse::new(TAU_LOW))
        .with_difficulty_filter(BreakevenDifficultyFilter::default())
        .with_batch_gate(ColinearityBatchGate::default())
    };
    let mut lp_cgsp = cgsp_loop;
    let mut scratch_cgsp = ScratchBuffers::new(8, 8);

    force_collapse(&mut lp_cgsp, COLLAPSED_ARM);
    let h_collapsed = entropy_nats(lp_cgsp.bandit().priorities());
    println!("  Forced one-hot on arm {COLLAPSED_ARM}.");
    println!("  Collapsed entropy H = {h_collapsed:.6} nats (τ_low = {TAU_LOW})");
    print_priorities("collapsed", lp_cgsp.bandit().priorities());

    let (cycles_with, recovered_with) = count_cycles_to_recover(
        &mut lp_cgsp, &target, &mut scratch_cgsp, TAU_LOW, RECOVERY_CAP,
    );

    separator("After recovery attempt (CGSP with collapse-aware)");
    let h_after = entropy_nats(lp_cgsp.bandit().priorities());
    println!("  Cycles to recover : {cycles_with} (cap {RECOVERY_CAP})");
    println!("  Recovered         : {recovered_with}");
    println!("  Entropy now       : {h_after:.4} nats");
    print_priorities("recovered", lp_cgsp.bandit().priorities());

    // ── Section 2: Baseline (no collapse exploration) ────────────────────
    separator("Section 2: Baseline (NeverCollapse) — same setup, no exploration injection");

    let baseline_loop = {
        let conj = PoolConjecturer::new(pool.clone(), 42);
        let guide = HlaProjectionGuide::new(2.0, 1.0, ComplexityWeights::default());
        let solver = DotSolver { sharpness: 1.0 };
        let bandit = VecBandit::uniform(8);
        CgspLoop::new(conj, guide, solver, bandit, CgspConfig {
            tau_low: TAU_LOW,
            ..CgspConfig::default()
        })
        .with_collapse(NeverCollapse)
        .with_difficulty_filter(NoOpDifficultyFilter)
        .with_batch_gate(NoOpBatchGate)
    };
    let mut lp_base = baseline_loop;
    let mut scratch_base = ScratchBuffers::new(8, 8);

    force_collapse(&mut lp_base, COLLAPSED_ARM);
    let h_collapsed_b = entropy_nats(lp_base.bandit().priorities());
    println!("  Forced one-hot on arm {COLLAPSED_ARM}.");
    println!("  Collapsed entropy H = {h_collapsed_b:.6} nats");

    let (cycles_without, recovered_without) = count_cycles_to_recover(
        &mut lp_base, &target, &mut scratch_base, TAU_LOW, RECOVERY_CAP,
    );

    let h_after_b = entropy_nats(lp_base.bandit().priorities());
    println!("  Cycles to recover : {cycles_without} (cap {RECOVERY_CAP})");
    println!("  Recovered         : {recovered_without}");
    println!("  Entropy now       : {h_after_b:.4} nats");

    // ── Section 3: Asymmetric comparison ─────────────────────────────────
    separator("Section 3: Asymmetric comparison (the G2 proof)");

    let speedup = if cycles_with > 0 {
        cycles_without as f64 / cycles_with as f64
    } else {
        f64::INFINITY
    };
    println!("  ┌──────────────────────────────────────────────────────────────────┐");
    println!("  │ Config              │ Cycles to recover                          │");
    println!("  ├──────────────────────────────────────────────────────────────────┤");
    println!("  │ CGSP (collapse-aware)│ {cycles_with:>4}                                       │");
    println!("  │ Baseline (never)     │ {cycles_without:>4}                                       │");
    println!("  └──────────────────────────────────────────────────────────────────┘");
    println!();
    println!("  Speedup: {speedup:.1}× (CGSP with collapse-aware vs baseline)");

    // The enforced GOAT gate (G2) criterion: with ≤ 50, baseline ≥ 200.
    // The example is more relaxed — we just assert CGSP beats baseline.
    assert!(
        cycles_with <= cycles_without,
        "CGSP ({cycles_with}) should recover at least as fast as baseline ({cycles_without})"
    );
    assert!(recovered_with, "CGSP should recover within {RECOVERY_CAP} cycles");

    println!();
    println!("  ✓ CGSP with EntropyCollapse recovers in {cycles_with} cycle(s)");
    println!("  ✓ Baseline (NeverCollapse) takes {cycles_without} cycle(s)");
    if cycles_without >= RECOVERY_CAP && !recovered_without {
        println!("  ✓ Baseline stays collapsed for the full {RECOVERY_CAP}-cycle window");
    }
    println!();
    println!("  Why this matters: in a live NPC runtime, a degenerate conjecturer");
    println!("  (e.g. always proposing the same sub-goal) would freeze the agent.");
    println!("  CGSP's collapse-aware layer detects the entropy drop and re-injects");
    println!("  exploration mass automatically — no external coordinator needed.");
    println!();

    // ── Section 4: Mechanism walkthrough ─────────────────────────────────
    separator("Section 4: How EntropyCollapse works");

    println!("  1. After each cycle, `CgspLoop` computes H = entropy(priorities).");
    println!("  2. If H < τ_low ({TAU_LOW}), the collapse detector fires.");
    println!("  3. `inject_exploration(priorities, magnitude)` mixes the current");
    println!("     priority vector with the uniform distribution:");
    println!("       p'[i] = (1 − m) · p[i] + m · (1/N)");
    println!("     where m = exploration_magnitude (default 0.35).");
    println!("  4. The mixed table has H > τ_low, so the next cycle's conjecturer");
    println!("     samples diversely again — recovery is immediate.");
    println!();
    println!("  The cost: one entropy computation (O(N)) + one priority mix (O(N)).");
    println!("  For N=8 this is negligible — sub-microsecond even in debug builds.");
    println!();
}

// TL;DR: Demonstrates CGSP's defining property — collapse recovery. Forcing
// the priority table into a one-hot state and counting cycles until entropy
// returns above τ_low shows that `EntropyCollapse` recovers in ~1 cycle,
// while a `NeverCollapse` baseline stays collapsed for the full 200-cycle
// window. This is the asymmetric G2 proof from the GOAT gate benchmark.
