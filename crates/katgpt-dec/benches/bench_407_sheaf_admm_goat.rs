//! Plan 407 Phase 2 — Sheaf-ADMM GOAT perf gates (G4 latency, G5 zero-alloc).
//!
//! Exercises the two perf gates for `sheaf_admm_step`:
//!
//! - **G4 (latency)** — one `sheaf_admm_step` call, K=100 vertices, d_v=8,
//!   d_e=5, T=5 diffusion steps. Target: mean latency < 5 µs.
//! - **G5 (zero-alloc)** — 0 allocations in steady state per `sheaf_admm_step`
//!   call (after warmup). The scratch buffers are pre-allocated; the `_into`
//!   entry point is zero-alloc by contract.
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/sheaf_admm_phase2 \
//! cargo bench -p katgpt-dec --features sheaf_admm --no-default-features \
//!   --bench bench_407_sheaf_admm_goat -- --nocapture
//! ```
//!
//! (If `cargo bench` stalls on macOS due to dyld/trustd, run the binary
//! directly: `target/release/deps/bench_407_sheaf_admm_goat-* --nocapture`.)

#![cfg(feature = "sheaf_admm")]

use katgpt_dec::{
    AdmmScratch, CellComplex, CochainField, LocalObjective, SheafMaps, sheaf_admm_step,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Inline CountingAllocator (katgpt-dec has no tests/common/counting_allocator
// macro — we inline the pattern here per Plan 407).
// ---------------------------------------------------------------------------

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

// ---------------------------------------------------------------------------
// SplitMix64 PRNG (deterministic, no external dep)
// ---------------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let bits = z >> 40;
        let u01 = (bits as f32) / ((1u64 << 24) as f32);
        u01 * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

/// Build the G4/G5 workload: K=100 vertices (10×10 grid), d_v=8, d_e=5, T=5.
fn build_workload() -> (
    CellComplex,
    SheafMaps,
    CochainField,
    CochainField,
    CochainField,
    LocalObjective,
    AdmmScratch,
) {
    let cx = CellComplex::grid_2d(10, 10);
    let d_v = 8usize;
    let d_e = 5usize;
    let maps = SheafMaps::identity(&cx, d_v, d_e);
    let total = cx.n_vertices() * d_v;

    let mut rng = SplitMix64::new(0xBADC_407A_6026_0707); // deterministic seed
    let mut primal_x = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut consensus_z = CochainField::zeros(0, cx.n_vertices(), d_v);
    let mut dual_u = CochainField::zeros(0, cx.n_vertices(), d_v);
    for k in 0..total {
        primal_x.data[k] = rng.next_f32();
        consensus_z.data[k] = rng.next_f32();
        dual_u.data[k] = rng.next_f32() * 0.1;
    }
    let objective = LocalObjective::DiagonalQuadratic {
        diag_q: vec![1.0; total],
        q: vec![-0.5; total],
    };
    let scratch = AdmmScratch::new(&cx, d_v, d_e);
    (cx, maps, primal_x, consensus_z, dual_u, objective, scratch)
}

const RHO: f32 = 1.0;
const ETA: f32 = 0.2;
const T_STEPS: usize = 5;

// ===========================================================================
// G4 — latency: mean per-call latency < 5 µs
// ===========================================================================

fn g4_latency() -> (f64, bool) {
    let (cx, maps, mut primal_x, mut consensus_z, mut dual_u, objective, mut scratch) =
        build_workload();

    // Warmup: 10 iterations (not timed). Stabilizes caches, branch predictor.
    for _ in 0..10 {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            RHO,
            ETA,
            T_STEPS,
            &mut scratch,
        );
    }

    // Measure: 1000 iterations, total elapsed time.
    let iters = 1000usize;
    let start = Instant::now();
    for _ in 0..iters {
        sheaf_admm_step(
            black_box(&cx),
            black_box(&maps),
            black_box(&mut primal_x),
            black_box(&mut consensus_z),
            black_box(&mut dual_u),
            black_box(&objective),
            RHO,
            ETA,
            T_STEPS,
            black_box(&mut scratch),
        );
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() as f64 / iters as f64;
    let per_call_us = per_call_ns / 1000.0;

    // Gate: < 5 µs (5000 ns).
    let pass = per_call_ns < 5000.0;
    (per_call_us, pass)
}

// ===========================================================================
// G5 — zero-alloc: 0 allocations in steady state
// ===========================================================================

fn g5_zero_alloc() -> (usize, bool) {
    let (cx, maps, mut primal_x, mut consensus_z, mut dual_u, objective, mut scratch) =
        build_workload();

    // Warmup: 1 iteration. Allocations during warmup are allowed (scratch setup,
    // lazy initialization, etc.). The contract is zero-alloc in STEADY STATE.
    sheaf_admm_step(
        &cx,
        &maps,
        &mut primal_x,
        &mut consensus_z,
        &mut dual_u,
        &objective,
        RHO,
        ETA,
        T_STEPS,
        &mut scratch,
    );

    // Snapshot alloc counter after warmup.
    let before = ALLOC_COUNT.load(Ordering::Relaxed);

    // Measured run: 100 iterations.
    for _ in 0..100 {
        sheaf_admm_step(
            &cx,
            &maps,
            &mut primal_x,
            &mut consensus_z,
            &mut dual_u,
            &objective,
            RHO,
            ETA,
            T_STEPS,
            &mut scratch,
        );
    }

    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let delta = after - before;

    // Gate: 0 allocations in steady state.
    let pass = delta == 0;
    (delta, pass)
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn verdict(pass: bool) -> &'static str {
    if pass {
        "PASS ✅"
    } else {
        "FAIL ❌"
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 407 — Sheaf-ADMM GOAT Perf Gate (G4 latency, G5 zero-alloc) ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");
    println!();

    let mut all_pass = true;

    let (lat_us, g4) = g4_latency();
    println!(
        "G4 latency (K=100, d_v=8, d_e=5, T=5): mean = {lat_us:.3} µs  (gate < 5.0)  → {}",
        verdict(g4)
    );
    all_pass &= g4;

    let (allocs, g5) = g5_zero_alloc();
    println!(
        "G5 zero-alloc (100 calls, steady state): allocs = {allocs}  (gate = 0)  → {}",
        verdict(g5)
    );
    all_pass &= g5;

    println!();
    if all_pass {
        println!("══ Phase 2 PERF GATES PASS — G4 + G5 both pass ══");
    } else {
        println!("══ Phase 2 PERF GATES FAIL — see above ══");
    }
    std::process::exit(if all_pass { 0 } else { 1 });
}
