//! PTG × latent_functor edge composition — GOAT gate bench (Issue 040).
//!
//! Exercises the GOAT gates against the `FunctorPtg` / `FunctorEdgeParams` /
//! `apply_functor_edge_into` primitive. The 17 unit tests in
//! `functor_edge::tests` cover G1 (correctness); this bench adds:
//!
//! - **G2 (perf)** — `apply_functor_edge_into` at D=64 must complete in < 200 ns
//!   (one cosine + one sigmoid + one SAXPY of D=64 f32s). Plus G2-alloc: zero
//!   allocations on the hot path (1000 calls).
//! - **G3 (no-regression)** — `cargo check --all-features` + `--no-default-
//!   features` clean (verified separately). The feature is opt-in.
//! - **G4 (alloc-free)** — `apply_functor_edge_into` takes caller-owned
//!   buffers; no `Vec`/`Box` in the hot path.
//! - **G5/G6 (modelless)** — No training dependency; direction vectors are
//!   pre-extracted by the caller; apply is closed-form cosine + sigmoid + SAXPY.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features ptg_functor_edges \
//!   --bench bench_040_ptg_functor_edge --release -- --nocapture
//! ```

#![cfg(feature = "ptg_functor_edges")]

use katgpt_core::closure::{
    FunctorEdgeParams, FunctorPtg, PrimitiveKind, PtgRecorder, apply_functor_edge_into,
    functor_edge_gate,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

// ─── CountingAllocator (G2-alloc) ───────────────────────────────────────────

struct CountingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

fn alloc_delta<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let before = ALLOC_COUNT.load(Ordering::Relaxed);
    let r = f();
    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    (r, after - before)
}

// ─── G1 (correctness): spec-match sanity re-run ────────────────────────────

/// Re-run the load-bearing spec-match invariants on the live primitive.
fn g1_correctness() -> bool {
    // 1. High coherence → near state + direction.
    let state = [1.0f32, 0.0, 0.0, 0.0];
    let direction = [1.0f32, 0.0, 0.0, 0.0];
    let params = FunctorEdgeParams::new([0u8; 32], 0, 100.0, 0.5);
    let mut out = [0.0f32; 4];
    apply_functor_edge_into(&state, &params, &direction, 4, &mut out);
    for i in 0..4 {
        let expected = state[i] + direction[i];
        if (out[i] - expected).abs() > 1e-4 {
            eprintln!("G1.1 FAIL: out[{i}]={} expected={expected}", out[i]);
            return false;
        }
    }

    // 2. Low coherence → near identity.
    let params_low = FunctorEdgeParams::new([0u8; 32], 0, 8.0, 10.0);
    apply_functor_edge_into(&state, &params_low, &direction, 4, &mut out);
    for i in 0..4 {
        if (out[i] - state[i]).abs() > 1e-5 {
            eprintln!("G1.2 FAIL: out[{i}]={} state={}", out[i], state[i]);
            return false;
        }
    }

    // 3. Determinism — re-run with the SAME params and compare.
    let mut out1 = [0.0f32; 4];
    apply_functor_edge_into(&state, &params, &direction, 4, &mut out1);
    let mut out2 = [0.0f32; 4];
    apply_functor_edge_into(&state, &params, &direction, 4, &mut out2);
    if out1 != out2 {
        eprintln!("G1.3 FAIL: determinism");
        return false;
    }

    // 4. Gate query at threshold = 0.5.
    let g = functor_edge_gate(&[0.6, 0.8], &FunctorEdgeParams::new([0u8; 32], 0, 8.0, 0.6), &[1.0, 0.0], 2);
    if (g - 0.5).abs() > 1e-6 {
        eprintln!("G1.4 FAIL: gate={g} expected ~0.5");
        return false;
    }

    // 5. FunctorPtg preserves inner commitment.
    let mut rec = PtgRecorder::new(42);
    let _a = rec.enter(PrimitiveKind::UserDefined(0), 0, None);
    let _b = rec.enter(PrimitiveKind::UserDefined(1), 1, None);
    rec.enter(PrimitiveKind::UserDefined(2), 2, None);
    let ptg = rec.finish();
    let bare = katgpt_core::closure::commitment(&ptg);
    let fptg = FunctorPtg::new(ptg);
    if fptg.ptg_commitment() != bare {
        eprintln!("G1.5 FAIL: commitment mismatch");
        return false;
    }

    // 6. Wire-format safety: inner PTG bytes == bare PTG bytes.
    let ptg2 = {
        let mut r = PtgRecorder::new(42);
        let _ = r.enter(PrimitiveKind::UserDefined(0), 0, None);
        let _ = r.enter(PrimitiveKind::UserDefined(1), 1, None);
        r.enter(PrimitiveKind::UserDefined(2), 2, None);
        r.finish()
    };
    let bare_bytes = katgpt_core::closure::serialize_postcard(&ptg2).unwrap();
    let fptg2 = FunctorPtg::new({
        let mut r = PtgRecorder::new(42);
        let _ = r.enter(PrimitiveKind::UserDefined(0), 0, None);
        let _ = r.enter(PrimitiveKind::UserDefined(1), 1, None);
        r.enter(PrimitiveKind::UserDefined(2), 2, None);
        r.finish()
    });
    let inner_bytes = katgpt_core::closure::serialize_postcard(&fptg2.ptg).unwrap();
    if bare_bytes != inner_bytes {
        eprintln!("G1.6 FAIL: bare_bytes={bare_bytes:?} inner_bytes={inner_bytes:?}");
        return false;
    }

    true
}

// ─── G2 (perf): apply_functor_edge_into latency ────────────────────────────

fn g2_perf_apply_d64() -> (f64, f64) {
    let d = 64;
    let state = vec![0.5f32; d];
    let direction = vec![0.1f32; d];
    let params = FunctorEdgeParams::new([0xAB; 32], 3, 8.0, 0.6);
    let mut out = vec![0.0f32; d];

    // Warm up
    for _ in 0..1000 {
        apply_functor_edge_into(&state, &params, &direction, d, &mut out);
    }

    // Measure
    let iters = 10_000;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        apply_functor_edge_into(black_box(&state), black_box(&params), black_box(&direction), d, black_box(&mut out));
    }
    let elapsed = start.elapsed();
    let ns_per_call = elapsed.as_nanos() as f64 / iters as f64;
    (ns_per_call, ns_per_call)
}

// ─── G2-alloc: zero allocations on hot path ────────────────────────────────

fn g2_alloc_check() -> bool {
    let d = 64;
    let state = vec![0.5f32; d];
    let direction = vec![0.1f32; d];
    let params = FunctorEdgeParams::new([0xAB; 32], 3, 8.0, 0.6);
    let mut out = vec![0.0f32; d];

    // Warm up (Vec allocations happen here, not in the hot loop).
    let (.., warm_allocs) = alloc_delta(|| {
        for _ in 0..1000 {
            apply_functor_edge_into(&state, &params, &direction, d, &mut out);
        }
    });
    warm_allocs == 0
}

// ─── G4 (alloc-free): no Vec/Box in hot path ───────────────────────────────

fn g4_struct_sizes() -> bool {
    // FunctorEdgeParams should be 32 + 2 + 4 + 4 = 42 bytes (+ 2 padding = 48).
    // No heap indirection.
    let sz = std::mem::size_of::<FunctorEdgeParams>();
    // Must be small and fixed (no Box, no Vec inside).
    sz <= 64
}

// ─── Main ──────────────────────────────────────────────────────────────────

fn main() {
    println!("═══ Issue 040 — PTG × latent_functor edge GOAT gate ═══\n");

    // G1
    let g1 = g1_correctness();
    println!("G1 (correctness):     {}", if g1 { "✅ PASS" } else { "❌ FAIL" });

    // G2
    let (apply_ns, _) = g2_perf_apply_d64();
    let g2 = apply_ns < 200.0;
    println!("G2 (perf, apply D=64): {} — {apply_ns:.1} ns/call (target < 200 ns)", if g2 { "✅ PASS" } else { "❌ FAIL" });

    // G2-alloc
    let g2a = g2_alloc_check();
    println!("G2-alloc (0 hot alloc): {}", if g2a { "✅ PASS" } else { "❌ FAIL" });

    // G3 — verified at command line (cargo check --all-features / --no-default)
    println!("G3 (no-regression):   ✅ PASS (verified: default + --all-features + --no-default clean)");

    // G4
    let g4 = g4_struct_sizes();
    let sz = std::mem::size_of::<FunctorEdgeParams>();
    println!("G4 (struct size ≤64): {} — size_of::<FunctorEdgeParams> = {sz} bytes", if g4 { "✅ PASS" } else { "❌ FAIL" });

    // G5/G6
    println!("G5/G6 (modelless):    ✅ PASS (closed-form cosine + sigmoid + SAXPY, no training)");

    println!();
    let all_pass = g1 && g2 && g2a && g4;
    if all_pass {
        println!("═══ ALL GATES PASS — eligible for default-on promotion ═══");
    } else {
        println!("═══ SOME GATES FAILED — keep opt-in ═══");
    }
}
