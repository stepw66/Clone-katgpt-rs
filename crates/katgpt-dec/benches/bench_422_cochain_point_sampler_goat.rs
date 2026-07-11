//! Plan 422 — Cochain Point Sampler GOAT perf gates (G4 zero-alloc, G5 latency).
//!
//! Exercises the two perf gates for the continuous cochain point sampler:
//!
//! - **G4 (zero-alloc)** — 0 allocations in steady state per
//!   `sample_cochain_at_point_quad_into` / `sample_point_tri_into` call (after
//!   warmup). The `*_into` paths use caller-provided slices / pre-allocated
//!   `PointSamplerScratch`; zero-alloc is the contract.
//! - **G5 (latency)** — single `sample_cochain_at_point_quad_into` query on a
//!   64×64 grid (bilinear Whitney 0-form reconstruction). Target: < 100 ns;
//!   gate at < 200 ns to be safe. Also reports the full `sample_point_quad_into`
//!   (CartesianSincos encoding) and `sample_point_tri_into` (BarycentricSortCdf)
//!   paths for visibility — these are not gated (the plan only gates the raw
//!   bilinear interp path).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/cochain_point_sampler_goat \
//! cargo bench -p katgpt-dec --features cochain_point_sampler --no-default-features \
//!   --bench bench_422_cochain_point_sampler_goat -- --nocapture
//! ```

#![cfg(feature = "cochain_point_sampler")]

use katgpt_dec::{
    CellComplex, CochainField, LocalCoordEncode, PointSamplerScratch,
    sample_cochain_at_point_quad_into, sample_point_quad_into, sample_point_tri_into,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Inline CountingAllocator (katgpt-dec benches inline the pattern per Plan 407)
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
    /// Returns f32 in [0, 1).
    fn next_u01(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        let bits = z >> 40; // top 24 bits → mantissa
        (bits as f32) / ((1u64 << 24) as f32)
    }
}

// ---------------------------------------------------------------------------
// Shared workload builders
// ---------------------------------------------------------------------------

/// Quad workload: 64×64 grid, dim=8 cochain field. Returns the pre-allocated
/// output buffer sized for `dim`.
fn build_quad_workload() -> (CellComplex, CochainField, [f32; 8]) {
    let (gw, gh) = (64usize, 64usize);
    let cx = CellComplex::grid_2d(gw, gh);
    let dim = 8usize;
    let mut field = CochainField::zeros(0, gw * gh, dim);

    // Deterministic but non-trivial field data.
    let mut rng = SplitMix64::new(0x4220_0710_2026); // 422, 2026-07-10
    for v in 0..gw * gh {
        let features = field.cell_features_mut(v);
        for d in 0..dim {
            features[d] = rng.next_u01() * 2.0 - 1.0;
        }
    }
    let out = [0.0f32; 8];
    (cx, field, out)
}

/// Triangle workload: single triangle [[0,0],[4,0],[0,4]], dim=8 field.
fn build_tri_workload() -> (
    CochainField,
    [[f32; 2]; 3],
    [usize; 3],
    PointSamplerScratch,
) {
    let dim = 8usize;
    let mut field = CochainField::zeros(0, 3, dim);
    let mut rng = SplitMix64::new(0x4221_0710_2026);
    for v in 0..3 {
        let features = field.cell_features_mut(v);
        for d in 0..dim {
            features[d] = rng.next_u01() * 2.0 - 1.0;
        }
    }
    let tri_pos = [[0.0f32, 0.0], [4.0, 0.0], [0.0, 4.0]];
    let tri_idx = [0usize, 1, 2];
    // BarycentricSortCdf → aug_dim = 3.
    let scratch = PointSamplerScratch::new(dim, 3);
    (field, tri_pos, tri_idx, scratch)
}

// ===========================================================================
// G4 — zero-alloc: 0 allocations in steady state
// ===========================================================================

fn g4_zero_alloc_quad() -> (usize, bool) {
    let (cx, field, mut out) = build_quad_workload();

    // Warmup: 1 iteration. Allocations during warmup are allowed.
    sample_cochain_at_point_quad_into(&cx, &field, 32.5, 32.5, &mut out);

    let before = ALLOC_COUNT.load(Ordering::Relaxed);

    // Measured run: 100 iterations at varied interior points.
    let mut px = 10.0f32;
    let mut py = 10.0f32;
    for _ in 0..100 {
        sample_cochain_at_point_quad_into(
            black_box(&cx),
            black_box(&field),
            black_box(px),
            black_box(py),
            black_box(&mut out),
        );
        // Drift the point to exercise different quads (stays interior on 64×64).
        px += 0.37;
        py += 0.29;
        if px > 60.0 {
            px = 5.0;
        }
        if py > 60.0 {
            py = 5.0;
        }
    }

    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let delta = after - before;
    (delta, delta == 0)
}

fn g4_zero_alloc_tri() -> (usize, bool) {
    let (field, tri_pos, tri_idx, mut scratch) = build_tri_workload();

    // Warmup: 1 iteration.
    sample_point_tri_into(
        &field,
        &tri_pos,
        tri_idx,
        1.0,
        1.0,
        LocalCoordEncode::BarycentricSortCdf,
        &mut scratch,
    );

    let before = ALLOC_COUNT.load(Ordering::Relaxed);

    // Measured run: 100 iterations at varied interior points.
    let mut px = 0.5f32;
    let mut py = 0.5f32;
    for _ in 0..100 {
        sample_point_tri_into(
            black_box(&field),
            black_box(&tri_pos),
            black_box(tri_idx),
            black_box(px),
            black_box(py),
            black_box(LocalCoordEncode::BarycentricSortCdf),
            black_box(&mut scratch),
        );
        // Small drift, stays inside [[0,0],[4,0],[0,4]] (x+y < 4).
        px += 0.03;
        py += 0.02;
        if px + py > 3.5 {
            px = 0.3;
            py = 0.3;
        }
    }

    let after = ALLOC_COUNT.load(Ordering::Relaxed);
    let delta = after - before;
    (delta, delta == 0)
}

// ===========================================================================
// G5 — latency: mean per-call latency
// ===========================================================================

/// Primary gate: `sample_cochain_at_point_quad_into` on 64×64 grid, dim=8.
/// Gate: < 200 ns (plan target < 100 ns).
fn g5_latency_quad_raw() -> (f64, bool) {
    let (cx, field, mut out) = build_quad_workload();

    // Warmup: 100 iterations (stabilize caches, branch predictor).
    let mut px = 10.0f32;
    let mut py = 10.0f32;
    for _ in 0..100 {
        sample_cochain_at_point_quad_into(&cx, &field, px, py, &mut out);
        px += 0.37;
        py += 0.29;
        if px > 60.0 {
            px = 5.0;
        }
        if py > 60.0 {
            py = 5.0;
        }
    }

    // Measure: 10_000 iterations.
    let iters = 10_000usize;
    let mut px = 10.0f32;
    let mut py = 10.0f32;
    let start = Instant::now();
    for _ in 0..iters {
        sample_cochain_at_point_quad_into(
            black_box(&cx),
            black_box(&field),
            black_box(px),
            black_box(py),
            black_box(&mut out),
        );
        px += 0.37;
        py += 0.29;
        if px > 60.0 {
            px = 5.0;
        }
        if py > 60.0 {
            py = 5.0;
        }
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() as f64 / iters as f64;

    // Gate: < 200 ns.
    let pass = per_call_ns < 200.0;
    (per_call_ns, pass)
}

/// Full quad path: `sample_point_quad_into` with CartesianSincos { n_harmonics: 4 }.
/// Report only (not gated — the plan only gates the raw bilinear interp).
fn g5_latency_quad_full_sincos() -> f64 {
    let (cx, field, _) = build_quad_workload();
    let dim = 8usize;
    let encode = LocalCoordEncode::CartesianSincos { n_harmonics: 4 };
    let aug_dim = katgpt_dec::local_coord_aug_dim(encode, 2); // 2 * 2 * 4 = 16
    let mut scratch = PointSamplerScratch::new(dim, aug_dim);

    // Warmup.
    for i in 0..100 {
        let p = 5.0 + (i as f32 * 0.5).rem_euclid(50.0);
        sample_point_quad_into(&cx, &field, p, p, encode, &mut scratch);
    }

    // Measure.
    let iters = 10_000usize;
    let mut px = 10.0f32;
    let mut py = 10.0f32;
    let start = Instant::now();
    for _ in 0..iters {
        sample_point_quad_into(
            black_box(&cx),
            black_box(&field),
            black_box(px),
            black_box(py),
            black_box(encode),
            black_box(&mut scratch),
        );
        px += 0.37;
        py += 0.29;
        if px > 60.0 {
            px = 5.0;
        }
        if py > 60.0 {
            py = 5.0;
        }
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iters as f64
}

/// Triangle path: `sample_point_tri_into` with BarycentricSortCdf.
/// Report only.
fn g5_latency_tri() -> f64 {
    let (field, tri_pos, tri_idx, mut scratch) = build_tri_workload();
    let encode = LocalCoordEncode::BarycentricSortCdf;

    // Warmup.
    let mut px = 0.5f32;
    let mut py = 0.5f32;
    for _ in 0..100 {
        sample_point_tri_into(&field, &tri_pos, tri_idx, px, py, encode, &mut scratch);
        px += 0.03;
        py += 0.02;
        if px + py > 3.5 {
            px = 0.3;
            py = 0.3;
        }
    }

    // Measure.
    let iters = 10_000usize;
    let mut px = 0.5f32;
    let mut py = 0.5f32;
    let start = Instant::now();
    for _ in 0..iters {
        sample_point_tri_into(
            black_box(&field),
            black_box(&tri_pos),
            black_box(tri_idx),
            black_box(px),
            black_box(py),
            black_box(encode),
            black_box(&mut scratch),
        );
        px += 0.03;
        py += 0.02;
        if px + py > 3.5 {
            px = 0.3;
            py = 0.3;
        }
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / iters as f64
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
    println!("╔═════════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 422 — Cochain Point Sampler GOAT Perf Gate (G4 alloc, G5 lat) ║");
    println!("╚═════════════════════════════════════════════════════════════════════╝");
    println!();

    let mut all_pass = true;

    // G4: zero-alloc (quad + tri)
    let (allocs_q, g4q) = g4_zero_alloc_quad();
    println!(
        "G4 zero-alloc (quad, 100 calls): allocs = {allocs_q}  (gate = 0)  → {}",
        verdict(g4q)
    );
    all_pass &= g4q;

    let (allocs_t, g4t) = g4_zero_alloc_tri();
    println!(
        "G4 zero-alloc (tri,  100 calls): allocs = {allocs_t}  (gate = 0)  → {}",
        verdict(g4t)
    );
    all_pass &= g4t;

    // G5: latency (primary gate = quad raw; full sincos + tri = report only)
    let (ns_raw, g5) = g5_latency_quad_raw();
    println!(
        "G5 latency  (quad raw, 64×64, dim=8): {ns_raw:.1} ns/call  (gate < 200)  → {}",
        verdict(g5)
    );
    all_pass &= g5;

    let ns_sincos = g5_latency_quad_full_sincos();
    println!(
        "  [report] quad full (Sincos n=4):     {ns_sincos:.1} ns/call"
    );

    let ns_tri = g5_latency_tri();
    println!(
        "  [report] tri full (BarycentricSort):  {ns_tri:.1} ns/call"
    );

    println!();
    if all_pass {
        println!("══ PERF GATES PASS — G4 + G5 both pass ══");
    } else {
        println!("══ PERF GATES FAIL — see above ══");
    }
    std::process::exit(if all_pass { 0 } else { 1 });
}
