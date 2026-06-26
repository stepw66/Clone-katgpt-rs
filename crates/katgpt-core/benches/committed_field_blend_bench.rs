//! CommittedFieldBlend — G4 zero-alloc GOAT gate bench (Plan 321 Phase 2 T2.4).
//!
//! Verifies that the hot path `apply_blended()` allocates ZERO times after a
//! single warmup call, via a global `CountingAllocator`. The commit path is
//! also audited (it collects field commitments into a stack-fixed array, so
//! it must also be zero-alloc).
//!
//! # Gates
//!
//! - **G4 (alloc-free hot path)** — `apply_blended()` in a 1000-iteration tight
//!   loop must allocate 0 times. Scratch buffers are pre-allocated outside the
//!   measured region and reused.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features committed_field_blend --bench committed_field_blend_bench --release -- --nocapture
//! ```

#![cfg(feature = "committed_field_blend")]

use katgpt_core::committed_field_blend::{ArchetypeFieldSource, TriArchetypeBlend};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// ─── CountingAllocator (G4) ─────────────────────────────────────────────────

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

// ─── Test fields (frozen, zero-alloc evolve) ────────────────────────────────

/// Linear field: f(z) = scale · z. Frozen at construction.
struct LinearField {
    scale: f32,
    commitment: [u8; 32],
}

impl LinearField {
    fn new(scale: f32, id: u8) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"LinearField");
        hasher.update(&[id]);
        hasher.update(&scale.to_le_bytes());
        Self {
            scale,
            commitment: *hasher.finalize().as_bytes(),
        }
    }
}

impl ArchetypeFieldSource<32> for LinearField {
    fn evolve<'a>(&self, z: &[f32], dz_scratch: &'a mut [f32]) -> &'a mut [f32] {
        for j in 0..32 {
            dz_scratch[j] = self.scale * z[j];
        }
        &mut dz_scratch[..32]
    }
    fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

// ─── G4: alloc-free hot path ────────────────────────────────────────────────

fn g4_apply_blended_zero_alloc() -> (bool, usize) {
    let f0 = LinearField::new(0.5, 0);
    let f1 = LinearField::new(-0.3, 1);
    let f2 = LinearField::new(0.8, 2);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

    // Three orthogonal-ish direction vectors (stack-fixed).
    let mut dirs = [[0.0f32; 32]; 3];
    for j in 0..32 {
        dirs[0][j] = if j < 11 { 1.0 } else { 0.0 };
        dirs[1][j] = if (11..22).contains(&j) { 1.0 } else { 0.0 };
        dirs[2][j] = if j >= 22 { 1.0 } else { 0.0 };
    }
    let summary: [f32; 32] = core::array::from_fn(|i| (i as f32) * 0.1);

    let mut blend = TriArchetypeBlend::uncommitted();
    blend.commit(&summary, &dirs, &fields, 1);

    // Pre-allocate scratch + output (stack arrays — no heap).
    let mut z = [0.5f32; 32];
    let mut scratch = [0.0f32; 32];
    let mut dz_out = [0.0f32; 32];

    // Warmup: one apply call (in case any lazy init hides inside — there
    // shouldn't be any, but the warmup makes the audit airtight).
    blend.apply_blended(&fields, &z, &mut scratch, &mut dz_out);

    // Measured region: 1000 apply_blended calls. Must allocate 0 times.
    let (_, apply_allocs) = alloc_delta(|| {
        let mut sink = 0u64;
        for i in 0..1000 {
            // Vary z slightly each iteration so the compiler can't hoist the
            // call out of the loop, but do so without allocating.
            for j in 0..32 {
                z[j] = ((i + j) as f32) * 0.001;
            }
            let dz = blend.apply_blended(&fields, &z, &mut scratch, &mut dz_out);
            sink = sink.wrapping_add(dz[0].to_bits() as u64);
        }
        std::hint::black_box(sink)
    });

    (apply_allocs == 0, apply_allocs)
}

fn g4_commit_zero_alloc_after_warmup() -> (bool, usize) {
    let f0 = LinearField::new(0.5, 0);
    let f1 = LinearField::new(-0.3, 1);
    let f2 = LinearField::new(0.8, 2);
    let fields: [&dyn ArchetypeFieldSource<32>; 3] = [&f0, &f1, &f2];

    let mut dirs = [[0.0f32; 32]; 3];
    for j in 0..32 {
        dirs[0][j] = if j < 11 { 1.0 } else { 0.0 };
        dirs[1][j] = if (11..22).contains(&j) { 1.0 } else { 0.0 };
        dirs[2][j] = if j >= 22 { 1.0 } else { 0.0 };
    }
    let summary: [f32; 32] = core::array::from_fn(|i| (i as f32) * 0.1);

    let mut blend = TriArchetypeBlend::uncommitted();

    // Warmup commit (any first-time init).
    blend.commit(&summary, &dirs, &fields, 1);

    // Measured: re-commit must allocate 0 times (field_commitments is a
    // stack-fixed [[u8;32]; N], blake3 Hasher is stack-allocated).
    let (_, commit_allocs) = alloc_delta(|| {
        let mut sink = 0u64;
        for v in 1..=100 {
            let h = blend.commit(&summary, &dirs, &fields, v);
            sink = sink.wrapping_add(h[0] as u64);
        }
        std::hint::black_box(sink)
    });

    (commit_allocs == 0, commit_allocs)
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  Plan 321 Phase 2 — CommittedFieldBlend G4 Zero-Alloc Gate      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let (apply_pass, apply_allocs) = g4_apply_blended_zero_alloc();
    println!("── G4a: apply_blended alloc-free (1000 iters) ──");
    println!("   allocs:     {apply_allocs}");
    println!("   Threshold:  0");
    println!(
        "   Result:     {}",
        if apply_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();

    let (commit_pass, commit_allocs) = g4_commit_zero_alloc_after_warmup();
    println!("── G4b: commit alloc-free (100 re-commits) ──");
    println!("   allocs:     {commit_allocs}");
    println!("   Threshold:  0");
    println!(
        "   Result:     {}",
        if commit_pass { "PASS ✓" } else { "FAIL ✗" }
    );
    println!();

    let all_pass = apply_pass && commit_pass;
    println!("═══ Phase 2 G4 exit ─══");
    if all_pass {
        println!("   G4a ✓ G4b ✓ → ZERO-ALLOC confirmed. apply_blended + commit are");
        println!("   heap-free after warmup. G4 PASSES.");
    } else {
        println!("   G4 FAILED — audit the allocation source before promotion.");
        std::process::exit(1);
    }
}
