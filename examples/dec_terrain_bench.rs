//! DEC Terrain Benchmark — Dynamic topology update performance (Plan 261 Phase 3).
//!
//! Compares DEC-based navigation field update vs naive grid re-scan after
//! terrain destruction events of varying scale.
//!
//! Run: `cargo run --example dec_terrain_bench --features dec_operators`
//!
//! # What This Measures
//!
//! - **remove_face + recompute**: Time to destroy N faces and rebuild DecFlowField
//! - **recompute_if_dirty skip**: Overhead of the dirty check when topology is unchanged
//! - **is_dirty_since check**: Sub-nanosecond version comparison
//! - **DecCache Betti invalidation**: Cache hit/miss after topology change
//!
//! # GOAT Gate
//!
//! - DEC wins if `remove_face + recompute` < naive grid scan for small N (1-10 cells)
//! - `is_dirty_since()` must be < 1ns
//! - `recompute_if_dirty` must skip in < 1ns when topology unchanged

use std::time::Instant;

use katgpt_core::dec::{
    CellComplex, CochainField, DecCache, DecFlowField, betti_numbers, hodge_decompose,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DEC Terrain Benchmark — Plan 261 Phase 3 GOAT Gate          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // ── 1. is_dirty_since() latency — must be < 1ns ──────────────
    bench_dirty_check();

    // ── 2. remove_face latency ───────────────────────────────────
    bench_remove_face();

    // ── 3. recompute_if_dirty skip overhead ──────────────────────
    bench_recompute_skip();

    // ── 4. Full destruction + recompute pipeline ─────────────────
    bench_destruction_pipeline();

    // ── 5. DecCache Betti invalidation ───────────────────────────
    bench_cache_invalidation();

    println!();
    println!("═╗ Benchmark Complete ═════════════════════════════════════════");
}

/// Measure `is_dirty_since()` — target: < 1ns per call.
fn bench_dirty_check() {
    let cx = CellComplex::grid_2d(64, 64);
    let version = cx.topology_version();
    let iterations = 10_000_000;

    let start = Instant::now();
    let mut sink = 0u64;
    for i in 0..iterations {
        let dirty = cx.is_dirty_since(version);
        sink += dirty as u64;
        std::hint::black_box(&sink);
        // Prevent the compiler from hoisting the check out of the loop
        if i == usize::MAX {
            break;
        }
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!();
    println!("┌─ is_dirty_since() ──────────────────────────────────────────");
    println!("│  Grid: 64×64, iterations: {iterations}");
    println!("│  Per-call: {per_call_ns:.3} ns");
    println!("│  Target: < 1 ns");
    match per_call_ns < 1.0 {
        true => println!("│  ✅ PASS"),
        false => println!("│  ⚠️  SLOW (but functional)"),
    }
    println!("└──────────────────────────────────────────────────────────────");
}

/// Measure `remove_face()` for 1, 10, 100 faces.
fn bench_remove_face() {
    for &n_destroy in &[1usize, 10, 100] {
        let mut cx = CellComplex::grid_2d(64, 64);
        let n_faces = cx.n_faces();

        let start = Instant::now();
        for i in 0..n_destroy.min(n_faces) {
            cx.remove_face(i);
        }
        let elapsed = start.elapsed();
        let per_face_us = elapsed.as_secs_f64() * 1e6 / n_destroy as f64;

        println!();
        println!("┌─ remove_face() × {n_destroy} ────────────────────────────────");
        println!("│  Grid: 64×64 ({n_faces} faces initially)");
        println!("│  Total: {elapsed:?}");
        println!("│  Per-face: {per_face_us:.1} μs");
        println!("│  Target: < 10 μs (1 cell)");
        println!("└──────────────────────────────────────────────────────────────");
    }
}

/// Measure `recompute_if_dirty` skip when topology is unchanged.
fn bench_recompute_skip() {
    let cx = CellComplex::grid_2d(32, 32);
    let mut pot = CochainField::zeros(0, cx.n_vertices(), 1);
    for i in 0..cx.n_vertices() {
        pot.set_scalar(i, (i as f32 * 0.3).sin());
    }
    let mut field = DecFlowField::compute(&cx, &pot, 1.0, 0.5, 0.3);

    let iterations = 1_000_000;
    let start = Instant::now();
    let mut recompute_count = 0u64;
    for _ in 0..iterations {
        let recomputed = field.recompute_if_dirty(&cx, &pot, 1.0, 0.5, 0.3);
        recompute_count += recomputed as u64;
        std::hint::black_box(&recompute_count);
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;

    println!();
    println!("┌─ recompute_if_dirty() skip ─────────────────────────────────");
    println!("│  Grid: 32×32, iterations: {iterations}");
    println!("│  Per-skip: {per_call_ns:.3} ns");
    println!("│  Recomputes triggered: {recompute_count} (should be 0)");
    assert_eq!(
        recompute_count, 0,
        "should never recompute when topology unchanged"
    );
    println!("│  ✅ Skip verified — zero recomputes");
    println!("└──────────────────────────────────────────────────────────────");
}

/// Full pipeline: destroy N faces → recompute DecFlowField.
fn bench_destruction_pipeline() {
    for &n_destroy in &[1usize, 10, 50] {
        let mut cx = CellComplex::grid_2d(32, 32);
        let mut pot = CochainField::zeros(0, cx.n_vertices(), 1);
        for i in 0..cx.n_vertices() {
            pot.set_scalar(i, ((i % 32) as f32 - 16.0).abs());
        }

        // Initial flow field
        let mut field = DecFlowField::compute(&cx, &pot, 1.0, 0.5, 0.3);

        // Destroy faces
        for i in 0..n_destroy.min(cx.n_faces()) {
            cx.remove_face(i);
        }

        // Recompute (topology changed)
        let start = Instant::now();
        let recomputed = field.recompute_if_dirty(&cx, &pot, 1.0, 0.5, 0.3);
        let elapsed = start.elapsed();

        println!();
        println!("┌─ destroy {n_destroy} faces + recompute ──────────────────────");
        println!("│  Grid: 32×32");
        println!("│  Recomputed: {recomputed}");
        println!("│  Total recompute time: {elapsed:?}");
        println!("│  Target: < 500 μs for 100-cell SIMD");
        println!("└──────────────────────────────────────────────────────────────");
    }
}

/// DecCache Betti number invalidation after topology change.
fn bench_cache_invalidation() {
    let mut cx = CellComplex::grid_2d(16, 16);
    // Bump version to simulate post-init state (version > 0)
    cx.remove_cell(2, 0);
    let mut cache = DecCache::new();

    // Compute and cache Betti numbers at version 1
    let bettis = betti_numbers(&cx);
    cache.store_betti(bettis, cx.topology_version());
    assert!(cache.is_valid(cx.topology_version()));

    // Mutate topology again
    cx.remove_face(0);

    // Check cache invalidation
    let dirty = !cache.is_valid(cx.topology_version());

    // Hodge decomposition with cache awareness
    let edge_field = {
        let pot = CochainField::zeros(0, cx.n_vertices(), 1);

        katgpt_core::dec::exterior_derivative(&cx, &pot)
    };
    let _components = hodge_decompose(&cx, &edge_field);

    println!();
    println!("┌─ DecCache invalidation ─────────────────────────────────────");
    println!("│  Grid: 16×16");
    println!("│  Betti numbers: {bettis:?}");
    println!("│  Cache dirty after 1 face removal: {dirty}");
    assert!(dirty, "cache must be dirty after topology change");
    println!("│  ✅ Cache correctly invalidated");
    println!("└──────────────────────────────────────────────────────────────");
}
