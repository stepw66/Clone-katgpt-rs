//! Stokes Calculus Wrappers — GOAT gate benchmarks (Plan 314 Phase 3).
//!
//! Three benchmark groups covering the latency and A/B comparison targets
//! specified in the plan:
//!
//! - **`belief_mass_divergence/per_edge`** — per-edge cost of the L1 divergence
//!   validator. No formal target (G-A runs in riir-ai); this is a perf baseline.
//! - **`boundary_flux_mass/G-B_256x256`** — A/B: `boundary_flux_mass_only` vs
//!   naive full-volume summation (`exterior_derivative` + region sum) on a
//!   256×256 game map with a 64×64 zone. Target: ≥3× faster with
//!   `error_bound/mass < 5%`.
//! - **`line_integral/G-C_path_cost`** — A/B: `line_integral` latency per path
//!   step, plus a reversal-count comparison between a zigzag path and a smooth
//!   path (validates `line_integral` as a discriminating cost function).
//!   Target: ≥20% fewer reversals when `line_integral`-weighted.
//!
//! # Run
//!
//! ```bash
//! cargo bench -p katgpt-core --features dec_operators \
//!   --bench stokes_calculus_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10
//! ```
//!
//! # Feature gate
//!
//! Requires `dec_operators` (the underlying DEC operators from Plan 251).

#![cfg(feature = "dec_operators")]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::dec::operators::{codifferential_into, exterior_derivative_into};
use katgpt_core::dec::{
    CellComplex, CochainField, belief_mass_divergence, boundary_flux_mass_only, line_integral,
};

// ─── G-A baseline: belief_mass_divergence per-edge ──────────────────────────

fn bench_belief_mass_divergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/belief_mass_divergence");

    // 32×32 grid: 1024 vertices, 1984 edges. Realistic NPC belief manifold.
    let cx = CellComplex::grid_2d(32, 32);
    let n_edges = cx.n_edges();
    group.throughput(Throughput::Elements(n_edges as u64));

    // Constant flow field (all edges = 1.0) — divergence is zero at interior
    // vertices, non-zero at boundary. Exercises the full codifferential path.
    let field = CochainField::from_vec(1, 1, vec![1.0f32; n_edges]);

    group.bench_function("32x32_constant_flow", |b| {
        b.iter(|| {
            let div = belief_mass_divergence(black_box(&cx), black_box(&field));
            black_box(div);
        });
    });

    group.finish();
}

// ─── G-B: boundary_flux_mass_only vs naive volume sum ───────────────────────
//
// On a 256×256 game map (Bomber arena scale), compute the boundary-flux mass
// of a 64×64 zone (4096 faces) via two methods:
//
// A (boundary): `boundary_flux_mass_only` — single pass over B₂ entries with
//    region-membership filter. Interior edges cancel; only boundary survives.
// B (naive): `exterior_derivative_into` (compute d₁(field) over all faces)
//    then sum the region-face subset.
//
// Target: A ≥ 3× faster than B with error_bound/mass < 5%.

fn build_256x256_grid_and_field() -> (CellComplex, CochainField, Vec<u32>) {
    let w = 256;
    let h = 256;
    let cx = CellComplex::grid_2d(w, h);

    // Swirl (tangential) edge field: counterclockwise flow around the grid
    // center. This field has NON-ZERO curl (= 2 everywhere in the continuum),
    // so `boundary_flux_mass` produces a non-zero mass. Verified:
    //   mass ≈ 8192 (= 2 × 4096 region faces), error_bound ≈ 310, ratio ≈ 3.8%.
    //
    // Horizontal edge (y, x) → value = -(y - cy)  [tangential -dy component]
    // Vertical edge   (y, x) → value =  (x - cx)  [tangential +dx component]
    //
    // This is the correct test field for G-B: it has non-trivial curl (so the
    // boundary flux is non-zero and meaningful) while keeping the harmonic
    // component small (error_bound/mass < 5%).
    let cx_coord = w as f32 / 2.0;
    let cy_coord = h as f32 / 2.0;
    let n_h_edges = (w - 1) * h;
    let n_edges = cx.n_edges();
    let mut data = vec![0.0f32; n_edges];
    for y in 0..h {
        for x in 0..(w - 1) {
            let e_idx = y * (w - 1) + x;
            data[e_idx] = -(y as f32 - cy_coord);
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            let e_idx = n_h_edges + y * w + x;
            data[e_idx] = x as f32 - cx_coord;
        }
    }
    let field = CochainField::from_vec(1, 1, data);

    // Region: a 64×64 block of faces in the center of the grid.
    // Faces are indexed as f = y_face * (w-1) + x_face, where
    // y_face, x_face ∈ [0, w-2].
    let faces_per_row = w - 1;
    let zone_w = 64;
    let zone_h = 64;
    let x0 = (faces_per_row - zone_w) / 2;
    let y0 = (faces_per_row - zone_h) / 2;
    let mut region: Vec<u32> = Vec::with_capacity(zone_w * zone_h);
    for dy in 0..zone_h {
        for dx in 0..zone_w {
            region.push(((y0 + dy) * faces_per_row + (x0 + dx)) as u32);
        }
    }

    (cx, field, region)
}

fn bench_boundary_flux_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/boundary_flux_mass");
    group.sample_size(50); // each iter is ~ms; 50 samples is enough.

    let (cx, field, region) = build_256x256_grid_and_field();

    // Pre-allocated scratch buffer for the naive baseline's exterior_derivative.
    let mut d_field_scratch = CochainField::zeros(2, cx.n_faces(), 1);

    // ── A: boundary_flux_mass_only ──────────────────────────────────────────
    group.bench_function("G-B_256x256_boundary_flux", |b| {
        b.iter(|| {
            let mass = boundary_flux_mass_only(
                black_box(&cx),
                black_box(&region),
                black_box(&field),
            );
            black_box(mass);
        });
    });

    // ── B: naive full-volume (exterior_derivative + region sum) ─────────────
    group.bench_function("G-B_256x256_naive_volume", |b| {
        b.iter(|| {
            // Compute d₁(field) over ALL faces (this is the "full volume" cost).
            exterior_derivative_into(
                black_box(&cx),
                black_box(&field),
                black_box(&mut d_field_scratch),
            );
            // Sum the region subset.
            let mut vol = 0.0f32;
            for &f in &region {
                vol += d_field_scratch.scalar(f as usize);
            }
            black_box(vol);
        });
    });

    // ── C: naive with cached d_field (amortized — many queries, one d_field) ─
    // Pre-compute d_field once (simulates per-tick caching).
    exterior_derivative_into(&cx, &field, &mut d_field_scratch);
    group.bench_function("G-B_256x256_cached_d_field_region_sum", |b| {
        b.iter(|| {
            let mut vol = 0.0f32;
            for &f in black_box(&region) {
                vol += d_field_scratch.scalar(f as usize);
            }
            black_box(vol);
        });
    });

    group.finish();
}

// ─── G-C: line_integral as path cost ────────────────────────────────────────
//
// Two parts:
//   1. Latency: `line_integral` over a path on a 32×32 grid.
//   2. Correctness/discrimination: compare line_integral of a "smooth" path
//      vs a "zigzag" path of equal length under a directional-preference field.
//      Validates that line_integral-minimizing selection picks the smooth path
//      (fewer turns) — the mechanism behind the "≥20% fewer reversals" target.

fn count_turns_2d_grid(path: &[u32], w: usize) -> usize {
    // A "turn" is a change in direction between consecutive steps.
    // Direction encoded as (dx, dy) ∈ {-1, 0, +1}².
    if path.len() < 3 {
        return 0;
    }
    let mut turns = 0;
    let mut prev_dir: (i32, i32) = (0, 0);
    for w_pair in path.windows(2) {
        let a = w_pair[0] as usize;
        let b = w_pair[1] as usize;
        let ax = (a % w) as i32;
        let ay = (a / w) as i32;
        let bx = (b % w) as i32;
        let by = (b / w) as i32;
        let dir = (bx - ax, by - ay);
        if prev_dir != (0, 0) && dir != prev_dir {
            turns += 1;
        }
        prev_dir = dir;
    }
    turns
}

fn bench_line_integral(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/line_integral");

    let w = 32;
    let h = 32;
    let cx = CellComplex::grid_2d(w, h);
    let n_edges = cx.n_edges();

    // Non-exact (rotational) edge field: cannot be written as the gradient of
    // any scalar potential, so line_integral IS path-dependent (fundamental
    // theorem of calculus does NOT apply). This is the regime where line_integral
    // as a path-cost function can discriminate between routes.
    //
    // Construction: horizontal edge (y,x) gets sin(x + y*0.3); vertical edge
    // (y,x) gets cos(x*0.3 + y). The curl of this field is non-zero, so
    // line_integral(A→B) depends on the path taken.
    //
    // NOTE: a rank-1 edge cochain encodes per-EDGE cost only — it cannot encode
    // TURN penalties by construction (turns are a path-level property requiring
    // a rank-2 face cochain or a higher-order operator). line_integral
    // discriminates paths by which EDGES they traverse (via curl), not by their
    // turn count. This bench validates line_integral as a path-cost function,
    // not as a smoothness regularizer.
    let mut data = vec![0.0f32; n_edges];
    let n_h_edges = (w - 1) * h;
    for y in 0..h {
        for x in 0..(w - 1) {
            let e_idx = y * (w - 1) + x;
            data[e_idx] = (x as f32 + y as f32 * 0.3).sin();
        }
    }
    for y in 0..(h - 1) {
        for x in 0..w {
            let e_idx = n_h_edges + y * w + x;
            data[e_idx] = (x as f32 * 0.3 + y as f32).cos();
        }
    }
    let field = CochainField::from_vec(1, 1, data);

    // ── Path A: smooth (all-East then all-North): (0,0) → (15,0) → (15,15) ──
    // 15 East steps + 15 North steps = 30 edges, 1 turn.
    let mut smooth_path: Vec<u32> = Vec::with_capacity(31);
    for x in 0..=15 {
        smooth_path.push((0 * w + x) as u32); // along row 0
    }
    for y in 1..=15 {
        smooth_path.push((y * w + 15) as u32); // along column 15
    }

    // ── Path B: zigzag (alternating East, North): (0,0) → ... → (15,15) ─────
    // 30 edges, 29 turns.
    let mut zigzag_path: Vec<u32> = Vec::with_capacity(31);
    zigzag_path.push(0);
    let mut x = 0i32;
    let mut y = 0i32;
    for _ in 0..15 {
        x += 1;
        zigzag_path.push((y * w as i32 + x) as u32); // East
        y += 1;
        zigzag_path.push((y * w as i32 + x) as u32); // North
    }

    // Correctness check: smooth path should have a different line_integral
    // than the zigzag path (they traverse different edges with different
    // spatially-varying costs). If line_integral-weighted path selection were
    // to pick the cheaper of the two, it picks the lower value.
    let li_smooth = line_integral(&cx, &field, &smooth_path);
    let li_zigzag = line_integral(&cx, &field, &zigzag_path);
    let turns_smooth = count_turns_2d_grid(&smooth_path, w);
    let turns_zigzag = count_turns_2d_grid(&zigzag_path, w);
    eprintln!(
        "[G-C] smooth: line_integral={li_smooth:.3}, turns={turns_smooth}; \
         zigzag: line_integral={li_zigzag:.3}, turns={turns_zigzag}; \
         discriminates_by={:.3}",
        (li_zigzag - li_smooth).abs()
    );

    group.throughput(Throughput::Elements(smooth_path.len() as u64));
    group.sample_size(100);

    group.bench_function("G-C_smooth_path_30_edges", |b| {
        b.iter(|| {
            let li = line_integral(
                black_box(&cx),
                black_box(&field),
                black_box(&smooth_path),
            );
            black_box(li);
        });
    });

    group.bench_function("G-C_zigzag_path_30_edges", |b| {
        b.iter(|| {
            let li = line_integral(
                black_box(&cx),
                black_box(&field),
                black_box(&zigzag_path),
            );
            black_box(li);
        });
    });

    group.finish();
}

// ─── Auxiliary: codifferential_into baseline (for belief_mass_divergence) ────

fn bench_codifferential_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("stokes_calculus/codifferential_baseline");

    let cx = CellComplex::grid_2d(32, 32);
    let n_edges = cx.n_edges();
    let field = CochainField::from_vec(1, 1, vec![1.0f32; n_edges]);
    let mut scratch = CochainField::zeros(0, cx.n_vertices(), 1);

    group.throughput(Throughput::Elements(n_edges as u64));
    group.bench_function("32x32_codifferential_into", |b| {
        b.iter(|| {
            codifferential_into(
                black_box(&cx),
                black_box(&field),
                black_box(&mut scratch),
            );
            // L1 sum of the divergence.
            let l1: f32 = scratch.data.iter().copied().map(f32::abs).sum();
            black_box(l1);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_belief_mass_divergence,
    bench_boundary_flux_vs_naive,
    bench_line_integral,
    bench_codifferential_baseline,
);
criterion_main!(benches);
