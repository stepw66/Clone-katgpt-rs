# Plan 407 Phase 3 — Sheaf-ADMM Amplification GOAT Gate Results

**Date:** 2026-07-07
**Plan:** [katgpt-rs/.plans/407_sheaf_admm_coordination_primitive.md](../.plans/407_sheaf_admm_coordination_primitive.md)
**Bench:** `katgpt-dec/benches/bench_407_phase3_sheaf_admm.rs`

## Gate summary

| Gate | Task | Criterion | Target | Measured | Verdict |
|---|---|---|---|---|---|
| T3.2 | Sparse selector latency (K=1000) | compact < 50% of dense | ratio < 0.50 | **0.197** (5.1× speedup) | ✅ PASS |
| T3.1 | CG vs GD residual (K=1000, κ>100) | CG residual < GD residual at equal matvec count | CG < GD | CG=1025.73 < GD=1101.80 (**6.9% lower**) | ✅ PASS |
| T3.3 | Soft-constraint latency overhead | overhead < 20% | ratio < 0.20 | **+11%** | ✅ PASS |

**All three Phase 3 amplification gates PASS.**

## T3.2 — Sparse selector maps (the unblocking primitive)

### What shipped

- `SheafMaps::selector_per_edge(cx, d_v, dim_indices_per_edge: &[&[usize]])` — Issue 396 API. Per-edge heterogeneous dim selection. Compact u16 storage (`n_edges * 2 * d_e` entries), no dense matrix materialized.
- `SheafMaps::selector_per_edge_topk(cx, d_v, scores_per_edge: &[&[f32]], k: usize)` — Issue 397 API. Top-k dims from per-edge importance scores via partial sort.
- **Gather-scatter fast path** in `sheaf_laplacian_matvec`: when `maps.is_selector`, the matvec does `O(d_e)` per edge (gather selected dims, compute disagreement, scatter back) instead of `O(d_e·d_v)` for the dense general-maps path.

### Bench result (K=1000 path graph, d_v=8, d_e=5, non-identity dims [3,4,5,6,7])

| Path | Latency (ns) | Per-edge work |
|---|---|---|
| Dense general-maps (`SheafMaps::selector`) | 221,819 | `O(d_e·d_v)` = 40 muladds/edge |
| Compact gather-scatter (`SheafMaps::selector_per_edge`) | 43,798 | `O(d_e)` = 10 gather-scatter/edge |
| **Speedup** | **5.1×** | **4× fewer ops + no SIMD dot overhead** |

The dense path is much slower than expected because the general explicit-maps matvec does `d_e` SIMD dot products (length `d_v`) per edge for the disagreement, PLUS `d_e × d_v` scatter-adds for the `F^T·d` accumulation — most of which multiply against zero (selector maps have only `d_e` nonzeros per `d_e × d_v` matrix). The compact path skips all the zero-multiplies.

### Downstream unblock

This ships the open primitive that riir-ai Plan 394 T3.5 + Issues 396/397 were blocked on. The runtime can now consume `selector_per_edge` / `selector_per_edge_topk` via `ZoneSheafState::set_maps()` for per-edge-type heterogeneous consensus.

## T3.1 — CG z-update variant

### What shipped

- `sheaf_admm_step_cg_into(cx, maps, ..., max_cg_iters, tol, scratch)` — CG on the singular sheaf Laplacian `L_F z = 0`. Converges to the kernel projection (same target as the GD diffusion path) but in `O(√κ)` iterations vs GD's `O(κ)`.
- Extra scratch: `cg_r`, `cg_p`, `cg_ap` in `AdmmScratch`.
- Zero-alloc: the matvec borrow is split (`sheaf_laplacian_matvec` takes individual field slices) so `scratch.cg_p` can be passed as input without an aliasing violation.

### Bench result (K=1000 path graph, d_v=8, d_e=5, 20 matvecs)

| Path | L1 disagreement after 1 step |
|---|---|
| GD (T=20 diffusion steps, η=0.2) | 1101.80 |
| CG (20 iterations, tol=1e-12) | **1025.73** |

CG residual is **6.9% lower** at equal matvec count. The gain is modest because:
1. The path graph's condition number κ ≈ N²/π² ≈ 100 (for N=1000), so √κ ≈ 10 — CG's theoretical advantage is ~10× fewer iterations, but we're only running 20 iterations (not enough for the asymptotic advantage to dominate).
2. CG's per-iteration cost is ~3× GD's (3 axpy + 1 matvec + 2 dot vs 1 matvec + 1 axpy).

**Stays opt-in.** The GD path (`sheaf_admm_step_into`) remains the default entry point. CG wins when `√κ < matvec_count` AND the graph is ill-conditioned enough to justify the per-iteration overhead.

## T3.3 — Soft-constraint variant

### What shipped

- `sheaf_admm_step_soft_into(cx, maps, ..., gamma, diffusion_steps, scratch)` — soft-constraint z-update.
- When `gamma == 0.0`: delegates to the hard-constraint path (bit-identical to `sheaf_admm_step_into`).
- When `gamma > 0`: z-update minimizes `½ z^T L_F z + (γ/2)‖z − b‖²` via `z ← z − η(L_F z + γ(z − b))`. The `γ(z − b)` term resists full consensus, preserving individual variation.

### Bench result (K=1000, γ=0.5, T=5 diffusion steps)

| Path | Latency (ns) |
|---|---|
| Hard constraint (`sheaf_admm_step_into`) | 36,367 |
| Soft constraint γ=0.5 (`sheaf_admm_step_soft_into`) | 40,475 |
| **Overhead** | **+11%** (gate < 20%) |

The overhead is the extra `γ(z − b)` term in the diffusion loop — one extra FMA per element per diffusion step. Well under the 20% gate.

### Correctness tests

- `soft_constraint_gamma_zero_matches_hard`: γ=0 produces bit-identical results to the hard path (all `x`, `z`, `u` fields match exactly).
- `soft_constraint_gamma_positive_preserves_variation`: γ>0 retains MORE edge disagreement than the hard path after 30 ADMM steps (the `γ(z − b)` pull prevents full consensus).

## Validation

- `cargo test -p katgpt-dec --features sheaf_admm --lib`: **131 passed, 0 failed** (9 original + 11 new sheaf_admm tests + 111 other DEC tests).
- `cargo test -p katgpt-core --lib`: **1326 passed, 0 failed** (no regressions from the `AdmmScratch` field additions).
- `cargo bench --bench bench_407_sheaf_admm_goat`: G4 (1.767 µs < 5.0 µs) + G5 (0 allocs) **both PASS** — no Phase 2 regression.
- `cargo bench --bench bench_407_phase3_sheaf_admm`: all three Phase 3 gates **PASS**.
- `cargo check` (full workspace): **clean**.
