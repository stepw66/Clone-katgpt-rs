# Benchmark 314: Stokes Calculus Wrappers — GOAT Gate

**Date:** 2026-06-24
**Plan:** 314 (Stokes Calculus Wrappers — Fokker-Planck Validator + Boundary-Flux Mass + Line Integral)
**Features:** `--features dec_operators` (root alias: `stokes_calculus`)
**Commands:**
  - Tests: `cargo test -p katgpt-core --features dec_operators --lib dec::stokes_calculus`
  - Bench:  `cargo bench -p katgpt-core --features dec_operators --bench stokes_calculus_bench -- --warm-up-time 1 --measurement-time 2 --sample-size 10`
**Source papers:**
  - [arXiv:2202.11322](https://arxiv.org/abs/2202.11322) — *Efficient CDF Approximations for Normalizing Flows* (TMLR 2022)
  - [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/cbf8710b43df3f2c1553e649403426df-Abstract.html) — *Neural Manifold ODEs* (Lou et al.)

---

## Summary Verdict

| Gate | Target | Result | Status |
|------|--------|--------|--------|
| **G-B** (boundary-flux mass) | ≥3× faster, error < 5% | **5.36× faster**, error 3.78% | ✅ **PASS** |
| **G-C** (line integral) | ≥20% fewer reversals | line_integral discriminates paths (Δ=1.872) but **cannot encode turn penalties** | ⚠️ **STRUCTURAL FAIL** — see §G-C Honest Finding |
| **G-A** (Fokker-Planck) | ≥1.5× earlier / ≥2× cheaper | Deferred to riir-ai (needs live HLA) | ⏳ DEFERRED |

**Promotion decision (T3.5):** G-B passes; G-C fails on a structural limitation (not a perf miss). Per the plan's split rule, **`stokes_calculus` stays opt-in** — the boundary-flux mass is a genuine win, but the headline "line-integral smoothness" claim doesn't hold for rank-1 cochains. File `004_stokes_calculus_g_c_turn_penalty.md` for the rank-2 extension.

---

## Unit Tests (Phase 2, recap)

All 12 unit tests pass (`cargo test -p katgpt-core --features dec_operators --lib dec::stokes_calculus`):

| Primitive | Tests | Status |
|-----------|-------|--------|
| `belief_mass_divergence` | 4 (identity×2, scaling, anomaly) | ✅ |
| `boundary_flux_mass` | 4 (Stokes identity×2, exact-field, empty) | ✅ |
| `line_integral` | 4 (straight, reversal antisymmetry, closed-loop exact, short path) | ✅ |

Full DEC suite: **96/96 tests pass** (12 new + 84 pre-existing).

---

## Bench Results (MacBook Pro, release profile, criterion)

### G-A baseline — `belief_mass_divergence` (32×32 grid)

| Bench | Median | Per-edge | Notes |
|-------|--------|----------|-------|
| `belief_mass_divergence/32x32_constant_flow` | **5.00 µs** | 2.5 ns/edge | Full `codifferential` + L1 sum |
| `codifferential_baseline/32x32_codifferential_into` | **5.20 µs** | 2.6 ns/edge | `_into` variant + manual L1 sum |

The wrapper (`belief_mass_divergence`) is on par with the raw `codifferential_into` — the wrapper overhead (function call + L1 reduction) is negligible. **G-A formal gate deferred to riir-ai** (needs live HLA branching events).

### G-B — `boundary_flux_mass_only` vs naive volume (256×256 map, 64×64 zone)

| Bench | Median | Speedup vs naive | Notes |
|-------|--------|------------------|-------|
| `G-B_256x256_boundary_flux` | **115.53 µs** | — | `boundary_flux_mass_only` |
| `G-B_256x256_naive_volume` | **619.31 µs** | **5.36× slower** | `exterior_derivative_into` + region sum |
| `G-B_256x256_cached_d_field_region_sum` | **2.79 µs** | 0.024× (42× faster) | Pre-cached `d_field`, region-only sum |

**Error bound:** mass = 8192.0 (= 2 × 4096 region faces, matching continuum curl = 2), error_bound = 309.56, **ratio = 3.78% < 5% → PASS**.

#### Why boundary_flux wins (and the honest caveat)

The 5.36× speedup does **not** come from the theoretical O(boundary) vs O(volume) advantage — the current `boundary_flux_mass_only` implementation scans ALL `B₂` entries (O(|B₂|) = 260k entries), same asymptotic class as the naive `exterior_derivative_into`. The win comes from **memory access patterns**:

- `exterior_derivative_into` writes ALL 65k face values (full output materialization)
- `boundary_flux_mass_only` only reads `field.scalar()` for entries where `in_region[kp1_cell]` is true (sparse read, no output write)

The theoretical O(boundary) advantage requires a **coboundary index** (for each (k+1)-cell, its boundary (k)-cells) to avoid scanning all B entries. CellComplex doesn't expose one. This is filed as a future optimization (see Issues).

**The cached_d_field result (2.79 µs)** shows that if a caller pre-computes `d₁(field)` once per tick (O(|B₂|)), each subsequent region query is O(|region|) = 4096 ops — 42× faster than boundary_flux. **The boundary_flux primitive's niche is single-query scenarios where you don't want to materialize the full d_field**, not multi-query batched zone computation.

### G-C — `line_integral` as path cost (32×32 grid)

| Bench | Median | Per-edge | Line integral | Turns |
|-------|--------|----------|---------------|-------|
| `G-C_smooth_path_30_edges` | **9.30 µs** | 310 ns/edge | 2.231 | 1 |
| `G-C_zigzag_path_30_edges` | **10.76 µs** | 359 ns/edge | 0.359 | 29 |

**Discrimination:** `line_integral` correctly distinguishes the two paths (Δ = 1.872) on a non-exact (rotational) edge field. On an exact (gradient) field, both paths give the same line_integral (path-independence, fundamental theorem of calculus) — verified during bench development.

#### G-C Honest Finding: line_integral CANNOT reduce reversals

The plan's G-C target ("≥20% fewer direction reversals") is **structurally unreachable** for `line_integral` on a rank-1 (edge) cochain. Reason:

> A rank-1 edge cochain encodes per-EDGE cost only. **Turn penalties are a path-level property** — they depend on the angle between consecutive edges, which requires either a rank-2 (face) cochain (integrating curl over the enclosed area) or a path-level cost function. `line_integral` sums per-edge scalars and cannot "see" turns.

Evidence from the bench: the smooth path (1 turn) and the zigzag path (29 turns) have **different** line_integral values (2.231 vs 0.359), but this difference comes from which EDGES they traverse (different spatial positions in a non-exact field), NOT from the turn count. On a uniform field, both give the same value regardless of turns.

**The line_integral primitive is still useful** as a pure path-cost function (path energy, work, terrain-friction accumulation) — just not as a smoothness/reversal regularizer. The "≥20% fewer reversals" framing in the plan was based on a misclassification of what rank-1 cochains can express.

**Path forward (filed as issue):** A rank-2 face cochain wrapper (`area_integral` or `circulation_integral`) could encode turn penalties by integrating curl over the area enclosed by a closed path. This is the natural Stokes-theorem companion to `line_integral`.

---

## Perf Observation: `line_integral` latency

`line_integral` takes ~310 ns/edge on a 32×32 grid. For each path step, it scans ALL `B₁` entries via `chunks_exact(2)` to find the connecting edge — **O(path_length × |B₁|)**. For a 30-edge path on a 32×32 grid (|B₁| ≈ 4k entries × 2 = 8k), that's 30 × 4000 = 120k pair-comparisons.

A precomputed vertex-pair → edge index (HashMap or CSR) would make this **O(path_length)** = ~30 lookups. This is the same class of optimization as Plan 312's CSR adjacency. Filed as a future issue.

---

## Files Touched (Phase 3)

| File | Change |
|------|--------|
| `katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs` | Extracted `boundary_flux_mass_only` (mass without hodge_decompose) from `boundary_flux_mass` |
| `katgpt-rs/crates/katgpt-core/src/dec/mod.rs` | Re-export `boundary_flux_mass_only` |
| `katgpt-rs/crates/katgpt-core/benches/stokes_calculus_bench.rs` | NEW — criterion bench (5 benchmarks across 4 groups) |
| `katgpt-rs/crates/katgpt-core/Cargo.toml` | Register `stokes_calculus_bench` under `[[bench]]` with `required-features = ["dec_operators"]` |
| `katgpt-rs/.benchmarks/314_stokes_calculus_goat.md` | NEW — this file |

---

## Promotion Decision (T3.5)

**`stokes_calculus` stays OPT-IN** (not promoted to default-on).

Rationale:
1. **G-B PASSES** (5.36× speedup, 3.78% error) — `boundary_flux_mass` is a genuine modelless win for single-query zone-mass computation on 2D game maps.
2. **G-C FAILS on a structural limitation** — `line_integral` cannot encode turn penalties on rank-1 cochains. This is not a perf miss or a tuning issue; it's a mathematical fact about what edge cochains can express. The primitive is still correct and useful as a path-cost function, but the headline "smoothness" claim doesn't hold.
3. **Per the plan's split rule**: if only one gate passes, keep the feature opt-in. The winning primitive (`boundary_flux_mass`) is available to callers who want it; the losing primitive (`line_integral` for smoothness) is documented honestly.
4. **G-A is deferred to riir-ai** — its result feeds back into the decision when available.

**Future promotion path:**
- If G-A passes in riir-ai (Fokker-Planck validator catches ICT branching earlier/cheaper) → re-evaluate promotion of the full feature.
- If a rank-2 `area_integral` / `circulation_integral` wrapper is added (issue `004`) → the smoothness claim becomes achievable.
- If a coboundary index is added to CellComplex → `boundary_flux_mass` achieves true O(boundary) and the G-B win widens.

---

## Honest Risk Notes (recap from plan, updated with findings)

- ✅ **G-B passes** but NOT for the theoretical reason (O(boundary) vs O(volume)). The current implementation is O(|B₂|) for both approaches; the win is from memory access patterns (no output materialization). True O(boundary) requires a coboundary index.
- ⚠️ **G-C fails structurally** — line_integral of a rank-1 cochain cannot encode turn penalties. This is a mathematical limitation, not a fixable bug. The plan's risk note ("G-C may fail if manifold_geodesic paths are already near-optimal") understated the issue — the failure is more fundamental than path optimality.
- ⏳ **G-A is the highest-value gate** but runs in riir-ai. If it passes, the Fokker-Planck validator becomes the headline application regardless of G-B/G-C outcomes.
- The three primitives are all **correct** (12/12 unit tests pass, Stokes identities hold by construction). The GOAT gate is about whether they provide a modelless WIN, not whether they're correct.

---

## References

- Plan 314 — Stokes Calculus Wrappers
- Research 296 — Stokes Calculus DEC Vocabulary Crosswalk
- Plan 251 — DEC operators (the underlying machinery these wrappers call)
- Plan 312 — Viable Manifold Graph (CSR adjacency precedent for the coboundary-index optimization)

---

## Update: Plan 318 — Coboundary Index (2026-06-24)

**Origin:** Issue 006 (`boundary_flux_mass` coboundary index).
**Goal:** Widen the G-B win from the current 5.36× memory-access-pattern win
  toward true O(boundary) via a CSR coboundary index on `CellComplex`.

### New primitive

`boundary_flux_mass_indexed(cx, region_cells, field) -> f32` — uses the
pre-built coboundary index (`CellComplex::build_coboundary_index(k)`) to do
`O(|region| × boundary_per_cell)` direct CSR lookups instead of the
`O(|B_{k+1}|)` full-matrix scan.

### G-B indexed results (same 256×256 grid, 64×64 zone)

| Bench | Median | Speedup vs full-scan | Notes |
|-------|--------|----------------------|-------|
| `G-B_256x256_full_scan_baseline` | **132.93 µs** | 1.0× | `boundary_flux_mass_only` (current winner) |
| `G-B_256x256_indexed_cold` | **1.3435 ms** | 0.099× (10× SLOWER) | clone + `build_coboundary_index` + 1 query per iter |
| `G-B_256x256_indexed_warm` | **14.718 µs** | **9.03× FASTER** ✅ | pre-built index, query only |

**GOAT gate: PASS (warm = 9.03× ≥ 3× target).**

The cold-cache path is 10× slower (clone + build cost dominates a single
query). This is the honest "you must amortize the build" signal — the indexed
path is the right choice ONLY when the caller does many region queries on the
same topology. For single-query scenarios, `boundary_flux_mass_only` remains
the winner.

### Why warm is faster than the estimate

Plan 318's pre-implementation estimate was "~7 µs". Actual warm = 14.72 µs.
The discrepancy is because:
- The estimate assumed the dominant cost was the CSR scan (`4096 × 4 = 16k` ops).
- In practice, the indexed path also avoids the `Vec<bool>` region-membership
  allocation (`260k bools` = 260KB) that `boundary_flux_mass_only` does every
  call. The allocation + memset alone is ~50-100 µs.
- So the warm path pays neither the allocation nor the full-matrix scan —
  just the CSR lookups.

The 9.03× win (not the theoretical 16×) reflects the remaining work: iterating
4096 region cells × 4 edges each = 16k scalar reads + multiplies.

### Promotion decision (Plan 318)

**`stokes_calculus` stays opt-in.** G-B-indexed widens the G-B margin but does
NOT change the feature's promotion status, because G-A and G-C still fail.
The coboundary index is a modelless architectural win available to callers who
enable the feature.

### `merkle_root` lesson audit

All 5 topology-mutation paths invalidate the coboundary cache via
`invalidate_coboundary_cache()`:
- `remove_face` (rank 2)
- `remove_cell` rank 0 (vertex)
- `remove_cell` rank 1 (edge)
- `remove_cell` rank 2 → delegates to `remove_face` (covered)
- `remove_cell` rank 3 (volume)

Test `test_coboundary_index_remove_cell_invalidates_all_ranks` verifies all 4
explicit ranks (0/1/2/3) invalidate all 3 coboundary indices. No `add_incidence`
method exists (confirmed by grep), so there are no other mutation paths.

### Test count update

| Suite | Before Plan 318 | After Plan 318 |
|-------|-----------------|----------------|
| `dec::` (with `--features dec_operators`) | 99 | **111** (+12) |
| Full `katgpt-core --lib` (default features) | 509 | **509** (unchanged — dec is opt-in) |

New tests: 7 `CoboundaryIndex` tests (types.rs) + 5 `boundary_flux_mass_indexed`
tests (stokes_calculus.rs).

### Cross-refs

- Plan 318 — Coboundary Index for `boundary_flux_mass`
- Issue 006 — originally tracked in Issue 006 (closed + removed; this benchmark is the canonical record).
