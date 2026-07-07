# Plan 410: Linking-Number Detector + Fold Correction Primitive

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/391_Low_Dimensional_Topology_Linking_Number.md](../.research/391_Low_Dimensional_Topology_Linking_Number.md)
**Source paper:** [arXiv:2606.31856](https://arxiv.org/abs/2606.31856) — Ren & Lim, *Low-dimensional topology of deep neural networks*, ICML 2026 (PMLR 306)
**Target:** `katgpt-rs/crates/katgpt-core/src/linking_fold.rs` (new module) + Cargo feature `linking_fold`
**Status:** Active — Phase 1 ✅, Phase 2 ✅, Phase 3 ✅, Phase 4 ✅ (G2 detector FAILS original budget — resolved via Option C feature split, see Issue 050), Phase 5 ✅.

**GOAT gate summary (verified 2026-07-07, bench + alloc test run):**
| Gate | Status | Evidence |
|---|---|---|
| G1 (correctness) | ✅ PASS | 16/16 unit tests + bench G1 smoke (Hopf = −1, unlinked = 0, fold unlinks) |
| G2 fold hot-path | ✅ PASS | 12.5 ns @ D=8 (Abs), 16.1 ns @ D=8 (Gelu), 16.8 ns @ D=64 (Abs), 16.9 ns @ D=64 (Gelu) — all under 50 ns / 500 ns budgets |
| G2 detector cold-path | ❌ FAIL original (50 ms @ n=2×1000); ⚠️ PASS recalibrated (500 ms @ n=2×200, audit cadence) | 407 ms measured @ n=2×200; minutes extrapolated @ n=2×1000. Resolved via Option C split (2026-07-07): detector stays opt-in (`linking_fold_detector`); fold promoted to default (`linking_fold_fold`). See [Issue 050](../.issues/050_linking_fold_detector_cold_path_perf.md) |
| G3 (no-regression) | ✅ PASS | `cargo check --features linking_fold` clean |
| G4 (alloc-free hot path) | ✅ PASS | `linking_fold_alloc_check`: 0 allocs / 1000 calls × 4 (Abs/Gelu × D=8/D=64) |
| G5 (determinism) | ✅ PASS | bit-identical detector (link ×3) + fold (×100) |

**Promotion decision (T4.4):** **Option C EXECUTED (2026-07-07).** The bundled `linking_fold` feature was split into two independently-gated sub-features:
- **`linking_fold_fold`** (the hot-path fold correction) — **DEFAULT-ON.** Passes every GOAT gate modellessly (G1 ✅ fold unlinks Hopf link, G2 ✅ 12–17 ns, G3 ✅, G4 ✅ 0 allocs, G5 ✅ bit-identical). This is the valuable per-tick primitive; it ships immediately.
- **`linking_fold_detector`** (the cold-path detector) — **opt-in.** Fails its original G2 budget (407 ms @ n=2×200 vs planned 50 ms @ n=2×1000). Stays opt-in until [Issue 050](../.issues/050_linking_fold_detector_cold_path_perf.md) resolves (accept recalibrated audit-cadence budget, optimize, or keep the split permanent).
- **`linking_fold`** — umbrella = `linking_fold_fold + linking_fold_detector` (backward-compat for consumers who wrote `linking_fold`). Opt-in.

Verified across 4 feature combinations: default (fold only) ✅, `--no-default-features --features linking_fold_fold` ✅, `--no-default-features --features linking_fold_detector` ✅, `--all-features` ✅. All 7 fold tests + 9 detector tests + 1 cross-feature unlink test + G4 alloc test pass.

---

## Goal

Distill Ren & Lim's modelless linking-number detector (paper Algorithm 1) and the coordinate-fold unlinking correction (paper §5) into a single generic MIT-licensed Rust module. Two composable primitives:

1. **`linking_detector`** — given two point clouds X, Y in R^d, PCA-project to R^3, build ε-filtered k-NN graphs, extract a fundamental cycle basis per graph via BFS spanning forest, compute the Gauss linking integral over O(β_X · β_Y) basis-cycle pairs. Returns `LinkingVerdict { linked: bool, link: i32, witness: Option<(usize, usize)> }`.
2. **`fold_projection_into`** — coordinate-wise `x ↦ c + |x − c|` (paper Eq. 1: `|x| = x + 2·ReLU(−x)`, realized as a single fold) applied in-place to a latent subspace. The deterministic modelless correction when the detector fires. Plus a GELU-surrogate variant (`fold_gelu_into`) that uses a smooth local-extremum fold instead of the hard `|·|`.

**Why modelless:** the detector is pure point-cloud geometry (PCA + k-NN + cycle basis + Gauss quadrature). The fold is a closed-form `|x − c|` map. No weights, no training, no backprop.

**Why GOAT, not Super-GOAT:** Q1 (no prior art) ✅, Q2 (new diagnostic class) ✅, Q4 (multi-pillar) ✅ — but Q3 (product selling point) is moderate (quality/retrieval gate, not headline). See Research 391 §3 for the full Q1–Q4 verdict.

**What this plan does NOT do:**
- Does NOT integrate with HLA / `latent_functor` / `NeuronShard` retrieval — those are private riir-ai / riir-neuron-db follow-ups, conditional on this open primitive shipping and proving useful.
- Does NOT fuse with DEC `harmonic_projector` (Research 391 §2.4 #1) — separate cross-paper fusion pass.
- Does NOT pursue a Lean 4 proof — the paper proves the unlinking theorem over ℝ; an f32 spec-match test on the synthetic Hopf link suffices for v1 (mirrors the riir-ai HLA-boundedness pattern).
- Does NOT redirect anything to riir-train. The fold is a §3.5 path-3 latent-space correction (deterministic, closed-form, no GD).

---

## GOAT Gate (per AGENTS.md Feature Flag Discipline)

| Gate | Criterion | Measurement |
|---|---|---|
| **G1 (correctness)** | `linking_detector` returns `link = ±1` on a synthetic thickened Hopf link (paper §G.1 parametrization), `link = 0` on two unlinked circles. `fold_projection_into` unlinks: after one coordinate-fold pass on each axis, the detector returns `link = 0` on the folded Hopf link. | `cargo test -p katgpt-core --features linking_fold --lib linking_fold` |
| **G2 (perf)** | Detector cold-path (audit cadence, not per-tick): ≤ 50 ms on n = 2×1000 point clouds at d = 8 (HLA scale). Fold hot-path: ≤ 50 ns/call at d = 8 (HLA tick budget), ≤ 500 ns at d = 64 (shard scale), 0 allocs. | `cargo bench -p katgpt-core --features linking_fold --bench bench_410_linking_fold_goat` |
| **G3 (no-regression)** | Default features still build clean; `--features linking_fold` builds clean; `--all-features` builds clean. Existing tests unaffected. | `cargo check`, `cargo check --features linking_fold`, `cargo check --all-features` |
| **G4 (alloc-free hot path)** | `fold_projection_into` and `fold_gelu_into`: 0 allocations per 1000 calls (CountingAllocator). The detector may allocate — it's cold-path. | Separate `linking_fold_alloc_check` test binary |
| **G5 (determinism)** | `linking_detector` returns the same `link` (integer) across runs on the same input. `fold_projection_into` is bit-identical given the same input + center (closed-form). | Bit-identical assertions in the G1 test |

**UQ check (Report the Floor rule, AGENTS.md):** This primitive does NOT claim a probability distribution, predictive interval, quantile, coverage guarantee, or calibrated uncertainty. The linking number is an integer-valued topological invariant; the verdict is boolean. The conformal-naive floor does not apply.

**Promotion rule:** If G1–G5 all PASS AND the primitive is pure-modelless → promote `linking_fold` to `default`. If G1 FAILS (detector misses the Hopf link or fold fails to unlink) → keep opt-in, file issue. If G3 FAILS → block promotion, fix before merge.

---

## Phase 1 — Unblocking Skeleton (CORE)

Goal: a compiling, feature-gated module with the public API surface frozen. No implementation yet — just types + signatures + doc.

**STATUS: ✅ COMPLETE (2026-07-07)**

### Tasks

- [x] **T1.1** Add feature flag `linking_fold = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[features]` section. No new deps (detector is brute-force k-NN; fold is closed-form).
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/linking_fold.rs` with module-level doc referencing Research 391 and arXiv:2606.31856.
- [x] **T1.3** Add `#[cfg(feature = "linking_fold")] pub mod linking_fold;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (alphabetical, after `karc`).
- [x] **T1.4** Define public types:
  - `LinkingDetectorConfig { k_neighbors, epsilon_quantile, min_cycle_len, n_subdivisions }` with sane defaults (`k=8`, `epsilon_quantile=0.7`, `min_cycle_len=4`, `n_subdivisions=4`).
  - `LinkingVerdict { linked: bool, link: i32, witness: Option<(usize, usize)> }`.
  - `FoldConfig { center: &[f32], variant: FoldVariant }` with `FoldVariant::{ Abs, Gelu }`.
- [x] **T1.5** `cargo check --features linking_fold` passes.

---

## Phase 2 — `linking_detector` implementation

Goal: the full Algorithm 1 pipeline. Cold-path; allocation allowed.

**STATUS: ✅ COMPLETE (2026-07-07)**

### Tasks

- [x] **T2.1** `pca_project_to_3d_into(points, output)` — closed-form 3×3 covariance + power-iteration + deflation for the top 3 eigenvectors. No `rustfft` dep (it's eigenvectors of a 3×3, not an FFT). Returns the projected 3D points.
- [x] **T2.2** `build_epsilon_knn_graph(points, k, epsilon)` — brute-force O(n²) k-NN with ε threshold (cold-path; matches paper §H.1). Returns adjacency lists (`Vec<Vec<usize>>`).
- [x] **T2.3** `fundamental_cycle_basis(n, adjacency)` — BFS spanning forest; for each non-tree edge, emit the cycle (tree-path + edge). Returns `Vec<Vec<usize>>` (paper §H.2 Definition H.1).
- [x] **T2.4** `gauss_linking_integral(cycle_x, cycle_y, points_x, points_y, n_sub)` — midpoint quadrature of `1/(4π) ∮_X ∮_Y (x−y)·(dx×dy)/|x−y|³` over two piecewise-linear cycles (paper §H.3). Returns the rounded integer link.
- [x] **T2.5** `detect_linking(x_d, y_d, config) -> LinkingVerdict` — top-level: PCA-project both clouds, build graphs, extract cycle bases, iterate basis pairs, return on first non-zero link.
- [x] **T2.6** Unit tests: synthetic Hopf link (paper §G.1 parametrization, thickened) → `link = ±1`; two unlinked circles → `link = 0`; degenerate inputs (empty, single point, all-coincident) → `NotLinked`.

---

## Phase 3 — `fold_projection_into` implementation

Goal: the modelless unlinking correction. Hot-path; zero-alloc.

**STATUS: ✅ COMPLETE (2026-07-07)**

### Tasks

- [x] **T3.1** `fold_projection_into(state, center)` — in-place coordinate-wise `state[i] = center[i] + |state[i] − center[i]|`. The paper's `|x| = x + 2·ReLU(−x)` identity realized as a single fold. Zero allocations, `#[inline]` inner loop.
- [x] **T3.2** `fold_gelu_into(state, center, alpha)` — smooth GELU-surrogate fold. Uses GELU's local minimum near −0.75 (paper §F.2 extension to activations with a strict local extremum). The `alpha` parameter rescales the data into the V-shape's effective region.
- [x] **T3.3** Unlinking test: take the synthetic Hopf link, apply one coordinate-fold pass per axis (3 passes total, paper Fig. 9), re-run the detector → `link = 0`.
- [x] **T3.4** Determinism test: `fold_projection_into` is bit-identical across calls given the same input + center.

---

## Phase 4 — GOAT gate (benchmarks + tests)

**STATUS: COMPLETE (2026-07-07) — all gates run. G2 detector FAILS original budget; promotion resolved via Option C feature split (T4.4 below).**

All five gates have been measured. The fold (hot-path) passes everything; the detector (cold-path) fails its original 50 ms @ n=2×1000 budget but passes a recalibrated 500 ms @ n=2×200 audit-cadence budget. **Resolution (Option C, 2026-07-07):** the feature was split — `linking_fold_fold` (the fold) promoted to default-on; `linking_fold_detector` (the detector) stays opt-in. The fold ships immediately; the detector's budget is tracked in Issue 050.

### Tasks

- [x] **T4.1** Bench `bench_410_linking_fold_goat` created and run. G1 smoke ✅, G2 fold hot-path ✅ (12–17 ns), G2 detector cold-path **407 ms @ n=2×200** (FAILS original 50 ms @ n=2×1000; the brute-force implementation is O(β²) and the original budget underestimated β). G5 determinism ✅. The bench reports both the original and a recalibrated audit-cadence budget transparently. See [Issue 050](../.issues/050_linking_fold_detector_cold_path_perf.md) for the budget-revision decision.
- [x] **T4.2** Alloc test `linking_fold_alloc_check` created and run. G4 ✅ — `fold_projection_into` and `fold_gelu_into` both 0 allocs / 1000 calls at D=8 and D=64 (CountingAllocator). Detector is cold-path and explicitly NOT gated.
- [x] **T4.3** All five gates run; verdicts recorded in the header table above and in the bench output.
- [x] **T4.4** **RESOLVED via Option C (feature split, 2026-07-07).** The bundled `linking_fold` feature was split into `linking_fold_fold` (DEFAULT-ON — the fold passes every gate modellessly) and `linking_fold_detector` (opt-in — the detector fails its original G2 budget). The umbrella `linking_fold = [fold, detector]` preserves backward compat. This unblocks the fold's promotion without silently relaxing the detector's budget. [Issue 050](../.issues/050_linking_fold_detector_cold_path_perf.md) tracks the detector's residual budget-acceptance/optimization decision (now non-blocking — the fold is already shipped).

---

## Phase 5 — Cross-references + commit

**STATUS: IN-PROGRESS (2026-07-07) — T5.1 done (table fix in Research 391), T5.2 being executed in this commit.**

### Tasks

- [x] **T5.1** Table fix in Research 391 §1.3 (literal `|x|` inside markdown table cells broke the column count; replaced with `abs(x)` and prose). Committed as `docs:` separate from the implementation.
- [x] **T5.2** Commit on `develop` with `feat:` prefix (this commit). Note: the implementation ships **opt-in only** — G2+G4 must pass before promotion to `default`. Two separate commits land in this session: `docs: research 391 table fix` then `feat(katgpt-core): linking_fold primitive (Plan 410, opt-in)`. A third follow-up commit (out of scope for this session) will land the bench + alloc test + promotion once G2/G4 pass.

---

## TL;DR

Shipped `linking_detector` (Algorithm 1: PCA-3D + ε-kNN + cycle basis + Gauss integral) + `fold_projection_into` / `fold_gelu_into` (coordinate-wise `|x−c|` unlinking correction, paper §5 / Eq. 1) behind feature flag `linking_fold` in `katgpt-rs/crates/katgpt-core/src/linking_fold.rs`. Pure modelless (closed-form PCA + brute-force k-NN + Gauss quadrature + abs-fold; no training, no GD).

**GOAT gate status (honest accounting, 2026-07-07, all gates run):**
- **Fold (hot-path) — all gates PASS:** G1 ✅, G2 ✅ (12–17 ns/call at D=8/D=64), G4 ✅ (0 allocs/1000 calls × 4), G5 ✅ (bit-identical).
- **Detector (cold-path) — G2 FAILS original budget:** 407 ms @ n=2×200 vs planned 50 ms @ n=2×1000. Passes a recalibrated 500 ms audit-cadence budget, but silent goalpost-moving is not honest.
- **G3 ✅** (`cargo check --features linking_fold` clean).

**Promotion: Option C executed (2026-07-07).** `linking_fold_fold` (the fold) is **DEFAULT-ON** — it passes every GOAT gate modellessly (G1 ✅ fold unlinks Hopf link, G2 ✅ 12–17 ns, G3 ✅, G4 ✅ 0 allocs, G5 ✅ bit-identical). `linking_fold_detector` (the detector) stays **opt-in** — it fails its original 50 ms @ n=2×1000 G2 budget (407 ms @ n=2×200 measured). The umbrella `linking_fold = [fold, detector]` preserves backward compat. See [Issue 050](../.issues/050_linking_fold_detector_cold_path_perf.md) for the detector's residual decision (now non-blocking). Verified: default ✅, fold-only ✅, detector-only ✅, all-features ✅; 7 fold + 9 detector + 1 cross-feature + G4 alloc tests all pass.
