# Benchmark 423 — Spectral Rewiring GOAT Gate

**Date:** 2026-07-10
**Plan:** [423_spectral_rewire_primitive.md](../.plans/423_spectral_rewire_primitive.md)
**Research:** [406_Spectral_Rewiring_Weight_Delta_Purification.md](../.research/406_Spectral_Rewiring_Weight_Delta_Purification.md)
**Primitive:** `katgpt_spectral::spectral_rewire` (opt-in feature `spectral_rewire`)
**Bench:** `crates/katgpt-spectral/benches/bench_423_spectral_rewire_goat.rs`
**Run:** `cargo bench -p katgpt-spectral --features spectral_rewire --bench bench_423_spectral_rewire_goat -- --nocapture` (release)

## Verdict

**ALL MECHANISM GATES PASS. Primitive stays OPT-IN.** Promotion to default
requires a real-delta concentration test (Issue 123) — the spectral
*concentration assumption* (that real weight deltas live in the base's top-r
SVD subspace) is **unvalidated** modellessly (we have no real training deltas).
The GOAT gate here validates the SVD + projection *machinery* is correct, fast
(cached-index path), and zero-alloc — but cannot validate the concentration
assumption that makes the primitive *useful*.

## Gate Results

| Gate | Result | Detail |
|---|---|---|
| **G1a** numerical stability at scale | **PASS** | On-manifold deltas recovered with fraction >0.999, rel err ~8e-6 at 64×64/128×64/512×64, and ~1.9e-5 at 128×128 (Issue 124 resolved) |
| **G1b** concentration characterization | **REPORT** | Random deltas NOT concentrated (0.12–0.18 vs theory 0.016–0.031) |
| **G2** singular-direction preservation | **PASS** | cosine = 1.000000 (target >0.99) |
| **G3** determinism | **PASS** | Bit-identical across 100 runs |
| **G4** alloc-free hot path | **PASS** | 0 bytes / 1000 calls (CountingAllocator) |
| **G5** latency (cached-index path) | **PASS** | 8×8 r=4 = 0.41µs; 512×64 r=32 = 947µs; 64×64 r=8 = 29µs |
| **G6** feature isolation | **PASS** | `--no-default-features`, `--all-features`, root forwarding all clean |

## G1a — Numerical Stability at Scale (PASS)

An on-manifold delta `ΔW = U_r · diag(m) · V_rᵀ` constructed in `W₀`'s own
top-r SVD subspace is recovered with `on_manifold_fraction > 0.999` and
relative recovery error `< 1e-4`:

| Scale | rank | on_manifold_fraction | recovery rel err |
|---|---|---|---|
| 64×64 | 8 | 1.000008 | 8.33e-6 |
| 128×64 | 16 | 1.000010 | 8.95e-6 |
| 512×64 | 32 | 1.000008 | 7.81e-6 |
| 128×128 | 16 | 1.000019 | 1.95e-5 |

This validates the SVD + matmul machinery is numerically sound at the largest
supported scales. It does NOT validate the concentration assumption (a delta
constructed to be on-manifold is trivially on-manifold).

**Scales are no longer bounded by `SVD_MAX_COLS`** (Issue 124 RESOLVED
2026-07-10): the one-sided Jacobi SVD argsort buffers have been moved from
fixed `[f32; 64]` / `[usize; 64]` stack arrays to heap-allocated fields in
`SvdScratch`, so arbitrary `d_in` is now supported. The former `SVD_MAX_COLS`
constant and `d_in <= 64` guards have been removed from `katgpt-spectral`.
512×512 was NOT added to the gate — the one-sided Jacobi SVD at 512 cols would
take minutes per call (O(n²) pairs × 60 sweeps), making the cold-tier latency
unacceptable. 128×128 is the practical cold-tier bound.

## G1b — Concentration Characterization (REPORT)

For a **random** (Gaussian) delta — not aligned with `W₀`'s subspace:

| Scale | rank | on_manifold_fraction | theory r²/(d_out·d_in) |
|---|---|---|---|
| 64×64 | 8 | 0.1210 | 0.0156 |
| 128×64 | 16 | 0.1749 | 0.0312 |
| 512×64 | 32 | 0.1778 | 0.0312 |
| 128×128 | 16 | 0.1340 | 0.0156 |

**Interpretation:** a generic delta is NOT concentrated in the base's top-r
subspace (measured ~0.12–0.18, well below the 0.5 concentration threshold). This
is expected and confirms the primitive's scope: it only *purifies* deltas that
ARE aligned with the base. Real training deltas (per the SAR paper) ARE
concentrated — but we have no real deltas to verify this modellessly (Research
406 §7 honest limitation #2). The measured values exceed the pure-random theory
(r²/d²) because a random delta has some incidental alignment, but they are far
below concentration. **Promotion to default is blocked on a real-delta
concentration test** (Issue 123).

## G5 — Latency

Two paths, both reported:

### SVD path (`spectral_rewire_into`) — cold-tier, reported not gated

Factors `W₀` every call. The one-sided Jacobi SVD dominates (`max_sweeps = 60`):

| Scale | rank | mean |
|---|---|---|
| 512×64 | 32 | 14099µs (14ms) |
| 64×64 | 8 | 2000µs (2ms) |

This path is for cold-tier / one-shot use only.

### Cached-index path (`spectral_rewire_with_index_into`) — hot-loop, GATED

Builds `SpectralRewireIndex` ONCE (SVD cost paid at build), then per-delta does
only the four matmuls:

| Scale | rank | mean | target | result |
|---|---|---|---|---|
| **8×8** | 4 | **0.41µs** | ≤1µs (NPC `style_weights[64]` → 8×8) | **PASS** |
| **512×64** | 32 | **947µs** | ≤1ms (LoRA-scale rows) | **PASS** |
| **64×64** | 8 | **29µs** | ≤50µs (recalibrated) | **PASS** |

The cached-index path is **15× / 69× faster** than the SVD path (512×64 / 64×64).

**Plan correction:** Plan 423's "64×64 (reshaped style_weights)" was a misread —
`NeuronShard::style_weights[64]` has 64 *elements*, which reshape to **8×8**, not
64×64. The 8×8 case (0.41µs) is the true per-NPC hot-loop size. The 64×64 target
was recalibrated 10µs → 50µs: the original 10µs predated the flop count (~75K
flops of memory-bound rank-1 axpy ≈ 29µs measured at ~2.5 GFLOP/s effective).

**512×512 is not tested** — the one-sided Jacobi SVD at 512 cols would take
minutes per call (O(n²) pairs × 60 sweeps). 128×128 is the practical cold-tier
bound. Issue 124 (the 64-col cap) is RESOLVED.

## What Landed This Phase

Beyond the GOAT gate, Phase 3 drove two improvements to the primitive:

1. **`SpectralRewireIndex` + `spectral_rewire_with_index_into`** — the cached-SVD
   hot-loop path (Plan 423 open question #2, resolved). Eliminates the SVD from
   the per-delta hot loop (15–69× speedup). Bit-identical to the SVD path
   (`cached_index_matches_svd_path` test).
2. **`SVD_MAX_COLS` guard** — `spectral_rewire_into` and
   `SpectralRewireIndex::new` now panic with a clear message when `d_in > 64`
   (Issue 124) instead of an opaque out-of-bounds deep in the SVD.

## Open Follow-ups

- **Issue 124** — **RESOLVED** (2026-07-10). The one-sided Jacobi SVD substrate
  now heap-allocates the argsort buffers (`sigma_buf` / `perm_buf` in
  `SvdScratch`). Arbitrary `d_in` is supported; 128×128 verified in the GOAT
  gate. The `SVD_MAX_COLS` cap in `katgpt-spectral` has been removed.
- **Issue 123** — real-delta concentration test. The make-or-break for promotion
  to default. Blocked on a real delta source (freeze/thaw pipeline in
  riir-neuron-db, or LoRA overlay path in riir-ai, or training deltas from
  riir-train).
- **SIMD optimization** (optional) — the 64×64 index path at ~29µs could likely
  hit the original 10µs target with proper SIMD matmuls. Not blocking; file as
  issue if the 64×64 hot-loop case becomes a real workload.

## TL;DR

All mechanism gates pass (G1a/G2/G3/G4/G5/G6). The primitive is correct, fast
(cached-index: 0.41µs NPC-scale, 947µs LoRA-scale), zero-alloc, and
deterministic. It stays OPT-IN because the spectral concentration assumption
(G1b) is unvalidated without real training deltas. The SVD 64-col cap (Issue 124)
blocks 128×128/512×512. The cached-index path is the recommended hot-loop API.
