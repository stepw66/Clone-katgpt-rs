# Plan 422 — Cochain Point Sampler GOAT Gate Results

**Date:** 2026-07-10
**Plan:** [katgpt-rs/.plans/422](../.plans/422_cochain_point_sampler_primitive.md)
**Research:** [katgpt-rs/.research/404](../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md)
**Status:** ✅ **ALL GATES PASS — G1–G3 from unit tests, G4 + G5 from this benchmark. Stays opt-in (Gain verdict).**

---

## Gate-by-gate results

| Gate | Criterion | Target | Measured | Verdict |
|------|-----------|--------|----------|---------|
| **G1** | Linear-precision exactness: bilinear λ reproduces `f(x,y) = αx + βy + γ` exactly | error < 1e-5 | 1250 interior points, all < 1e-5 | ✅ PASS (unit test) |
| **G2** | Partition of unity: `Σⱼ λⱼ = 1`, `λⱼ ≥ 0` | tol 1e-6 | sum = 1, all non-negative (quad + tri) | ✅ PASS (unit test) |
| **G3** | C⁰ continuity across primitive boundaries | discontinuity < 1e-5 | sincos boundary u=±1 → 0 diff; barycentric sort across 6 vertex perms → 0 diff | ✅ PASS (unit test) |
| **G4** | Zero-alloc steady state (100 calls after warmup) | `0 allocs` | `0 allocs` (quad + tri) | ✅ PASS |
| **G5** | Latency: `sample_cochain_at_point_quad_into` on 64×64 grid, dim=8 | `< 200 ns` (target < 100 ns) | **11.2 ns/call** | ✅ PASS |

---

## G4 — zero-alloc methodology

The `CountingAllocator` wraps `std::alloc::System` and atomically counts every
`alloc()` call. After 1 warmup call (to allow any lazy initialization), the
counter is snapshotted, then 100 calls are issued with black-boxed inputs at
drifting interior points. The delta is the steady-state allocation count.

Both paths (`sample_cochain_at_point_quad_into` and `sample_point_tri_into`)
are zero-alloc by construction — all output is written into caller-provided
slices (`&mut [f32]`) or a pre-allocated `PointSamplerScratch`. No `Vec` growth
or heap allocation occurs in the hot path.

---

## G5 — latency methodology

**Workload:** 64×64 vertex grid (`CellComplex::grid_2d(64, 64)`), rank-0
`CochainField` with `dim = 8` (a realistic multi-channel field, matching HLA
scalar count). Deterministic SplitMix64 PRNG data (seed `0x4220_0710_2026`).

**Measurement:** 100 warmup iterations (cache + branch predictor stabilization),
then 10_000 measured iterations with `black_box` on all inputs. Points drift
across interior quads (`px += 0.37`, `py += 0.29`, wrapping at 60.0) to prevent
the branch predictor from learning a single access pattern.

### Results (two consecutive runs, Apple Silicon release build)

| Path | Run 1 | Run 2 | Gate |
|------|-------|-------|------|
| `sample_cochain_at_point_quad_into` (raw bilinear, **gated**) | 11.3 ns | 11.2 ns | < 200 ns ✅ |
| `sample_point_quad_into` (Sincos n=4, report only) | 82.3 ns | 82.9 ns | — |
| `sample_point_tri_into` (BarycentricSortCdf, report only) | 11.0 ns | 11.2 ns | — |

The raw quad path (the gated function) runs at **~11 ns** — **8.8× under the
plan's aspirational < 100 ns target** and **17.7× under the < 200 ns gate**.
The sincos encoding adds ~70 ns (16 `sinf` + 16 `cosf` calls for n_harmonics=4
across 2 axes), still well within budget. The triangle path is equally fast
(~11 ns) since barycentric interpolation + sort + CDF remap is all scalar
arithmetic.

---

## G4/G5 labeling note

Plan 422 uses a different G-number convention than Plan 407:
- **Plan 422:** G4 = zero-alloc, G5 = latency
- **Plan 407:** G4 = latency, G5 = zero-alloc

This benchmark follows the **Plan 422** convention (G4 = alloc, G5 = latency),
matching the plan's Phase 3 task labels and the module doc-comments in
`point_sampler.rs`.

---

## Promotion decision

**Stays opt-in** (`cochain_point_sampler = []`). This is a substrate-completeness
primitive (fills the DEC continuous-read gap), not a default-path improvement.
There is no incumbent to demote — continuous intra-primitive sampling is a new
capability with no predecessor. Per the Research 404 Gain verdict, it's a quality
knob for future consumers (continuous terrain queries, belief-field rendering),
not a perf/correctness win on the existing default path.

---

## Validation commands and results

```bash
export CARGO_TARGET_DIR=/tmp/bench_422_goat

# 1. Correctness gates (G1, G2, G3) — unit tests
cargo test -p katgpt-dec --features cochain_point_sampler --no-default-features --lib
# → 198 passed; 0 failed (13 new tests from point_sampler module)

# 2. Perf gates (G4, G5)
cargo bench -p katgpt-dec --features cochain_point_sampler --no-default-features \
  --bench bench_422_cochain_point_sampler_goat -- --nocapture
# → G4: 0 allocs (quad + tri) PASS; G5: 11.2 ns < 200 ns PASS

# 3. All-features combo check (merkle_root / can_freeze lesson)
cargo check --workspace --all-features
# → Finished
```

---

## Files changed

| File | Change |
|------|--------|
| `crates/katgpt-dec/benches/bench_422_cochain_point_sampler_goat.rs` | NEW: G4 (zero-alloc) + G5 (latency) perf gates |
| `crates/katgpt-dec/Cargo.toml` | Added `[[bench]]` entry for `bench_422_cochain_point_sampler_goat` |
| `.plans/422_cochain_point_sampler_primitive.md` | T3.4, T3.5 promoted from `[-]` to `[x]` with results |
| `.benchmarks/422_cochain_point_sampler_goat.md` | NEW: this file |

---

## Cross-references

- [Research 404](../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md) — Cells2Pixels paper distillation (Gain verdict)
- [Plan 422](../.plans/422_cochain_point_sampler_primitive.md) — task breakdown + GOAT gate spec
- Plan 407 — Sheaf-ADMM GOAT (bench file template + CountingAllocator pattern)
