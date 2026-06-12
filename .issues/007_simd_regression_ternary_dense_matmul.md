# Issue 007: SIMD Regression — Ternary Matvec (-34%) and Dense Matmul (-18%)

## Status: OPEN

## Severity: HIGH

## Affected Benchmarks

| Benchmark | Peak (762f2f72) | Current (HEAD) | Regression |
|-----------|----------------|----------------|------------|
| Ternary matvec 64×64 | 950,728 ops/s | ~600K ops/s | -36% |
| Ternary matvec 128×128 | 236,933 ops/s | ~156K ops/s | -34% |
| Ternary matvec 256×256 | 60,883 ops/s | ~39K ops/s | -35% |
| Dense matmul 64×16 | 11,986,815 ops/s | ~9.8M ops/s | -18% |
| Dense matmul 128×32 | 4,502,140 ops/s | ~3.6M ops/s | -19% |

## Root Cause

Regression introduced between `762f2f72` and `f88342ca` (689 commits).
The SIMD kernel implementations themselves were refactored in `crates/katgpt-core/src/simd.rs`:

### Suspect Commits (touching simd.rs, 17 total)

1. `35e7526a` — `#[inline(always)]` on SIMD kernels (24 public dispatch functions)
2. `4254bb4e` — SIMD inlining, branch-free sparse, field reorder
3. `2776eab8` — SIMD pipeline hiding, fused kernels
4. `74528b1b` — branchless ternary, SIMD reciprocal fusion
5. `c31853af` — 4-accumulator NEON/AVX2 sum & max reductions
6. `27d26c5b` — SIMD Wall Attention hot-path

### Specific Changes in neon_dot_f32

- Added SIMD remainder loop (4-wide FMA tail) — neutral for 16/32 col sizes
- Added `#[inline(always)]` — forces inlining at every call site, causes i-cache pressure

### Possible Causes

1. **`#[inline(always)]` code bloat** — 24 public dispatch functions force-inlined, duplicating dispatch+implementation at every call site. With 120+ default features, many more call sites exist.
2. **Compilation unit growth** — simd.rs gained 550 lines (+30%), changing function alignment and i-cache layout
3. **Feature flag bloat** — 32 → 120+ default features increase total binary size (1.83MB), affecting i-cache hit rates for hot SIMD loops

## Action Items

- [ ] Bisect the 17 simd.rs commits to identify the exact regression commit
- [ ] Profile with `perf stat` to check L1i-cache miss rate difference
- [ ] Consider reverting `#[inline(always)]` to `#[inline]` on public SIMD dispatch functions
- [ ] Consider feature-gating `Mutex` fields in `BanditPruner` to reduce struct size (related: Δ-Bandit remaining 2x gap from 65M to 140M peak)

## Notes

- The regression is consistent across multiple cold runs — not thermal
- Forward (flat), game benchmarks, and noise benchmarks are unaffected
- Only SIMD-heavy microbenchmarks (ternary matvec, dense matmul) are affected
- Binary size: 1.83MB (vs unknown at peak, but feature bloat clearly contributes)
