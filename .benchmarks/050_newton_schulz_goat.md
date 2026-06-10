# Benchmark 050: Newton-Schulz Orthogonalization + River-Valley Diagnostics — GOAT Proof

**Date:** 2026-05-30
**Plan:** 152 (Newton-Schulz Orthogonalization + River-Valley Diagnostics)
**Source:** Research 114 (AMUSE — Anytime Muon with Stable Gradient Evaluation)
**Features:** `newton_schulz`, `river_valley` (default-on)
**Command:** `cargo test --features "newton_schulz,river_valley" --test bench_152_newton_schulz_goat -- --nocapture`

## GOAT Proof Results

| # | Criterion | Threshold | Result | Status |
|---|-----------|-----------|--------|--------|
| T1.1a | 8×8 orthogonality (max |X@X^T - I|) | < 0.5 | 0.382 | ✅ PASS |
| T1.1b | 32×32 orthogonality | < 0.5 | 0.334 | ✅ PASS |
| T1.1c | 64×64 orthogonality | < 0.5 | 0.360 | ✅ PASS |
| T1.1d | All outputs finite (5 sizes × 3 seeds) | No NaN/Inf | All finite | ✅ PASS |
| T1.2 | Convergence in 5 iters (5 seeds) | < 0.5 each | 0.21–0.44 | ✅ PASS |
| T1.3a | Non-square 12×6 | < 0.5 | 0.335 | ✅ PASS |
| T1.3b | Non-square 6×12 | < 0.5 | 0.294 | ✅ PASS |
| T1.3c | Non-square 64×16 (tall-skinny) | < 0.5 | 0.358 | ✅ PASS |
| T1.3d | Allocating vs scratch API match | < 1e-6 | 0 | ✅ PASS (5 shapes) |
| T2.1a | Subspace ratios Pythagorean | r_dom²+r_bulk²=1 | 1.000000 | ✅ PASS |
| T2.1b | Full projection r_dom | ≈ 1.0 | 1.000000 | ✅ PASS |
| T2.1c | Zero gradient | (0, 1) | (0, 1) | ✅ PASS |
| T2.2a | Effective rank identity (4×4) | ≈ 4.0 | 4.0000 | ✅ PASS |
| T2.2b | Effective rank rank-1 | ≈ 1.0 | 1.0000 | ✅ PASS |
| T2.2c | Effective rank nonsquare (2×4) | ≈ 2.0 | 2.0000 | ✅ PASS |
| T2.3a | Cosine similarity constant direction | = 1.0 | 1.000000 | ✅ PASS |
| T2.3b | Cosine similarity orthogonal | = 0.0 | 0.000000 | ✅ PASS |
| T2.3c | Cosine similarity opposite | = -1.0 | -1.000000 | ✅ PASS |
| T2.3d | Cosine similarity single update | = 1.0 | 1.000000 | ✅ PASS |
| T3.1a | Muon output approximately orthogonal | < 1.0 | 0.989 | ✅ PASS |
| T3.1b | 100 Muon steps no NaN/Inf | All finite | All finite | ✅ PASS |
| T3.1c | muon_update vs muon_update_into match | < 1e-6 | 0 | ✅ PASS (3 seeds) |
| T3.2 | Momentum accumulation (5 steps) | Strictly increasing | [2.13, 4.05, 5.78, 7.33, 8.73] | ✅ PASS |
| T3.3a | Scratch/alloc API throughput ratio | < 1.5× | 0.987× | ✅ PASS |
| T3.3b | Muon 16×16 update latency | < 5ms | 653 µs | ✅ PASS |

**25/25 GOAT proofs passed.**

## Orthogonality Threshold Note

Newton-Schulz with 5 fixed iterations and coefficients a=3.4445, b=-4.7750, c=2.0315 produces **approximate** orthogonalization. Singular values converge to [0.68, 1.12] (Keller Jordan's Muon blog), meaning the diagonal entries of X@X^T are in that range (not exactly 1.0). This is by design — for Muon optimizer use, the key property is that the update direction is well-conditioned, not that it's exactly on the Stiefel manifold.

## Throughput Results (Debug Build, Apple Silicon)

| Operation | Size | Latency |
|-----------|------|---------|
| newton_schulz5 (allocating) | 32×32 | 4,198 µs |
| newton_schulz5_into (scratch) | 32×32 | 4,144 µs |
| muon_update_into (full pipeline) | 16×16 | 653 µs |

## Files Changed

| File | Change |
|------|--------|
| `tests/bench_152_newton_schulz_goat.rs` | GOAT proof test (25 tests) |
| `.benchmarks/050_newton_schulz_goat.md` | This file |

## Commands to Reproduce

```bash
# GOAT proof (25 tests)
cargo test --features "newton_schulz,river_valley" --test bench_152_newton_schulz_goat -- --nocapture

# Unit tests only (newton_schulz)
cargo test --features newton_schulz --lib -- newton_schulz --nocapture

# Unit tests only (river_valley)
cargo test --features "newton_schulz,river_valley" --lib -- river_valley --nocapture
```
