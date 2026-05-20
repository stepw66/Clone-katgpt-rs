# Benchmark 013: TurboQuant vs SpectralQuant MaxSim CPU Results

**Date:** 2025-01-25
**Plan:** 080 (MaxSim Late-Interaction Scoring)
**Command:** `cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, release profile

## Corrigendum

Initial version of this benchmark compared **4-bit TurboQuant** vs **~3-bit SpectralQuant** with **identity eigenvectors** (no calibration). This was an unfair comparison — different bit budgets and SQ degraded to random rotation fallback. The corrected benchmark below uses the **same 3-bit budget** with **real calibration data** from `calibrate_eigenbasis`, matching the methodology of `tests/bench_spectralquant.rs::bench_spectralquant_cosine_vs_turboquant`. Latest version adds a **4-way matrix** (TQ/SQ × Cosine/MaxSim) to measure the interaction between quantization method and scoring method.

## 1. MaxSim Core Primitive (T2 Correctness)

| Metric | Value |
|--------|-------|
| MaxSim score (Lq=8, Ld=32, dim=64) | 82.6837 |
| Naive reference | 82.6837 |
| Match | ✓ exact |

## 2. SIMD Speedup (T4 Performance)

| Metric | Value |
|--------|-------|
| Config | Lq=32, Ld=256, dim=128, 1000 iterations |
| MaxSim latency | 48.3 µs/call |
| Throughput | 20,721 scores/s |
| Naive latency | 360.0 µs/call |
| **Speedup** | **7.46×** |

## 3. Block Scoring: MaxSim vs Mean-K (T7 Quality)

| Method | Needle | Noise | Separation |
|--------|--------|-------|------------|
| MaxSim | 435.1998 | 21.7600 | **20.00×** |
| Mean-K dot | 2.8900 | 0.6800 | 4.25× |

**MaxSim separation: 4.71× better** at distinguishing needle from noise.

## 4. TurboQuant MaxSim (T9 Correctness, 4-bit)

| Metric | Value |
|--------|-------|
| Config | kv_dim=16, 8 positions, 2 query tokens, 4-bit quantization |
| TurboQuant score | 18.9444 |
| Uncompressed score | 19.1255 |
| **Relative error** | **0.95%** |
| Status | ✓ PASS |

## 5. SpectralQuant MaxSim (T10 Correctness, ~3-bit)

| Metric | Value |
|--------|-------|
| Config | kv_dim=16, 8 positions, 2 query tokens, ~3-bit spectral quantization |
| SQ MaxSim (streaming) | 16.9787 |
| SQ MaxSim (dequantized) | 16.9787 |
| **Roundtrip error** | **0.00%** (exact match) |
| Status | ✓ PASS |

## 6. 4-Way Matrix: TQ/SQ × Cosine/MaxSim (Same 3-bit, Calibrated)

kv_dim=16, 3-bit budget, 16 doc positions, 4 query tokens.
Calibration via `from_keys()` — auto-calibrates from actual data, cannot forget.
Ground truth MaxSim score: 40.8321.

### Results Table

```
┌──────────────────────────────────┬──────────────┬──────────────┐
│ Metric                            │ TurboQuant   │ SpectralQuant│
├ ─ ─ Scoring Quality ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│ Key cosine (reconstruction)       │ 0.9715       │ 0.9845       │
│ MaxSim error (vs uncompressed)    │  40.54%       │  18.90%       │
│ Compression ratio                 │ 5.3×         │ 9.7×         │
├ ─ ─ Latency (10K iters) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
│ Cosine: dequant+cos (16 pos)     │   2.71 µs    │   3.16 µs    │
│ MaxSim: dequant+maxdot (4q×16d) │  10.55 µs    │  11.24 µs    │
└──────────────────────────────────┴──────────────┴──────────────┘
```

### Interaction Analysis: Does MaxSim Amplify Quantization Error?

MaxSim's `max` operation selects the highest dot product per query token. If quantization noise shifts which doc token has the maximum, the error compounds beyond per-vector reconstruction error.

**Amplification factor** = MaxSim error ÷ cosine error:
- **TQ**: 40.5% MaxSim error from 2.8% cosine error = **14.2× amplification**
- **SQ**: 18.9% MaxSim error from 1.6% cosine error = **12.2× amplification**

Both methods show significant amplification (12–14×), meaning MaxSim is inherently sensitive to quantization noise. However, SQ's better base fidelity means its amplified error is still **2.1× lower** than TQ's.

### Summary

| Dimension | Winner | Evidence |
|-----------|--------|----------|
| **Cosine similarity** | **SpectralQuant ✓** | +0.0129 higher (1.3% better) |
| **MaxSim fidelity** | **SpectralQuant ✓** | 2.1× less error (18.90% vs 40.54%) |
| **Compression** | **SpectralQuant ✓** | 81% more (9.7× vs 5.3×) |
| **Cosine latency** | **TurboQuant** | 14% faster (2.71 vs 3.16 µs) |
| **MaxSim latency** | **TurboQuant** | 7% faster (10.55 vs 11.24 µs) |
| **MaxSim amplification** | **SpectralQuant ✓** | 12.2× vs 14.2× (less compounding) |

### Cross-validation with bench_spectralquant test

The existing `tests/bench_spectralquant.rs::bench_spectralquant_cosine_vs_turboquant` confirms the same conclusion at kv_dim=16 (debug build):

| Metric | TurboQuant (3-bit) | SpectralQuant (3-bit) | Delta |
|--------|--------------------|-----------------------|-------|
| Key cosine | 0.9692 | **0.9917** | **+0.0225** |
| Value cosine | 0.9827 | **0.9917** | **+0.0089** |
| Compression | 5.3× | **9.1×** | **+72%** |

## 7. Why the Initial Comparison Was Wrong

The first version of Section 7 showed TQ winning because:

1. **Different bit budgets**: TQ at 4-bit vs SQ at ~3-bit — TQ had 33% more bits
2. **No calibration**: Identity eigenvectors caused SQ to fall back to random rotation (same as TQ), eliminating its eigenbasis advantage
3. **Missing SQ store loop**: `from_keys()` calibrates but does not store keys — the example forgot to call `store_key` after creating the SQ cache
4. **Result**: "4-bit TQ" vs "empty SQ cache" — TQ naturally won

With fair comparison (same bits, real calibration, both caches populated):
- SQ's eigenbasis rotation concentrates variance into fewer dimensions
- SQ's two-regime bit allocation spends bits where they matter
- SQ achieves higher fidelity AND better compression simultaneously
- MaxSim amplifies quantization error 12–14×, but SQ's lower base error means amplified error is still 2.1× better than TQ

## GOAT Gate Summary

| Gate | Metric | Result | Status |
|------|--------|--------|--------|
| T2 | Correctness: naive within 1e-6 | exact match | ✅ |
| T4 | Speedup: ≥2× vs naive | **7.46×** | ✅ |
| T7 | Separation: ≥5% better than mean-K | **371% better** | ✅ |
| T9 | TQ error: < 10% vs uncompressed | **0.95%** (4-bit) | ✅ |
| T10 | SQ streaming vs dequantized | **0.00%** exact | ✅ |
| T15 | Example exercises all primitives | **7/7 sections** | ✅ |

## Confirms Existing Decision

`Cargo.toml` default features include `spectral_quant` and exclude `turboquant` (labeled "legacy baseline for bench/educate only"). This benchmark proves that decision was correct:

- **SpectralQuant**: higher quality, better compression, lower MaxSim amplification, default-on ✓
- **TurboQuant**: simpler, slightly faster, useful as baseline/comparison ✓
- **MaxSim + SpectralQuant** is the optimal combination for late-interaction scoring on compressed KV

## Test Commands

```sh
# Run all tests (683 pass)
cargo test --features "maxsim,turboquant,spectral_quant" --lib --quiet

# Run benchmark example (7 sections including 4-way matrix)
cargo run --example core_05_maxsim --features "maxsim,turboquant,spectral_quant" --release

# Run dedicated SQ vs TQ cosine comparison
cargo test --features "spectral_quant,turboquant" --test bench_spectralquant bench_spectralquant_cosine_vs_turboquant -- --nocapture

# Clippy clean
cargo clippy --features "maxsim,turboquant,spectral_quant" --examples --quiet
```
