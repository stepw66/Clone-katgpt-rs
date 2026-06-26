# Plan 246: Spectral Irrep Pruner — Inference-Time Spectral Collapse Detection

**Status**: Complete ✅ GOAT PASS (G4 overhead +3.6%, G5 peaked 100%, G5 flat 3.9%)
**Feature Flag**: `spectral_pruner`
**Research**: 214_Spectral_Irrep_Compression_Inference.md
**Depends on**: ConstraintPruner trait, FreqBandit infrastructure (Plan 189)

---

## Motivation

Paper arXiv:2606.02993 proves that converged neural networks exhibit **spectral concentration** — each neuron's Fourier coefficients collapse to a single irreducible representation. At inference time, we can detect whether logit distributions exhibit this "converged" structure (low spectral flatness = few dominant modes) vs "still competing" (high spectral flatness = many competing modes).

This gives us a modelless, training-free pruning signal for speculative decoding.

## Tasks

- [x] Implement `SpectralFlatness` utility function
  - Input: logit slice `&[f32]`
  - Output: spectral flatness score (geometric mean / arithmetic mean of FFT magnitudes)
  - Use existing FFT infrastructure or rustfft crate
  - Zero-alloc: pre-allocated scratch buffer

- [x] Implement `IrrepPruner` struct implementing `ConstraintPruner`
  - Fields: `convergence_threshold: f32`, `max_modes: usize`, `scratch: Vec<f32>`
  - `is_valid()`: FFT → spectral flatness < threshold → valid
  - `batch_is_valid()`: SIMD-batch spectral flatness check
  - Behind feature flag `spectral_pruner`

- [x] Add `IrrepPruner` to pruner registry
  - `IrrepPrunerConfig` struct with defaults (threshold=0.7, top_k=10)
  - `IrrepPruner::from_config()` + top-level `irrep_pruner_from_config()` factory
  - `spectral_pruner` feature propagated to `katgpt-rs/Cargo.toml`
  - GOAT-gated, not in default feature set

- [x] Benchmark throughput impact
  - G1: `set_logits` latency = 2.8μs (FFT on 256 elements, O(n log n))
  - G2: `is_valid` latency = 16ns
  - G3: `batch_is_valid` (256 candidates) = 23-28ns
  - G4: DDTree overhead = +3.2-3.6% (target <5%) ✅

- [x] Benchmark accuracy impact
  - G5: Peaked distribution → 100% acceptance (target ≥95%) ✅
  - G5: Flat distribution → 3.9% acceptance = top_k/256 (target ≤4.3%) ✅

- [x] GOAT gate: PASS — overhead +3.6% < 5%, accuracy 100% converged / 3.9% uncertain
  - Promote to default: add `spectral_pruner` to default features
  - Benchmarks: `bench_246_irrep_pruner_goat.rs` (6 tests, all pass)

## Architecture

```
logits → FFT → |spectrum|² → spectral_flatness() → < threshold → valid branch
                                                          ≥ threshold → prune branch
```

## Constraints

- No allocations in hot path (pre-allocated scratch buffer)
- SIMD-friendly FFT (process 4/8 logit vectors at once)
- CPU/GPU auto-route: CPU for small batches, GPU for large batches
- Compatible with existing DDTree branch evaluation

## Testing

- Unit test: known convergent distribution (single peak) → passes pruner
- Unit test: uniform distribution → rejected by pruner
- Unit test: two-peak distribution with threshold=1 → passes (max_modes=2)
- Integration test: end-to-end speculative decoding with IrrepPruner enabled
- Benchmark: throughput comparison with/without

## References

- arXiv:2606.02993 — Theorem 4.3 (single representation convergence)
- Plan 189 — FreqBandit DFT infrastructure
- katgpt-rs ConstraintPruner trait
