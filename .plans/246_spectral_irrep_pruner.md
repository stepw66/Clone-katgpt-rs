# Plan 246: Spectral Irrep Pruner — Inference-Time Spectral Collapse Detection

**Status**: Planned
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

- [ ] Add `IrrepPruner` to pruner registry
  - Register in `pruner_factory()` with config
  - Default OFF (behind feature flag)
  - Config: threshold, max_modes

- [ ] Benchmark throughput impact
  - Compare: baseline (no pruner) vs SynPruner vs IrrepPruner vs combined
  - Measure: tokens/sec, latency p50/p99
  - Goal: <5% throughput overhead for spectral check

- [ ] Benchmark accuracy impact
  - Compare acceptance rates with/without IrrepPruner
  - Measure: acceptance rate, speculation accuracy
  - Goal: ≥same acceptance rate, higher quality accepted tokens

- [ ] GOAT gate: promote to default if throughput < 5% overhead AND accuracy ≥ baseline
  - If fail: keep behind feature flag, document why
  - If pass: add to default feature set

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
