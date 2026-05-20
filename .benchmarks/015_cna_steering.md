# Benchmark 015: CNA Steering — Discovery Latency, Modulation Overhead, Quality Preservation

**Date:** 2025-07
**Plan:** 087 (CNA Contrastive Neuron Attribution), Task T9
**Command:** `cargo test --features cna_steering --test bench_cna_steering -- --nocapture`
**Machine:** macOS (Apple Silicon)
**Rust:** edition 2024, release profile

## Test Design

Synthetic benchmark measuring CNA discovery latency, modulation overhead, quality preservation, and game-domain behavior change.

### Configuration

| Parameter | Value |
|-----------|-------|
| Model layers | 6 |
| MLP hidden dim | 128 |
| Total MLP activations | 768 |
| Default top_pct | 0.1% |
| Modulation iterations | 1000 |

## Results

### Benchmark A: Discovery Latency

Measures time to discover a circuit from N contrastive pairs.

| Pairs | Total Slots | Top-K | Time (µs) |
|-------|-------------|-------|-----------|
| 10    | 768         | 1     | TBD       |
| 50    | 768         | 1     | TBD       |
| 100   | 768         | 1     | TBD       |
| 500   | 768         | 1     | TBD       |

**Expectation:** < 100µs for 100 pairs on 6-layer model. Linear in pairs × slots.

### Benchmark B: Modulation Overhead

Measures per-call overhead of `cna_modulate()` with K circuit neurons.

| Circuit Size (K) | Iterations | Total Time (µs) | Per-Call (ns) | Overhead vs Baseline |
|-------------------|------------|-----------------|---------------|----------------------|
| 0 (empty)         | 1000       | TBD             | TBD           | —                    |
| 10                | 1000       | TBD             | TBD           | TBD                  |
| 50                | 1000       | TBD             | TBD           | TBD                  |
| 100               | 1000       | TBD             | TBD           | TBD                  |
| 500               | 1000       | TBD             | TBD           | TBD                  |

**Expectation:** < 1% overhead for K=50 (typical circuit size). O(K) scaling.

### Benchmark C: Quality Preservation

Measures cosine similarity between original and modulated hidden activations.

| Multiplier (m) | Non-Circuit Cosine | Circuit Cosine | Δ Non-Circuit | Δ Circuit |
|----------------|--------------------|----------------|---------------|-----------|
| 0.0 (ablate)  | 1.000              | TBD            | 0.000         | TBD       |
| 0.5            | 1.000              | TBD            | 0.000         | TBD       |
| 1.0 (baseline) | 1.000             | 1.000          | 0.000         | 0.000     |
| 1.5            | 1.000              | TBD            | 0.000         | TBD       |
| 2.0 (amplify) | 1.000              | TBD            | 0.000         | TBD       |

**Paper benchmark:** CNA quality > 0.97 at all strengths, CAA < 0.60 at max.

### Benchmark D: Game Domain Contrastive Pair Collection

Measures contrastive pair collection from Go games.

| Games | Moves/Game | Positive Obs | Negative Obs | Ratio |
|-------|------------|--------------|--------------|-------|
| 5     | ~150       | TBD          | TBD          | TBD   |
| 10    | ~150       | TBD          | TBD          | TBD   |
| 20    | ~150       | TBD          | TBD          | TBD   |

**Expectation:** Game domains produce natural contrastive pairs without manual labeling.

## GOAT Verdict

| Test | Metric | Threshold | Result | Pass |
|------|--------|-----------|--------|------|
| A: Discovery | Latency (100 pairs) | < 100µs | TBD | TBD |
| B: Modulation | Overhead (K=50) | < 1% | TBD | TBD |
| C: Quality | Non-circuit cosine | > 0.99 | TBD | TBD |
| D: Game pairs | Obs count (20 games) | > 0 both | TBD | TBD |

## Architecture Notes

### Why CNA over CAA

| Property | CNA (neuron-level) | CAA (residual-stream) |
|----------|-------------------|----------------------|
| Target | 0.1% MLP neurons | Full residual stream |
| Quality at max steering | > 0.97 | < 0.60 |
| Overhead | O(K), K ≈ 10-50 | O(d_model) |
| No gradients needed | ✓ | ✓ |
| Sufficient statistics | Mean activation difference | Mean activation difference |

### Implementation

- Discovery: `cna_discover()` in `src/pruners/cna.rs`
- Modulation: `cna_modulate()` forward hook in `src/transformer.rs`
- Feature gate: `cna_steering = ["bandit"]`
- Game pairs: `GoContrastivePairs`, `BomberContrastivePairs`, `FftContrastivePairs`

## References

- Paper: [arXiv:2605.12290](https://arxiv.org/pdf/2605.12290)
- Research: `.research/53_CNA_Contrastive_Neuron_Attribution.md`
- Plan: `.plans/087_cna_contrastive_neuron_attribution.md`
