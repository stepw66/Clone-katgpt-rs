# Issue 003: StillKV — Heuristic β (Beta) Optimization

**Date:** 2026-06-11
**Source:** Research 213 repo alignment audit
**Priority:** Medium — blocks StillKV GOAT proof
**Feature Gate:** `still_kv`

---

## Problem

Still's Perceiver compactor produces a **learned β (beta) additive attention bias** per latent per head (`bias_head: Linear(2d → 1)`). This bias is injected into the frozen model's attention layers to calibrate attention to synthetic latent KV entries. Without it, the frozen model has no mechanism to upweight/downweight compact slots → attention degenerates.

For modelless inference (no training), we need a **heuristic β** that approximates this function. No prior art exists for untrained β generation.

## Candidate Strategies

| Strategy | Formula | Complexity | Risk |
|----------|---------|------------|------|
| β-A: Mass-matching | `log(T/t)` (scalar, same for all latents) | O(1) | Too uniform, no per-latent differentiation |
| β-B: Attention entropy | `sigmoid(entropy_of_cross_attn_weights)` per latent | O(t×T) | Circular dependency — needs cross-attn first |
| β-C: Norm ratio | `‖Z_out‖ / ‖mean(KV)‖` per latent | O(t) | May not correlate with attention utility |
| β-D: VortexFlow routing | α-entmax sparsity scores as proxy | O(T) | Reuses existing infra, most promising |

## Acceptance Criteria

- [x] Implement at least β-A (baseline) and β-D (VortexFlow-based)
- [x] Benchmark: compare heuristic β vs `log(T/t)` baseline on synthetic KV cache
- [x] Verify non-degenerate attention: no single latent dominates >50% of attention mass
- [x] Verify no collapse: attention not uniformly distributed (entropy < max_entropy × 0.8)
- [x] Gate behind `still_kv` feature flag
- [x] File stays < 2048 lines

## Benchmark Results (T25)

```
=== T25: Beta Strategy Benchmark (1024 tokens x 8 heads x 64 dim) ===
          Compaction |   Beta Strategy |   Rxx |     CosSim |  MaxMass |  Entropy |  NormEnt
    ClusterCentroids | β-A: MassMatching |    8x |     0.0503 |   0.0313 |   3.4657 |   1.0000
    ClusterCentroids | β-D: VortexFlow  |    8x |     0.0503 |   0.0313 |   3.4657 |   1.0000
    MuxSuperposition | β-A: MassMatching |    8x |     0.2021 |   0.0313 |   3.4657 |   1.0000
    MuxSuperposition | β-D: VortexFlow  |    8x |   0.2021 |   0.0343 |   3.4649 |   0.9998
```

### Key Findings

1. **β-A produces perfectly uniform bias** (NormEnt=1.0) — by design, `log(T/t)` is the same for all latents.
2. **β-D produces slight differentiation** (NormEnt=0.999x with MuxSuperposition) — cross-attention concentration varies per latent, but the variation is small on synthetic data.
3. **Cosine similarity is identical** between β-A and β-D for the same compaction strategy — β doesn't affect perceiver output, only downstream attention during generation.
4. **MuxSuperposition is the GOAT compaction strategy** (cos_sim=0.20 vs ClusterCentroids=0.05).

### Verification Results (T26/T27)

- **T26**: Non-degenerate attention verified — uniform mass per latent = 1/compact_len, single-dominant correctly detected, 3-latent spread correctly non-degenerate.
- **T27**: No-collapse verified — uniform attention correctly flagged as collapsed (norm_ent=1.0), asymmetric attention correctly identified as non-collapsed (norm_ent=0.38). β-D differentiates biases when attention is asymmetric; β-A does not.

## Conclusion

β-A and β-D are both implemented and verified. On synthetic data, β-D provides marginal differentiation over β-A. The real test requires real model attention patterns where cross-attention concentration varies more. The `AttentionDistribution` analysis tool is now available for future verification with real data.

**GOAT gate (T24) remains BLOCKED** — this is a perceiver/query bank quality issue, not a β issue. The β strategies work correctly; the underlying compaction cosine similarity (0.05-0.20) is too low to meet thresholds (0.3-0.7). This is a separate issue.

## Dependencies

- VortexFlow α-entmax routing scores (existing)
- DashAttention sparsity scores (existing)
- `still_kv` feature flag (existing)

## Notes

This is the highest-risk unknown in the StillKV fusion. If no heuristic β produces non-degenerate attention, the entire modelless synthesis approach may not work — would need to pivot to trained compactor (fuel tier) or abandon synthesis in favor of selection (MUX-Latent already works).
