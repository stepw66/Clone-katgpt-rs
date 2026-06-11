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

- [ ] Implement at least β-A (baseline) and β-D (VortexFlow-based)
- [ ] Benchmark: compare heuristic β vs `log(T/t)` baseline on synthetic KV cache
- [ ] Verify non-degenerate attention: no single latent dominates >50% of attention mass
- [ ] Verify no collapse: attention not uniformly distributed (entropy < max_entropy × 0.8)
- [ ] Gate behind `still_kv` feature flag
- [ ] File stays < 2048 lines

## Benchmark Plan

1. Generate synthetic KV cache (T=8192, d=64, H=32)
2. Run StillKV compaction with each β strategy (t=128, t=256, t=512)
3. Measure attention distribution over compact slots during generation
4. Compare against MUX-Latent selection baseline
5. Report: per-latent attention mass, entropy, generation quality metric

## Dependencies

- VortexFlow α-entmax routing scores (existing)
- DashAttention sparsity scores (existing)
- `still_kv` feature flag (new)

## Notes

This is the highest-risk unknown in the StillKV fusion. If no heuristic β produces non-degenerate attention, the entire modelless synthesis approach may not work — would need to pivot to trained compactor (fuel tier) or abandon synthesis in favor of selection (MUX-Latent already works).
