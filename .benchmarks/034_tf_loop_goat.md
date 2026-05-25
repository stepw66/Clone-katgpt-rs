# GOAT Proof 034: Training-Free Loop Wrapper — ODE-Refined Sub-Stepping (Plan 136)

> **Date:** 2026-05-25
> **Feature Gate:** `tf_loop` (depends on `lt2_looped`)
> **Depends on:** Plan 136 Phase 0-1 (SubStepStrategy, IterationMode, CacheStrategy, TrainingFreeLoopConfig, tf_loop module)

## Summary

GOAT proof for Training-Free Loop Wrapper: pure inference-time retrofit that re-applies a contiguous mid-stack block of layers with ODE-motivated damped sub-stepping. Core result: **K-stage RK with β=0.5 produces finite, bounded output for K=2,3,4 with cache size independent of K** — confirming the sub-stepping approach adds effective depth without growing memory.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Dim | 64 (layer mode), 128 (cache proof) |
| K (loop count) | 1, 2, 3, 4, 8, 16 |
| β (blend factor) | 0.0, 0.25, 0.3, 0.5, 0.75, 1.0 |
| Strategy | DampedEuler, KStageRK |
| Build | Debug |

## GOAT Proofs

### Proof 1: `proof_tf_loop_finite`

K-stage RK loop with synthetic affine transforms (scale=0.8, bias=0.1) produces finite, non-NaN output for K∈{2,3,4}, β∈{0.25,0.5,0.75}.

**Target:** All outputs finite ✓

### Proof 2: `proof_tf_loop_cache_size`

Output vector dimension is always `dim`, regardless of K. KV cache written once from final (or first) state.

**Target:** cache_entries == dim for all K ∈ {1,2,4,8,16} ✓

### Proof 3: `proof_tf_loop_bypass_free`

When K=0 or window is empty, computation is identity (no state change). β=0 anchor blend is also identity.

**Target:** ||x_out - x_original|| < ε ✓

### Proof 4: `proof_tf_loop_layer_mode_stable`

Layer-by-layer iteration within window produces finite, bounded output (max_abs < 1e6).

**Target:** All outputs finite, no divergence ✓

## Benchmark Targets (Future)

| Metric | Target |
|--------|--------|
| Per-token overhead (K=2) | < 30% over baseline |
| Per-token overhead (K=4) | < 60% over baseline |
| Memory overhead vs baseline | 0% (same KV cache size) |
| Quality (perplexity Δ) | ≤ 0 (not worse) |

## Run

```bash
cargo test --features tf_loop --test test_136_tf_loop -- --nocapture
```
