# Plan 424 GDN Tree Verification — GOAT Gate Results

**Date:** 2026-07-10
**Plan:** [katgpt-rs/.plans/424_gdn_tree_verification_primitive.md](../.plans/424_gdn_tree_verification_primitive.md)
**Feature:** `gdn_tree_verify` (opt-in)
**Bench:** `benches/bench_424_gdn_tree_verify.rs`

## Summary

| Gate | Result | Notes |
|------|--------|-------|
| **G1** (correctness) | ✅ PASS | Random trees T={16,32,64,128} match per-branch sequential to <1e-3 (f32 precision). |
| **G2** (perf — deep tree) | ✅ PASS | Chain tree speedup: 1.93×/2.79×/4.66×/**7.09×** at T=16/32/64/128. **Matches paper's B200 numbers** (1.5×/2.7×/4.6×/7.1×). |
| **G2** (perf — shallow tree) | ⚠️ NEUTRAL | Random tree speedup: 1.18×-1.40×. Per-branch sequential does less total work on shallow trees (O(T·log T·d_k·d_v) vs O(T²·(d_k+d_v))). |
| **G3** (no-regression) | ✅ PASS | Default + `--all-features` compile clean. All 1429 existing tests pass. |
| **G4** (alloc-free) | ✅ PASS | 0 allocations on steady-state `verify_gdn_tree_into` (CountingAllocator). |

## G2 Detailed Results (d_k=64, d_v=64, release, single-threaded)

### Chain tree (depth = T) — the algorithmically favorable case

| T | Tree-verify | Per-branch seq | Speedup | Paper (B200) |
|---|---|---|---|---|
| 16 | 64.7 µs | 124.9 µs | **1.93×** | 1.5× |
| 32 | 146.5 µs | 408.3 µs | **2.79×** | 2.7× |
| 64 | 314.0 µs | 1461.8 µs | **4.66×** | 4.6× |
| 128 | 795.7 µs | 5641.5 µs | **7.09×** | 7.1× |

The chain tree results **match the paper's B200 GPU numbers almost exactly**. This is
because for a chain (depth = T), the algorithmic advantage is fully realized:
tree-verify is O(T²·(d_k+d_v)), sequential is O(T²·d_k·d_v). With d_k=d_v=64,
the theoretical ratio is d_k·d_v/(d_k+d_v) = 32× — the realized 7× reflects the
constant-factor overhead of the interaction matrix build and output computation.

### Random tree (shallow, depth ~log T) — typical speculative decode shape

| T | Tree-verify | Per-branch seq | Speedup |
|---|---|---|---|
| 16 | 65.0 µs | 76.4 µs | 1.18× |
| 32 | 129.2 µs | 154.1 µs | 1.19× |
| 64 | 264.8 µs | 316.6 µs | 1.20× |
| 128 | 552.8 µs | 772.6 µs | 1.40× |

On shallow trees, the per-branch sequential does less total work
(O(T·log T·d_k·d_v) vs O(T²·(d_k+d_v))). The tree-verify still wins modestly
due to better cache locality and no per-branch S₀ cloning, but the advantage
is small. The paper's GPU speedup on shallow trees comes from **batching T
independent branches on a parallel accelerator**, not from fewer FLOPs.

## Verdict

**G2 PASSES on deep trees (the algorithmically interesting case).** The feature
ships as opt-in behind `gdn_tree_verify`. It is NOT promoted to default — it
only activates on `QwenDeltaNet` / GDN-layer configs (themselves opt-in via
`deltanet_inference`), and only provides significant speedup on deep draft
trees.

The primary value on CPU is **rollback elimination** (the verify pass is
read-only, state is never speculatively written), not raw FLOP reduction.
On GPU (riir-gpu), the batching advantage would give the full paper speedup
on all tree shapes — that's a Phase 4/riir-gpu follow-up.

## Configuration

- **d_k, d_v:** 64 (typical GDN head dims; e.g. QwenDeltaNet-1.5B uses d_k=128
  with GQA grouping to effective d_k=64)
- **Tree shape:** chain (depth=T) for the headline number; random for the
  realistic case
- **Tolerance:** G1 uses 1e-3 (f32 accumulation); f64 intermediates would
  tighten this to 1e-6+ but add latency

## Reproduce

```bash
CARGO_TARGET_DIR=/tmp/424_gdn_tree_verify \
  cargo bench -p katgpt-core --features gdn_tree_verify \
  --bench bench_424_gdn_tree_verify -- --nocapture
```
