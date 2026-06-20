# Issue 014: MSA Arena RULER Benchmark Infrastructure

**Created:** 2026-06-13
**Source:** Plan 256 Phase 3 (deferred task: "Run arena benchmark: msa_sparse vs vortex_flow vs dash_attn vs dense attention")
**Blocking:** Plan 256 full GOAT gate (arena accuracy portion)
**Status:** CLOSED (blocked on trained model weights + RULER dataset + inference harness — none exist in katgpt-rs)

**Closure rationale (2026-06-20):** All three prerequisites (trained model weights, RULER dataset, end-to-end attention inference harness) are explicitly out of scope for the public modelless `katgpt-rs` engine and belong in `riir-ai`/`riir-gpu`. The three Phase 2 micro-benchmarks all FAILED their GOAT gates (per-group 1.003×, KV-outer 1.14×, adaptive-k 0.629), strongly predicting the full arena would also fail. The optimization candidates were already spun out to Issue 015. Reopen when a trained transformer with KV-cache attention lands in riir-ai.

---

## Problem

Plan 256 (MSA Blockwise Sparse Distillation) requires a full arena benchmark comparing `msa_sparse` family routers against `vortex_flow` baseline, `dash_attn`, and dense attention on:

- **RULER-8K, RULER-32K, RULER-128K accuracy** (needle-in-haystack, multi-key-value, variable-tracking)
- **Prefill latency** at 32K, 128K, 512K context
- **Decode latency** at 32K, 128K context
- **Block selection latency** (micro-bench)

This requires:
1. A **trained transformer model** with KV-cache attention layers (the current codebase is modelless inference — router selection is tested with synthetic centroids, not real model attention)
2. **RULER task datasets** (needle-in-haystack JSON, multi-hop QA, etc.)
3. A **full inference harness** that runs the router through actual attention computation (not just `forward_indexer` in isolation)

None of these are available in the current modelless inference codebase.

## What Was Done Instead (Modelless Proxies)

Three Phase 2 micro-benchmarks were run as modelless RULER proxies:

| Benchmark | File | Metric | Result | Verdict |
|-----------|------|--------|--------|---------|
| Per-group | `tests/bench_256_per_group.goat.rs` | Coverage ratio | 1.003× (need ≥1.5×) | ❌ FAIL |
| KV-outer | `tests/bench_256_kv_outer.goat.rs` | Speedup @ 128K | 1.14× (need ≥1.5×) | ❌ FAIL |
| Adaptive-k | `tests/bench_256_adaptive_k.goat.rs` | Recall ratio | 0.629 (need ≥0.90) | ❌ FAIL |

All three **FAILED** their GOAT gates. The micro-benchmark failures predict the full RULER arena would also fail to show ≥5% accuracy gain, because:
- Per-group doesn't increase block coverage (the diversity proxy saturated)
- KV-outer doesn't speed up prefill at long context (block sharing drops)
- Adaptive-k trades recall for compute savings (mathematically bounded)

## Scope of This Issue

### Prerequisites (must exist before arena can run)
- [-] Trained model weights with KV-cache attention (Gemma2-2B scale or equivalent) — blocked: katgpt-rs is modelless; this belongs in riir-ai/riir-gpu
- [-] RULER task dataset downloaded + parsing (needle-in-haystack, multi-key-value, variable-tracking) — blocked: no download/parse code exists; worthless without trained model
- [-] End-to-end attention inference harness that integrates `VortexRouter` selection into actual softmax-weighted value accumulation — blocked: only `KvOuterPrefill::prefill_sparse` has a real attention path; generalizing needs trained model first

### Arena benchmark implementation (once prereqs exist)
- [-] `tests/bench_256_arena_ruler.goat.rs` — compares router configs on real model attention — blocked on prereqs
- [-] Measure RULER accuracy (exact-match / F1) per task per context length — blocked on prereqs
- [-] Measure prefill/decode latency — blocked on prereqs (decode needs real transformer forward)
- [-] GOAT gate: `msa_sparse` ≥5% RULER gain + ≥10% selection speedup → promote to default — blocked on prereqs

### Optimization candidates (if arena confirms failure)
- [-] Per-group: redesign coverage metric to measure per-call partition spread, not cross-query union — **spun out to Issue 015** (tractable now, no prereqs)
- [-] KV-outer: add query batching / increase effective n_queries per block to restore sharing at long context — **spun out to Issue 015** (tractable now, no prereqs)
- [-] Adaptive-k: replace recall@fixed_k with precision@adaptive_k or weighted recall — **spun out to Issue 015** (tractable now, no prereqs)

The three optimization candidates were spun out to [Issue 015](./015_msa_microbench_metric_refinement.md)
because they're metric redesigns on existing synthetic micro-benchmarks and don't
need the arena infrastructure. Everything else here is blocked on trained model
weights + RULER dataset, which are outside katgpt-rs scope.

## Priority

**Low** — the micro-benchmark failures strongly suggest the arena would not flip the GOAT verdict. This issue exists to track the full evaluation for completeness, not because the current verdict is in doubt.

## Related
- Plan 256: `.plans/256_msa_blockwise_sparse_distillation.md`
- Benchmark files: `tests/bench_256_{per_group,kv_outer,adaptive_k}.goat.rs`
