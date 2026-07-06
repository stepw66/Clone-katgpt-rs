# Plan 397 HGA GOAT Gate Report — G2-proxy NEGATIVE RESULT

**Date:** 2026-07-06
**Plan:** [`.plans/397_hierarchical_global_attention.md`](../.plans/397_hierarchical_global_attention.md)
**Research:** [`.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md`](../.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md)
**Paper:** [arxiv 2606.30709](https://arxiv.org/abs/2606.30709) — Frank, Fedosov, Grinenko (BMW Group) 2026
**Test:** `tests/bench_397_hga_goat.rs` (run: `cargo test --features hga,dash_attn,msa_sparse --test bench_397_hga_goat -- --nocapture`)
**Verdict:** **G2-proxy FAIL — HGA group-tier routing does not improve needle retrieval over DashAttention chunk-only routing on random-key NIAH. Keep `hga` opt-in.**

---

## Executive Summary

The G2-proxy (modelless NIAH routing comparison) **FAILED**: HGA's sub-chunk group tier won only 2/12 trials against DashAttention's chunk-only routing. The root cause is that group summaries (mixed-RoPE mean-pool of 16 random keys) dilute the single-needle signal below the dot-product detection threshold.

This mirrors the **MSA R225 GOAT-FAILED precedent** (blockwise sparse with per-GQA-group + max-pool failed on our harness) — both share the class of "sub-chunk routing via summary scoring," and both fail when the summary averages over random distractor keys.

The full G2 (transformer-level loss-gap) is deferred to riir-train and may still pass — the paper's result uses trained model keys with semantic structure, where group summaries are more informative than random-key summaries.

---

## Gate Results

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** Full-coverage = SDPA | < 1e-5 abs diff | ✅ PASS (Phase 1, `forward_hga_full_coverage_equals_causal_sdpa`) | ✅ |
| **G2-proxy** NIAH routing (load-bearing) | HGA wins ≥ 50% of trials | **2/12 trials won** (need ≥ 6) | ❌ **FAIL** |
| **G3** No-regression | `--all-features` clean | ✅ PASS (Phase 1) | ✅ |
| **G4** Alloc-free hot path | 0 allocs steady-state | ~8 Vec allocs/call (Phase 1 reference) | ⚠️ Informational (optimization target) |
| **G5** Latency | HGA ≤ 1.5× DashAttention | **1.12×** (3.53ms vs 3.14ms at 32K context) | ✅ **PASS** |

---

## G2-proxy Detailed Results

### Config
- n_chunks=128, C=64, GS=16, D=64, rope_theta=10000 (Gemma 2-style)
- Total tokens: 8192
- Depths tested: 25%, 50%, 75% of sequence
- Budgets tested: 3.13%, 6.25%, 12.5%, 25% of total tokens

### Full table

| Depth | Budget | DashAttention (fetched/cos/hit) | HGA (fetched/cos/hit) |
|-------|--------|----------------------------------|-----------------------|
| 25% | 256 (3.13%) | 384/-0.031/no | 384/0.023/no |
| 25% | 512 (6.25%) | 640/0.005/no | 640/-0.017/no |
| 25% | 1024 (12.5%) | 1152/-0.036/no | 1152/-0.026/no |
| 25% | 2048 (25%) | 1728/0.071/no | 1728/0.071/no |
| 50% | 256 (3.13%) | 384/0.026/**YES** | 384/0.023/no |
| 50% | 512 (6.25%) | 640/0.051/**YES** | 640/0.005/**YES** |
| 50% | 1024 (12.5%) | 1152/0.001/**YES** | 1152/0.010/**YES** |
| 50% | 2048 (25%) | 1728/0.099/**YES** | 1728/0.099/**YES** |
| 75% | 256 (3.13%) | 384/-0.031/no | 384/0.023/no |
| 75% | 512 (6.25%) | 640/0.005/no | 640/-0.017/no |
| 75% | 1024 (12.5%) | 1152/-0.036/no | 1152/-0.026/no |
| 75% | 2048 (25%) | 1728/0.071/no | 1728/0.071/no |

### Key observations

1. **Both methods retrieve the needle iff the needle's chunk is selected by entmax.** The needle's chunk (at 50% depth) scores high in chunk-level entmax because the needle's distinctive key inflates the chunk summary. At 25% and 75% depths, the needle's chunk is not in the top-k_c selected chunks, so neither method retrieves it.

2. **HGA's group-tier routing does NOT change which chunks are selected.** Both methods use the same chunk-level entmax scoring. The group tier only affects which tokens WITHIN selected chunks are fetched — it can't help if the needle's chunk wasn't selected.

3. **At 50% depth (needle chunk selected), HGA sometimes misses the needle that DashAttention catches.** At the 3.13% budget, DashAttention fetches the entire needle chunk (including the needle token), while HGA's group-level scoring may NOT select the needle's group within the chunk — the group summary of 16 random keys dilutes the needle signal.

4. **Cosine similarities are near-zero for both methods** when the needle is not fetched. When the needle IS fetched, the cosine is low (~0.05-0.10) because SDPA distributes attention across all fetched tokens, not just the needle.

### Root cause analysis

The G2-proxy fails because **group summaries of random keys dilute the needle signal**:

- A group summary is the mixed-RoPE mean-pool of 16 keys. When 15 of those keys are random, the summary is dominated by noise. The needle's key (1/16 of the summary) is below the dot-product detection threshold.
- The paper's result uses a TRAINED model where keys have semantic structure — in that setting, group summaries cluster meaningfully and the needle's group scores distinctly. In our modelless synthetic with random keys, there is no semantic structure to exploit.

This is the **same failure mode as MSA R225** (GOAT-FAILED): sub-chunk routing via summary scoring fails when the summary averages over noise.

### D=128, theta=1M (Qwen3-style) result

- DashAttention: fetched=640, needle=no, cos=-0.047
- HGA: fetched=640, needle=no, cos=-0.092

No improvement at larger D or different rope_theta.

### Tight budget test (256 tokens, 256 chunks)

- DashAttention: fetched=384, needle=no, cos=-0.089
- HGA: fetched=384, needle=no, cos=-0.047

HGA's theoretical advantage (spread budget across more chunks via groups) did not materialize — the group summaries are too noisy to identify the needle's group.

---

## G5 Latency Result

- **DashAttention:** 3,143,957 ns/iter (3.14ms)
- **HGA:** 3,532,404 ns/iter (3.53ms)
- **Ratio:** 1.12× (target ≤ 1.5×)
- **Verdict: PASS** — the group-tier scoring pass adds only 12% overhead.

This means the group tier is computationally cheap; the quality issue is in the summary construction, not the latency.

---

## G4 Allocation Count

- **Phase 1 reference implementation:** ~8 Vec allocations per routing call (chunk scores, chunk probs, scored chunks, group scores, working set keys/values, SDPA logits + output).
- **Verdict:** Informational. Zero-alloc optimization is a Phase 3 target only if G2 passes (it didn't).

---

## What HGA ships anyway (survives the negative result)

Despite the G2-proxy failure, the HGA primitive ships as opt-in because:

1. **The mechanism is correct** (G1 proves full-coverage = SDPA).
2. **The latency overhead is acceptable** (G5: 1.12×).
3. **The G2-proxy is a MODELLESS proxy** — it uses random keys, not trained model keys. The paper's result uses Qwen3 with semantic key structure. The full G2 (transformer-level) may still pass with a trained model.
4. **The tiered KV store (`TieredKvStore`)** is independently useful as a generic route-and-fetch abstraction (not gated by `hga`).
5. **The mixed-RoPE summarizer** is an alternative summary construction that may be useful in other contexts (e.g., as a replacement for DashAttention's learned summary query).

---

## Phase 3 Decision: T3.3 (G2 FAIL)

Per Plan 397 Phase 3 T3.3:

- ✅ Keep `hga` opt-in.
- ✅ Document as negative result (this file).
- ✅ Investigate sub-component failures:
  - **Group tier routing** (the group-level dot-product scoring) is the failing component — group summaries of random keys dilute the needle signal.
  - **Mixed-RoPE summarizer** is not the bottleneck (the summaries are correctly computed; the issue is that averaging random keys produces noise regardless of RoPE handling).
  - **Route-and-fetch store** works correctly (G1 proves it).
- The primitive may be revisited if a trained model (riir-train) shows that semantic key structure makes group summaries informative.

---

## Comparison to MSA R225 (GOAT-FAILED precedent)

| Aspect | MSA R225 | HGA G2-proxy |
|--------|----------|-------------|
| Class | Blockwise sparse + per-GQA-group + max-pool | Sub-chunk group routing + dot-product on summaries |
| Failure mode | max-pool over random keys loses needle signal | mean-pool (mixed-RoPE) over random keys loses needle signal |
| Root cause | Summary scoring on random keys | Same |
| Verdict | GOAT-FAILED, kept opt-in | GOAT-FAILED (proxy), kept opt-in |

Both failures share the same fundamental issue: **summary-level routing on random keys does not preserve single-token signals**. This is a modelless-vs-trained gap — trained models have semantic key structure that makes summaries informative.

---

## Next steps

1. **Full G2 (transformer-level)** — deferred to riir-train. Train a micro-GPT with HGA routing vs DashAttention routing at matched sparsity. The paper's result (0.01-0.02 nat loss gap) suggests HGA should work with trained keys.
2. **Alternative summary constructions** — the mixed-RoPE summarizer is one option; DashAttention's learned summary query is another. A head-to-head with trained keys would clarify which is better.
3. **TieredKvStore as standalone** — the generic route-and-fetch abstraction is independently useful (riir-neuron-db dendritic branch retrieval, riir-ai mmap-backed store).
