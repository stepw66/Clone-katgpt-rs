# FlashLib: GPU Classical ML Operators — Distillation & Verdict

**Date:** 2026-05-28
**Source:** https://flashml-org.github.io/ (Yang et al., 2026)
**Code:** https://github.com/FlashML-org/flashlib
**Local mirror:** `.raw/flashlib/`

---

## Summary

FlashLib is a GPU library (Triton + CuteDSL, NVIDIA-only) that accelerates classical ML operators (KMeans, KNN, PCA, SVD, HDBSCAN, t-SNE, MultinomialNB, etc.) with up to 26×–208× speedups over cuML on H200. Four design principles:

1. **Reformulation** — mathematically equivalent rewriting for GPU-friendliness (e.g., KMeans streaming fused assign avoids materializing N×K distance matrix)
2. **Hardware-aware kernels** — multiple kernel variants per operator, heuristic selection (no autotune loops)
3. **Tolerance routing** (`tol` parameter) — user declares precision budget, dispatcher picks fastest Pareto-optimal variant (bf16, 3xbf16, Ozaki-II INT8, Halko subspace)
4. **Cost-predictable API** (`flashlib.info.estimate`) — ~5µs CPU-only cost prediction (runtime, FLOPs, HBM, roofline regime) without importing torch/triton/cutlass

---

## Distillation: What Maps to Our Stack

### Direct Mapping: No (different domain)

FlashLib targets **NVIDIA GPU + Python + CUDA/Triton/CuteDSL**. Our stack is **Rust CPU SIMD + wgpu (Metal/Vulkan)**. The 15 primitives (KMeans, KNN, PCA, etc.) are GPU-specific kernels we cannot use directly.

### Principle-Level Mapping: Partial

| FlashLib Principle | Our Equivalent | Status | Gap |
|-------------------|---------------|--------|-----|
| **Reformulation** (algorithm→hardware) | Our SIMD kernels: `simd_dot_f32`, `simd_matmul_rows`, `simd_sparse_matmul_rows`, CODA fused kernels, PlasmaPath ternary SIMD | ✅ Already doing this | CPU SIMD reformulation, not GPU tensor core |
| **Hardware-aware kernels** (multiple variants) | `SimdLevel` (Scalar/Neon/Avx2) dispatch, `AttentionMode`, `HlaMode`, `ModelArchitecture` configs | ✅ Already doing this | We have 3 SIMD levels, FlashLib has ~10 GPU kernel variants |
| **Tolerance routing** (`tol` parameter) | Feature gates control precision/accuracy tradeoffs (e.g., `spectral_quant` vs `turboquant`, `asymmetric_kv` key_bits/val_bits) | ⬜ Partial | No unified `tol` dispatcher — our tradeoffs are binary (feature on/off) or per-config |
| **Cost-predictable API** (`info.estimate`) | `spec_cost_model` (Amdahl cost model for speculative decoding), `sr2am_configurator` (UCB1 planning decisions) | ⬜ Partial | We have cost models for specific subsystems, not a unified cost prediction surface |

### What We Already Cover Better

1. **CPU SIMD path**: FlashLib is GPU-only. Our entire inference stack runs on CPU SIMD (ARM NEON, x86 AVX2). For our target (Apple M-series, edge devices), this is the correct choice.
2. **wgpu GPU training**: riir-ai has wgpu-based LoRA training (Metal/Vulkan) — FlashLib's Triton/CuteDSL stack doesn't run on Metal.
3. **Feature gates as tolerance routing**: Our ~80 feature gates serve a similar purpose to FlashLib's `tol` parameter — users declare precision/speed tradeoffs via Cargo features.
4. **SR²AM configurator**: Our `sr2am_configurator` already does budget-aware planning decisions — similar to FlashLib's cost-predictable API but for inference planning, not operator selection.

### What FlashLib Does That We Don't

1. **Unified cost prediction surface**: FlashLib's `info.estimate()` returns runtime/FLOPs/HBM/roofline for any primitive in ~5µs on CPU. We have `spec_cost_model` for speculative decoding only.
2. **Tolerance-driven dispatch**: A single `tol` float routes to Pareto-optimal variant. Our feature gates are compile-time, not runtime.
3. **Agent-native cost API**: Designed for LLM agents to compose pipelines and budget before execution. Our SR²AM does this for planning, but not for the full inference pipeline.

---

## Verdict: NO GAIN — Research Only

| Criterion | Assessment |
|-----------|------------|
| **Direct code reuse** | ❌ Python/CUDA, no Rust path |
| **Algorithm transfer** | ⬜ Classical ML operators (KMeans/KNN/PCA) aren't in our inference hot path |
| **Design principle transfer** | ⚠️ Tolerance routing and cost API concepts are interesting but we already have feature gates + spec_cost_model + SR²AM |
| **Performance gain** | ❌ No perf gain — different hardware target (GPU vs CPU) |
| **GOAT proof potential** | ❌ Nothing to prove — nothing to implement |
| **Game-specific value** | ❌ FlashLib operators are general ML, not game-domain |
| **Super-GOAT potential** | ❌ No secret sauce for game AI |

### Why No Plan, No Feature Gate

1. **Wrong platform**: FlashLib = NVIDIA GPU + Python. We = Rust CPU SIMD + wgpu. Zero code reuse.
2. **Wrong primitives**: KMeans/KNN/PCA aren't in our inference pipeline. Our hot path is attention + sparse matmul + KV cache, which FlashLib doesn't cover.
3. **Already covered**: Our feature gates + `spec_cost_model` + `sr2am_configurator` + `SimdLevel` dispatch cover the same design principles adapted to our platform.
4. **No new insight**: The 4 principles (reformulation, hw-aware, tolerance, cost-prediction) are already in our engineering DNA. FlashLib validates our approach; it doesn't teach us anything new.

### What FlashLib Validates (Indirect)

- **Feature gate philosophy**: FlashLib's `tol` routing validates our ~80 feature gate approach — users want precision/speed control.
- **Cost model importance**: FlashLib's `info.estimate()` validates our `spec_cost_model` and `sr2am_configurator` — predicting cost before execution is architecturally correct.
- **Kernel variant strategy**: FlashLib's multiple kernel variants per operator validates our `SimdLevel` + `AttentionMode` dispatch — no one-size-fits-all.
- **Agent-native design**: FlashLib's cost API for LLM agents validates our `sr2am_configurator` — agents need predictable cost surfaces.

---

## Tasks

- [x] Read FlashLib paper + code
- [x] Map to katgpt-rs architecture
- [x] Map to riir-ai architecture
- [x] Cross-reference with MMO GOAT Pillars decision matrix
- [x] Verdict: NO GAIN — research only, no plan, no feature gate

---

## Reference

```bibtex
@misc{yang2026flashlib,
  title  = {FlashLib: Bringing Flash Magic to Classical Machine Learning Operators},
  author = {Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Mang, Qiuyang and
            Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Gonzalez, Joseph E. and
            Han, Song and Xu, Chenfeng and Stoica, Ion},
  year   = {2026},
  url    = {https://flashml-org.github.io/},
}
```
