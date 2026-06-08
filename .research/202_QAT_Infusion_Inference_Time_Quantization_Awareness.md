# Research: QAT Infusion — Inference-Time Quantization Awareness (Modelless)

**Date:** 2026-06-08
**Source:** Google Gemma 4 QAT blog + LR-QAT (2406.06385) + LoTA-QAF (2505.18724) + ZeroQAT (2509.00031) + Scaling Law for QAT (2505.14302) + APEX (2506.03296) + CoA-LoRA (2509.25214)
**Status:** Verdict Pending
**Domain:** katgpt-rs (MIT Engine — Modelless Inference)

---

## Source Distillation: Gemma 4 QAT

Google's Gemma 4 QAT release applies quantization-aware training to compress models for edge devices. Key techniques:

| Technique | What | Why |
|-----------|------|-----|
| **QAT vs PTQ** | Simulate quantization noise during training | Minimizes quality loss when compressed |
| **Static activations** | Pre-calculate activation scaling during training | Eliminates runtime scaling computation |
| **Channel-wise quant** | Structure data for mobile accelerator lanes | Native execution, no workarounds |
| **Targeted 2-bit** | Heavy compression on token-gen layers, high precision on reasoning | Save storage, keep intelligence |
| **Embedding + KV compression** | Compress vocabulary + short-term memory | Reduce active memory for long chats |
| **Modality pruning** | Deploy only needed modalities | 1GB text-only E2B |

**Fundamental insight:** QAT's power comes from **making the system aware of its future compression during optimization**. The system learns to distribute information in ways that survive lossy encoding.

---

## Fusion Ideas: Fundamental Applications (Not Direct Mapping)

We do NOT train LLMs. So "QAT during training" doesn't directly apply. But the **fundamental principle** — *optimize for the precision you'll actually use* — applies everywhere in our inference stack.

### Fusion 1: Precision-Aware Speculative Drafting (PASD)

**Core idea:** When speculative decoding drafts tokens, the draft model runs at lower precision than the verifier. Currently, both run at whatever precision the KV cache quantization gives them. What if the **drafter is explicitly trained/optimized to produce drafts that survive quantization error**?

**How it's modelless:** The drafter is a small speculative model (not an LLM). We can apply inference-time precision awareness:
- Feed the drafter's output through the same quantization pipeline the verifier will use
- Compute the quantization-induced error distribution per-token-position
- Use this as a **draft scoring signal** — penalize drafts that land near quantization boundaries (where rounding flips the token)
- This is a runtime calibration step, not training

**Connection to existing work:**
- `SpeculativeGenerator` trait already produces drafts
- `NextLat` (Plan 217) already has belief-state speculative drafting
- KVarN already quantizes KV cache — we just feed the quantization error back into draft scoring

**GOAT gate:** `precision_aware_draft`

**Expected gain:** Fewer draft rejections due to quantization-induced token flips. Current draft acceptance rate ~70-80% → target 85-90% by avoiding boundary tokens.

---

### Fusion 2: Static Calibration Tables (SCT) — QAT's Static Activations, Modelless

**Core idea:** Gemma 4 QAT pre-calculates activation scaling during training. We can pre-calculate quantization scaling at **model load time** instead of per-inference.

**How it's modelless:** Instead of computing per-token activation scales (which KVarN's Sinkhorn normalization does online), we:
1. At model load, run a calibration pass with representative prompts
2. Record the per-layer, per-head activation statistics
3. Bake them into **static scale tables** — one `f32` per (layer, head) pair
4. At inference, use the static tables instead of online Sinkhorn

**Connection to existing work:**
- KVarN's `var_norm.rs` does online Sinkhorn normalization
- TurboQuant/OCT+PQ already do rotation — the scales are stable post-rotation
- This is the modelless equivalent of "static activations"

**GOAT gate:** `static_cal_tables`

**Expected gain:** Eliminate Sinkhorn iterations (currently 4-8 iterations per decode step) at the cost of a one-time calibration. ~10-15% decode speedup for long sequences. Quality loss: near-zero for well-calibrated models.

**Risk:** If input distribution shifts (new domain), static scales may be suboptimal. Mitigation: periodic recalibration trigger when River Valley signal detects distribution shift.

---

### Fusion 3: Channel-Wise SIMD Routing — QAT's Channel-Wise Quant, Modelless

**Core idea:** Gemma 4 structures quantized data to match mobile accelerator lanes. We can structure our SIMD quantization to match **actual CPU cache line boundaries**.

**How it's modelless:** Current SIMD ternary matvec (`simd_ternary_matvec`) processes all channels uniformly. But:
- NEON has 128-bit lanes (4 × f32)
- AVX2 has 256-bit lanes (8 × f32)
- Cache lines are 64 bytes (16 × f32)

We can **pre-arrange weight layout** so that quantization boundaries align with SIMD lane boundaries. This eliminates cross-lane gather/scatter in the quantize/dequantize step.

**Connection to existing work:**
- `TernaryWeights::quantize_from_f32` already does row-wise quantization
- SIMD kernels in `simd.rs` already have NEON/AVX2 paths
- Just need to align the storage layout with the compute layout

**GOAT gate:** `channel_simd_align`

**Expected gain:** ~5-10% SIMD throughput improvement by eliminating cross-lane operations. This is pure data layout optimization — no algorithmic change.

---

### Fusion 4: Targeted Precision Budget (TPB) — QAT's Targeted 2-Bit, Modelless

**Core idea:** Gemma 4 applies 2-bit to token-gen layers, higher precision to reasoning layers. We can apply **different KV cache quantization precision per attention head** based on sensitivity.

**How it's modelless:** At calibration time (model load):
1. For each attention head, measure how much quantization error degrades the output
2. Rank heads by sensitivity
3. Assign bit-budget: sensitive heads get 4-bit, robust heads get 2-bit
4. The total budget matches our current KVarN average, but distributed non-uniformly

**Connection to existing work:**
- KVarN already does variance-normalized quantization — this extends it to per-head bit allocation
- River Valley signal already identifies peaked vs flat attention — peaked heads need higher precision
- OCT+PQ already does per-layer rotation — per-head precision is the natural extension

**GOAT gate:** `targeted_precision`

**Expected gain:** Same total KV cache size, but better perplexity. The scaling law paper (2505.14302) shows that **FC2 layer outliers are the primary bottleneck** — in our case, it's specific attention heads that dominate error. Targeting them is higher-leverage than uniform quantization.

---

### Fusion 5: Modality-Pruned Context Loading — QAT's Modality Pruning, Modelless

**Core idea:** Gemma 4 drops vision/audio encoders when not needed. We can drop **inference components** when the query doesn't need them.

**How it's modelless:** Not all queries need the full inference stack:
- Simple factual queries: Skip speculative decoding, skip adaptive CoT, use direct decode
- Code generation: Enable DDTree + SynPruner, skip KV compression (precision matters)
- Long context: Enable VortexFlow sparse attention + KV compression, skip speculative
- Reasoning: Enable adaptive CoT + ThoughtFold, full precision

This is a **query-classification → pipeline-pruning** system. The TriggerGate already routes CPU/GPU/ANE — this routes **inference features** based on query type.

**Connection to existing work:**
- TriggerGate already has the routing infrastructure
- Three-Mode Router (Plan 211) already classifies queries
- SubstrateGate (Plan 216) already gates features — this is the precision-aware extension

**GOAT gate:** `modality_pruned_load`

**Expected gain:** 20-40% latency reduction for simple queries that currently run through the full stack. No quality loss because we're skipping components that don't help.

---

### Fusion 6: APEX-Style Async CPU/GPU Quantize/Dequantize Overlap

**Core idea:** APEX (arXiv 2506.03296) overlaps CPU KV cache operations with GPU attention. We can overlap **quantization/dequantization** with **attention computation**.

**How it's modelless:** Current flow:
```
dequantize KV → attention on GPU → quantize new KV → next step
```

With overlap:
```
Thread A: dequantize KV[0..k]     → attention on KV[0..k]
Thread B:                   dequantize KV[k..n] → attention on KV[k..n] (overlapped)
```

The CPU dequantize and GPU attention run concurrently. The CPU is always one chunk ahead, feeding the GPU a stream of dequantized KV blocks.

**Connection to existing work:**
- InferenceRouter already dispatches to GPU
- KVarN already quantizes/dequantizes KV cache
- TileRT (Plan 102) already tiles the execution pipeline

**GOAT gate:** `async_qdq_overlap`

**Expected gain:** ~15-25% throughput improvement on GPU when KV cache is the bottleneck (long sequences). The dequantize cost is hidden behind GPU attention compute.

---

## Verdict: Commercial Open Source Strategy Alignment

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:

| Fusion | Engine (MIT) | Fuel (SaaS) | Verdict |
|--------|-------------|-------------|---------|
| PASD (Draft precision awareness) | ✅ Fits `SpeculativeGenerator` trait | Could be `lora.bin` quality signal | **Engine — modelless** |
| SCT (Static calibration tables) | ✅ Pure inference optimization | N/A | **Engine — default-ON if GOAT** |
| Channel SIMD alignment | ✅ Data layout optimization | N/A | **Engine — default-ON if GOAT** |
| TPB (Targeted precision budget) | ✅ Extends KVarN | Could be calibration data | **Engine — modelless** |
| Modality-pruned loading | ✅ Pipeline pruning | N/A | **Engine — modelless** |
| Async Q/DQ overlap | ✅ GPU pipeline optimization | N/A | **Engine — opt-in (GPU only)** |

**Key insight:** All 6 fusions are modelless inference-time optimizations. They fit the MIT engine perfectly. None require LLM training. None conflict with the engine/fuel split.

The **commercial angle**: If SCT + TPB require calibration data that improves with domain-specific usage, the calibration tables could become part of the fuel (Episode DB → better calibration → better quantization → better translations). This is the same flywheel as `lora.bin` but for quantization parameters.

---

## Research Papers Survey

| Paper | ID | Key Finding | Relevance |
|-------|----|-------------|-----------|
| Scaling Law for QAT | 2505.14302 | FC2 activation outliers are primary quantization bottleneck | Guides TPB head selection |
| Compute-Optimal QAT | 2509.22935 | `tokens-per-param-byte` predicts optimal QAT ratio | Validate SCT quality without experiments |
| ZeroQAT | 2509.00031 | Zeroth-order (forward-only) QAT at inference cost | Could enable on-device recalibration |
| LR-QAT | 2406.06385 | Quantization-grid-aware LoRA, zero overhead after merge | Relevant to riir-ai training side |
| LoTA-QAF | 2505.18724 | Ternary {-1,0,+1} adapters, lossless merge into quantized model | Relevant to PlasmaPath ternary |
| CoA-LoRA | 2509.25214 | One adapter works across all quant configs | Config-aware meta-adapter |
| APEX | 2506.03296 | Async CPU-GPU overlap for KV cache + attention | Blueprint for Fusion 6 |
| KTransformers | SOSP 2025 | 671B MoE on single GPU via CPU/GPU hybrid | Expert placement strategy |

---

## GOAT Gate Feature Flags

All fusions behind individual feature flags for A/B benchmarking:

```toml
[features]
default = ["hybrid_oct_pq", "kvarn", "plasma_path", "spectral_quant", "kv_share",
           "dash_attn", "vortex_flow", "rv_gated_routing", "substrate_gate"]

# QAT Infusion fusions (modelless)
precision_aware_draft = []       # Fusion 1: PASD
static_cal_tables = []           # Fusion 2: SCT
channel_simd_align = []          # Fusion 3: Channel SIMD
targeted_precision = []          # Fusion 4: TPB
modality_pruned_load = []        # Fusion 5: Modality pruning
async_qdq_overlap = []           # Fusion 6: Async Q/DQ (GPU only)
```

**Default-ON rule:** Only enable by default if GOAT proof shows gain AND no perf hurt. Start all as opt-in, promote after benchmarking.

---

## TL;DR

Gemma 4 QAT's fundamental insight — *optimize for the precision you'll deploy at* — translates to **6 modelless inference-time fusions** for katgpt-rs:

1. **PASD**: Draft tokens that survive quantization (precision-aware speculative decoding)
2. **SCT**: Pre-compute quantization scales at load time (static calibration)
3. **Channel SIMD**: Align data layout with SIMD lanes (cache-line-aware quantization)
4. **TPB**: Different bit-widths per attention head (targeted precision budget)
5. **Modality pruning**: Skip inference features the query doesn't need (pipeline pruning)
6. **Async Q/DQ**: Overlap dequantize with GPU attention (APEX-style)

All 6 are modelless, fit the MIT engine, and respect the engine/fuel commercial split. Papers LR-QAT and LoTA-QAF are the strongest cross-pollination for the riir-ai (model-based) side. The scaling law (2505.14302) is the diagnostic tool — compute `tokens-per-param-byte` at load to validate quantization quality without running experiments.
