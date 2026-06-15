# Research 101: CachePrune — Privacy-Aware Fine-Grained KV Cache Sharing

**Paper:** [arXiv:2605.23640](https://arxiv.org/abs/2605.23640) — CachePrune: Privacy-Aware and Fine-Grained KV Cache Sharing for Efficient LLM Inference
**Authors:** Guanlong Wu, Zhaohan Li, Yao Zhang, Zheng Zhang, Jianyu Niu, Ye Wu, Yinqian Zhang (SUSTech + ByteDance, 2026)
**Date:** 2026-05-25
**Verdict:** 🟡 **Conditional Adopt — token-granularity KV sharing + SAT attention analysis useful for our compression pipeline, but primarily a serving infrastructure paper. Game-specific privacy patterns go to riir-ai.**

---

## TL;DR

CachePrune breaks the privacy–efficiency trade-off in multi-tenant KV cache sharing by operating at **token granularity** instead of fixed chunks. Three key algorithms: (1) summed-area table for O(n²) → O(1) intra/inter-attention evaluation, (2) rolling-hash retrieval for variable-length segment matching, (3) sensitivity-masked selective sharing. 4.5× TTFT reduction, 94.57% cache hit rate, 0% direct leakage.

---

## Core Mechanisms

### 1. Token-Granularity KV Cache Sharing

Existing systems share at fixed chunk boundaries (512 tokens in CacheBlend/CacheCraft) or prefix matches only. CachePrune shares at individual token level:

- Sensitive tokens (PII, user input) act as **natural separators**
- Non-sensitive tokens between separators form **dynamic-length reusable segments**
- A single sensitive token in a 512-token chunk no longer invalidates the entire chunk

**Key insight for us:** Our SpecHop prefix matching currently works at prompt level. Token-granularity matching could improve multi-user cache reuse in MMO serving scenarios.

### 2. Summed-Area Table (SAT) for Attention Analysis

The paper's most algorithmically interesting contribution. Computes self-contextualization (intra-attention vs inter-attention) in O(1) per substring after O(n²) preprocessing:

```
T[i,j] ← A[i,j] + T[i-1,j] + T[i,j-1] - T[i-1,j-1]  // in-place

sum(R) = T[x2,y2] - T[x1-1,y2] - T[x2,y1-1] + T[x1-1,y1-1]  // O(1) query
```

**Reusability criterion:** `IntraAttn(l,r) > InterAttn(l,r)` — substring P[l:r] is reusable when tokens attend more to each other than to outside context.

**Key insight for us:** This is a general-purpose attention analysis primitive. Our DashAttention, RTPurbo, and SpectralQuant could use SAT for O(1) importance scoring instead of O(n) scans.

### 3. Rolling-Hash Retrieval

Two-phase matching for variable-length segments:
1. **Prefix filtering:** Slide 128-token window, rolling hash in O(1) per shift → O(n) scan
2. **Full verification:** SHA-256 of matched substring to eliminate collisions

Uses polynomial rolling hash modulo Mersenne prime 2⁶¹-1 with prefix-hash array for O(1) substring hashing.

**Key insight for us:** Our SpecHop pipeline does hash-based cache lookup. Rolling hash could enable sub-prompt matching across SpecHop hops.

### 4. Sensitivity Detector (Pluggable)

Three policies:
- **User-defined:** Regex patterns for PII
- **Classifier-based:** NER models (Presidio, spaCy)
- **Strict masking:** All user input masked, only system prompts shared

The detector runs **post-inference** (doesn't affect TTFT).

---

## Key Results

| Metric | CachePrune | No-Sharing | Δ |
|--------|-----------|------------|---|
| TTFT (QASPER) | 273ms | 797ms | 2.9× faster |
| TTFT (NarrativeQA) | 705ms | 1675ms | 2.4× faster |
| TTFT (QMSum) | 342ms | 1554ms | **4.5× faster** |
| Cache hit rate (QMSum) | 94.57% | 0% | — |
| Cache hit rate (QASPER) | 82.88% | 0% | — |
| Direct leakage (all datasets) | **0%** | — | — |
| Contextual leakage (exact) | 2.2% avg | — | — |
| Contextual leakage (semantic) | 6.4% avg | — | — |

**vs fixed-chunk methods (without privacy):** CachePrune achieves +44% higher cache hit rate than CacheCraft/CacheBlend/EPIC at token-level granularity.

---

## Distillation to Our Stack

### What Aligns

| Our Component | CachePrune Connection | Synergy |
|--------------|----------------------|---------|
| **SpectralQuant / OCTOPUS** | SAT for intra/inter attention analysis | Could use SAT to compute per-segment compression quality instead of global metrics |
| **DashAttention** | Self-contextualization criterion | DashAttn uses α-entmax for sparsity. CachePrune's intra/inter ratio could be an alternative/complementary importance signal |
| **RTPurbo** | Token-level importance scoring | SAT could accelerate RTPurbo's retrieval head identification from O(n) to O(1) per window |
| **SpecHop** | Rolling-hash retrieval | Multi-hop speculation could use rolling hash for sub-prompt matching across hops |
| **PagedKVCache** | Token-granularity sharing | Our PagedKVCache fork() shares prefix pages. Token-granularity extends this to position-independent reuse |
| **Event Log (Plan 124)** | Sensitivity masking | Game event traces have sensitive strategy tokens. CachePrune's selective sharing applies directly |
| **GDN2 / HLA** | O(1) attention analysis | These O(1) decode layers could benefit from SAT-based segment reusability scoring during prefill |

### What Doesn't Align

| Aspect | Why It's Limited |
|--------|-----------------|
| **Multi-tenant serving focus** | Paper assumes vLLM-style multi-user serving. Our current use is single-user inference. Multi-tenant relevance is MMO-server future (Issue 015). |
| **GPU→CPU attention transfer** | KV Annotator transfers full attention matrix to CPU (383ms for 10K tokens). Not needed in our CPU-only stack. |
| **Privacy detector overhead** | Presidio adds up to 8s for 30K tokens. Our game contexts don't need PII detection. |
| **Chunk-based baselines irrelevant** | We don't use CacheBlend/CacheCraft/EPIC. Our KV compression is quantization-based, not chunk-based. |
| **No training implications** | Paper is pure inference optimization. No model weight changes. |

---

## Model-Based vs Modelless

CachePrune is **modelless infrastructure**:
- No model weight changes
- No training required
- Operates on attention scores at inference time
- SAT is a pure algorithmic technique

**Implication for our stack:**
- The SAT algorithm and rolling-hash retrieval are generic primitives → katgpt-rs (MIT)
- Game-specific sensitivity patterns (which game tokens are "sensitive") → riir-ai (private)
- Per-domain recompute thresholds (ρ per game type) → riir-ai (private)

---

## GOAT Pillar Assessment

Per [27_mmo_goat_pillars_decision_matrix.md](../../riir-ai/.docs/27_mmo_goat_pillars_decision_matrix.md):

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| GOAT passed | ❌ (external) | Paper has results but we haven't proven it in our stack |
| MMO-product | ⬜ | Indirect — would matter when MMO server runs multi-tenant inference (Issue 015). Not blocking any current pillar. |
| LoRA-independent | ✅ | Pure algorithmic. No ML model. Works on any attention output. |
| Defensible | ⬜ | The algorithms are public (SAT, rolling hash). Game-specific sensitivity patterns are somewhat defensible. |
| Secret coverage | A2 + B | Game sensitivity patterns (A2) + per-domain episode segmentation (B). Not core secrets. |

**Verdict: NOT a pillar.** This is infrastructure optimization that supports future MMO serving. It doesn't directly contribute to any of the 4 pillars but would strengthen the MMO backbone (Gap 2 in decision matrix).

---

## Super GOAT / Selling Point Assessment

| Aspect | katgpt-rs (MIT) | riir-ai (Private) |
|--------|-----------------|-------------------|
| SAT attention analysis primitive | ✅ Generic algorithm | — |
| Rolling-hash KV segment retrieval | ✅ Generic retrieval | — |
| SensitivityDetector trait | ✅ Generic interface | — |
| Game-specific sensitivity patterns | — | ✅ Private — per-game token classification |
| Per-domain ρ recompute thresholds | — | ✅ Private |
| MMO cross-player KV reuse policy | — | ✅ Private — per-game cache sharing rules |

**The "super GOAT" angle:** In an MMO context, multiple players interact with the same game world. Cross-player KV reuse is a genuine efficiency win. The **game-specific sensitivity patterns** are private IP. A naive implementation shares too much or too little.

**Keep secret:** Game-specific sensitivity patterns, per-domain recompute thresholds. Ship the generic SAT + rolling hash infrastructure.

---

## Connection to Existing Research

- **Research 039 (SpectralQuant):** SAT could accelerate per-segment compression quality evaluation
- **Research 071 (DashAttention):** Intra/inter attention ratio is complementary to α-entmax routing
- **Research 086 (RTPurbo):** SAT for O(1) retrieval head importance scoring
- **Research 091 (SpecHop):** Rolling hash for multi-hop sub-prompt matching
- **Research 199 (Asymmetric KV):** Token-granularity sharing could use asymmetric K/V compression per sensitivity level
- **Research 066 (TileRT):** SAT fits as a pre-tile analysis step
- **Plan 124 (Event Log):** Sensitivity masking directly applicable to game trace sharing

---

## Algorithmic Deep Dive: What We Actually Want

### A. Summed-Area Table for Attention Analysis (HIGH VALUE)

This is the paper's most reusable contribution for our stack. A generic SAT primitive that:

1. Takes an n×n attention matrix A
2. Computes summed-area table T in-place, O(n²)
3. Answers arbitrary rectangular region sum queries in O(1)

**Use cases in our stack:**
- **DashAttention:** Compute per-head sparsity patterns without scanning full attention
- **RTPurbo:** Identify retrieval heads via intra/inter attention ratio
- **SpectralQuant:** Evaluate compression quality per segment
- **EGA (Plan 139):** Compute spectral energy per position using SAT instead of per-token projection

**Implementation:** ~100 lines of Rust. No dependencies. Works on any 2D matrix.

### B. Rolling-Hash Retrieval (MEDIUM VALUE)

Our SpecHop already does cache matching. Rolling hash would enable:

1. Sub-prompt matching across SpecHop hops
2. Position-independent KV reuse across game sessions
3. O(n) scan for incoming prompts against cached segments

**Implementation:** ~200 lines. Needs prefix-hash array + precomputed powers. We already have blake3 for hashing; rolling hash is lighter-weight for prefix filtering.

### C. Sensitivity Masking (LOW VALUE — riir-ai domain)

Game-specific. Not useful for generic katgpt-rs. The trait/interface belongs in katgpt-rs; the implementations belong in riir-ai.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SAT memory overhead | Low | Medium — O(n²) for full attention | Only compute during prefill, not decode. Or compute per-head SAT on smaller matrices. |
| Multi-tenant not needed yet | High | Low | Feature gate. Build for future MMO backbone. |
| Rolling hash collision | Low | Low — SHA-256 verification eliminates false matches | Already handled by paper's two-phase design |
| Over-engineering for single-user | Medium | Medium | Only SAT primitive is worth extracting now. Full CachePrune system can wait for Issue 015. |
| Game sensitivity patterns wrong | Medium | Medium | Start with strict masking (all game tokens = sensitive), relax per domain over time |

---

## Open Questions

1. Can SAT replace our current O(n) per-head importance scan in DashAttention?
2. What is the SAT overhead for our micro configs (block_size=256, n_head=8)?
3. Can rolling hash improve SpecHop's prefix matching speed?
4. What game tokens count as "sensitive" in the MMO context? (e.g., chat = sensitive, world state = not sensitive?)
5. Does token-granularity sharing interact correctly with our PagedKVCache fork() mechanism?

---

## Verdict Summary

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Novelty** | ⭐⭐⭐⭐ | Token-granularity sharing + SAT + rolling hash is genuinely new combination |
| **Relevance** | ⭐⭐⭐ | Useful for future MMO serving, SAT is broadly applicable now |
| **Adoptability** | ⭐⭐⭐⭐ | SAT is ~100 lines, rolling hash ~200 lines. No model changes needed. |
| **GOAT potential** | ⭐⭐ | SAT primitive can be GOAT-proven. Full system requires MMO backbone (Issue 015). |
| **Moat contribution** | ⭐⭐ | Game-specific sensitivity patterns are defensible but niche |

**Actionable takeaway:** Extract the **SAT attention analysis primitive** into katgpt-rs as a generic utility. It's the paper's most reusable contribution and fits our existing attention pipeline. Defer full CachePrune system to riir-ai when MMO backbone is ready.

---

## References

- Wu, G. et al. (2026). CachePrune: Privacy-Aware and Fine-Grained KV Cache Sharing for Efficient LLM Inference. arXiv:2605.23640.
- Crow, F.C. (1984). Summed-area tables for texture mapping. SIGGRAPH.
- CacheBlend (Yao et al., 2025). Fast LLM serving for RAG with cached knowledge fusion.
- CacheCraft (Agarwal et al., 2025). Managing chunk-caches for efficient RAG.
- EPIC (Hu et al., 2024). Efficient Position-Independent Caching for Serving LLMs.
