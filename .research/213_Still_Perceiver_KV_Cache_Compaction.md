# Research 213: Still — Perceiver-Based KV Cache Compaction for Modelless Inference

**Date:** 2026-06-10
**Source:** [Still: Amortized KV Cache Compaction in a Single Forward Pass](https://arxiv.org/pdf/2606.07878v1) (O'Neill et al., Baseten, 2026)
**Target:** katgpt-rs modelless inference engine
**Status:** Verdict Pending

---

## Paper Summary

Still introduces a per-layer Perceiver-style compactor that synthesizes compact KV caches from full-context caches in a single forward pass. Key innovations:

1. **Amortized synthesis**: Learned latent queries cross-attend to full KV cache, producing compact keys/values — no per-context optimization
2. **Position-free compaction**: Un-rotate RoPE before compaction, re-rotate after — eliminates position-content coupling
3. **Iterative chunked compaction**: Recurrent compression with fixed local ratio + 1-chunk lookahead buffer
4. **KL divergence training**: Forward KL from full-context teacher to compact-cache student, top-200 vocab support
5. **Identity-style initialization**: Near pass-through at t=T for training stability

**Results**: 8-200x compression, 8k-128k context, 50M params (~1% of base model), 8-22 point RULER advantage over KV-Distill.

---

## Distillation for katgpt-rs (Modelless)

### Constraint Check
- ✅ No LLM training required — Still's Perceiver compactor is a **separate module** from the base model
- ✅ Inference-time only — compaction is a forward pass through the compactor
- ❌ The compactor IS trained (requires KL training against a base model)
- ✅ But: the *pattern* (cross-attention synthesis, position-free, iterative) is applicable modellessly

### Fusion Idea 1: StillKV — Heuristic Perceiver Compaction (NO TRAINING)

**Core Insight**: Replace Still's *learned* latent queries with *heuristic* query banks generated at inference time.

**Architecture**:
```
Full KV cache (T tokens)
    → Un-rotate RoPE (position-free frame)
    → Concat [K; V] per head
    → Cross-attention from heuristic latent queries:
        - TF-IDF centroids from token frequencies
        - Attention-sink patterns (first 4 + last K)
        - VortexFlow α-entmax routing scores as query importance
        - BFCF region centroids as spatial anchors
    → Self-attention refinement (2 blocks)
    → Project to compact Ck, Cv
    → Re-rotate RoPE
```

**Heuristic Latent Query Generation** (replaces learned Z ∈ R^{H×t×d}):
- **Method A**: Cluster-based — run mini-batch k-means on [K;V] concatenation for t clusters, use centroids as queries
- **Method B**: Importance-weighted — use existing DashAttention/VortexFlow attention scores to weight token positions, then subsample weighted average
- **Method C**: Spectral — use existing SpectralQuant eigenbasis to project KV cache to top-t eigenvectors
- **Method D**: MUX-Latent superposition — use existing MUX encoder to produce t superposed latent representations

**Integration Points**:
- `QuantizedKVCache` trait extension: add `compact_into(&self, budget: usize) -> CompactKVCache`
- `VortexFlow` provides the cross-attention routing mechanism
- `DashAttention` provides the α-entmax sparsity for attention scoring
- `MUX-Latent` provides the vocabulary superposition encoder
- `BFCF` provides region-based spatial partitioning
- `ThoughtFold` provides the iterative refinement through chain folding
- `KVarN` provides the variance normalization for position-free frame

**Position-Free Compaction** (pure engineering):
- Un-rotate: apply inverse RoPE to cached keys before compaction
- Compact: run compaction in position-free frame
- Re-rotate: apply RoPE at evenly-spaced output positions
- Offset: continuation tokens get position offset = original_prefix_len - compact_len

**Iterative Chunked Compaction**:
- Fixed local compression ratio c (e.g., c=8 means compress every 8t tokens to t)
- 1-chunk lookahead buffer (raw KV) between compressed chunks
- Matches existing SegmentCheckpoint's growing memory pattern

### Fusion Idea 2: StillCoT — CoT Trace Compaction via Synthesis

**Core Insight**: Apply Still's synthesis compaction to *thinking traces*, not just prefill context.

**Problem**: ThoughtFold prunes CoT steps by *selection* (keep important, discard rest). Still shows synthesis > selection for information preservation.

**Architecture**:
1. Model generates thinking trace (CoT tokens)
2. After trace complete, compact the thinking KV cache via StillKV synthesis
3. The compact thinking trace becomes "compressed working memory"
4. Generation continues against compact trace + prompt + response prefix

**Gain over ThoughtFold**: ThoughtFold achieves 78% CoT reduction via *selection*. StillKV synthesis could achieve similar or better reduction while preserving more distributed information (not just the "important" tokens but blended summaries).

**Integration**:
- Extends `ChainFolder` trait with `compact_trace()` method
- Uses `FoldCache` for KV rollback + `StillKV` for synthesis compaction
- Gated by feature flag `still_cot`

### Fusion Idea 3: StillRSM — Perceiver-Augmented Routing Slot Memory

**Core Insight**: Raven RSM maintains O(1) routing slots. Still's Perceiver can *synthesize* better slot representations from the full KV cache.

**Architecture**:
- Instead of selecting top-K KV entries for RSM slots
- Use cross-attention from t fixed latent queries to synthesize slot representations
- Each slot becomes a *blended summary* of related KV entries, not a single entry

**Gain**: Higher information density per slot → better routing decisions in O(1) time.

---

## Verdict: GOAT/Gain Analysis

### Fusion 1: StillKV (Heuristic Perceiver Compaction)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ No training needed — heuristic queries replace learned ones |
| Gain vs existing | 🟡 Moderate — MUX-Latent already gives 14-29x TTFT reduction, StillKV adds synthesis quality |
| Novel fusion | ✅ Perceiver pattern + VortexFlow routing + BFCF regions is novel |
| Complexity | 🔴 High — cross-attention + self-attention + RoPE handling is non-trivial |
| Hot path impact | 🔴 Risky — cross-attention is O(t*T), only worth it if t << T |

**Verdict: GAIN but GATE** — Implement behind `still_kv` feature flag. Gate on GOAT proof that shows quality improvement over MUX-Latent selection at same compression ratio. The synthesis-vs-selection insight is the key differentiator.

### Fusion 2: StillCoT (CoT Trace Compaction)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ Inference-time compaction of thinking traces |
| Gain vs existing | ✅ High — ThoughtFold is selection-based, StillCoT is synthesis-based |
| Novel fusion | ✅ Still applied to CoT compression is novel |
| Complexity | 🟡 Moderate — reuses StillKV infra |
| Hot path impact | 🟡 Acceptable — compaction happens after trace complete, not during generation |

**Verdict: GAIN** — Natural evolution of ThoughtFold. Implement after StillKV. Can be tested against ThoughtFold's 78% reduction benchmark.

### Fusion 3: StillRSM (Perceiver-Augmented RSM)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ Inference-time only |
| Gain vs existing | 🟡 Moderate — better slot quality vs current selection |
| Novel fusion | ✅ Perceiver for routing slot synthesis is novel |
| Complexity | 🟡 Moderate — smaller scale than full StillKV |
| Hot path impact | 🟡 Acceptable — RSM is already O(1) |

**Verdict: DEFER** — Lower priority. Implement if StillKV proves the synthesis pattern works.

---

## Commercial Strategy Alignment

Per the 003 verdict:
- StillKV is **engine** (MIT, open) — modelless inference-time compaction
- If a trained Perceiver compactor is added later, it becomes **fuel** (riir-ai, private SaaS)
- The position-free compaction pattern is pure engineering → stays in engine
- The heuristic query generation strategies are the open-source moat demonstration
- A trained compactor (KL against base model) would be the SaaS premium

### Engine/Fuel Split:
| Component | Layer | License |
|-----------|-------|---------|
| Position-free RoPE handling | Engine | MIT |
| Iterative chunked compaction pipeline | Engine | MIT |
| Heuristic latent query generation | Engine | MIT |
| Cross-attention synthesis module | Engine | MIT |
| QuantizedKVCache trait extension | Engine | MIT |
| Trained Perceiver compactor weights | Fuel | Private (riir-ai) |
| KL training pipeline for compactor | Fuel | Private (riir-ai) |

---

## Related Work in Our Stack

| Our Feature | Still Analog | Relationship |
|-------------|-------------|-------------|
| MUX-Latent (Plan 238) | Amortized compression | MUX uses vocabulary superposition; Still uses cross-attention synthesis |
| VortexFlow (Plan 196) | Sparse routing | VortexFlow routes tokens; Still routes entire KV cache to latents |
| KVarN (Plan 179) | Variance normalization | KVarN quantizes; Still compresses via synthesis |
| ThoughtFold (Plan 195) | Iterative compaction | ThoughtFold prunes by selection; Still compresses by synthesis |
| BFCF (Plan 213) | Spatial partitioning | BFCF regions → natural heuristic query clusters |
| SegmentCheckpoint (Plan 226) | Growing memory | SegCheckpoint caches segments; Still compresses them iteratively |
| SP-KV (Research 042) | Token utility prediction | SP-KV predicts utility; Still synthesizes from all tokens |
| ShardKV (Research 109) | Asymmetric K/V | ShardKV separates K/V processing; Still does the same |

---

## Key Reference Equations

### Cross-Attention Synthesis (adapted for modelless)
```
Z' = Z + CrossAttn(RMSNorm(Z), [K; V])
Z'' = Z' + SelfAttn(RMSNorm(Z'))
Ck = Z'' @ W_key
Cv = Z'' @ W_val
```
Where Z = heuristic latent queries (not learned), W_key/W_val = identity or PCA projection (not learned).

### Position-Free Compaction
```
K_free = un_rotate(K, positions)  // Strip RoPE
X = [K_free; V]                   // Position-free concatenation
Z_out = PerceiverBlocks(X, Z)     // Compact
Ck = re_rotate(Z_out @ W_key, new_positions)  // Restore RoPE
```

### Iterative Chunked Compaction
```
retained_cache = []
for chunk in chunks:
    prefill(chunk, conditioned_on=retained_cache + lookahead_raw)
    compact_chunk = compact(recent_kv_chunk)
    retained_cache.append(compact_chunk)
// Total: T/c + c*t entries (linear at rate 1/c)
```

---

## TL;DR

Still's amortized Perceiver KV compaction distills into three modelless fusion ideas:
1. **StillKV** (GATE) — Heuristic Perceiver compaction replacing learned latents with TF-IDF/attention/BFCF cluster centroids
2. **StillCoT** (GAIN) — Synthesis-based CoT trace compaction, evolution of ThoughtFold from selection to synthesis
3. **StillRSM** (DEFER) — Perceiver-augmented routing slot memory

The synthesis-over-selection insight is the key takeaway. Our existing stack already has all the building blocks (VortexFlow, BFCF, DashAttention, MUX-Latent) — the fusion is connecting them through the Perceiver cross-attention pattern. Position-free compaction and iterative chunked processing are pure engineering wins.
