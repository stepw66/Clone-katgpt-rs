# Research 385: SoftMatcha 2 — Smooth-Min Soft Pattern Matching

> **Source:** SoftMatcha 2: A Fast and Soft Pattern Matcher for Trillion-Scale Corpora — Yoneda, Matsushita, Kamoda, Suenaga, Akiba, Waga, Yokoi (ICML 2026)
> **arXiv:** [2602.10908](https://arxiv.org/abs/2602.10908)
> **Date:** 2026-07-06
> **Status:** Done
> **Related Research:** 278 (Engram), 296 (DEC vocabulary crosswalk — smooth-min ↔ cochain), 012 (riir-neuron-db ItemEmbedIndex)
> **Related Plans:** 299 (Engram — Zipfian cache), 362 (riir-neuron-db ItemEmbedIndex)
> **Classification:** Public

---

## TL;DR

SoftMatcha 2 is a **systems paper** about sub-300ms soft (semantic) pattern matching over trillion-token corpora via suffix arrays + word vectors. The systems contribution (disk-aware staged suffix array, 5.6 TB index, 53-hour build) does **not** transfer to our 20Hz game runtime — we don't search trillion-scale text corpora per tick. The transferable primitive is the **smooth-min similarity function** with Zipfian-norm-based insertion/deletion penalties: a deterministic, modelless latent-space operation for fuzzy multi-token retrieval. It is a **Gain** — useful as a small utility that AnyRAG / ItemEmbedIndex / Engram could consume when they need multi-token soft matching, but not a new capability class and not provably better than plain cosine without a PoC.

**Distilled for katgpt-rs (modelless, inference-time):**
The smooth-min similarity `sim(q,p) = 1 - log_β(Σ_i(β^{1-c_i} - 1) + 1)` (where `c_i` = cosine similarity of the i-th token pair, `β` = sharpness) is a parametrized aggregation that interpolates between plain-min (`β→∞`, the SoftMatcha v1 behavior) and plain-sum (`β≈1`). Combined with a `exp(-v/γ)` scaling per insertion/deletion (where `v` = squared Zipfian-whitened norm of the edited token), it scores variable-length pattern similarity without decoding to a single vector. This is a zero-allocation, sigmoid-friendly (never softmax), deterministic function — a natural katgpt-core utility.

---

## 1. Paper Core Findings

### 1.1 The three contributions

| Contribution | What it is | Transferable to us? |
|---|---|---|
| **Dynamic corpus-aware pruning** (§3.3) | Iterative prefix-based candidate filtering: at each query-token position i, enumerate patterns similar to q₁..qᵢ, then filter to those that *occur in the corpus* via suffix-array lookup. Exploits Zipf/Heaps law to prove sublinear scaling O(\|C\|^{1/δ}) in corpus size (Theorem 1). | **Partially.** The Zipf-exploitation idea already ships in Engram's `ZipfianCacheHierarchy` (Plan 299 Phase 6). The prefix-pruning pattern is structurally similar to our `ConstraintPruner` / `ScreeningPruner` cascade. |
| **Disk-aware staged suffix array** (§3.4) | Two-tier index: sorted array X of all L-grams on disk + sparse array Y = [X₀, X_B, X_{2B}, …] in RAM. One random disk access per exact lookup (vs O(log\|C\|) for standard suffix array). Run-length compression reduces index 7.1×. | **No.** Designed for 5.6 TB indexes on NVMe SSD arrays. Our game corpora are bounded (quest text, dialog, item catalogs — MB-scale, RAM-resident). The CompressionDrafter benchmark (.benchmarks/287) noted this: suffix-array would fix its latency failure but is "a different algorithm entirely." |
| **Smooth-min similarity** (§3.5) | `sim(q,p) = 1 - log_β(Σ_i(β^{1-c_i} - 1) + 1)` for substitutions; `× exp(-v/γ)` per insertion/deletion where v = squared Zipfian-whitened norm. Interpolates plain-min (β→∞) ↔ plain-sum (β≈1). Norm-based penalties keep low-information words ("the", "of") cheap to edit. | **Yes — this is the transferable primitive.** Deterministic, modelless, latent-space. β=10⁴ empirically best. |

### 1.2 Theoretical results (§4)

- **Theorem 1 (sublinearity):** Under Zipf n-gram distribution with exponent δ>1, expected total exact lookups = O(|C|^{1/δ}). Empirically δ≈1.5 → O(|C|^{2/3}).
- **Theorem 2 (query-length bound):** Under exponential bound E[|Rᵢ|] ≤ a·rⁱ with r<1, total lookups = O(1) in query length m.
- **Empirical validation (Table 22):** |Rᵢ|/|Sᵢ| (corpus-occurring / all-similar) drops rapidly with i — at α=0.5, i=5: 0.003 without insert/delete, 0.001 with. The pruning is what makes soft search tractable.

### 1.3 Practical results (§5–6)

- p95 latency 278ms for soft search, 0.34ms for exact search on FineWeb-Edu (1.4T tokens).
- 33× faster than infini-gram on exact lookup; 475× faster than infini-gram mini at 273B tokens.
- Contamination detection: found 36 additional dirty benchmark samples (1.4%) that exact match missed; 81% precision on manual verification (29/36 true contamination).
- β ablation (Table 19): β=10⁴ consistently best; β→∞ (plain min) loses multi-mismatch discrimination; β≈1 (sum) lets unrelated patterns with one exact word rank high.

---

## 2. Distillation

### 2.1 The smooth-min similarity as a modelless latent-space primitive

The core transferable function, stripped of the suffix-array machinery:

```rust
/// Smooth-minimum similarity for variable-length soft pattern matching.
///
/// `cosines` = per-position cosine similarities c₁..cₘ (each in [-1, 1]).
/// `β` = sharpness (paper uses 1e4; β→∞ = plain min, β≈1 = plain sum).
/// Returns similarity in [0, 1].
fn smooth_min_similarity(cosines: &[f32], beta: f32) -> f32 {
    // 1 - log_β( Σ(β^(1-c_i) - 1) + 1 )
    // Numerically stable via logsumexp on the β^(1-c_i) terms.
    let sum = cosines.iter()
        .map(|&c| (1.0 - c).mul_add(beta.ln(), 0.0).exp() - 1.0)  // β^(1-c) - 1
        .sum::<f32>() + 1.0;
    1.0 - sum.ln() / beta.ln()
}

/// Insertion/deletion penalty using Zipfian-whitened norm.
/// `norm_sq` = squared norm of the edited token's embedding (post-Zipfian whitening).
/// `gamma` = penalty scale (paper: γ = m·γ' where γ' tuned so penalty = 1/e at m=5, 50th-lowest norm).
fn edit_penalty(norm_sq: f32, gamma: f32) -> f32 {
    (-norm_sq / gamma).exp()  // exp(-v/γ)
}
```

Properties:
- **Deterministic** — no training, no gradients. Pure arithmetic on pre-computed embeddings.
- **Sigmoid-friendly** — output is in [0,1], composes with sigmoid gates (never softmax per AGENTS.md).
- **Zero-allocation** — operates on `&[f32]` slices, no heap.
- **Parametrized** — β and γ are runtime-tunable (could be freeze/thaw-versioned direction-vector parameters).

### 2.2 Why the systems contribution doesn't transfer

The disk-aware staged suffix array is the paper's headline (33× faster exact lookup). But:
- Our game corpora are MB-scale (quest text, dialog trees, item catalogs), not TB-scale.
- MB-scale corpora fit in RAM — no disk-access optimization needed.
- The 53-hour index build is a non-starter for a game runtime.
- CompressionDrafter (.benchmarks/287) already considered suffix arrays and rejected them as "a different algorithm entirely" from its MatchLengthScorer.

The dynamic corpus-aware pruning is more interesting algorithmically, but Engram's `ZipfianCacheHierarchy` (Plan 299 Phase 6) already exploits Zipf distribution for cache tiering. The prefix-pruning pattern (filter candidates at each token position) is structurally similar to our ConstraintPruner/ScreeningPruner cascade — we already do staged candidate filtering.

### 2.3 Fusion (novel combination — not yet planned)

The 2–3 closest existing primitives and what fusing them produces:

| Fusion | Primitives combined | What it produces that none has alone |
|---|---|---|
| **Smooth-min ItemEmbedIndex** | This note (smooth-min) × Research 012 (ItemEmbedIndex) × Plan 362 | Multi-token item queries ("enchanted silver sword") ranked by smooth-min over per-token cosine instead of single-vector cosine. Currently ItemEmbedIndex does single 8-dim cosine; smooth-min would handle adjective+noun queries where each token has its own embedding. |
| **Soft Engram** | This note (smooth-min + edit penalty) × Research 278 (Engram) × Plan 299 | Engram is hash-addressed EXACT pattern memory. A "soft Engram" layer that falls back to smooth-min cosine retrieval when the exact hash misses would give NPCs fuzzy pattern recall. The ZipfianCacheHierarchy already exists; smooth-min would be the warm-tier scorer. |
| **AnyRAG retrieval engine** | This note (smooth-min) × riir-neuron-db gateway.rs (stub) | AnyRAG's `request_ruling` is currently a stub. Smooth-min similarity over an external corpus (when AnyRAG eventually gets a real backend) would provide semantically-aware escalation retrieval. |

The strongest fusion is **Smooth-min ItemEmbedIndex** — it's the most concrete, the consumer already ships (Plan 362, default-on), and the gain is testable (does smooth-min beat single-cosine on multi-token item queries?). But it's a Gain-tier extension, not a new capability class.

### 2.4 Latent-space reframing (mandatory per skill §1 step 3)

| Substrate | How smooth-min looks on it | Verdict |
|---|---|---|
| (a) HLA per-NPC latent state | Smooth-min over per-dimension HLA similarities (valence/arousal/…). But HLA already uses dot-product + sigmoid per AGENTS.md. Smooth-min would replace sigmoid aggregation — marginal, and violates the "sigmoid not softmax" rule's spirit (smooth-min is a different aggregation). | Weak |
| (b) latent_functor operations | Smooth-min as a functor on latent vectors. Stretch — functors are operator-valued, smooth-min is scalar-valued. | Weak |
| (c) cgsp_runtime curiosity | Not obviously relevant. | N/A |
| (d) LatCal fixed-point | Smooth-min similarity → LatCal scalar bridge. The similarity output [0,1] could be LatCal-committed, but this is just "commit a scalar," not novel. | Weak |
| (e) NeuronShard / AnyRAG / ItemEmbedIndex | **Strongest angle.** AnyRAG retrieval quality, ItemEmbedIndex multi-token queries, shard style-weight soft matching. | **Strong** |
| (f) DEC Stokes operators | Smooth-min as a cochain aggregation? The smooth-min is a nonlinear reduction; DEC operators are linear. Not a natural fit. | Weak |

The latent reframing confirms: this is a **retrieval primitive** (angle e), not a manifold/functor/HLA primitive. It belongs in the retrieval/matching family alongside Engram and ItemEmbedIndex.

---

## 3. Verdict

### Tier: **Gain**

**One-line reasoning:** The smooth-min similarity + Zipfian-norm edit penalty is a deterministic, modelless, latent-space utility that could incrementally improve multi-token retrieval in AnyRAG/ItemEmbedIndex/Engram, but the paper's core contribution (trillion-scale suffix-array systems) doesn't apply to our 20Hz runtime, the similarity function itself is a known aggregation (softmin/LogSumExp family), and there's no provable gain over our existing single-cosine retrieval without a PoC.

### Why not Super-GOAT

- **No prior art?** Smooth-min (softmin) is a well-known function. Zipfian-norm penalties are a reasonable heuristic but not novel. The combination is tasteful but not a new mechanism.
- **New class of behavior?** No — we already have cosine retrieval (ItemEmbedIndex), exact pattern matching (Engram), and external escalation (AnyRAG stub). Smooth-min is a better aggregation, not a new capability.
- **Product selling point?** Cannot finish "Our NPCs do X that no competitor can" — smooth-min retrieval is incremental.
- **Force multiplier ≥2 pillars?** Touches Pillar 1 (Egg/Shell — ItemEmbedIndex) and Pillar 6 (NPC Dialog — latent RAG), but the connection is weak (better retrieval scoring, not a new pillar-level mechanism).

### Why not GOAT

GOAT requires a **provable gain** over existing approach. We'd need a PoC comparing smooth-min vs plain cosine on ItemEmbedIndex multi-token queries to prove the gain. Without that PoC, this stays at Gain. A GOAT promotion is possible if:
1. A PoC shows smooth-min beats single-cosine on multi-token item retrieval (e.g., "enchanted silver sword" → correct sword vs wrong weapon type).
2. The latency overhead is < 100ns per query (smooth-min is O(m) where m = query length, should be sub-µs for m ≤ 10).

### Why not Pass

The smooth-min similarity IS relevant to modelless/latent retrieval — it's a deterministic latent-space operation usable at inference time. Pass would require it to be training-only or irrelevant to our stack. It's neither.

### MOAT gate (§1.6)

| Domain | In scope? | Verdict |
|---|---|---|
| katgpt-rs (public engine) | **Yes** — smooth-min is a generic modelless retrieval primitive. | Neutral Gain. Ship behind feature flag if planned; do not overclaim moat. |
| riir-neuron-db (shards) | Consumer (AnyRAG, ItemEmbedIndex), not primitive owner. | Neutral. |
| riir-ai (runtime) | Consumer (NPC dialog RAG), not primitive owner. | Neutral. |

The primitive itself is a small katgpt-core utility. No private guide needed (not Super-GOAT). No pillar-level moat contribution.

---

## 4. What would a plan look like? (not created — Gain tier)

If promoted to a plan in the future, the minimal scope would be:

**Plan: Smooth-Min Soft Similarity Utility (katgpt-core)**

- **Target:** `katgpt-rs/crates/katgpt-core/src/similarity.rs` (or extend an existing retrieval module)
- **Feature flag:** `smooth_min_similarity` (opt-in)
- **Phase 1:** `smooth_min_similarity(cosines: &[f32], beta: f32) -> f32` + `edit_penalty(norm_sq: f32, gamma: f32) -> f32` + `zipfian_whitened_norm(embedding: &[f32], freq_rank: usize) -> f32`
- **Phase 2:** GOAT gate — PoC comparing smooth-min vs plain cosine on synthetic multi-token retrieval task. Gate: smooth-min recall@5 > plain cosine recall@5 on queries with ≥2 token mismatches.
- **Phase 3:** Wire into ItemEmbedIndex as an optional multi-token query path.

**No plan created in this session** — Gain tier, and the GOAT gate (PoC) is the blocking step before committing to implementation.

---

## 5. What does NOT transfer (honest scope)

- **Disk-aware staged suffix array** — our corpora are RAM-resident (MB-scale). The 5.6 TB index, 53-hour build, NVMe/RAID0 tiering, and single-disk-access lookup are all solving a problem we don't have.
- **Trillion-scale search latency** — 278ms p95 is impressive at 1.4T tokens but irrelevant to our 20Hz (50ms budget) game tick.
- **Benchmark contamination detection** — interesting application but not our use case (we don't train LLMs; riir-train is out of scope for this workflow).
- **k-gram pruning / last-bits pruning** — specific to suffix-array lookup; our retrieval is hash-based (Engram) or cosine-based (ItemEmbedIndex), not suffix-array-based.

---

## 6. Cross-references

- **Research 278** (Engram) — exact hash-addressed pattern memory; ZipfianCacheHierarchy already exploits Zipf.
- **Research 012** (riir-neuron-db ItemEmbedIndex) — single-vector cosine item retrieval; the most natural consumer of smooth-min for multi-token queries.
- **Research 006** (riir-ai NPC Dialog Engine + Latent RAG) — NPC dialog retrieval; could consume smooth-min when matching multi-token dialog cues.
- **.benchmarks/287** (CompressionDrafter) — FAILED GOAT on latency; noted suffix-array would help but is "a different algorithm." Confirms suffix-array is not our path.
- **Plan 299** (Engram) — ZipfianCacheHierarchy Phase 6; the Zipf-exploitation pattern already ships.

---

## TL;DR

SoftMatcha 2 is a systems paper about fast soft pattern matching at trillion scale. The systems contribution (disk-aware suffix array) doesn't apply to our RAM-resident game corpora. The transferable primitive is the **smooth-min similarity function** with Zipfian-norm edit penalties — a deterministic, modelless, latent-space utility for fuzzy multi-token retrieval. **Verdict: Gain.** It could incrementally improve ItemEmbedIndex (multi-token queries), Engram (soft fallback when exact hash misses), and AnyRAG (when it gets a real retrieval engine). No plan created — a GOAT promotion would require a PoC proving smooth-min beats plain cosine on our retrieval tasks. No Super-GOAT: smooth-min is a known aggregation function, not a new capability class, and the paper's core value (trillion-scale search infrastructure) is orthogonal to our 20Hz game runtime.
