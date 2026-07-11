# Research 378: HOLA — Hippocampal Exact KV Cache for Linear Attention

> **Source:** [A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets](https://arxiv.org/abs/2607.02303) — Wanyun Cui, Shanghai University of Finance and Economics, 2 Jul 2026
> **Date:** 2026-07-05
> **Status:** Active
> **Classification:** Public
> **Related Research:** 070 (GDN2 backbone), 233 (Attention Matching KV Compaction), 243 (Temporal Derivative surprise signal), 249 (DecentMem DualPool — semiparametric split), 213 (StillKV β-bias), 109 (Shard Drop), 252 (Unified Surprise Bus), 242/276 (recurrent belief kernel family)
> **Related Plans:** 105 (GDN2 — default-on backbone), 277 (temporal_deriv — default-on surprise bus), 271 (Attention Matching KV Compaction), 378 (this note's plan)

---

## TL;DR

HOLA gives the GDN (Gated DeltaNet) recurrent state a **hippocampal complement**: a bounded exact KV cache that stores the tokens with the largest **delta-rule write magnitude `β·‖e‖`** (the intrinsic surprise signal the delta rule already computes), read by a **decoupled RMSNorm-γ** sharpened softmax path that recovers near-argmax retrieval instead of soft averaging. At 340M / 15B SlimPajama tokens it cuts Wikitext perplexity 27.32 → 22.92 (−16.1%, below full-attention Transformer++ at 26.88) and stays length-robust on RULER S-NIAH-1 out to 32k (16× training length: 0.58 vs GDN 0.14).

**Distilled for katgpt-rs (modelless, inference-time):** a **surprise-evicted bounded KV cache** primitive — the cache is a `top-w` heap keyed by an intrinsic write-magnitude score `β·‖e‖` that GDN2 already computes per token, paired with a **decoupled cache-read RMSNorm-γ** that turns the exact copies into sharp near-argmax retrieval rather than a soft average. The eviction policy is parameter-free (no learned scorer — HOLA's headline contribution vs LTE which uses a CNN). The sharpened read needs a small learnable γ (≈ 0.004% of params) but is a model parameter, not a training recipe → still modelless at inference. This is a **per-stack KV-compression-slot primitive** that fuses cleanly with the shipped GDN2 backbone (Plan 105, default-on), the shipped `temporal_deriv` surprise bus (Plan 277, default-on), and the shipped `DualPoolBandit` semiparametric split (Research 249).

**Latent-space reframing (the part that decides GOAT vs Super-GOAT):** the HOLA "neocortex + hippocampus" pattern is the *exact* pattern the codebase already ships across three repos — `temporal_deriv` + δ-Mem write gate (Plan 277/053), `DualPoolBandit` E/X pools (Research 249), Raven consolidation + AnyRAG escalation + `temp_loss_fingerprint` (riir-neuron-db). What HOLA adds is **the specific instantiation on the GDN linear-attention backbone** with β·‖e‖ as the eviction score and a decoupled RMSNorm-γ read. That is a GOAT — a novel primitive with a provable gain on a shipped backbone — but **not a Super-GOAT**, because the capability class ("hippocampal complement to a compressive recurrent state") already ships, and the unified surprise bus (Plan 277) already drives the parametric/non-parametric split elsewhere.

---

## 1. Paper Core Findings

### 1.1 The framework: semiparametric test-time memory regression (§3.2)

The paper formalizes memory readout as **test-time regression**:

```
M_t = Write(M_{t-1}, k_t, v_t)
o_t = Read(q_t, M_t) = f̂_t(q_t)
```

Three instances of this framework:
- **Pure parametric (GDN):** `M_t = S_t` (fixed-size matrix), `Read = q^T S_t`. Cheap, captures the linearly-compressible part, but lossy once context exceeds capacity.
- **Pure non-parametric (full attention):** `M_t = D_t` (every causal KV pair), `Read = softmax kernel`. Lossless but O(T) memory / O(T²) compute.
- **Bounded semiparametric (HOLA):** `M_t = (S_t, A_t)` where `A_t ⊆ D_t` is a bounded exact KV set. `Read = q^T S_t + λ_t g_t(q_t)` — parametric estimate + non-parametric correction.

This locates HOLA on a spectrum between GDN (pure parametric) and full attention (pure non-parametric). The framework dictates two design choices: **what to store** (§1.2) and **how to read** (§1.3).

### 1.2 What to store — β·‖e‖ as intrinsic surprise (§3.3)

The delta rule writes `S_t = S_{t-1} + Δ_t` where `Δ_t = β_t · k_t · e_t^T` (rank-1, with ‖k_t‖=1). The Frobenius norm of the write is:

```
m_t = ‖Δ_t‖_F = β_t · ‖k_t‖ · ‖e_t‖ = β_t · ‖e_t‖
```

`m_t` is **how much token t changed the state**. Interpretation:
- Large `‖e_t‖` = the state couldn't predict `v_t` → token brought new information (Kalman innovation, LMS error).
- Large `β_t` = the model actually committed the correction to S.
- Their product = the surprising-and-committed tokens.

**Cache policy:** keep the top-w tokens by `m_t = β·‖e‖`, **regardless of distance** (not recency). The score is fixed at write time, so the same top-w set can be maintained online or blockwise without order dependence — training and inference use the same cache semantics.

**Ablation evidence (Table 4, 46M):** the *product* matters — `‖e‖` alone lacks the write-strength utility signal and underperforms on far-needle; `β·‖v‖` lacks the residual innovation signal; `β·‖e‖` is best or tied-best on every column. Importance eviction also beats recency eviction at 340M (matched control HOLA+recency): S-NIAH-1 at 32k is 0.58 (importance) vs 0.24 (recency) vs 0.14 (no cache).

### 1.3 How to read — decoupled RMSNorm-γ (§3.4)

If the cache reuses GDN's unit-L2-normalized q/k, the effective logit is `τ · (1/√d) cos ≈ 0.83 cos` → softmax is nearly uniform → cache degenerates into a soft average.

**Fix:** apply Qwen3-style RMSNorm-γ to the cache-path q/k only (decoupled from the state-update path):

```
cache_o_t = Σ_{j ∈ V_t} softmax_j(q̃_t^T k̃_j / √d) v_j
where q̃ = RMSNorm_γ(q), k̃ = RMSNorm_γ(k)
```

`‖q̃‖ ≈ ‖k̃‖ ≈ √d ≈ 11` (vs 1 for unit-L2), so the effective logit is `≈ √d · cos ≈ 11 cos` → near-argmax retrieval. **Decoupling is essential**: the delta rule relies on `‖k‖=1` to keep the update operator `I − β k k^T` within eigenvalues `[0,1]` (stable); a `√d` norm in the state update would give eigenvalues `1 − β·d` and diverge.

**Sharpening is the single largest lever** (Table 5, 46M): unit-L2 read gives Wiki PPL 70.10 (barely better than no cache 70.21); RMSNorm-γ read gives 59.5 (−10.6 points). Multi-key capacity at 16 keys goes from 0.31 → 0.41. This is the design choice that turns a bounded exact copy from a soft average into actual retrieval.

### 1.4 Instantiation on GDN (§3.5) and overhead

Backbone: GDN (Yang et al. 2024a) with data-dependent decay gate `α_t ∈ (0,1]`. The residual becomes `e_t = v_t − α_t · k^T S_{t-1}`; β·‖e‖ is unchanged; the gate is orthogonal to the method.

Overhead: 12,480 trainable scalars (cache-path Q/K RMSNorm scales + per-head sink + cache gate) = **<0.004% of 340M**. Cache is inference state (bf16): ~31 MB at L=24, w=64, C=256, head_dim=256. Peak GPU alloc 0.75 GB vs 0.72 GB for GDN at 32k/128k context — **~5% peak-memory overhead, flat with context length**.

### 1.5 Headline results (340M / SlimPajama 15B / ctx 2048)

| Metric | GDN (anchor) | HOLA | T++ (full attn) | KDA (SOTA linear) |
|---|---|---|---|---|
| Wikitext PPL ↓ | 27.32 | **22.92** | 26.88 | 26.18 |
| LAMBADA PPL ↓ | 30.95 | **30.26** | 42.15 | 31.37 |
| FDA retrieval ↑ | 11.7 | **20.1** | 46.1 | 13.9 |
| SWDE retrieval ↑ | 29.0 | **35.9** | 25.9 | 34.1 |
| RULER S-NIAH-1 @ 32k ↑ | 0.14 | **0.58** | 0 (RoPE extrap) | — |
| RULER S-NIAH-1 @ 16k ↑ | ~0.10 | **~0.74** | 0 | — |

Gain is consistent across scales (46M/170M/340M: −15–16% Wikitext PPL vs same-backbone GDN).

---

## 2. Distillation

### 2.1 What is transferable (modelless, inference-time)

**T1 — Bounded exact KV cache with β·‖e‖ eviction (the open primitive).** A per-layer fixed-capacity `w` heap of `(k, v, score)` triples where `score = β · ‖e‖`. The score is computed *for free* by any delta-rule update (GDN/GDN2) — the residual `e` and the write gate `β` are already on the hot path. Maintain the top-w via a binary heap (insert O(log w), evict-min O(log w)); the score is set at write time and never updated. This is **parameter-free** — HOLA's headline contribution over LTE (which learns a CNN eviction scorer).

**T2 — Decoupled RMSNorm-γ cache-read path.** The cache read applies a *separate* RMSNorm-γ to q/k (keeping norm at √d), then a sharpened softmax over `V_t = cache ∪ current_block ∪ sink`. The state-update path keeps unit-L2 q/k for delta-rule stability. The decoupling is the design point — applying the same norm to both breaks the delta rule's stability contract. The γ vector is a learnable model parameter (not a runtime learned value) → still modelless at inference (it ships with the model weights, like any other RMSNorm γ).

**T3 — Cache composition `V_t = top-w cache ∪ current block (≤C) ∪ null sink`.** The persistent component is the top-w; the block and sink are bounded bookkeeping for causal chunked processing and stability. Total visible cache size is `w + C + 1 ≈ 321` per layer at the paper's defaults.

### 2.2 What is NOT transferable

- **Training the model end-to-end on SlimPajama** → riir-train. (Out of scope per workflow.)
- **The GDN backbone from scratch** → we already ship GDN2 (Plan 105, default-on, GOAT 14/14) which is the *stronger* GDN variant (channel-wise erase/write gates). The HOLA primitive is backbone-agnostic over the delta-rule family; it ports directly to GDN2. Research 070 §Verdict explicitly identifies GDN2 + SWA hybrid as Phase 4 future work — HOLA's bounded exact cache is an alternative complement to SWA, pluggable into the same backbone.

### 2.3 Prior-art check — the part that decides GOAT vs Super-GOAT

Per the skill's mandatory two-layer check (notes + shipped code, with vocabulary translation).

**Vocabulary translation (paper → codebase):**
- "hippocampal memory" / "exact KV cache" / "episodic memory" → δ-Mem segment, DualPoolBandit X-pool, AnyRAG escalation set, KarcShard
- "delta rule" / "linear attention" → GDN2 (Plan 105), delta_mem (Plan 053)
- "surprise" / "write magnitude β·‖e‖" / "innovation residual" → `temporal_deriv` surprise signal (Plan 277), `surprise_norm()`, δ-Mem write gate, `temp_loss_fingerprint` (Plan 005)
- "semiparametric" / "parametric state + non-parametric KV" → `DualPoolBandit` E/X pools (Research 249), Raven consolidation (parametric) + AnyRAG (non-parametric)
- "sharpened read" / "decoupled RMSNorm-γ" → `rmsnorm_with_gamma` (in `katgpt-core/src/types.rs` / `katgpt-types/src/math.rs`)

**Notes layer (grep results — both vocabularies):**

| Paper concept | Closest shipped note | Coverage |
|---|---|---|
| GDN backbone | Research 070 (GDN2 — Decoupled Erase/Write) | ✅ Backbone fully distilled; Phase 1-2 ships as Plan 105. |
| Surprise signal β·‖e‖ | Research 243 (Temporal Derivative Kernel) + Research 252 (Unified Surprise Bus) | ⚠️ Concept ships as the **dual-fast/slow EMA derivative** driving the δ-Mem write gate (Plan 277 default-on). Different mechanism (EMA derivative vs instantaneous Frobenius norm), same *idea* (surprise-driven memory write). |
| Semiparametric state + bounded KV | Research 249 (DecentMem DualPool) + Research 213 (StillKV) + Research 233 (Attention Matching) | ⚠️ DecentMem ships `ReachableDualPoolRouter` — the parametric(E) + non-parametric(X) split — for **curiosity direction pools**, not for KV cache. AM (Plan 271) is the closest KV-compaction cousin but uses mass-preserving OMP, not surprise eviction. |
| KV cache eviction policy | Research 109 (Shard Drop), 213 (StillKV), 233 (AM), 258 (Sink-Aware) | ⚠️ Multiple KV-compaction policies ship. **None driven by delta-rule write magnitude.** See code-layer table below. |
| Sharpened cache read | (none — closest is `rmsnorm_with_gamma` primitive in `katgpt-types`) | ✅ Primitive exists; not wired to a *decoupled* cache-read path. |

**Code layer (mandatory — the layer that catches overclaims):**

Grep across `katgpt-rs/**/*.rs` for `evict|top_w|write_magnitude|β.*∥.*e∥|delta_rule_write|beta.*e_norm`:

| Match | Location | Is this HOLA's mechanism? |
|---|---|---|
| FIFO ring-buffer eviction (motif/conformal) | `katgpt-core/src/closure/motif.rs`, `conformal/ring.rs` | NO — pure recency/FIFO, not surprise. |
| E-pool priority eviction on consolidation | `katgpt-core/src/cgsp/dual_pool.rs` `consolidate_growing` | PARTIAL — evicts lowest-priority E-pool arm on overflow, but the pool is curiosity *directions*, not KV pairs, and the priority is bandit reward, not β·‖e‖. |
| Tau-calibrator window eviction | `katgpt-attn/src/chiaroscuro/tau.rs` | NO — entropy threshold window, not delta-rule write magnitude. |
| Zone bank FIFO cap | `benches/alien_sampler_goat.rs` | NO — recency cap, not surprise. |
| Stale-cache invalidation | `benches/bench_002_density_routing_goat.rs` | NO — regime-transition invalidation, not eviction-by-score. |

**Grep for `top_w|write_magnitude|delta_rule_write|β.*e_norm|frobenius.*residual` → NO MATCHES in any `.rs`.** No shipped KV-cache eviction policy uses the delta-rule write Frobenius norm as its score.

**Grep for `rmsnorm_with_gamma` → SHIPS** in `katgpt-types/src/math.rs` (and re-exported via `katgpt-core`). Not currently wired to a decoupled cache-read path.

**Conclusion:** The HOLA *mechanism* — surprise-evicted bounded KV cache with decoupled RMSNorm-γ read on the GDN/GDN2 backbone — is **novel as shipped code**. The HOLA *latent-space reframing* — hippocampal complement to a compressive recurrent state — **already ships across three repos**:
- katgpt-rs: `temporal_deriv` + δ-Mem write gate = surprise-gated parametric memory (Plan 277, default-on).
- katgpt-rs: `DualPoolBandit` = parametric E-pool + non-parametric X-pool (Research 249, shipped in `cgsp/dual_pool.rs`).
- riir-neuron-db: Raven consolidation (parametric neocortex) + AnyRAG escalation (non-parametric hippocampus); `temp_loss_fingerprint` (Plan 005, default-on) = surprise-diverse wake-event selection before averaging into style_weights.

### 2.4 Fusion (the GOAT-tier combination)

| Fusion | Existing system | What HOLA adds | Gate |
|---|---|---|---|
| **F1: GDN2 + HOLA cache** | GDN2 recurrent decode (Plan 105, default-on backbone) | A bounded exact KV cache that recovers long-range exact recall the fixed-size GDN2 state loses. The β·‖e‖ score is *free* from the existing delta-rule update — no extra compute on the hot path beyond heap maintenance. | On a synthetic multi-key associative recall suite at d_model=512, does HOLA+GDN2 recover ≥80% of needles at 8× training length where bare GDN2 collapses to ≤20%? (Paper: HOLA 0.98 vs GDN 0.83 at S-NIAH-1 8k.) |
| **F2: HOLA × temporal_deriv unified surprise bus** | temporal_deriv (Plan 277, default-on surprise signal driving δ-Mem write gate, collapse detector, curiosity) | HOLA's β·‖e‖ is a *second* surprise channel — instantaneous Frobenius-norm surprise vs the dual-fast/slow EMA derivative. Fusing: HOLA cache eviction uses β·‖e‖ (instantaneous, per-token); δ-Mem write gate uses `temporal_deriv.surprise_norm()` (slow-integrated, per-event). Two complementary timescales — exactly the complementary-learning-systems pattern. | Does a 2-signal cache (instantaneous β·‖e‖ + slow temporal_deriv) beat either alone on long-context retrieval? |
| **F3: HOLA × DualPoolBandit** | DualPoolBandit E/X pool (Research 249, shipped) | The HOLA cache IS a dual-pool: parametric GDN2 state = E-pool (compressive); non-parametric top-w KV = X-pool (exact). The consolidation gate ("which X-pool items promote to E-pool") maps to "which exact KV pairs the GDN2 state should absorb at its next chunk update". | Does the DualPoolBandit consolidation rule (sigmoid reward threshold) subsume or improve HOLA's static top-w? |
| **F4: HOLA × Attention Matching (AM, Plan 271)** | AM mass-preserving KV compaction (Plan 271) | HOLA selects tokens to keep *online* by intrinsic surprise; AM compacts a *full* KV cache *offline* by mass-preserving OMP. The two compose: HOLA cache (online, w=64) ⊕ AM-compacted prefix (offline, t<\<T). AM is the cold-tier compaction; HOLA is the warm-tier exact set. | On a 32k context, does HOLA-online ∪ AM-compacted-prefix recover more needles than either alone at fixed total budget? |

The GOAT-tier claim is **F1 alone** — the open primitive + GDN2 backbone. F2–F4 are fusion-potential, validated by their respective gates before any further escalation.

---

## 3. Verdict

### Tiers

| Tier | Criteria | Routing |
|---|---|---|
| Super-GOAT | Novel mechanism + new capability class + selling point + force multiplier | Open primitive + private guide + plans |
| **GOAT (this)** | **Provable gain over existing approach, not a new class. Promote if it wins.** | **Plan + implement + benchmark (Plan 378).** |
| Gain | Incremental, useful but not headline | Plan only, behind flag |
| Pass | Not relevant, OR training-only | One-line note |

**Verdict: GOAT.**

**One-line reasoning:** The HOLA cache mechanism (β·‖e‖-evicted bounded KV cache with decoupled RMSNorm-γ read) is a **novel primitive on a shipped backbone (GDN2)** with a **provable gain** (paper: −16% Wikitext perplexity, robust at 16× training length, +0.44 needle recall vs no-cache at 32k), but it is **not a new capability class** — the latent-space reframing ("hippocampal complement to a compressive recurrent state") already ships across katgpt-rs (`temporal_deriv` + δ-Mem write gate + `DualPoolBandit`) and riir-neuron-db (Raven + AnyRAG + `temp_loss_fingerprint`).

### Novelty gate (Q1–Q4)

- **Q1 — No prior art?** YES at mechanism level (no shipped KV eviction policy uses delta-rule write Frobenius norm; no shipped decoupled RMSNorm-γ cache-read path). NO at latent-reframe level (the dual-pool / surprise-bus pattern ships in 3 repos). **Mixed → not novel enough for Super-GOAT.**
- **Q2 — New capability class?** NO. "Surprise-gated bounded exact memory complement to a compressive recurrent state" is the shipped unified-surprise-bus + DualPool pattern. HOLA is a new instantiation on the GDN backbone, not a new class.
- **Q3 — Product selling point?** WEAK. "Our linear-attention decode has bounded exact recall" is a perf/quality optimization on a shipped backbone. Not a moat until F1 validates on a real long-context game benchmark, and even then it's an engine optimization, not a product capability.
- **Q4 — Force multiplier?** MODERATE — connects GDN2 (shipped), temporal_deriv (shipped), DualPoolBandit (shipped), AM (shipped), δ-Mem (shipped). But the unified surprise bus already wires these (Plan 277). HOLA is a *new consumer* of the bus, not a new bus.

**Q2 + Q3 fail → GOAT, not Super-GOAT.** No riir-ai / riir-chain / riir-neuron-db guide created. Plan the open primitive + GOAT gate.

### MOAT gate per domain (§1.6)

| Domain | In scope? | MOAT contribution |
|---|---|---|
| `katgpt-rs` (public engine) | ✅ YES — paper-derived KV-compression-slot primitive for the transformer stack | **Promote/demote tracked per stack.** Lands in the **KV-compression slot** alongside AM (Plan 271), StillKV (Plan 245), Shard Drop (Research 109), Sink-Aware (Plan 287). HOLA competes for the same slot. GOAT gate decides promote-to-default vs demote-loser vs coexist-by-feature-flag. |
| `riir-ai` (private runtime) | NO — generic engine primitive, no game IP. | — |
| `riir-chain` (private chain) | NO — no commitment / sync / LatCal angle. | — |
| `riir-neuron-db` (private shards) | NO at mechanism level (the surprise-evicted KV cache is a transformer-stack primitive). YES at *latent-reframe* level (Raven + AnyRAG already implement the hippocampal-complement pattern for shards) — but that pattern is shipped, so this is a cross-reference, not new IP. | — |

**Per-stack promote/demote ledger (the engine's quality contract):**

| Slot | Current default | HOLA's claim | Gate |
|---|---|---|---|
| KV compression (long-context decode) | AM (Plan 271, GAIN, opt-in) + Sink-Aware (Plan 287, opt-in) + StillKV (Plan 245) | Surprise-evicted bounded exact cache on GDN2 backbone | G1 eviction correctness (top-w by β·‖e‖, deterministic, no order dependence) + G2 latency (heap maintain < 100 ns/token) + G3 no-regression on bare GDN2 (no cache = identical decode) + G4 retrieval gain on synthetic multi-key suite + G5 perplexity proxy on a real model (deferred to riir-train — needs trained GDN2 weights) |

G5 is the load-bearing gate and requires a trained GDN2 model — **out of scope for this workflow** (modelless). The modelless gates (G1–G4) ship in Plan 378; G5 is a tracked riir-train follow-up.

### §3.6 PoC requirement check

This verdict makes **no quality-parity claim** with the paper. Claims made:
- (a) **Architectural** — GDN2 ships (grep-proven), `rmsnorm_with_gamma` ships (grep-proven), `DualPoolBandit` ships (grep-proven), `temporal_deriv` ships (grep-proven). Each proven by reading the code.
- (b) **Latent-reframe coverage** — the hippocampal-complement pattern ships across 3 repos (grep + read proven above).
- (c) **Mechanism novelty** — no shipped KV eviction policy uses β·‖e‖ (grep-proven, zero hits).

No claim that "our runtime analog matches the paper's perplexity numbers". The paper's numbers come from a 340M model trained from scratch on 15B tokens — that is a riir-train job, not a modelless claim. **No PoC required.** The Plan 378 GOAT gate (G1–G4) is the modelless validation; G5 (perplexity on real weights) is explicitly deferred.

---

## 4. Implementation Sketch (delegates to Plan 378)

1. **`HippocampalCache<const D, const W>`** in `crates/katgpt-core/src/hippocampal_cache.rs` — generic, no model semantics. Feature flag `hippocampal_cache`.
   - Internal: fixed-capacity min-heap of `(score: u32 (or f32 bits), idx: u32)` + ring of `(k: [f32; D], v: [f32; D])` slots.
   - `observe(k, v, beta, residual_norm)` → push `(beta * residual_norm, slot)`, evict min if over W. O(log W).
   - `read_cache(q, gamma) -> [f32; D]` → RMSNorm-γ on (q, cached k's), sharpened softmax over `V = cache ∪ block ∪ sink`. O(W · D).
2. **Decoupled γ** — `gamma: [f32; D]` stored as a model parameter (not learned at runtime; initialized to ones, like any RMSNorm γ). The cache path uses its own γ, separate from any layer-norm γ on the state-update path.
3. **GDN2 integration** — `gdn2::State` exposes the per-token `β_t` (write gate) and `‖e_t‖` (residual norm) on every delta-rule update; pipe these to `HippocampalCache::observe`. Zero extra compute on the hot path (both are already computed).
4. **GOAT gate (G1–G4 modelless):**
   - G1 — eviction correctness: deterministic top-w by β·‖e‖, no order dependence (insert same multiset in different orders → identical cache set).
   - G2 — latency: heap maintain ≤ 100 ns/token at W=64, D=256; cache read ≤ 1 µs at W=64.
   - G3 — no-regression: with cache disabled (W=0), GDN2 decode is byte-identical to bare GDN2.
   - G4 — retrieval: synthetic multi-key associative recall (8 keys in a 4k stream, query one) — bare GDN2 < 30%, HOLA-cache ≥ 80%.
5. **G5 (riir-train follow-up):** perplexity + RULER needle on a trained GDN2 model with vs without HOLA cache. Tracked in `.issues/` — **not blocking** the modelless promotion.

### Latent vs raw boundary

The HOLA cache is **inference-local state** — never crosses the sync boundary. The cache contents (k, v tensors, scores) are per-layer decode scratch, not per-entity truth. No sync, no LatCal commitment, no quorum. The cache γ vector ships as a model parameter (committed via the model's BLAKE3 weight hash, same as any RMSNorm γ — no special handling).

If a future fusion routes HOLA cache contents to riir-neuron-db (e.g., persisting a "memorable tokens" set per NPC across sessions), that persistence goes through `MerkleFrozenEnvelope` as raw bytes (the cache is fixed-layout Pod-friendly: `[(k, v, score); W]`). That is a riir-neuron-db follow-up, out of scope here.

---

## 5. Open Questions / Risks

1. **w tuning.** Paper uses w=64 at L=24, head_dim=256. Our micro GDN2 configs (hd=4, n_embd=48) need w << 64 to avoid the cache dominating the state. Plan 378 sweeps w ∈ {8, 16, 32, 64} on micro; the GOAT gate uses w=16 as the modelless default.
2. **γ initialization and stability.** RMSNorm-γ starts at γ=1 (identity) — does the cache read stay stable without training γ? The paper trains γ end-to-end. Our modelless gate (G4 retrieval) uses γ=1 (fixed); if retrieval fails at γ=1, that flags a modelless-vs-trained gap → G5 follow-up. **This is the canonical "systematic bias" path from the modelless unblock protocol §3.5**: if the bias is "cache logits too flat at γ=1", a deterministically-constructed γ (e.g., γ_i = √d / ‖k_i‖ heuristic, or a closed-form sharpening factor) may close the gap without gradient descent. Exhaust this before deferring γ-tuning to riir-train.
3. **Heap vs sorted-vec at W=64.** At small W a sorted `[f32; W]` with linear-scan eviction may beat a binary heap (cache locality, branch predictor). Plan 378 G2 benchmarks both.
4. **Block + sink bookkeeping.** Paper includes current chunk (≤C=256) + null sink in `V_t` for causal chunked processing. Our GDN2 already does chunkwise decode — the cache must compose cleanly with the existing chunk boundary. Plan 378 Phase 2.
5. **GDN2 vs GDN mismatch.** HOLA instantiates on GDN (Yang 2024a, scalar decay). Our shipped GDN2 (Plan 105) has *channel-wise* erase/write gates (Research 070). β·‖e‖ is unchanged (Frobenius norm of the rank-1 write), but the per-channel gate structure may interact with the cache read path. Plan 378 Phase 3 integration test.
6. **Coexistence with AM (Plan 271).** Both target the KV-compression slot. HOLA is online + intrinsic-surprise; AM is offline + mass-preserving. They compose (F4 fusion), but the GOAT gate must verify they don't fight for the same budget. Plan 378 Phase 4.

---

## TL;DR

**Verdict: GOAT.** HOLA's surprise-evicted bounded KV cache (top-w by β·‖e‖) with decoupled RMSNorm-γ read is a **novel primitive on the shipped GDN2 backbone** (Plan 105, default-on) with a strong paper-claimed gain (−16% perplexity, robust at 16× training length). It is **not a Super-GOAT**: the latent-space reframe (hippocampal complement to compressive recurrent state) already ships across katgpt-rs (`temporal_deriv` + δ-Mem write gate, Plan 277 default-on; `DualPoolBandit`, Research 249) and riir-neuron-db (Raven + AnyRAG + `temp_loss_fingerprint`, Plan 005). Lands in `katgpt-rs` as a KV-compression-slot primitive (Plan 378), feature-flagged `hippocampal_cache`, GOAT gate G1–G4 modelless + G5 deferred to riir-train. The β·‖e‖ score is **free** from the existing delta-rule update; the decoupled γ is a model parameter (not runtime-learned); the cache is inference-local state (no sync boundary). The strongest fusion is **F1 (GDN2 + HOLA cache)**; F2–F4 (temporal_deriv, DualPool, AM) are tracked fusion-potential.
