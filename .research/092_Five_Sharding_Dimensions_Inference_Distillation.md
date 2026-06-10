# Research 92: Five Sharding Dimensions — Inference Distillation

> **Source:** [The Five Things You Can Shard](https://sakshamconsul.substack.com/p/the-five-things-you-can-shard) — Saksham Consul (Noise2Signal), 2025-05
> **Date:** 2026-07, distilled 2026-07
> **Related Research:** 037 (REAP Model-Based/Modelless), 059 (MoE+SD Co-Design), 073 (LT2 Looped), 058 (GRAM), 070 (GDN2), 028 (HLA)
> **Related Plans:** — (verdict: no direct plan needed, conceptual alignment only)
> **Verdict: LOW DIRECT VALUE — Training-time distributed sharding (multi-node, multi-GPU) is orthogonal to our single-device inference stack. However, three conceptual distillations are useful: (1) the bandwidth hierarchy lens for our model-based/modelless cost model, (2) expert parallelism's all-to-all pattern maps to our domain router dispatch, (3) the composition recipe (TP×PP×DP nesting) is a blueprint for our LT2 hybrid dispatch layer nesting. No feature gate needed — these are mental models, not code.**

---

## TL;DR

The article organizes every distributed training parallelism paradigm into **five dimensions** you can split work along:

| # | Axis | Splits | Communication | Fabric Tier |
|---|------|--------|---------------|-------------|
| 1 | **Data Parallelism (DP)** | Batch | AllReduce (gradients) | Slowest (inter-rack IB) |
| 2 | **Tensor Parallelism (TP)** | Matmul weights | AllReduce (per layer) | Fastest (NVLink) |
| 3 | **Pipeline Parallelism (PP)** | Layers | Point-to-point (send/recv) | Medium (inter-node IB) |
| 4 | **Context Parallelism (CP)** | Sequence (attention) | Ring rotation (K/V shards) | Medium-fast |
| 5 | **Expert Parallelism (EP)** | MoE experts | All-to-all (dispatch + combine) | Medium-fast |

Plus two modifiers (not separate axes):
- **SP** = TP companion (shard elementwise activations between TP regions)
- **FSDP/ZeRO** = DP refinement (shard optimizer state/grads/params along DP axis)

The key insight: **each axis maps to a tier of the bandwidth hierarchy**. TP lives on NVLink (chatty, per-layer sync). PP tolerates IB (large infrequent messages). DP tolerates the slowest fabric (one allreduce per step). Composition = nesting axes at the right level of zoom.

**Why this matters for us:** It doesn't directly — we're single-device inference. But the *mental model* of "what gets split, what communication it forces, what tier it lives on" applies to our model-based/modelless dispatch at a micro scale.

---

## 1. The Bandwidth Hierarchy Lens — Applied to Inference

The article's core thesis: **each parallelism is a bet about where you can afford to communicate**. This maps to our model-based/modelless spectrum:

### 1.1 Our "Bandwidth Hierarchy" (Inference-Time)

| Tier | Component | "Bandwidth" | Cost |
|------|-----------|-------------|------|
| L0 (fastest) | `ConstraintPruner` | Zero-cost lookup | O(1) table check |
| L1 | `NoScreeningPruner` / `FlowPruner` | Zero-inference | Pure formula |
| L2 | `BanditPruner<P>` | O(1) Q-value lookup | Amortized free |
| L3 | `DeltaBanditPruner` | Single log-probe | One inference call |
| L4 (slowest) | Full model verification | Full forward pass | O(L×D²) per token |

The article says: "DP tolerates the slowest fabric, TP requires the fastest." Our analogue: **modelless pruners tolerate the "slowest information" (no model output), model-based pruners require the "fastest fabric" (full inference pass)**.

### 1.2 The Composition Recipe Analogy

Frontier training uses: `DP × PP × TP` (outer to inner, slow to fast fabric).

Our dispatch composes: `Domain Router → ScreeningPruner → BanditPruner → ConstraintPruner → Verification`

| Layer | Analogy | Why |
|-------|---------|-----|
| Domain Router | DP (outer) | Batch-level, cheap, once per request |
| ScreeningPruner | PP (stage) | Graded relevance, per-depth decision |
| BanditPruner | TP (inner) | Per-token, O(1), hot path |
| ConstraintPruner | SP (companion) | Free with TP, elementwise check |
| Verification | — | The "ground truth" all paths converge to |

This is conceptual only — but it explains *why* our layering works: each tier operates at the right granularity with the right cost budget.

---

## 2. Expert Parallelism (EP) → Domain Router Dispatch

### 2.1 The EP Pattern

EP shards MoE experts across ranks. Communication: all-to-all, twice per MoE layer (dispatch tokens → expert ranks, combine expert outputs → originating ranks).

Key properties:
- **Load imbalance is the enemy**: if router sends 30% of tokens to one expert, that rank receives 30% of all traffic
- **All-to-all is the densest pattern**: every rank talks to every other rank
- **Temporal correlation helps**: adjacent tokens route to overlapping experts (38% overlap at step 1 vs 6.3% uniform baseline — Research 059)

### 2.2 Our Mapping

We don't have multi-rank EP, but our domain router + LoRA dispatch is the single-device analogue:

```
Frontier EP:  token → router → all-to-all → expert rank → compute → all-to-all → combine
Our dispatch: token → domain router → LoRA selection → forward pass → screening → accept/reject
```

| EP Concept | Our Analogue |
|------------|-------------|
| Expert rank | Domain LoRA adapter |
| Router top-k | Prompt router confidence threshold |
| All-to-all dispatch | Domain inference budget TOML |
| Load imbalance | Bandit exploration/exploitation |
| Temporal correlation | `RoutingOverlapSnapshot` (Plan 096) |

**Insight from EP:** The article notes EP all-to-all can be the single largest component of step time. Our analogue: domain LoRA switching cost. The `domain_latent` feature already caches domain context (Plan 038). The EP lesson is: **cache aggressively, batch domain switches when possible**.

### 2.3 Expert Choice Routing

The article mentions "expert choice" routing (tokens don't choose experts; experts choose tokens). This is the inverse of our current top-k routing.

**Potential application:** Instead of tokens choosing domains (prompt router), domains could "claim" tokens based on their specialty. This is essentially what `ScreeningPruner` does — it grades which domain a token fits. The "expert choice" framing suggests we could invert: have domains bid for tokens rather than tokens request domains.

**Verdict:** Interesting framing, but our current approach (domain router + screening) already captures this. No code change needed.

---

## 3. Context Parallelism (CP) → Our Long-Context Strategy

### 3.1 The CP Pattern

CP shards the sequence dimension across the attention computation itself. Each rank holds 1/N of Q, K, V. K/V chunks rotate in a ring while attention is computed in pieces using online-softmax.

Key constraint: **CP only pays for itself past ~16K-32K sequence** — below that, rotation communication exceeds memory savings.

### 3.2 Our Long-Context Stack

We already have multiple long-context strategies, each occupying a different "tier":

| Strategy | Effective Context | "Fabric" | Cost |
|----------|-------------------|----------|------|
| TurboQuant / SpectralQuant / OCTOPUS | Compressed KV cache | Memory-bound | Quantization error |
| DashAttention (Plan 106) | Sparse hierarchical | Compute-bound | Retrieval head selection |
| GDN2 recurrent state | Unbounded (O(1) memory) | State carry | Gate decay |
| LT2 looped AHLA | O(T·w) per layer | Loop count T | Re-loop cost |
| HLA / AHLA | Unbounded (O(1) memory) | Linear attention | Rank-1 state |

**The CP lesson:** Ring attention is essentially "stream K/V through compute." Our GDN2 and HLA do the same thing without the ring — they compress the K/V stream into a fixed-size state. We've already solved the CP problem by eliminating the need to shard the sequence at all.

**Verdict:** Our recurrent attention architectures (GDN2, HLA, LT2) are strictly superior to CP for single-device inference. CP exists because distributed training can't afford O(N²) attention on any single device. We avoid O(N²) entirely via compression and recurrence.

---

## 4. Pipeline Parallelism (PP) → LT2 Looped Pipeline

### 4.1 The PP Pattern

PP splits layers across stages. Forward activations flow stage 0 → 1 → 2 → ... Key challenge: pipeline bubble (idle time) and stashed activations.

Scheduling variants: GPipe (all forward, all backward), 1F1B (interleave forward/backward), interleaved 1F1B (multiple stages per rank for smaller bubbles), zero-bubble PP (weight-only recomputation).

### 4.2 Our LT2 Loop = Micro-PP

LT2 (Plan 108) loops T=4 times through the same weights. This is PP where every "stage" shares the same weights — eliminating the parameter cost but introducing the same scheduling question:

| PP Concept | LT2 Analogue |
|-----------|--------------|
| Pipeline stage | Loop iteration |
| Stage boundary | SDPA↔GDN2 transition |
| Pipeline bubble | Loop overhead (negligible: same device) |
| Activation stash | Recurrent state carry |
| Interleaved PP | Hybrid 1:4 SDPA:GDN2 ratio |
| Zero-bubble | SDPA output gate (eliminate attention sink) |

**The PP lesson:** PP's bubble fraction is `(N-1)/(M+N-1)` for N stages, M microbatches. With N=4 loops and M=1 (single token decode), bubble = 75%. But our "bubble" is just the loop overhead — no inter-device latency — so it's negligible. The interleaving insight (1:4 ratio) is already in our LT2 hybrid recipe.

**Verdict:** Already implemented in Plan 108. PP's scheduling tricks don't apply (no multi-device latency to hide).

---

## 5. Tensor Parallelism (TP) → Our SIMD/TileRT Dispatch

### 5.1 The TP Pattern

TP splits the matmul itself — weight matrices sharded column-wise or row-wise across ranks. Communication: AllReduce inside each layer (320 hard sync points per step for ~80 layers).

**Key constraint:** TP must stay on NVLink. Push it across IB and MFU collapses.

### 5.2 Our Single-Device "TP"

We don't shard across GPUs, but we do split computation across SIMD lanes and tile blocks:

| TP Concept | Our Analogue |
|-----------|--------------|
| Weight column shard | TileRT tile (Plan 102) |
| AllReduce sync | SIMD reduction |
| SP companion | Activation tiling in `decode_specialize` |
| NVLink requirement | L1/L2 cache locality |

**The TP lesson:** "320 hard sync points per step" explains why we see diminishing returns from fine-grained tiling. Each tile boundary is a synchronization point. Our TileRT pipeline (Plan 102) already manages this by:

1. Persisting tiles in cache (`contiguous_weights`)
2. Specializing decode stages (`decode_specialize`)
3. Overlapping compute with dequantization

**Verdict:** Already implemented. TP's lesson (minimize sync points, keep data local) is what TileRT already does.

---

## 6. Data Parallelism (DP) → Our Bandit Episode Dispatch

### 6.1 The DP Pattern

DP splits the batch. Each rank processes different examples through the same weights. Communication: one AllReduce per step on full gradients.

**Key insight:** DP is the outermost layer because it tolerates the slowest fabric.

### 6.2 Our "DP" = Bandit Episode Batching

When we run bandit episodes (e.g., G-Zero self-play, arena benchmarks), we process multiple games in parallel via rayon:

```
Frontier DP:  batch shard → same model → gradient → AllReduce → update
Our dispatch: game shard → same forward model → reward → bandit update → Q-value
```

The DP lesson: **batch-level parallelism is the cheapest because it requires no intra-step synchronization**. Our rayon parallelism over games already exploits this. The `rayon::prelude::*` in our transformer forward pass is the DP analogue.

**Verdict:** Already implemented. No change needed.

---

## 7. The Composition Recipe — Our "5D" Stack

The article's frontier recipe: `DP × PP × TP × CP × EP` (nested outer→inner).

Our inference "5D" composition:

```
Domain Router (DP-analogue: batch-level, cheap)
  └→ LT2 Hybrid Dispatch (PP-analogue: loop iterations)
       └→ TileRT Tile Execution (TP-analogue: SIMD/tiling)
            └→ GDN2/AHLA State Carry (CP-analogue: compress sequence)
                 └→ Domain LoRA Selection (EP-analogue: expert routing)
                      └→ ScreeningPruner → BanditPruner → ConstraintPruner
                           └→ Verification (ground truth)
```

Each level operates at the right granularity with bounded communication cost. This isn't code — it's the mental model that explains why the architecture works.

---

## 8. What We Could Borrow (Actionable)

| Idea | Source Axis | Our Application | Priority | Effort |
|------|------------|-----------------|----------|--------|
| Bandwidth-tiered cost model | All | Formalize L0-L4 pruner cost hierarchy in `SpecCostModel` | Low | Small |
| Expert choice routing (inverted) | EP | Domains bid for tokens vs tokens request domains | Low | Medium |
| Hierarchical sharding policy (HSDP) | DP/FSDP | Domain router: FSDP within domain, DDP across domains | None | — |
| Interleaved stage assignment | PP | Already in LT2 1:4 hybrid ratio | Done | — |
| Online-softmax ring attention | CP | Already superseded by GDN2/HLA recurrence | Done | — |

---

## 9. Verdict Summary

| Dimension | Direct Applicability | Reason |
|-----------|---------------------|--------|
| DP | ✅ Already have | Rayon batch parallelism |
| TP | ✅ Already have | TileRT SIMD tiling |
| PP | ✅ Already have | LT2 looped pipeline |
| CP | ✅ Superseded | GDN2/HLA recurrence > ring attention for single-device |
| EP | ⚠️ Conceptual | Domain router analogue, no multi-rank needed |

**Bottom line:** The article is about distributed training scale-out. We're about single-device inference efficiency. The five dimensions are the *wrong abstraction level* for our code, but the *right mental model* for understanding why our layered architecture works. Each layer in our stack maps to a sharding dimension, and our constraint (single device) means we've already found the optimal solutions for each.

**No feature gate needed. No plan needed.** This is a conceptual distillation that validates existing architectural decisions.

---

## 10. Key Takeaways for GOAT Proof Context

If we were to prove something from this research, it would be:

> **Claim:** Our inference stack's layered dispatch (domain router → screening → bandit → constraint → verify) naturally corresponds to the bandwidth hierarchy of distributed training. Each layer operates at the right cost tier, and the composition is optimal for single-device inference because we've replaced inter-device communication (the bottleneck in all five sharding dimensions) with either (a) compressed recurrent state, (b) SIMD tile locality, or (c) zero-cost pruner lookups.

This is an architectural alignment proof, not a benchmark proof. It belongs in the "why the architecture works" narrative, not in a GOAT benchmark.