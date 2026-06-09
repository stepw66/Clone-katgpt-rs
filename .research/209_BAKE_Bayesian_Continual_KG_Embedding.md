# Research 209: BAKE — Bayesian-Guided Continual KG Embedding for Inference-Time Embedding Evolution

**Date:** 2026-06-09
**Source:** arXiv:2508.02426 — "Learning to Evolve: Bayesian-Guided Continual Knowledge Graph Embedding" (Li et al., WWW'26)
**Verdict:** GAIN — High priority for katgpt-rs (modelless)
**Target:** Modelless (katgpt-rs primary), Model-based (riir-ai secondary)
**Relates To:** Research 196 (KG Latent Octree), 192 (NextLat Belief States), 193 (BFCF Region × LFU), 199 (Memory Caching RNN)
**Feature Gate:** `bake_precision` (opt-in, GOAT gate before default)

---

## Executive Summary

BAKE's fundamental insight is NOT "Bayesian KG update" — it's **per-dimension precision as a forgetting-resistant update protocol**. Each dimension of any vector-valued state tracks its own evidence count (precision λ). High-precision dimensions resist change (anchors). Low-precision dimensions absorb new evidence eagerly (exploration). The update is pure arithmetic: O(d), zero-allocation, SIMD-friendly — no gradients, no replay buffer, no stored history.

**Our fusion opportunity:** Apply BAKE's precision-weighted posterior updates to our `KgEmbedding` system (currently has scalar `confidence: f32`), making KG embeddings evolve across inference sessions without forgetting. This is NOT replicating BAKE's training pipeline — it's distilling the inference-time artifact: **a [f32; 8] precision vector per embedding that gates how much new evidence can shift each dimension**.

---

## Paper Core Ideas

### 1. Sequential Bayesian Posterior-as-Prior
- Entity/relation embeddings modeled as Gaussian: `e_t ~ N(μ_{e,t}, λ_{e,t}^{-1})`
- Update rules (closed-form, O(d)):
  - `λ_{e,t} = λ_{e,t-1} + λ_obs` (precision grows monotonically)
  - `μ_{e,t} = (λ_{t-1}⊙μ_{t-1} + λ_obs⊙ê_t) / λ_t` (precision-weighted mean)
- New entities initialized with uninformative priors (random means, small precisions)
- Observation precision `λ_obs` controls plasticity/stability balance

### 2. Precision-Weighted KL Regularizer
- `L_Bayes = Σ_i β·√(λ_{i,t-1} ⊙ (θ̂_{i,t} - μ_{i,t-1})²)`
- High precision → strong anchor → resists change
- Low precision → weak anchor → allows exploration
- `√(λ·δ²)` form: precision-weighted Mahalanobis distance
- β controls regularization strength

### 3. Continual Clustering (Contrastive)
- Entities sorted by importance: `IE(e) = f_nc(e) + f_bc(e)` (centrality + betweenness)
- Fixed-size clusters, momentum-updated centroids: `c_{k,t} = (1-η)c_{k,t-1} + η·mean(emb)`
- Contrastive loss `L_FCC` with per-cluster fairness weights `α_k = 1/N_k`
- Proxy vectors `v_k` optimize cluster center positions
- Prevents semantic drift between snapshots

### 4. Results
- 8 datasets, all SOTA MRR/Hits@1/3/10
- Ablation: removing Bayesian = -7.05% MRR, removing clustering = -11.16% H@1
- FWT (forward transfer) best across all baselines
- 1.2× training time overhead (Bayesian updates are negligible)

---

## Distillation: Modelless vs Model-Based

### What's Training-Time Only (riir-ai territory)
- The KGE training loss `L_KGE` (TransE scoring)
- The clustering loss `L_FCC` (requires labels/centroids during training)
- Learning observation precision `λ_obs` as a hyperparameter
- Centroid initialization from graph structure analysis

### What's Inference-Time (katgpt-rs territory)
- The **precision vector** `[f32; 8]` per embedding — tracks certainty per dimension
- The **posterior update rules** (eq 2, 3) — pure arithmetic, SIMD-friendly
- The **precision-weighted regularization** — prevents embedding drift across sessions
- The **momentum centroid update** — maintains cluster consistency
- The **importance scoring** `IE(e)` — centrality + betweenness from KG structure

---

## Fusion Ideas: Creative, Not Direct Mapping

### Fusion 1: Precision-Gated KgEmbedding (Primary — katgpt-rs)

**Core Insight:** Replace `KgEmbedding.confidence: f32` with `precision: [f32; 8]`. This upgrades from a single scalar "how confident am I?" to per-dimension "which dimensions of my KG knowledge are well-established vs. still being explored?"

**Current State:**
```rust
pub struct KgEmbedding {
    pub entity_hash: u64,
    pub relation_hash: u64,
    pub embedding: [f32; 8],
    pub sign: bool,
    pub confidence: f32,  // ← scalar, no per-dimension certainty
}
```

**Proposed Extension:**
```rust
pub struct KgEmbedding {
    pub entity_hash: u64,
    pub relation_hash: u64,
    pub embedding: [f32; 8],
    pub sign: bool,
    pub confidence: f32,       // retained for backward compat
    #[cfg(feature = "bake_precision")]
    pub precision: [f32; 8],   // per-dimension certainty (λ), default [1.0; 8]
}
```

**Impact:** 32 bytes added per embedding. For 10K embeddings = 320KB. Negligible.
**Backward compat:** `confidence` computed as `mean(precision) / max(precision)` when feature enabled.

### Fusion 2: BFCF Region Stability via Precision Anchoring

**Core Insight:** BFCF Tree regions (Accept/Maybe/Reject) can oscillate when embeddings shift between decode steps. BAKE's precision anchoring prevents this: high-precision regions resist boundary movement.

**Mechanism:**
- Each BFCF region has a `boundary_precision: f32` derived from the mean precision of its constituent embeddings
- When region boundaries shift between decode steps, precision-weighted smoothing prevents oscillation
- `boundary_new = (precision_old ⊙ boundary_old + λ_obs ⊙ boundary_obs) / (precision_old + λ_obs)`
- This is literally BAKE eq 3 applied to region boundaries instead of entity embeddings

### Fusion 3: Session-Level Embedding Evolution

**Core Insight:** Across inference sessions, KG embeddings accumulate evidence. BAKE's posterior-as-prior naturally enables this:
- Session N: embedding = posterior from session N-1 + new observations
- No replay buffer needed — the precision vector IS the compressed history
- High-precision dimensions are "I've seen this many times" — resistant to outlier sessions
- Low-precision dimensions are "I'm still learning about this" — absorb eagerly

**Implementation:**
- Store `KgEmbedding` with precision in persistent cache (BFCF × LFU shard)
- On session start, load embeddings with their precision vectors
- On session end, apply Bayesian update with session observations
- New entities start with uninformative priors: precision = [0.1; 8]

### Fusion 4: ThoughtFold Precision-Gated Fold Confidence

**Core Insight:** ThoughtFold's fold decisions (keep/fold/revisit) can be precision-gated:
- Steps where the KG embedding has HIGH precision → fold is safe (knowledge is well-established)
- Steps where the KG embedding has LOW precision → fold is risky (knowledge is uncertain)
- The bandit already tracks fold success rate → precision adds a prior signal

### Fusion 5: SenseBandit Precision-Weighted Exploration

**Core Insight:** The existing SenseBandit uses scalar confidence for sense trial weighting. BAKE's per-dimension precision enables:
- Sense trials directed at LOW-precision dimensions (where learning is needed)
- Sense acceptance gated by HIGH-precision dimensions (where knowledge is reliable)
- Decay function: `decay_direction(precision_dim) = sigmoid(-λ * (1.0 - precision_dim))`

---

## GOAT Gate

| Criteria | Target | Measurement |
|----------|--------|-------------|
| Precision update SIMD auto-vectorizes | ≥95% of theoretical peak | Benchmark vs scalar loop |
| Zero-cost when feature disabled | 0 bytes added, 0 instructions | `cargo bench` compare |
| Embedding drift reduction | ≥30% less drift over 5 sessions | Session replay benchmark |
| BFCF region oscillation reduction | ≥50% fewer region flips | Region stability counter |
| Backward compatibility | All existing tests pass | `cargo test` |

---

## Prior Art in Our Stack

| Component | Status | Gap BAKE Fills |
|-----------|--------|---------------|
| `KgEmbedding.confidence` | Scalar f32 | Per-dimension precision vector |
| BFCF region boundaries | No temporal stability | Precision-anchored boundaries |
| SenseBandit | 1D confidence | Per-dimension certainty for directed exploration |
| ThoughtFold | Bandit-gated folding | Precision-gated folding safety |
| `FreqBandit` | Spectral frequency bands | Precision as Bayesian frequency evidence |

---

## Verdict

**GAIN for katgpt-rs.** The precision vector is a 32-byte extension to an existing struct, feature-gated, auto-vectorizes, and enables a new capability we don't have: **inference-time continual learning for KG embeddings without any LLM training**. The update rule is pure arithmetic (O(8) per embedding), fits our zero-alloc hot-path constraints, and naturally composes with BFCF, SenseBandit, and ThoughtFold.

**Confidence: High.** BAKE's ablation shows 7% MRR improvement from Bayesian alone, 11% H@1 from clustering. Even distilling just the precision vector (no clustering) should yield measurable drift reduction.

**Priority: High.** This is a natural extension of Research 196 (KG Latent Octree) and Plan 218 (BFCF × LFU Sharding), both already default-ON.

---

## TL;DR

BAKE's per-dimension precision vector replaces scalar confidence with certainty budgets per embedding dimension. O(d) update, zero-alloc, SIMD-friendly. Apply to `KgEmbedding` (32B extension), BFCF region boundaries (oscillation prevention), SenseBandit (directed exploration), and ThoughtFold (precision-gated folding). Feature-gated `bake_precision`, GOAT gate before default.
