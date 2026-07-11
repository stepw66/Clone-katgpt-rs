# Research 401: Bottlenecked Transformers — Periodic KV Cache Consolidation for Generalised Reasoning

> **Source:** [Bottlenecked Transformers: Periodic KV Cache Consolidation for Generalised Reasoning](https://arxiv.org/abs/2505.16950) — Adnan Oomerjee, Zafeirios Fountas, Haitham Bou-Ammar, Jun Wang (UCL + Huawei Noah's Ark), May 2025 (v4: Mar 2026)
> **Date:** 2026-07-09
> **Status:** Active — GOAT verdict, PoC-gated
> **Related Research:** 213 (Still Perceiver KV compaction), 233 (Attention Matching KV compaction), 024/053 (δ-Mem — negative result), 241 (SwiR entropy switch), 325 (Latent Reasoning Survey), 199 (Memory Caching Growing RNN), 318 (Sleep-Time)
> **Related Plans:** 420 (this research's implementation plan — modelless KV consolidation)
> **Classification:** Public (katgpt-rs/MIT) — open primitive only

---

## TL;DR

A cache-operator ALSC (Auxiliary Latent-Space Computation) method that periodically rewrites KV cache entries in-place at reasoning step boundaries, justified by Information Bottleneck (IB) theory: autoregressive training incentivizes the KV cache to be *minimally compressive* of input (maximize I(X;Z)), and periodic rewrites can reduce I(X;Z) while preserving predictive information I(Z;Y), improving generalization. Two operations — **consolidation** (rewrite recent step's entries) and **reconsolidation** (rewrite top-k attention-selected prior entries) — run at every newline-delimited step boundary via a trained auxiliary "Cache Processor." Paper reports +6.6pp on SVAMP (Llama-3.2 1B), +4.6pp on GSM8K (3B), consistent gains across 7 math benchmarks.

**Distilled for katgpt-rs (modelless, inference-time):** The paper's Cache Processor is a TRAINED auxiliary Transformer. The modelless distillation replaces it with a deterministic **sigmoid-gated value mean-shift**: recalled entries' values move toward the recent step's mean value, weighted by attention relevance, gated by sigmoid. The selection mechanism (top-k by attention mass) and the IB theoretical framework transfer directly. The paper itself flags (§7) that surprise/prediction-error gating would be better than newline triggers — which our codebase already ships (δ-Mem surprise gate, SwiR entropy switch), making the modelless version potentially BETTER-triggered than the paper.

**Verdict: Gain (architectural only) — PERMANENTLY.** Novel KV cache operation class (periodic in-place consolidation for reasoning quality, NOT footprint reduction). Our existing KV work (213 Still Perceiver compaction, 233 Attention Matching selection) is all compression/selection — none rewrites entries in-place. **Quality claim REFUTED on BOTH untrained AND trained models**: Plan 420 Phase 1 (untrained) and Plan 313 (trained, 31% accuracy) both show consolidation ≈ baseline ≈ random-rewrite — the paper's gain comes from a TRAINED Cache Processor, not from mean-shift as a deterministic operation. Plan 420 Phases 2–4 permanently shelved.

---

## 1. Paper Core Findings

### 1.1 The ALSC taxonomy (positions this work)

The paper introduces **Auxiliary Latent-Space Computation (ALSC)** — inference-time procedures that transform the model's KV cache `h_t` and/or final hidden state `o_t` between decoding steps: `(h', o') = T(h_t, o_t)`. Three execution pathways:

| Pathway | Mechanism | Our coverage |
|---------|-----------|-------------|
| (i) Token-mediated | Append latent tokens / pause tokens / latent rollouts to the cache | SwiR (241, DEFAULT-ON), MUX (158), Coconut family |
| (ii) Residual-operator | Modify `o_t` only (activation steering), leave `h_t` unchanged | CHaRS (389), spherical steering (382), latent field steering (153) |
| (iii) Cache-operator | Transform `h_t` directly (prune, merge, compress) | 213 Still, 233 AM, 109 ShardDrop, 101 CachePrune, etc. |

**The gap:** all our cache-operators are **compression-oriented** (reduce footprint). The Bottlenecked Transformer is a cache-operator that **rewrites for quality without reducing footprint** — a fourth sub-category: *consolidation/reconsolidation*.

### 1.2 The IB theoretical framework (the transferable insight)

**Theorem 4.1:** In a decoder-only Transformer, the KV cache + final hidden state `C_{0:n} = (K_{0:n}, V_{0:n}, O_n)` is the **terminal information bottleneck** mediating input `S_{0:n}` and output `S_{n+1}`.

**Theorem 4.2:** Autoregressive training (maximize next-step log-likelihood `L(θ)`) encourages BOTH:
- High `I(S_{0:n}; C_{0:n})` — the cache retains maximal input information (minimally compressive)
- High `I(C_{0:n}; S_{n+1})` — the cache is maximally predictive of the next step

The problem: the KV cache carries **extraneous detail** from processed sequences (a high-fidelity step-by-step predictive trace of all right-shifted tokens), not a single compressed summary. This hinders generalization.

**IB solution:** a rewrite `Ẑ = T(Ẑ)` that increases predictive efficiency `I(Ẑ;Y)/I(X;Ẑ)`. By the data processing inequality, any such transformation satisfies `I(X;Ẑ) ≤ I(X;Ẑ_orig)` — input information only decreases. By training T to minimize future prediction error, `I(Ẑ;Y)` is preserved or improved.

**Generalization bound (Kawaguchi 2023):** `ε ≤ O(√((I(X;Z)+1)/n))` — reducing I(X;Z) directly reduces the generalization error bound. This is WHY consolidation helps reasoning: it's not about memory, it's about removing irrelevant information that causes overfitting to the specific reasoning trace.

### 1.3 The Cache Processor mechanism

At each newline-delimited reasoning step boundary:

1. **Selection** — for each layer ℓ, construct index set `I_n^(ℓ) = J_n ∪ TopK_n^(ℓ)(P_{<n})`:
   - `J_n` = token indices of the just-completed step (recent step window, RSW)
   - `TopK_n^(ℓ)(P_{<n})` = k prior positions with largest attention mass from the current step
   - Attention mass: `α_i^(ℓ) = (1/(|J_n|·H)) Σ_h Σ_{j∈J_n} A^(ℓ,h)_{j,i}`
2. **Rewrite** — convert selected `(k,v)` to "KV-tokens", project to processor hidden space via learned `W_in`, process through non-causal Transformer block, project back via `W_out`, apply gated residual:
   - `k ← k + σ(g)·Δk`, `v ← v + σ(g)·Δv`
   - `g` is a learnable layer-wise scalar gate initialized small, σ = logistic sigmoid
3. **Resume** decoding from the rewritten cache.

**Training:** two-stage. Stage 1: SFT backbone on reasoning trajectories. Stage 2: freeze backbone, train Processor ω via next-step CE loss with BPTT truncated across step boundaries.

### 1.4 Empirical results

| Backbone | Task | SFT | SFT+pause | SFT+latent-rollout | Bottlenecked | Δ (best) |
|----------|------|-----|-----------|---------------------|-------------|----------|
| Llama-3.2 1B | SVAMP | 38.0 | 37.4 | 35.8 | **44.6** | **+6.6** |
| Llama-3.2 3B | GSM8K | 46.78 | 45.51 | 44.05 | **51.33** | **+4.6** |
| Qwen-3 0.6B | MATH | 26.68 | 26.21 | 25.84 | **29.08** | **+2.4** |
| Llama-3.1 8B | LogiQA | 20.74 | 20.61 | 19.83 | **23.81** | **+3.1** |

Gains concentrate on in-distribution math (GSM8K, MATH, SVAMP, GSM-Hard). OOD tasks (Gaokao-Math, Chinese) often favor plain SFT — distribution shift beyond the Processor's training.

### 1.5 The critical empirical finding (§6.5) — guides the modelless distillation

**Figure 4 analysis:** The Processor mainly edits **VALUE vectors** (keys barely change). Rewrite magnitudes:
- Value cosine distance: ~0.1–0.3 (moderate, consistent adjustments — NOT dramatic rewrites)
- Key cosine distance: ~0.0 (essentially unchanged — addressing preserved, content refined)
- **Layer concentration:** edits concentrate in **earliest layers**, small changes in middle/late layers
- **Temporal dynamics:** largest rewrites at early processing steps, settle to stable plateau after ~10 invocations

**Interpretation:** the Processor learns to reshape low-level value representations that propagate forward, rather than rewriting deep layers directly. This is a gentle refinement — which a deterministic modelless analog can approximate.

---

## 2. Distillation

### 2.1 What transfers modellessly (no training needed)

| Component | Paper version | Modelless replacement |
|-----------|---------------|----------------------|
| Step-boundary trigger | Newline token | **Entropy/surprise gate** (SwiR block-relative entropy, δ-Mem surprise gate) — BETTER than newline per paper §7 |
| Selection: top-k prior entries | Attention mass from recent step | **Identical** — deterministic attention scores, already computed |
| Selection: recent step window | Fixed R tokens | **Identical** — mechanical |
| Consolidation (recent entries) | Trained Processor rewrite | **Sigmoid-gated value mean-shift** toward step mean |
| Reconsolidation (recalled entries) | Trained Processor rewrite | **Sigmoid-gated value mean-shift** toward recent step's mean, weighted by attention relevance |
| IB theoretical framework | Theorems 4.1–4.2 | **Transfers directly** — justifies ANY deterministic I(X;Z)-reducing, I(Z;Y)-preserving rewrite |

### 2.2 The modelless consolidation primitive

Replace the trained Processor with a deterministic value update. For each selected entry `i` at layer ℓ:

```
// Consolidation (recent step entries j ∈ J_n):
v_j^(ℓ) ← v_j^(ℓ) + σ(g_c) · (μ_v^(ℓ) − v_j^(ℓ))
//   where μ_v^(ℓ) = mean(v_{J_n}^(ℓ))  — recent step's mean value
//   g_c = consolidation gate scalar (layer-wise, sigmoid-bounded)

// Reconsolidation (recalled entries i ∈ TopK):
v_i^(ℓ) ← v_i^(ℓ) + σ(g_r · α_i) · (μ_v^(ℓ) − v_i^(ℓ))
//   where α_i = normalized attention mass of entry i from recent step
//   g_r = reconsolidation gate scalar (layer-wise, sigmoid-bounded)
```

**Why this is the modelless analog:**
- Moves recalled values toward the recent step's mean — "recalled memories get updated with new contextual information" (the paper's reconsolidation definition)
- Moves recent values toward their own mean — "stabilise newly formed memory traces" (the paper's consolidation definition)
- Sigmoid-gated per AGENTS.md constraint #2 (sigmoid, never softmax)
- Attention-weighted — entries more relevant to the recent step get larger updates
- Only edits values (not keys) — matches paper's §6.5 finding that keys barely change
- **Reduces I(X;Z)** by projecting onto a lower-information subspace (the recent step's mean direction)
- **Preserves I(Z;Y)** because the update is toward the step most predictive of the next step

**IB justification for the modelless version:** the value mean-shift is a deterministic projection that reduces variance of the selected entries. By the data processing inequality, `I(X;Z') ≤ I(X;Z)` holds for any deterministic transformation. The question is whether `I(Z';Y)` is preserved — which is the §3.6 PoC's job to verify.

### 2.3 Layer-wise gating schedule (from §6.5)

The paper shows edits concentrate in early layers. The modelless version should use a **decaying layer gate**:
```
g_c^(ℓ) = g_max · sigmoid(−λ · (ℓ / L))   // early layers get larger gates
```
where `g_max` is the max consolidation strength and λ controls the decay rate. This matches the empirical observation without needing a learned per-layer gate.

### 2.4 Surprise-triggered consolidation (the fusion upgrade)

The paper uses a fixed newline trigger. §7 explicitly states: "reconsolidation appears to depend on prediction error at retrieval... surprise/PE gating (rather than a fixed newline trigger) would be more suitable."

**Our codebase already ships this:**
- δ-Mem surprise gate (`pruners/delta_mem/state.rs`) — triggers on prediction error
- SwiR block-relative entropy switch (Plan 275) — triggers on entropy spike
- Temporal derivative kernel (Plan 277) — temporal surprise signal

**Fusion:** trigger consolidation when the surprise signal exceeds a threshold, NOT on every newline. This is:
- Biologically more accurate (prediction error opens the reconsolidation window)
- More efficient (consolidate only when needed, not every step)
- Already supported by our entropy/surprise infrastructure

### 2.5 Connection to δ-Mem (the revival hypothesis)

δ-Mem (Research 024, Plan 053) was a **negative result for DDTree**: the delta-rule `S' = (1-β)S − β(S·k)⊗k + β·v⊗k` converged but corrections were too small to flip branch ordering, with 26× latency overhead.

**The Bottlenecked Transformer reframes why δ-Mem failed for DDTree but might work for reasoning:**
- DDTree branches are discrete (select/drop) — small value corrections can't flip a hard branch decision
- KV cache reasoning is continuous (next-token distribution) — small value corrections CAN shift the output distribution
- The IB framework says the correction should reduce I(X;Z), not just store a new association

**Fusion hypothesis:** δ-Mem's delta-rule × Bottlenecked Transformer's IB framework × attention-selected reconsolidation = a modelless KV consolidation that works where δ-Mem-alone failed. The delta-rule provides the update mechanism; the IB framework provides the objective (reduce I(X;Z)); the attention selection provides the sparsity (only rewrite relevant entries). This needs the PoC to verify.

---

## 3. Fusion Ideas — Modelless (katgpt-rs)

### F1: IB-Gated Value Consolidation (the core primitive)
**What:** sigmoid-gated value mean-shift at surprise-triggered step boundaries, layer-decaying gate.
**Connects:** paper's IB framework + δ-Mem delta-rule + SwiR entropy trigger + SpectralQuant eigenbasis.
**Why it matters:** fills the cache-operator gap (consolidation, not compression). First KV primitive that improves reasoning quality without reducing footprint.

### F2: Subspace-Projected Reconsolidation
**What:** instead of mean-shift toward step mean, project recalled values onto the **principal subspace** of the recent step's values (PCA of the RSW value vectors).
**Connects:** SpectralQuant (039) eigenbasis + paper's IB framework.
**Why it matters:** the PCA subspace is the maximum-variance direction of the recent step — projecting onto it maximally preserves I(Z;Y) while maximally reducing I(X;Z). This is the IB-optimal deterministic projection.

### F3: Conformal Consolidation Confidence Gate
**What:** use conformal prediction (Plan 340 conformal floor) to decide WHEN to consolidate — only consolidate when the conformal interval width exceeds a threshold (high uncertainty = needs consolidation).
**Connects:** conformal UQ overlay (340, 322) + paper's surprise-gating suggestion.
**Why it matters:** principled trigger for consolidation based on calibrated uncertainty, not just raw entropy.

---

## 4. Verdict

**Tier: Gain (architectural only, downgraded from GOAT per §6 PoC Addendum).**

| Criterion | Assessment |
|-----------|------------|
| Novel mechanism (no prior art in our corpus) | ✅ — confirmed: all existing KV work (213, 233, 109, 101, 083, 063, 042, 039, 165, 159) is compression/selection/quantization. NONE does periodic in-place consolidation for quality. The `cgsp/dual_pool.rs consolidate()` is for the bandit pool, not KV cache. |
| Provable gain | ❌ — quality claim REFUTED on untrained AND trained models (§6 PoC Addendum). The modelless mean-shift has no measurable quality effect even when the KV cache carries learned structure (31% accuracy, Plan 313). The paper's gain comes from a TRAINED Cache Processor, not from mean-shift as an operation. |
| New class of capability | ✅ — first KV cache operator that improves reasoning quality without reducing footprint (consolidation, not compression) |
| Modelless | ✅ — selection mechanism transfers directly; trained Processor replaced by deterministic sigmoid-gated value mean-shift (§3.5 path 3: latent-space correction) |
| Force multiplier | ✅ — connects to reasoning pack (P8), KV cache stack, δ-Mem revival hypothesis, neuron-db consolidation analogy |

**Downgrade rationale:** the §3.6 PoC (Plan 420 Phase 1) showed modelless consolidation is inert on untrained models, and riir-train Plan 313 confirmed it is ALSO inert on trained models (31% accuracy, 0.00pp gain). The paper's Cache Processor is a TRAINED auxiliary Transformer — the quality gain is inseparable from the learned consolidation rule. The deterministic mean-shift uniformly compresses all values toward an uninformative centroid, which is not the IB-prescribed selective removal of extraneous detail. Plan 420 Phases 2–4 permanently shelved.

**MOAT gate (katgpt-rs):** ✅ in-scope. KV cache primitives are explicitly in katgpt-rs's MOAT scope ("Transformer stack — KV cache"). This is a paper-derived fundamental primitive for the KV cache stack. Public per commercial strategy (the adoption funnel depends on engine-quality KV primitives).

### §3.6 PoC requirement (mandatory before promotion)

**Claim to defend:** "the modelless sigmoid-gated value mean-shift achieves quality parity (or meaningful fraction) with the paper's trained Cache Processor on a reasoning task."

**PoC design (Plan 420 Phase 1):**
- Three competitors on a controlled toy reasoning task (e.g., multi-step arithmetic in a micro-GPT):
  1. **No consolidation** (baseline — vanilla KV cache)
  2. **Modelless consolidation** (sigmoid-gated value mean-shift, surprise-triggered)
  3. **Paper's mechanism analog** (if feasible without training — else skip and compare to baseline only)
- Metric: reasoning accuracy (exact-match) at fixed token budget
- If modelless consolidation beats baseline by ≥2pp → GOAT confirmed, proceed to feature flag
- If modelless consolidation ≈ baseline → the consolidation primitive is architectural-only (no quality gain modellessly), demote to Gain or shelve

**The PoC defends OR refutes.** If it refutes quality parity, record raw numbers in a §"PoC Addendum" and downgrade the verdict honestly.

---

## 5. Related Work Map

| Cousin | Relationship |
|--------|-------------|
| **213 Still Perceiver** | Sibling — synthesis-based COMPACTION (reduces count). This is CONSOLIDATION (rewrites in-place, no count reduction). |
| **233 Attention Matching** | Sibling — SELECTION-based compaction (keep/drop with optimal β). This rewrites ALL selected entries, doesn't drop any. |
| **024/053 δ-Mem** | Revival hypothesis — δ-Mem's delta-rule is a modelless consolidation mechanism that failed for DDTree (discrete branches). KV cache reasoning (continuous distribution) may be where it works. |
| **241 SwiR** | Trigger source — entropy switch provides the surprise-triggered consolidation boundary (better than paper's newline trigger). |
| **318 Sleep-Time** | Conceptual cousin — sleep-time anticipates future queries; this consolidates current working memory. Both are "offline-ish" memory operations during inference. |
| **226 SegmentCheckpoint** | Mechanical cousin — saves KV at segment boundaries. This REWRITES KV at segment boundaries. Could compose: checkpoint after consolidation. |
| **CaM (Zhang 2024, cited in paper)** | Prior art — cache MERGING for memory efficiency. This is consolidation for QUALITY, not footprint. |
| **325 Latent Reasoning Survey §7.2** | The G1–G8 gaps (distilled as 343–349) do NOT cover KV cache consolidation. Confirmed novel against the survey. |

---

## 6. PoC Addendum (Plan 420 Phase 1 + Plan 313 trained-model validation, 2026-07-09)

**Outcome: QUALITY CLAIM REFUTED — on BOTH untrained and trained models.**

The §3.6 PoC ran the modelless sigmoid-gated value mean-shift against a no-consolidation baseline and a random-rewrite control on 200 few-shot addition problems × 3 seeds = 1800 evaluations in a single-layer micro-GPT (d_model=64, 8 heads, random weights).

### Raw numbers

| Competitor | EM_rate | Token_acc | Mean_NLL | DigitMass | N_consol |
|---|---|---|---|---|---|
| Baseline | 0.0000 | 0.0275 | 7.8057 | 0.4931 | 0.0 |
| Modelless consolidation | 0.0000 | 0.0269 | 7.8058 | 0.4931 | 5.0 |
| Random-rewrite control | 0.0000 | 0.0275 | 7.8053 | 0.4931 | 5.0 |

- **Δ(consolidation − baseline)**: token_acc −0.06pp, NLL +0.0001 — no meaningful difference.
- **Δ(consolidation − random-rewrite)**: NLL +0.0005 — consolidation is NOT better than random perturbation.
- **Hyperparameter sweep** (g_max ∈ {0.1, 0.3, 0.5}, k ∈ {16, 32, 64}): zero sensitivity — all configs produce identical token_acc (0.0133) and near-identical NLL (7.8627–7.8630).
- **Self-test**: consolidation code verified correct (keys unchanged, values modified, reconsolidation works, variance reduced 74.6%, no NaN).

### What was confirmed vs refuted

| Claim type | Result |
|---|---|
| **Architectural** (mechanism exists) | ✅ CONFIRMED — consolidation runs, modifies values, preserves keys, reduces variance. |
| **Latency** (modelless, no GD) | ✅ CONFIRMED — pure deterministic, no training. |
| **Quality** (matches paper's gains) | ❌ REFUTED on untrained AND trained models — the modelless mean-shift has no measurable quality change even when the KV cache carries learned structure (31% accuracy). The paper's gain comes from a TRAINED Cache Processor, not from mean-shift as an operation. |

### Why refuted (the mechanism analysis)

The paper's IB argument (Theorems 4.1–4.2) applies specifically to TRAINED models: autoregressive training incentivizes the KV cache to be minimally compressive of input (maximize I(X;Z)), creating "extraneous detail" that consolidation removes. On an UNTRAINED model (random weights), the KV cache has no learned "extraneous detail" — value vectors are random projections of random embeddings. Consolidation (mean-shift toward step mean) changes the values, but this change is not information-preserving or information-reducing in any meaningful sense — it's just a different random-ish distribution that happens to have lower variance.

The random-rewrite control confirms this: a perturbation of the same magnitude but random direction produces identical results. The mean-shift direction (toward the step mean) has no special property on untrained models because the step mean itself is uninformative.

### Revised verdict

**Tier: Gain (architectural only) — PERMANENTLY.** The consolidation mechanism is novel, correct, and fills a genuine gap in the KV cache operator taxonomy. But the quality claim is REFUTED on both untrained and trained models. The paper's Cache Processor is a TRAINED auxiliary Transformer — the gain comes from the learned consolidation rule, not from mean-shift as a deterministic operation. The modelless mean-shift uniformly compresses ALL values (signal + noise) toward an uninformative centroid, which is not what the IB argument prescribes.

Plan 420 Phases 2–4 are permanently shelved. This is NOT a modelless failure or a task-appropriateness boundary — it's a fundamental mismatch: the paper's quality gain is inseparable from the TRAINED consolidation rule, and no deterministic mean-shift captures it.

### riir-train follow-up — RESOLVED (Plan 313, 2026-07-09)

**The trained-model validation has been completed** in riir-train Plan 313 (`tests/plan_313_kv_consolidation_trained.rs`). A micro-GPT (same architecture as bench_420: d_model=64, 8 heads, single layer) was trained from scratch on few-shot addition using a manually-backpropagated AdamW optimizer (gradient check: 18/18 weights pass, max rel_err 1.53%). The trained model achieves **31.5% train / 31.1% test token accuracy** — well above the ~10% chance floor, confirming the KV cache carries genuine learned structure.

The 3-competitor PoC was then re-run on the trained model:

| Competitor | EM_rate | Token_acc | Mean_NLL | DigitMass |
|---|---|---|---|---|
| Baseline | 0.0250 | 0.2758 | 4.9986 | 0.9997 |
| Modelless consolidation | 0.0250 | 0.2758 | 4.9983 | 0.9997 |
| Random-rewrite control | 0.0250 | 0.2758 | 4.9980 | 0.9997 |

Consolidation shows **0.00pp** gain on both exact-match and token-accuracy (bit-identical to baseline), and only −0.0003 NLL improvement (smaller than random-rewrite's −0.0006 — statistical noise). **The quality claim is REFUTED on trained models too.**

### Why refuted even on trained models

The paper's Cache Processor is a TRAINED auxiliary Transformer that LEARNS what to consolidate. The modelless distillation replaces it with a deterministic mean-shift — but the mean-shift direction (toward the step mean) has no learned signal about WHAT to remove. On a trained model, the KV cache DOES carry learned structure (31% accuracy proves this), but the blind mean-shift toward the step centroid doesn't selectively remove "extraneous detail" — it uniformly compresses ALL values, signal and noise alike, toward an uninformative centroid. The paper's gain comes from the TRAINED consolidation rule, not from mean-shift as an operation.

This confirms the primitive is **permanently Gain (architectural only)** — the consolidation mechanism is novel and correct, but the modelless quality claim does not hold. Plan 420 Phases 2–4 are permanently shelved.

---

## TL;DR

**Verdict: Gain (architectural only) — PERMANENTLY REFUTED on quality.** The Bottlenecked Transformer introduces a genuinely novel KV cache operation — periodic in-place consolidation/reconsolidation at reasoning step boundaries, justified by Information Bottleneck theory (reduce I(X;Z), preserve I(Z;Y)). This fills a gap: all our existing KV work (213 Still, 233 AM, 109 ShardDrop, etc.) is compression/selection; none rewrites for quality without reducing footprint. The modelless distillation replaces the paper's trained Cache Processor with a deterministic sigmoid-gated value mean-shift. **Quality claim REFUTED on BOTH untrained models** (Plan 420 Phase 1: consolidation ≈ baseline ≈ random-rewrite, zero sensitivity to hyperparameters) **AND trained models** (Plan 313: same architecture trained to 31% accuracy, still 0.00pp gain — the mean-shift uniformly compresses signal+noise toward an uninformative centroid, not selectively removing extraneous detail). The paper's gain is inseparable from the TRAINED consolidation rule. Plan 420 Phases 2–4 permanently shelved.
