# Research 399: Distributed Sparse Interventions (DSI) — PASS

> **Source:** [Distributed Sparse Interventions in Language Models](https://arxiv.org/pdf/2607.07128) — Ernst, Linhardt, Peikert, Eberle (TU Berlin / MPI Human Development / BIFOLD), arXiv:2607.07128v1, 8 Jul 2026
> **Date:** 2026-07-09
> **Status:** Done — PASS
> **Classification:** Public

**Verdict:** → PASS. Core mechanism already ships as **CNA Steering** (R053 / Plan 087, `cna_steering` feature, **DEFAULT-ON**, GOAT-proved quality >0.97). Iterative refinement + LRP-modified gradients are offline discovery improvements (analysis-time, not runtime modelless). The strictly stronger causal head-importance scoring already ships as **HydraHead** (R362 / Plan 358, activation-patching counterfactual IE score). The one novel conceptual contribution — **set-based task decomposition** (intersection/union of sparse neuron sets, §5) — does not map to our substrate (we operate on HLA 8-dim affective state, `latent_functor` vector ops, `NeuronShard` fixed `style_weights[64]`, DEC cochains — NOT sparse LLM attention-head neuron sets), and the latent analog already exists via **CommittedFieldBlend** (R302/Plan 321 archetype intersection) + **RIZZ NonInterferenceProjection** (R310/Plan 327 orthogonal subspace branch isolation). No files, no plan, no issue.

---

## TL;DR

DSI identifies **sparse sets of 8–64 neurons** (0.01–0.04% of total, distributed across attention heads and layers) that, when **additively modulated** during a forward pass, activate task behavior in instruction-tuned LLMs at accuracy matching or exceeding 10-shot ICL. Discovery uses (1) mean activation differences between k-shot and 0-shot prompts, (2) LRP-modified gradients for robustified first-order effect estimation, (3) iterative refinement via ZeroFPR nonconvex sparse optimization to account for nonlinear neuron-pair interactions. The paper's second contribution is a **set-based perspective on task composition**: tasks decompose into shared (e.g., "copy") and task-specific neuron subsets via set operations (intersection/union), enabling finer-grained analysis and control than direction-based steering.

**Why PASS:** every load-bearing mechanism already ships under a different name.

| DSI component | Codebase prior art | Status |
|---|---|---|
| Sparse neuron selection via mean activation difference (k-shot − 0-shot) | **CNA Steering** (R053/Plan 087, `cna_steering` **DEFAULT-ON**) — same math: `δℓj = mean(P+) − mean(P−)` → top-k 0.1% sparse set | ✅ Ships, GOAT-proved, quality >0.97 |
| Additive activation modulation at inference (`a ← a + δ`) | **CNA** (`cna_modulate`, multiplicative `hidden[neuron] *= m`) + **Latent Field Steering** (R290/Plan 309, additive `s' = s + α·v`) | ✅ Ships, both additive and multiplicative variants |
| Robustified gradient effect estimation (LRP: identity/half/LayerNorm rules) | **HydraHead** (R362/Plan 358) — strictly stronger: causal counterfactual IE score via activation patching, forward-pass-only, stable from ~6 samples | ✅ Ships, strictly stronger (causal > first-order gradient) |
| Iterative refinement (ZeroFPR nonconvex sparse optimization) | Offline discovery procedure; the *application* is modelless. CNA's single-shot discovery already achieves quality >0.97. | ⚠️ Marginal discovery improvement; not a runtime primitive |
| Set-based task composition (intersection/union of neuron sets, §5) | **CommittedFieldBlend** (R302/Plan 321) — archetype direction intersection/union at the latent level; **RIZZ NonInterferenceProjection** (R310/Plan 327) — orthogonal subspace branch isolation | ✅ Latent analog ships; DSI's neuron-set variant doesn't fit our substrate |

---

## 1. Paper Core Findings

### 1.1 The DSI algorithm (paper §3)

Four-step discovery + application:

1. **Activation differences** — `a− = mean(D_k) − mean(D_0)` per neuron (k-shot minus 0-shot).
2. **Robustified gradient** — `∂_r f` using LRP rules (identity for activations, half for gated MLP/attention, LayerNorm rule for RMSNorm) to mitigate gradient shattering.
3. **Sparse initial set** — top-n neurons by expected effect `ē = g ⊙ a−`.
4. **Iterative refinement** — ZeroFPR (nonconvex optimizer with L0 ball penalty), 50 steps × 10 perturbed restarts, re-evaluating effects at the updated intervention point to capture nonlinear neuron-pair interactions.

**Application** (paper §3, step 4): `f_δ := f(x | a ← a + δ)` where `δ = s ⊙ a−` and `s` is the per-neuron scaling factor from the optimizer. Additive, inference-time, no weight mutation.

### 1.2 Empirical results (paper §4)

| Set size | Llama 3.2 (3B) abstractive | Qwen 3 (8B) abstractive | Gemma 3 (4B) abstractive |
|---|---|---|---|
| baseline (no intervention) | 0.02 | 0.04 | 0.01 |
| 8 neurons (DSI) | 0.63 | 0.54 | 0.51 |
| 64 neurons (DSI) | 0.75 | 0.73 | 0.72 |

Both gradient robustification and iterative refinement contribute consistently. Neurons are distributed across heads and layers (broadly spread, not confined to predefined layers — contradicting prior layer-fixed steering approaches).

### 1.3 Set-based task composition (paper §5 — the novel conceptual contribution)

Decompose the `present-past` task into:
- **Copy neurons** (intersection with a copy-task neuron set) — reproduce the input.
- **Task-specific neurons** (set difference) — apply the tense transformation without copying.

Set operations (∩, ∪, \) on sparse neuron index sets provide finer-grained control than direction-based steering's additive vector composition. Quantitatively: intersecting `present-past` with `copy` (23 neurons) solves copy at 0.92; the 9 task-specific neurons solve `present-past` at 0.37 with reduced copy behavior.

---

## 2. Why PASS — prior-art accounting

### 2.1 CNA Steering is the direct prior art (the load-bearing finding)

**Research 053 / Plan 087 / `cna_steering` (DEFAULT-ON, GOAT-proved)** ships the exact DSI mechanism:

| DSI step | CNA equivalent | Difference |
|---|---|---|
| `a− = mean(D_k) − mean(D_0)` | `δℓj = mean(P+) − mean(P−)` | **None** — both are mean activation differences over contrastive sets. DSI's k-shot/0-shot is one instance of CNA's positive/negative framing. |
| Top-n sparse selection | Top-0.1% by `|δℓj|` | **None** — same top-k by activation-difference magnitude. |
| Additive modulation `a ← a + δ` | Multiplicative modulation `hidden[neuron] *= m` | **Variant** — CNA uses multiplicative (m=0 ablate, m>1 amplify); DSI uses additive. Both are inference-time activation hooks. Our Latent Field Steering (R290) ships the additive variant at the latent level. |
| MLP post-ReLU activations | MLP post-ReLU activations (`ctx.hidden`) | **None** — same activation tensor. |
| Universal neuron filtering | (CNA §2) — neurons firing ≥80% across diverse prompts excluded | **CNA has this; DSI does not explicitly.** |

CNA's GOAT gate (Benchmark 015) proves quality >0.97 at max steering, >50% refusal reduction, MMLU within 1 point. DSI's 0.72–0.75 accuracy on abstractive tasks at 64 neurons is comparable but NOT better than what CNA achieves on its target behavior. **DSI is not a quality improvement over CNA — it is the same mechanism with a different discovery procedure.**

### 2.2 HydraHead ships the strictly stronger discovery mechanism

DSI's "iterative refinement + LRP gradients" discovery is a **first-order + iterative** approach. **HydraHead (R362 / Plan 358)** ships **causal counterfactual** head-importance scoring via activation patching:

- **DSI first-order**: `E(δ) ≈ ∂_r f · δ` — a gradient approximation.
- **HydraHead causal**: `IE_l,h = (m(x) − m(x; O_l,h ← O_l,h(x'))) / (m(x) − m(x'))` — a counterfactual direct effect via patching corrupted activations.

A head may have high first-order effect (DSI picks it) yet be overridden downstream (a *correlated bystander*). HydraHead's causal patching filters these by construction. The paper itself (§4.2) shows first-order picks "some neurons [that] reduce the output logit" — exactly the correlated-bystander failure HydraHead's counterfactual avoids. **HydraHead is strictly stronger and already ships.**

### 2.3 The set-based composition maps to latent analogs we already ship

DSI's novel angle — set operations on sparse neuron sets for task decomposition — does not fit our substrate:

- **Our runtime substrate**: HLA 8-dim affective state, `latent_functor` vector ops, `NeuronShard` fixed `style_weights[64]`, DEC cochain fields. **No sparse LLM attention-head neuron sets.**
- **DSI's substrate**: 8–64 neurons out of ~147K (Qwen3-8B: 36 layers × 32 heads × 128 neurons/head). Sparse attention-head outputs pre-projection.

The latent analog of "set-based composition" already ships:

| DSI set operation | Latent analog | Where |
|---|---|---|
| Intersection (∩) of task neuron sets → shared "copy" neurons | Archetype direction intersection via CommittedFieldBlend's sigmoid-gated blend `f_π = Σ_k sigmoid(π_k/τ) · f_k` | R302 / Plan 321 |
| Union (∪) → combined task activation | Multi-archetype blend (all K archetypes active) | R302 / Plan 321 |
| Set difference (\) → task-specific neurons without shared | NonInterferenceProjection — orthogonal subspace per branch, updates projected onto one direction don't contaminate others (dot-product = 0 across branches) | R310 / Plan 327 |
| "Copy vs task-specific" decomposition | "Shared archetype vs specific personality" — CommittedFieldBlend with near-zero weight on shared archetypes | R302 + R158 (riir-ai guide) |

The DSI set perspective is a **finer-grained interpretability lens** at the LLM-neuron level; our latent analogs operate at a coarser (direction-vector) granularity that is the *correct* granularity for our substrate (8-dim HLA, not 147K sparse neurons). Forcing sparse-neuron-set operations onto HLA would be a category error.

### 2.4 Compute-unit mismatch (the R368 lesson applied)

DSI's compute unit is "sparse set of LLM attention-head neurons." Our runtime's compute units for the same decisions:

| DSI decision | Our runtime compute unit | Translation |
|---|---|---|
| "Which neurons to intervene on for task T" | "Which archetype directions to blend for personality P" | `CommittedFieldBlend` weight vector `π` |
| "How strongly to modulate neuron j" | "Steering strength α for direction v" | `Latent Field Steering` `α(τ)` calibration (R290) |
| "Discover the neuron set from k-shot/0-shot" | "Mine the direction from verdict-conditioned activation shifts" | **MAG** (R397 / Plan 418, `mag_mining` **DEFAULT-ON**) — the unsupervised acquisition step |

MAG (just shipped 2026-07-09, default-on) is the **unsupervised direction-mining** analog of DSI's k-shot/0-shot activation-difference discovery, operating at the residual-stream direction level (correct granularity for us) rather than the sparse-neuron level (correct for LLM interpretability, wrong for our runtime).

---

## 3. Verdict

### Tier: **Pass**

| Question | Answer |
|---|---|
| Q1 No prior art? | **NO.** CNA Steering (R053, default-on) ships the exact sparse-neuron-selection + runtime-modulation mechanism. HydraHead (R362) ships the strictly stronger causal discovery. MAG (R397, default-on) ships the unsupervised direction-mining analog. CommittedFieldBlend (R302) + RIZZ (R310) ship the latent analog of set-based composition. Vocabulary translation done: "sparse intervention"→"CNA circuit", "activation difference"→"contrast direction", "neuron set operations"→"archetype blend / non-interference projection". |
| Q2 New class of behavior? | **NO.** Every capability DSI demonstrates (sparse activation steering, task activation via few neurons, set-based composition) has a shipped equivalent in our codebase. |
| Q3 Product selling point? | **NO.** "Our NPCs steer via sparse neuron interventions" is already true (CNA, default-on). The set-based decomposition angle doesn't produce a new selling point for our latent-state runtime. |
| Q4 Force multiplier? | **NO.** Would connect to systems already connected (CNA → MAG → CommittedFieldBlend → Latent Field Steering). No new integration surface. |

**One-line reasoning:** DSI is a rediscovery of CNA Steering (R053, default-on) with a more elaborate offline discovery procedure (iterative + LRP gradients) that HydraHead (R362) already supersedes with causal counterfactual scoring; the one novel angle (set-based neuron composition, §5) is a finer-grained interpretability lens at the LLM-neuron level that does not map to our latent-state substrate and whose latent analog already ships via CommittedFieldBlend (R302) + RIZZ NonInterferenceProjection (R310).

### MOAT gate

Not applicable — PASS verdict. No primitive lands in any repo.

---

## 4. What would have made this non-Pass

- If we shipped an **LLM-serving path** (we don't — katgpt-rs is a modelless inference engine, riir-ai is game-AI runtime on HLA/functor/shard substrate), DSI's sparse-neuron approach would be a direct competitor to CNA, and the iterative refinement might be a marginal GOAT over single-shot CNA discovery.
- If our substrate were **sparse attention-head neurons** (it isn't), the set-based composition would be a novel primitive.
- If CNA did not exist, DSI would be a GOAT (provable gain: 0.01% of neurons activates task behavior). But CNA exists, is default-on, and achieves quality >0.97.

---

## 5. Notes for future reference

- The **set-based perspective** (intersection/union of behavior-relevant component sets) is a useful *analytical lens* even if not a primitive. If a future riir-ai guide needs to explain "why does CommittedFieldBlend with K=3 archetypes decompose a personality into shared + specific?", the DSI §5 case study (copy vs task-specific neurons) is the cleanest pedagogical reference for the same idea at a different granularity.
- DSI's finding that **interventions are distributed across layers** (not confined to predefined layers, contradicting CAA/steering-vector approaches) corroborates HydraHead's finding (R362 §1.2: "retrieval is head-localized, not layer-localized; critical heads scattered across layers"). This is a cross-paper confirmation of the head/component-level granularity principle our codebase already operates on.
- The **LRP-modified gradient** rules (identity/half/LayerNorm) are a potentially useful detail if we ever need attribution-patching at scale (HydraHead §1.7 flags this as the scalable alternative to full activation patching). Not actionable now; noted for the FaithfulnessProbe (R244) extension surface.

---

## TL;DR

DSI is a PASS — the core mechanism (sparse neuron selection via activation difference + runtime additive modulation) already ships as CNA Steering (R053/Plan 087, `cna_steering`, **DEFAULT-ON**, GOAT-proved quality >0.97). The iterative refinement + LRP gradients are offline discovery improvements superseded by HydraHead's (R362) causal counterfactual scoring. The one novel conceptual contribution — set-based task composition (intersection/union of neuron sets) — is a finer-grained LLM-interpretability lens that doesn't map to our latent-state substrate (HLA 8-dim, functor vectors, NeuronShard style_weights[64]); its latent analog already ships via CommittedFieldBlend (R302) + RIZZ NonInterferenceProjection (R310). MAG (R397, default-on, shipped 2026-07-09) is the unsupervised direction-mining analog at the correct granularity for our runtime. No files, no plan, no issue.
