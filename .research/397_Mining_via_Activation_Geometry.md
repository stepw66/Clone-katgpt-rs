# Research 397: Mining via Activation Geometry (MAG) — Unsupervised Direction Mining + Modelless Transfer Prediction

> **Source:** [Unsupervised Features Mining via Activation Geometry](https://arxiv.org/abs/2607.04222) — LeVi, David, Fomin (Zenity / Technion), ICML 2026 FAGEN Workshop, Jul 2026
> **Date:** 2026-07-09
> **Status:** Active — Super-GOAT; primitive + plan + private guide created this session
> **Related Research:** 144 (Functional Emotions — supervised cousin), 276 (PersonalityWeightedComposition — direction consumer), 290 (Latent Field Steering — injection cousin), 302 (FAME/CommittedFieldBlend — archetype-direction consumer), 357 (Neural Procedural Memory Activation Steering), 388 (Jacobian Lens — SVD readout cousin), 393 (Block-Sparse Featurizer), 196 (KG Latent Octree — supervised direction extraction)
> **Related Plans:** 162 (EmotionDirections — supervised), 297 (PersonalityWeightedComposition), 309 (Latent Field Steering), 321 (CommittedFieldBlend), 405 (Spherical Steering), 412 (Subspace Steering), 418 (this primitive — open MAG)
> **Cross-ref (riir-ai):** Research 316 — *MAG Unsupervised Direction Mining Game Runtime Guide* (private Super-GOAT selling-point doc)
> **Classification:** Public

---

## TL;DR

MAG (Mining via Activation Geometry) is a **modelless, unsupervised** framework for extracting reasoning-feature direction vectors from activation shifts. Prepend a fixed natural-language transformation Q to every input p; measure the activation delta `m(Q‖p) − m(p)` at a single readout point; the mean shift `v_Q` is a feature direction. The model's own verdict `y_M` (its yes/no answer under Q) serves as the label — **no human annotation needed**. The paper proves: (1) MAG features predict the model's verdict better than raw activations, (2) a single linear direction approximates the prefix shift (reconstruction error ϵ_Q ∈ [0.59, 0.97]), (3) class-mean directions steer verdicts under calibrated α, (4) context-loaded prefixes produce geometrically distinct directions, and (5) **MAG geometry predicts dataset transfer at 94.7% Top-1 accuracy** while raw centroid cosine is near-uninformative (ρ ∈ [0.01, 0.05]).

**Why it matters here:** This codebase has a rich direction-vector *injection* ecosystem (Latent Field Steering R290/153, PersonalityWeightedComposition P297, CommittedFieldBlend P321, Spherical Steering P405, Subspace Steering P412) and a *supervised* acquisition method (EmotionDirections P162 = mean-difference on labeled emotion stories; KG Latent Octree R196 = mean-difference on labeled triples). **MAG is the missing unsupervised acquisition step** — mine direction vectors from the model/runtime's own verdicts without labeled data. Plus the §4 transfer-prediction experiment is a genuinely new capability class: modelless prediction of which experiences/datasets best improve a target capability.

**Distilled for katgpt-rs (modelless, inference-time):**
- `mine_direction(with_prefix, without_prefix)` → mean shift `v_Q = E[Δ_Q]`, unit-norm, BLAKE3-committed
- `mine_contrast_direction(positive, negative)` → class-mean `u_Q = (μ⁻ − μ⁺)/‖·‖` using model-self-labels (NOT human labels)
- `reconstruction_error(...)` → ϵ_Q (linearity diagnostic: 0 = exact single-direction, 1 = no effect)
- `calibrate_alpha(tau, prefix_norm, dir_norm)` → `α(τ) = τ·‖A_prefix‖/‖d‖`
- `transfer_score(candidate, target, ...)` → modelless dataset/experience transfer ranking via class-conditional MAG-operator centroid geometry
- The 8 MAG operators (Direct, Prefixed, Answered, InputDelta, QuestionDelta, Interaction, Verdict, FewShot) as a generic operator family

No training, no gradients, no backprop. The "label" is whatever verdict the host model/runtime emits — model-relative by construction.

---

## 1. Paper Core Findings

### 1.1 The MAG mechanism

Given a transformation Q (a fixed natural-language instruction/context/question) and inputs p:

```
Δ_Q(p) := m(Q ‖ p) − m(p)          // prefix-induced activation shift
v_Q    := E_p[ Δ_Q(p) ]             // mean shift = the feature direction
```

where `m(x)` is the residual-stream readout at the last token of the final block. The shift bundles four mechanisms (direct prefix-token value, attention reweighting, MLP nonlinearities, downstream head interactions); the operator family below isolates parts of it.

The model's **self-label** under a yes/no prefix:
```
y_M(p) = 𝟙[ Pr(yes | Q‖p) > Pr(no | Q‖p) ]
```
This is NOT the dataset label — it is the verdict the model emits. Contrast sets `P⁻ = {p : y_M(p)=1}`, `P⁺ = {p : y_M(p)=0}` are induced from `y_M`, never from external annotation.

### 1.2 The 8 MAG operators

Each operator ϕ is a vector summary of the prompt under MAG:

| Operator | Formula | What it isolates |
|----------|---------|------------------|
| Direct | `A(p)` | Unprefixed baseline |
| Prefixed | `A(Q‖p)` | Canonical question-conditioned activation |
| Answered | `A(Q‖p‖y_M(p))` | Commits the model to its verdict before readout |
| InputDelta | `A(Q‖p) − A(p)` | Subtracts the bare input |
| QuestionDelta | `A(Q‖p) − A(Q)` | Subtracts the bare question |
| Interaction | `A(Q‖p) − A(Q) − A(p) + A(∅)` | Residual capturing Q×p relation |
| Verdict | `A(y_M(p))` | Standalone yes/no activation |
| FewShot | `A(E‖Q‖p)` | Fixed in-context preamble E |

Each ϕ induces a contrast direction `v_{ϕ,Q} = (μ⁻_ϕ − μ⁺_ϕ)/‖μ⁻_ϕ − μ⁺_ϕ‖` defined entirely from `y_M`.

### 1.3 Five empirical results (the load-bearing findings)

1. **Readability (§3.1):** MAG operators (Prefixed, InputDelta) predict the model's verdict `y_M` better than raw activations on every (concept, model) cell. Concept tasks: ROC 0.86 → 0.96 (Gemma ocean). PI corpus: smaller but consistent gains.
2. **Model-relative, not dataset-memorizing:** On disagreement rows (dataset label ≠ y_M), the MAG classifier sides with the model 69.3% (Llama) / 73.7% (Gemma) — Wilson CIs disjoint from 50%. MAG encodes the model's verdict, not the dataset signature.
3. **Linearity (§3.2):** A single readout-time direction reconstructs the prefix shift better than no steering on every cell. ϵ_Q ∈ [0.59, 0.97]. Concept prefixes are more linear than PI; Gemma more linear than Llama. Final-layer-only steering beats layer-wise (per-layer increments overshoot).
4. **Steering (§3.3):** Class-mean direction `u_Q` at calibrated `α(τ)` flips 11–12/12 neutral yes/no verdicts on concept tasks (Gemma at τ=0.3, Llama at τ=1.0). The PI direction does NOT flip neutral prompts — binary safety calls are more entrenched (ϵ_Q ≈ 1 predicts non-steerability).
5. **Composition (§3.4):** Context-loaded prefix (Bob) induces a direction near-orthogonal to its named pieces (cos(d_Bob, d_desert) = −0.04 on Llama) that still generalizes to held-out objects (LOO-AUC ≥ 0.83).

### 1.4 Transfer prediction (§4 — the killer capability)

Given base pool B, target T, candidate pool C: predict which C_i gives the largest transfer gain `Δ(C_i, T) = Acc(B∪C_i, T) − Acc(B, T)`. The predictor ranks candidates using a geometric score over MAG operators × metrics.

**Headline numbers:**
- Raw centroid cosine (the standard baseline): ρ ∈ [0.01, 0.05] on 6/8 metrics — **essentially uninformative**.
- MAG operators Y3/Y5/Y8: mean ρ = 0.33.
- Best class-conditional triple {Y3+cos_ben, Y5+cos_ben, Y8+cos_mal}: **94.7% Top-1, 100% Top-2** on 19 valid shuffles.
- Best full-coverage pair {Y2+cos, Y4+CKA}: 62.0% Top-1, 78.0% Top-2 (random = 16.7%).
- Centroids stabilize at K ≥ 64 prompts → practical with small unlabelled samples.

The top-six triples all share the structure `{cos_ben, cos_ben, cos_mal}` — complementary class-conditional geometry, not a single dominant feature.

---

## 2. Distillation

### 2.1 The transferable primitive (stripped of the paper's LLM setting)

The paper operates on 4096-dim LLM residual streams with natural-language prefixes. The transferable insight is **modelless and substrate-agnostic**: any system with a readout function `m(x) → ℝ^d` and a transformation `Q` can run MAG. The math is:

1. **Mean-shift direction:** `v_Q = mean(with_Q) − mean(without_Q)`, normalized.
2. **Contrast direction:** `u_Q = mean(negative_class) − mean(positive_class)`, normalized — where the classes come from the model/runtime's own verdict, not human labels.
3. **Linearity diagnostic:** `ϵ_Q` tells you whether a single direction suffices (ϵ ≈ 0) or the feature is entangled (ϵ ≈ 1 → non-steerable).
4. **Calibrated injection:** `α(τ) = τ · ‖A_prefix‖ / ‖d‖` makes steering strength a fraction of the activation magnitude — model- and direction-invariant.
5. **Transfer ranking:** class-conditional centroid cosine on MAG operators predicts which candidate dataset/experience best improves a target — orders of magnitude more informative than raw cosine.

### 2.2 Where the pieces already live (the fusion map)

| Piece | Existing location | Relationship |
|-------|-------------------|--------------|
| Direction injection | `apply_latent_steering` (R290/Plan 309), `spherical_steering` (Plan 405), `subspace_steering` (Plan 412) | **MAG feeds these** — mines the directions they inject |
| Supervised direction extraction | `EmotionDirections::extract_direction` (P162), civ_emotion mean-difference (R196) | **MAG is the unsupervised sibling** — same mean-difference math, label from `y_M` not human annotation |
| Direction-consuming composition | `PersonalityWeightedComposition` (P297), `CommittedFieldBlend` (P321/FAME) | **MAG mines the archetype/personality directions** they blend |
| Class-mean contrast | `EmotionDirections` (P162) uses mean-difference on labeled data | **MAG generalizes to model-self-labeled data** |
| BLAKE3 commitment | `MerkleFrozenEnvelope` (riir-neuron-db/freeze.rs) | **MAG directions are frozen artifacts** — same envelope |
| Curiosity / exploration | CGSP (R126/P299), Curiosity Pulse (R041) | **MAG transfer prediction = modelless curiosity signal** — predicts which experiences teach the most |
| External escalation | AnyRAG gateway (riir-neuron-db/gateway.rs) | **MAG transfer ranking = which source to escalate to** |
| Consolidation | Raven/δ-Mem (riir-neuron-db/consolidation.rs) | **MAG transfer prediction ranks which shards/experiences to consolidate** |

**Nothing here is new math.** The mean-difference is identical to EmotionDirections. What is new: (a) the **unsupervised label source** (`y_M` — the model/runtime's own verdict), (b) the **operator family** that isolates different parts of the shift, (c) the **linearity diagnostic** `ϵ_Q` that predicts steerability, and (d) the **transfer-prediction** application.

### 2.3 Closest cousins (3)

1. **EmotionDirections (R144/Plan 162)** — closest mechanism (mean-difference contrast direction). Differs critically in the **label source**: EmotionDirections needs human-authored emotion stories; MAG uses the model's own verdict `y_M`. MAG is the unsupervised generalization.
2. **Latent Field Steering (R290/Plan 309)** — the WRITE side (injection). MAG is the READ side (mining). Together they close the loop: MAG mines → Latent Field Steering injects.
3. **Jacobian Lens (R388/Plan 409)** — SVD-based concept readout. Different mechanism (corpus-averaged Jacobian SVD vs activation delta), refuted as a FaithfulnessProbe prefilter. MAG is simpler (no SVD) and has the transfer-prediction capability the J-lens lacks.

### 2.4 Fusion (the Super-GOAT angle)

**F1 (PRIMARY — riir-ai): MAG × Latent Field Steering × EmotionDirections — the unsupervised acquisition loop.**
Today: designer authors direction vectors (R290) OR supervised extraction from labeled stories (P162). With MAG: the NPC's own runtime verdicts mine direction vectors unsupervised. The loop: NPC acts → MAG mines `v_Q` from verdict-conditioned activation shifts → direction frozen as `MerkleFrozenEnvelope` → Latent Field Steering injects it on future ticks. **NPCs discover their own reasoning directions from experience.** Connects R290 + R144 + P162 + P297 + P321 + P405 (≥6 systems).

**F2 (SECONDARY — riir-ai): MAG transfer prediction × CGSP curiosity × AnyRAG escalation.**
The §4 experiment — predict which dataset/experience transfers best — is a modelless curiosity/exploration signal. An NPC with a target skill T asks: "which of these candidate experiences will teach me the most about T?" MAG geometry answers modellessly (94.7% Top-1 on the paper's task). This is the missing *directed curiosity* signal: not "what's novel?" (entropy) but "what transfers to my goal?" (MAG geometry). Connects R126 (CGSP) + R041 (Curiosity Pulse) + AnyRAG gateway.

**F3 (TERTIARY — katgpt-rs): MAG × CommittedFieldBlend × PersonalityWeightedComposition — unsupervised archetype discovery.**
CommittedFieldBlend (P321) blends K archetype dynamics fields with frozen weights π. Currently archetypes are designer-authored or trained. MAG mines archetype directions unsupervised from the model's verdicts under different "context prefixes" (e.g., "is this a combat situation?" → mine the combat-archetype direction). Connects P321 + P297 + R302.

**F4 (QUATERNARY — riir-neuron-db): MAG transfer prediction × Raven/δ-Mem consolidation.**
Consolidation picks which wake-events to freeze into a shard. MAG transfer prediction ranks which candidate experiences best improve a target capability — the consolidation selector becomes transfer-aware, not just diversity-aware (TEMP, P341) or flatness-aware (can_freeze). Connects riir-neuron-db consolidation + AnyRAG.

---

## 3. Verdict

### Tier: **Super-GOAT**

| Question | Answer | Notes |
|----------|--------|-------|
| Q1 No prior art? | **YES.** Grep for `prefix.*(delta\|shift)`, `direction.*(mining\|extract\|unsupervised)`, `model.verdict`, `self.label`, `transfer.predict` across all 5 repos: all direction extraction is SUPERVISED mean-difference (EmotionDirections P162 on labeled stories; KG Latent Octree R196 on labeled triples). No unsupervised prefix-delta mining ships. No transfer-prediction primitive ships (AnyRAG uses raw cosine, which the paper proves is ~uninformative ρ∈[0.01,0.05]). Vocabulary translation done: "activation shift"→"direction vector", "model verdict"→"claim verifier"/"CLR vote", "transfer prediction"→"curiosity"/"AnyRAG escalation". | The injection side (R290/309) ships; the acquisition side (this) does not. |
| Q2 New class of behavior? | **YES.** (a) Unsupervised direction mining — NPCs discover reasoning directions from their own verdicts without labeled data. (b) Modelless transfer prediction — predict which experiences best improve a target capability. Neither exists today; both are new capabilities, not optimizations. | |
| Q3 Product selling point? | **YES.** "Our NPCs mine their own reasoning directions from runtime experience, and predict which experiences will teach them the most — no labeled training data, no gradient descent." Concrete, differentiated, demoable. | |
| Q4 Force multiplier? | **YES.** The missing acquisition step for ≥6 direction-consuming systems: Latent Field Steering (R290), EmotionDirections (P162), PersonalityWeightedComposition (P297), CommittedFieldBlend (P321), Spherical Steering (P405), KARC (P308). Plus curiosity (R126) + AnyRAG + consolidation. | |

**Selling point:** NPCs discover their own reasoning directions from runtime verdicts and predict which experiences transfer best — closing the acquisition gap in the direction-vector ecosystem. No labeled data, no training.

**Not Super-GOAT if:** G2 (contrast direction separates model-self-labeled classes at ≥threshold accuracy on a controlled toy) fails — if model-self-labels produce non-separable directions, the unsupervised acquisition is no better than random and demotes to Gain (research-only).

### MOAT gate (per domain, §1.6)

| Domain | Verdict | Reasoning |
|--------|---------|-----------|
| **katgpt-rs** (public engine) | **IN SCOPE — strengthens moat.** Fundamental modelless primitive (direction mining + transfer prediction math). Paper-derived base-foundation primitive that passes via fusion. Ships behind feature flag with GOAT gate. | Generic math, no game/chain/shard semantics. |
| **riir-ai** (private runtime) | **IN SCOPE — pillar-level.** Connects to ≥2 pillars: NPC Dialog (P3 — direction-conditioned dialog), Reasoning Pack (P8 — unsupervised reasoning direction mining), Self-Learn NPCs (CGSP curiosity). The per-NPC selling point lives here. | The F1+F2 fusions are pillar amplifiers. |
| **riir-neuron-db** | Cross-ref only (F4 consolidation angle). Not a primary target — no separate guide. | Consolidation consumer, not a shard primitive. |

### Mandatory outputs (created this session)

1. **Open primitive** → `katgpt-rs/.plans/418_mag_activation_geometry_primitive.md` (feature flag `mag_mining`).
2. **Private guide** → `riir-ai/.research/316_mag_unsupervised_direction_mining_guide.md` (selling-point doc).
3. **This note** → `katgpt-rs/.research/397_Mining_via_Activation_Geometry.md`.

---

## 4. Constraints check

| Constraint | Status |
|------------|--------|
| Modelless / inference-time | ✅ No training, no gradients. The "label" is the model/runtime's own verdict `y_M` — a runtime observation, not a training target. Transfer prediction is pure geometry (centroid cosine on MAG operators). |
| Latent-to-latent preferred | ✅ Operates entirely in activation/latent space. Direction mining is `mean(with_Q) − mean(without_Q)` on latent readouts. Never decodes to tokens. |
| Use sigmoid not softmax | ✅ Steering calibration `α(τ)` is a scalar fraction; the paper's class-mean direction is used with additive injection (same as Latent Field Steering R290). No softmax anywhere. |
| Freeze/thaw over fine-tuning | ✅ Mined directions are frozen as `MerkleFrozenEnvelope` artifacts. Atomic Arc swap for direction hot-reload. No weight mutation. |
| 5-repo discipline | ✅ Open primitive (math) → katgpt-rs. Game integration (per-NPC wiring) → riir-ai. No chain/shard IP in the open primitive. |
| Raw scalars at sync boundary | ✅ The mined direction vector stays latent (local to entity). Only the resulting scalar projections cross sync (same boundary discipline as HLA). The transfer-prediction score is a local scalar. |
| Zero-alloc hot path | ✅ Direction mining is offline (consolidation/sleep-cycle tier). Transfer scoring is `O(N·d)` centroid cosine — SIMD-able, pre-allocated. |

---

## 5. Open questions / risks

1. **Does the unsupervised label `y_M` produce separable directions on our substrate?** The paper validates on LLM residual streams. Our HLA is 8-dim; the readout is different. **Mitigation:** G2 gate measures LOO/LODO accuracy of the contrast direction on a controlled toy; gate requires ≥ threshold. If HLA's 8 dims are too low-rank, the primitive may need a higher-dim readout (latent_functor state, NeuronShard style_weights[64]).
2. **Is the transfer-prediction result an artifact of the paper's specific 18-dataset PI corpus?** The paper itself flags this (§5 limitations: "best combinations selected and evaluated on the same shuffles"). **Mitigation:** the open primitive's G4 gate runs on a controlled synthetic transfer task, not the paper's corpus. The selling point claim is hedged until independent validation.
3. **Compute-unit translation (the R368 lesson).** The paper's "model verdict `y_M`" is an LLM yes/no answer. Our runtime's "verdict" is an NPC action selection / claim verification / CLR vote. The translation: `y_M` = any binary runtime observable (did the NPC succeed? did the claim pass the rubric? did the action hit the target?). The MAG mechanism is the same; the verdict source differs. **Mitigation:** the open primitive is generic over the verdict source (host-supplied `&[bool]` labels).
4. **PI-direction non-steerability (ϵ_Q ≈ 1).** The paper found the prompt-injection direction does NOT steer (entrenched verdict). This is a feature, not a bug — `ϵ_Q` predicts steerability. For our substrate: some NPC reasoning features may be entrenched (non-steerable) and MAG's `ϵ_Q` correctly diagnoses this. No mitigation needed; it's the diagnostic working as designed.
5. **Operator selection.** The paper's 8 operators are not equally useful (Interaction/Verdict are near-zero on average; Prefixed/InputDelta/Answered/FewShot carry the signal). The open primitive ships all 8 but the GOAT gate focuses on the 3-4 load-bearing operators.

---

## TL;DR

MAG is a modelless, unsupervised direction-mining framework: prepend a fixed transformation Q, measure `m(Q‖p) − m(p)`, extract the mean shift as a feature direction using the model's own verdict as the label. It is the **missing acquisition step** for our direction-vector ecosystem — today every direction is designer-authored (R290) or supervised-extracted (P162). MAG mines them unsupervised from runtime verdicts. Plus the §4 transfer-prediction experiment is a genuinely new capability (modelless "which experience teaches the most" — 94.7% Top-1 vs raw cosine's ρ≈0.03). Super-GOAT via F1 fusion (MAG × Latent Field Steering × EmotionDirections): NPCs discover their own reasoning directions from experience, freeze them, and inject them on future ticks. Open primitive in katgpt-rs (Plan 418, feature `mag_mining`); private guide at riir-ai/.research/316. Kills itself if G2 (model-self-labeled contrast direction separability) fails.
