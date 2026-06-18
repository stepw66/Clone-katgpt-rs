# Research 267: FPCG — Future Probe Controlled Generation (Detection ≠ Prediction Features)

> **Source:** [Predicting Future Behaviors in Reasoning Models Enables Better Steering](https://openreview.net/forum?id=48NnVTsirb) — Kortukov, Komorowski, Klein, Engl, Sarti, Oh, Lapuschkin, Samek, NeurIPS 2026 / Mech Interp Workshop at ICML 2026
> **Repo:** <https://github.com/kortukov/future_probes> (uv-managed Python; reference implementation of probes + FPCG loop)
> **Date:** 2026-06-18
> **Status:** Active — **GOAT** (provable gain over activation steering on quality preservation; not a new capability class)
> **Related Research:** 144 (Functional Emotions / Emotion Vector — *detection* sibling), 053 (CNA — *detection* sibling), 211 (PosteriorGuided Pruner — precision vector cousin), 215 (Self-Revising Discovery), 240 (CGSP — candidate-sampling + Guide scoring cousin), 243 (Bebop — linear α forecast cousin), 244 (FaithfulnessProbe — behavioral delta cousin)
> **Related Plans:** 162 (Emotion Vector Inference), 087 (CNA Steering), 239 (Posterior-Guided Pruner), 277 (Temporal Derivative), 274 (CGSP)
> **Related Issues:** 023 (Adaptive γ from entropy-linear acceptance forecast — closest cousin in spirit)
> **Classification:** Public — generic math, no game semantics

---

## TL;DR

The paper's central conceptual contribution is the **separation of detection features from prediction features** in LLM activations. Standard activation steering (difference-in-means, CAA, CNA) operates on **detection features** — what behavior already exists in the text. The paper shows these are *poor predictors* of what the model will do next. There is a separate, *distinctly encoded* set of **prediction features** in the residual stream that linearly forecast the probability a given behavior will appear in the future output (64%–91% binarized accuracy on 6 behaviors × 4 models, MAE 0.10–0.32).

Building on this, FPCG (Future Probe Controlled Generation) is a sentence-level candidate-sampling steering algorithm: sample M candidate next-sentences, score each by an MLP probe on its intermediate-layer activation, pick the argmax/min. The probe is a future-behavior forecast trained offline on resampling data. FPCG preserves output quality (perplexity barely moves, <10% format-filtered) where difference-in-means steering breaks outputs entirely on the same behaviors — and enables steering on 3 model×behavior combos where activation steering fails outright.

**Distilled for katgpt-rs (modelless, inference-time):**

1. **The "two features" frame** sharpens what we already ship: `EmotionDirections::project` (Plan 162) and CNA (Plan 087) are **detection-side** primitives; nothing currently computes a **prediction-side** future-behavior forecast from intermediate activations. This is a real gap, not a rename.
2. **The FPCG algorithm pattern** = sample-M-candidates → score-via-projection → argmax-select. We already have this shape three times: CGSP's Conjecturer+Guide (`crates/katgpt-core/src/cgsp/`), CompressionDrafter beam search (`riir-games/.../compression_draft.rs`, GOAT FAILED), and best-buddies speculative drafting (`Plan 199`). FPCG is the same skeleton applied to a *behavior forecast probe* instead of a quality/compression/correspondence score.
3. **The proven linear forecast insight** rhymes with Research 243 / Issue 023: a simple linear model on a cheap signal (entropy in 243; mid-layer activation projected on a direction vector here) forecasts a future metric with remarkable accuracy. FPCG generalizes the Issue 023 *acceptance* forecast to a *behavior* forecast.

---

## 1. Paper Core Findings

### 1.1 Detection vs Prediction features — the conceptual core

| Feature class | What it represents | How it's extracted | Used by |
|---|---|---|---|
| **Detection** | Behavior already realized in the text | Mean activation difference over contrastive final-answer pairs (Rimsky et al. 2024) | CAA, difference-in-means steering, CNA (Nous), our `EmotionDirections`, our `cna_discover` |
| **Prediction** | Probability the model will exhibit behavior B in the future output, *given current prefix* | Linear/MLP probe trained on (intermediate-step activations, future-behavior-probability labels gathered by resampling) | This paper's FPCG |

**Empirical evidence they're different (Fig. 5):** A probe trained on detection features (final-answer activations only) predicts future behavior with MAE ~0.30–0.40 and binarized accuracy 50–75% on early-CoT activations. The same probe class trained on prediction features (all-sentence activations) achieves MAE ~0.10–0.20 and accuracy 75–93% on the same early-CoT activations. The gap shrinks as you approach the final answer (where detection becomes prediction by definition) but is large in the first half of the CoT.

**Practical implication:** If you want to *steer* a model towards/away from a future behavior, intervening on detection features (the standard recipe) targets the wrong subspace. The model's *intentions* live elsewhere.

### 1.2 Behavior distribution dynamics (§3.2)

Per prompt, the paper samples S=10 base responses, splits each into sentences, and at each prefix re-samples M=10 completions to measure `B̄(p_{i ← r_{j:k}})` — the empirical future-behavior probability after the k-th sentence. Two findings:

- 23–88% of dataset prompts are "behaviorally uncertain" — the model has not yet decided.
- A single sentence ("Okay, so I need to…" vs "Alright, so I need to") can shift future-behavior probability from 40% → 70%.

This is the **ground truth** the prediction probe learns to estimate without resampling.

### 1.3 Probe accuracy (§3.3.1, Fig. 4)

Linear probes (logistic regression on a single layer's residual-stream activation at sentence-end position) achieve:

| Behavior class | MAE range | Binarized accuracy range |
|---|---|---|
| MCQ behaviors (Myopic/Wealth/Survival) | 0.17–0.32 | 64–86% |
| Free-form behaviors (Refusal/PromptInjection/Sycophancy) | 0.10–0.20 | 76–91% |

MLP probes (1 hidden layer, dim 1024) are only marginally better — most of the signal is linear. (Rhymes with our sigmoid-margin rule and the linear representational claim from Research 144.)

### 1.4 FPCG algorithm (§4.1, Fig. 6)

```
def fpcg(model, prompt, future_probe, layer, num_candidates, direction):
    response = ""
    while not finished(response):
        candidates = generate_sentence_candidates(model, prompt + response, num_candidates)
        for c in candidates:
            acts = extract_activations(model, prompt + response + c, layer)
            c.score = future_probe(acts)
        best = argmax(c.scores) if direction == "positive" else argmin(c.scores)
        response += best
    return response
```

Three things to note for our stack:

1. **Sentence-level atomicity** (not token-level). The probe is applied once per generated sentence. The atomic unit matters: it matches the paper's CoT-understanding pretense (Bogdan 2025, Macar 2026) that sentences, not tokens, are where decisions crystallize.
2. **No residual stream modification.** Unlike CAA, FPCG never adds a vector to activations. It only *selects* among already-generated candidates. This is why output quality is preserved — the model is never pushed off-manifold.
3. **`num_candidates` is the quality/steering-strength knob.** Paper Fig. 25 ablation: 2 candidates already gives meaningful steering; 10–15 saturates. Each candidate costs one extra forward pass up to `layer`.

### 1.5 FPCG vs Activation Steering (§4.2, Figs. 7–8, Tables 1, D.2)

| Model | FPCG wins | ActSteer wins | Quality tie | Cases only FPCG works |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-8B | 6/6 (all) | 0 | — | 0 |
| Qwen3-14B | 5/6 | 0 | 1 (Survival) | 0 |
| gpt-oss-20b | 3/6 (strength) | 3/6 (strength) | — | Refusal, PromptInj, Wealth (ActSteer breaks outputs) |
| QwQ-32B | 1/6 (Survival) | 5/6 | — | 1 (Survival) |

Key metric: **format-filtered rate** (proxy for output degradation). ActSteer with multipliers large enough to materially shift behavior filters 10–100% of outputs (model produces gibberish or wrong format). FPCG filters <10% in nearly all settings. Perplexity tells the same story: ActSteer raises PPL in 9/12 scenarios, FPCG in 1/12.

**The complementarity is the headline.** FPCG is not strictly dominant — it is weaker in raw steering strength on QwQ-32B — but it works where ActSteer catastrophically fails, with much better quality preservation. Hybrid is open future work (paper §5).

---

## 2. Distillation

### 2.1 What's already in katgpt-rs (verified by code grep)

| Paper mechanism | Our shipped equivalent | Class | Status |
|---|---|---|---|
| Difference-in-means steering vector | CNA contrastive neuron discovery (`Plan 087`, `crates/katgpt-core/src/...`, `BomberContrastivePairs`, `GoContrastivePairs`) | **Detection** | ✅ shipped, GOAT 4/4 |
| Linear projection on residual stream for behavior | `EmotionDirections::project` (`Plan 162`, `src/pruners/emotion_vector.rs`), valence/arousal/desperation/calm | **Detection** | ✅ shipped, default-on |
| Sentence-level CoT as decision unit | `AdaptiveTraceCompactor::observe_entropy` (`src/attn_match/adaptive_cot.rs:159`), `RejectionReason::EntropySpike` (`src/distill/trd.rs:56`) | Both | ✅ shipped (entropy, not behavior) |
| Causal-intervention behavioral delta | `FaithfulnessProbe` (`Plan 278`, `crates/katgpt-core/src/cgsp/dual_pool.rs:1868`), `behavior_delta` trait method | Detection (intervention) | ✅ shipped |
| Sample-M-candidates → score → argmax-select | CGSP Conjecturer+Guide loop (`crates/katgpt-core/src/cgsp/loop_.rs`); CompressionDrafter beam (`riir-games/src/quest_grammar/compression_draft.rs`) | Mechanism skeleton | ✅ shipped (different domains) |
| Linear forecast from cheap signal | Issue 023 `AcceptanceForecast { a, b }` (`α ≈ a − b·H(p)`, Research 243 §2.3) | Forecast | 📋 planned, not yet implemented |
| Bayesian precision vector for arm lifecycle | `PosteriorGuidedPruner` (`Plan 239`, `src/pruners/posterior/wrapper.rs`) | Posterior (not future) | ✅ shipped, default-on |
| Self-revising discovery on regime shift | Regime-Transition Inference (`Plan 215`, `src/pruners/regime_transition.rs`) | Regime detection | ✅ shipped, default-on |
| Per-NPC recurrent belief kernel | `evolve_hla` (`crates/katgpt-core/src/sense/reconstruction.rs`, Research 242); `LeakyIntegrator` in micro_belief (Plan 276) | Belief state | ✅ shipped (no behavior-forecast framing) |

### 2.2 What's NOT in katgpt-rs (the gap)

1. **Future-behavior probability probe.** No primitive computes "given this activation, what is the probability the model will exhibit behavior B in its future output?" The closest is `EmotionDirections::project`, but that returns the *current* emotion projection, not a future-behavior probability. The probe requires (a) a direction vector trained against future-outcome labels (gathered via resampling), and (b) a sigmoid readout returning a probability, not a signed activation.
2. **Detection-vs-prediction vocabulary in the trait stack.** Our `ScreeningPruner::relevance()` returns a scalar with no tag distinguishing "this pruner reads detection features" vs "this pruner reads prediction features". The paper's framing would let us mark which primitives are safe for steering (prediction-side, won't push off-manifold) vs which are detection-side (good for *monitoring*, risky for *intervention*).
3. **Sentence-atomic candidate selection.** Our existing candidate samplers operate at action/subgoal/compression-token granularity. We have no sampler whose atomic unit is "the model's next utterance span ending in a sentence terminator", scored by a probe on the post-span residual. This is a new shape.
4. **Non-invasive steering primitive.** Every steering-shaped primitive we ship (CNA modulation, Emotion Vector desperation monitor) either writes to the residual stream or signals downstream consumers that mutate behavior. FPCG's "select among already-generated candidates without modifying activations" is a *read-only-at-the-LLM-level* steering shape — the intervention is at the *sample selector*, not the residual stream.

### 2.3 Fusion

**Fusion: FPCG × EmotionDirections × Issue 023 AcceptanceForecast × CGSP.**

The three-way fusion produces a thing none of the parts has alone: a **calibrated, manifold-preserving future-behavior steering loop** that requires zero weight updates and zero residual-stream writes. The recipe:

```
                                  ┌─→ (detection, read-only)      EmotionDirections::project / CNA
activation[mid_layer, sent_end] ──┼─→ (forecast, read-only)       FutureProbe::sigmoid(dot(act, w_B))
                                  └─→ (posterior, update)         PosteriorGuidedPruner::record_evidence

sentence_candidates[M] ──► score each by FutureProbe ──► argmax/min ──► append to response
                                                  │
                                                  └─► (this is CGSP's Conjecturer→Guide skeleton,
                                                       applied at utterance granularity with a
                                                       forecast probe instead of a quality rubric)
```

**What the fusion adds that no part has alone:**

- **EmotionDirections (R144)** gives us the linear projection math and the zero-alloc dot-product hot path. It currently has no notion of "this is a *future* forecast, treat differently from a *current* reading."
- **Issue 023 AcceptanceForecast** gives us the precedent that a 2-parameter linear model on a cheap signal can forecast a future metric with calibrated accuracy. FPCG generalizes this from `α ≈ a − b·H(p)` (forecast of acceptance from entropy) to `P(B) ≈ σ(w·act + b)` (forecast of behavior from activation).
- **CGSP loop skeleton (R240)** gives us the candidate-sampler + score + select machinery at subgoal granularity. Re-targeting it to sentence granularity with a future-behavior probe is mechanical.
- **PosteriorGuidedPruner (R211)** gives us the precision-tracking primitive for *when to trust the probe*: a freshly-trained probe on a new behavior has low precision; the PGP decorator would gate FPCG's selection by probe-confidence, falling back to unsteered generation when the probe's posterior is wide.

**Cousins in the corpus this fuses with (cross-references):**

- **Research 144 + Plan 162 (Emotion Vector)** — the projection primitive, promoted from "current-state read" to "future-state forecast".
- **Research 243 + Issue 023 (Bebop / AcceptanceForecast)** — the precedent that linear-forecast-from-cheap-signal is a profitable pattern. We should land Issue 023 first; FPCG generalizes it.
- **Research 240 + Plan 274 (CGSP)** — the candidate-sample-score-select skeleton at utterance granularity.
- **Research 053 + Plan 087 (CNA)** — the detection-side counterpart. FPCG and CNA are complementary: CNA for monitoring/attribute-extraction, FPCG for non-invasive steering.
- **Research 211 + Plan 239 (PosteriorGuided)** — confidence gating on the probe.
- **Research 244 + Plan 278 (FaithfulnessProbe)** — `behavior_delta` semantics. The probe's forecast *is* a behavioral delta estimate.
- **Research 215 + Plan 215 (Regime Transition)** — when FPCG's probe indicates the model is about to "tip" into an undesired behavior, regime-transition machinery can be triggered.
- **Plan 277 (Temporal Derivative)** — the dual fast/slow surprise kernel could feed an online drift detector for when the probe's calibration itself goes stale.

**What's novel in the fusion:** The detection-vs-prediction **vocabulary** in the trait stack. Today every primitive that reads activations is implicitly treated as detection-side. Marking some primitives as forecast-side (with the implication: "safe to steer on, won't push off-manifold") is a small architectural change with large downstream consequences — it lets us build non-invasive steering on top of the existing read path.

**What's NOT novel:** Linear projection on activations (shipped), candidate-sample-score-select (shipped), sigmoid readout (shipped), per-NPC belief kernels (shipped), probe-trained-on-labels (a 2-line generalization of `EmotionDirections::new`). The contribution is the *recipe* and the *vocabulary distinction*, not any single primitive.

---

## 3. Verdict

### Decision: **GOAT** — Plan + implement, feature-flagged, benchmarked

**One-line reasoning:** Provable gain (output-quality preservation + steering-in-cases-where-actsteer-fails) over our existing detection-side primitives (CNA, Emotion Vector), achieved by combining three existing mechanisms (linear projection, candidate-score-select, calibrated linear forecast) at a new abstraction layer (sentence-atomic future-behavior probe). Not Super-GOAT: every component has a strong cousin already shipped, no new capability class is created (both methods achieve steering — FPCG just does it without breaking the model).

### Novelty gate (4 questions)

| Gate | Question | Answer |
|---|---|---|
| **No prior art?** | Does any existing note/plan/code cover FPCG's mechanism? | **NO — but partial.** The *forecast-from-cheap-signal* pattern is documented (Issue 023). The *candidate-sample-score-select* skeleton ships (CGSP). The *linear projection on activations* ships (Emotion Vector, CNA). The specific combination — *future-behavior-probability* probe (not acceptance, not emotion, not quality) scored over *sentence-level* candidates with *non-invasive* selection — is not in any of `katgpt-rs/.research/`, `katgpt-rs/.plans/`, `riir-ai/.research/`, `riir-ai/.plans/`, or shipped code (grepped `future_probe`, `FPCG`, `detection_feature`, `prediction_feature`, `candidate_sentence`, `behavior_probe`, `intent_probe`, `future_intent` — zero hits in notes; zero hits in code; `behavior_delta` exists in FaithfulnessProbe but as an intervention diagnostic, not a forecast). Closest is Issue 023 + R243 (acceptance forecast) + R144 (emotion vector) — three separate primitives. |
| **New class of behavior?** | New capability, not just better numbers? | **NO.** The capability is "steer model towards/away from behavior" — we already do this (CNA modulation, Emotion Vector desperation monitor). FPCG does it with better quality preservation and in cases where activation steering fails. That is a *better number*, not a new capability. |
| **Product selling point?** | "Our system does X that no competitor can"? | **Partial.** "Our modelless inference stack can steer LRM behavior at utterance granularity without degrading output quality, using a future-behavior probe rather than a detection-side steering vector." Sellable, but the LLM-steering space is crowded (CAA, SAE-steering, attribution-guided decoding, GeDi, FUDGE). The *game-AI* framing — "NPCs whose future-dialogue behavior is steerable by sampling and probing, not by mutating their residual stream" — is more differentiated but speculative (no shipped game integration). |
| **Force multiplier?** | Connects to ≥2 existing pillars? | **YES.** Connects Emotion Vector × CNA × PosteriorGuided × CGSP × Issue 023 × FaithfulnessProbe × Plan 277 (≥7 cousins). |

**3/4 PASS, 1/4 (new-class) FAIL → GOAT, not Super-GOAT.** No riir-ai guide created (that's reserved for Super-GOAT). Plan to `katgpt-rs/.plans/` only.

### Why not Super-GOAT

The detection-vs-prediction distinction is genuinely clarifying for our trait stack, but in our system the *practical* difference is smaller than in the paper:

1. Our `EmotionDirections` is already read-only and zero-cost; we don't ship a residual-stream-mutating steering vector for emotions on the hot path. So the "FPCG preserves quality where activation steering breaks it" advantage is less pronounced against *our* baseline than against vanilla CAA.
2. We have no deployed setting where we currently try to steer LLM behavior and break outputs. The paper's wins are against a baseline we don't ship.
3. The mechanism skeleton (sample-score-select) is shared with CGSP, CompressionDrafter, and best-buddies drafting. FPCG is a new *target* for an old *pattern*.

The honest framing: this paper is a **quality bar for our detection-side primitives** (it tells us they're detection-side, and that matters) and a **recipe for adding a prediction-side sibling**. Both are valuable. Neither is a moat.

### Why not Gain

The vocabulary distinction (detection vs prediction feature tagging in the trait stack) is more than incremental — it clarifies which of our shipped primitives are safe for *intervention* vs only *monitoring*. That architectural clarification alone is worth a plan, independent of the FPCG algorithm itself.

---

## 4. Distillation to riir-ai (private game runtime)

The game-side application is **deferred** pending katgpt-rs GOAT proof. The sketch:

- **NPC dialogue steering without voice breakage**: NPCs in MMORPG-scale game AI have emergent personalities (per-NPC HLA emotion vectors, freeze/thaw snapshots). FPCG-style sentence-level steering would let a quest director nudge an NPC's dialogue trajectory towards a desired behavior (refuse a bribe, offer a hint) without breaking the NPC's voice — the FPCG paper's perplexity-preservation result directly translates to "NPC stays in character".
- **Latent-side only**: the future-behavior probe operates entirely on latent activations (mid-layer residual). The scalar probability `σ(w·act + b)` is the only thing that crosses to the steering selector. **No full embedding crosses the sync boundary** (per `AGENTS.md` latent-vs-raw rules).
- **Semantic domain only**: NPC personality, mood, dialogue tendency → latent. NPC position, HP, wallet → raw, untouched. The probe operates exclusively on the semantic domain.

This is a future `riir-ai/.research/` guide **only if** the katgpt-rs plan ships and GOAT-proves a quality-preservation gain on a real model. Until then it stays in katgpt-rs as a generic primitive.

---

## 5. Plan sketch (to be elaborated in `.plans/267_*.md`)

- **Phase 1 — Vocabulary in the trait stack.** Add `FeatureClass::{Detection, Prediction}` tag to `ScreeningPruner`. Mark `EmotionDirections`, CNA, FaithfulnessProbe as `Detection`. Reserve `Prediction` slot. Zero-cost enum tag.
- **Phase 2 — FutureBehaviorProbe primitive.** New struct in `src/pruners/future_probe.rs`: holds `w_B: Vec<f32>` direction + bias. Method `forecast(&[f32]) -> f32` returning sigmoid probability. Trained offline (Python script following Kortukov's resampling recipe) → loaded at init. **No online training.** Direction vector is a frozen artifact (freeze/thaw-compatible).
- **Phase 3 — SentenceCandidateSelector.** Sampler that produces M candidate next-utterance-spans (sentence-terminated) and selects argmax/min by probe score. Reuses CGSP's Conjecturer trait shape.
- **Phase 4 — GOAT gate.** Benchmark vs (a) unsteered baseline (perplexity parity, <5% behavior shift), (b) Emotion Vector desperation monitor (behavior shift ≥ 30pp at <5% perplexity increase), (c) CNA modulation (format-filter rate ≤ 10% where CNA filters >30%). Synthetic test on a small model.
- **Phase 5 — Promotion / demotion.** If GOAT passes: promote to default-on for the probe primitive (opt-in for the selector loop, since it costs M forward passes). If GOAT fails on quality: demote to opt-in, keep only the vocabulary tag (Phase 1) as the shippable output.

**Feature flag:** `future_probe` (Phase 2+), `fpcg_selector` (Phase 3+).

**GOAT gate rule:** the headline metric is **perplexity-vs-steering-strength Pareto frontier** vs `EmotionDirections` projection-based steering. FPCG must dominate on at least one behavior class to promote.

---

## 6. References

- **Paper:** Kortukov et al., "Predicting Future Behaviors in Reasoning Models Enables Better Steering", NeurIPS 2026 / Mech Interp Workshop at ICML 2026. <https://openreview.net/forum?id=48NnVTsirb>
- **Reference implementation:** <https://github.com/kortukov/future_probes> (uv, Python). Files of interest: `behavior_distribution_analysis.py` (resampling label generation), `gather_activations.py` + `train_probe.py` + `evaluate_probe.py` (probe training pipeline), `future_probe_controlled_generation.py` (FPCG loop), `activation_steering.py` (baseline), `compute_steering_vector.py` (diff-of-means baseline).
- **Cousin research notes:**
  - `katgpt-rs/.research/144_Functional_Emotions_Linear_Representations_Behavior_Control.md` — emotion vectors (detection)
  - `katgpt-rs/.research/053_CNA_Contrastive_Neuron_Attribution.md` — CNA (detection)
  - `katgpt-rs/.research/211_Bayesian_Agent_Posterior_Guided_Skill_Evolution.md` — PosteriorGuided (precision vector cousin)
  - `katgpt-rs/.research/215_ECHO_Environment_Prediction_Inference_Time.md` — env prediction (prediction-side cousin, different target)
  - `katgpt-rs/.research/240_SGS_Curiosity_Guided_Self_Play.md` — CGSP (candidate-sample-score-select skeleton)
  - `katgpt-rs/.research/243_Bebop_Entropy_Bounded_MTP_Acceptance_Adaptive_Gamma.md` — linear forecast from cheap signal (closest in spirit)
  - `katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md` — FaithfulnessProbe (behavioral delta)
- **Cousin plans/issues:**
  - `katgpt-rs/.plans/162_*` (Emotion Vector), `087_*` (CNA), `239_*` (Posterior-Guided), `277_*` (Temporal Deriv), `274_*` (CGSP), `278_*` (FaithfulnessProbe)
  - `katgpt-rs/.issues/023_adaptive_gamma_from_entropy_forecast.md` — AcceptanceForecast (the linear-forecast precedent to land first)

---

## TL;DR

FPCG's headline is conceptual: **detection features ≠ prediction features** in LLM activations, and the standard steering recipe targets the wrong one. Operationally, FPCG is a sentence-level candidate-sampler scored by a future-behavior-probability probe — a pattern we already have three times in different domains (CGSP, CompressionDrafter, best-buddies drafting). The proven results (perplexity-preserving steering, works-where-activation-steering-fails) are a quality bar, not a new capability. **Verdict: GOAT.** Ship (1) a `FeatureClass::{Detection, Prediction}` vocabulary tag in the trait stack — small architectural clarification with large downstream consequences for which primitives are safe-to-steer-on — and (2) a `FutureBehaviorProbe` primitive + `SentenceCandidateSelector` behind feature flags, benchmarked vs our existing detection-side primitives on perplexity-vs-steering Pareto. **Not Super-GOAT** because every component has a strong shipped cousin; the contribution is the recipe and the vocabulary, not any single primitive. riir-ai game-side integration (NPC dialogue steering without voice breakage) is deferred pending katgpt-rs GOAT proof.
