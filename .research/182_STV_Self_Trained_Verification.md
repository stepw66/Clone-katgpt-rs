# Research 182: STV — Self-Trained Verification for Inference-Time Self-Improvement

**Date:** 2026-06-07
**Paper:** [Self-Trained Verification for Training- and Test-Time Self-Improvement](https://arxiv.org/abs/2605.30290) — Wu, Raghunathan (CMU)
**Code:** github.com/ar-forum/stv
**Status:** Active — Fusion Research
**Verdict:** GOAT — Reference-conditioned verification maps directly to ConstraintPruner + Episode DB; V-R loop is a natural DDTree extension; Episode-Guided Constraint Synthesis is a novel modelless fusion not in the paper.
**Cross-ref:** Research 172 (MUSE Skill Evolution), 175 (ThoughtFold), 107 (SynPruner), 007 (Screening Absolute Relevance), 156 (Speculative Reconciliation), 004 (LoRA Architecture)
**Commercial:** engine (MIT) = Episode-Guided ConstraintPruner, V-R DDTree loop, self-distilling pruner bandit; fuel (SaaS) = episode DB with reference solutions, constraint synthesis heuristics per domain

---

## Paper TL;DR

STV trains verifiers to provide better feedback for generator refinement without human annotation. The key insight: **diagnosis is easier with a reference solution**. A model that struggles to find flaws on its own can locate them when shown a reference. STV turns this asymmetry into supervision.

**Core contributions:**

1. **Reference-conditioned teacher verifier** (V*): Sees problem + candidate solution + correct solution → can locate specific errors. Never deployed at inference — used only for training signal.
2. **On-Policy Distillation (OPD)**: Student verifier V_θ (no reference) matches teacher V* (with reference) using α-divergence (Jensen-Shannon, α=0.5). SFT fails because student encounters prefixes never seen during training (off-policy drift).
3. **Verdict-RL**: Binary reward for correct accept/reject verdict. Combined with OPD: L_STV = L_OPD + λ·L_RL.
4. **Verification-Refinement (V-R) loop**: Generate solution → Verifier returns (verdict, feedback) → If reject, generator refines with feedback → Repeat up to R rounds.
5. **Verifier-in-the-Loop (ViL) training**: Train generator with RL inside the V-R loop using frozen STV verifier's feedback. Continues past RLVR plateau — 33% relative gain in pass@1 with verifier, 30% standalone.

**Key empirical results:**
- ~2× accuracy on hard math (Qwen3-8B, DAPO Hardest split)
- 14× on scientific reasoning (SciKnowEval: 1.5% → 21%)
- 33% gain past RLVR ceiling with ViL training
- 30% standalone gain without verifier at inference
- OPD beats SFT, RL-only, and meta-verifier baselines
- Qwen3-8B + STV verifier outperforms Qwen3-32B generator alone

---

## What's Training (Paper) vs. Inference (Ours)

| Paper Mechanism | Training-Time | Inference-Time (Modelless) |
|-----------------|--------------|---------------------------|
| Reference-conditioned teacher V* | ✅ LoRA weight update | ✅ **Episode DB as reference** |
| On-Policy Distillation (OPD) | ✅ LoRA weight update | ❌ N/A (requires gradient) |
| Verdict-RL | ✅ Policy gradient | ✅ **Bandit reward for pruner arms** |
| V-R loop (generate→verify→refine) | ✅ Multi-turn generation | ✅ **DDTree + ConstraintPruner loop** |
| Verifier feedback | ✅ Teacher-generated text | ✅ **CompilerFeedback + SynPruner** |
| ViL generator training | ✅ LoRA weight update | ❌ N/A (requires gradient) |
| Score calibration | ✅ RL reward signal | ✅ **Bandit Q-value normalization** |

---

## What We Already Have (~80%)

| STV Component | Our Analog | Status |
|---------------|-----------|--------|
| Generator | `SpeculativeGenerator` trait | ✅ Working |
| Verifier (accept/reject) | `ConstraintPruner::is_valid()` | ✅ Working |
| Verifier (relevance scoring) | `ScreeningPruner::relevance()` | ✅ Working |
| Reference solutions | Episode DB (anyrag) | ✅ Architecture exists |
| Feedback mechanism | `CompilerFeedback` from syn errors | ✅ Working |
| V-R loop | DDTree explores pruned branches | ✅ Partial — single-pass |
| Verdict signal | `InferenceResult.reward` | ✅ Working |
| Multi-round refinement | Speculative decoding + rejection | ✅ Partial |
| Domain validators | WASM validators (Bomber, Go) | ✅ Working |

**Gap (~20%):**
1. No iterative V-R loop (generate → verify → inject new constraints → re-generate)
2. No reference-conditioned constraint synthesis from episode DB
3. No bandit that learns pruner strategies from episode verification outcomes

---

## Fusion Ideas — Modelless (katgpt-rs)

### F1: Episode-Guided Constraint Synthesis (EGCS) 🔥

**Core idea:** When the episode DB has a reference solution for a similar problem, diff the candidate against the reference, extract structural patterns, and synthesize new constraints that guide the candidate toward the reference.

This is STV's "diagnosis is easier with reference" applied modellessly — no gradient, no training, pure algorithmic constraint injection.

**How it works:**
1. Generator produces candidate tokens via DDTree
2. ConstraintPruner filters invalid tokens (existing)
3. New: EpisodePruner looks up similar prompts in episode DB
4. If reference exists: structural diff → constraint synthesis
   - Example: reference has `map.get(key).copied().unwrap_or(default)`, candidate has `map[key]`
   - Synthesized constraint: "reject direct index access when `.get()` pattern exists in reference"
5. Inject synthesized constraints into the ConstraintPruner for this generation
6. Cache synthesized constraints by pattern hash for reuse

**Landing:** New `EpisodePruner` implementing `ConstraintPruner` trait. Wraps any inner pruner and adds episode-guided constraints. Uses anyrag for episode retrieval.

**Why it's modelless:** No weight updates. Pure algorithmic constraint synthesis from structural diffs. The "teacher" is the episode DB (reference solution), the "student" is the deployed ConstraintPruner.

**Expected gain:** 2-5× accuracy improvement on problems where episodes exist. Zero cost on novel problems (no episode → no synthesis → fallback to base pruner).

### F2: V-R Loop as Iterative DDTree Refinement

**Core idea:** Extend DDTree from single-pass to multi-round verification-refinement. After DDTree generates candidates, verify them, extract failure feedback, inject as new constraints, and re-generate.

**How it works:**
1. Round 0: DDTree generates candidates with ConstraintPruner
2. Verify candidates via CompilerFeedback (existing)
3. If all candidates fail: extract error patterns from compiler output
4. Inject error patterns as new constraints (dynamic constraint injection)
5. Round 1+: DDTree re-generates with augmented constraints
6. Repeat until a candidate passes or max rounds reached

**Landing:** New `VRLoop` struct wrapping `SpeculativeGenerator` + `SpeculativeVerifier`. Adds `max_rounds: usize` parameter. Each round injects new constraints from previous round's failures.

**Why it's modelless:** Uses existing CompilerFeedback mechanism. The "verifier feedback" is the compiler error message, which is already structured and actionable. The iterative loop is just DDTree re-invocation with new constraints.

**Expected gain:** 30-50% improvement on hard problems (where initial candidates fail but are "close"). Cost: 1-2 additional DDTree passes. Net positive because hard problems benefit disproportionately.

### F3: Self-Distilling Pruner Bandit

**Core idea:** A bandit that learns which pruner configurations work best, where the "teacher" signal comes from episodes (known-good outcomes). This is the modelless analog of STV's OPD.

**How it works:**
1. Multiple pruner configurations as bandit arms
2. Episode DB provides ground truth: "this prompt should produce this output"
3. After generation, compare actual output to episode's reference
4. If match: reward the arm (configuration) that was used
5. Thompson sampling selects best arm for similar prompts
6. Per-domain arms: different configurations for different problem types

**Landing:** Extends existing `BanditPruner` with episode-guided reward signal. Adds `episode_reward()` method that queries anyrag.

**Why it's modelless:** Bandit learning is online, no gradient. The "distillation" is Thompson sampling updating Q-values from episode outcomes instead of random exploration.

**Expected gain:** 10-20% improvement in pruner accuracy over time. Compounding — more episodes → better arms → better pruning → more episodes.

---

## Fusion Ideas — Model-Based (riir-ai)

### M1: STV-LoRA — Verifier-in-the-Loop Game AI Training 🔥

**Core idea:** STV's ViL training applied to game AI LoRA training. Train the game LoRA adapter inside a V-R loop where the game validator (BomberWASMValidator, GoValidator) provides feedback.

**How it works:**
1. Game self-play generates trajectories (generator)
2. Game validator verifies each action (verifier) — already working
3. When validator rejects: feedback is the invalid action + valid alternatives
4. LoRA training continues with verifier feedback as additional context
5. This is ViL: the LoRA learns to use verifier feedback during training
6. At inference: LoRA runs standalone but has learned from verifier feedback

**Landing:** New `vil_training.rs` in riir-gpu. Extends existing GRPO training loop with verifier-in-the-loop feedback. Uses existing game validators.

**Why it's model-based:** Requires LoRA weight updates. The ViL training is exactly STV's method applied to game AI: continue past GRPO/RLVR plateau using verifier feedback.

**Expected gain:** 30%+ past GRPO plateau (matching STV's 33% result). The standalone gain (without verifier at inference) is the key — the LoRA internalizes the verifier's knowledge.

### M2: Self-Trained Game Verifiers

**Core idea:** Train game verifiers using STV's reference-conditioned teacher approach. The "reference" is the optimal play (from MCTS or stronger player analysis).

**How it works:**
1. Collect winning trajectories from self-play (reference solutions)
2. Teacher verifier sees: game state + candidate action + winning action
3. Teacher can identify WHY a suboptimal action is bad (by comparing to winning line)
4. Student verifier sees only: game state + candidate action
5. On-policy distillation trains student to match teacher's feedback
6. At inference: student runs without reference → better action validation

**Landing:** New `stv_game_verifier.rs` in riir-gpu. Extends existing game validator training with OPD loss.

**Expected gain:** Better game validators → better LoRA training signal → compounding improvement.

### M3: Reference-Augmented Game LoRA (RAG-LoRA)

**Core idea:** During LoRA forward pass for game AI, inject winning trajectory from episode DB as additional context. The LoRA learns to use reference context. At inference, reference is removed but LoRA has learned reference-quality play patterns.

**How it works:**
1. During training forward pass: concatenate game state + reference trajectory
2. LoRA learns to condition on reference context for better action selection
3. During inference: no reference, but LoRA has learned patterns from reference-conditioned training
4. This is STV's "imitate a more informed version of itself" applied to game AI

**Landing:** Extends existing LoRA forward pass with optional reference context injection.

---

## 🔥 Novel Fusion: Episode-Guided Self-Improving Pruner Pipeline

**The fundamental insight:** STV's three innovations (reference-conditioned teacher, OPD, ViL) form a self-improving loop. Our modelless analog combines:

1. **Episode DB as teacher** (reference solutions = "privileged information")
2. **ConstraintPruner as student** (learns constraints from reference diffs)
3. **Bandit as OPD analog** (on-policy learning from episode outcomes)
4. **V-R DDTree loop as ViL analog** (iterative refinement with constraint injection)

**The pipeline:**
```
Problem → DDTree generate → ConstraintPruner filter → Verify (compiler/validator)
                                                            ↓ (if reject)
                                                    Extract failure patterns
                                                            ↓
                                                    Episode DB lookup (reference?)
                                                            ↓ (if found)
                                                    Constraint synthesis from diff
                                                            ↓
                                                    Inject new constraints → Re-generate
                                                            ↓ (repeat up to R rounds)
                                                    Accept or max rounds → Output
                                                            ↓
                                                    New episode → Episode DB update
                                                            ↓
                                                    Bandit arm reward update
```

**Self-improving loop:**
1. Each generation either succeeds (new episode) or fails (new failure pattern)
2. Episodes enrich the reference DB → better constraint synthesis for future queries
3. Bandit learns which constraint strategies work → better arm selection
4. V-R loop gets better over time as more episodes accumulate
5. No training, no gradient, pure inference-time self-improvement

**Why this is better than STV for our domain:**
- STV requires LLM training (OPD, ViL). Our version is modelless.
- STV's verifier is a language model. Our verifier is a deterministic ConstraintPruner — faster, cheaper, more reliable.
- STV's feedback is text. Our feedback is structured constraints — directly actionable.
- STV scales with GPU compute. Our version scales with episode DB size — amortized cost.

---

## GOAT Verdict

### Modelless (katgpt-rs)

| Fusion | GOAT Potential | Risk | Verdict |
|--------|---------------|------|---------|
| **F1: Episode-Guided Constraint Synthesis** | ⭐⭐⭐ HIGH | LOW — extends existing ConstraintPruner | **GO — Default ON if GOAT passes** |
| F2: V-R DDTree Loop | ⭐⭐ MEDIUM | MEDIUM — latency concern for hot path | GOAT-gate, Plan B |
| F3: Self-Distilling Bandit | ⭐⭐ MEDIUM | LOW — extends existing bandit | GOAT-gate, Plan C |

**F1 is the GOAT candidate.** It maps STV's core insight ("reference-conditioned diagnosis") to our existing ConstraintPruner trait. The episode DB is already designed for this — each episode has a prompt + successful output. The constraint synthesis is O(diff_length) — negligible cost. Zero perf hurt when no episode exists (fallback to base pruner).

### Model-Based (riir-ai)

| Fusion | GOAT Potential | Risk | Verdict |
|--------|---------------|------|---------|
| **M1: STV-LoRA ViL Game Training** | ⭐⭐⭐ HIGH | MEDIUM — new training loop | **GO — Feature-gated** |
| M2: Self-Trained Game Verifiers | ⭐⭐ MEDIUM | HIGH — new OPD infrastructure | Defer |
| M3: Reference-Augmented Game LoRA | ⭐ LOW | MEDIUM — context injection complexity | Defer |

**M1 is the GOAT candidate.** Game AI self-play has perfect verifiers (game rules are deterministic). ViL training with verifier feedback can break through the GRPO plateau. The existing game validators (BomberWASMValidator, GoValidator) are already providing binary accept/reject — adding feedback extraction is ~100 LOC.

### Commercial Strategy Alignment

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:
- **F1 (Episode-Guided ConstraintPruner)** → MIT katgpt-rs engine. Inference-time constraint synthesis is "plumbing" — open, attracts adoption.
- **M1 (STV-LoRA ViL Training)** → Private riir-ai SaaS. Training intelligence is "fuel" — closed, monetizable.
- The engine without the episode DB still works — just uses base pruners.
- The episode DB enriches constraint synthesis → better translations → more episodes → flywheel.
- This directly strengthens the "Data Flywheel" (Phase 4 of execution plan).

**Engine/Fuel split intact.** ✅

---

## Related Research Cross-References

| Research | Connection |
|----------|------------|
| 004 LoRA Architecture | STV-LoRA training pattern (M1) |
| 007 Screening Absolute Relevance | ScreeningPruner as continuous verifier |
| 107 SynPruner/Validator | ConstraintPruner as verifier |
| 156 Speculative Reconciliation | V-R loop analog |
| 172 MUSE Skill Evolution | Episode-guided skill lifecycle |
| 175 ThoughtFold | Chain folding + V-R loop synergy |
| 178 MUX | Multiplexed latent reasoning with V-R |
| 192 ITSE | Inference-time skill evolution + episode DB |
| riir-045 ANE Verdict | CPU/GPU routing for V-R loop |
| riir-064 ThoughtFold Mask-DPO | Mask-DPO + STV-OPD fusion |

---

## TL;DR

STV = **reference-conditioned teacher → on-policy distillation → iterative V-R refinement**. The training-time ideas (OPD, ViL) land in riir-ai's game AI LoRA training. The inference-time ideas (reference-guided diagnosis, V-R loop, bandit learning) land in katgpt-rs's ConstraintPruner trait. The creative fusion (Episode-Guided Constraint Synthesis) uses the episode DB as "reference solution" for modelless constraint synthesis — STV's core insight without any training. GOAT candidates: F1 (modelless, default-on if proven), M1 (model-based, feature-gated). Engine/fuel split intact.
