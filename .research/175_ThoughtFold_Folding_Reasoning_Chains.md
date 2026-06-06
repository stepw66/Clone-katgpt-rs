# Research: ThoughtFold — Folding Reasoning Chains via Introspective Preference Learning

**Date:** 2026-06
**Source:** arXiv:2606.03503 (ICML 2026)
**Status:** GOAT Verdict — Distilled

---

## Paper TL;DR

ThoughtFold reduces LRM "overthinking" by 56% token usage while *improving* accuracy +2.82% (DeepSeek-R1-Distill-Qwen-7B). The key insight: RLVR uniformly reinforces all steps in a correct trajectory — both essential deductions and redundant explorations. ThoughtFold adds fine-grained step-level preference learning on top.

**Three mechanisms:**
1. **Introspective Redundancy Identification** — binary search to find minimal correct sub-trajectory (tail truncation + internal folding)
2. **Attention-based step importance** — middle-layer attention from answer tokens → reasoning tokens as importance proxy
3. **Dynamic Mask Strategy (Mask-DPO)** — step-level masks that penalize redundant steps and encourage "Fold Anchors" (bridges between essential segments)

**Results:** 56.1% token reduction on 7B, 42.6% on 14B, +0.98-2.82% accuracy improvement across 5 benchmarks.

---

## What's Training (Paper) vs. Inference (Ours)

| Paper Mechanism | Training-Time | Inference-Time (Modelless) |
|-----------------|--------------|---------------------------|
| GRPO outcome reward | ✅ LoRA weight update | ❌ N/A |
| Mask-DPO fine-grained loss | ✅ LoRA weight update | ❌ N/A |
| Introspective binary search | ✅ Generates preference pairs | ✅ **Prune-and-verify at inference** |
| Attention-based step importance | ✅ Ranking for folding | ✅ **Already in ForwardContext.scores** |
| Dynamic mask strategy | ✅ Training loss masking | ✅ **ScreeningPruner relevance masks** |
| Fold Anchor concept | ✅ Loss concentrates on bridge tokens | ✅ **DDTree branch cache reuse** |
| ML@k minimum length metric | ✅ Evaluation metric | ✅ **Bandit reward signal** |

---

## Fusion Ideas — Modelless (katgpt-rs)

### F1: ThoughtFold Inference — Chain Folding via DDTree

**Core idea:** At inference time, when ThinkingController selects Latent/CpuResample mode, apply introspective pruning to the *already-generated* CoT prefix before continuing generation.

**How it works:**
1. Generate CoT prefix until a "fold point" is detected (entropy spike → exploration began)
2. Use `ForwardContext.scores` (middle-layer attention) to rank prefix steps by importance
3. Binary search: prune low-importance steps, verify if model still produces coherent continuation
4. If yes → commit fold, resume from compressed prefix (KV cache rollback + replay)
5. If no → keep original prefix, continue generating

**Landing:** New `ChainFolder` struct implementing `ScreeningPruner` trait — returns `relevance = 0.0` for foldable steps, `1.0` for essential steps.

**Why it's modelless:** No weight updates. Uses existing attention scores + speculative verification. The "introspective" search is the same binary search the paper uses for training data construction, but we use it *live* during inference.

**Expected gain:** 30-50% CoT token reduction on hard queries (where thinking mode activates). Zero cost on easy queries (where ThinkingController picks Direct mode).

### F2: FoldCache — KV Rollback + Replay

**Core idea:** When ChainFolder identifies foldable segments, rollback the KV cache to the last essential step and replay only the essential prefix. This is the inference analog of the paper's "Internal Folding."

**How it works:**
1. During CoT generation, maintain a `Vec<StepBoundary>` marking reasoning step boundaries (detected by `\n\n` or think-tag transitions)
2. When ChainFolder decides to fold at step i, truncate KV cache to step_boundary[i]
3. Replay essential steps only (those with `relevance > threshold`)
4. Continue generation from compressed state

**Landing:** Extends `MultiLayerKVCache` with `truncate_to_step()` + `replay_essential()`.

### F3: ML@k Bandit Reward — Self-Learning Fold Budget

**Core idea:** Use the paper's ML@k metric as a bandit reward signal. The ThinkingBandit already decides *when* to think. Add a "fold budget" arm that decides *how aggressively* to fold.

**How it works:**
1. `ThinkingMode::Latent` gets a fold budget parameter `k ∈ [0.5, 1.0]` (fraction of steps to keep)
2. After each query, compute effective token reduction ratio
3. If accuracy maintained → increase fold aggressiveness (lower k)
4. If accuracy dropped → decrease fold aggressiveness (higher k)
5. Uses existing `ThinkingBandit` Thompson sampling

**Landing:** Add `fold_budget: f32` to `ThinkingMode::Latent`, bandit reward from speculative verifier acceptance rate.

---

## Fusion Ideas — Model-Based (riir-ai, LoRA-only)

### M1: ThoughtFold LoRA Training — Mask-DPO for Game AI

**Core idea:** Apply the paper's Mask-DPO training method to game AI LoRA training. When self-play generates correct trajectories (e.g., bomber winning game), introspectively identify and fold redundant moves.

**How it works:**
1. GZeroLoop generates trajectory τ = (game_states, actions, reward=1)
2. Introspective search: binary search on action subsequence → verify via replay
3. Construct preference pairs: (concise winning trajectory) ≻ (verbose winning trajectory)
4. Mask-DPO loss: penalize redundant actions, encourage Fold Anchors (transition moves)
5. Joint GRPO + Mask-DPO training (λ coefficient)

**Landing:** New `loss_thoughtfold.rs` in riir-gpu, extending existing `loss_dpo.rs` with step-level masks. `FoldAnchor` annotation on game action transitions.

### M2: Attention-Guided LoRA Curriculum

**Core idea:** Use middle-layer attention (from the base model during LoRA forward pass) to weight training samples. High-attention-redundancy samples get lower curriculum priority.

**Landing:** New `thoughtfold_curriculum.rs` in riir-gpu, integrating with existing `curvature_curriculum.rs` (Plan 205).

---

## GOAT Verdict

### Modelless (katgpt-rs)

| Fusion | GOAT Potential | Risk | Verdict |
|--------|---------------|------|---------|
| **F1: ChainFolder ScreeningPruner** | ⭐⭐⭐ HIGH | LOW — extends existing ScreeningPruner | **GO — Default ON if GOAT passes** |
| F2: FoldCache KV Rollback | ⭐⭐ MEDIUM | MEDIUM — KV cache mutation is tricky | GOAT-gate, Plan B |
| F3: ML@k Fold Budget | ⭐⭐ MEDIUM | LOW — extends existing bandit | GOAT-gate, Plan C |

**F1 is the GOAT candidate.** It lands naturally on the `ScreeningPruner` trait (which already returns `f32` relevance scores). The introspective binary search is O(log N) in CoT steps. Verification reuses `SpeculativeVerifier`. Zero perf hurt on Direct mode (no thinking path executed).

### Model-Based (riir-ai)

| Fusion | GOAT Potential | Risk | Verdict |
|--------|---------------|------|---------|
| **M1: Mask-DPO for Game AI** | ⭐⭐⭐ HIGH | MEDIUM — new loss function + mask logic | **GO — Feature-gated** |
| M2: Attention Curriculum | ⭐ LOW | LOW — incremental | Defer |

**M1 is the GOAT candidate.** Game AI self-play generates perfect training data for ThoughtFold: winning trajectories with redundant moves. The existing `loss_dpo.rs` already has `PreferencePair` with chosen/rejected masks. Adding step-level dynamic masks is ~200 lines.

### Commercial Strategy Alignment

Per `003_Commercial_Open_Source_Strategy_Verdict.md`:
- **F1 (ChainFolder)** → MIT katgpt-rs engine. Inference-time chain folding is "plumbing" — open, attracts adoption.
- **M1 (Mask-DPO training)** → Private riir-ai SaaS. Training intelligence is "fuel" — closed, monetizable.
- The engine without the trained fold-budget model still works — just uses default fold aggressiveness.
- The trained model learns *which steps are typically redundant per domain* — better fold decisions.

**Engine/Fuel split intact.** ✅

---

## Novel Fusion (Creative, Not Direct Mapping)

### 🔥 ThoughtStream: Streaming Chain Fold + Bandit Self-Tuning

**The fundamental insight:** ThoughtFold's introspective search is *not just about training data construction* — it's a **general meta-cognitive loop** that works at inference time. The model "thinks about its own thinking" and compresses on the fly.

**Fusion:** ChainFolder (F1) + ThinkingController (Plan 194) + MUX (Plan 178) in a unified pipeline:

1. **ThinkingController** decides: Direct or Think?
2. If Think: generate CoT prefix via RiM buffer slots
3. **ChainFolder** introspects on the buffer: attention scores rank buffer positions
4. **MUX superposition**: instead of pruning (losing information), multiplex low-importance positions into a single latent token
5. Continue generation from compressed buffer
6. **Bandit** learns per-domain fold aggressiveness

**Why this is better than the paper:** The paper does discrete pruning (remove step or keep). MUX superposition preserves *all* information in compressed form. No information loss — just dimensionality reduction.

**Why it's modelless:** No weight updates. Superposition happens in the existing MUX vocabulary space. Bandit learns the right compression ratio online.

**Expected outcome:** Same quality as full CoT at 40-60% of tokens. No accuracy loss because MUX preserves information that discrete pruning loses.

---

## Related Research Cross-References

| Research | Connection |
|----------|------------|
| 012 TRT | Rejection knowledge → fold decisions |
| 016 AutoTTS | β-budget → fold aggressiveness |
| 076 SR²AM | Configurator → ThinkingController mode selection |
| 158 MUX | Vocabulary superposition → ThoughtStream fusion |
| 172 RiM | Buffer slots → reasoning workspace for folding |
| 194 Adaptive CoT | ThinkingController → fold trigger |
| riir-043 RiM | Two-stage curriculum → game AI folding |
| riir-048 MUX LoRA | MUX loss → Mask-DPO fusion |

---

## TL;DR

ThoughtFold = **introspective binary search + attention-based importance + Mask-DPO**. The training-time ideas (Mask-DPO loss) land in riir-ai's `loss_dpo.rs`. The inference-time ideas (introspective folding, attention ranking) land in katgpt-rs's `ScreeningPruner` trait. The creative fusion (ThoughtStream) combines ChainFolder + MUX superposition for information-preserving chain compression — better than discrete pruning. GOAT candidates: F1 (modelless, default-on if proven), M1 (model-based, feature-gated). Engine/fuel split intact.
