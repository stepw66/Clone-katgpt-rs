# Research: RLM-GEPA — Reflective Prompt Evolution for Modelless Stack

**Date:** 2026-05-31
**Source:** Gabriel Lespérance, "Going recursive (part I): Applying RLM-GEPA to AppWorld" (X article)
**Papers:** [RLM arXiv:2512.24601](https://arxiv.org/abs/2512.24601), [GEPA arXiv:2507.19457](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral)
**Status:** GOAT Verdict Pending

---

## Summary

RLM-GEPA combines two techniques:
1. **Recursive Language Models (RLM):** LLMs treat long context as an external environment, recursively calling themselves on snippets, decomposing tasks programmatically.
2. **GEPA (Genetic-Pareto):** Reflective prompt evolution using natural language reflection instead of RL gradients — outperforms GRPO by 6% average, up to 20%, with 35x fewer rollouts.

Result on AppWorld: 0.940 TGC / 0.911 SGC — state-of-the-art.

---

## RLM Analysis — REDUNDANT with Our Stack

Our architecture already covers RLM's core ideas:

| RLM Concept | Our Equivalent | Plan |
|-------------|----------------|------|
| Recursive self-calls on context snippets | DDTree branch exploration | Core |
| External context management | ScreeningPruner + BanditPruner | Plan 021, 030 |
| Looped inference | LT2 (Looped Inference Pipeline) | Plan 108 |
| Multi-hop speculation | SpecHop continuous pipeline | Plan 131 |
| Decomposition | TemplateProposer task decomposition | Plan 049 T2 |

**Verdict: RLM adds nothing new.** Our DDTree + ScreeningPruner + LT2 already does recursive, looped, multi-hop inference with external context gating.

---

## GEPA Analysis — GAP IDENTIFIED

GEPA's core loop:

```
1. Sample K trajectories (rollouts) with current prompts
2. Reflect: natural language diagnosis of failures
3. Propose: generate candidate prompt updates from reflection
4. Test: evaluate candidates on held-out trajectories
5. Combine: Pareto-frontier selection of complementary lessons
6. Update: apply winning prompts → repeat
```

**What GEPA optimizes:** System-level prompts, not model weights. A few rollouts + reflection > thousands of RL gradient updates. The key insight: language is a richer learning medium than scalar rewards for LLMs.

### Mapping to Our Architecture

| GEPA Component | Our Equivalent | Status |
|----------------|----------------|--------|
| Trajectory sampling | DDTree branches + TrialLog (JSONL) | ✅ |
| Failure reflection | MeMo Reflection QA (Plan 094) | ✅ |
| Helpfulness tracking | ReviewMetrics helpful/harmful (Plan 036) | ✅ |
| Pareto frontier | BanditPruner UCB1/ThompsonSampling | ✅ |
| **Prompt update from reflection** | **Missing** | ❌ |
| **Genetic crossover** | **Missing** | ❌ |
| **Candidate testing** | **Missing** | ❌ |

### The Gap

Our reflection pipeline (MeMo → BanditPruner) feeds **arm-level rewards** but doesn't feed back into **system-level configuration**:

- Rubric weights (ROPD, Plan 071) are static
- Bandit hyperparameters (ε, UCB1 c) are fixed
- TemplateProposer hints (Plan 049 T2) are hand-written
- AbsorbCompress thresholds (Plan 030) are constant

GEPA's insight: **let the system evolve these configurations from trajectory reflection**, using Pareto-frontier selection to keep the best variants.

### Where It Fits: SR²AM Configurator Bandit (Plan 112)

Our SR²AM already makes adaptive planning decisions (tree budget, draft lookahead, early exit). GEPA adds a **meta-level**: evolve the SR²AM's own configuration from trajectory reflection.

```
Current:  Episode → TrialLog → BanditPruner (arm rewards)
GEPA:     Episode → TrialLog → Reflection → Prompt Candidates → Pareto Test → Config Update
```

This is modelless prompt optimization — no gradients, no LoRA, just bandit-driven configuration evolution from natural language reflection. Exactly our paradigm.

---

## Verdict: CONDITIONAL GAIN (Modelless Path Only)

### What to Take

1. **Reflective Config Evolution** — use MeMo reflection to propose configuration updates (rubric weights, template hints, bandit params) for SR²AM
2. **Pareto-Frontier Selection** — keep the Pareto-optimal configurations across evaluation rounds
3. **Few-Shot Reflection** — a few trajectories + reflection > many blind rollouts

### What to Skip

1. **RLM recursive context** — we already have this via DDTree + LT2 + SpecHop
2. **Model-based prompt optimization** — GEPA's full pipeline requires LLM self-reflection calls, which is model-based and expensive
3. **Genetic crossover of prompts** — overkill for our parameter space (we have ~10 config knobs, not free-form text prompts)

### Modelless Distillation Targets

| Distillation | Component | What |
|-------------|-----------|------|
| **D1: Reflective Arm** | `ReflectiveBanditPruner` | Bandit arm = config variant, reward = reflection score |
| **D2: Pareto Config** | `ParetoConfigFrontier` | Track Pareto-optimal config variants by (reward, cost) |
| **D3: Template Hint Evolution** | `TemplateProposer` extension | Generate hint variants from reflection, test via bandit |

### Estimated Gain

- **Quality:** MeMo reflection already produces structured feedback. Using it to evolve configs should improve rubric targeting and template relevance.
- **Cost:** Near-zero — reflection is already computed, config evaluation is just running bandit episodes with different configs.
- **Risk:** Low — feature-gated, additive to existing BanditPruner, doesn't touch hot path.

### Optimization Alignment

Per `optimization.md`:
- Config evolution happens **between episodes**, not in the hot path (zero decode overhead)
- Pareto tracking uses fixed-size arrays (bounded config space)
- No new allocations in the decode loop
- Bandit reward computation is already O(1) per arm

---

## Comparison with Related Work in Our Stack

| System | Updates | From Reflection? | Pareto? |
|--------|---------|-------------------|---------|
| BanditPruner (Plan 030) | Arm Q-values | ❌ (environment reward) | ❌ |
| ReviewMetrics (Plan 036) | Helpful/harmful counts | ✅ (intervention tracking) | ❌ |
| G-Zero Hint-δ (Plan 049) | Dense reward from log-prob shift | Partial (intrinsic signal) | ❌ |
| ROPD Rubric (Plan 071) | Per-criterion gap targeting | ❌ (static rubric) | ❌ |
| SR²AM (Plan 112) | Adaptive planning decisions | ❌ (domain config) | ❌ |
| **GEPA-D (proposed)** | **Config variant selection** | **✅ (MeMo reflection)** | **✅** |

---

## Conclusion

RLM is redundant. GEPA's reflective prompt evolution fills a real gap: **evolving system-level configuration from trajectory reflection without gradient updates**. This maps cleanly to our modelless stack as a BanditPruner extension where each arm is a config variant, scored by reflection quality.

The gain is in the SR²AM → reflection → config evolution loop. Implementation is additive, feature-gated, and doesn't touch the hot path. Worth pursuing as a modelless distillation target.

---

## References

- Alex L. Zhang, Tim Kraska, Omar Khattab. "Recursive Language Models." arXiv:2512.24601 v3, May 2026.
- Lakshya A. Agrawal et al. "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning." ICLR 2026 Oral. arXiv:2507.19457 v2, Feb 2026.
- Gabriel Lespérance. "Going recursive (part I): Applying RLM-GEPA to AppWorld." X article, May 2026.
