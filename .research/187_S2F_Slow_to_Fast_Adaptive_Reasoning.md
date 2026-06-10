# Research: S2F Slow-to-Fast Adaptive Reasoning + DeGRPO Hybrid Mode Selection

**Date:** 2026-06
**Papers:**
1. "To Think or Not To Think" (ICML 2026) — S2F (Slow-to-Fast) reasoning + T2M (Think-to-Match)
2. "Thinkless: LLM Learns When to Think" (NeurIPS 2025) — DeGRPO decoupled policy optimization
**Verdict: GOAT** — Fuses into existing SelectivityRouter + ThinkingController with novel additions

---

## Paper 1: "To Think or Not To Think" — Core Insights

### Key Finding: Reasoning Collapse
- Extended deliberation in ToM/ambiguous tasks **correlates with failure**
- Response length distribution shows errors clustered in high-length regions (8K-10K chars)
- More reasoning effort → worse performance on complex tasks (GPT-o3: 0.838→0.693 on HiToM)
- Token length limits act as **performance catalysts** — Qwen3-8B: 1500 token limit beats unconstrained

### Key Finding: Complementary Strengths
- Reasoning and non-reasoning models solve **disjoint sets** of hard problems
- At Order 4 HiToM: Reasoning uniquely solves 49, Non-reasoning uniquely solves 39, overlap only 58
- Neither pure reasoning nor non-reasoning alone achieves optimal results
- **Adaptive strategy** based on task complexity is the answer

### Intervention: S2F (Slow-to-Fast Reasoning)
- Count "wait" tokens in reasoning trace
- When count exceeds threshold τ, terminate slow thinking → force fast answer
- Token-level generation control: replace next "wait" with preset sentence
- Results: R1-Distill-Qwen-32B surges 0.571→0.701 on HiToM

### Intervention: T2M (Think-to-Match)
- Phase 1: Think without options (pure reasoning, S2F-gated)
- Phase 2: Reintroduce options for matching
- Prevents option-matching shortcut where models justify choices post-hoc
- DeepSeek-R1: 0.549→0.691 on HiToM when options removed

### Critical Insight: Complexity-Triggered Collapse
- Reasoning collapse is **specifically triggered by high cognitive load**
- Less pronounced on simpler benchmarks (ToMATO, ToMBench)
- No single optimal token length — varies by model AND benchmark
- **Implies: adaptive per-instance budget, not global threshold**

---

## Paper 2: "Thinkless" — Core Insights

### Architecture: Hybrid Reasoning via Control Tokens
- Two control tokens: `<short>` (concise) and `..` (detailed CoT)
- First token emitted determines reasoning mode
- SFT warm-up from dual experts (reasoning model + instruction model)
- Paired dataset: same query → both long and short responses

### Algorithm: Decoupled GRPO (DeGRPO)
- **Problem:** Vanilla GRPO treats control token and response tokens uniformly
- Mode-Accuracy imbalance: 1 control token vs T response tokens
- Think-Short imbalance: `T_think >> T_short` suppresses gradient on `..` token
- **Solution:** Separate normalization:
  - Control token: `α · L_{i,0}(θ)` (length-independent weight)
  - Response tokens: `(1/T_i) · Σ L_{i,t}(θ)` (per-token normalization)
- α = 1/1000 in practice

### Reward Design
- Short + correct: reward = 1.0
- Long + correct: reward = 1.0 - γ (preference for efficiency)
- Wrong answer: reward = -1.0
- γ ∈ (0,1) controls efficiency-accuracy tradeoff

### Results
- 50-90% reduction in long-chain thinking on simple benchmarks
- On AIME (hard): model naturally uses 100% long reasoning
- On GSM8K: only 13.31% thinking mode activated
- Minerva Algebra: 25.88% thinking mode, 3× fewer tokens, <1% accuracy loss

### Training Dynamics: U-Shape Curve
- Early: long-chain preference (higher initial accuracy)
- Mid: short-chain accuracy improves (RL + difficulty routing)
- Late: only hard queries remain in long mode (shift in task allocation)

---

## Novel Fusion: Collapse-Aware Adaptive Bandit (CAAB)

### What's New vs Existing Infrastructure

| Existing | Gap This Fills |
|----------|---------------|
| SelectivityRouter (kurtosis) | Static statistical signal, no runtime collapse detection |
| ThinkingController (bandit) | Bandit converges globally, no per-instance early exit |
| RiM slots (latent workspace) | No signal to stop thinking mid-reasoning |
| PPoT (resample) | Resamples from same distribution — no mode switch |
| Parallel-Probe | Explores N branches simultaneously — no early stop |

### Fusion Idea: Three-Layer Adaptive Stack

**Layer 1: Pre-Decide (Before Thinking)**
- Existing SelectivityRouter kurtosis → Direct vs CoT routing
- **New:** Add DeGRPO-style reward shaping to bandit feedback
- Short + correct gets γ-boosted reward (prefer efficiency)
- This makes the bandit learn the **value of not thinking**

**Layer 2: Mid-Think Collapse Detector (During Thinking)**
- **New:** "Wait"-frequency monitor from S2F paper
- Count "wait"-like hesitation tokens in the reasoning trace
- If count > τ within a sliding window → **collapse predicted**
- Action: truncate thinking, switch to fast mode
- This is the missing piece — current infrastructure has no runtime mechanism to stop mid-reasoning

**Layer 3: Post-Verify (After Thinking)**
- Existing ConvergenceSelector (BestQ, BtRank, MajorityVote)
- **New:** T2M-inspired option stripping
- Run verification WITHOUT options first (pure reasoning check)
- Then match against options if present
- Prevents the option-matching shortcut the paper identified

### Modelless Distillation

| ID | Component | From Paper | Maps To |
|----|-----------|-----------|---------|
| D1 | `CollapseDetector` | S2F wait-count | New trait in `katgpt-core` — counts hesitation patterns in token stream |
| D2 | `ThinkingBudget` | Token length limits | Per-position adaptive budget (EMA of optimal length from bandit) |
| D3 | `OptionStripper` | T2M | Wrapper around ScreeningPruner that strips options for first pass |
| D4 | `EfficiencyReward` | DeGRPO γ parameter | Reward shaping: sigmoid(budget_saved/max_budget) for short+correct |
| D5 | `CollapseAwareBandit` | U-shape + S2F | ThinkingBandit with early-exit signal from CollapseDetector |

### Model-Based Distillation (riir-ai)

| ID | Component | From Paper | Maps To |
|----|-----------|-----------|---------|
| M1 | DeGRPO Training | Decoupled GRPO | Separate α-weighted loss on thinking mode token vs response tokens in `riir-gpu` |
| M2 | Warm-up Distillation | Dual expert SFT | Paired long/short dataset from reasoning + instruction models |
| M3 | U-Shape Curriculum | Training dynamics | Phase 1: encourage exploration of thinking mode. Phase 2: shift to short mode as accuracy improves |

---

## Verdict

| Criterion | Score | Notes |
|-----------|-------|-------|
| Modelless-first | 5/5 | D1-D5 all inference-time, no training required |
| Land in riir-ai domain | 5/5 | M1-M3 are LoRA training, not full LLM training |
| SOLID/DRY | 5/5 | CollapseDetector is a new trait, composes with existing |
| Perf impact | 5/5 | Wait-counting is O(1) per token, no allocation |
| Tests/examples | 4/5 | Clear before/after: thinking vs collapsed vs adaptive |
| CPU/GPU auto-route | 5/5 | Collapse signal feeds into existing ThinkingController |

**Decision: ADOPT.** The CollapseDetector (D1) is the novel missing piece — no existing infrastructure detects reasoning collapse at runtime. S2F's wait-counting maps directly to the token stream in the decode loop. The DeGRPO reward shaping (D4) makes the existing bandit learn the VALUE of not thinking, not just the value of thinking.

**GOAT Gate:** `collapse_aware_thinking` — on by default if gain proven (it should be — stopping collapse early saves tokens AND improves accuracy).

---

## Related Research

| Ref | Overlap |
|-----|---------|
| katgpt-rs 012 (TRT) | Rejection knowledge → PPoT. This adds: runtime collapse detection |
| katgpt-rs 175 (ThoughtFold) | Reduces overthinking by folding chains. This adds: mid-chain collapse stop |
| katgpt-rs 180 (Polarization) | Kurtosis-based routing. This adds: per-instance budget from collapse signal |
| Plan 204 (SelectivityRouter) | Pre-decide routing. This adds: mid-think collapse exit |
| Plan 194 (ThinkingController) | Bandit-based adaptive. This adds: collapse-aware early exit + efficiency reward |
| riir-ai 042 (Thinking Pixel) | Multi-LoRA routing. This adds: DeGRPO loss decomposition for mode token |
| riir-ai 043 (RiM) | Latent workspace. This adds: runtime stop signal for latent iterations |
| riir-ai 050 (SDPG) | REINFORCE+GRPO. This adds: decoupled α-weight for mode vs response |

TL;DR: Two papers prove "more thinking ≠ better" and give us both the detection mechanism (wait-count → collapse) and the training mechanism (DeGRPO → learn when to think). The novel fusion is CollapseDetector as a runtime mid-reasoning early exit, composed with existing SelectivityRouter + ThinkingController.
