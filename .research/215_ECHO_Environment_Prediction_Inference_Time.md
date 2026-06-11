# Research 215: ECHO — Environment Prediction as Inference-Time Dense Supervision

**Date:** 2026-06
**Source:** arXiv:2605.24517 — ECHO: Terminal Agents Learn World Models for Free (Shrivastava et al., 2026)
**Verdict:** GOAT — High-gain for modelless inference-time environment prediction scoring
**Target:** Modelless (katgpt-rs) primary

---

## Executive Summary

ECHO proves that **environment observations are a free, dense supervision signal** already present in every rollout. By adding a cross-entropy loss on environment tokens alongside the standard policy-gradient loss on action tokens, ECHO doubles GRPO pass@1 (Qwen3-8B: 2.70→5.17%, Qwen3-14B: 5.17→10.79%) — with **zero architecture changes, zero extra rollouts**, same forward pass.

**The modelless fusion opportunity is NOT replicating ECHO's training loss** (that's model-based, riir-ai territory). Our opportunity is distilling ECHO's core insight — **prediction quality correlates with policy quality** — into our existing DDTree + BanditPruner + ScreeningPruner pipeline at inference time.

---

## Paper Core

### 1. The Hybrid Objective

```
L_ECHO(θ) = L_GRPO(θ; A) + λ · L_Env(θ; O')
```

Where:
- **A** = indices of assistant-action token positions in the rollout
- **O'** = indices of terminal-output (environment) token positions (excluding harness warning prefix)
- **L_GRPO** = standard clipped policy-gradient loss with group-normalized advantages (Eq. 2 in paper)
- **L_Env** = mean cross-entropy on environment observation tokens:

```
L_Env(θ; O') = -1/|O'| · Σ_{t ∈ O'} log p_θ(x_t | x_{1:t-1})
```

- **λ = 0.05**: Environment loss weight (0.01–0.05 safe, 0.2 degenerate)
- Same forward pass — zero overhead on inference compute

#### Key Mechanism

1. **Single forward pass**: The model already computes logits at every position for GRPO's action-token loss. ECHO simply **gathers the already-computed logits at environment-token positions** and adds their cross-entropy to the same backward pass.
2. **No extra rollouts, no teacher model, no architecture changes**.
3. **On-policy by construction**: Targets come from the current policy's own rollouts. As the agent improves and visits new terminal states, the environment produces new responses → **self-evolving curriculum**.
4. **Auto-annealing**: As the model learns terminal-output statistics, L_Env falls rapidly, reducing auxiliary contribution without explicit schedule.

#### What Gets Trained On

Rollout structure:
```
[sys] [task] [action₁] [obs₁] [action₂] [obs₂] ... [action_K] [obs_K]
```

- **GRPO**: trains only on `action` positions, driven by sparse binary outcome reward
- **ECHO**: additionally trains on `obs` positions (terminal output only, NOT harness warnings)

**Why exclude warnings?** Warning tokens (format violation messages) have near-zero entropy and get memorized in ~60 steps. Terminal-output tokens (file names, test failures, stack traces, byte counts) have irreducible entropy of 0.05–0.10 nats and provide sustained gradient throughout training.

#### Hyperparameters

| Parameter | Value |
|-----------|-------|
| λ (loss weight) | 0.05 (productive range: 0.01–0.05) |
| GRPO rollouts per prompt | n = 16 |
| Learning rate | 1e-6 (constant, no warmup/decay) |
| Gradient clip | 0.2 |
| Sampling temperature | 0.8 (train), 0.6 (eval) |
| Training steps | 500 GRPO steps |
| Hardware | 8× A100/B200 |

### 2. Key Results

#### Main Performance (TerminalBench-2.0)

| Model | GRPO pass@1 | ECHO pass@1 | Multiplier |
|-------|------------|-------------|------------|
| Qwen3-8B | 2.70% | 5.17% | ×1.9 |
| OT-SFT (8B) | 7.64% | 7.87% | ×1.03 |
| Qwen3-14B | 5.17% | 10.79% | ×2.1 |

#### Training Efficiency

- **8B**: ECHO reaches GRPO's peak performance in **1.5–2.3× fewer steps**
- **14B**: Both peak at same step, but ECHO reaches a **higher plateau**
- **Inference**: ECHO cuts TB2 timeouts from 19.8% → 9.0% (8B), reduces completion tokens by 30%

#### World Model Transfer

On **held-out off-policy trajectories from Qwen3-32B** (a model that didn't generate these trajectories):

| Model | val100 CE drop | ITD CE drop | TBLite CE drop |
|-------|---------------|-------------|----------------|
| Qwen3-8B | 0.29→0.07 | 0.46→0.32 | 0.35→0.25 |
| Qwen3-14B | 0.24→0.07 | 0.39→0.31 | 0.30→0.23 |

GRPO alone **barely changes** env-token cross-entropy. ECHO sharply lowers it across all slices.

#### Expert SFT Gap Recovery

ECHO from base Qwen3-8B recovers:
- **101.6%** of expert-SFT gap on val100
- **103.9%** on ITD
- **88.9%** on TBLite
- **~50%** on TerminalBench-2.0

*Without using any of the ~15k expert demonstrations.*

### 3. What Worked / Didn't

| What | Verdict |
|------|---------|
| λ ∈ [0.01, 0.05] | ✅ Safe range. Below 0.01: gradient too small. Above 0.1: competes with policy. |
| Auto-annealing λ | ✅ As model learns env statistics, L_Env drops → natural decay without scheduling |
| Env-only targets (stdout, not warnings) | ✅ Warning tokens memorized in ~60 steps, then provide zero gradient. Env tokens sustain gradient. |
| Clean rollout filtering for verifier-free | ✅ Filtering to parseable tool calls is critical for OOD verifier-free adaptation |
| Verifier-free on PyTerm | ✅ +10pp from env-loss alone |
| λ = 0.2 | ❌ Degenerate rollouts — terminal outputs easy to predict but no longer useful |
| TBLite verifier-free | ❌ -3.9pp. Weak env-action coupling; filesystem orchestration needs broader shell state |
| SFT-initialized models | ❌ Less marginal gain (already has interaction prior) |
| 14B internal gains smaller | ❌ Gains appear on TB2 (harder benchmark) not internal evals. Policy/env-prediction compete more. |

#### Verifier-Free Adaptation (§5.5)

Starting from best ECHO checkpoint, mask GRPO, train only L_Env for 100 steps:

| Target | Δ pass rate | Filter |
|--------|-----------|--------|
| val100 (in-dist) | +3.8pp | none |
| PyTerm (OOD) | +10.0pp | clean tool calls |
| ITD (OOD) | +5.2pp | clean tool calls |
| TBLite (OOD) | -3.9pp | clean tool calls |

**Key insight**: Verifier-free env-only adaptation works best when clean exploration exposes **predictive, action-linked feedback** (e.g., Python tracebacks). Fails when terminal output is weakly coupled to action quality (e.g., filesystem orchestration).

### 4. Related Work

#### Closest Cousins (Training-Time Auxiliary Prediction)

| Paper | Key Idea | Relation to ECHO |
|-------|----------|-----------------|
| **PaW** arXiv:2606.02388 | Co-train policy + world model on next-obs | Nearly identical, concurrent. ECHO has deeper ablations + verifier-free result. |
| **CWM** (FAIR CodeGen, 2025) arXiv:2510.02387 | Separate 32B world model on execution traces | Separate model, offline corpus. ECHO is on-policy, in-line, no separate stage. |
| **RLTF** (Song et al., 2026) arXiv:2602.02482 | Predict judge-generated critiques as auxiliary loss | Needs judge/teacher. ECHO predicts raw env tokens directly. |
| **OpenClaw-RL** (Wang et al., 2026) arXiv:2603.10165 | Next-state signals via judge-extracted hints | Uses judge. ECHO uses raw env tokens. Complementary. |
| **Self-Distillation** (Hübotter et al., 2026) arXiv:2601.20802 | RL via self-distillation of agent experience | Related in using agent's own experience, different mechanism. |

#### Classical Auxiliary Prediction in RL

ECHO follows the lineage of UNREAL (Jaderberg 2017), curiosity-driven (Pathak 2017), SPR (Schwarzer 2021), future prediction (Kwon 2024) — but applied to **multi-turn LM-agent setting** where targets are textual observations already in the rollout.

#### Inference-Time / Training-Free Approaches (Key Gap Area)

| Approach | What It Does | Training Required? |
|----------|-------------|-------------------|
| **Speculative Decoding** (EAGLE, Medusa) | Draft model predicts future tokens via feature-level autoregression | Yes — draft model trained separately |
| **MCTS / Tree Search at inference** | Use model's own probability estimates to search (AlphaCode, QwQ) | No extra training, but compute budget |
| **Process Reward Models at inference** | Score intermediate steps during generation to guide search | Yes — PRM must be trained |
| **Self-consistency** | Sample multiple completions, select by majority vote | No training |
| **Verifier-free ECHO** (§5.5) | Continue training on env-prediction only | Still updates weights |

**Critical observation**: There is **no known inference-time-only technique** that directly applies ECHO-style "predict environment to improve policy" without weight updates. This is a research gap.

---

## Fusion: Novel Modelless Applications of ECHO Insight

The paper's training-time approach is **not directly applicable** to our modelless constraint. But ECHO proves a deeper principle that IS applicable at inference time:

### Insight 1: Prediction Quality ≈ Policy Quality

ECHO shows that policies that better predict environment dynamics also better navigate those dynamics. At inference time, we can **score actions by how predictable their outcomes are** — not by training a predictor, but by using the game's own forward model speculatively.

**Our novel fusion: `EnvPredictorPruner`** — a `ScreeningPruner` that:
1. For each candidate action, runs the game's deterministic forward model (already exists for game engines)
2. Scores the resulting state by how "expected" it is (entropy of state features vs historical average)
3. Boosts actions leading to predictable states, suppresses actions leading to chaotic/surprising states
4. Uses bandit to learn which environments benefit from this scoring

This is **not ECHO's training loss** — it's the inference-time dual: instead of training to predict the environment, use the environment to score predictions.

### Insight 2: Failed Rollouts Are Information

ECHO's key finding: failed rollouts contain rich evidence about environment dynamics. Standard GRPO discards this. At inference time, our DDTree already explores failed branches — but we don't currently **learn from them across sessions**.

**Our novel fusion: `PredictionVerifier` bandit arm** — track prediction-vs-reality across DDTree branches:
1. During DDTree exploration, log predicted outcomes per branch
2. After verification (LeviathanVerifier), compare actual vs predicted
3. Feed accuracy signal into BanditPruner reward
4. AbsorbCompress promotes prediction strategies with high verification accuracy

### Insight 3: Dense Intra-Trajectory Credit

ECHO's environment prediction creates dense per-token credit. At inference time, we can approximate this via **step-level scoring** in DDTree:

**Our novel fusion: `ShapedBanditPruner`** — from Research 025 (StepCodeReasoner):
1. Intra-trajectory advantage: `Â(i) = r_i × (1 + future_accuracy)`
2. Steps that "pave the way" (lead to verified good outcomes) get boosted
3. Steps that are locally plausible but lead nowhere get suppressed
4. Pure post-hoc computation on DDTree verification paths — modelless

### Insight 4: Verifier-Free Self-Improvement

ECHO's most striking result: **environment prediction loss alone** (+10pp) enables self-improvement without any reward signal. At inference time, this maps to:

**Our novel fusion: `PredictionConsistencyGate`** — if the model's marginal predictions are consistent (low entropy across multiple DDTree branches), the action is likely correct. If predictions are inconsistent (high inter-branch entropy), the action needs more exploration. This is a **modelless consistency check** that requires no training — just entropy measurement on the existing DDTree output.

---

## Research Gap: Inference-Time World Model for Policy Improvement

ECHO proves that environment prediction capability **correlates with policy quality**. The key question is:

> Can we use a model's internal environment-prediction capability at **inference time** (no weight updates) to improve action selection?

### Potential Approaches (untested/speculative)

1. **Implicit world-model scoring**: At each action step, sample multiple candidate actions. For each, use the model's own prediction of environment response as a **scoring signal** (higher likelihood of "good" env tokens = better action). No training, just beam-search-like scoring.

2. **Verification via environment simulation**: Before committing an action, the model "imagines" the terminal output. If the imagined output contains error patterns (exit code ≠ 0, stack traces), reject/modify the action. This is speculative decoding applied to environment tokens.

3. **Contrastive action selection**: Given multiple action candidates, compute p(action | context) × p(env_response | action, context). The joint probability picks actions that the model can both generate and predict consequences for.

4. **Lookahead with world model head**: If the model has been trained with ECHO-style auxiliary loss, its env-prediction head can be repurposed at inference for **MCTS**-style rollouts: action → predicted env → next action → predicted env → ... → evaluate terminal state.

5. **Self-play with environment model**: At inference, generate K actions, for each predict the env response, then continue the trajectory in "imagined" mode. Score the imagined trajectories by whether they reach a completion signal.

### Why This Hasn't Been Done

- ECHO-style auxiliary loss is very new (May 2026)
- Most inference-time work focuses on **mathematical reasoning** (MCTS, PRM) not **interactive environments**
- Speculative decoding research focuses on **latency**, not **policy quality**
- The connection between "model can predict environment" and "model makes better decisions" was only recently formalized by ECHO

---

## Distillation: What's Training-Time Only (riir-ai)

- The auxiliary cross-entropy loss `L_Env` on environment tokens
- Joint GRPO + env-prediction training
- λ scheduling and auto-annealing
- Any weight updates (LoRA or full)

## What's Inference-Time (katgpt-rs)

- Environment forward model scoring (game engines have deterministic forward models)
- Prediction-vs-reality verification (compare speculative branches against actual outcomes)
- Bandit-driven prediction strategy selection
- Consistency-based confidence scoring (entropy across DDTree branches)
- Shaped intra-trajectory credit (post-hoc computation)

---

## Existing Infrastructure (80% Built)

| Component | What | ECHO Role |
|-----------|------|-----------|
| `ScreeningPruner::relevance()` | Token quality scoring | Environment prediction as relevance signal |
| `BanditPruner<P>` | Adaptive arm selection | Track which prediction strategies work |
| `DDTree` | Speculative tree search | Multi-step environment rollouts |
| `WasmPruner` | Deterministic validation | Verify predictions against game rules |
| `AbsorbCompress` | Promote winning patterns | Lock in working prediction strategies |
| `TrialLog` | Episode history | Prediction vs reality log |
| `HotSwapPruner` | Runtime swap | Change prediction strategies dynamically |
| `ConstraintPruner::is_valid()` | Binary accept/reject | Reject actions with invalid predicted outcomes |
| `NextLat` (Plan 217) | Belief-state drafter | Frozen MLP as environment predictor |
| Freeze/thaw | Cross-session persistence | Prediction skills survive sessions |

## Missing Primitives (The 20% Gap)

1. **`EnvPredictorPruner`** — ScreeningPruner that scores actions by predicted outcome quality
2. **`PredictionVerifier`** — Compare predicted state vs actual state, feed into bandit reward
3. **`PredictionConsistencyGate`** — Entropy-based confidence from DDTree branch consistency

---

## GOAT Gate

Feature flag: `echo_env_predictor` (default-OFF until GOAT proof passes)

### GOAT Proofs Required

| # | Metric | Threshold | Measurement |
|---|--------|-----------|-------------|
| G1 | Bomber HL score with EnvPredictorPruner | ≥ baseline (no regression) | Arena benchmark |
| G2 | Prediction accuracy bandit convergence | ≥70% correct after 100 rounds | Unit test |
| G3 | DDTree branch consistency improvement | ≥15% entropy reduction on hard queries | Benchmark |
| G4 | No hot-path latency regression | ≤5% overhead per token | Micro-bench |

---

## Verdict by 003 Commercial Strategy

- **Modelless first** ✅ — inference-time only, no LLM training
- **Engine territory** ✅ — fits katgpt-rs engine, no fuel dependency
- **SOLID/DRY** ✅ — extends existing ScreeningPruner/BanditPruner traits
- **Tests/examples** ✅ — bomber arena before/after, prediction accuracy test
- **CPU/GPU auto-route** ✅ — forward model is CPU (game engine), no GPU needed
- **Tier aware** ✅ — prediction scoring in Hot tier, verification in Warm tier
- **Adaptive threshold** ✅ — bandit learns when env prediction helps vs hurts

**Decision: GAIN — implement as feature-gated plan, GOAT gate before promotion.**

---

## TL;DR

ECHO proves environment observations are dense supervision. We distill this to modelless: score DDTree actions by predicted-outcome quality using the game's own forward model, verify predictions against reality, and bandit-learn which prediction strategies work. The infrastructure is 80% built — need 3 new primitives (EnvPredictorPruner, PredictionVerifier, PredictionConsistencyGate) wired into existing BanditPruner + DDTree + AbsorbCompress pipeline.
