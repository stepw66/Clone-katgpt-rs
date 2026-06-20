# Research 272: Reversal Q-Learning (RQL) — Pass (→ riir-train)

> **Source:** [Reversal Q-Learning](https://arxiv.org/abs/2606.17551) — Oberai, Park, Levine (UC Berkeley). arXiv:2606.17551v1, 2026-06-16.
> **Date:** 2026-06-20
> **Status:** Closed — Pass on novelty (training-only, redirect to riir-train)
> **Related Research:** 012 (LEO All-Goals), 125-riir-train (QGF Critic Training Verdict), 236 (QGF — sibling modelless paper), 271 (MIT 6.S184 vocabulary crosswalk)
> **Classification:** Public (katgpt-rs)

---

## TL;DR

RQL is a **training algorithm** for offline RL with flow policies. It treats each Euler step of a flow policy as a separate MDP action ("expanded MDP"), then synthesizes virtual on-policy flow trajectories by **running the flow ODE in reverse** (`x_{f-1} ← x_f − v(s, x_f, f)`) starting from dataset action `x_F = a`. Because the virtual trajectories are deterministic, multi-step TD returns computed on them are unbiased and zero-variance, collapsing the effective horizon from T×F back to T. A value network is trained via **expectile regression** (IQL-style); the velocity field is trained via **DDPG-style policy gradient + BC regularizer**.

**Verdict: Pass on novelty.** RQL backpropagates through both V (expectile loss) and v (DDPG ascent), so it falls squarely under the skill's training-only redirect: *"DPO / GRPO / SFT / RL training pipelines"* and *"Anything that requires backpropagation through base weights."* No files created in this session beyond this redirect note.

---

## 1. Paper Core

| Component | Mechanism | Where it lives |
|---|---|---|
| Expanded MDP | Each Euler step of flow policy = a separate action; horizon ×F | Training framework |
| Flow reversal | Given (s,a,r,s') from offline dataset, run ODE backwards from `x_F = a` to synthesize virtual flow trajectory | Training-data construction trick |
| Multi-step returns | Skip intermediate flow transitions in TD target; zero-variance because virtual trajectories are deterministic & on-policy | Value horizon reduction |
| Value learning | Expectile regression on `V(s, x_f, f)` (IQL-style IVL loss, ensemble with pessimistic target) | **Backprop** |
| Policy learning | DDPG-style gradient ascent on `V` through velocity field `v` + BC flow-matching regularizer | **Backprop** |

Empirical: best aggregate score across 50 OGBench tasks (56 vs 46 next-best QAM-E), particularly strong on long-horizon (humanoidmaze-large, antmaze-giant, puzzle-4x4, cube-quadruple).

---

## 2. Distillation attempt (mandatory per workflow §1.5)

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase-equivalent term(s) | Status |
|---|---|---|
| Flow policy `v(s, x, f)` | `SpeculativeGenerator::generate()` velocity, drafter step | Shipped (speculative/) — but training a flow `v` is not |
| Flow reversal `x_{f-1} ← x_f − v` | Backward trajectory walker, `ReplayBackwardWalker` | Shipped modellessly — `katgpt-rs/src/pruners/bomber/replay_backward.rs` |
| Expanded MDP / horizon reduction | Decision stage, functor application, cgsp cycle | Training paradigm, no modelless analog |
| Expectile regression (IQL) | `LeoHead` critic training | Already covered — Research 012 LEO, Research 125-riir-train QGF Critic |
| Multi-step return | Trajectory folding, leaky integrator | Different mechanism; RQL needs value learning to use it |

### 2.2 Latent-space reframing (mandatory §1.5 step 3)

Attempted reframes against the five Super-GOAT factory modules:

- **HLA**: reverse-flow ≈ "given final HLA state, walk back the recurrence to recover prior belief states." But RQL uses it to construct *training data*, not as a runtime op. Modelless analog already ships as `ReplayBackwardWalker`.
- **latent_functor**: expanded MDP ≈ "decision stage per functor application." But this is a control paradigm needing weight updates; no modelless extraction.
- **cgsp_runtime**: multi-step return ≈ "skip cycles in curiosity signal." Requires value learning (backprop) to be useful.
- **LatCal**: no natural mapping — LatCal is deterministic raw↔latent bridge, not a flow.
- **Adapter routing**: classifier-free guidance `(1-w)u(∅) + w·u(y)` is the GOAT-tier framing per R269 warning; latent-functor/HLA reframing is not stronger here. Already covered by QGF (Research 236).

**None of the reframes survive the modelless-first constraint.** RQL's value is its training loop; the only modelless component (reverse-flow data synthesis) already ships as `ReplayBackwardWalker`.

### 2.3 Fusion check (prior art across BOTH repos, BOTH layers)

The Levine group's flow-policy RL program is **already fully distilled**:

| Existing note / plan | Covers | Match to RQL |
|---|---|---|
| **katgpt-rs/.research/236 QGF** (arxiv 2606.11087, sibling paper) | Test-time gradient guidance of flow policies — the **modelless half** of the same program | RQL is the **training half** of the same program; QGF explicitly redirects training side → riir-train |
| **katgpt-rs/.research/271 MIT 6.S184 crosswalk** | Vocabulary: flow ODE, score, classifier-free guidance, denoiser | RQL uses standard flow-matching vocabulary already crosswalked |
| **riir-train/.research/125 QGF Critic Training** | IQL critic training on top of game LoRA — **the exact expectile-regression objective RQL uses** | RQL adds the expanded-MDP framing; does not change the recipe |
| **katgpt-rs/.research/012 LEO All-Goals + Plan 155** | All-goals Q-head training infrastructure | Absorbs IQL-style training |
| `katgpt-rs/src/pruners/bomber/replay_backward.rs` | GFlowNet-inspired backward trajectory extraction (modelless) | Closest codebase analog to RQL's "reverse flow" — already ships, for inference-time replay mining |
| **katgpt-rs/.plans/268 QGF** | Test-time Q-gradient guidance primitive (Plan, GAIN) | Public engine side already covered |

**No novel fusion is possible.** RQL sits strictly inside the territory already framed by QGF + LEO + QGF Critic Training.

---

## 3. Verdict

**Pass on novelty.** Training-only paper → redirect to riir-train.

**One-line reasoning:** RQL backpropagates through both V (expectile loss) and v (DDPG ascent); the only modelless component (reverse-flow data synthesis) already ships as `ReplayBackwardWalker`; the Levine group's flow-policy RL program is already fully distilled by Research 236 (modelless half) + Research 125-riir-train (IQL critic training half, identical expectile objective) + Research 012 (LEO infrastructure).

### Novelty gate (§1.5)

| Q | Answer |
|---|---|
| 1. No prior art? | ❌ Heavy prior art: QGF (236), QGF Critic (125-riir-train), LEO (012), MIT 6.S184 crosswalk (271) |
| 2. New class of behavior? | ❌ Improved offline RL training; not a new capability for our modelless/game runtime |
| 3. Product selling point? | ❌ Cannot finish "our NPCs do X that no competitor can" from RQL |
| 4. Force multiplier (≥2 pillars)? | ❌ Touches only critic training, which is already covered |

All four ❌ → not Super-GOAT, not GOAT, not Gain.

---

## 4. What NOT to do

- ❌ Do not create a plan in katgpt-rs (no modelless primitive to ship)
- ❌ Do not create a riir-ai guide (no Super-GOAT selling point)
- ❌ Do not implement `ReplayBackwardWalker`-for-flows (we don't ship continuous flow policies — same verdict as Research 010 ColaDLM, 041 RePlaid, 044 ELF, 079 ELF Plan, 268 QGF Plan §"What This Plan Does NOT Do")
- ❌ Do not add a flow-policy training pipeline to riir-train as a result of this note (RQL is incremental on existing IQL/LEO infrastructure there; if riir-train wants to track it, that's a separate decision in the riir-train repo)

---

## 5. Why this note exists

Per the skill TL;DR: *"If a paper is training-only → note '→ riir-train' in one line and stop."* This note exists so a future grep for `RQL`, `Reversal Q-Learning`, `flow reversal`, `expanded MDP`, or `2606.17551` finds the prior-art cross-references (QGF 236, QGF Critic 125-riir-train, LEO 012) and does not re-distill. This is the prophylactic pattern established by Research 271 (MIT 6.S184 crosswalk §5: "issues closed NOT NOVEL because vocabulary translation revealed the mechanism already ships").

---

## TL;DR

RQL (arxiv 2606.17551) is a **training algorithm** for offline RL with flow policies — backprops through both value network (expectile loss) and velocity field (DDPG ascent). **Verdict: Pass on novelty → riir-train.** The Levine group's flow-policy RL program is already fully distilled: the modelless/test-time half is Research 236 (QGF, Plan 268), the IQL critic-training half is Research 125 in riir-train (with the *exact* expectile-regression objective RQL uses), and the closest codebase analog to RQL's reverse-flow tool — `ReplayBackwardWalker` — already ships modellessly for inference-time replay mining. No files created beyond this redirect note.
