# Research 309: Neurosymbolic Comparative Study (LNN vs LLM-SS) — PASS (already shipped)

> **Source:** Michael K. Chen. *A Comparative Study of Neurosymbolic AI Approaches to Interpretable Logical Reasoning*. [arXiv:2508.03366](https://arxiv.org/abs/2508.03366). NeSy 2025 (19th Conference on Neurosymbolic Learning and Reasoning). Raffles Institution.
> **Date:** 2026-06-25
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 184 (FOL-LNN — **the integrative half, already distilled**), 186 (Three-Mode NS Router — validates hybrid vs integrative), 295 (AC-GPT — constraining-generation cousin), 308 (Bisimulation + ASP operator inference — **the hybrid Stage-3 cousin**), 144/156/276/302 (affect composition — the latent reframing that beats binary gates)
> **Related Plans:** 209 (FOL-LNN DDTree→rules), 007 (SynPruner / rustc+syn hybrid solver), 259 (spec_compile grammar-constrained decoding), 324 (bisimulation ASP operator inference)
> **Verdict: PASS.** Both approaches the paper compares are already distilled in our quintet at higher fidelity. The paper's value is its **comparative verdict** (hybrid > integrative for general logical reasoning), which validates architectural choices we already made. The training recipes (LNN gradient descent on 4 gate weights; LLM fine-tuning) → **riir-train**. No file/plan/guide created beyond this classification note.

---

## TL;DR

The paper introduces two best-in-class models representing the two main neurosymbolic approaches and compares them:

1. **LNN (Logic Neural Network)** — *integrative*: each neuron is a differentiable logic gate `c = σ(w₁a + w₂b + w₃(a·b) + w₄)`, choosing among **16 logic gates** (the full binary truth table on 2 inputs). Weights trained via gradient descent; at inference, each neuron discretizes to its nearest gate. Comparable to Logic Gate Network (LGN) on synthetic + Breast Cancer / Adult Census, converges ~3× faster than LGN.
2. **LLM-SS (LLM-Symbolic Solver)** — *hybrid*: 3-stage pipeline. (1) LLM generates natural-language premises; (2) LLM translates premises to Clingo (Answer Set Programming) code **with a grammar-constraining program that masks logits of grammar-violating tokens**; (3) Clingo performs deterministic deductive reasoning. 54.5% accuracy / 1.5% error on StrategyQA (vs 48.5% / 17.8% unconstrained).

**Comparative verdict:** the hybrid approach is more promising for *general* logical reasoning because (i) its reasoning chain is more interpretable (Clingo code, not 640 neurons of gates), and (ii) it retains existing LLM capabilities (knowledge retrieval, generalization) that the integrative approach strips away by replacing the Transformer.

**Distilled for katgpt-rs (modelless, inference-time):** nothing not already shipped. The integrative LNN is covered by **R184/P209** ("the DDTree IS the LNN, constructed on-the-fly from marginals — no training needed"). The hybrid LLM-SS pipeline is covered by **P007** (SynPruner — rustc/syn as deterministic referee), **P259** (spec_compile — grammar-constrained decoding via ConstraintPruner + SpecDFA), and **R308/P324** (bisimulation + ASP operator inference). The specific Clingo ASP solver is one more Stage-3 tool we lack, but it is an external solver integration, not a modelless inference primitive. The comparative verdict validates our existing hybrid-first architecture.

---

## 1. Paper Core Findings

### 1.1 LNN (integrative) — the 16-gate relaxation

The single transferable formula:

```
c = σ(w₁·a + w₂·b + w₃·(a·b) + w₄)
```

where `a, b` are two input neurons and `w₁..w₄` are trainable weights. By evaluating at the 4 input combinations `(a,b) ∈ {(0,0),(0,1),(1,0),(1,1)}` and discretizing each output `oᵢ > 0.5 → 1`, the neuron snaps to one of **16 logic gates** (the complete truth table on 2 binary inputs: AND, OR, XOR, NAND, NOR, XNOR, ⇒, ⇐, ¬A, ¬B, A, B, True, False, and their negations). This generalizes Petersen et al. 2022's Logic Gate Network, which used a separate relaxation formula per gate + categorical average; LNN's single 4-weight formula converges ~3× faster.

**Training-only pieces** (→ riir-train): gradient descent on `w₁..w₄` per neuron; Adam optimizer; the convergence-speed comparison vs LGN.

### 1.2 LLM-SS (hybrid) — the 3-stage pipeline

| Stage | What it does | Novel piece |
|---|---|---|
| **1. Premise generation** | LLM (Llama2-7B) generates NL premises (declarative + conditional sentences only) for the input question | Constraining the LLM to two sentence types |
| **2. NL→logical-form translation** | LLM (CodeQwen1.5-7B) translates premises to Clingo ASP code | **Grammar-constraining program**: masks logits of tokens that violate Clingo's formal grammar (e.g., after `=`, only `True`/`False` tokens survive). Cuts error rate from 17.8% → 1.5%. |
| **3. Symbolic solving** | Clingo (ASP solver) deductively derives the answer | Deterministic, interpretable; final output is necessarily true given accurate premises |

**Bottleneck identified:** Stage 2 *semantic* errors (not syntactic — those are solved by the constraining program). Semantic errors = inconsistent naming conventions across premises ("1519" vs "16th century"), or code that doesn't correspond to the premise.

### 1.3 The comparative verdict (Table 5)

| Criterion | Integrative (LNN) | Hybrid (LLM-SS) |
|---|---|---|
| Symbolic reasoning | ✓ (limited — fixed inter-layer connections) | ✓ (full — external solver) |
| Interpretability | ~ (scales poorly — 640 neurons uninterpretable) | ✓ (Clingo code is readable) |
| Retains LLM abilities | ✗ (replaces Transformer) | ✓ (Stage 1 keeps the LLM) |

**Conclusion:** hybrid is more promising for *general* logical reasoning.

---

## 2. Distillation

### 2.1 What's training-only → riir-train

- **LNN gate-weight training** via gradient descent on `w₁..w₄` (Adam, lr 0.01, 1000 iterations for gate-ID; 200 epochs for classification).
- **LLM fine-tuning** for Stage 1/2 premise generation and Clingo-code translation (Llama2-7B, CodeQwen1.5-7B).
- **Convergence-speed comparison** LNN vs LGN — a training-side benchmark.

### 2.2 What's modelless but already shipped

| Paper piece | Shipped cousin | Plan / Research | Notes |
|---|---|---|---|
| **LNN AND-OR-NOT gates** (integrative) | **DDTree AND-OR decomposition** | P190, **R184**, P209 | "The DDTree IS the LNN, but constructed on-the-fly from marginals — no training needed" (R184 §0). Our 3-gate AND-OR-NOT basis is **functionally complete** — expresses all 16 gates — so we lose no expressiveness vs the paper's 16-gate enumeration. |
| **16-gate relaxation** `σ(w₁a+w₂b+w₃(ab)+w₄)` | DDTree marginals + bandit scores as path weights | R184, P209 | At inference, LNN discretizes to nearest gate (4-float → 4-bit lookup). Our DDTree paths carry the same information as continuous weights; we don't need the discretization step because we never trained the 4 floats. |
| **FOL extraction from text** | `ConstraintPruner` trait / `SynPruner` | P007, P209 T1 | SynPruner extracts syntactic constraints from Rust code; FolPruner (P209) extracts FOL constraints from prompts via regex + keyword tables — **no LLM call**, <1µs. |
| **Reward-weighted gate connections** | `BanditPruner` (UCB1 / Thompson Sampling) | P030, P209 T3 | Already has reward signals on DDTree branch selection. |
| **Rule extraction from trained LNN** | **Rule extraction from DDTree paths** (TOP-K highest-scoring paths → FOL rules) | P209 T2 | Modelless inference-time rule extraction; no training needed (the paper needs training to learn the weights first). |
| **Interpretable decision traces** | `decision_trace` feature | P209 T4 | Opt-in debug/audit trace explaining *why* tokens were chosen. |
| **Stage 2 grammar-constrained decoding** (logit masking) | **`spec_compile` SpecDFA + ConstraintPruner** | **P259** | DFA-based format validation (email/phone/date/URL) + TokenBias marginals (-20.0 for blocked, 0.0 for allowlist) + AND/OR chain composition. The paper's CFG-based Clingo-grammar masking is a minor variant; our `SynPruner` (rustc/syn AST) already handles the stricter context-sensitive case. |
| **Stage 3 symbolic solver (Clingo/ASP)** | `rustc`/`syn` validator (P007), WASM validator (artifact def), Lean4 agent (R198), bisimulation ASP operator inference (R308/P324) | P007, P324 | We ship multiple Stage-3 solvers. R308/P324 explicitly notes the ASP solver step is "out of scope for the open primitive — that's the heavier-weight CWM path." Clingo is one more external solver option; not a modelless primitive. |
| **3-stage modular framework** (premises → translate → solve) | NS-CSG three-mode router (L4R/R4L/LR) | R186, P211 | Our router dynamically selects the mode per decode step; the paper's framework is a static 3-stage decomposition. |
| **Hybrid > integrative verdict** | Our architecture already chose hybrid | R186, R308 | Strategic validation, not a primitive. |

### 2.3 Fusion — none novel

The prior-art surface is dense. Every individual piece of both approaches ships in our quintet:

- **Integrative half** → R184/P209 (DDTree IS the LNN) ships the modelless equivalent, stronger because no training is needed.
- **Hybrid half** → P007 (SynPruner) + P259 (spec_compile grammar constraint) + R308/P324 (ASP operator inference) ship all three stages.

The only genuinely additive items are:
1. **The explicit 16-gate enumeration** as a named basis. Our AND-OR-NOT is functionally complete (expresses all 16), so this is a redescription, not a capability gain. The 4-weight-per-neuron parameterization is only useful *if you train the weights* → riir-train.
2. **The Clingo ASP solver** as a Stage-3 tool. We lack it specifically, but we have rustc/syn, WASM, Lean4, and the deferred ASP path from R308/P324. Adding Clingo is a tool-integration task, not a modelless primitive.
3. **The comparative verdict** as a design principle. Validates our choices; codifies what we already practice.

### 2.4 Latent vs raw boundary (mandatory check)

Not applicable — no new boundary-crossing behavior. The paper operates entirely in the symbolic/text domain (NL premises → Clingo code → boolean answer). No latent-state operation, no sync-boundary implication.

---

## 3. Verdict

**Tier: PASS.** Both approaches already distilled at higher fidelity. Training recipes → riir-train. Comparative verdict validates existing architecture.

| Gate | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **FAIL.** Integrative LNN → R184/P209 ("DDTree IS the LNN"). Hybrid LLM-SS → P007 + P259 + R308/P324. Every piece ships. |
| **Q2** New class of behavior? | **FAIL.** "Differentiable logic gates in-network" = DDTree AND-OR (shipped). "LLM + external symbolic solver" = SynPruner + WASM + Lean4 + bisimulation-ASP (shipped). |
| **Q3** Selling point? | **FAIL for new selling point.** "Interpretable logical reasoning via hybrid LLM+solver" IS the SynPruner/bisimulation-ASP selling point — already ours. |
| **Q4** Force multiplier? | **YES — but only as a validation** of capabilities we already have. Connects R184, P007, P259, R308 — all already connected. |

### Latent-space reframing check (mandatory per skill — primary framing)

- **HLA framing:** the 16-gate logic neuron operates on 2 scalar activations → 1 scalar. *Could* be a per-NPC affect combiner (valence ∧ arousal → action readiness). **But** our existing affect composition (R144 Functional Emotions, R156 Clifford wedge, R276 Personality-Weighted Composition, R302 CommittedFieldBlend) uses **continuous** sigmoid/dot-product/Clifford-wedge ops — strictly more expressive than binary gates. Restricting to binary loses information for continuous affect dimensions. The 16-gate basis is a step backward for HLA.
- **Latent functor framing:** a logic gate IS a rank-0→rank-0 functor; the 16 gates are the 16 functors on 2 binary inputs. Our functors operate on continuous latents — binary restriction loses the continuous functor's expressiveness.
- **CGSP framing:** not relevant (no curiosity/exploration signal).
- **Neuron-shard framing:** the 4 weights per gate could be frozen as a `MerkleFrozenEnvelope` artifact class ("logic personality"). But we already freeze/thaw LoRA adapters, functor direction vectors, kernel snapshots, NeuronShard style_weights — another artifact class adds no new mechanism.
- **LatCal framing:** the `oᵢ > 0.5 → 1` discretization is a trivial fixed-point bridge. LatCal already handles deterministic commitment at higher fidelity.
- **DEC framing:** not relevant (logic gates are rank-0, not differential forms).

**No latent-space reframing yields a new capability.** The binary restriction inherent to logic gates is a step down from our continuous latent operations for every reframing angle.

### Honest one-line reasoning

This is a comparative study, not a primitive paper. Both approaches it compares are already distilled in our quintet — the integrative LNN by R184/P209 (modelless, stronger because no training), the hybrid LLM-SS by P007/P259/R308-P324 (all three stages shipped). The comparative verdict (hybrid > integrative) validates our existing architecture. The three specific additives (16-gate enumeration, Clingo ASP solver, the verdict itself) are either redescription of shipped expressiveness, external tool integration, or strategic validation — none is a new modelless inference primitive.

---

## 4. Routing

- **Training recipes** (LNN gate-weight gradient descent; LLM fine-tuning for premise generation and Clingo-code translation) → **riir-train** (one-line note, out of scope for this workflow).
- **Open primitive** → none new. The 16-gate parameterization is a redescription of DDTree AND-OR-NOT (functionally complete). The Clingo ASP solver is an external tool, not a primitive.
- **Architectural guide** → none required. R184 (FOL-LNN) + R186 (three-mode router) + R308 (bisimulation ASP) already cover the selling points at higher fidelity.
- **Plan** → none required. No new code needed.
- **Architecture-doc follow-up (optional, low priority):** if `.docs/` ever needs a "why hybrid over integrative" design-principle note, this paper's Table 5 is a clean citation. The principle is already implicit in our choice to ship SynPruner (rustc/syn) + WASM validator + Lean4 agent + bisimulation-ASP rather than an in-network logic-gate layer. Not actioned now.

---

## TL;DR

arXiv:2508.03366 is a comparative study of two neurosymbolic approaches (integrative LNN vs hybrid LLM-SS) that concludes hybrid > integrative for general logical reasoning. Both approaches are already distilled in our quintet at higher fidelity: integrative → R184/P209 ("the DDTree IS the LNN, modelless, no training"); hybrid → P007 (SynPruner) + P259 (spec_compile grammar-constrained decoding) + R308/P324 (bisimulation ASP). The paper's specific additives (16-gate relaxation formula, Clingo solver, the comparative verdict) are redescription / external tool / strategic validation respectively — none is a new modelless inference primitive. The verdict validates our existing hybrid-first architecture. Training recipes → riir-train. Closing this research path; no plan, no guide, no open primitive.
