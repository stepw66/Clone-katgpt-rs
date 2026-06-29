# Research 344: Implicit Language Models are RNNs — Balancing Parallelization and Expressivity

> **Source:** [Implicit Language Models are RNNs: Balancing Parallelization and Expressivity](https://arxiv.org/abs/2502.07827) — Schöne*, Rahmani*, Kremer, Falck, Ballani, Gladrow (Microsoft Research Cambridge / Hessian AI / TUD Dresden). ICML 2025 Spotlight. arXiv:2502.07827v3, 12 Jun 2025. **Code:** github.com/microsoft/implicit_languagemodels
> **Note on the assignment URL:** the pre-assigned fetch URL pointed at `arxiv.org/pdf/2504.01280`, which is *not* this paper — it is an unrelated economics paper (Schipper & Zhang, "Matching, Unanticipated Experiences, Divorce, Flirting, Rematching"). The correct arxiv ID is **2502.07827**. This was recovered by web-searching for the title + author list.
> **Date:** 2026-06-29
> **Status:** Done
> **Survey context:** Fills gap **G2** from Research 325 (§7.2) — *"SSM-block-as-fixed-point-solver framing may unlock a modelless recurrent kernel that halts adaptively per-NPC"*. Pre-flight skepticism: *"Pre-check: does the fixed-point solver admit a modelless construction, or does it require learned dynamics?"*
> **Related Research:** 035 (Attractor Models — FP refinement on a backbone), 073 (LT2 Looped), 079 (EqR Equilibrium Reasoners), 097 (Training-Free Looped), 113 (NITP implicit token prediction), 192 (NextLat Coconut), 230 (SSD Duality / Mamba-2), 242 (MicroRecurrentBeliefState design), 265 (CoFRe/FP-MGM sibling of FPRM), **266 (FPRM — closest cousin, damped FP halting on transformer loop)**, **273 (ELT elastic any-time)**, **276 (Personality-Weighted Latent Layer — ships MicroRecurrentBeliefState; Plan 276 benchmark is the decisive null-result prior art)**, **282 (LoopCoder-V2 — gain/cost loop halting, second-closest cousin)**, 286 (Depth-Invariance Diagnostic), 325 (Survey, §7.2 G2)
> **Related Plans:** 108 (LT2 — `LoopMode::WeightShared`), 119 (EqR Convergence Selector), 136 (TF-Loop — `LoopMode::TrainingFree` K-stage RK β=0.5), 152 (Newton-Schulz cubic FP / River-Valley Diagnostics), 231 (PathwayTracker stability exit), 276 (MicroRecurrentBeliefState GOAT — AttractorKernel demoted, LeakyIntegrator promoted), 283 (Self-Advantage Gate residual halt), 304 (Gain/Cost Loop Halting)
> **Classification:** Public

---

## TL;DR

The paper trains an SSM block (Mamba2) or a Transformer block (Llama3) as a **Deep Equilibrium Model** — `z* = F_θ(z*, x)` is the model output, found by self-iterating the block until convergence (≤16 self-iterations during training for the S5 word problem; ≤24/32 for the 1.3B PILE LM; halt at relative residual `|z^(s) − z^(s−1)| / |z^(s−1)| < ε=0.05`). The headline results: (a) **Theorem 1** — fixed-point iteration of a generic-weight SSM block yields a **non-linear, non-diagonal state-to-state Jacobian**, i.e. implicit SSMs recover RNN expressivity that linear SSMs provably lack (TC⁰ ⊊ NC¹ = regular languages = RNN-expressive class); (b) **≤8 self-iterations during training** suffice to solve the S5 word problem out-of-distribution (vs. L=256 sequential steps a classical RNN needs); (c) implicit Mamba2/Llama3 up to **1.3B params / 207B tokens** outperform explicit baselines on D-PILE perplexity + Hellaswag/ARC/Lambada; (d) **simultaneous (parallel) mode and sequential (per-token) mode converge to functionally equivalent fixed points** (97.6% token match on 3M tokens), enabling autoregressive generation from a model trained in parallel.

**Distilled for katgpt-rs (modelless, inference-time):** Almost nothing. The paper's value lives overwhelmingly on the **training side** — phantom gradients (Eq. 5, truncated von-Neumann series Jacobian-vector products with smoothing λ), DEQ implicit-function-theorem differentiation (Eq. 4), the bounded-then-free curriculum (`(4+1)` phantom-gradient phase → `(24+4)`/`(32+4)` free FP phase), and the **trained-weight** fact that the fixed-point operator converges to a *useful* non-linear RNN-equivalent dynamics. The **inference artifact** — a single weight-tied block iterated to a fixed point with a relative-residual halt — is already shipped in this codebase as `LoopMode::WeightShared { loop_count }` (Plan 108) and the strictly-more-sophisticated `LoopMode::TrainingFree` (Plan 136, K-stage RK β=0.5 = functionally a damping factor). The closest cousins (FPRM 266, LoopCoder-V2 282, ELT 273, Self-Advantage Gate 283) already cover the FP-halting / gain-cost-halt / any-time / residual-halt design space. The MicroRecurrentBeliefState benchmark (`.benchmarks/276_micro_belief_goat.md`) is the **decisive empirical prior art**: iterating a per-NPC state kernel to a fixed point (the `AttractorKernel` Family A) **fails G1.4 latency (~273 ns) AND G2.1 coherence (569× more flip-flops than the leaky integrator)** — exactly the "random-init attractor flip-flops instead of converging to useful dynamics" regime. The paper's RNN-equivalence theorem requires *generic trained weights*; the modelless attractor we shipped does not satisfy that precondition.

**Verdict: Gain** (training value → riir-train, see §3.5 modelless-unblock protocol failure; modelless inference delta is a config variant of shipped primitives, not a new capability class). **Not Super-GOAT, not GOAT.** This note exists to (a) close survey gap G2, (b) document the §3.5 modelless-unblock failure for the FP-solver-needs-trained-dynamics question, and (c) record the prior-art surface so a future agent doesn't re-evaluate. **No plan created in this session.**

---

## 1. Paper Core Findings

### 1.1 Architecture — DEQ on a single weight-tied SSM/Transformer block

Following Bai et al. (2019), the model is defined implicitly via the fixed point of an input-conditional network. For an SSM block (Mamba2 backbone):

```
h_t^(s) = Λ(z_t^(s−1), x_t) · h_{t−1}^(s) + u(z_t^(s−1), x_t)     (Eq. 6)
z_t^(s) = f_θ(z_t^(s−1), h_{t−1}^(s), x_t)                          (Eq. 7)
```

Iterated until `z_t^(s) → z_t*` (relative residual < ε). In the limit `s → ∞`:

```
h_t* = Λ(z_t*, x_t) · h_{t−1}* + u(z_t*, x_t)                      (Eq. 8)
z_t* = f_θ(z_t*, h_{t−1}*, x_t)                                     (Eq. 9)
```

Two evaluation modes emerge from exchanging the `s` (depth) and `t` (sequence) loops:
- **Simultaneous mode** — find fixed points for all t in parallel. Exploits SSM/Transformer sequence parallelism. Used for training.
- **Sequential mode** — resolve one token at a time, passing only converged hidden states forward. Constant memory. Used for autoregressive generation.

### 1.2 Theorem 1 — implicit SSMs recover RNN expressivity

An implicit SSM (Eqs. 8–9) with generic weights yields a **non-linear, non-diagonal state-to-state transition** `h_{t−1}* → h_t*`. The state-to-state Jacobian (Eq. 10) contains a diagonal term `Λ(z_t*, x_t)` (the linear SSM recurrence) **plus** non-diagonal corrections from the implicit dependency of `z_t*` on `h_{t−1}*` via the implicit-function-theorem derivative `∂φ/∂h_{t−1}*`. **Generic weights** is the load-bearing precondition — random or trivial weights do not satisfy the theorem's "continuously differentiable φ" with non-trivial non-linearity.

Circuit-complexity framing: explicit transformers and SSMs are bounded in **TC⁰** (constant-depth, poly-width, unbounded-fan-in majority gates). RNNs reach **NC¹** (= regular languages, via Barrington's theorem on the S5 word problem). TC⁰ ⊊ NC¹ is open but widely conjectured. The implicit construction lifts an SSM from TC⁰ into NC¹ by giving it adaptive effective depth.

### 1.3 Phantom gradients — the training primitive (→ riir-train)

Naive differentiation through the fixed-point solver is prohibitive. The **phantom gradient** (Geng et al. 2021, Eq. 5) computes:

```
d∂Φ/∂θ = λ · ∂F_θ/∂θ |_{z*} · Σ_{i=0}^{k−1} (λ · ∂F_θ/∂z |_{z*} + (1−λ)·I)^i
```

A small smoothing λ ∈ (0, 1] keeps the condition number bounded; truncation length k sets approximation accuracy. Memory cost is **proportional to k, not to the depth of forward iterations** — this is what makes deep FP training feasible. This is unambiguously training machinery → riir-train.

### 1.4 Curriculum — `(s+k)` notation, bounded → free

- **Bounded phase (4+1):** 4 self-iterations + 1 phantom gradient step. Trains on 80% of D-PILE.
- **Free phase (24+4) for Mamba2 / (32+4) for Llama:** 24/32 self-iterations + 4 phantom gradient steps. Trains on the remaining 20%. Stop threshold ε=0.05.

The 80/20 split was empirically tuned on a 130M Mamba2 model; 70% bounded overfits, 90% bounded underfits.

### 1.5 Empirical headline results

- **S5 word problem** — implicit Mamba2 (1 layer, unbounded test-time iterations) solves sequences of length 128 trained on length 32. Explicit Mamba2 (16 layers) cannot length-extrapolate past 32.
- **Sparse hard tokens** — implicit Mamba2 generalizes from D₀.₁ (10% hard S5 tokens) to D₀.₅ (50% hard) OOD with as few as **8 training-time self-iterations** (vs. L=256 sequential RNN steps — a 32× parallelization factor).
- **CatbAbI** — implicit Mamba2 1-layer beats explicit Mamba2 1-layer on most of the 20 reasoning tasks.
- **D-PILE 207B pretrain, 130M–1.3B** — implicit models consistently beat explicit baselines on perplexity and most downstream tasks (LAMBADA/Hellaswag/ARC-E/ARC-C/PIQA); implicit Llama 760M ≈ explicit Llama† 1.3B.
- **Simultaneous ↔ sequential duality** — 97.6% token match between the two modes on 3M tokens (1.3B Mamba2). This is the first demonstration that DEQ-style models trained in parallel can be deployed for autoregressive generation.
- **Length extrapolation** — implicit Mamba2 maintains lower perplexity at 4× the training context length, while explicit Mamba2 degrades catastrophically (760M: 9.23 → 231.25 ppl from 2048 → 16384 tokens).

### 1.6 Path independence

Gradients of a FP iteration depend only on the fixed point, not the path to it (implicit function theorem). This is why simultaneous training → sequential inference works. Related to Anil et al. 2022 (path-independent DEQs).

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper ↔ codebase)

| Paper term | Codebase equivalents (≥2) | Where it ships |
|---|---|---|
| "implicit SSM" / "implicit language model" | `LoopMode::WeightShared`, `LoopMode::TrainingFree`, "weight-tied looped block" | `katgpt-rs/src/looped.rs`, Plans 108/136 |
| "fixed-point iteration until convergence" | "convergence halting", "FP optimizer", `Newton-Schulz cubic FP`, `ResidualRelevanceScorer.is_converged` | Plans 152 (NS cubic), 085 (L2/KL residual scorer), 266 (FPRM damping) |
| "self-iteration" / "depth iteration" | "looped forward", "weight-shared loop", `latent_functor` application cycle | `LoopMode::*` |
| "phantom gradient" / "DEQ implicit differentiation" | (none — this is training machinery → riir-train) | — |
| "damped fixed-point optimizer (FPOpt)" | `FPRM FPOpt Algorithm 1`, "patience-based geometric η decay", K-stage RK β=0.5 damping | Research 266, Plan 136 |
| "gain/cost halting" (cross-paper) | `GainCostLoopHalter`, "coherence-decay re-estimation trigger" | Research 282, `latent_functor/reestimation.rs` |
| "non-linear non-diagonal state transition" / "RNN expressivity" | `AttractorKernel` (failed), `LeakyIntegrator` (won), "per-NPC recurrent belief kernel" | Plan 276 — **null result is decisive prior art** |
| "simultaneous vs sequential mode" | "parallel prefill vs autoregressive decode", "batched vs streaming loop" | standard inference modes; the duality is already true for any weight-shared loop |
| "curriculum (4+1) → (24+4)" | (training → riir-train) | — |

### 2.2 What we already ship (the prior-art surface)

**The FP-halting family is the single best-covered primitive family in this codebase.** Direct mapping:

| Shipped primitive | Paper analog | Note |
|---|---|---|
| `LoopMode::WeightShared { loop_count }` (Plan 108, default) | "implicit SSM/Transformer block iterated K times" | **Identical** — weight-tied block, fixed K. The only delta is the paper's `K → ∞` (until-convergence) framing, which our convergence-halt cousins below cover. |
| `LoopMode::TrainingFree` (Plan 136, default) | "implicit block with ODE-motivated damping" | **Strictly more sophisticated** than the paper's plain FP iteration — K-stage RK sub-stepping with β=0.5 is functionally a damping factor that the paper achieves only via the FPRM-style pre-norm + residual scaling (Research 266). |
| **FPRM damped FP optimizer** (Research 266, plan-only behind `fpopt_halt`) | Paper's "fixed-point solver halts on convergence" | Closer cousin than this paper — patience-based geometric η decay when residual stalls. |
| **LoopCoder-V2 gain/cost halting** (Research 282, Plan 304) | Paper's "halt when refinement irrelevant" | **Strictly more principled** than the paper's single-curve (gain-only) residual halt — tracks two crossing curves (gain vs cost). |
| **Self-Advantage Gate** (Plan 283) | Paper's "halt on residual" | Per-NPC residual-halt cousin. |
| **ELT elastic any-time** (Research 273) | Paper's "adaptive compute per token" | Elastic any-time inference, broader framing. |
| **PathwayTracker stability exit** (Plan 231) | Paper's "convergence halt" | Stability-based halt, shipped. |
| **River-Valley Diagnostics** (Plan 152) | Paper's "fixed point reached" | Effective-rank trajectory signal — a richer convergence signal than the paper's L2 residual. |
| **Newton-Schulz cubic FP** (Plan 152) | Paper's "fixed-point solver" | A *specific* FP solver the paper does not use (the paper uses plain iteration). |
| **ResidualRelevanceScorer** (Plan 085, default-on, GOAT 6/6) | Paper's `|z^(s) − z^(s−1)| / |z^(s−1)| < ε` | Generic L2/KL residual scorer with `is_converged(residual, tolerance)` — **exactly the paper's halt test, already shipped as a default-on primitive.** |
| **MicroRecurrentBeliefState / AttractorKernel** (Plan 276, opt-in, demoted) | Paper's "iterate state kernel to FP" | **Decisive null result** — see §2.3. |

### 2.3 The decisive prior-art data point — Plan 276's AttractorKernel null result

The user's prompt flagged this as critical prior art. It is. From `.benchmarks/276_micro_belief_goat.md`:

- `AttractorKernel` (Family A, `s_t = 2σ(W_s·s + W_x·x + b) − 1`, iterated K times per tick) **failed G1.4 latency (~273 ns, target <100 ns) AND G2.1 coherence (569× more flip-flops than LeakyIntegrator)**.
- `LeakyIntegrator` (Family C, monotone additive with `±max_delta` clamp) **won** — byte-identical to `evolve_hla`, promotable.
- The `evolve_hla` refactor made `evolve_hla` delegate to the ungated `leaky_core::leaky_step` — zero behavior change.

**Why this is decisive for this paper:** Schöne et al.'s Theorem 1 requires **generic trained weights** for the implicit SSM to yield a non-trivial non-linear state transition. With random or untrained weights (the modelless regime — the only regime we can ship without riir-train), the FP iteration `z* = lim_s T^s(z)` either:
1. Does not converge (diverges or oscillates) — exactly the AttractorKernel flip-flop failure mode.
2. Converges to a trivial fixed point (identity / zero) that does not exhibit the RNN-equivalent non-linearity.

This is not a coincidence — it is the **same mathematical phenomenon**. The paper's RNN-equivalence is a statement about the *trained* FP operator's Jacobian, not about FP iteration in general. The MicroRecurrentBeliefState null result is the empirical proof that the modelless attractor regime does not satisfy the theorem's preconditions.

### 2.4 Fusion — what novel combination could the paper unlock?

Following the fusion protocol, the candidate fusions are:

| Fusion | Novel capability? | Verdict |
|---|---|---|
| Paper × FPRM (266): use FPRM's damped FP optimizer as the solver for an implicit SSM block. | No new capability — FPRM already covers transformer-loop FP halting; substituting an SSM block is a config variant (`LoopMode::WeightShared` over an SSM layer instead of a transformer layer). | Gain at most — already covered by the LoopMode abstraction. |
| Paper × LoopCoder-V2 (282): apply gain/cost halting to an implicit SSM's self-iterations. | No new capability — LoopCoder-V2's halting is architecture-agnostic; applying it to an SSM block instead of a transformer loop is again a config variant. | Gain at most. |
| Paper × MicroRecurrentBeliefState (276): "iterate an NPC's affect latent to a fixed point = deliberation until emotional equilibrium." | **This is the strongest per-NPC reframing the user's prompt flagged.** But Plan 276 already tried it (AttractorKernel) and the null result killed it: random-init attractors flip-flop. The paper's RNN-equivalence requires trained dynamics; per-NPC deliberation via FP iteration would require either (a) a trained attractor per NPC (→ riir-train, doesn't scale to 10k NPCs) or (b) a leaky integrator that doesn't actually iterate to a FP (what we already ship). | **Blocked by §3.5 — see §3 below.** Not a fusion we can ship modellessly. |
| Paper × latent_functor/reestimation.rs: "re-estimation trigger fires when an NPC's HLA state fails to converge under its current functor." | The reestimation trigger already fires on coherence decay (`coherence < tau_reest`) — which is functionally the "FP not reached" signal. Adding a FP-iteration count as an additional trigger is a parameterization, not a new mechanism. | Gain at most. |
| Paper × DEC Stokes substrate (Plan 251, Research 219): "iterate Hodge decomposition to a FP = manifold equilibrium." | Speculative. Hodge decomposition is a linear-algebraic decomposition (exact in one shot via the DEC operators), not an iterative FP problem. No clear gain. | No clear path. |

**Honest fusion verdict:** No fusion in this session produces a capability class that the shipped primitives lack. The FP-halting family is saturated; the per-NPC deliberation angle is blocked by the §3.5 modelless-unblock failure (see §3). The fusion protocol's standing rule applies: if every fusion is a config variant of shipped primitives, the verdict is Gain at most, not Super-GOAT.

---

## 3. Verdict

### 3.1 §3.5 modelless-unblock protocol — MANDATORY pre-check

The survey G2 entry explicitly asked: *"does the fixed-point solver admit a modelless construction, or does it require learned dynamics?"* Run the protocol:

**Gate:** The paper's RNN-equivalence (Theorem 1) and the empirical S5/length-extrapolation results require the FP operator `T = f_θ(·)` to be **trained**. The construction is `z* = lim_s T^s(z)`; for the limit to exist AND produce useful non-linear dynamics, T must be trained.

**Does the failure have a SYSTEMATIC, characterizable cause (§3.5)?**
- The "failure" here is: a modelless (random-init or deterministically-constructed) FP operator does not produce the paper's RNN-equivalent non-linear state transition.
- This is **not a bias** (like AC-Prefix G1's doubled signal — Research 313). It is the **absence of a dynamics**. There is no systematic correction to apply; the dynamics itself is what training produces.

→ NO systematic, characterizable bias. **Proceed to the genuine-riir-train-dependency branch.**

**Check the three paths anyway (defensive):**
1. **Freeze/thaw snapshot correction** (path 1) — N/A. Freeze/thaw swaps a frozen *snapshot*; it does not synthesize a *dynamics*. The paper's value is the dynamics, not a state.
2. **Raw/lora reader-writer hot-swap** (path 2) — N/A. A deterministically-constructed LoRA overlay corrects a *systematic bias* in an existing computation; it cannot synthesize an FP-convergent non-linear operator from nothing. There is no closed-form construction of "the LoRA that makes a random-init Mamba2 block converge to an S5-solving FP" — that construction *is* training.
3. **Latent-space projection/gate** (path 3) — N/A. The non-linear state transition is a *result* of the FP iteration, not a correctable bias on a separate computation. Projecting onto a correction direction presupposes the dynamics exists.

→ **All three paths fail. Genuine riir-train dependency.** The paper's value (RNN-equivalent non-linear state transitions via FP iteration) requires gradient descent to construct the operator T. Document and stop — no plan in this repo.

### 3.2 Novelty gate Q1–Q4

| Q | Answer | Evidence |
|---|---|---|
| **Q1. No prior art?** | **NO** — saturated prior art. | §2.2 above: FPRM (266), LoopCoder-V2 (282), ELT (273), Self-Advantage Gate (283), PathwayTracker (231), River-Valley Diagnostics (152), Newton-Schulz cubic FP (152), ResidualRelevanceScorer (085, default-on), LT2 WeightShared (108), TF-Loop TrainingFree (136), AttractorKernel null result (276). The FP-halting + looped-block design space is one of the best-covered in the codebase. |
| **Q2. New class of behavior?** | **NO.** | The paper's contribution is an *architectural choice* (SSM-block-as-FP-solver vs transformer-loop-as-FP-solver) + a *training curriculum*. The inference artifact is a weight-tied block iterated to a FP — which is exactly what `LoopMode::WeightShared` already does. No capability that the shipped primitives lack. |
| **Q3. Product selling point?** | **NO** (modellessly). | The paper's selling point ("implicit models beat explicit on perplexity + length extrapolation") is a *training* result, not an inference primitive. We cannot finish the sentence "Our NPCs do X that no competitor can" without riir-train producing the trained FP operator. |
| **Q4. Force multiplier?** | **NO.** | Does not connect to ≥2 pillars in a novel way. The FP-halting family already multiplies the reasoning/freeze-thaw/CLR pillars; this paper adds no new connection. |

**All 4 NO → not Super-GOAT. Proceed to GOAT/Gain.**

### 3.3 GOAT vs Gain

- **GOAT** requires a provable gain (latency/quality/security) over an existing approach on the modelless path. The paper provides *no* modelless gain — every empirical result in the paper is a *training* result (phantom gradients, curriculum, 207B-token pretrain).
- **Gain** is the appropriate tier for: (a) closing survey gap G2 with an honest "no, this is training-bound" verdict, (b) documenting the §3.5 modelless-unblock failure for the FP-solver-needs-trained-dynamics question, (c) recording the prior-art surface so a future agent does not re-evaluate, and (d) one narrow inference-time config insight (see §3.4 below).

**Verdict: Gain.** No plan created in this session. No Super-GOAT guide created. No open primitive created.

### 3.4 The one narrow modelless insight worth recording

The paper's *simultaneous ↔ sequential mode duality* (97.6% token match, §1.5) is a useful **empirical validation** of a property our `LoopMode::WeightShared` already has by construction (path-independence of the FP — gradients depend only on the FP, not the path, per the implicit function theorem). This is **not a new primitive** — it is supporting evidence that our existing simultaneous-prefill / sequential-decode loop is correct. No code change; the duality is already true.

The paper's *pre-norm + residual scaling* architectural choice (mirrored in FPRM Research 266) is a config recommendation for any future looped-block we train — but that is a riir-train config note, not a katgpt-rs primitive.

### 3.5 → riir-train (out of scope, noted for completeness)

The following are unambiguously training-side and route to `riir-train/.research/` if pursued:

- **Phantom gradient** method (Eq. 5) — truncated von-Neumann series Jacobian-vector products with smoothing λ. Constant-memory training of deep FP models.
- **DEQ implicit-function-theorem differentiation** (Eq. 4) — `∂Φ/∂θ = −J⁻¹_{G,z*} · ∂F_θ/∂θ`.
- **Bounded-then-free curriculum** — `(4+1)` phantom-gradient phase → `(24+4)`/`(32+4)` free FP phase, 80/20 split.
- **Theorem 1's "generic weights" precondition** — the trained FP operator's non-linear non-diagonal Jacobian is a *training* result, not a modelless construction.
- **Pre-norm + residual scaling** as a stable loop configuration (mirrors FPRM Research 266's architectural move).
- **Length-extrapolation** gains — a training-data + curriculum result.

If riir-train ever produces a trained implicit-SSM artifact, the *inference* path is already shipped here (`LoopMode::WeightShared` over an SSM block + `ResidualRelevanceScorer.is_converged` halt). The artifact would load via the existing model loader; no new katgpt-rs primitive needed.

---

## 4. What this note prevents (canonical failure modes averted)

1. **False Super-GOAT on "SSM-block-as-FP-solver is novel."** It is not — `LoopMode::WeightShared` is architecture-agnostic; the SSM-vs-transformer choice is a config variant. The FP-halting family (FPRM, LoopCoder-V2, ELT, Self-Advantage Gate, PathwayTracker, River-Valley Diagnostics, Newton-Schulz, ResidualRelevanceScorer) saturates the design space.

2. **False Super-GOAT on "per-NPC deliberation via FP iteration."** Plan 276 already tried this (AttractorKernel) and the null result killed it: random-init attractors flip-flop (G2.1 coherence FAIL, 569× more flip-flops than leaky). The paper's RNN-equivalence requires trained dynamics; per-NPC deliberation via FP would require a trained attractor per NPC, which doesn't scale to 10k NPCs and is firmly riir-train territory.

3. **Mis-routing the paper's value to katgpt-rs.** The paper's value is overwhelmingly training-side (phantom gradients, DEQ differentiation, curriculum, 207B-token pretrain). Routing any of it here would violate the modelless-first mandate (constraint #1).

4. **Re-evaluation by a future agent.** This note + the §3.5 modelless-unblock failure documentation + the prior-art table (§2.2) should prevent any future session from re-running the novelty gate on this paper or its close cousins.

5. **Treating the wrong arxiv ID as canonical.** The assignment URL (`2504.01280`) points at an unrelated economics paper. The correct ID is `2502.07827`. Recorded in the header so the next agent does not repeat the fetch error.

---

## 5. Action items

- [x] **Close survey gap G2** (Research 325 §7.2). This note is the closure. Update Research 325 §7.5 G2 line from "pre-check" to "closed — Gain, training-bound, see Research 344" if a future doc sweep touches it.
- [-] **No plan in katgpt-rs.** The modelless inference delta is a config variant of shipped primitives; no new primitive warranted.
- [-] **No Super-GOAT guide in riir-ai.** The per-NPC deliberation-via-FP fusion is blocked by the §3.5 modelless-unblock failure (Plan 276 AttractorKernel null result).
- [-] **→ riir-train** (if pursued): phantom gradient method, DEQ implicit differentiation, bounded-then-free curriculum, pre-norm + residual scaling config. Out of scope for this workflow; noted for completeness.
- [-] **Track as context, not action:** the simultaneous ↔ sequential mode duality (97.6% token match) is empirical validation of a property our `LoopMode::WeightShared` already has by construction. No code change.

---

## TL;DR

**Verdict: Gain.** Schöne et al. (ICML 2025 Spotlight, arXiv:2502.07827 — *not* 2504.01280, which is an unrelated economics paper) train an SSM/Transformer block as a Deep Equilibrium Model: iterate the block until its output converges to a fixed point, recover RNN-equivalent non-linear non-diagonal state transitions (Theorem 1), scale to 1.3B / 207B tokens with phantom gradients + a bounded-then-free curriculum, and demonstrate simultaneous-train / sequential-decode duality (97.6% token match). **Almost none of this is modelless-distillable.** The paper's value is overwhelmingly training-side (phantom gradients, DEQ implicit differentiation, curriculum, "generic trained weights" precondition) → riir-train. The inference artifact (weight-tied block iterated to a FP with a relative-residual halt) is already shipped as `LoopMode::WeightShared` (Plan 108) + `ResidualRelevanceScorer.is_converged` (Plan 085, default-on), and the FP-halting design space is one of the most saturated in this codebase (FPRM 266, LoopCoder-V2 282, ELT 273, Self-Advantage Gate 283, PathwayTracker 231, River-Valley Diagnostics 152, Newton-Schulz 152, AttractorKernel 276). **§3.5 modelless-unblock protocol fails on all three paths** — the paper's RNN-equivalence is the *absence of a dynamics* in the modelless regime, not a correctable bias; the Plan 276 AttractorKernel null result (G1.4 latency FAIL ~273ns, G2.1 coherence FAIL 569× flip-flops vs leaky) is the empirical proof. Novelty gate Q1–Q4 all NO. No plan, no Super-GOAT guide, no open primitive created in this session. The one narrow modelless insight (simultaneous ↔ sequential mode duality) is empirical validation of a property our existing loop already has by construction — no code change.
