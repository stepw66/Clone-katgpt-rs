# Research 298: Inverting the Bellman Equation — Closed-Form World-Model Extraction

> **Source:** [Inverting the Bellman Equation: From Q-Values to World Models](https://arxiv.org/pdf/2606.21173) — Letcher, Fellows, Goldie, Richens, Foerster, Richardson (FLAIR Oxford, Google DeepMind, Mila), arxiv 2606.21173v1, 19 Jun 2026
> **Reference impl:** [github.com/aletcher/inverting-bellman](https://github.com/aletcher/inverting-bellman) (JAX, PQN+HER training + tabular/LP P-learning)
> **Date:** 2026-06-25
> **Status:** Done — verdict locked (GOAT)
> **Related Research (katgpt-rs):** 275 (Induced CWM — closest cousin, the LLM-induction path), 192 (NextLat — forward latent dynamics), 118 (LEO All-Goals — the Q-function source); 295 (AC-Prefix — same "extract a property from a frozen forward pass" pattern)
> **Related Research (riir-ai):** 012 (LEO game runtime), 145 (CWM Runtime — Super-GOAT guide, where this lands as a sibling path)
> **Related Plans:** 155 (LEO All-Goals trait framework — Q substrate), 217 (NextLat belief-state drafter), 296 (Induced CWM primitive — sibling extraction path)
> **Classification:** Public

---

## TL;DR

The paper proves that **goal-conditioned Q-values implicitly encode a unique world model**, and gives a closed-form procedure (**P-learning**) to extract it: invert the Bellman equation by minimising `‖T^π_φ(Q) − Q‖²` over candidate transition kernels `P_φ`. In the tabular setting the fixed point is `P_∞(s,a) = M⁺Q(s,a) + (I − M⁺M)P₀(s,a)` (Theorem 1), where `M⁺` is the Moore–Penrose pseudo-inverse of the test-function matrix `M_{lk} = r(s'_k, g_l) + γV(s'_k, g_l)`. When the goal set spans the state space, `M` is full-rank and the extraction is unique — empirically as few as `|G|=4` goals on a 6-dim continuous Reacher environment recover a quasi-perfect world model (NMSE 1.2e-4), including over dimensions the rewards never touched (velocity, given position-only training).

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is **"closed-form world-model extraction from a frozen Q-function"**: given a frozen Q-table (or UVFA snapshot) and a fixed goal-reward family `(G, r)`, recover the implied transition kernel `P` via a single linear solve — `P̂(s,a) = M⁺Q(s,a)` (stochastic) or column-matching `argmin_{s'} ‖M(·,s') − Q(s,a,·)‖₁` (deterministic). No LLM, no gradient descent, no policy rollouts, no observation trajectories. The extracted `P̂` is an `InducedCwmKernel` candidate (Plan 296) — it can be unit-tested, BLAKE3-committed, and hot-swapped exactly like the LLM-induced variant. **This is the modelless cousin of the Induced CWM primitive** (R275/Plan 296) — same output contract (`InducedCwmKernel: GameState`), different extraction path (closed-form algebra vs LLM synthesis).

The whole training pipeline (PQN+HER, neural-network Q-function fitting) is `→ riir-train` and explicitly out of scope here; what is in scope is the **post-hoc extraction primitive** that runs against a frozen Q snapshot at cold/warm tier, plus the **identifiability gate** that decides whether the extraction is sound before promoting the candidate kernel.

---

## 1. Paper Core Findings

### 1.1 The inversion: Q → P instead of P → Q

Standard Q-learning fixes the kernel `P` and searches for a `Q` that satisfies `T⋆Q = Q`. **P-learning** fixes `Q` (the agent's frozen value function) and searches for a `P_φ` that satisfies the same Bellman equation:

```
L(φ) = ‖T^π_φ(Q) − Q‖²_d = ‖δ_φ‖²_d,    δ_φ = T^π_φ(Q) − Q
```

The gradient (Lemma in Appx A, derived via the score-function trick with a TD baseline) is

```
∇_φ L(φ) = E_{(g,s,a)∼d, s'₁,s'₂∼P_φ, a'₁,a'₂∼π}[ δ̂₁ · δ̂₂ · ∇_φ log P_φ(s'₂ | s,a) ]
```

— a double-sample estimator where both samples come from the *model* `P_φ` (free), not the environment. For deterministic MDPs parameterised as a successor function `P_φ: S×A → S`, the inner expectation collapses and a single reparameterised sample suffices.

### 1.2 The closed form: `P_∞ = M⁺Q + (I − M⁺M)P₀` (Theorem 1)

The tabular case decouples per `(s,a)` into a linear system. Defining `M_{lk} = r(s'_k, g_l) + γV(s'_k, g_l)` (the "Bellman test functions"), the Bellman equation becomes `M · P(s,a) = Q(s,a)`. P-learning by gradient descent converges to

```
P_∞(s,a) = M⁺ Q(s,a) + (I − M⁺M) P₀(s,a)
```

where `M⁺` is the Moore–Penrose pseudo-inverse. Two regimes:

- **M full column-rank** → `(I − M⁺M) = 0`, the extraction is unique and independent of the prior `P₀`: `P_∞ = M⁺Q`. If `Q = Q^π` exact, this is the true kernel `P`.
- **M rank-deficient** → value equivalence; `(I − M⁺M)P₀` is the underdetermined component and the prior `P₀` selects among indistinguishable kernels.

The deterministic special case replaces `M⁺Q` with **column-matching**: `P̂(s,a) = argmin_{s'} ‖M(·,s') − Q(s,a,·)‖₁`, which is unique iff `M` is column-injective (`M(·,s) ≠ M(·,s')` for all `s ≠ s'`).

### 1.3 Identifiability theorems (Section 4)

Sufficient conditions on the goal-reward family `(G, r)` for the extraction to be unique — i.e. for `M` to be full-rank or column-injective:

| Regime | Goal count | Reward family | Identifiability |
|---|---|---|---|
| Deterministic, finite `S` | `|G| ≥ 1` | Generic (Lebesgue-a.e.) | `∃!P(Q^π)` — **Thm 2** |
| Deterministic, finite `S` | `|G| = 1` | Gaussian `r_g(s) = exp(−‖φ(s)−g‖²/2σ²)` | `∃!P(Q^π)` for a.e. `g` — **Prop 1** |
| Stochastic, finite `S` | `|G| ≥ |S|` | Spanning (`rank(R) = |S|`, includes indicator `r(s,g) = δ_{sg}`) | `∃!P(Q^π)` — **Thm 3** |
| Stochastic, finite `S` | `|G| ≥ |S|` | Generic | `∃!P(Q^π)` for a.e. `r` — **Thm 3(a)** |
| `N`-local stochastic, finite `S`, known support | `|G| ≥ N−1` | Generic | `∃!P(Q^π)` — **Thm 5(a)** |
| Deterministic, continuous `S ⊆ ℝᵈ`, indicator + termination-on-arrival | `G ⊇ S + B_σ(0)` | Indicator | `∃!P(Q^π)` — **Thm 4** |
| Deterministic, continuous, analytic `V^π`, `D` | `|G| ≥ 2d+1` | Gaussian | `∃!P(Q^π)` for a.e. goal tuple — **Thm 9** |

Approximate bounds: if `Q` is `ε`-approximate and `M` is column-injective with separation `Δ`, deterministic column-matching succeeds iff `ε < Δ(1+γm)/2` (Thm 2b). Stochastic: `‖P̂ − P‖₁ ≤ ‖M⁺‖₁ (1+γm) ε` (Thm 3c).

**Theorem 5 (local MDPs)** is particularly relevant for game maps: an `N`-local kernel (each `(s,a)` transitions to ≤ N neighbours, e.g. cardinal-4 movement on a grid) is identifiable from `|G| ≥ N−1` generic goals when the support is known (or `2N−1` when unknown) — far less than `|S|`.

### 1.4 Empirical: 4 goals recover a 6-dim continuous world model

- **Reacher** (`|S|=6`, `|A|=9` discretised): `|G|=4` position-only goals → extracted `P̂` matches ground truth at NMSE 1.2e-4 despite Q-values being coarse (NMSE 5.7e-1). The extracted WM is so accurate that planning inside it produces quasi-optimal policies on **out-of-distribution velocity-based goals** — variables the rewards never depended on. Spearman `ρ = −0.98` between agent return and WM error across a 42-architecture sweep.
- **MountainCar**: position-trained and velocity-trained agents induce near-identical WMs (NMSE 1.7e-4 between them, vs 7.2e-3 to ground truth) — the "local-global" hypothesis: agents trained on local goals implicitly encode global dynamics.
- **FourRooms variants**: deterministic (`|G|=1`), windy/local (`|G|=4`), teleporting/stochastic (`|G|=20`, vs `|S|=68` worst-case bound).

### 1.5 The resolvent identity — the structural bridge

Lemma 1 proves the matrix `M` factors as

```
M_l = R_l (I − γ P̄^π_l)⁻¹
```

where `P̄^π_l = (P^π_l)ᵀ` is the policy-induced kernel transpose. **This is the same `(I − γP)⁻¹` resolvent that appears in LatCal-style deterministic commitment and in our latent-functor arithmetic.** The Bellman-inversion problem is algebraically a *resolvent inversion*: given `M = R(I − γP̄)⁻¹` and `Q`, solve for `P`.

---

## 2. Distillation

### 2.1 What ships in katgpt-rs (public, open, generic)

| Primitive | What it is | Why open |
|---|---|---|
| `tabular_p_extract<M,Q>` | Closed-form `P̂(s,a) = M⁺Q(s,a)` per `(s,a)`. Linear-algebra solve. Optional simplex projection for the rank-deficient case. | Pure linear algebra — same flavour as `dirichlet_energy`, `peira` regressor, `karc` ridge solve. |
| `column_match<P,Q>` | Deterministic successor recovery: `P̂(s,a) = argmin_{s'} ‖M(·,s') − Q(s,a,·)‖₁`. SIMD L1-distance argmin. | Pure metric-space primitive — same shape as `maxsim_score`. |
| `identifiability_gate<M>` | Pre-extraction gate returning `rank(M)`, `‖M⁺‖₁`, column separation `Δ`, and a verdict `{Unique, Underdetermined, InsufficientGoals}`. | Pure linear-algebra diagnostic — same shape as `subspace_phase_gate`. |
| `InducedCwmKernel` impl for the extracted `P̂` | The output `P̂` already implements `GameState`-shaped advance — wrap it as an `InducedCwmKernel` so it inherits `CwmCommitment`, `verify_transition`, and `InducedCwmSlot` hot-swap. | **Zero new trait surface** — purely a new constructor for the existing Plan 296 primitive. |

**What stays out of katgpt-rs:** the Q-function training pipeline (PQN+HER, neural-network fitting), the LLM-induction counterpart (R275), game-specific NPC integration, chain commitment bridging, observation-trajectory collection. Those are private IP — see §5.

### 2.2 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalents |
|---|---|
| Q-values / action-value function | `LeoHead` output (Plan 155), `QGradientOracle`, belief-state snapshot, posterior snapshot, frozen shard readout |
| world model / transition kernel `P` | `GameState::advance`, `InducedCwmKernel`, forward model, deterministic kernel, latent functor application chain |
| Bellman residual `‖T^π_φ(Q) − Q‖²` | TD error, `delta_signal`, `coherence decay` (latent_functor/reestimation), surprise |
| Bellman inversion / P-learning | `tabular_p_extract` (closed form), `column_match` (deterministic), re-estimation, re-derive, thawed-snapshot projection |
| pseudo-inverse `M⁺` | `linalg/ridge_solve.rs` (Plan 308), Schur solver (Plan 318), least-squares lift |
| value equivalence | latent ambiguity, underdetermination, subspace collapse, coherence `< τ` |
| identifiability / column-injectivity | `subspace_phase_gate` participation ratio, numerical rank, `Dirichlet` separation |
| resolvent `M = R(I − γP̄)⁻¹` | latent functor arithmetic, LatCal `(I − γP)⁻¹` form, KARC delay-basis |
| goal-conditioned RL / GCRL | LEO All-Goals (Plan 155), `LeoHead`, `sigmoid_bounded_q` |
| Bellman test function `m^π_g(s')` | `StateHeuristic`, value function prior, MCTS leaf evaluator |
| successor representation / features | Raven RSM (`rim_slots`), δ-Mem, occupancy measure |
| `ε`-approximate Q | belief drafter residual, `MicroRecurrentBeliefState` coherence |

### 2.3 Latent-space reframing (mandatory before verdict)

Re-cast the paper's mechanism as a latent-to-latent operation on each Super-GOAT factory module:

**(a) HLA's per-NPC latent state.** The paper's `Q(s,a,g)` is a value function over `(state, action, goal)`. HLA's 8-dim per-NPC latent state `(valence, arousal, desperation, calm, fear, …)` plays the role of a *goal-indexed value summary*: each NPC's affect IS a low-dimensional projection of its value function across its active objective set. P-learning on HLA state = "extract the per-NPC implied world model from its affect vector" — i.e. decode what the NPC *believes* about transition dynamics, given what it *values*. This is the **think-brain / two-brain model made algebraic**: the think brain's belief about dynamics is the Moore–Penrose inverse of its value profile.

**(b) `latent_functor/` operations.** The resolvent identity `M_l = R_l(I − γP̄^π_l)⁻¹` is *exactly* a latent functor: `(I − γP̄)⁻¹` is the functor application, `R_l` is the per-goal direction vector, and `M` is the composed operator. The Bellman inversion `P̂ = M⁺Q` is the **functor inverse** — given the composed operator and the output `Q`, recover the underlying kernel `P`. This generalises `latent_functor/arithmetic.rs::extract_functor` from "extract displacement direction from source/target latent pairs" to "extract transition kernel from value-function snapshots". The coherence-decay re-estimation scheduler (`latent_functor/reestimation.rs`) is the natural place to trigger re-extraction when `coherence < τ_reest`.

**(c) `cgsp_runtime/` curiosity signals.** The paper's identifiability gate (`∃!P(Q)` requires goals that span `S`) maps to a **curiosity signal**: if the NPC's active goal set does *not* span the state space, the extracted world model has an underdetermined component `(I − M⁺M)P₀` — exactly the region where the NPC should be *curious*. Areas of high value-equivalence ambiguity = areas of high exploration value. This fuses cleanly with Curiosity Pulse (R041) and the ICT branching-point detector (R270).

**(d) LatCal fixed-point commitment (riir-chain).** The extracted `P̂` is a *deterministic, closed-form-derived* transition kernel — perfect LatCal material. Once `identifiability_gate` returns `Unique`, the kernel can be canonicalised and BLAKE3-committed exactly like the LLM-induced variant (R275). The `(I − γP̄)⁻¹` resolvent is itself LatCal-shaped arithmetic. A faction of NPCs that co-extract `P̂` from a shared frozen Q-snapshot can establish rule consensus without ever running an LLM.

**(e) `NeuronShard` / `MerkleFrozenEnvelope` (riir-neuron-db).** The frozen Q-function is naturally stored as a `NeuronShard` (the Q-table or UVFA weights are the shard's `style_weights`). The extracted `P̂` is itself freezable: a `MerkleFrozenEnvelope` over the canonical `P̂` bytes gives tamper-evident "this NPC extracted this world model from this Q-snapshot at tick T" provenance. Raven/δ-Mem consolidation generalises from "consolidate experience into a shard" to "consolidate a frozen Q into an extracted `P̂` shard".

**(f) DEC Stokes-calculus operators (katgpt-rs `dec/`).** `M_l = R_l(I − γP̄^π_l)⁻¹` has the form of a **resolvent Green's function** — the inverse of `(I − γP̄)`, which is a discrete elliptic operator. The Bellman inversion is structurally a Green's-function inversion. The DEC substrate's `hodge_laplacian` `(δd + dδ)` is the canonical elliptic operator on a cell complex; `belief_mass_divergence` (Plan 314) gives the divergence of a belief flow. **The paper's `M⁺` is the pseudo-Green's-function of the policy-induced kernel.** This is a thin connection for `d ≤ 3` (game maps, HLA regions) — curse-of-dimensionality rules it out for high-dim shards.

### 2.4 Fusion — what novel combination does Inverting-Bellman × our stack produce?

| Fusion | Source A | Source B | Novel combination? |
|---|---|---|---|
| **Bellman-inversion × Induced CWM (R275)** | R298 (closed-form `P̂ = M⁺Q`) | R275 (LLM-induced CWM, Plan 296) | **Refinement** — two extraction paths feed the same `InducedCwmKernel` substrate. CWM-LLM works on observation-only data with no Q; R298 works on frozen-Q with no observations. **The two are complementary**, not competitive. R298 is the modelless path; R275 is the data-driven path. A runtime NPC picks the path by what it has: frozen Q → R298; observation stream → R275. |
| **Bellman-inversion × LEO All-Goals (R118/Plan 155)** | R298 (extract `P̂` from `Q`) | R118 (`LeoHead` produces all-goals `Q` in one forward pass) | **Force multiplier** — LEO is the *natural Q-source* for R298. LEO already produces the all-goals Q-vector the paper requires. With LEO default-on, every LEO-trained NPC has the input for R298 extraction. |
| **Bellman-inversion × Latent Functor re-estimation (R123/Plan 303)** | R298 (closed-form extraction) | `latent_functor/reestimation.rs` (coherence-driven re-extraction) | **Force multiplier** — when `coherence(Q, P̂) < τ_reest`, re-run `tabular_p_extract`. The Bellman residual `‖T^π_φ(Q) − Q‖²` is itself a coherence signal. Identical scheduler pattern to DiPOD's ELBO-drift self-distillation. |
| **Bellman-inversion × Freeze/Thaw (riir-neuron-db)** | R298 (`P̂` is closed-form from frozen `Q`) | `MerkleFrozenEnvelope`, `InducedCwmSlot` atomic hot-swap | **GOAT-tier wiring** — extract `P̂` from a thawed Q-snapshot, BLAKE3-commit the canonical `P̂`, hot-swap into the `InducedCwmSlot`. Identical machinery to R275. |
| **Bellman-inversion × LatCal (riir-chain)** | R298 (deterministic closed-form `P̂`) | LatCal fixed-point bridge | **NEW INSIGHT** — `P̂` derived from a frozen Q is *bit-reproducible* (closed-form solve, no sampling). A faction can establish rule consensus by co-extracting `P̂` from a shared Q-snapshot and LatCal-committing the result — no LLM call, no observation sharing, no quorum on trajectories. The commitment is on the *extracted kernel bytes*, which any node can re-derive from the same `Q` + `(G,r)`. |
| **Bellman-inversion × Curiosity Pulse (R041) / ICT (R270)** | R298 (identifiability gate flags underdetermined regions) | Curiosity-driven exploration | **NEW CAPABILITY** — value-equivalence underdetermination `(I − M⁺M)P₀ ≠ 0` IS a curiosity signal. When `identifiability_gate` returns `Underdetermined`, the NPC knows its active goal set does not span some state subspace — *that* is where it should explore. This is a **principled, algebraic curiosity signal** derived from the rank structure of `M`, not from entropy or surprise. Fuses cleanly with Curiosity Pulse and the ICT branching detector. |
| **Bellman-inversion × Two-Brain model (riir-armageddon)** | R298 (extract think-brain `P̂` from frozen `Q`) | Info-brain = ground-truth synced `P`, think-brain = subjective `P̂` | **NEW INSIGHT** — the paper's "subjective world model" framing *is* the two-brain model. The think brain's `P̂` extracted from its (possibly stale, possibly biased) `Q` diverges from the info brain's ground-truth `P` *by design*. Fog-of-war re-observation is the regularizer that bounds divergence. This makes the two-brain divergence *algebraically characterizable* — `‖P̂ − P‖₁ ≤ ‖M⁺‖₁(1+γm)ε` (Thm 3c) gives a closed-form bound on how wrong an NPC's subjective model can be given how wrong its Q is. |
| **Bellman-inversion × AC-Prefix (R295/Plan 313)** | R298 (extract `P̂` from a single forward-pass `Q`) | R295 (single-pass arbitrary-conditional `p(xe | xc)`) | **Force multiplier** — AC-Prefix gives single-pass conditional Q-evaluation; R298 consumes the resulting Q to extract `P̂`. The two primitives compose: AC-Prefix → `Q(s,a,g)` for arbitrary `(s,a,g)` triples → R298 → `P̂`. Same "extract a property from a frozen forward pass" shape. |

The two **NEW CAPABILITY/INSIGHT** fusions (principled algebraic curiosity, characterizable two-brain divergence) are interesting but **do not cross the Super-GOAT bar** because they require a *trained Q-function* as input — which is `→ riir-train` per §3.5 unless the Q can be supplied modellessly (see §3.5 analysis below).

---

## 3. Verdict

### **GOAT**

**Novelty gate:**

| Q | Answer | Evidence |
|---|---|---|
| **Q1: No prior art?** | ⚠️ NO | The "extract a forward model from a frozen value function" pattern is conceptually adjacent to **R275 Induced CWM** (Plan 296, GOAT 4/4 PASS, shipped) — same output contract (`InducedCwmKernel`), same commitment/hot-swap machinery. The closest latent-side cousin is **R192 NextLat** (Plan 217, belief-state latent dynamics — but forward direction, not inverse). The Q-substrate is **R118 LEO All-Goals** (Plan 155, Super-GOAT, default-on). Matthews is on both the LEO and Inverting-Bellman author lists — they are explicitly related work. **The closed-form `M⁺Q` extraction itself is not shipped** — but it is a linear-algebra solve in the same family as `karc/ridge_solve` (Plan 308) and `peira` (Plan 153), and the output plugs into already-shipped `InducedCwmKernel`. |
| **Q2: New class of behavior?** | ❌ NO | R275 already opened the "NPC extracts a forward model" capability class. R298 is a *different extraction path* (closed-form vs LLM-synthesis), not a new class. The two interesting novel angles — algebraic curiosity from `rank(M)` and characterizable two-brain divergence — are insights about *existing* pillars (curiosity, two-brain model), not new capability classes. |
| **Q3: Product selling point?** | ⚠️ PARTIAL | The selling point requires a trained Q-function (LEO+PQN+HER). Without riir-train, the primitive is a *consumer* of trained artifacts, not a standalone capability. The modelless-correctable path (§3.5) is plausible — a frozen Q-snapshot thawed at inference, plus closed-form extraction, is modelless — but it's a *refinement* of the existing CWM selling point, not a new one. |
| **Q4: Force multiplier?** | ✅ YES (≥4 pillars) | LEO All-Goals (Q source), Induced CWM (output substrate), Latent Functor re-estimation (scheduler), LatCal commitment (chain provenance), Curiosity Pulse (identifiability-as-curiosity), Two-Brain model (subjective `P̂`). |

**Not all 4 YES → GOAT, not Super-GOAT.** The primitive is novel modelless math, but it sits *inside* the capability class opened by R275 Induced CWM, complementing the LLM-induction path with a closed-form alternative. It is a **gain-bearing refinement** of an existing Super-GOAT — exactly the GOAT tier.

**One-line reasoning:** R298 ships the modelless cousin of Induced CWM — a closed-form `P̂ = M⁺Q` extraction that produces an `InducedCwmKernel` candidate without an LLM call — strengthening the existing Super-GOAT (R275) by adding a no-LLM extraction path, an algebraic curiosity signal, and a characterizable two-brain divergence bound, but not opening a new capability class.

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | n/a — verdict is GOAT |
| **GOAT** ← **this** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → `katgpt-rs/.plans/319_bellman_inversion_p_extract.md` (open primitive, feature flag `bellman_inversion`). The gain to prove: (G1) extraction correctness on a known-P tabular fixture, (G2) extraction latency < 100µs per `(s,a)` on a 1000-state MDP, (G3) identifiability gate matches paper Thm 2/3/5 predictions, (G4) zero-alloc hot path. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | n/a |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | n/a |

---

## 3.5. Modelless-unblock protocol (mandatory before any riir-train deferral)

The paper's empirical results use PQN+HER-trained neural-network Q-functions. Does this force a `→ riir-train` deferral for the extraction primitive itself? **No — the extraction is modelless; only the Q-source is potentially training-dependent.** Walking the §3.5 protocol:

1. **Freeze/thaw snapshot correction (path 1).** Can a frozen Q-snapshot, thawed at inference, supply the input to `tabular_p_extract`? **YES** — this is the canonical use case. A Q-table or UVFA snapshot is exactly the artifact freeze/thaw is designed to ship. The extraction primitive consumes the thawed snapshot modellessly. **No training needed at runtime.**
2. **Raw/lora reader-writer hot-swap (path 2).** Can a *deterministically constructed* Q-function (not trained) supply a useful `P̂`? **PARTIAL** — for deterministic MDPs where `Q(s,a,g) = M(P(s,a), g)`, a constructed `Q` recovers a constructed `P` (circular but valid for testing). For real games, `Q` must come from training. **Path 2 does not unblock production extraction but does enable modelless validation of the extraction primitive itself** (G1 tests use constructed Q-P pairs).
3. **Latent-space correction (path 3).** Can a biased extracted `P̂` be corrected by projecting onto a correction direction and gating? **YES for systematic biases** — if the Q-source has a known systematic bias (e.g., the AC-Prefix G1 doubled-signal bias, Plan 313), the *resulting* `P̂` bias is also systematic and characterizable via the Thm 3c bound `‖P̂ − P‖₁ ≤ ‖M⁺‖₁(1+γm)ε`. A latent projection gate on `P̂` can downweight the biased components.

**Verdict:** the extraction primitive is **MODELLESS-VALIDABLE** via path 1 (frozen Q-snapshot) + path 2 (constructed Q-P pairs for G1 tests). The Q- *training* pipeline is `→ riir-train`, but the Q-*consumption* (extraction) primitive ships modellessly in katgpt-rs. No premature deferral.

---

## 4. Latent vs raw boundary (riir-armageddon compliance check)

- **The frozen `Q(s,a,g)` snapshot** is a latent artifact (per-NPC subjective value). It does NOT cross the sync boundary directly — it is consumed locally by the extraction primitive. ✅ latent-local.
- **The extracted `P̂(s,a)`** in a game context IS the transition kernel — raw, deterministic, syncable. Once `identifiability_gate` returns `Unique` and `P̂` is BLAKE3-committed via `CwmCommitment`, the canonical bytes can cross the sync boundary for anti-cheat replay. ✅ raw-committed.
- **Bridge function:** `tabular_p_extract` is the latent→raw bridge. It consumes a latent Q-snapshot and produces a raw deterministic kernel. It is zero-allocation (caller-supplied scratch), gateable by feature flag, and introduces no sync dependency. ✅ compliant.
- **KG triple emission:** semantic encounters derived from `P̂` structure ("NPC believes action A from state S leads to state S'") → KG triple. Physical events (NPC actually moves) → TxDelta with raw values. ✅ compliant.
- **Two-brain model:** info brain = ground-truth synced `P`. Think brain = subjective `P̂` extracted from NPC's `Q`. They diverge by design; fog-of-war re-observation of transitions is the regularizer. **Thm 3c gives the closed-form divergence bound** `‖P̂ − P‖₁ ≤ ‖M⁺‖₁(1+γm)ε` — a *characterizable* two-brain divergence, which is novel.

---

## 5. Connection map (force-multiplier analysis)

| Existing pillar | How R298 multiplies it |
|---|---|
| **Induced CWM (R275, Plan 296)** | Adds a closed-form extraction path alongside LLM-induction. Same output contract (`InducedCwmKernel`). |
| **LEO All-Goals (R118, Plan 155)** | LEO is the natural Q-source for R298. LEO's all-goals Q-vector is exactly what `tabular_p_extract` consumes. |
| **Latent Functor (R123, Plan 303)** | The re-estimation scheduler (`coherence < τ_reest`) triggers R298 re-extraction. Bellman residual is the coherence signal. |
| **NextLat (R192, Plan 217)** | NextLat is the forward direction (state → next state); R298 is the inverse direction (Q → P). Together they form a belief-state ↔ world-model duality. |
| **LatCal (riir-chain)** | Extracted `P̂` is bit-reproducible closed-form output — perfect LatCal commitment material. Faction rule consensus via co-extraction. |
| **MerkleFrozenEnvelope (riir-neuron-db)** | Frozen Q is a shard; extracted `P̂` is a derived shard; both wrap in `MerkleFrozenEnvelope` for tamper-evident provenance. |
| **Curiosity Pulse (R041) / ICT (R270)** | `rank(M)` identifiability gap IS a curiosity signal — explore where the goal set doesn't span. |
| **Two-Brain model (riir-armageddon)** | Thm 3c gives the closed-form divergence bound between think-brain `P̂` and info-brain `P`. |
| **AC-Prefix (R295, Plan 313)** | Single-pass conditional Q-evaluation feeds R298 extraction. |

---

## 6. What NOT to take

1. **PQN+HER training pipeline.** Training-only → `→ riir-train`. We consume the resulting frozen Q-snapshot, not the training loop.
2. **Neural-network parametrisation of `P_φ`.** The paper's primary P-learning uses a residual MLP `P_φ(s,a) = s + h_φ(s,a)`. The closed-form tabular extraction is the modelless transferable piece; the MLP variant is `→ riir-train`.
3. **Indicator goals with `|G| ≥ |S|` worst-case.** Theorem 3's worst case `|G| ≥ |S|` is impractical for large `S`. Theorem 5 (local MDPs, `|G| ≥ N−1` for known support) is the practical regime for game maps. We take the *local* result, not the global stochastic bound.
4. **The double-sample score-function gradient estimator (Appx A).** Only relevant for the MLP-parametrised P-learning, which is `→ riir-train`. The tabular extraction is closed-form and needs no gradient.
5. **Continuous-state Gaussian/indicator reward theory (Thms 4, 6, 7, 8, 9).** Mathematically beautiful but impractical for game maps (the bounds require `G ⊇ S + B_σ(0)`, i.e. a goal set larger than the state space). The finite-state theory (Thms 2, 3, 5) covers our use cases.

---

## 7. Validation protocol (G1–G4 GOAT gate)

**G1 — Correctness on a known-P fixture.** Construct a small tabular MDP (e.g. 5-state deterministic chain, 4-state stochastic gridworld), construct `Q^π` analytically from the known `P`, run `tabular_p_extract(M, Q)`, assert `‖P̂ − P‖₁ < 1e-6`. Repeat for: deterministic (column-match), stochastic (pseudo-inverse), and `N`-local (LP-restricted-to-support). At least 3 fixtures × 3 regimes = 9 tests.

**G2 — Latency.** Microbench `tabular_p_extract` on a 1000-state × 10-action MDP with `|G|=20` goals. Target: < 100µs per `(s,a)` extraction on SIMD CPU. The pseudo-inverse `M⁺` (20×1000) is computed once, applied `|S|·|A|` times — pre-compute `M⁺` once (cold), apply in hot loop.

**G3 — Identifiability gate matches theory.** Construct fixtures matching Thms 2, 3, 5 exactly: (a) deterministic + 1 generic goal → `identifiability_gate` returns `Unique`; (b) stochastic + `|G| < |S|` → returns `InsufficientGoals`; (c) `N`-local + `|G| = N−1` with known support → returns `Unique`. Assert the gate verdict matches the theorem prediction in all three cases.

**G4 — Zero-alloc hot path + feature isolation.** `tabular_p_extract_into` takes caller-supplied scratch; no `Vec` allocation in the per-`(s,a)` loop. Feature-gated behind `bellman_inversion`; `cargo check` with feature off is byte-identical to baseline.

**Promotion rule:** all 4 PASS → keep `bellman_inversion` opt-in, ready for downstream consumption. Any FAIL → stay opt-in, file `.issues/NNN_*` follow-up.

**Modelless-validation caveat:** G1 uses *constructed* Q-P pairs (path 2 in §3.5), not trained Q. This validates the extraction primitive's correctness; it does NOT validate that real trained Q produces useful `P̂`. The latter is `→ riir-train` (LEO+PQN+HER) or requires a thawed Q-snapshot from production (path 1). The GOAT gate is on the *primitive*, not on the trained-Q pipeline.

---

## 8. Paper metadata

- **Title:** Inverting the Bellman Equation: From Q-Values to World Models
- **Authors:** Alistair Letcher¹, Mattie Fellows¹, Alexander D. Goldie¹, Jonathan Richens², Jakob N. Foerster¹, Oliver Richardson³·¹  (¹FLAIR Oxford, ²Google DeepMind, ³Mila Montreal)
- **arXiv:** 2606.21173v1 [cs.LG], 19 Jun 2026
- **Code:** github.com/aletcher/inverting-bellman (JAX, PQN+HER training, tabular/LP P-learning)
- **Author overlap with our corpus:** Mattie Fellows is on Bayesian Exploration Networks (cited in §6); Matthews (LEO, R118) is *not* an author here but the LEO and Inverting-Bellman papers are explicitly related (Inverting-Bellman cites goal-conditioned RL and HER, which LEO generalises).

---

## TL;DR

**Verdict: GOAT** (not Super-GOAT). The paper proves Q-values implicitly encode a unique world model and gives a closed-form extraction `P̂ = M⁺Q + (I − M⁺M)P₀` (Theorem 1) — the **modelless cousin of Induced CWM (R275/Plan 296)**. Same output contract (`InducedCwmKernel`), different extraction path (closed-form algebra vs LLM synthesis). The primitive is **MODELLESS-VALIDABLE** (§3.5 path 1: frozen Q-snapshot + closed-form solve; path 2: constructed Q-P pairs for G1 tests) — no `→ riir-train` deferral needed for the primitive itself, only for the Q-training pipeline that produces its input. Plan 319 will ship `tabular_p_extract` + `column_match` + `identifiability_gate` behind feature flag `bellman_inversion`, with G1–G4 GOAT gates. The interesting novel angles — algebraic curiosity from `rank(M)`, characterizable two-brain divergence via Thm 3c — are insights about *existing* pillars, not new capability classes, so the verdict stays at GOAT.
