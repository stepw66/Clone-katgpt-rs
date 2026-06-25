# Research 308: NSM vs VLA — "The Price Is Not Right" + Bisimulation-Based Operator Abstraction

> **Source:** [The Price Is Not Right: Neuro-Symbolic Methods Outperform VLAs on Structured Long-Horizon Manipulation Tasks with Significantly Lower Energy Consumption](https://arxiv.org/pdf/2602.19260) — Duggan, Lorang, Lu, Scheutz (Tufts HRI Lab + AIT Vienna), 22 Feb 2026, arXiv:2602.19260 [cs.RO]
> **Underlying NSM method:** [Lorang et al. "Few-shot neuro-symbolic imitation learning for long-horizon planning and acting"](https://arxiv.org/abs/2508.21501) — arXiv:2508.21501, Aug 2025
> **Date:** 2026-06-25
> **Status:** Done
> **Related Research:** 275 (CWM — closest cousin, the existing Super-GOAT), 264 (Closure-Expansion Instrument — explicitly flagged the gap this closes), 188 (NS-CSG), 172 (MUSE/ITSE skill lifecycle), 211 (three-mode NS router), 185 (INSIGHT symbolic distillation)
> **Related Plans:** 296 (Induced CWM primitive — the prior art this validates), 290 (Closure-Expansion Instrument — the gap this partially closes), 324 (Bisimulation Operator Inference — the new plan from this note)
> **Classification:** Public

---

## TL;DR

A head-to-head empirical comparison between a fine-tuned open-weight VLA (π0 via OpenPi, LoRA on PaliGemma-2B + Gemma-300M) and a neuro-symbolic architecture (PDDL planner + diffusion-policy skills) on Towers of Hanoi manipulation. **The neuro-symbolic model (NSM) wins by a wide margin** — 95% vs 34% success on 3-block, **78% vs 0% on the unseen 4-block variant** (zero-shot generalization from training on simpler stacking demos only), trained on 50 demos vs 300, in **34 minutes vs 1.5+ days**, at **0.85 MJ vs 65–69 MJ total training energy** (~80× less), and runs **CPU-only at inference** with ~10× lower per-episode energy. The paper's punchline, *"VLMs cannot reliably plan"* (Kambhampati), is empirically demonstrated: GPT-5 produces only 84% optimal plans; Qwen-7B and PaliGemma-3B produce 0% optimal, 100% invalid.

**Distilled for katgpt-rs (modelless, inference-time):**

This paper is **empirical validation of the existing CWM/NSM thesis** that already ships as the Super-GOAT in Research 275 + Plan 296 — *"the LLM is the rule-induction engine, classical search is the policy."* The capability class ("system observes a structured task, induces verifiable rules, plans via search") is already shipped. **Not** a new Super-GOAT.

**However**, the paper's NSM pipeline contains one transferable primitive that the codebase explicitly lacks and that Research 264 (Closure-Expansion Instrument) flagged as a gap: **a bisimulation-based state-equivalence compactor + ASP/PDDL operator-inference pipeline that materializes a transition graph from observed trajectories, collapses redundant states, and emits reusable symbolic operators.** This is the missing half of Research 264's "Primitive Transition Graph (PTG) + motif mining + operator promotion" loop, in a concrete, shippable form. **GOAT verdict — plan only (324), no Super-GOAT guide.**

---

## 1. Paper Core Findings

### 1.1 Headline empirical result

| Setting | E2E-VLA | PG-VLA | **NSM** |
|---|---|---|---|
| Individual Move success | 87.0% | 59.6% | **99.0%** |
| 3-Block Hanoi success | 34.0% | 0.0% | **95.0%** |
| 4-Block Hanoi success (unseen) | 0.0% | 0.0% | **78.0%** |
| Training time | 1d 16h 26m | 1d 15h 42m | **34m** |
| Training energy (total, MJ) | 68.5 | 64.9 | **0.85** (~80× less) |
| Inference GPU | required | required | **none (CPU only)** |
| Per-episode energy (3-blk, kJ) | 7.96 | 6.94 | **0.83** (~10× less) |
| Training demos | 300 full Hanoi runs | 300 | **50 stacking-only** (never sees a full Hanoi resolution) |

VLM planners (GPT-5 / Qwen-7B / PaliGemma-3B) tasked with producing pick-and-place sequences: GPT-5 = 84% optimal / 16% invalid; Qwen = 0% optimal / 100% invalid; PaliGemma = 0% optimal / 100% invalid. Confirms Kambhampati et al. *"LLMs can't plan"* (arXiv:2402.01817).

### 1.2 The NSM architecture (from Lorang et al. 2508.21501)

The paper compares against the prior NSM, whose pipeline has five stages:

1. **Symbolic abstraction from demonstrations.** From raw demo trajectories `D`, extract node transitions `τ_node = (n, l, n′)` where `n`, `n′` are high-level abstract states and `l` is a human-assigned label. These form a graph `G = ⟨V, E, L⟩` whose **nodes are abstract states and edges are skills**.

2. **Minimal bisimulation compaction.** Compute a minimal bisimulation `G̅` that removes redundant states while preserving equivalence — i.e., quotient the graph by the bisimulation relation (two states equivalent iff their outgoing edge-labels lead to equivalent successor classes).

3. **ASP-based PDDL domain inference.** Using an Answer Set Programming solver (Bonet & Geffner 2020, Rodriguez et al. 2021), infer a symbolic domain `σ = ⟨E, F, S, O⟩` in PDDL form from the compacted graph.

4. **Skill decomposition + diffusion policies.** Each operator `o_i ∈ O` is associated with a neural policy `π_i` trained from its demo segments. Inspired by the **options framework** (Sutton, Precup, Singh 1999), each skill is further decomposed into action-step sub-policies `π_{i,j}` with learned **termination conditions**. A task-relevant feature selector `φ` projects observations to operator-relevant objects `E_{o_i}` in end-effector-relative coordinates.

5. **Hierarchical execution.** At test time: user specifies `(s_0, s_g)` → PDDL instance → MetricFF planner → plan `P = [o_1, …, o_{|P|}]` → each `o_i` realized by `π_i`, which internally sequences `π_{i,j}` until their termination conditions fire.

### 1.3 Why NSM generalizes where VLA fails

NSM never observes a Towers of Hanoi resolution during training — only 50 random stacking demos. It **infers** the symbolic operator schema from the compacted transition graph, and MetricFF composes those operators into novel plans at test time. The 4-block variant is a zero-shot generalization: same operators, longer plan, no retraining.

VLAs, by contrast, memorize trajectories. PG-VLA was given correct instructions for 4-block and still failed — it executed the 3-block trajectory it had memorized, dropping the first block on the *right* platform (3-block first move) instead of the *middle* platform (4-block first move).

### 1.4 The energy framing

Training energy is dominated by LoRA fine-tuning of PaliGemma-2B + Gemma-300M (full GPU saturation, 100% util, ~410–424 W mean GPU power, 1.5+ days). NSM training is dominated by diffusion-policy fitting on 50 short stacking segments (34 min, CPU-bound). At inference, VLA requires GPU (~70 W GPU + ~43 W CPU = ~115 W total); NSM is CPU-only at ~19 W. Per-episode energy ratios compound because VLA episodes also take ~2× longer (more retries, more steps).

---

## 2. Distillation

### 2.1 What's already shipped in our stack (the prior art)

| Paper mechanism | Closest shipped cousin | Status |
|---|---|---|
| Induce forward model from trajectories, plan via search | **Induced CWM Kernel** (Research 275, Plan 296, arXiv:2510.04542) — `InducedCwmKernel: GameState`, `verify_transition`, `ismcts_search_with_inference`, `ValueFnTournament`, `CwmCommitment` (BLAKE3) | ✅ Super-GOAT, shipped, GOAT-proved (47/47 tests) |
| Symbolic-extraction pipeline (LLM-as-induction-engine) | CWM REx refinement loop (Thompson tree over CWM hypotheses) | ✅ Private (riir-ai Plan 326) |
| Skill catalog + lifecycle (create / memory / refine / retire) | **MUSE/ITSE** (Research 172, Plan 192) — `SkillCatalog`, `PrunerMemory`, `PrunerTestGate`, `FrozenSkillBank` | ✅ DEFAULT ON, GOAT-proved |
| Skill lifecycle exploration arm (try / patch / split) | `AbsorbCompressLayer`, Bayesian posterior-guided skill evolution (R211, P239) | ✅ Shipped |
| Three NS modes (L4R / R4L / LR) routing | **Three-Mode Neuro-Symbolic Router** (Research 186, Plan 211) — bandit-selected per-decode mode | ✅ Shipped |
| MDL-gated primitive admission | **Regime-Transition Inference** (Plan 215, Research 190) — `RegimeTransitionGate::evaluate()`, `ProvenanceChain` (BLAKE3) | ✅ DEFAULT ON, GOAT 8/8 |
| AND-OR subgoal decomposition | **AND-OR DDTree** (Plan 190, Research 170) — `AndOrNode`, `BlueprintPass`, `DecompositionReviewer`, `ProofGoalCache` | ✅ GOAT-proved |
| Options framework / termination conditions | **Salience Tri-Gate** (Plan 303, Research 281) — per-tick Speak/Silent/Delegate emit decision | ✅ Shipped |
| Symbolic distillation (skill → expression) | **INSIGHT Symbolic Distillation** (Plan 210, Research 185) — `symbolic_distill`, `concept_grounding` | ✅ Shipped |

**Conclusion:** the broad capability class — *"system observes a structured task, induces verifiable rules, plans via search, decomposes into skills with termination conditions"* — is already shipped as the CWM Super-GOAT plus its force-multiplier stack. This paper does **not** open a new capability class.

### 2.2 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalents |
|---|---|
| symbolic abstraction / PDDL domain | `GameState` impl, `InducedCwmKernel`, induced forward model, LatCal program |
| minimal bisimulation `G̅` | partition refinement, BFCP refinement (R188 NS-CSG), state quotient, MDL gate, `RegimeCollapseClassifier` |
| ASP solver / answer set programming | WASM validator, Lean4 agent (R198/P223), `rustc`/`syn` validator (Plan 007) |
| operator `o_i`, skill `π_i` | `ConstraintPruner`, `SkillDescriptor`, induced CWM kernel, functor application |
| action-step sub-policy `π_{i,j}` with termination | `SalienceTriGate::decide` (per-tick emit), leaky integrator decay gate, `latent_functor` stage gating |
| node transition `τ_node = (n, l, n′)` | `EventLog` entry (Plan 124), `TrialLog` row (BanditPruner), KG triple (`vibe.rs`) |
| task-relevant feature selector `φ` | `ScreeningPruner`, zone gating, attention matching, salience probe |
| demonstration / imitation learning | replay folder (Plan 039), episode buffer, posterior-guided skill evolution |
| MetricFF planner | `mcts_search`, `ismcts_search_with_inference`, AND-OR DDTree blueprint pass |

### 2.3 Latent-space reframing (mandatory before verdict)

The NSM pipeline has two latent-space angles:

1. **Operator = direction vector in skill-latent space.** A "skill" `π_i` is, in our reframing, a direction in the latent subspace spanned by HLA / `latent_functor` / `polytope_router`. Selecting an operator `o_i` to apply = projecting the current latent state onto the skill direction and gating via sigmoid (already the `polytope_router` / `personality_weighted_composition` pattern). The options-framework termination condition is a per-tick salience-tri-gate decision. The NSM paper's "skill library" is structurally the `SkillCatalog` (Plan 192) plus the per-NPC frozen CWM pool (R275 fusion).

2. **Bisimulation compaction = state-equivalence quotient = the missing PTG primitive.** This is where Research 264 explicitly flagged a gap: *"Primitive Transition Graph (PTG) as explicit runtime data structure… motif mining → motif wrapping → higher-order primitive promotion."* The NSM paper's minimal-bisimulation step is the concrete algorithmic instantiation of that gap. Quotienting a transition graph by bisimulation equivalence (two states equivalent iff same observable future) produces the minimal deterministic representation that preserves planning-relevant behavior — a *lossless* compression of the state space into operator-keyed classes. This is the **raw-side** counterpart to the latent-side `latent_functor` projection.

**Bridge to chain:** the quotient graph is small, discrete, and deterministic — perfect LatCal commitment material. A bisimulation quotient is a chain-committable artifact: BLAKE3 over `(state_classes, edges, operator_labels)` gives a tamper-evident canonical rule reference, identical to how `CwmCommitment` works for code.

### 2.4 Fusion — what does this paper × R275 (CWM) × R264 (CEI) produce?

| Fusion | Source A | Source B | Novel combination? |
|---|---|---|---|
| **Bisimulation quotient as PTG** | NSM paper (minimal bisimulation `G̅`) | R264 gap (PTG data structure missing) | **Closes R264 gap #1 concretely.** PTG = bisimulation quotient graph of observed trajectories. Operator nodes = equivalence classes; edges = labeled transitions. |
| **Motif mining = repeated sub-path in quotient** | NSM paper (operator consolidation) | R264 gap #2 (motif mining missing) | **Closes R264 gap #2 partially.** Recurring sub-paths in the quotient graph *are* the candidate motifs; wrap as composite operator via MDL gate (already shipped, Plan 215). |
| **Bisimulation-induced CWM** | NSM paper (symbolic domain σ) | CWM primitive (Plan 296 — induce *code*) | **Complementary.** CWM induces executable code (richer but heavier). NSM's bisimulation + ASP induces a PDDL-like operator schema (lighter, more compositional). Two ends of the same induction spectrum; the runtime can pick per-task. |
| **Per-NPC heterogeneous rule models** | NSM paper (different rule models per NPC) | CWM per-NPC pool fusion (R275) | **Refinement.** Different NPCs observe different demos → different bisimulation quotients → emergent diversity in their induced rule models. Already foreshadowed in R275; this paper gives the lighter-weight induction path. |
| **Chain-committed rule consensus** | NSM paper (PDDL as canonical rule reference) | LatCal commitment | **Force multiplier.** Faction agrees on rules → co-induce quotient → BLAKE3-commit → canonical for anti-cheat replay. Same pattern as `CwmCommitment`, applied to a smaller artifact. |

**The genuinely novel transferable piece** is the **bisimulation-based operator-abstraction primitive**: an algorithm that takes an observed transition graph and emits a minimal quotient + a PDDL-like operator schema, with zero training, that closes the R264 PTG/motif gap. This is Plan 324.

---

## 3. Verdict

### **GOAT — empirical validation of an existing Super-GOAT capability class + one transferable primitive that closes a flagged gap**

**One-line reasoning:** The paper's capability-class contribution ("system observes structured task, induces verifiable rules, plans via search, generalizes to unseen variants") is already shipped as the CWM Super-GOAT (R275/Plan 296). What's genuinely transferable and unshipped is the **bisimulation-based state-equivalence compactor + PDDL-like operator-inference pipeline**, which closes the explicit R264 gap on the Primitive Transition Graph and motif-mining loop. The empirical head-to-head is supporting evidence for the existing selling point, not a new one.

### Novelty gate (§1.5 of skill protocol)

| Question | Answer |
|---|---|
| **Q1: No prior art?** | **NO.** CWM (R275/Plan 296) ships the *capability class* — "induce forward model from trajectories + plan via search." MUSE/ITSE (R172/Plan 192) ships skill lifecycle. Plan 215 ships MDL-gated primitive admission. The specific missing piece (bisimulation compaction + operator inference from a transition graph) is incremental over these foundations, not greenfield. |
| **Q2: New class of behavior?** | **NO.** "NPC observes new game, induces rules, plans, generalizes" is already the CWM Super-GOAT capability class. NSM is a *narrower variant* (PDDL operators vs code, bisimulation vs REx refinement). |
| **Q3: Product selling point?** | Partial. "Empirical evidence: our architecture beats π0 VLA 3× on success, ∞× on unseen variants, ~80× on training energy, ~10× on inference energy" is supporting evidence for the existing selling point — strong for marketing the CWM moat, not a new moat. |
| **Q4: Force multiplier ≥2 pillars?** | YES. Closes the R264 PTG gap. Connects CWM (R275) × Closure-Expansion (R264) × MUSE skill lifecycle (R172) × Regime-Transition MDL gate (Plan 215) × LatCal chain commitment. |

**2 NO → not Super-GOAT.** Proceed to GOAT.

### Why not Pass

The bisimulation-compaction primitive is (a) *concretely unshipped* (confirmed by R264 §2.2 explicit gap), (b) *modelless* (graph algorithm + ASP-like inference, no training), (c) *closes a flagged gap* in our open-ended-intelligence stack, (d) *force-multiplier across ≥4 pillars*. Pass would lose genuine value.

### Why not Super-GOAT

- Q1 fails: the capability class is already shipped (CWM).
- Q2 fails: no new behavior — NPCs can already learn-new-game-by-observation via CWM; this paper's contribution is a *lighter-weight induction path* (PDDL operators instead of executable code), not a new capability.
- Q3 fails: the empirical result strengthens the existing selling point, does not create a new one.

### Routing

| Deliverable | Repo | Action |
|---|---|---|
| Research note (this file) | `katgpt-rs/.research/308_*.md` | ✅ Created |
| Plan: Bisimulation Operator Inference primitive | `katgpt-rs/.plans/324_bisimulation_operator_inference.md` | ✅ Will create |
| Open primitive: `BisimulationQuotient` + `OperatorInference` + `TransitionGraph` | `katgpt-rs/crates/katgpt-core/src/bisimulation/` (new module) | Plan 324 phases 1–4 |
| riir-ai guide | — | **NOT created** (verdict ≠ Super-GOAT; the existing CWM guide at `riir-ai/.research/145_CWM_Runtime_Induced_Game_Rules_Guide.md` covers the selling point this paper validates) |
| riir-train routing | — | The diffusion-policy skill training is riir-train territory; the bisimulation/ASP/operator-inference half is modelless and stays here. One-line note: *"diffusion policy training for NSM-style skills → riir-train; bisimulation quotient + PDDL operator inference → katgpt-rs Plan 324."* |

---

## 4. Latent vs raw boundary (riir-armageddon compliance check)

The bisimulation quotient straddles the boundary cleanly:

| Artifact | Space | Synced? | Why |
|---|---|---|---|
| `StateClassId: u32` (bisimulation equivalence class) | Raw (discrete tag) | YES if committed | Bit-identical replay requires deterministic class enumeration |
| `OperatorLabel: u8` (`#[repr(u8)]` enum) | Raw | YES | Anti-cheat: same operator → same plan → same outcome |
| `Edge { from: u32, to: u32, op: u8 }` | Raw | YES | Deterministic transition structure |
| `blake3_hash` of canonical quotient graph | Raw (32-byte commitment) | YES (audit) | Tamper-evidence — same as `CwmCommitment` |
| `class_embedding: [f32; K]` (latent vector per class, for similarity/mining) | Latent | NO — local only | Used for PRI / motif clustering / similarity search, not for state reconstruction |
| `motif_reuse_count: u32` | Raw counter | YES (aggregate) | Deterministic given same execution history |

**Bridge functions:** `transition_graph_to_quotient()` (raw graph → raw quotient, deterministic partition-refinement algorithm — *modelless*); `class_id_to_embedding()` (raw→latent, dot-product projection onto class direction vectors, sigmoid-bounded, zero-allocation); `embedding_to_motif_score()` (latent→raw scalar, clamp to [0,1]). All bridge functions gateable by feature flag, zero sync dependency.

**Anti-pattern avoided:** never encode the latent `class_embedding` as the synced state identity — the `StateClassId` (raw u32) is the sync identity, the embedding is local-only for similarity search. This mirrors the HLA rule: sync the 5 scalars, not the 64-dim vector.

---

## 5. Connection map (force-multiplier analysis)

```
                          ┌─ Plan 296 Induced CWM ────┐
                          │  (induce executable code)  │
                          └─────────────┬──────────────┘
                                        │ complements
                                        ▼
            ┌─ R308 NSM paper ──► Plan 324 Bisimulation Operator Inference
            │                          (induce PDDL-like operator schema)
            │                                  │
            │                                  │ closes gap
            ▼                                  ▼
    R264 Closure-Expansion ◄─── PTG data structure + motif mining
    Instrument (Plan 290)        now concretely instantiable
            │
            │ feeds
            ▼
    Plan 215 Regime-Transition MDL gate  ◄── admits promoted motifs as new primitives
            │
            │ feeds
            ▼
    Plan 192 MUSE/ITSE SkillCatalog  ◄── new operators register as skills
            │
            │ consumed by
            ▼
    Plan 303 Salience Tri-Gate  ◄── per-tick termination for each skill
            │
            ▼
    Plan 211 Three-Mode NS Router  ◄── picks L4R/R4L/LR per decision
            │
            ▼
    riir-chain LatCal commitment  ◄── quotient graph is chain-committable
```

---

## 6. What NOT to take

- **Diffusion policy training** for low-level skills `π_{i,j}`. Training-side → riir-train. Our plasma/hot runtime consumes frozen skills; we do not train them.
- **YOLOv8 object detector** for pose estimation. Domain-specific perception; out of scope for the modelless engine.
- **VLA fine-tuning methodology** (PaliGemma-2B + Gemma-300M LoRA configs). Training-side → riir-train.
- **Robosuite simulation harness.** Game/sim-domain; not engine territory.
- **π0 / OpenVLA / GR00T / UniVLA comparisons.** Reference baselines; nothing to distill.
- **GPT-5 / Qwen / PaliGemma planner evaluations.** Confirms Kambhampati; no primitive here.

---

## 7. Validation protocol (for Plan 324 GOAT gate)

The bisimulation primitive's GOAT gate (G1–G5):

- **G1 — Bisimulation correctness.** Given a known graph with a known minimal bisimulation, the partition-refinement algorithm produces the canonical quotient. Property test: re-running on the same graph yields bit-identical class assignments.
- **G2 — Operator inference soundness.** Given a quotient graph with labeled edges, the inferred operator schema covers every observed transition (no missing operators) and admits no spurious operators (every operator is exercised by at least one edge).
- **G3 — Plan validity.** A classical planner running on the inferred operator schema produces plans that, when executed against the original transition graph, never violate preconditions and always reach the goal (when a plan exists).
- **G4 — Latency.** Partition refinement on a graph of N nodes completes in O(N² log N) worst case; target ≤ 1 ms for N=1024 on plasma-tier CPU SIMD.
- **G5 — Zero-alloc hot path.** The quotient lookup (`class_id(state) -> u32`) is O(1) hash or direct index, no heap allocation across 10⁶ queries.

No quality gate against a real robot/game — that requires the diffusion-skill half which lives in riir-train. The modelless half is validated structurally.

---

## 8. Energy framing → existing perf tiers (supporting evidence, not new primitive)

The paper's ~80× training-energy and ~10× inference-energy ratios validate the existing plasma→hot→warm→cold→freeze tiering strategy:

- **NSM training (34 min, 0.85 MJ)** = warm-tier offline induction (modelless graph algorithm + ASP solver). No GPU.
- **NSM inference (CPU-only, 19 W)** = plasma/hot tier. The quotient graph and operator schema fit in L1/L2 cache; `class_id` lookups are sub-µs.
- **VLA training (1.5+ days, 65 MJ)** = the anti-pattern we avoid. LoRA fine-tuning at 100% GPU saturation.
- **VLA inference (GPU required, ~115 W)** = hot/warm tier with GPU dependency — *exactly what the plasma-tier (µs CPU SIMD) design avoids.*

**Selling-point reinforcement** (for the existing CWM Super-GOAT, not a new claim): *"Our runtime matches the energy profile of the winning architecture in this paper — CPU-only inference, 19 W, no GPU — because the induced rule model is a frozen, BLAKE3-committed artifact, not a running neural network."* This goes into the CWM guide's marketing section, not a new guide.

---

## 9. Paper metadata

- **Authors:** Timothy Duggan¹, Pierrick Lorang¹·², Hong Lu¹, Matthias Scheutz¹·∗
- **Affiliations:** ¹Human-Robot Interaction Lab, Tufts University; ²AIT Austrian Institute of Technology, Vienna
- **Funding:** ONR grant #N00014-24-1-2024
- **Code/models:** https://price-is-not-right.github.io
- **Underlying NSM method:** Lorang, Lu, Huemer, Zips, Scheutz. "Few-shot neuro-symbolic imitation learning for long-horizon planning and acting." arXiv:2508.21501, Aug 2025.
- **Related cited work we already distill:**
  - Bonet & Geffner 2020 (ASP-based symbolic domain inference) — the SPASS-style learning-first-order-representations thread
  - Sutton, Precup, Singh 1999 (options framework) — maps to Salience Tri-Gate (R281/P303)
  - Chi et al. 2024 (diffusion policy) — maps to DFlash (R034) and D2F (R066); training-side → riir-train
  - Kambhampati et al. 2024 (LLMs can't plan, LLM-modulo) — validates our NS-router three-mode design (R186/P211)
  - Konidaris, Kaelbling, Lozano-Perez 2018 (from skills to symbols) — the inverse of CWM (symbols from skills vs skills from symbols); bisimulation is the bridge

---

## TL;DR

Empirical validation paper (arXiv:2602.19260): neuro-symbolic (PDDL planner + diffusion skills) beats π0 VLA on Towers of Hanoi (95% vs 34% on 3-block, 78% vs 0% on unseen 4-block, ~80× less training energy, ~10× less inference energy, CPU-only at runtime). **Capability class already shipped as CWM Super-GOAT (R275/Plan 296)** — this is empirical reinforcement, not a new moat. **GOAT verdict**: one transferable primitive unshipped, **bisimulation-based operator abstraction** (graph → minimal bisimulation quotient → PDDL-like operator schema), which closes the explicit R264 (Closure-Expansion Instrument) gap on the Primitive Transition Graph + motif mining. Plan 324 ships it as an open primitive in `katgpt-rs/crates/katgpt-core/src/bisimulation/`. Diffusion-skill training half → riir-train. No private guide (verdict ≠ Super-GOAT); the existing CWM guide at `riir-ai/.research/145_*.md` covers the selling point this paper validates.
