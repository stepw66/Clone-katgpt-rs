# Research 374: OTF-LAM — Factorized Transition Primitives for Compositional Action Abstraction

> **Source:** [Latent Actions from Factorized Transition Effects under Agent Ambiguity](https://arxiv.org/abs/2606.30544) — Heejeong Nam, Chandradithya S Jonnalagadda, Harshit Aggarwal, Eric Xu, Randall Balestriero (Brown University), arXiv:2606.30544v1, 30 Jun 2026
> **Date:** 2026-07-03
> **Status:** Done — verdict locked (**GOAT for katgpt-rs**)
> **Classification:** Public (this note). Training recipe → riir-train.
> **Related Research:** 123 (Latent Functor Runtime — **ships the monolithic version as Super-GOAT; this paper is the factorized refinement**), 358 (SMWM — same-author Balestriero, PASS, monolithic runtime analog), 360 (AdaJEPA — PASS, monolithic runtime analog), 275 (Induced CWM — frozen forward model), 192 (NextLat — belief-state latent dynamics), 138 (LeJEPA — same-author Balestriero, LOW-MODERATE GAIN), 303 (FUNCATTN predecessor — Galerkin-style attention)
> **Related Plans:** 273 (latent_functor arithmetic — the monolithic baseline), 375 (this paper's open primitive), 297 (PersonalityWeightedComposition — weighted layer composition cousin), 296 (InducedCwmKernel — the frozen `g_φ`)
> **Domain:** katgpt-rs (this note, public — the open primitive). Runtime wiring → riir-ai (cross-ref only, no guide this session — GOAT, not Super-GOAT).

---

## TL;DR

OTF-LAM factorizes each observation transition `(x_t, x_{t+1})` into a **sparse set of K reusable observed-transition primitives** via a patchwise VQ codebook, then aggregates them into a compact action-like latent via a **state-aware sigmoid relevance gate + normalized weighted average**. The codebook transfers **zero-shot across visual carriers and morphologies** (walker→cheetah, digit-0→digit-5), and the factorization suppresses distractor entanglement that monolithic latent action models suffer. A decoder-free variant (OTF-LAM-Dino) predicts future states in a frozen DINO representation space and **outperforms** the pixel-decoder version.

**Verdict: GOAT for katgpt-rs.** The paper's factorized mechanism is genuinely novel vs our shipped **monolithic** latent functor (`extract_functor`/`apply_functor`, Research 123/Plan 273 — a single mean-displacement vector or single rank-k operator). The transferable inference-time primitive — "given a frozen codebook of K effect primitives, compute per-primitive sigmoid relevance conditioned on current state, then normalized-gated-average into an action latent" — does NOT ship (grep confirms codebooks exist only for KV-cache compression, never for transition factorization). The modelless path is viable: k-means codebook construction (deterministic, runtime, no gradient) + linear patch encoding + sigmoid gate + normalized weighted average. The training-only parts (VQ-VAE codebook learning, behavioral cloning policy, action decoder) → riir-train.

**Distilled for katgpt-rs (modelless, inference-time):** a `FactorizedActionAbstraction` primitive — frozen codebook `C = {c(1)..c(K)}` of D-dim effect vectors + per-primitive sigmoid relevance gate `α_k = σ(G_θ(r_t,k))` + normalized weighted average `z = Σ α_k r_k / (Σ α_k + ε)`. The codebook can be (a) a trained artifact loaded from disk, (b) a runtime k-means fit on observed transitions, or (c) a `NeuronShard`-stored Pod (cross-ref riir-neuron-db). The aggregation is zero-allocation, sigmoid-gated (never softmax), feature-flagged.

---

## 1. Paper Core Findings

### 1.1 The agent-ambiguity problem (the motivation)

A Latent Action Model (LAM) encodes `(x_t, x_{t+1})` into a latent action `z_t` and predicts `x_{t+1}` from `(x_t, z_t)`. In controlled settings (robot arm, dominant agent), the visual transition is mostly agent motion. In distractor-rich scenes (DCS: cheetah-run, walker-run with background motion, camera dynamics), the transition mixes agent motion + distractors + camera + background — all arrive through the same pixel channel. A monolithic latent action entangles these; the model has "no direct basis for deciding which observed changes should be attributed to the controlled agent."

### 1.2 Observed Transition Factorization (OTF) — the vocabulary stage

Decompose each transition into a sparse composition of **reusable observed-transition primitives**:

1. **Motion input** `o_t`: gradient/Sobel transform applied per-frame, then temporal difference `o_t = ϕ(x_{t+τ}) − ϕ(x_t)`. Suppresses static appearance, preserves transition evidence.
2. **Patchify**: partition `o_t` into P non-overlapping spatial patches; encode each via shallow MLP into token `f_{t,i} ∈ ℝ^D`.
3. **Patchwise VQ**: assign each token to nearest codebook entry `c(k*)` via top-1 NN quantization (Eq. 3). Codebook `C = {c(1)..c(K)}`, K typically 16–128.
4. **Per-code occupancy** `M(k)_t` (spatial support) + **activation strength** `w(k)_t` (fraction of patches assigned). Together form the transition factor set `E_t = {(c(k), M(k)_t, w(k)_t)}`.
5. **Train** via VQ-VAE objective: reconstruction `L_rec` + codebook `L_code` + commitment `L_commit` + orthogonality `L_orth` (Eq. 5–8). Codebook updated by EMA; rarely-used entries reinitialized.

**Key design choice (Remark 1):** quantization structures *local observed effects*, NOT the full action as a single code. The discrete bottleneck is at the patch level; the final action latent is continuous (aggregated from primitives).

### 1.3 OTF-LAM — action abstraction on frozen factors

Freeze the OTF factorizer. Train a LAM on the extracted factor set:

1. **State-aware factor token**: `r_{t,k} = Γ_θ(c(k), M(k)_t, w(k)_t, x_t)` — embeds code identity + occupancy + activation + current-frame context.
2. **Sigmoid relevance gate**: `α_{t,k} = σ(G_θ(r_{t,k})) ∈ [0,1]` — per-primitive relevance, inactive codes masked out. **Sigmoid, not softmax** (the paper uses sigmoid gating throughout).
3. **Normalized gated average**: `z^fac_t = Σ_k α_{t,k} r_{t,k} / (Σ_k α_{t,k} + ε)`.
4. **Projection**: `z^act_t = P_θ(z^fac_t)`.
5. **Forward dynamics**: `x̂_{t+τ} = p_θ(x_t, z^act_t)` — residual prediction (next frame = current + predicted delta).

The action abstraction module **does not assume the factorizer identified the controlled agent** — it only learns which observed-transition factors are useful for forming a compact latent for prediction.

### 1.4 OTF-LAM-Dino — decoder-free JEPA variant

Replace pixel-space prediction with prediction in a **frozen DINOv2 representation space**. Both the observed-transition vocabulary AND the visual state encoder are frozen/reusable; the learned model focuses on abstracting transition info into `z^act` and predicting its effect in representation space. **Empirically outperforms** the decoder-based OTF-LAM (cheetah-run: 43.80 vs 25.79 mean return) because frozen DINO suppresses pixel-level nuisance factors (texture, lighting, background) while preserving control-relevant state.

### 1.5 Empirical results

- **Cross-morphology transfer** (Table 1): OTF codebook trained on walker-run, evaluated zero-shot on cheetah-run. Transfer degradation 20–52% (OTF) vs 58–72% (monolithic VQ-VAE). The factorized codebook is less sensitive to morphology shift.
- **Carrier transfer** (Moving MNIST): trained on digits {0–4}, evaluated on {5–9}. OTF maintains usable motion reconstruction; monolithic falls back toward static reference appearance.
- **Policy learning** (Table A1): OTF-LAM-Dino 43.80 (cheetah) / 28.87 (walker) — competitive with or beats FLAM(8) 31.95/34.97, HiLAM 22.64/25.12, LAPO 11.48/28.71.
- **Codebook size** (Table A2): OTF-LAM improves monotonically with K on cheetah (10→26 as K goes 16→128); OTF-LAM-Dino is non-monotonic but consistently above decoder-based at every K.

---

## 2. Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent | Where it ships / would land |
|---|---|---|
| latent action `z^act_t` | action-direction vector, motor gate latent, policy latent | `latent_functor/` direction vectors; HLA motor channels |
| observed transition `(x_t, x_{t+1})` | frame delta, TxDelta, temporal derivative, belief-state transition | `katgpt-core/src/temporal_deriv.rs` (DEFAULT-ON Plan 277); `latent_functor/arithmetic.rs` source/target pairs |
| observed-transition primitive `c(k)` | codebook entry, effect atom, shard style dimension | **DOES NOT SHIP for transitions** — codebooks exist only for KV compression (`katgpt-kv`, Lloyd-Max). This primitive would be new. |
| motion input `o_t` (gradient/Sobel + temporal diff) | temporal derivative signal | `katgpt-core/src/temporal_deriv.rs` (DEFAULT-ON) — ships the dual fast/slow surprise signal |
| patchify + patchwise VQ | block quantization, patch embedding | KV-cache has K-means VQ on groups of 4 channels; no transition-patch equivalent |
| occupancy map `M(k)_t` | spatial support mask, attention mask | DEC cochains (`terrain_cochains.rs`); zone density maps |
| activation strength `w(k)_t` | code usage frequency, slot utilization | `ShardIndex` utilization; MoE expert load |
| state-aware factor token `r_{t,k} = Γ(c(k), M, w, x_t)` | state-conditioned embedding, FiLM-modulated token | `ega_attn.rs` FiLM-style gating; `funcattn` cross-feature interaction |
| sigmoid relevance gate `α_k = σ(G_θ(r))` | sigmoid gate (hard rule, never softmax) | Pervasive: `ega_attn`, `gdn2/kernel`, `rat_bridge/fuse`, `manifold_power_iter_router`, `latent_functor/arithmetic.rs::functor_gate` |
| normalized gated average `Σ α_k r_k / (Σ α_k + ε)` | soft attention, mean-field aggregation | `mean_field/` module; `set_attention.rs`; `PersonalityWeightedComposition::compose_into` (Plan 297) |
| frozen DINO representation space | frozen encoder, fixed embedding target | `InducedCwmKernel` (frozen `g_φ`); `BeliefInferenceFn` (observation→belief); sleep-time frozen rollforward |
| inverse dynamics (IDM) | `extract_functor` (estimate displacement from pairs) | `latent_functor/arithmetic.rs` — **monolithic** (single mean displacement) |
| forward dynamics (FDM) | `apply_functor` (predict target from source + functor); `InducedCwmKernel::advance` | `latent_functor/arithmetic.rs`; `katgpt-core/src/induced_cwm/` |
| agent ambiguity / distractor entanglement | curiosity = prediction-error signal (Pathak-style distractor filter) | `katgpt-core/src/temporal_deriv.rs` (DEFAULT-ON); CGSP (Plan 274) |
| cross-morphology / cross-carrier transfer | cross-game transfer, shard reuse across zones/archetypes | `latent_functor/cross_game.rs`; `NeuronShard` zone transfer; `ArchetypeBlendShard` |
| behavioral cloning policy | training-only | → riir-train |
| VQ-VAE codebook learning (k-means init + EMA + commitment loss) | training-only (OR runtime k-means — modelless unblock path) | → riir-train (trained); katgpt-rs (runtime k-means) |
| action decoder (latent → true action) | training-only | → riir-train |

---

## 3. Distillation — fusion angle (factorized vs monolithic)

### 3.1 The decisive prior-art check

The mandatory two-layer novelty check (notes + shipped code, per the Research 242 `evolve_hla` lesson) found:

| Paper mechanism | Shipped prior art | Match |
|---|---|---|
| Monolithic latent action (`z^act` = single vector) | `extract_functor`/`apply_functor` (Research 123, Plan 273) — Super-GOAT | ✅ Monolithic version ships |
| Inverse dynamics (`z_{t+1} ≈ z_t + ρ(a)`) | `latent_functor/arithmetic.rs` + `sleep_time::HlaSleepTimeOp` (DEFAULT-ON) | ✅ Ships (Research 358/360 PASS precedent) |
| Plan-execute-adapt-replan loop | `ReestimationScheduler` (Research 123, Plans 303/317) | ✅ Ships (Research 360 PASS precedent) |
| Frozen forward model (`g_φ`) | `InducedCwmKernel: GameState` (Plan 296) | ✅ Ships |
| Curiosity / distractor filter | Temporal Deriv Kernel (Plan 277, DEFAULT-ON) | ✅ Ships |
| **Factorized codebook of K transition primitives** | **NOT FOUND** — codebooks exist only for KV compression (`katgpt-kv` Lloyd-Max), never for transition/observation factorization | ❌ **Novel** |
| **Per-primitive sigmoid relevance gate over a codebook** | Sigmoid gates are pervasive, but NOT applied over a factorized transition codebook | ❌ **Novel as a combination** |
| **Normalized gated average of codebook primitives → action latent** | `PersonalityWeightedComposition` does weighted composition over LAYERS, not over a transition codebook | ❌ **Novel as a combination** |
| Cross-morphology codebook transfer | `latent_functor/cross_game.rs` transfers functors (monolithic); no codebook-transfer mechanism | ❌ **Novel** |

**The factorization is the novel angle.** The monolithic latent-action primitive already ships as Super-GOAT (Research 123). OTF-LAM is the **factorized/compositional refinement**: instead of one displacement vector, K discrete primitives combined via state-aware sigmoid gating.

### 3.2 Why factorization matters (the paper's empirical contribution)

The paper proves three things the monolithic approach cannot do:

1. **Distractor suppression via factorization.** In DCS (cheetah-run, walker-run with background motion), the monolithic LAPO baseline entangles agent motion with distractors (11.48 / 28.71 mean return). OTF-LAM's factorization lets the relevance gate select only action-relevant primitives (25.79 / 26.94). OTF-LAM-Dino reaches 43.80 / 28.87.
2. **Cross-morphology codebook transfer.** The OTF codebook trained on walker-run transfers zero-shot to cheetah-run with 20–52% degradation vs 58–72% for monolithic VQ-VAE (Table 1). The codebook captures reusable transition structure, not embodiment-specific templates.
3. **Compositional action representation.** Actions are mixtures of primitives. The same codebook serves different embodiments with different gating patterns — strictly richer than a single displacement vector per (source, target) pair.

### 3.3 Fusion — OTF-LAM × NeuronShard × latent_functor × HLA

The novel combination this paper enables:

```
                    ┌─────────────────────────────────────────┐
                    │  Frozen Effect Codebook (K primitives)   │
                    │  c(1)..c(K) ∈ ℝ^D                        │
                    │  Stored as: NeuronShard Pod (future)     │
                    │             OR runtime k-means fit       │
                    │             OR trained artifact (disk)   │
                    └─────────────┬───────────────────────────┘
                                  │
    transition (x_t, x_{t+1}) ───►│
                                  ▼
                    ┌─────────────────────────────────────────┐
                    │  Patchify + assign to codebook           │
                    │  → occupancy M(k), activation w(k)       │
                    └─────────────┬───────────────────────────┘
                                  │
                         HLA state x_t ──►───┐
                                  │           │
                                  ▼           ▼
                    ┌─────────────────────────────────────────┐
                    │  State-aware factor token               │
                    │  r_k = Γ(c(k), M(k), w(k), x_t)         │
                    │  Sigmoid relevance gate:                │
                    │  α_k = σ(G(r_k)) ∈ [0,1]                │
                    │  Normalized gated average:              │
                    │  z = Σ α_k r_k / (Σ α_k + ε)            │
                    └─────────────┬───────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────────────┐
                    │  Action latent z^act                     │
                    │  → feeds apply_functor (existing)        │
                    │  → or InducedCwmKernel::advance          │
                    │  → or forward dynamics predictor         │
                    └─────────────────────────────────────────┘
```

**The fusion produces a capability none of the pieces has alone:**
- `latent_functor` (monolithic) gives ONE displacement per transition — no compositional structure, no distractor selection.
- NeuronShard stores style vectors — but as a single embedding, not as a codebook of discrete action-effect primitives.
- HLA provides per-NPC state — but doesn't gate a codebook of primitives.
- **Fused:** the codebook provides a vocabulary of reusable effects; the HLA-conditioned sigmoid gate selects which effects are action-relevant FOR THIS NPC AT THIS MOMENT; the normalized average produces a per-NPC action latent that is compositional (a mixture) rather than monolithic (a single vector).

**Closest cousins across all 5 repos:**

| Cousin | Domain | Verdict | Overlap |
|---|---|---|---|
| **Research 123 (Latent Functor Runtime)** | riir-ai | **Super-GOAT, shipped** | The monolithic baseline — OTF-LAM is the factorized refinement |
| **Research 358 (SMWM)** | katgpt-rs | PASS | Same JEPA world-model domain; monolithic runtime analog already ships |
| **Research 360 (AdaJEPA)** | katgpt-rs | PASS | Same; plan-execute-adapt-replan already ships via ReestimationScheduler |
| **Research 297 (PersonalityWeightedComposition)** | katgpt-rs | Shipped | Weighted composition over LAYERS — the aggregation pattern, but not over a transition codebook |
| **Research 275 (Induced CWM)** | katgpt-rs | Shipped | The frozen `g_φ` forward model — the prediction target |
| **Plan 277 (Temporal Deriv Kernel)** | katgpt-rs | **DEFAULT-ON** | The motion-input `o_t` analog (dual fast/slow surprise signal) |
| **`katgpt-kv` (Lloyd-Max VQ)** | katgpt-rs | Shipped | K-means VQ on KV channels — the codebook mechanism, but for compression not transition factorization |

---

## 4. Mandatory latent-space reframing (per SKILL §1 step 3)

| Target substrate | OTF-LAM reframing | Status |
|---|---|---|
| **(a) HLA per-NPC latent state** | The state `x_t` that conditions the relevance gate `α_k = σ(G(r_k, x_t))`. Each NPC gates the SAME codebook differently → per-NPC compositional action understanding. | Novel application of HLA as gate-conditioner |
| **(b) `latent_functor/` operations** | The factorized cousin of `extract_functor`/`apply_functor`. Instead of one displacement, K primitives + sigmoid gate + normalized average. Composes with existing `apply_functor` as the final prediction step. | Novel — factorized extension of shipped monolithic functor |
| **(c) `cgsp_runtime/` curiosity signals** | The motion input `o_t` (temporal derivative) already ships as Plan 277 (DEFAULT-ON). The curiosity signal drives which transitions to add to the codebook-fitting buffer. | Already shipped (Plan 277); feeds the codebook construction |
| **(d) LatCal fixed-point commitment** | A frozen trained codebook could be committed via BLAKE3 as a `NeuronShard` Pod (cross-ref riir-neuron-db). The codebook is latent (D-dim vectors); commitment is raw (BLAKE3 root). | Future fusion — codebook-as-shard |
| **(e) `NeuronShard` style_weights / dendritic branch** | A new shard subtype (`EffectCodebookShard`) storing K×D codebook entries as a `#[repr(C)]` Pod. Cross-zone transfer = same shard, different NPC gating. | Future fusion — new shard type |
| **(f) DEC Stokes operators** | No direct reframing — OTF-LAM is action/composition-centric, not divergence/curl-centric. The occupancy maps `M(k)` could be cochains on a spatial grid, but the paper's mechanism doesn't require DEC. | N/A |

The factorized action abstraction is a genuine latent-to-latent operation: transition → patchify → codebook assignment (latent) → sigmoid gate (latent) → normalized average (latent) → action latent (latent). Only the final scalar projections (if any) cross the sync boundary. This satisfies AGENTS.md constraint #2 (latent-to-latent preferred, sigmoid never softmax).

---

## 5. §3.5 Modelless unblock check — MANDATORY before any riir-train deferral

The paper IS training-heavy (VQ-VAE codebook learning, behavioral cloning, action decoder). Per §3.5, check all three modelless paths before deferring:

### Path 1: Freeze/thaw snapshot correction
- **Can a frozen snapshot fix the issue?** The codebook IS a frozen artifact once constructed. The question is whether it can be constructed modellessly.
- **Verdict:** The codebook CAN be frozen after construction, but construction itself needs Path 2 or 3.

### Path 2: Raw/lora reader-writer hot-swap (deterministic construction)
- **Can a deterministically constructed adapter fix the issue?** The codebook can be constructed via **k-means clustering** on observed transition patches. K-means is deterministic iterative refinement (Lloyd's algorithm), NOT gradient descent. It converges to a local optimum from a fixed seed.
- **The modelless OTF construction:**
  1. Collect transition patches `{f_{t,i}}` from observed `(x_t, x_{t+1})` pairs (the `o_t` signal already ships as Plan 277).
  2. Run k-means with K clusters → codebook `C = {c(1)..c(K)}`.
  3. Assign each patch to nearest codebook entry → occupancy `M(k)`, activation `w(k)`.
  4. State-aware sigmoid gate + normalized weighted average → action latent.
- **Verdict:** K-means codebook construction is modelless. The patch encoder can be a linear projection (flatten + normalize) — also modelless. The full inference path (gate + aggregate) is modelless.

### Path 3: Latent-space correction (dot-product projection + sigmoid gate)
- **Can a latent-space projection/gate fix the issue?** The relevance gate `α_k = σ(G(r_k))` IS a latent-space projection + sigmoid. This is already the modelless analog of a learned gate.
- **Verdict:** Already modelless by construction.

### Decision

**MODELLESS-VALIDABLE.** The factorized action abstraction primitive can be implemented modellessly via:
- K-means codebook construction (Path 2 — deterministic, runtime)
- Linear patch encoding (flatten + normalize)
- Sigmoid relevance gate (Path 3 — modelless by construction)
- Normalized weighted average (modelless arithmetic)

The trained version (VQ-VAE with EMA codebook updates + commitment loss + MLP patch encoder) will produce a higher-quality codebook, but the modelless baseline is sufficient for the GOAT gate. The training-only refinement → riir-train (one-line note, §8).

**What genuinely requires gradient descent:** behavioral cloning policy (mapping latent actions to true environment actions) and the action decoder. These are RL training pipelines → riir-train. But the inference-time action abstraction module (codebook + gate + aggregate) does NOT require training.

---

## 6. Novelty gate (§1.5)

| Q | Answer | Evidence |
|---|---|---|
| **1. No prior art?** | **PARTIAL YES** | The factorized codebook of K transition primitives + per-primitive sigmoid gate + normalized weighted average does NOT ship (grep confirms codebooks exist only for KV compression). The monolithic version ships (Research 123). The factorization is novel. |
| **2. New capability class?** | **PARTIAL** | Compositional action understanding (actions as mixtures of primitives) is richer than monolithic displacement, and distractor suppression via factorization is a qualitative gain. But it's an extension of the existing latent-functor capability class, not a fundamentally new class. |
| **3. Product selling point?** | **MODERATE** | "Our NPCs understand actions as compositions of reusable effect primitives, with the same codebook serving different archetypes via different HLA-conditioned gating patterns" — real selling point, but incremental over the committed-personality + latent-functor pitch. |
| **4. Force multiplier (≥2 pillars)?** | **YES** | Connects NeuronShard (P2 — codebook storage), Reasoning Pack (P8 — action abstraction), HLA (state-aware gating), Committed Personality (per-NPC gating divergence). ≥3 pillars. |

**Q2 is the decisive NO for Super-GOAT.** The factorized action abstraction is a provable improvement over the monolithic latent functor, but it does not create a new capability CLASS — it enriches an existing one. The recent precedent (358 SMWM PASS, 360 AdaJEPA PASS, 138 LeJEPA LOW-MODERATE GAIN) confirms the codebase's high bar for world-model papers: the runtime analog must be genuinely absent, not just a refinement.

**Verdict: GOAT.** Provable quality gain (compositional understanding, distractor suppression, cross-zone codebook transfer) over the existing monolithic approach. Not a new capability class. Plan + feature flag + benchmark.

---

## 7. MOAT gate per domain (§1.6)

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| `katgpt-rs` (public) | **In-scope** | Paper-derived fundamental primitive (factorized codebook + sigmoid gate + soft attention) that passes GOAT via fusion with latent_functor + HLA. Generic math, no game semantics. | **Open primitive** → `katgpt-rs/crates/katgpt-core/src/` (new module `factorized_action/`). Plan 375. |
| `riir-ai` (private runtime) | **In-scope** | Per-NPC wiring (HLA state → gate conditioner), zone-transfer tuning, cross-archetype codebook sharing. Fusion-GOAT connecting ≥2 pillars. | **Plan cross-ref** (no guide this session — GOAT, not Super-GOAT). Future riir-ai plan for runtime wiring. |
| `riir-chain` (private chain) | Out of scope | N/A — no chain commitment angle | — |
| `riir-neuron-db` (private shards) | **Future fusion** | `EffectCodebookShard` Pod subtype storing K×D codebook entries. Cross-zone transfer = same shard, different gating. | **Cross-ref** — future plan for new shard type |
| `riir-train` (private training) | **In-scope** | VQ-VAE codebook learning, behavioral cloning policy, action decoder training | **→ riir-train** (one-line note, §8) |

---

## 8. → riir-train (one-line redirect per SKILL §"Redirect to riir-train")

The training-only parts of OTF-LAM:
- **VQ-VAE codebook learning** (k-means init + EMA updates + commitment loss + orthogonality regularizer) — produces a higher-quality codebook than runtime k-means.
- **Behavioral cloning policy** — distills the learned latent action space into a policy `π(z^act | x_t)`.
- **Action decoder** — maps latent actions to true environment actions using a small action-labeled dataset (32 trajectories in the paper).

If prioritized, file a plan in `riir-train/.plans/` implementing the VQ-VAE objective `L_vocab = L_rec + λ1 L_code + λ2 L_commit + λ3 L_orth` with EMA codebook updates, and A/B-test against the modelless k-means baseline on a controlled toy domain (Moving MNIST or DCS-style). Hypothesis: trained codebook wins on reconstruction quality but the modelless k-means baseline captures most of the cross-morphology transfer benefit (the paper's Table 1 shows even simple motion inputs transfer well). Not pursued here — out of scope for this workflow.

---

## 9. GOAT gate design (for Plan 375)

The GOAT gate must prove the factorized primitive provides a **provable quality gain** over the monolithic baseline. Per §3.6 (defend-wrong PoC), the quality claim is qualitative ("distractor suppression", "cross-zone transfer") and therefore requires a head-to-head PoC on a controlled toy benchmark.

**Benchmark domain:** Moving MNIST-style synthetic transitions (digits moving on a 2D plane, with optional distractor motion). This is the paper's controlled diagnostic (§4.1) and maps cleanly to our 2D top-down arena.

**Three competitors (minimum per §3.6):**
1. **Monolithic baseline** — `extract_functor` + `apply_functor` (single mean displacement per transition).
2. **Factorized OTF (modelless)** — k-means codebook (K=32) + linear patch encoder + sigmoid relevance gate + normalized weighted average.
3. **Frozen/no-adaptation baseline** — identity transition (predict `x_{t+1} = x_t`).

**Gates:**
- **G1 (correctness):** reconstruction MSE on in-distribution transitions. Factorized ≤ monolithic.
- **G2 (distractor suppression):** reconstruction MSE on transitions WITH distractor motion. Factorized << monolithic (the paper's key claim).
- **G3 (cross-carrier transfer):** codebook trained on digit-{0–4}, evaluated on digit-{5–9}. Factorized transfer degradation < monolithic.
- **G4 (latency):** factorized aggregation < 500 ns per transition at K=32, D=8. Zero-alloc after warmup.
- **G5 (sigmoid never softmax):** the relevance gate uses sigmoid, verified by construction + test.
- **G6 (feature isolation):** the primitive compiles under `--features factorized_action` and `--no-default-features`.

**Promote/demote:** if G1–G3 pass, promote `factorized_action` to default-on (it enriches the latent-functor stack). If G2 fails (no distractor suppression gain), keep opt-in and note the modelless k-means codebook is insufficient (→ riir-train for trained VQ-VAE).

---

## TL;DR

**Paper:** *Latent Actions from Factorized Transition Effects under Agent Ambiguity* (Nam et al., Brown, arXiv:2606.30544, 2026-06-30). OTF-LAM factorizes observation transitions into a sparse VQ codebook of K reusable observed-transition primitives, then aggregates them via a state-aware sigmoid relevance gate + normalized weighted average into a compact action latent. The codebook transfers zero-shot across morphologies; the factorization suppresses distractor entanglement.

**Verdict: GOAT for katgpt-rs.** The factorized mechanism is genuinely novel vs our shipped **monolithic** latent functor (Research 123/Plan 273 — single displacement vector). The transferable inference-time primitive — frozen codebook + sigmoid relevance gate + normalized weighted average — does NOT ship (codebooks exist only for KV compression). The modelless path is viable (k-means codebook + linear encoding + sigmoid gate). Training-only parts (VQ-VAE, behavioral cloning, action decoder) → riir-train. Plan 375 implements the open primitive behind `factorized_action` feature flag; GOAT gate compares factorized vs monolithic on Moving-MNIST-style transitions (G1 correctness, G2 distractor suppression, G3 cross-carrier transfer, G4 latency, G5 sigmoid, G6 isolation).

**Files created this session:** `katgpt-rs/.research/374_OTF_LAM_Factorized_Transition_Primitives.md` (this note) + `katgpt-rs/.plans/375_factorized_transition_action_abstraction.md` (the plan). No private guide (GOAT, not Super-GOAT). No riir-train file (one-line redirect only).

---

## 10. Code Verification Addendum (2026-07-03)

The official code repo ([Hazel-Heejeong-Nam/lam_agent_ambiguity](https://github.com/Hazel-Heejeong-Nam/lam_agent_ambiguity), MIT license) was inspected to verify the distillation above against the actual implementation.

### Verified accurate (my distillation matches the code)

| My claim (§1.2–1.3) | Code evidence | Status |
|---|---|---|
| Sigmoid relevance gate `α_k = σ(G_θ(r))` | `GateNetwork.forward()`: `return torch.sigmoid(self.out_linear(x))` | ✅ **Exact match** — sigmoid, not softmax |
| Normalized gated average `z = Σ α_k r_k / (Σ α_k + ε)` | `OTFLAM.forward()` step 6: `alpha_sum = alpha.sum(dim=1).clamp_min(self.eps); z_factor = (alpha * factor_embedding).sum(dim=1) / alpha_sum` | ✅ **Exact match** |
| VQ codebook with k-means init + EMA updates | `default_config.yaml`: `codebook_init: kmeans`, `codebook_update: ema`, `ema_decay: 0.99`, `dead_code_steps: 1000`, `kmeans_iters: 20` | ✅ **Exact match** |
| Motion input `o_t = ϕ(x_{t+τ}) − ϕ(x_t)` (velocity) or acceleration | `motion_transforms.py::compute_motion_signal()`: velocity = `transformed_next - transformed_current`; acceleration = `transformed_next - 2.0 * transformed_current + transformed_previous`. Default is **acceleration** (second-order). | ✅ **Match** (default is acceleration, not velocity) |
| Inactive codes masked out | `OTFLAM.forward()`: `if self.mask_inactive_factors: alpha = alpha_raw * active_mask.unsqueeze(-1)` | ✅ **Exact match** |
| Codebook defaults K=128, D=32 | `default_config.yaml`: `codebook_size: 128`, `latent_dim: 32` | ✅ **Match** |

### New findings (refinements to capture)

1. **`aggregator_type` flag — gate vs mean ablation.** The code supports two aggregation modes:
   - `"gate"` (default): sigmoid relevance gate produces `α_k`, then normalized weighted average.
   - `"mean"`: `α_k = 1` for all active factors (uniform weighted average, no learned gate).
   This is the natural ablation for testing whether the sigmoid gate adds value over uniform aggregation. **Plan 375 should support both modes** as a config flag — the `"mean"` mode is the G2 ablation baseline.

2. **FiLM conditioning is pervasive.** The code uses Feature-wise Linear Modulation `(1 + γ) * x + β` at every layer of every module:
   - State encoder: FiLM on occupancy weights at every encoder block.
   - Occupancy encoder: FiLM on global state at every conv layer.
   - Factor embedding: FiLM on `[global_state, occupancy_embedding]`.
   - Gate network: FiLM on `[global_state, occupancy_embedding]` at every hidden layer (4 linear layers, not 2).
   - Forward decoder: FiLM on `z_action` (and optionally state features) at every decoder block, with two modes (`"z_action"` vs `"z_action_and_state"`).
   The modelless version can use a simplified FiLM: `r_k = (1 + γ_k) * c(k) + β_k` where `γ_k, β_k` are derived from `dot(state, projection_k)`.

3. **Factor token construction is richer than concatenation.** The `FactorEmbedding` module concatenates `[codebook_vector, weight, occupancy_embedding]` AND FiLM-conditions on `[global_state, occupancy_embedding]`. The occupancy embedding is itself a CNN/MLP-encoded spatial map, not just a scalar. My note's factor token description was simplified; the code is more elaborate.

4. **Decoder FiLM modes.** The forward decoder supports two FiLM modes:
   - `"z_action"`: channel-wise FiLM from z_action alone (lighter).
   - `"z_action_and_state"`: spatially-varying FiLM from z_action AND state features (heavier, default in the class but config defaults to `"z_action"`).

### Impact on Plan 375

- **Add `aggregator_type` config** (`"gate"` | `"mean"`) to the primitive. The `"mean"` mode is the G2 ablation.
- **Use simplified FiLM** in the modelless factor token: `r_k = (1 + γ_k) * c(k) + β_k` where `γ_k = dot(state, g_proj_k)`, `β_k = dot(state, b_proj_k)`.
- **Update GOAT gate hyperparameters** to match the paper's defaults: K=128, D=32 (not K=32, D=8).
- **Default motion input is acceleration** (second-order temporal derivative), which is already shipped as Plan 277 (Temporal Deriv Kernel, DEFAULT-ON).

**Verdict unchanged: GOAT.** The code verification confirms the factorized mechanism is exactly as distilled. The refinements (aggregator_type flag, FiLM pervasiveness, K=128/D=32 defaults) are implementation details that enrich Plan 375 but do not change the novelty assessment or the modelless unblock path.
