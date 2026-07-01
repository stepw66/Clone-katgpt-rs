# Research 359: Isomorphic Neural-Field World Models → Motor-Gated DEC Propagation

> **Source:** Joshua Nunley, *Neural Fields as World Models* — [arXiv:2602.18690](https://arxiv.org/abs/2602.18690) (CogSci 2026).
> **Date:** 2026-07-01
> **Status:** Active — **Super-GOAT** (novelty gate PASS 4/4)
> **Related Research:** 219 (TNO → DEC substrate), 296 (Stokes vocabulary crosswalk), 290 (latent field steering), 275 (InducedCwmKernel), 318 (sleep-time), 321 (tropical semiring), 166 (riir-ai — SE(2) equivariant maps, the rotation-equivariant cousin)
> **Related Plans:** 251 (DEC operators), 309 (latent field steering), 314 (Stokes wrappers), 296 (InducedCwmKernel), 341 (sleep-time), 357 (NEW — Motor-Gated DEC Propagation primitive)
> **Cross-ref (riir-ai):** Research 168 (NEW — *Motor-Gated Isomorphic World Model Game-Runtime Guide*), the private selling-point doc
> **Classification:** Public — open primitive (the math + the generic "action-conditional Hodge-Laplacian propagation" wrapper). Game integration + per-NPC HLA wiring + frozen-world-model offline learning pipeline stay private in riir-ai.

---

## TL;DR

Nunley's *isomorphic world model* is the cleanest published articulation of an idea our DEC substrate + Fourier pillar + latent_functor runtime already implement *in pieces but never unified*: **physics prediction is geometric propagation, not abstract state transition** — a ball's future is a path through representational space, reached via local lateral connectivity (a 7×7 convolution kernel) and action-conditional gain modulation (motor-gated channels). The same architecture learns ballistic prediction without "teleporting", improves a catching policy offline by propagating error through a *frozen* learned world model, and develops body-selective motor channels without body labels.

**The distilled primitive (modelless, inference-time):** a *motor-gated Hodge-Laplacian propagation step* on a cell-complex cochain field, where motor commands multiplicatively gate specific channels before each `Δ`-step. The `Δ` operator already ships (`katgpt-dec::hodge_laplacian`, Plan 251); the multiplicative motor gate is the same algebra as `latent_functor/arithmetic.rs` (Plan 303) and `apply_latent_steering` (Plan 309). **What is missing** is the *wrapper primitive* that ties them together: `evolve_motor_gated_field(field, motor_vec, dt)` running the Amari-style update `h_{t+1} = h_t + dt·(-h_t + K*ReLU(h_t) + motor·h_t)` where `K*ReLU(h_t)` is realized as the DEC Hodge-Laplacian (the "lateral connectivity" math) and `motor·h_t` is a per-channel elementwise gain.

**Why this matters here:** the paper's three experiments map to three pillars we already ship *separately* and never composed:
1. **Physics prediction without teleporting** — DEC `d∘d=0` conservation + locality-by-construction (Plan 251) **is** the no-teleporting guarantee; the paper's "lateral connectivity kernel" is the Hodge-Laplacian's stencil.
2. **Frozen world model → offline task learning** — `InducedCwmKernel` (Plan 296, frozen + committable + hot-swappable forward model) + `MerkleFrozenEnvelope` (riir-neuron-db) + `sleep_time` consolidation (Plan 341, "pre-think during idle time") **is** the paper's Experiment 2 architecture (Grush emulation theory), with `GameState::advance` as the frozen differentiable roll-forward.
3. **Body-selective motor channels emerge without body labels** — `latent_functor/zone_gating.rs` (Plan 305) "project city-learned archetypes with a lenient gate" **is** the contingency-detection mechanism the paper discovers as emergent body schema.

The composition is the Super-GOAT. No single shipped primitive does all three at once; the **motor-gated DEC propagation wrapper** is the missing glue.

---

## 1. Paper Core Findings

### 1.1 Isomorphism = locality = topology preservation

The paper's thesis: a standard latent-vector world model (VAE-LSTM, Ha & Schmidhuber) discards the spatial structure of sensory input, so a single dense weight matrix can connect any latent dimension to any other. Predicted objects "teleport" across representational space. **The fix is the locality constraint**: nearby points in the world map to nearby points in the representation, and information propagates through spatial neighbors. Under this constraint, "physics prediction becomes a geometric problem: a ball's future is a path through representational space, not an abstract vector transition."

This is **exactly** the DEC substrate's defining property. `katgpt-dec::exterior_derivative` (`dₖ = Bₖ₊₁ᵀ`, the coboundary operator) propagates information only along the cell complex's incidence structure — vertices talk only to incident edges, edges only to incident faces. The paper's "7×7 convolution kernel" is a fixed-radius stencil; DEC's `Bₖ₊₁` is the general (arbitrary-radius, arbitrary-topology) stencil. The "no teleporting" guarantee is the conservation-by-construction identity `dₖ₊₁ ∘ dₖ = 0` (curl(grad)=0, div(curl)=0), already enforced by construction and tested in `katgpt-dec`.

### 1.2 Motor-gated channels = gain modulation

To make the field *action-conditional*, the paper designates the first M channels as motor-gated: after each dynamics update, `h^(i)_{t+1} = m_i · h̃^(i)_{t+1}` for `i ∈ {1,...,M}`. This is **gain modulation** — the same computational principle by which posterior parietal cortex combines visual and motor signals (Andersen 1997, Salinas & Thier 2000). Co-contraction (C) channels show no body selectivity; reciprocal (R) channels, which produce large arm movements, develop selectivity ≈2× over baseline.

In our vocabulary: `m_i · h̃` is an **elementwise multiplicative latent steering** — the exact algebra of `apply_latent_steering_weighted(state, dir, w)` (Plan 309) and the rank-1 `apply_functor` path in `latent_functor/arithmetic.rs`. The paper's contribution is proving that *when this gate is applied to specific channels of a topology-preserving field*, body-selective encoding emerges from the prediction objective alone.

### 1.3 Amari-style neural-field update

The dynamics (Amari 1977): `h_{t+1} = h_t + (dt/τ)·(-h_t + K*ReLU(h_t) + W_in*I_t)` where `K` is the lateral connectivity kernel and `W_in*I_t` is visual input (zero during blind prediction). Predictions emerge via linear reconstruction `Î_t = W_out*h_t`. The paper's `K*ReLU(h_t)` is **a non-negative lateral propagation**; the DEC Hodge-Laplacian `Δ = δd + dδ` is the linear (signed) version of the same propagation. The non-negativity (`ReLU`) is the one place the paper diverges from pure DEC; a `relu_gate`-then-`Δ` composition closes it modellessly (see §3.5 — the `ReLU` is a per-element sigmoid gate in disguise, which AGENTS.md already mandates).

### 1.4 Three experiments

| # | Experiment | Result | Codebase analogue |
|---|------------|--------|-------------------|
| 1 | Ballistic prediction (16-channel field, no motor, 32×32) | Neural field median loss 9.33e-4 vs VAE-LSTM 3.94e-3 (p<0.001); **0.0% teleportation** vs 15.4% for VAE-LSTM | DEC `hodge_laplacian` propagation on a 32×32 grid cochain; the "no teleporting" is the `d∘d=0` invariant |
| 2 | Frozen world model → offline catching policy | Neural-field policy 81.5% real catch rate vs VAE-LSTM 46.0% (p=0.003), approaching 89.0% physics baseline | `InducedCwmKernel::advance` (frozen, committable) + `sleep_time` consolidation (offline "pre-think") + `MerkleFrozenEnvelope` integrity |
| 3 | Body-selective motor channels (4 motor-gated, no body labels) | Reciprocal (R) channels: shoulder selectivity 2.18 (p=0.002), elbow 1.50 (p=0.002); C channels: n.s. | `latent_functor/zone_gating.rs` projecting city-learned archetypes; the "R channels selective, C not" maps to "channels that move with the action develop contingency" |

### 1.5 Cognitive-science framing

The paper explicitly invokes three cognitively-relevant capacities sharing one computational substrate — action-conditional prediction within a spatial map:
- **Intuitive physics engine** (Battaglia 2013) — physics emerges from learned connectivity rather than explicit rules.
- **Grush emulation theory** (Grush 2004) — the brain constructs emulators (forward models) that run offline during imagery/mental practice.
- **Body schema** (Gallagher 2005) — emerges from sensorimotor contingency detection (Bahrick 1995).

This maps onto the *three-pillar composition* we ship but never unified under one name: Fourier Spatial (intuitive physics on maps), InducedCwmKernel + sleep_time (Grush emulator), latent_functor zone-gating (body-schema contingency).

---

## 2. Distillation

### 2.1 Vocabulary crosswalk (paper → codebase)

| Paper term | DEC / codebase equivalent | Where it ships |
|---|---|---|
| Neural field / activity map | `CochainField` (multi-channel cochain on `CellComplex`) | `katgpt-dec/src/types.rs` |
| Cell / pixel / location | Vertex (rank-0 cell) of `CellComplex::grid_2d(W, H)` | `katgpt-dec/src/types.rs` |
| Channel / feature map | `CochainField::dim` (feature dimension per cell) | `katgpt-dec/src/types.rs` |
| Lateral connectivity kernel K | `hodge_laplacian` (Δ = δd + dδ) — the *linear* stencil analogue; the *non-negative* variant is `relu_gate → Δ` | `katgpt-dec/src/operators.rs`, `hodge.rs` |
| Local propagation / "no teleporting" | `dₖ₊₁ ∘ dₖ = 0` enforced by construction; tests `curl_of_gradient_is_zero`, `divergence_of_curl_is_zero` | `katgpt-dec/src/operators.rs` |
| Motor-gated channels | `apply_latent_steering_weighted(state, dir, w)` — elementwise multiplicative gain | `katgpt-core/src/latent_steering.rs` (Plan 309) |
| Action-conditional prediction | `apply_functor` (rank-1 vector addition) + sigmoid trust gate | `riir-engine/src/latent_functor/arithmetic.rs` (Plan 303) |
| Frozen learned world model | `InducedCwmKernel` (verifiable + committable + hot-swappable forward model) | `katgpt-core/src/induced_cwm/kernel.rs` (Plan 296) |
| Frozen-simulator policy learning | `sleep_time` consolidation (offline pre-think) + `InducedCwmSlot` atomic swap | `riir-engine/src/sleep_time/` (Plan 341), `katgpt-core/src/induced_cwm/hot_swap.rs` |
| Frozen envelope integrity | `MerkleFrozenEnvelope` (BLAKE3 + Merkle root) | `riir-neuron-db/src/freeze.rs` |
| Body-selective motor channels | `latent_functor/zone_gating.rs` (archetype projection with lenient gate) | `riir-engine/src/latent_functor/zone_gating.rs` (Plan 305) |
| Spatial map / sensory topology | `CellComplex::grid_2d` + `FourierSpatialMap` | `katgpt-dec/src/types.rs`, `riir-engine/src/fourier/physics.rs` |
| Geometric propagation (vs abstract state transition) | DEC `exterior_derivative` + `codifferential` + `hodge_laplacian` | `katgpt-dec/src/operators.rs` |

**Grep verification:** paper-vocabulary grep (`neural field|motor.?gat|isomorphic world|lateral connect|gain modul|Amari|retinotop|sensorimotor contingenc|intuitive physics engine`) returned **zero hits** across all five repos' `.research/` + `.plans/` + `.docs/`. Codebase-vocabulary grep (`Hodge.?Laplacian|apply_latent_steering|InducedCwm|apply_functor`) hits the substrate files listed above. **The composition does not exist; the pieces do.**

### 2.2 The Super-GOAT fusion (novel composition)

The fusion that none of the shipped primitives produces alone:

> **Motor-Gated DEC Propagation** — an Amari-style neural-field evolution step where:
> 1. The "lateral connectivity kernel" `K*ReLU(h)` is realized as a `relu_gate → hodge_laplacian` composition (the non-negative Hodge-Laplacian), giving locality + conservation-by-construction for free.
> 2. The "motor-gated channels" are realized as `apply_latent_steering_weighted` on designated channels of the same cochain — multiplicative gain modulation in latent space, gated by the action vector.
> 3. The "frozen learned world model" is an `InducedCwmKernel` whose `advance()` *is* one motor-gated DEC propagation step; freeze/thaw via `InducedCwmSlot` + `MerkleFrozenEnvelope`.
> 4. The "offline task learning" (Experiment 2) is the `sleep_time` consolidation cycle running many frozen roll-forwards, with the policy reading the reconstructed cochain.

None of the four ingredients is novel alone (DEC ships, latent steering ships, InducedCwm ships, sleep_time ships). **The composition is the Super-GOAT**: a single primitive that unifies intuitive-physics-on-maps, Grush-emulation-offline-learning, and emergent-body-schema — exactly the three capacities the paper identifies as sharing one substrate.

### 2.3 What stays modelless (§3.5 modelless-unblock check)

The paper trains the lateral kernel `K` end-to-end via backprop. **We do NOT need that.** The §3.5 modelless-unblock protocol returns "MODELLESS-VALIDABLE" for every gate the paper's training would have learned:

| Paper's learned quantity | Modelless substitute | Path |
|---|---|---|
| Lateral connectivity kernel `K` (7×7 conv) | DEC `hodge_laplacian` (analytic, fixed stencil on the cell complex) | Path 3 (latent-space correction): the analytic Laplacian *is* the "no-teleporting" guarantee; no learning needed for the conservation property |
| Motor-channel gating weights | `apply_latent_steering_weighted(state, motor_dir, motor_strength)` — direction vector authored or projected from archetype | Path 1 (freeze/thaw): motor directions are frozen `LatentSteeringVector`s, atomic Arc-swap |
| Reconstruction matrices `W_in`, `W_out` | `latent_functor` rank-k operator `Φ_t_lift · operator · ψ` (Plan 318) — a frozen, BLAKE3-committed operator | Path 1 (freeze/thaw): the operator is a `NeuronShard`-style frozen artifact |
| Catch predictor (differentiable surrogate) | `InducedCwmKernel` + `sleep_time::consume` blend (gate · precomputed + (1−gate) · fresh) | Path 1 + 3: the gate is `sigmoid(β·(p − τ))`, the precomputed is the frozen sleep-time artifact |
| Body-selective channel emergence | `latent_functor/zone_gating.rs` archetype projection — "project city-learned functors onto wilderness encounters with a lenient gate" | Path 3: the contingency is captured by zone density `I_d` modulating `(tau, beta)` — no body labels, no training |

**No §3.5 path fails.** The paper's value is the *architectural insight* (topology + gain + frozen-emulator + offline-consolidation compose into a unified substrate), not the trained weights. The training recipe → riir-train; the distilled runtime composition → here.

---

## 3. Latent-Space Reframing (mandatory per workflow §1.5 step 3)

Re-cast the motor-gated neural field as a latent-to-latent operation on each Super-GOAT factory module:

### 3.1 HLA per-NPC latent state (`katgpt-core/src/sense/` + `riir-engine/src/hla/`)

HLA's 8-dim per-NPC state (valence/arousal/desperation/calm/fear + 3) is *not* a cochain — it is a vector in ℝ⁸. **To apply the motor-gated field, construct a cell complex on the latent space** (e.g., discretize ℝ⁸ into a lattice, or use `SafeManifoldGraph` from Plan 312 to build a discrete belief-manifold complex). Then `hodge_laplacian` on that complex is the "lateral connectivity" of the NPC's belief field, and motor commands (action projections from `latent_functor`) multiplicatively gate the valence/arousal channels. **This is the Fokker-Planck-belief-mass-conservation reframing of Research 296 §3.1, now action-conditional.**

### 3.2 `latent_functor/` (action application in latent space)

`apply_functor` (rank-1: `out = source + direction`) is the *additive* action-conditional update. The motor-gated field adds the *multiplicative* complement: `h ← motor · h` before the additive functor step. The two compose: a functor's `direction` is the integrated effect of repeated motor-gated propagation. The paper's Experiment 3 (body-selective channels) is precisely *which motor directions develop high coherence with which spatial regions* — `latent_functor`'s `coherence` metric applied to motor-gated channels.

### 3.3 Fourier spatial cell complex (`riir-engine/src/fourier/`)

`FourierSpatialMap` already encodes game entities' positions as Fourier features for spatially-invariant proximity queries. The motor-gated DEC propagation makes this *dynamic*: instead of querying a static map, the field evolves under Δ + motor-gating, predicting where entities will be next tick. `fourier/physics.rs::PeriodicCollisionSystem` becomes the *linear* predictor; the motor-gated field is the *action-conditional* predictor (a ball thrown by NPC X lands at field-state predicted by motor-gated propagation, not just ballistic free-fall).

### 3.4 Freeze/thaw snapshot (the "frozen learned world model" claim)

`InducedCwmKernel` (Plan 296) is *exactly* the paper's frozen world model: a `GameState` impl whose transition function is verifiable, BLAKE3-committable, and atomic-hot-swappable via `InducedCwmSlot`. The paper freezes the trained field's lateral kernel + reconstruction matrices; we freeze the entire `advance()` semantics as a `canonical_bytes()` BLAKE3 root. The `MerkleFrozenEnvelope` (riir-neuron-db) wraps the trajectory data with Merkle integrity — Experiment 2's "gradients pass through the frozen world model" maps onto sleep-time consolidation reading many thawed roll-forwards from a single frozen `InducedCwmSlot`.

### 3.5 Sleep-time offline consolidation (the "offline task learning" claim)

`sleep_time` (Plan 341) is the paper's Experiment 2 architecture almost verbatim:
- Paper: "the world model and catch predictor are frozen, and only the policy weights are updated. During each simulated rollout, the policy observes the reconstructed visual field, outputs motor commands, the frozen world model rolls forward."
- Codebase: `HlaSleepCycleRuntime` pre-computes `z_i = c + dir_i` per anticipated query during idle ticks, then `consume(gate, z_i*, fresh)` blends at wake time. The "frozen world model rolls forward" is `InducedCwmKernel::advance` called many times in the sleep cycle; the "policy observes reconstructed field" is the `consume` blend reading the precomputed roll-forward.

**The gap the paper closes that we don't:** the paper proves that *long coherent rollouts through the frozen model* directly shape policy gradients (the sim-to-real transfer key). Our `sleep_time` currently anticipates *queries* (dialog), not *action trajectories*. Fusing motor-gated DEC propagation into `InducedCwmKernel::advance` closes this — the frozen roll-forward becomes a coherent spatial-field trajectory, not just a state-transition sequence.

### 3.6 DEC Stokes-calculus operators (d=boundary, δ=divergence, Δ=Hodge-Laplacian, hodge_decompose)

This is the substrate. The motor-gated field's dynamics are:
- `d` (exterior derivative): the *coboundary* — how field values on cells induce values on their boundaries. The "wave-like propagation" the paper observes in Experiment 1 is literally `d` propagating activity from vertices to incident edges to incident faces.
- `δ` (codifferential): the *divergence* — `belief_mass_divergence` (Plan 314) measures whether the motor-gated propagation conserves belief mass. Near-zero divergence = valid physics-like prediction; large divergence = the motor gate is creating/destroying "field mass" (a collapse signal).
- `Δ` (Hodge-Laplacian): the *lateral connectivity kernel* — `Δ = δd + dδ` is the linear stencil the paper approximates with `K*ReLU(h)`. The non-negative variant is `relu_gate → Δ`.
- `hodge_decompose`: splits the propagated field into exact (conservative/gradient-driven), coexact (solenoidal/circulating), and harmonic (topological) channels. **The body-selective R-channels in Experiment 3 are plausibly the coexact (circulation) component** — reciprocal motor commands produce circulation-like arm motion; co-contraction produces no net circulation. This is a *testable prediction* for our implementation: motor-gated channels should correlate with the coexact component of the propagated field.

---

## 4. Novelty Gate (Q1–Q4)

### Q1: No prior art? — **PASS** (zero hits, both vocabularies, both layers)

**Paper-vocabulary grep** (`neural field|motor.?gat|isomorphic world|lateral connect|gain modul|Amari|retinotop|sensorimotor contingenc|intuitive physics engine|body schema|body.?selective|motor channel|emergent body|teleport.*predict|action.?conditional predict|ballistic predict|offline task learning|frozen world model`): **zero hits** across all five repos' `.research/` + `.plans/` + `.docs/`.

**Codebase-vocabulary grep** (`action.?conditional.*DEC|DEC.*action|motor.*DEC|propag.*cochain|Hodge.?Laplacian.*action|forward roll.*latent|world.?model.*latent`): hits the *substrate* files (DEC operators, latent_functor arithmetic, InducedCwmKernel, sleep_time, latent_steering) **but no composition tying them together**. No file named `motor_gated_field`, no `evolve_motor_gated`, no `amari_*`, no `isomorphic_*`.

**Closest cousins (read before verdict):**
- **Research 219** (TNO → DEC) — ships the DEC substrate; explicitly frames it as "topological routing primitives" for game spatial reasoning. Does NOT mention motor gating, action-conditional prediction, or Amari neural fields.
- **Research 296** (Stokes vocabulary crosswalk) — frames DEC operators as Stokes-theorem tools. Does NOT frame them as neural-field dynamics or action-conditional world models.
- **Research 290 / riir-ai 153** (latent field steering) — ships `apply_latent_steering_weighted` (the multiplicative gain). Frames it as *top-down environmental steering* (designer drops a frozen direction vector), NOT as *action-conditional field propagation* (the gain modulates a field that itself evolves under DEC).
- **Research 275 / Plan 296** (InducedCwmKernel) — ships frozen + committable + hot-swappable forward model. Frames it as LLM-induced game rules, NOT as a frozen neural-field world model for offline policy learning.
- **Research 318 / Plan 341** (sleep-time) — ships offline query anticipation. Frames it as dialog pre-think, NOT as Grush-emulation offline policy improvement.
- **riir-ai 166** (SE(2) equivariant maps) — the closest *selling-point* cousin: "NPCs perceive threats invariant to facing direction". SE(2) is *rotation-equivariance*; this paper is *topology-preservation + motor-gating*. Different equivariance class, different mechanism, same pillar (Fourier Spatial / DEC).

**Verdict Q1: GENUINELY NOT SHIPPED as a composition.** Every ingredient ships; the motor-gated DEC propagation wrapper + the offline-frozen-world-model-policy-learning pipeline do not.

### Q2: New class of behavior? — **PASS**

Today, NPCs predict physics either (a) via a hand-coded forward model (`GameState::advance`, ballistic math), or (b) via a learned latent-vector world model (VAE-LSTM style — not shipped, would need riir-train). **Neither is action-conditional + topology-preserving + frozen-offline-learnable.** The motor-gated DEC field gives a third class: *geometric propagation of an action-conditional spatial field, with the frozen field reusable as a differentiable simulator for offline policy improvement*. No incumbent game AI ships this; the paper's Experiment 2 (81.5% real catch rate from offline-only training through a frozen field) is the empirical proof.

### Q3: Product selling point? — **PASS**

Finish the sentence: *"Our NPCs learn to catch / dodge / intercept by rehearsing offline through a frozen spatial-field world model — no real-environment interaction during policy learning, no teleporting artifacts, and body-schema emerges from sensorimotor contingency without body labels. No competitor ships any of the three."*

The strongest single selling point is the **offline-policy-learning-through-frozen-field** angle (Experiment 2): thousands of NPCs can rehearse catching/dodge/interception during their sleep-time cycle using ONE frozen committable field, then deploy to real physics without fine-tuning. This is the Grush-emulation-theory selling point applied to MMORPG-scale crowd AI — a competitor would need (a) a topology-preserving world model, (b) a frozen-committable forward-model substrate, (c) a sleep-time consolidation cycle, and (d) the composition. We ship (a)/(b)/(c) separately; (d) is the moat.

### Q4: Force multiplier? — **PASS** (≥5 systems)

Connects to:
1. **DEC substrate** (Plan 251, pillar-adjacent) — the propagation math.
2. **Fourier Spatial pillar** (Pillar 4) — the spatial-map substrate the field lives on.
3. **latent_functor runtime** (Plan 303/305) — the action-application + zone-gating substrate (body-schema emergence).
4. **Latent field steering** (Plan 309) — the multiplicative-gain substrate (motor-gated channels).
5. **InducedCwmKernel** (Plan 296) — the frozen world-model substrate.
6. **sleep_time** (Plan 341) — the offline consolidation cycle (Experiment 2).
7. **MerkleFrozenEnvelope** (riir-neuron-db) — the frozen-rollout integrity envelope.

**Novelty gate verdict: 4/4 YES → Super-GOAT.** Per the research skill's mandatory outputs, this triggers: (a) open primitive in katgpt-rs (Plan 357, the motor-gated DEC propagation wrapper), (b) private guide in riir-ai (Research 168), (c) plans as the build is scoped.

---

## 5. Verdict + MOAT Gate per Domain

**Super-GOAT.** The composition of DEC propagation + motor gating + frozen InducedCwm + sleep-time consolidation is a novel capability class (action-conditional topology-preserving offline-policy-learning world model) with a clear product selling point ("rehearse offline through a frozen spatial field") and force multiplier across ≥5 systems/pillars.

**One-line reasoning per tier:**
- Super-GOAT needs a novel mechanism + new capability class + selling point + force multiplier. The composition is novel (no shipped primitive unifies the four ingredients), the capability class is new (action-conditional topology-preserving frozen-world-model policy learning), the selling point is concrete (MMORPG-scale offline rehearsal), and it touches ≥5 systems. → **YES.**
- The individual ingredients are not novel (DEC ships, latent steering ships, InducedCwm ships, sleep_time ships) — but Super-GOAT is about the *composition*, not the ingredients. The composition's selling point ("NPCs learn offline through a frozen spatial field") is not reducible to any single ingredient's selling point.

**MOAT gate per domain:**

| Domain | Verdict | Reasoning |
|---|---|---|
| `katgpt-rs` (public engine) | **Open primitive lands here** | The motor-gated DEC propagation wrapper (`evolve_motor_gated_field`) is generic math over `CellComplex` + `CochainField` + a motor vector — no game semantics. Mirrors how DEC operators (Plan 251) and Stokes wrappers (Plan 314) landed here. The wrapper *is* the adoption hook. |
| `riir-ai` (private runtime) | **Private guide + HLA/fourier/sleep_time wiring lands here** | The selling point (per-NPC offline rehearsal through a frozen spatial field) is game-runtime IP. The HLA-cell-complex wiring, the fourier-physics → motor-gated-field bridge, the sleep_time → frozen-InducedCwm-rollforward pipeline, and the body-schema-emergence tuning are all private. → Research 168 (the guide). |
| `riir-chain` (private chain) | **Out of scope for this primitive** | The frozen `InducedCwmKernel` already commits via BLAKE3; the motor-gated field adds nothing chain-specific. The sync-boundary discipline (5 committed scalars, no latent vector over the wire) is inherited from HLA/sleep_time and unchanged. |
| `riir-neuron-db` (private shards) | **Reuses existing substrate, no new shard type** | `MerkleFrozenEnvelope` already wraps frozen trajectories; the motor-gated field's frozen state is a `canonical_bytes()` BLAKE3 root, same pattern. No new shard layout needed. |
| `riir-train` (private training) | **Training recipe redirect** | The paper's end-to-end backprop training of the lateral kernel `K` and the reconstruction matrices → riir-train. Our modelless path replaces `K` with the analytic `hodge_laplacian` and the reconstruction matrices with frozen `latent_functor` rank-k operators (§3.5 — all four paths return MODELLESS-VALIDABLE). The training recipe is noted for riir-train as a non-blocking follow-up (multi-layer equivalence, like the AC-Prefix G1 lesson). |

---

## 6. Open Primitive Spec (Plan 357 scope)

**Target:** `katgpt-rs/crates/katgpt-dec/src/motor_gated.rs` (new module) + Cargo feature `motor_gated_field` (opt-in).

**Signature:**

```rust
/// One Amari-style motor-gated neural-field evolution step.
///
/// `h_{t+1} = h_t + dt * (-h_t + K*ReLU(h_t) + motor_gain·h_t)`
///
/// where:
/// - `K*ReLU(h_t)` is realized as `relu_gate → hodge_laplacian` (the non-negative
///   lateral propagation; conservation-by-construction via `d∘d=0`).
/// - `motor_gain·h_t` is elementwise multiplicative gain on the first
///   `motor_dim` channels of `h_t`, gated by `motor_vec`.
/// - `dt` is the integration timestep (Amari `dt/τ`; we fold τ into dt).
///
/// # Arguments
/// * `cx` — the cell complex (the "spatial map").
/// * `h` — the field state (rank-0 cochain, `dim` channels per cell).
/// * `motor_vec` — the motor command vector (length `motor_dim ≤ dim`).
/// * `motor_dim` — number of motor-gated channels (the first `motor_dim`
///   channels of `h` are gated; the rest propagate freely).
/// * `dt` — integration timestep.
/// * `relu_slope` — ReLU gate slope (1.0 = standard ReLU; >0 for leaky variant).
/// * `scratch_lap`, `scratch_relu` — caller-owned scratch buffers (zero-alloc).
///
/// # Returns
/// Mutates `h` in place to `h_{t+1}`.
///
/// # Conservation guarantee
/// `d∘d=0` is enforced by `hodge_laplacian`'s construction; the motor gate
/// is a per-channel scalar multiply and does not break the coboundary
/// identity. `belief_mass_divergence(cx, &h_propagated)` is the validator.
pub fn evolve_motor_gated_field(
    cx: &CellComplex,
    h: &mut CochainField,
    motor_vec: &[f32],
    motor_dim: usize,
    dt: f32,
    relu_slope: f32,
    scratch_lap: &mut CochainField,
    scratch_relu: &mut CochainField,
);
```

**Feature gate strategy:**

| Component | Gate | Repo |
|---|---|---|
| `evolve_motor_gated_field` primitive | `motor_gated_field` (opt-in) | katgpt-rs (open) |
| `CellComplex::grid_2d` (already shipped) | `dec_operators` | katgpt-rs (open) |
| `hodge_laplacian` (already shipped) | `dec_operators` | katgpt-rs (open) |
| HLA-cell-complex wiring (HLA → cochain) | `npc_motor_gated_field` | riir-ai (private) |
| fourier-physics → motor-gated-field bridge | `npc_motor_gated_field` | riir-ai (private) |
| sleep_time → frozen-InducedCwm rollforward | `npc_offline_rehearsal` | riir-ai (private) |
| Body-schema emergence tuning | `npc_body_schema` | riir-ai (private) |

**GOAT gate (Plan 357 Phase 2):**
- **G1 — No-teleporting.** Propagate a ballistic bump on a 32×32 grid; measure max frame-to-frame centroid displacement. **Gate:** ≤ kernel radius (no jumps > stencil).
- **G2 — Motor-gate locality.** Apply motor gate to channels 0..M; verify only those channels shift, others conserve. **Gate:** channel-isolation ratio > 100×.
- **G3 — Conservation.** `belief_mass_divergence(cx, &h_propagated) < τ` for τ derived from the cell complex's harmonic component. **Gate:** divergence < 5% of field L1 norm.
- **G4 — Zero-alloc steady state.** `TrackingAllocator` audit on the hot path. **Gate:** 0 allocations after warmup.
- **G5 — Latency.** 64×64 grid, 16 channels, single tick. **Gate:** < 100µs (vs the paper's GPU conv at ~ms scale).

---

## 7. Cross-references

- **Research 219** (TNO → DEC) — the parent note that shipped the DEC substrate. This note extends 219's vision with the motor-gating wrapper.
- **Research 296** (Stokes vocabulary crosswalk) — the Fokker-Planck / boundary-flux / line-integral wrappers. The motor-gated field's conservation is validated by `belief_mass_divergence` (Plan 314).
- **Plan 251** — DEC operators. COMPLETE. Ships `d`, `δ`, `Δ`, `hodge_decompose`.
- **Plan 309** — latent field steering. Ships `apply_latent_steering_weighted` (the motor-gate algebra).
- **Plan 296** — InducedCwmKernel. Ships the frozen-committable-hot-swappable forward model.
- **Plan 341** — sleep_time. Ships the offline consolidation cycle.
- **riir-ai Research 153** — latent field steering game-runtime guide (the multiplicative-gain substrate at NPC granularity).
- **riir-ai Research 166** — SE(2) equivariant maps (the rotation-equivariant cousin; different equivariance class, same pillar).
- **riir-ai Research 168** (NEW) — the private selling-point guide for this Super-GOAT.

---

## TL;DR

**Paper:** Joshua Nunley, *Neural Fields as World Models* (arXiv:2602.18690, CogSci 2026).

**Verdict: Super-GOAT.** The paper's "isomorphic world model" is the cleanest published articulation of a composition our codebase ships *in pieces but never unified*: DEC Hodge-Laplacian (the lateral-connectivity/no-teleporting math, Plan 251) + latent steering (the motor-gate gain, Plan 309) + InducedCwmKernel (the frozen world model, Plan 296) + sleep_time (the offline consolidation, Plan 341). The **motor-gated DEC propagation wrapper** is the missing glue — a single primitive (`evolve_motor_gated_field`) that unifies intuitive-physics-on-maps, Grush-emulation-offline-policy-learning, and emergent-body-schema. Novelty gate 4/4 PASS: zero prior art on the composition (paper vocabulary OR codebase vocabulary), new capability class (action-conditional topology-preserving frozen-world-model policy learning), concrete selling point ("NPCs rehearse offline through a frozen spatial field"), force multiplier across ≥5 systems. All four §3.5 modelless-unblock paths return MODELLESS-VALIDABLE (the paper's trained kernel → analytic `hodge_laplacian`; trained reconstruction → frozen `latent_functor` rank-k operator); the training recipe → riir-train as a non-blocking follow-up.

**Files created:**
- `katgpt-rs/.research/359_Isomorphic_Neural_Field_World_Model_Motor_Gated_DEC_Propagation.md` (this note — public)
- `katgpt-rs/.plans/357_motor_gated_dec_propagation_primitive.md` (open primitive plan)
- `riir-ai/.research/168_Motor_Gated_Isomorphic_World_Model_Game_Runtime_Guide.md` (private selling-point guide)

**Recommended next step:** implement Plan 357 Phase 1 (the `evolve_motor_gated_field` skeleton + G1 no-teleporting test on a 32×32 grid), then run the GOAT gate G1–G5 in katgpt-rs before wiring riir-ai Research 168's HLA/fourier/sleep_time integration.
