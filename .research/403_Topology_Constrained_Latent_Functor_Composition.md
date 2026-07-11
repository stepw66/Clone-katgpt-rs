# Research 403: Topology-Constrained Latent Functor Composition — Canvas Schema × Flat FunctorTable × Freeze/Thaw Region Swap

> **Source:** Fusion research note (no single source paper). Lineage: Canvas Engineering (R398, Valdez 2026) × Latent Functor Runtime (R123/Plan 303, Minegishi et al. ICML 2026) × Percepta analytical weight construction (R031/032) × DEC Stokes substrate (R219/Plan 251) × FaithfulnessProbe causal attribution (R244/Plan 278).
> **Date:** 2026-07-10
> **Status:** Active
> **Related Research:** 398 (Canvas Engineering — the topology compiler), 123 (Latent Functor Runtime — the flat FunctorTable this fuses over), 031/032 (Percepta — analytical weight construction for deterministic regions), 219 (DEC substrate), 244 (FaithfulnessProbe — the attribution tool), 229 (ProgramAsWeights — the spec→compile axis), 311 (Analytic Lattice — closest cousin), 396 (region_subspace — DEFAULT-ON region steering)
> **Related Plans:** 419 (canvas_schema compiler — the topology + reachability), 303 (latent functor runtime — the flat table + bridges), 416 (region_subspace_steering — DEFAULT-ON), 278 (FaithfulnessProbe — already wired in riir-poc), 251 (DEC operators), 426 (Steering × Geometry Cookbook — the flagship showcase)
> **Classification:** Public

---

## TL;DR

R398 (Canvas Engineering) shipped a `CanvasSchema` compiler + reachability guarantee as an opt-in correctness primitive, parked because its behavioral gain was unproven and its headline value is training-dependent (a DiT must be trained within the declared mask). **This note dissolves the training dependency by changing the substrate.** The paper's "needs training" conclusion is an artifact of compiling the schema onto attention-token positions of a diffusion transformer. Our substrate is not a DiT — it is the **flat `FunctorTable`** (R123/Plan 303: papaya HashMap `RelationId → Arc<FunctorEntry>`, composed by hardcoded call order in `npc_integration.rs`). Distilling the canvas onto *that* substrate produces **topology-constrained latent functor composition**: a reachability graph over the existing latent_functor bridges, enforced by sigmoid subspace gating at functor boundaries, with no transformer backbone and no training.

**The clean distillation (the load-bearing reframe):** do NOT distill the canvas as attention masks (that forces the paper's "needs a backbone" trap). Distill it as a **reachability graph over latent_functor bridges**. The reachability guarantee still holds — absent edge `A↛B` ⟹ bridge A's output is structurally barred from bridge B's input (the gate is closed) — but it is enforced at the latent level (sigmoid-bounded subspace gate), not the attention level. No transformer. No training. The paper's attention mask is just one instantiation; ours is another.

**Verdict: GOAT** (structural merit — declared topology + reachability guarantee + isolated region swap + schema-keyed exchange + causal attribution, over the flat FunctorTable that ships today). Not Super-GOAT now: Q2 (new behavioral class) is unproven — the same attribution question Issue 043 botched. The fusion PoC (FaithfulnessProbe-attributed, modelless, reuses the wired `jlens_concept_readout_goat.rs` pattern) is the gate for Super-GOAT re-evaluation. A fusion issue is opened, not a deferred-commitment "candidate."

---

## 1. The idea and its lineage

### 1.1 The gap this fills

`riir-ai/crates/riir-engine/src/latent_functor/mod.rs` (read this session, lines 1-80) confirms the latent functor runtime is a **flat table**:

```
FunctorTable (lock-free papaya HashMap)
  RelationId → Arc<FunctorEntry { direction, coherence, commitment }>
                   │ atomic swap (freeze/thaw)
                   ▼
  npc_integration: predict_stance / rank / emit_kg_triplet / project_predicted_emotions
  (composition order HARDCODED in the tick loop, not declared in a schema)
```

There is **no declared topology between functors**. Any functor can influence any NPC scalar by construction — the tick loop calls them in a fixed order, but nothing structurally prevents cross-contamination. The bridges (`region_subspace_bridge`, `spectral_trajectory_bridge`, `gauge_invariant_bridge`, `ac_prefix_bridge`, ~30 total) are composed by caller code, not by a reachability graph.

Canvas Engineering (R398/Plan 419) ships the missing half: `compile_schema → AttentionMaskSpec + reachability_horizon + can_reach`. But R398 compiled the schema onto **attention-token positions** (the paper's DiT framing), which is why it concluded "needs training." The reframe: compile the schema onto the **latent_functor bridges** instead.

### 1.2 The three spec→compile axes (R229 lineage), and where this lands

| Variant | Spec | Lands in | Modelless-runnable? |
|---|---|---|---|
| Percepta (R031/032) | C program (WASM) | **Weights** (analytical construction, Futamura projection) | ✅ Yes — program is deterministic |
| Canvas as paper (R398) | Region schema + topology | **Attention mask + loss routing** (on a DiT) | ⚠️ Structure yes; backbone must train within mask |
| ProgramAsWeights (R229) | Symbolic spec | **Symbolic constraints** (bitmap) | ✅ Yes — constraints checked, not learned |
| **This note (R403)** | Region schema + topology | **Reachability graph over latent_functor bridges** (sigmoid subspace gates) | ✅ Yes — functors + freeze/thaw, no backbone |

R403 is the modelless-friendly distillation of canvas: it keeps R398's reachability guarantee but lands it on the latent-functor substrate instead of a transformer backbone, dissolving the training dependency.

---

## 2. Distillation (modelless, inference-time)

### 2.1 Training-dependent vs modelless split (the honest cut — and why R398's split no longer applies)

R398 §2.1 concluded the modelless value is "the compiler + reachability semantics" and everything behavioral needs training, because applying a mask to an untrained DiT backbone is a 19% loss. **That conclusion is substrate-specific.** On the latent-functor substrate:

| Component | Modelless? | Why |
|---|---|---|
| The compiler (`schema → topology → gated functor graph`) | **YES** | Pure structure compilation. |
| Reachability = exact marginal independence | **YES** | Graph-theoretic, by construction (absent edge = closed sigmoid gate = zero contribution). |
| Region content — affect | **YES** | `region_subspace_bridge` (Plan 416, DEFAULT-ON) + HLA dot-projection. Pre-discovered MFA artifact applied modellessly. |
| Region content — geometric/perception | **YES** | DEC `codifferential` on sensory cochain (DEFAULT-ON). |
| Region content — trajectory | **YES** | `spectral_trajectory_bridge` / `heat_kernel_trajectory_linear` (Plan 359). |
| Region content — memory | **YES (at runtime)** | `MerkleFrozenEnvelope` freeze/thaw swap (DEFAULT-ON). The snapshot was trained/consolidated earlier, but the swap is modelless. |
| Region content — deterministic action | **YES** | Percepta analytical weight construction (Futamura projection). |
| Region content — learned action | **YES (at runtime)** | freeze/thaw of a trained policy. Weights came from training; swap is modelless. |
| Behavioral QUALITY of any region | **NO (content)** | The canvas enforces structure; it cannot fix a bad affect direction vector. Quality bottoms out on region content. |

**The training dependency moves from "train the backbone within the topology" (R398, paper framing) to "produce region contents via the appropriate modelless path" (freeze/thaw, Percepta, DEC, steering).** That is an engineering question (pick the right producer per region), not a training question. Consistent with the modelless-first mandate (constraint #1): runtime weight mutations are freeze/thaw + deterministically-constructed LoRA + latent updates — exactly the three allowed paths.

### 2.2 §3.5 modelless-unblock check

R398 deferred the behavioral headline to riir-train after exhausting the three paths against the *paper's DiT substrate*. On the latent-functor substrate, re-check:

1. **Freeze/thaw** — can a frozen snapshot recover region content? **YES** — this is literally how the memory region works (`MerkleFrozenEnvelope`). Region content IS a frozen snapshot.
2. **Raw/lora hot-swap** — can a deterministically constructed adapter enforce the topology? **YES (as a gate, not a weight)** — the topology is a sigmoid subspace gate at each functor boundary (deterministically constructed from the schema's edge list). Not a LoRA on weights; a gate on latent flow. Modelless.
3. **Latent-space correction** — can a dot-product + sigmoid gate enforce region structure? **YES** — this is the gate itself. `region_subspace_bridge` already does this for one region; the canvas generalizes it to a declared topology of N regions.

**Conclusion: no riir-train deferral for the structural primitive.** The only thing that genuinely needs riir-train (or a modelless PoC with a non-designer-tuned producer) is proving the gated topology produces *better behavior* than free composition. That is the §3.6 PoC question, tracked as an issue, not a blocker.

### 2.3 Vocabulary translation (paper ↔ codebase) — MANDATORY before novelty claim

| Paper / R398 term | Codebase-equivalent (R403) | Verified shipped? |
|---|---|---|
| "canvas" / structured latent space | **gated functor graph** over `FunctorTable` | pieces ship; the gate graph does not |
| "region" / `RegionSpec` | a **latent_functor bridge** (region_subspace, spectral_trajectory, ...) gated as a region | **YES (bridges)**; no region-as-graph-node wrapper |
| "topology" / `Connection` | a **reachability edge** in the gated functor graph | **NO** — `latent_functor/mod.rs` is a flat table |
| "attention mask" | **sigmoid subspace gate** at functor boundary (NOT an attention mask) | pieces ship; the topology-of-gates does not |
| "reachability" / d-separation | `canvas_schema::can_reach` / `TransitiveClosure` (Plan 419) applied to functor graph | **YES (Plan 419, opt-in)**; not applied to functors |
| "compile_schema" | compile schema → gated functor graph (gate open/closed per edge) | **NO** — ships for attention masks (Plan 419), not functors |
| "coarse-grained bottleneck" | `CommittedFieldBlend` / StillPerceiver (per-region summary) | **YES (pieces)** |
| "schema-mediated latent exchange" | freeze/thaw a single region's `FunctorEntry` (BLAKE3-committed) | **NO** — freeze/thaw ships; schema-keyed per-region swap does not |

### 2.4 Prior-art surface — what already ships (verified grep + read this session)

1. **`riir-ai/crates/riir-engine/src/latent_functor/mod.rs`** — flat `FunctorTable` (papaya HashMap), coherence-driven re-estimation scheduler, `npc_integration` composes by hardcoded call order. **No declared topology.** (R123/Plan 303)
2. **`latent_functor/region_subspace_bridge.rs`** (Issue 424) — zone-conditioned two-mode NPC steering. **This is the affect-region producer.**
3. **`latent_functor/spectral_trajectory_bridge.rs`** — trajectory-region producer.
4. **`katgpt-core canvas/`** (Plan 419, opt-in) — `compile_schema`, `can_reach`, `TransitiveClosure`. **The topology/reachability primitive.**
5. **`katgpt-core faithfulness/probe.rs`** (Plan 278, opt-in, **wired in riir-poc**) — causal-intervention attribution. `DefaultFaithfulnessProbe`, `is_faithfully_used`. **The attribution tool.** Precedent: `riir-poc/benches/jlens_concept_readout_goat.rs`.
6. **`riir-neuron-db/src/freeze.rs`** — `MerkleFrozenEnvelope`. **The region-swap primitive.**
7. **Percepta** (R031/032, `katgpt-percepta` crate) — analytical weight construction for deterministic regions.
8. **DEC** (`katgpt-core/src/dec/`, DEFAULT-ON) — geometric region producer (`codifferential`, `heat_kernel_trajectory`).

**The gap (Q1):** no `compile_schema → gated functor graph` primitive; no `can_reach` applied to functor-to-functor edges; no schema-keyed per-region `FunctorEntry` freeze/thaw. These three are genuinely missing. The pieces all ship; the topology-over-functors wiring does not.

### 2.5 Fusion — what novel combination does this enable?

**Fusion (novelty TBD — needs PoC before Super-GOAT, tracked in issue):**

> Canvas topology (Plan 419) × flat FunctorTable (Plan 303) × region producer bridges (416/359/DEC/Percepta) × freeze/thaw region swap (neuron-db) × FaithfulnessProbe attribution (Plan 278) → "A gated NPC cognitive-stack graph where each region is a latent_functor bridge, the canvas schema declares which bridges may influence which (sigmoid gate open/closed per edge), regions swap independently via freeze/thaw, and a FaithfulnessProbe per region attributes behavioral deltas to specific gated paths."

**What's novel over the flat FunctorTable (the structural value-add):**

| Claim | latent_functor (today) | + canvas topology (R403) |
|---|---|---|
| Composition order | hardcoded in tick loop | **declared in schema**, swappable without code change |
| Reachability guarantee | none — any functor → any scalar | **absent edge = exact marginal independence** (closed sigmoid gate) |
| Isolated region swap | flat table; swap can cross-contaminate | topology decouples; freeze/thaw one region, others unaffected |
| Schema-keyed exchange | functors exchanged by RelationId | exchange by **schema region** (two NPCs, same schema, swap region content) |
| Causal attribution | manual tick-loop tracing | **FaithfulnessProbe per region** (empty a region's output, measure downstream delta) |

**The behavioral question (unchanged from Issue 043, but now tractable):** does gating the topology (perception→affect→action, no shortcut) beat free composition? On the DiT substrate this needed riir-train. On the functor substrate it is a **modelless PoC**: assemble the gated graph, run in riir-poc toy domain, FaithfulnessProbe-attributed. Reuses the `jlens_concept_readout_goat.rs` pattern already wired.

### 2.6 Compute-unit translation (R368 lesson — does NOT trigger here)

No "N LLM calls/step" structure. Compute unit is "one gated functor application over NPC latent state." No false-PASS risk.

---

## 3. Verdict — GOAT

**Tiers:** Super-GOAT > GOAT > Gain > Pass (per research skill).

**Verdict: GOAT.**

**One-line reasoning:** The gated-functor-graph is a novel modelless primitive (reachability graph over the flat FunctorTable, zero gradient descent) with a provable structural property (absent edge = exact marginal independence by construction) that unifies ≥5 existing pillars (latent_functor, canvas, freeze/thaw, DEC, Percepta, FaithfulnessProbe); but (a) the constituent sub-primitives ship, (b) Q2 (new behavioral class) is unproven — the same attribution question Issue 043 botched, (c) the reachability semantics is the same elegant reframing R398 already noted. It is a **structural unification** (turning the flat table into a decoupled, auditable, swap-isolated graph), not a new behavioral capability at the modelless level.

### 3.1 Novelty gate (Q1–Q4)

- **Q1 (No prior art for the gated functor graph?): YES.** `latent_functor/mod.rs` is flat; `canvas_schema` compiles to attention masks, not functor gates; no schema-keyed per-region freeze/thaw. Pieces ship; the topology-over-functors wiring does not.
- **Q2 (New class of behavior?): PARTIAL → NO at the modelless level (same as R398).** The graph enables declared topology + isolated swap + attribution, which are structural. Whether gating produces *better* NPC behavior than free composition is unproven — and the modelless PoC (FaithfulnessProbe) is the gate, not architectural reasoning.
- **Q3 (Product selling point?): YES (structural), PARTIAL (behavioral).** "Our NPC cognitive stack has a declared causal topology with reachability guarantees, isolated region swap, and schema-keyed exchange" is a real architectural selling point. "And it improves behavior modellessly" is unproven.
- **Q4 (Force multiplier?): YES.** latent_functor + canvas + freeze/thaw + DEC + Percepta + FaithfulnessProbe. ≥5 systems.

**Q2 fails at the modelless level → not Super-GOAT now.** Per the research skill's no-"candidate"-escape-hatch rule: this is a GOAT with a tracked fusion PoC, not a deferred Super-GOAT. A fusion issue is opened.

### 3.2 MOAT gate per domain

- **katgpt-rs (public engine):** the `compile_schema → gated functor graph` wrapper is generic (graph over typed nodes, no game IP). But it consumes `latent_functor` (private, riir-ai). **The public part is thin** — the reachability primitive already ships (Plan 419). The novel wiring (schema → functor gates) is runtime composition → **riir-ai**. katgpt-rs gets at most a generic "topology-over-typed-nodes" graph wrapper; riir-ai gets the gated NPC cognitive stack.
- **riir-ai (private runtime):** the fusion is **pillar-level if the PoC passes**. Until then, deferred — guide is NOT created now (GOAT verdict, not Super-GOAT).
- **riir-chain / riir-neuron-db:** schema-keyed region swap touches `MerkleFrozenEnvelope` (neuron-db), but the primary value is the gated graph (riir-ai). No reroute.

### 3.3 §3.6 defend-wrong PoC — RUN, RESULT: FAIL (NO Super-GOAT)

The PoC ran on 2026-07-10 (Plan 428 Phase 2). Three competitors (Floor / Free composition / Gated graph) tested via FaithfulnessProbe causal intervention on affect→action.

**Result:** Free composition has LARGER non-empty deltas than Gated (opposite of the Super-GOAT hypothesis). Both variants pass `is_faithfully_used` — the probe does not discriminate. The delta asymmetry is a baseline-construction artifact (Free's baseline includes perception, bilinear action amplifies perturbation). The gate's value is structural (reachability guarantee + isolated swap + zero latency overhead), not behavioral.

**R403 stays GOAT.** Phase 3 (Super-GOAT promotion) does NOT execute; `canvas_functor_gate` stays opt-in.

See: `riir-ai/.benchmarks/428_canvas_functor_poc.md` for raw `FaithfulnessProfile` numbers.

The PoC used `FaithfulnessProbe` (per R398 §7 lesson — the ad-hoc flee metric was the wrong tool; causal intervention is the right one) and shipped-bridge producers (per Issue 043 lesson — no designer-tuned classifier).

### 3.4 Probe→drift reward fusion — INVESTIGATED, MATHEMATICAL NO-OP (Plan 429)

The orthogonal follow-up (wire `FaithfulnessProbe` as the reward signal for
`DriftGate::tick`) was investigated in Plan 429 and found to be a mathematical
no-op **by construction**, before any code was written.

The drift kernel (`katgpt-personality/src/kernel.rs:210`) computes:
`Δw_i = alpha × (r_observed − r_expected_i) × Σ_j recent_direction_i[j]`.
The `sum(recent_direction)` term **is already a direction-content gate** —
dead writes with zero cognitive directions produce `Δw = 0` automatically. The
FaithfulnessProbe's Empty intervention is structurally uninformative here
(zeroing directions → zero drift → `empty_delta = 0` always), so the proposed
faithfulness multiplier is always > 1.0 (amplify only, never dampen).

**Positive interpretation:** the drift kernel is already robust to the
reward-hacking scenario the fusion was meant to address. The viable alternative
(action-level probe via the integrity layer `AuditRunner`, Plan 308) probes the
right thing (NPC action vs memory) and would be a wiring task, not a new
primitive.

See: `riir-ai/.benchmarks/429_probe_drift_fusion.md` for the full proof.

### 3.5 Action-level faithfulness→drift modulation — SHIPPED, G1 PASS (Plan 430)

The viable alternative identified in §3.4 (wire `AuditRunner`'s
`input_faithful_rate` → scalar multiplier on `DriftGate`'s reward) was
implemented in Plan 430 and **PASSED the decisive PoC**.

Unlike the drift-kernel-level probe (§3.4, no-op), the action-level rate is a
**strictly stronger check**: it tests whether the NPC's *behavior* (not the
drift kernel's directions) depends on injected memory. An NPC with legitimate
cognitive directions whose actions don't bind to memory is the real
 dead-injection failure mode — the drift-kernel probe sees "faithful" (nonzero
Δw), but the action-level rate → 0 → reward dampened.

**G1 result (the decisive PoC):** two NPCs, identical setup. Without
modulation: divergence = 0.000 (identical drift). With modulation: divergence
= 0.954 (NPC A drifts, NPC B frozen). Separation ratio = ∞.

The multiplier primitive (`faithfulness_reward_multiplier`), integration
contract (`FaithfulnessRateSource` trait + `AuditRunnerRateSource`), and PoC
are shipped behind `action_faithfulness_drift` (opt-in, default-off).
**Phase 4 T4.1 (production wiring) shipped 2026-07-10**: `AuditRunnerRateSource`
is now wired into `cognitive_branch.rs` at the DriftGate call site. The
`AuditRunner` is stored on `MapInstance` (all 5 construction sites). When empty
(no probes have run), rate=1.0 → multiplier=1.0 → bit-identical reward.
**Phase 4 T4.2 (production probe scheduling) shipped 2026-07-10**: new module
`riir-engine/src/integrity/composition_probe.rs` defines the production
consumer (`CompositionActivationConsumer`) — behavior =
`Σ_i w_i · Σ_j direction_{i,j}` (the NPC's cognitive activation). The probe is
scheduled at audit cadence (every N=64 ticks) in `cognitive_branch.rs`, ahead
of the drift phase. Dead injections (zero-weight compositions) →
`input_faithful_rate` drops → drift reward dampened. 11 unit tests + 1
production integration test all PASS.

**Plan 431 (production validation + promotion, 2026-07-10): PROMOTED to
DEFAULT-ON.** A production validation test ran the real map-tick loop (default
town map, 10 NPCs, 130 ticks, audit cadence at ticks 64/128) with a
persistently dead-injected NPC (`w` re-zeroed each tick — the real-world
failure mode of a shard that consistently fails to bind). **Maximal separation:
dead rate 0.0000 vs faithful 1.0000; dead multiplier 0.0000 vs faithful 1.0000;
zero false positives on a healthy population (min rate 1.0000).** The production
audit path (real `CompositionActivationConsumer` from `compositions[i].w` → real
`record_audit` → real `AuditRunnerRateSource` → real `fused_reward`) detects
dead-injection and dampens reward exactly as designed, with no false-positives.
`action_faithfulness_drift` is now DEFAULT-ON in both `riir-engine` and
`riir-games`. The only remaining piece (organic dead-injection in real gameplay)
is a production-telemetry question, not a mechanism question.

See: `riir-ai/.benchmarks/430_action_faithfulness_drift.md` (mechanism GOAT) and
`riir-ai/.benchmarks/431_action_faithfulness_drift_production_validation.md`
(production validation + promotion) for full results.

---

## 4. Distilled primitive — what ships

The public primitive is thin (reachability already ships in Plan 419). The novel work is the riir-ai wiring: schema → gated functor graph. Sketch:

```rust
// riir-engine/src/latent_functor/canvas_gate.rs (new, gated canvas_functor)
//
// A gated edge in the functor graph. Absent edge = closed gate = exact
// marginal independence (the sigmoid gate saturates to 0 contribution).
pub struct GatedFunctorEdge {
    pub src: RegionId,        // which bridge produces
    pub dst: RegionId,        // which bridge consumes
    pub gate: f32,            // sigmoid(gate_logit); 0 = closed (absent edge)
}

// Compile a CanvasSchema into a gated functor graph over a FunctorTable.
// Reuses canvas_schema::compile_schema for the topology, then lowers each
// Connection to a GatedFunctorEdge binding two latent_functor bridges.
pub fn compile_functor_graph(
    schema: &CanvasSchema,
    table: &FunctorTable,
) -> GatedFunctorGraph { ... }

// The reachability guarantee: if no path src→dst in the compiled graph,
// dst's output cannot depend on src's output (all gates on every path are 0).
// Reuses canvas_schema::TransitiveClosure (zero-alloc O(1) hot path).
pub fn can_influence(graph: &GatedFunctorGraph, src: RegionId, dst: RegionId) -> bool {
    graph.closure.reaches(src, dst)
}
```

**Consumers:** the gated graph replaces the hardcoded call order in `npc_integration.rs`. `FaithfulnessProbe` runs per region (empty a region's `FunctorEntry` output, measure downstream NPC-scalar delta — the `jlens_concept_readout_goat.rs` pattern). Region swap = freeze/thaw one `FunctorEntry` (BLAKE3-committed) without touching others (the topology decouples them).

**What does NOT ship here:**
- A new reachability primitive (Plan 419 ships it; this reuses it on a different substrate).
- New region producers (the bridges already ship; this wires them under a topology).
- Behavioral-quality proof (the PoC follow-up).

---

## 5. Risks and honest caveats

1. **The behavioral gain is unproven (same as R398/Issue 043).** Gating the topology enforces structure; it does not guarantee better behavior. The PoC (FaithfulnessProbe-attributed) is the gate. Do NOT claim modelless behavioral parity until the PoC runs.
2. **The PoC must NOT repeat Issue 043's mistakes.** (a) No designer-tuned classifier — the region producers must be the shipped bridges (region_subspace, DEC, trajectory), not a hand-tuned discriminator. (b) No noisy correlational flee metric — use `FaithfulnessProbe` causal intervention (empty/shuffle/corrupt the affect region's `FunctorEntry`, measure the action region's delta). (c) Report `is_faithfully_used(threshold)` for each region in both gated and free-composition variants.
3. **The level-mapping is the real engineering work.** latent_functor bridges emit heterogeneous representations (HLA 8-dim scalars, direction vectors, cochain fields, weight snapshots). The gate sits between them. A bridge function (raw↔latent, per AGENTS.md) is needed at every region boundary. A bad bridge silently corrupts the signal — the `Irrelevant` intervention catches it.
4. **Percepta fills only the deterministic-region slot.** It cannot analytically construct weights for "feel fear" — that's learned (region_subspace + HLA). Percepta is one region producer among several.
5. **The reachability guarantee requires binary gates (open/closed).** Soft gates (sigmoid not saturated) give *approximate* independence, not exact — same as R398's binary-mask caveat. For the guarantee to hold strictly, the gate must be saturated (0 or 1); for soft routing, it's a strong prior, not a proof.
6. **R398's canvas_schema is opt-in with zero consumers.** This fusion would be its first runtime consumer (the gated functor graph). That's a point in favor of pursuing it — but it means the fusion carries canvas_schema's "no production validation yet" risk.

---

## 6. Plan

→ Fusion PoC issue (tracked, not a plan yet — the PoC decides GOAT→Super-GOAT before a plan is warranted):

**Plan:** `riir-ai/.plans/428_canvas_functor_gate.md` (supersedes the deleted katgpt-rs/.issues/122 — folded in as Phase 2; the issue was misfiled in katgpt-rs when the work is riir-ai, and issues are a deletable vessel for a Super-GOAT gate this important). Phase 1 = structural primitive (opt-in `canvas_functor_gate`, ships on GOAT merits regardless). Phase 2 = the PoC: three competitors (floor / free composition / gated graph), `FaithfulnessProfile` per region via `DefaultFaithfulnessProbe` (reusing the `jlens_concept_readout_goat.rs` pattern), shipped-bridge producers only (NOT a designer classifier — the Issue 043 lesson). If the gated graph's `Empty`-intervention deltas are larger on affect-dependent regions (gate load-bearing) AND free-composition's are smaller (redundancy) → Phase 3 promotes to default + Super-GOAT. Else → Phase 4 records negative result, ships opt-in on structural merits.

**riir-ai wiring (Phase 3, if PoC passes):** wire `compile_functor_graph` into `npc_integration.rs`, replacing hardcoded call order. riir-ai private (runtime composition).

**riir-train follow-up (noted, not blocked):** none expected — the fusion is explicitly modelless (freeze/thaw + gates, no GD). The only training dependency is the region *content* quality, which is the bridges' job (already shipped), not the canvas's.

---

## 7. Relationship to R398 and Percepta (captured to avoid re-deriving)

- **vs R398 (Canvas Engineering):** R403 is the modelless-friendly distillation of R398. R398 compiled onto attention masks (paper framing, needs DiT training); R403 compiles onto latent_functor gates (our substrate, modelless). Same reachability guarantee, different substrate. R403 supersedes R398's "needs training" conclusion *for our codebase*; R398's conclusion still holds for anyone distilling onto a literal DiT.
- **vs Percepta (R031/032):** Percepta compiles a deterministic *program* into *weights* (fully modelless, nothing to learn). R403 compiles a *topology* over *bridges whose content is already produced modellessly*. Percepta is one region producer in R403's graph (the deterministic-action region); R403 is the topology that binds multiple producers. Neither subsumes the other. See the R398 comparison table (this session) for the full spec→compile axis.
- **Why R398 said "needs training" and R403 doesn't:** R398 followed the paper's DiT framing (mask on attention positions; backbone must learn to respect it). R403 changes the substrate to latent_functor (gate on functor outputs; the functors already produce content modellessly; no backbone to train). The training dependency was a paper-framing artifact, not a fundamental requirement.

---

## TL;DR

Canvas Engineering's "needs training" conclusion (R398) is a paper-framing artifact: it compiled the schema onto attention-token positions of a DiT, and a DiT must train within the mask. **Our substrate is not a DiT — it is the flat `FunctorTable` (R123/Plan 303).** Distilling the canvas onto that substrate produces **topology-constrained latent functor composition**: a reachability graph over the existing latent_functor bridges, enforced by sigmoid subspace gates, with no transformer and no training. The reachability guarantee holds (absent edge = closed gate = exact marginal independence). Every region's content comes from a modelless producer that already ships (region_subspace for affect, DEC for perception, spectral_trajectory for trajectory, freeze/thaw for memory, Percepta for deterministic action). **Verdict: GOAT** — structural unification (flat table → decoupled, auditable, swap-isolated, attributed graph), not a new behavioral class at the modelless level. Q2 (behavioral gain) is unproven; the fusion PoC (FaithfulnessProbe-attributed, reusing the wired `jlens_concept_readout_goat.rs` pattern) is the Super-GOAT gate, tracked in Issue 122. The fusion is R398's canvas + R123's functor table + freeze/thaw + DEC + Percepta + FaithfulnessProbe — all pieces ship; the topology-over-functors wiring is the novel gap.
