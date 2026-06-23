# Research 290: Latent Field Steering — Residual-State-Mutating Steering on the Hot Path

> **Source:** Synthesized from CAA literature (Panickssery et al. 2023), Anthropic Transformer Circuits (Research 144, functional emotions), and the Gemini "wave interference" reframing (2026-06-23)
> **Date:** 2026-06-23
> **Status:** Active
> **Related Research:** 144 (Functional Emotions), 257 (FUNCATTN), 267 (FPCG — explicit non-mutation baseline), 219 (TNO/DEC)
> **Related Plans:** 162 (Emotion Vector — read-only), 087 (CNA — neuron-level mutation), 292 (FPCG — sample-level, no mutation), 309 (this primitive)
> **Cross-ref (riir-ai):** Research 153 (Latent Field Steering Game Runtime Guide)
> **Classification:** Public

---

## TL;DR

Latent Field Steering is the **top-down control direction** complement to our
existing bottom-up emotion computation. Today, NPC affect is computed bottom-up
(HLA kernel reads raw state → projects to 5 scalars). Latent Field Steering
injects a **designer-side or environment-side direction vector directly into the
latent state**, shifting affect without recomputing HLA. This is the "wave
interference" mechanism Gemini described — mathematically, linear superposition
in activation space, applied as a top-down perturbation.

**Distilled for katgpt-rs (modelless, inference-time):**
A zero-allocation, SIMD-accelerated `apply_latent_steering(state: &mut [f32],
field: &LatentField)` plus a localized `LatentField` variant that applies the
steering only to entities within a support region (radius, zone, or graph
neighborhood). No training, no gradients — the direction vectors are frozen,
BLAKE3-committed artifacts loaded at init.

---

## 1. Mechanism

### 1.1 The reframe

Our existing emotion infrastructure is **read-only detection**:

- `EmotionDirections::project` (Plan 162) — reads emotion directions from
  mid-layer activations, zero-cost, no mutation.
- `CNA` (Plan 087) — mutates a sparse set of MLP neurons (~10–50), preserves
  quality ≥0.97. Closer to mutation, but neuron-level not latent-state-level.
- `FPCG` (Plan 292) — explicitly **refuses** to mutate the residual stream;
  intervenes at the sample selector.

Latent Field Steering is the **missing fourth quadrant**: mutate the latent
state directly, on the hot path, with a designer/environment-supplied direction
vector. The "wave interference" framing is the mathematical interpretation:
linear superposition of two latent fields (NPC's current state + injected
steering field) produces a new field that is the element-wise sum.

### 1.2 Why this is not CAA

CAA (Constitutional AI Activation Addition) on LLM residual streams degrades
quality to <0.60 at effective multipliers (Benchmark 015). We rejected it for
the LLM side. **The game-AI NPC hot path is a different setting**:

- NPCs aren't generating fluent text — quality degradation from residual
  perturbation doesn't apply the same way.
- NPC latent state is 5-dim affective (valence/arousal/desperation/calm/fear) +
  3 reserved, not a 4096-dim residual stream. Perturbation is small and bounded.
- The "quality" we care about is **behavior rank preservation** (does the NPC
  still choose the right action?), not text fluency.

### 1.3 The math

Given NPC latent state `s ∈ R^d` (d=8 for HLA) and steering direction `v ∈ R^d`
(unit norm), with strength `α ∈ [0, 1]`:

```
s' = s + α · v
```

For localized fields (only NPCs within support region R):

```
s'_i = s_i + α · v · kernel(distance(i, field_center), bandwidth)
```

The kernel is a **sigmoid falloff** (per AGENTS.md: sigmoid not softmax/Gaussian
for projections):

```
kernel(d, b) = sigmoid((b - d) · k)   // 1 inside bandwidth, 0 outside, smooth
```

### 1.4 Resolution invariance

Because the steering is element-wise on the latent state, it is
**resolution-invariant** in the same sense as FUNCATTN: a steering vector
trained on a 5-dim HLA applies identically to a 5-dim HLA at any crowd scale.
Per-NPC cost is `O(d)`, independent of crowd size.

---

## 2. Distillation

### 2.1 Transferable primitive

The primitive is the **direction-vector injection with localized support**,
stripped of any game semantics:

1. `LatentSteeringVector` — a unit-norm direction vector + scalar strength,
   BLAKE3-committed, freeze/thaw-compatible.
2. `apply_latent_steering` — element-wise SIMD add into a mutable latent slice.
3. `LatentField` — a steering vector + support descriptor
   (radius/zone/graph-neighborhood) + kernel.
4. `LatentField::apply_to_crowd` — batched application to a crowd of latent
   states, with per-entity kernel weighting.

### 2.2 Where the pieces already live

| Piece | Existing location | Reuse |
|---|---|---|
| Direction vector storage | `EmotionDirections` (Plan 162), `NeuronShard::style_weights` | ✅ same artifact format |
| Sigmoid projection | `EmotionDirections::project`, FUNCATTN sigmoid basis | ✅ same math |
| BLAKE3 commitment | `MerkleFrozenEnvelope` (`riir-neuron-db/src/freeze.rs`) | ✅ same envelope |
| Localized support | `latent_functor/zone_gating.rs`, `SpatialBelief::visible_radius` | ✅ same spatial reasoning |
| Crowd-scale batch | `latent_functor/npc_integration.rs` crowd loops, `crowd_mcgs` | ✅ same parallelism |
| Atomic swap | `riir-engine/src/snapshot.rs`, `LoRAHotSwap` | ✅ same pattern |

**Nothing here is new math.** What's new is the **control direction** (top-down
injection vs bottom-up computation) and the **application point** (latent state
on the hot path, not residual stream in a transformer).

### 2.3 Closest cousins (3)

1. **CNA (Plan 087)** — neuron-level steering. Closest existing mechanism.
   Differs in target (MLP neurons vs latent state) and granularity
   (sparse ~50 vs dense 8).
2. **EmotionDirections (Plan 162)** — same direction vectors, opposite
   direction (read vs write).
3. **PersonalityWeightedComposition (Plan 297)** — layer composition with
   sigmoid-gated direction drift. Closer to "steering over time" than
   "steering at a tick".

### 2.4 Fusion

**F1 (PRIMARY — riir-ai): Latent Field Steering × Zone Gating × CWM**
Induced game rules (CWM, Plan 145) become **steering fields** instead of hard
constraints. A "fog of war" rule becomes a low-valence field applied to NPCs
outside visible radius. A "danger zone" rule becomes a high-arousal field. This
unifies soft (field) and hard (constraint) game rules under one mechanism.

**F2 (SECONDARY — katgpt-rs): Latent Field Steering × FUNCATTN**
FUNCATTN transports latent fields between manifolds. A steering field defined
on the emotion manifold can be transported to the behavior manifold via the
FUNCATTN C operator, giving **cross-domain steering** (steer emotion, effect
propagates to behavior automatically).

**F3 (TERTIARY — speculative): Latent Field Steering × Freeze/Thaw**
Steering vectors are versioned alongside personality shards. A faction's
"battle stance" is a frozen steering field applied to all members; swapping
the field (atomic Arc swap) instantly shifts the faction's posture. This is
the "wave interference without breaking frozen snapshots" claim from Gemini —
operationally, it's an additive overlay on top of frozen state, not a mutation
of the frozen state itself.

---

## 3. Verdict

### Tier: **Super-GOAT (candidate — pending G1–G4 validation)**

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **YES (in our codebase).** Grep for `residual.stream.*steer`, `mutate.*activation`, `hot.path.*steer`, `wave.interference` returns zero code hits. CNA mutates neurons (not latent state); FPCG explicitly refuses mutation; EmotionDirections is read-only. **Game-AI NPC hot path with direct latent state injection is genuinely new for us.** External prior art (CAA) exists but we rejected it for LLMs; the game-AI reframing is the novel angle. | Vocabulary translation done: "steering vector" → "direction vector", "wave interference" → "linear superposition", "residual stream mutation" → "latent state injection". |
| Q2 New class of behavior? | **YES.** Today, NPC affect is bottom-up only (state → HLA → affect). Latent Field Steering enables **top-down control** (designer/environment → affect directly). New control direction, not an optimization. Enables: designer-time crowd steering, environment-driven mood fields, faction-wide posture shifts in 1 tick. | |
| Q3 Product selling point? | **YES.** "Designers drop a latent field vector; thousands of NPCs respond in 1 tick without re-deriving their emotion state." Concrete, differentiated, demoable. | |
| Q4 Force multiplier? | **YES.** Connects EmotionDirections (R144) + PersonalityWeightedComposition (P297) + HLA kernel + CWM (P145) + zone gating + FUNCATTN (R257) + freeze/thaw. ≥6 pillars. | |

**Selling point:** NPCs steered by latent field injection — designer drops a
vector, crowd responds in 1 tick without re-derivation. Top-down control
complements existing bottom-up computation.

**Not Super-GOAT if:** G2 (behavior rank preservation) fails — if steering
corrupts NPC decision-making, the primitive is dangerous and demotes to Gain
(research-only, never shipped to hot path).

### Routing

- **katgpt-rs/.plans/309_latent_field_steering_primitive.md** — open primitive.
  Zero-alloc SIMD `apply_latent_steering` + `LatentField` + crowd batch. Feature
  flag `latent_field_steering`. GOAT gate G1–G5.
- **riir-ai/.research/153_latent_field_steering_game_runtime_guide.md** —
  private guide (this Super-GOAT's selling-point doc).
- **riir-ai/.plans/** — deferred until katgpt-rs primitive passes G1–G2.

---

## 4. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Direction vectors are frozen artifacts; no gradients. |
| Latent-to-latent preferred | ✅ Operates entirely in latent space; never crosses to tokens. |
| Use sigmoid not softmax | ✅ Kernel is sigmoid falloff; strength α is sigmoid-bounded. |
| Freeze/thaw over fine-tuning | ✅ Steering vectors are BLAKE3-committed; atomic Arc swap for field hot-swap. **Critical:** steering is an additive overlay, NOT a mutation of frozen state. The frozen shard is read-only; the steering field is a separate mutable overlay. |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; game integration → riir-ai. |
| Raw scalars at sync boundary | ✅ Steering stays latent; only the 5 scalar affect outputs cross sync. |
| Zero-alloc hot path | ✅ SIMD element-wise add; `Vec::with_capacity` once for crowd batch. |

---

## 5. Open questions / risks

1. **Does steering preserve behavior rank?** The headline risk. If adding
   `α·v` to latent state changes which action the NPC selects (beyond the
   intended affect shift), the primitive is dangerous. **Mitigation:** G2 gate
   measures cosine similarity of action rankings pre/post steering; gate
   requires ≥0.95.
2. **Does steering leak across NPCs?** A localized field must not affect NPCs
   outside its support. **Mitigation:** `kernel(d, b) = sigmoid((b-d)·k)` is ~0
   outside bandwidth; G3 verifies zero leakage at distance > b + ε.
3. **Steering vs re-derivation: when is steering cheaper?** If the designer's
   intent can be expressed as a raw-state change (e.g., "spawn a predator"),
   bottom-up HLA re-derivation may be more accurate than top-down steering.
   **Mitigation:** steering is for **affect-level intent** that doesn't
   correspond to a raw-state change (e.g., "make this zone feel eerie",
   "faction battle stance").
4. **Steering strength calibration.** α ∈ [0,1] is designer-tunable. Too high →
   NPCs behave identically (lost individuality); too low → no effect.
   **Mitigation:** per-field α defaults derived from `EmotionDirections` norm;
   G1 sweeps α.

---

## TL;DR

Latent Field Steering is the missing top-down control direction for NPC affect:
inject a frozen direction vector directly into the latent state, with optional
localized support. Math is element-wise SIMD add (not new); application point
(latent state on hot path, not residual stream in transformer) and control
direction (top-down vs bottom-up) are new for us. Super-GOAT candidate pending
G1–G5: steering strength, behavior rank preservation, localization, crowd-scale
perf, zero-alloc. Kills itself if G2 fails (rank corruption). Plan 309; guide at
`riir-ai/.research/153`.
