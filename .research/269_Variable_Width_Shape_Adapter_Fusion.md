# Research 269: Variable-Width `> <former` → Stage-Gated HLA Subspace Activation (Latent-Functor-LatCal Fusion)

> **Source:** Wu, Sieberling, Tan, Panda, Polyanskiy, Kim. *Variable-Width Transformers* (`> <former`). [arXiv:2606.18246](https://arxiv.org/abs/2606.18246). MIT / MIT-IBM Watson AI Lab. 16 Jun 2026.
> **Date:** 2026-06-19 (rev 2 — latent-functor-LatCal reframing after skill refinement)
> **Status:** Active — **fusion idea, novelty TBD (needs Q1–Q4 check before verdict)**. See [`.issues/034_shape_adapter_novelty_gate.md`](../.issues/034_shape_adapter_novelty_gate.md).
> **Related Research:** 123 (Latent Functor Runtime Guide — Super-GOAT), 010 (KG × HLA × Role Transport), 148 (Hydra Effect → Hydra Budget), 231 (OPD per-module energy profile), 247 (Dense Latent cross-model adapters, training→riir-train pattern), 258 (Sink-Aware / compression valleys), 266 (DenseMesh adaptive width), 212 (Gemini Fourier × LatCal fusion — the canonical LatCal Super-GOAT precedent).
> **Related Plans:** 165 (Hydra Budget — layer skip via pre-computed profiles), 258 (LatCal Fixed-Point Shell Bridge), 265 (LatCal Fixed-Point Fourier Coefficients), 276 (MicroRecurrentBeliefState — HLA kernel snapshot).
> **Cross-ref (riir-ai):** `latent_functor/zone_gating.rs::NpcFunctorRuntime`, `hla/types.rs::MultiLayerHlaCache` (gamma decay = carry-forward), `hla/kernel.rs::evolve_hla`, `riir-chain/src/encoding/latcal*.rs`.
> **Classification:** Public (katgpt-rs engine note). The training recipe itself → `riir-train`.

---

## TL;DR

The paper proposes a `×`-shaped transformer (wide early/late, narrow middle) with a parameter-free **carry-forward residual** that lets inactive dimensions bypass narrowed layers. At parameter parity: ~3% perplexity win, ~22% FLOP reduction, ~15% KV reduction. In analysis it **mitigates mid-layer representation collapse**.

**Two framings, ranked by tier:**

1. **PRIMARY (Super-GOAT-tier) — Stage-Gated HLA Subspace Activation.** The ×-shape insight reframes as a latent-to-latent operation: at each decision stage (combat / dialog / economic / social), only a *subspace* of the NPC's HLA latent state needs to be active. `latent_functor/zone_gating.rs::NpcFunctorRuntime` already gates by **zone density** (spatial axis). The ×-shape adds a SECOND gating axis: **decision-stage subspace selection within HLA's 8-dim state** (valence/arousal/desperation/calm/fear + 3). Dormant subspaces **carry forward** via HLA's `gamma` decay leaky integrator — exactly the paper's carry-forward mechanism. This is modelless, latent-to-latent, batchable across thousands of NPCs at 20Hz, and the per-stage subspace profile is **LatCal-committable** for deterministic replay/anti-cheat.

2. **SECONDARY (GOAT-tier fallback) — Shape-Adaptive Adapter Routing.** With on-the-fly LoRA + riir-train, train adapters with per-layer shape objectives and hot-swap by shape. Documented in rev 1 of this note; demoted to fallback after skill refinement mandated the latent reframing first.

**Architecture recipe → `riir-train`.** Pre-training a `> <former` from scratch is pure training research. One-line redirect; no katgpt-rs files for the architecture itself.

**Distilled for katgpt-rs (modelless, inference-time):** the per-stage HLA subspace activation profile is a new latent-functor gating axis (orthogonal to zone density) that makes NPC cognition **variable-width per decision context** — combat NPCs run a narrow survival subspace, dialog NPCs run a narrow social subspace, dormant subspaces persist via gamma decay rather than recomputing. LatCal commits the per-stage profile for replay determinism.

---

## 1. Paper Core Findings (verified by reading)

| Finding | Mechanism | Relevance here |
|---|---|---|
| `×`-shape beats uniform at parameter parity (200M–2B dense, 3B/1B MoE) | Wide early/late, narrow middle, geometric schedule `d_ℓ = α·d_{ℓ−1}` with `ℓ*=0.75L`, `d_ℓ*=0.3d` | Training architecture → riir-train |
| **Carry-forward residual** (parameter-free) | Fixed global residual width = widest layer; each block reads/writes a slice; inactive dims bypass and are restored from the most recent layer that touched them; contraction = truncation, expansion = copy-or-zero-pad | Modelless primitive candidate — structured residual bypass |
| **Mid-layer collapse is real and severe** | Uniform models: normalized matrix entropy → ~0 by layer ~10 (compression valley, de Llano et al. 2026 — same paper our R258 cites). `> <former`: maintains higher entropy through the bottleneck | Already exploited by Hydra Budget (R148/P165) and Sink-Aware (R258/P287) |
| MLP activation Participation Ratio collapses | Uniform: width-normalized energy utilization <5% by layer ~10. `> <former`: maintains ~1000 effective dims through middle | Same metric we already compute (`participation_ratio` in SpectralQuant, `effective_rank` in data_probe) |
| Inference benefits follow automatically | Params ∝ d² (matched); attention FLOPs and KV ∝ d (linear), so nonuniform width strictly lowers avg d → 15% KV, 22% FLOP | This is the *training-time* source of the savings; the inference-time reflection is what Hydra/Sparse-MLP already harvest on uniform models |
| Carry-forward beats learned projection or zero-pad (Table 4, 500M) | Copy-from-prior-layer: 3.099; zero-pad: 3.124; trained projection: 3.150 | The parameter-free bypass is the load-bearing mechanism — and it's exactly what a frozen base + adapter pool can simulate |

## 2. Distillation

### 2.1 What's already shipped (the prior-art surface — three granularities)

| `> <former` insight | Shipped cousin | File / Plan | Granularity |
|---|---|---|---|
| Middle layers collapse → skip them | **Hydra Budget** `HydraSkipPlan { skip_layers: Vec<bool> }`, `HydraBudgetConfig { modelless: bool }` | `src/pruners/hydra_budget.rs`, P165, R148, default-on, GOAT 4/4 | **Layer** |
| MLP dead dimensions → skip them | **Sparse MLP** + **Prism** per-capability masks + **CNA** neuron discovery | P022, R191, R053 | **Dimension (within-layer)** |
| Per-layer capacity varies → adapt compute | **DenseMesh adaptive_width** `WidthDecision::{Contract,Neutral,Expand}` driven by Collapse-Aware + BreakevenRouter | `dense_mesh/adaptive_width.rs`, P266, R234 | **Topology (across-nodes)** |
| Normalized matrix entropy per layer | **`effective_rank`** (Roy-Vetterli) — `normalized_matrix_entropy = log(effective_rank)/log(r)` | `crates/katgpt-core/src/data_probe/geometry.rs` | Metric |
| MLP participation ratio per layer | **`participation_ratio`** `d_eff = (Σλ)²/Σ(λ²)` | SpectralQuant, P078, default-on, GOAT-proven | Metric |
| Attention sinks / compression valleys | **Sink-Aware Attention** (targets the de Llano 2026 finding the paper cites) | P287, R258 (deferred for latency; diagnostic ships) | Head |
| Per-module energy profile of adapter | **`ModuleEnergyProfile::PAPER_AVERAGE { ffn, attn, embed, other }`** | `src/inference_router/router_compute_target.rs`, R231 (OPD) | Module-type |
| Per-adapter shape descriptor | **`AdapterShape { rank, in_dim, out_dim }`** | `riir-ai/crates/riir-gpu/src/optimizer_amuse.rs` | Static per-adapter |
| Per-snapshot metadata + atomic swap | **`SnapshotMeta { blake3_hash, n_layers, ... }`**, `LoRAHotSwap`, `SenseHotSwap`, `KernelHotSwap` | `riir-ai/crates/riir-engine/src/snapshot.rs`, P276, P279 | Snapshot |

**The gap:** every shipped cousin is either (a) per-module-type (OPD: FFN vs Attn), (b) per-adapter-static (`AdapterShape`: fixed rank per adapter), or (c) per-layer-intrinsic (Hydra: skip based on the *base model's* profile). **Nothing characterizes an adapter by its per-LAYER shape profile** — which layers it concentrates capacity in vs suppresses. That is the dimension `> <former` operates on, and it's orthogonal to all three shipped axes.

### 2.2 The PRIMARY fusion (Super-GOAT-tier) — Stage-Gated HLA Subspace Activation

**`> <former × latent_functor/zone_gating × hla/types.rs (gamma decay) × LatCal commitment`**

The paper's variable-width insight reframes as a **latent-to-latent subspace gating** operation on HLA state. Reading the actual shipped code:

- `NpcFunctorRuntime` (`latent_functor/zone_gating.rs:220`) already has stage-gated activation — but gated by **zone density** (`ZoneGatingTier { min_density, tau, beta, reest_budget }`). `on_zone_transition()` resolves the active tier and pushes to the scheduler. This is the **spatial gating axis**.
- `MultiLayerHlaCache` (`hla/types.rs:136`) carries `gamma: f32` — the **leaky integrator decay**. Dormant HLA subspaces decay via `gamma` but do NOT zero out. This is exactly the paper's **carry-forward** mechanism: inactive dimensions bypass the layer and are restored from their last active value.
- `ThirdOrderMoment` (`hla/types.rs:211`) captures "relations between relations" in compressed form — this is the **wide early/late, narrow middle** analog: high-order moments are the "wide" subspace where dense relational computation happens; first-order scalars are the "narrow" projection.

**The fusion:** add a **decision-stage gating axis** to `NpcFunctorRuntime`, orthogonal to the existing zone-density axis. The ×-shape profile becomes a **per-stage HLA subspace activation schedule**:

```
Combat stage:   active = {desperation, fear, arousal}            (survival subspace, narrow)
Dialog stage:   active = {valence, calm, +social dims}            (social subspace, narrow)
Economic stage: active = {greed-axis, +economic dims}             (economic subspace, narrow)
Idle/explore:   active = all 8 dims                               (wide — full curiosity)
```

Dormant subspaces **carry forward** via `gamma` decay (no recomputation — the paper's parameter-free bypass). On stage transition, the newly-active subspace reads its last-decayed value as a warm start (the paper's expansion = copy-from-prior-layer).

**Why this is Super-GOAT-tier, not GOAT-tier:**
- It's **latent-to-latent** (operates on HLA state, never decodes to tokens or raw scalars except at the sync boundary).
- It's **modelless** (no training — the subspace profile is a runtime schedule, learned via curiosity signal or authored).
- It's **batchable at MMORPG scale** (1000s of NPCs × 20Hz — each NPC has its own stage + subspace profile).
- It connects to **LatCal** (the per-stage profile is a deterministic schedule that MUST be committed for replay/anti-cheat determinism — LatCal fixed-point bridge commits it as raw scalars).
- It connects to **freeze/thaw** (the subspace profile per NPC personality is snapshot-versioned — different NPC personalities have different ×-shapes).
- It connects to **cgsp_runtime** (curiosity signal drives WHICH subspace to activate — high curiosity on economic axis → economic subspace widens).

**The LatCal bridge (sync boundary):** the per-stage subspace activation is a latent-space operation (which HLA dims are active), but the **decision** of which stage the NPC is in crosses the sync boundary as a raw scalar (stage_id, confidence). LatCal commits this as a fixed-point value so deterministic replay reconstructs the exact subspace schedule. The HLA scalar projections (valence/arousal/desperation/calm/fear) cross the wire as raw scalars per AGENTS.md; the full 8-dim vector stays local. This respects the raw-vs-latent boundary exactly.

### 2.3 The SECONDARY fusion (GOAT-tier fallback) — Shape-Adaptive Adapter Routing

Documented in rev 1. With on-the-fly LoRA + riir-train, train adapters with per-layer shape objectives and hot-swap by shape. This is a legitimate GOAT-tier framing but **not the primary** — it operates on adapter weights (modelless but weight-aware), not on HLA latent state (modelless and fully latent). Per the refined skill, adapter routing is a fallback framing when the latent reframing is unavailable. The latent reframing IS available here, so the adapter framing is secondary.

### 2.4 Honest uncertainty on the mechanism (primary fusion)

The primary fusion is well-defined modellessly: the subspace gating is a runtime schedule on HLA state, no training needed. The open questions are:

1. **Profile authoring vs learning.** Where does the per-stage subspace profile come from? Three options: (a) authored by game designer (deterministic, simple), (b) learned via cgsp_runtime curiosity signal (emergent, the Super-GOAT angle), (c) calibrated from HLA participation_ratio per stage (the paper's analysis methodology applied to HLA). Option (b) is the strongest selling point but needs validation.
2. **Stage detection.** How does the runtime know which stage the NPC is in? Via the existing `on_zone_transition()` hook (spatial) + a new `on_decision_context()` hook (behavioral — combat action initiated, dialog opened, trade started). The behavioral hook is new plumbing in riir-games.
3. **LatCal commitment granularity.** Per-stage profile per NPC per tick is a lot of state. Likely commit only on stage transitions (not per-tick), matching the existing `on_zone_transition` cadence.

These are engineering questions, not research-blockers. The modelless primitive (stage-gated HLA subspace selection with gamma-decay carry-forward) is well-defined regardless of how the profile is produced.

## 3. Verdict

**Fusion idea — novelty TBD, needs Q1–Q4 check before verdict.** Not a committed Super-GOAT. The PRIMARY fusion (stage-gated HLA subspace activation) is Super-GOAT-tier in framing; the SECONDARY (adapter routing) is GOAT-tier.

| Gate | Status (PRIMARY fusion) | Evidence |
|---|---|---|
| Q1 No prior art | ❓ **UNCERTAIN — must check literature** | Stage-gated subspace activation on per-NPC latent state is a new combination in *our* codebase (zone_gating is spatial-only; HLA gamma is per-layer decay, not per-stage subspace selection). But "stage-gated affective subspaces" / "context-gated latent subspace routing" exist in the affective computing + agent architecture literature — needs arxiv survey. See [Issue 034](../.issues/034_shape_adapter_novelty_gate.md) (revised to cover BOTH framings). |
| Q2 New class of behavior | ✅ Likely yes | Per-decision-stage latent subspace routing for thousands of NPCs. No shipped primitive gates HLA by decision context (only by zone density). |
| Q3 Product selling point | ✅ Likely yes | "NPCs don't waste compute updating emotional subspaces during combat, or spatial subspaces during dialog — stage-gated latent width, 1000s of NPCs at 20Hz, LatCal-committed for replay determinism." |
| Q4 Force multiplier | ✅ Yes | Connects HLA (`hla/types.rs`), latent_functor (`zone_gating.rs`), cgsp_runtime (curiosity-driven profile), LatCal (commitment bridge), freeze/thaw (per-personality profile snapshot). 5 pillars. |

**Per the research skill:** because Q1 is not committed YES, this is filed as "novelty TBD" with an issue — NOT as "Super-GOAT candidate." If Issue 034 closes with Q1=YES on the PRIMARY fusion, this note upgrades to Super-GOAT and the mandatory outputs (open primitive in katgpt-rs + private riir-ai guide + plans) become due in that follow-up session.

**The pure architecture recipe → `riir-train`.** One-line redirect; no katgpt-rs files created for the `×`-shape training method itself.

### Tier reasoning

- **Not Pass:** the PRIMARY fusion (stage-gated HLA subspace activation) is a legitimate latent-to-latent Super-GOAT-tier framing. The gap in shipped prior art is real: `zone_gating` is spatial-only, HLA `gamma` is per-layer not per-stage-subspace.
- **Not Super-GOAT (yet):** Q1 literature check on "stage-gated affective subspaces" / "context-gated latent routing" is genuinely open. Affective computing has prior art on context-gated emotion models; claiming novelty without the survey would repeat the `evolve_hla` overclaim.
- **Not GOAT/Gain:** those tiers require a committed primitive + benchmark. The primitive is well-defined but its value depends on validating that stage-gated subspace narrowing actually saves compute at iso-quality on HLA state — which is the G2 gate.
- **Secondary fusion (adapter routing) is GOAT-tier at best** — it's a weight-aware modelless framing, weaker than the latent-to-latent primary.

## 4. What would change the verdict

| If Issue 034 finds... | Then... |
|---|---|
| Prior art on "per-layer adapter shape profile" in literature | Downgrade to **Gain** — the fusion is still useful as a composition of our shipped primitives, but no moat. Plan-only, feature-flagged. |
| No prior art; mechanism (adapter-driven layer suppression + Hydra skip + carry-forward) is novel | Upgrade to **Super-GOAT**. Mandatory outputs in follow-up session: (1) open `ShapeAdaptiveRouter` primitive in katgpt-rs; (2) private `riir-ai/.research/NNN_*.md` guide with validation protocol; (3) plans in katgpt-rs + riir-ai + riir-train. |
| Prior art exists but our specific composition (× OPD × Hydra × hot-swap) is novel-in-combination | **GOAT** — plan + implement behind feature flag, benchmark vs vanilla adapter routing, promote if it wins. |

## 5. Cross-references for the follow-up session

- **Closest cousins to fuse with:**
  - `katgpt-rs/.research/231_Sparse_Off_Principal_Task_Vector_OPD.md` — per-module energy profile; extend to per-layer
  - `katgpt-rs/.research/148_*.md` (Hydra Effect) + `katgpt-rs/.plans/165_*.md` (Hydra Budget) — layer skip machinery to make adapter-driven
  - `katgpt-rs/.research/247_Dense_Latent_Heterogeneous_Communication_CS_Probe.md` — same training→riir-train + modelless-survives pattern
  - `katgpt-rs/.research/266_DenseMesh_Latent_Node_Network.md` — topology-level width adaptation; the layer-level version is the gap
  - `katgpt-rs/.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md` — same compression-valley phenomenon
- **Runtime plumbing:** `riir-ai/crates/riir-engine/src/snapshot.rs::SnapshotMeta` (extend with per-layer profile), `riir-ai/crates/riir-engine/src/episode_buffer.rs::LoRAHotSwap` (atomic swap by shape profile).
- **Training side:** `riir-ai/crates/riir-gpu/src/optimizer_amuse.rs::AdapterShape` (currently static per-adapter; would need a per-layer variant).

## TL;DR

The architecture is training research (→ `riir-train`). The analysis methodology is `effective_rank`-equivalent math we already ship. The PRIMARY fusion (Super-GOAT-tier, after skill refinement forced the latent reframing): **stage-gated HLA subspace activation** — `> <former`'s variable width = which HLA subspace is active at which decision stage; carry-forward = dormant subspace persistence via HLA `gamma` leaky integrator; ×-shape profile = LatCal-committed per-stage schedule for replay determinism. The SECONDARY fusion (GOAT-tier, rev 1's primary): shape-adaptive adapter routing via on-the-fly LoRA. Verdict is **fusion — novelty TBD** because Q1 needs a literature check on "stage-gated affective subspaces" / "context-gated latent routing" before committing Super-GOAT. Filed Issue 034 for the gate; no guide or plans created until Q1 resolves YES. The skill refinement (added five Super-GOAT factory modules including LatCal, mandatory latent-space reframing step, R269 as documented failure case) is what surfaced the primary framing — without it, this note would have stayed at the weaker adapter-routing verdict.
