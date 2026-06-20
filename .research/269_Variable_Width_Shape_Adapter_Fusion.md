# Research 269: Variable-Width `> <former` → Stage-Gated HLA Subspace Activation (Latent-Functor-LatCal Fusion)

> **Source:** Wu, Sieberling, Tan, Panda, Polyanskiy, Kim. *Variable-Width Transformers* (`> <former`). [arXiv:2606.18246](https://arxiv.org/abs/2606.18246). MIT / MIT-IBM Watson AI Lab. 16 Jun 2026.
> **Date:** 2026-06-19 (rev 2 — latent-functor-LatCal reframing after skill refinement; rev 3 — 2026-06-20, Issue 034 closed, downgraded to Gain after literature survey)
> **Status:** **Gain (downgraded 2026-06-20).** Both PRIMARY (stage-gated HLA subspace) and SECONDARY (shape-adaptive adapter routing) fusions have material prior art. Plan-only, feature-flagged, low priority — no primitive opened, no Super-GOAT promotion. See [`.issues/034_shape_adapter_novelty_gate.md`](../.issues/034_shape_adapter_novelty_gate.md) (CLOSED) and §3 below for the verdict + citation table.
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

**Downgraded to Gain (2026-06-20, Issue 034 closure).** Both the PRIMARY (stage-gated HLA subspace activation) and SECONDARY (shape-adaptive adapter routing) fusions have material prior art in the literature. Neither clears the Super-GOAT bar (novel-in-mechanism); the SECONDARY doesn't even clear the GOAT bar (novel-in-combination) cleanly. R269 stays as a plan-only, feature-flagged, low-priority Gain — no primitive opened, no Super-GOAT promotion, no mandatory follow-up outputs triggered.

### Q1.a — PRIMARY fusion (stage-gated HLA subspace activation): **PARTIAL PRIOR ART**

Three legs: (a) per-decision-stage latent subspace selection on recurrent belief, (b) carry-forward persistence via leaky integrator, (c) deterministic commitment for replay.

Closest prior art:
- **VSG / SVSG (Jain et al., NeurIPS 2022, [arXiv:2210.11698](https://arxiv.org/abs/2210.11698))** — "Variational Sparse Gating." Update equation `h_t = u_t·h̃_t + (1-u_t)·h_{t-1}` with `u_t ~ Bernoulli(p)` is *exactly* sparse subspace selection on a recurrent latent state with carry-forward of dormant dimensions. Anticipates legs (a) generically (per-step, not per-decision-stage) and (b) via a binary gate (not a leaky integrator — `gamma` decay in our `MultiLayerHlaCache` is closer to the paper's intent than VSG's hard Bernoulli). Does NOT anticipate leg (c) — VSG's gates are stochastic for exploration, the antithesis of deterministic commitment.
- **Sparsely Changing Latent States (NeurIPS 2021)** — L0-regularized rectified gating on latent state updates. Same leg (a)/(b) family.
- **Latent Structure of Affective Representations in LLMs ([arXiv:2604.07382](https://arxiv.org/abs/2604.07382))** — geometric analysis of affective subspaces in LLMs. Establishes that "affective subspace" is a populated concept; not a gating mechanism.
- **Appraisal-based chain-of-emotion ([arXiv:2309.05076](https://arxiv.org/abs/2309.05076))** — decision-context emotion simulation in games. Establishes the affective + decision-context framing; no latent subspace selection mechanism.

**Verdict:** no paper composes all three legs for affective NPC cognition with replay commitment, but two of three legs (sparse subspace update on recurrent belief + dormant-dim carry-forward) are direct prior art in VSG. The remaining novelty is the **composition** (affective stage axis + leaky-integrator persistence + LatCal commitment + 1000s-of-NPCs batching) — novel-in-combination, not novel-in-mechanism. Per the resolution table, this fails the Q1=YES criterion for Super-GOAT.

### Q1.a' — SECONDARY fusion (shape-adaptive adapter routing): **PRIOR ART**

Three legs: (a) per-layer adapter capacity profile, (b) runtime hot-swap between profiles, (c) inference-time layer skip driven by the profile.

Closest prior art:
- **FIM-LoRA ([arXiv:2605.16800](https://arxiv.org/abs/2605.16800))** — produces "a standard LoRA with a per-layer rank pattern — no new parameters, no training overhead, no changes to serving." Directly anticipates leg (a). Per-layer maps are interpretable ("value projections and early-to-middle layers consistently receive higher rank").
- **ALoRA ([arXiv:2403.16187](https://arxiv.org/abs/2403.16187))** — re-allocates LoRA ranks per layer during fine-tuning. Leg (a).
- **MoLA (NAACL 2025)** — LoRA-MoE with layer-wise expert allocation. Leg (a), different mechanism.
- **La-LoRA, AdaLoRA, IGU-LoRA** — layer-wise adaptive rank via various signals. Leg (a) family.
- **vLLM / HuggingFace PEFT hotswap / Unsloth** — runtime LoRA hot-swap is production shipping. Leg (b).
- **LoRA-Switch ([arXiv:2405.17741](https://arxiv.org/abs/2405.17741))** — token-wise LoRA routing. Leg (b), per-token not per-profile.
- **LayerRoute ([arXiv:2606.01838](https://arxiv.org/abs/2606.01838))** — "Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning." Trains a per-layer router + LoRA jointly; tool-call inputs skip 15.25% of FLOPs, planning inputs skip 2.34%. Anticipates leg (c) — LoRA-driven layer skip — but the skip plan is a *separately learned router*, not *derived from the adapter's per-layer shape profile*.
- **LoRA-Drop ([arXiv:2601.02569](https://arxiv.org/abs/2601.02569))** — temporal compute schedule on a fixed LoRA subset. Leg (c) family.

**Verdict:** legs (a) and (b) are fully anticipated by separate papers; leg (c) is partially anticipated by LayerRoute (skip driven by LoRA, but not by shape profile). Per the issue's Q1.a' criterion — "If any two of the three exist together, the fusion is GOAT (novel-in-combination) not Super-GOAT" — the SECONDARY is at best GOAT, and the specific composition (skip plan *derived from* adapter shape profile, *atomically hot-swapped with* the adapter) is a thin novelty over LayerRoute + FIM-LoRA + vLLM-hotswap. Downgraded to Gain.

### Q1.c — adapter-driven Hydra skip plan: **feasible, novel as code, not as concept**

Grep of `HydraBudgetConfig` call sites in `katgpt-rs/src/` and `katgpt-rs/crates/katgpt-core/src/` (2026-06-20):
- `crates/katgpt-core/src/types.rs:4203` — struct definition `{ skip_threshold, cumulative_threshold, modelless: bool, skip_erasure_draft: bool }`. **No adapter-aware field.** `modelless` means "lookup vs logit-lens," not "base vs adapter."
- `crates/katgpt-core/src/lib.rs:184` — re-export.
- `src/pruners/hydra_budget.rs:11,95` — `hydra_layer_skip(profiles: &[HydraLayerProfile], config: &HydraBudgetConfig)`. Profile source is the caller's responsibility; nothing prevents passing an adapter-derived profile.
- `tests/bench_165_hydra_budget_goat.rs` — benchmarks pass synthetic profiles.

**Verdict:** no adapter-aware variant exists in katgpt-rs today. The mechanism is feasible — extend `HydraBudgetConfig` with `adapter_profile: Option<HydraLayerProfile>` (or pass adapter-derived profiles into `hydra_layer_skip`). But LayerRoute already demonstrates LoRA-driven layer skip in the literature, so this is novel-as-shipped-code, not novel-as-concept.

### Q1.d — `SnapshotMeta` forward-compat: **NOT forward-compatible as-is — migration prerequisite**

`riir-ai/crates/riir-engine/src/snapshot.rs:300-310` (read 2026-06-20, cross-repo accessible from this workspace):

```rust
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SnapshotMeta {
    pub blake3_hash: [u8; 32],
    pub n_layers: usize,
    pub size_bytes: usize,
    pub timestamp: u64,
}
```

**No `#[serde(default)]` on any field.** Adding `width_profile: Vec<u8>` (or any new field) would cause `serde::Deserialize` to reject old snapshots with a missing-field error. R269 §5's assumption ("forward-compatible serde-with-default fields") is **wrong**. Any snapshot extension requires a forward-compat migration first: add `#[serde(default)]` on the new field AND bump `SNAPSHOT_VERSION` (currently 1). This is an additional cost against the Gain tier.

### Tier reasoning (final)

- **Not Super-GOAT:** Q1=YES on PRIMARY fails. VSG (2210.11698) directly anticipates the load-bearing mechanism (sparse subspace update on recurrent latent + dormant-dim carry-forward). The remaining PRIMARY novelty is composition (affective stage axis + LatCal commitment + multi-NPC batching) — novel-in-combination, not novel-in-mechanism. Per the research skill, Super-GOAT requires novel-in-mechanism.
- **Not GOAT:** GOAT requires a committed primitive + benchmarked gain. R269 has neither. The SECONDARY fusion's three legs are each separately prior art; the composition is thin over LayerRoute + FIM-LoRA + vLLM-hotswap.
- **Gain (final):** the fusion is a useful composition of our shipped primitives (Hydra Budget + HLA + LatCal + hot-swap) that *might* yield a small win in NPC compute-per-tick at iso-quality, but the moat is weak and the snapshot forward-compat migration adds cost. Plan-only, feature-flagged (`shape_adaptive_router`), low priority. No primitive opened. No mandatory follow-up outputs.

### Citation table

| Paper | arXiv | Anticipates | Gap vs R269 |
|---|---|---|---|
| VSG / SVSG (Jain et al., NeurIPS 2022) | [2210.11698](https://arxiv.org/abs/2210.11698) | PRIMARY legs (a) generically + (b) via binary gate | No affective axis, no per-stage gating, no deterministic commitment |
| Sparsely Changing Latent States (NeurIPS 2021) | — | PRIMARY leg (a) via L0 gating | Same as VSG |
| FIM-LoRA (Sathyavageeswaran, 2026) | [2605.16800](https://arxiv.org/abs/2605.16800) | SECONDARY leg (a) — per-layer rank pattern as adapter property | No skip coupling, no hot-swap |
| ALoRA | [2403.16187](https://arxiv.org/abs/2403.16187) | SECONDARY leg (a) — rank re-allocation | No skip, no swap |
| MoLA (NAACL 2025) | — | SECONDARY leg (a) — layer-wise expert allocation | MoE not rank-profile |
| LayerRoute (Sikdar, 2026) | [2606.01838](https://arxiv.org/abs/2606.01838) | SECONDARY leg (c) — LoRA-driven per-input layer skip | Skip from learned router, not from adapter shape profile; no hot-swap |
| LoRA-Switch | [2405.17741](https://arxiv.org/abs/2405.17741) | SECONDARY leg (b) — token-wise LoRA routing | Per-token not per-profile |
| LoRA-Drop | [2601.02569](https://arxiv.org/abs/2601.02569) | SECONDARY leg (c) — temporal compute schedule on LoRA subset | Not shape-profile-derived |
| Latent Structure of Affective Reps in LLMs | [2604.07382](https://arxiv.org/abs/2604.07382) | "Affective subspace" as populated concept | Analysis only, no gating |
| Appraisal-based chain-of-emotion | [2309.05076](https://arxiv.org/abs/2309.05076) | Decision-context emotion in games | No latent subspace mechanism |

## 4. What would change the verdict

Issue 034 closed (2026-06-20) with **both Q1.a (PRIMARY) and Q1.a' (SECONDARY) showing prior art** → R269 downgraded to **Gain** per the resolution table row 3. The original hypothetical table below is retained for audit:

| If Issue 034 finds... | Then... |
|---|---|
| Prior art on "per-layer adapter shape profile" in literature | Downgrade to **Gain** ← *this row fired* |
| No prior art; mechanism (adapter-driven layer suppression + Hydra skip + carry-forward) is novel | Upgrade to **Super-GOAT**. |
| Prior art exists but our specific composition (× OPD × Hydra × hot-swap) is novel-in-combination | **GOAT** — plan + implement behind feature flag, benchmark vs vanilla adapter routing, promote if it wins. |

**Actual outcome:** PRIMARY has partial prior art (VSG anticipates the load-bearing subspace-gating mechanism); SECONDARY has prior art on 2 of 3 legs (FIM-LoRA + vLLM-hotswap). Neither composition clears Super-GOAT or GOAT. Gain it is.

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

**Verdict (rev 3, 2026-06-20): Gain — downgraded from fusion-novelty-TBD after Issue 034 literature survey.** The architecture is training research (→ `riir-train`). The PRIMARY fusion (stage-gated HLA subspace activation) has its load-bearing mechanism — sparse subspace update on a recurrent latent with dormant-dim carry-forward — directly anticipated by **VSG / SVSG (Jain et al., NeurIPS 2022, [arXiv:2210.11698](https://arxiv.org/abs/2210.11698))**; the remaining PRIMARY novelty is composition (affective stage axis + LatCal commitment + multi-NPC batching), novel-in-combination not novel-in-mechanism. The SECONDARY fusion (shape-adaptive adapter routing) has prior art on 2 of 3 legs: **FIM-LoRA ([2605.16800](https://arxiv.org/abs/2605.16800))** for per-layer rank pattern, **vLLM/HF/Unsloth hotswap** for runtime swap; LayerRoute ([2606.01838](https://arxiv.org/abs/2606.01838)) partially anticipates the LoRA-driven-skip leg. Neither fusion clears Super-GOAT or GOAT. Plan-only behind `shape_adaptive_router` feature flag, low priority. **Caveat uncovered:** `riir-ai/crates/riir-engine/src/snapshot.rs::SnapshotMeta` has no `#[serde(default)]` — any per-layer-width-profile extension requires a forward-compat migration (add `#[serde(default)]` + bump `SNAPSHOT_VERSION`) as a prerequisite. No primitive opened; no Super-GOAT promotion; no mandatory follow-up outputs triggered.
