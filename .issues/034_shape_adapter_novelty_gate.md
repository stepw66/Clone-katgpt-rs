# Issue 034: Shape-Adaptive Adapter Novelty Gate — close Q1 before verdict on Research 269

**Status:** CLOSED (**R269 downgraded to Gain** — both PRIMARY and SECONDARY fusions have prior art; plan-only, feature-flagged, low priority; no primitive opened.)

**Opened:** 2026-06-19
**Closed:** 2026-06-20
**Blocks:** Final verdict on [Research 269](../.research/269_Variable_Width_Shape_Adapter_Fusion.md) (`> <former` × on-the-fly LoRA × Hydra layer-skip fusion).
**Owner:** unassigned
**Type:** novelty gate (literature survey + mechanism feasibility check)

**Closure rationale (2026-06-20):** Literature survey of 12 arxiv keyword searches (3 returned 0 hits on exact-phrase — itself a weak novelty signal; the rest timed out at the arxiv search UI and were re-run via broader web search) plus full reads of the 3 closest papers produced prior art on BOTH framings. **Q1.a (PRIMARY — stage-gated HLA subspace activation):** PARTIAL PRIOR ART. VSG / SVSG (Jain et al., NeurIPS 2022, [arXiv:2210.11698](https://arxiv.org/abs/2210.11698)) directly anticipates the load-bearing mechanism — sparse subspace update on a recurrent latent state with dormant-dimension carry-forward via `h_t = u_t·h̃_t + (1-u_t)·h_{t-1}`, `u_t ~ Bernoulli(p)`. VSG covers PRIMARY legs (a) [subspace selection on recurrent belief] and (b) [carry-forward] but not leg (c) [deterministic commitment for replay]; it also lacks the affective / per-decision-stage axis. The remaining PRIMARY novelty is composition (affective stage axis + leaky-integrator persistence + LatCal commitment + 1000s-of-NPCs batching), which is novel-in-combination, not novel-in-mechanism — fails the Super-GOAT bar. **Q1.a' (SECONDARY — shape-adaptive adapter routing):** PRIOR ART on 2 of 3 legs. FIM-LoRA ([arXiv:2605.16800](https://arxiv.org/abs/2605.16800)) produces "a standard LoRA with a per-layer rank pattern" (leg a); vLLM / HuggingFace PEFT hotswap / Unsloth ship runtime LoRA hot-swap in production (leg b); LayerRoute ([arXiv:2606.01838](https://arxiv.org/abs/2606.01838)) trains LoRA + per-layer skip router jointly (leg c, partial — skip from learned router, not from adapter shape profile). Per the issue's Q1.a' criterion, "any two of the three exist together" → at best GOAT, and the specific composition (skip plan *derived from* adapter shape profile, atomically hot-swapped with the adapter) is a thin novelty over LayerRoute + FIM-LoRA + vLLM-hotswap. **Q1.c (adapter-driven Hydra skip):** grep of `HydraBudgetConfig` call sites in `katgpt-rs/src/` and `katgpt-rs/crates/katgpt-core/src/` confirms no adapter-aware variant exists today (struct at `types.rs:4203`, callers in `pruners/hydra_budget.rs` + `tests/bench_165_hydra_budget_goat.rs`); mechanism is feasible to add but LayerRoute already demonstrates LoRA-driven layer skip in the literature, so this is novel-as-shipped-code, not novel-as-concept. **Q1.d (SnapshotMeta forward-compat):** `riir-ai/crates/riir-engine/src/snapshot.rs:300-310` has NO `#[serde(default)]` on any field — R269 §5's forward-compat assumption was wrong; any per-layer-width-profile extension requires a migration (add `#[serde(default)]` + bump `SNAPSHOT_VERSION` from 1) as a prerequisite. **Action on R269:** per the resolution table row 3 ("Both have prior art → Downgrade R269 to Gain. Plan-only, feature-flagged, low priority. Close this issue."), R269 is downgraded to **Gain**. No primitive opened, no Super-GOAT promotion, no mandatory follow-up outputs triggered. If implemented later, it goes behind the `shape_adaptive_router` feature flag and must benchmark vs vanilla adapter routing before any promotion. See R269 §3 for the full citation table.

---

## Context

Research 269 documents a fusion idea sparked by `> <former` (arXiv:2606.18246): **shape-adaptive adapter routing** — train LoRA adapters with explicit per-layer shape objectives (e.g. ×-shape narrow-middle for fast/combat, wide-middle for deep/dialog), hot-swap between shape profiles at runtime, and drive Hydra Budget's layer-skip plan off the *adapter's* per-layer profile rather than the *base model's* intrinsic profile.

The in-codebase novelty check passed (vocabulary-translated grep across both repos, both layers — no shipped primitive characterizes an adapter by its per-layer shape profile; OPD is per-module-type, `AdapterShape` is static per-adapter, Hydra profiles the base). But the **broader literature check (Q1)** was not done in-session and the research skill explicitly forbids committing Super-GOAT without it.

The honest call in R269 was "fusion — novelty TBD" rather than "Super-GOAT candidate" precisely because this gate is open. This issue tracks closing it.

## The four sub-questions to resolve

### Q1.a — Is "stage-gated HLA subspace activation" novel? (PRIMARY fusion, latent-to-latent)

Survey arxiv for (use the keyword search URL from AGENTS.md):
- `stage-gated affective subspace`
- `context-gated latent subspace routing`
- `decision-context emotion model gating`
- `variable-width latent state agent`
- `subspace selection recurrent belief`
- `carry-forward dormant latent dimension`

**Pass criterion:** no paper proposes (per-decision-stage latent subspace selection on a recurrent belief state) × (carry-forward persistence via leaky integrator) × (deterministic commitment for replay). This is the PRIMARY fusion — if it's novel, R269 promotes to Super-GOAT.

### Q1.a' — Is "per-layer adapter shape profile" novel? (SECONDARY fusion, adapter-routing)

Survey arxiv for (use the keyword search URL from AGENTS.md):
- `layer-wise adapter capacity allocation`
- `adapter shape profile routing`
- `variable-width LoRA`
- `layer-skipping adapter composition`
- `per-layer adapter rank allocation` (this one almost certainly has hits — rank-per-layer is a known axis; check whether it's been combined with runtime routing)
- `adapter width profile hot-swap`

**Pass criterion:** no paper proposes (per-layer adapter capacity profile) × (runtime hot-swap between profiles) × (inference-time layer skip driven by the profile). If any two of the three exist together, the fusion is GOAT (novel-in-combination) not Super-GOAT (novel mechanism). If all three exist together, downgrade R269 to Gain.

### Q1.b — Is the "emergent narrowing" mechanism feasible?

The fusion cannot structurally narrow a frozen uniform base. It relies on:
1. Adapter learning to suppress its own contribution to middle-layer output dims (low-rank cancel).
2. Hydra Budget detecting the suppression via `effective_rank` / `participation_ratio` drop on a calibration set.
3. Residual stream carrying bypassed info forward (already structurally true).

**Open question for riir-train:** can a low-rank LoRA meaningfully "narrow" a layer's effective width without hurting quality? This is a training feasibility question, not modelless. File a separate riir-train issue if R269 promotes.

### Q1.c — Is "adapter-driven Hydra skip plan" novel?

Hydra Budget's `HydraBudgetConfig { modelless: bool }` today means "use a pre-computed profile of the *base model*." The fusion redefines this as "use a pre-computed profile of the *currently-loaded adapter*" — meaning the skip plan changes on hot-swap. Confirm this is not already implemented (grep `HydraBudgetConfig` call sites for any adapter-aware variant).

### Q1.d — Does `SnapshotMeta` extension break anything?

R269 proposes extending `riir-ai/crates/riir-engine/src/snapshot.rs::SnapshotMeta` with a per-layer width profile (BLAKE3-committed). Confirm the existing `SnapshotMeta` serialization is forward-compatible (serde-with-default fields) so old snapshots load without the profile.

## Resolution criteria

| Outcome | Action on R269 |
|---|---|
| Q1.a (PRIMARY) = no prior art AND mechanism feasible | **Promote R269 to Super-GOAT on the PRIMARY fusion.** Mandatory outputs due in the follow-up session: (1) open `StageGatedHlaSubspace` primitive in `katgpt-rs/crates/katgpt-core/src/sense/` (extends `NpcFunctorRuntime` with decision-stage axis); (2) private `riir-ai/.research/NNN_stage_gated_hla_subspace_guide.md` with validation protocol G1–Gn; (3) plans: katgpt-rs (modelless subspace selector), riir-ai (wire `on_decision_context()` hook + LatCal commitment of per-stage profile), optional riir-train (if profile is learned not authored). |
| Q1.a (PRIMARY) has prior art BUT Q1.a' (SECONDARY) is novel | **Downgrade PRIMARY to Gain, promote SECONDARY to GOAT.** Plan the adapter-routing fusion behind `shape_adaptive_router` feature flag. Benchmark vs vanilla adapter routing. |
| Both have prior art | **Downgrade R269 to Gain.** Plan-only, feature-flagged, low priority. Close this issue. |

## Tasks

- [x] **T1** Run the six arxiv keyword searches above; tabulate hits with one-line relevance assessment each. *(3 of 12 returned 0 exact-phrase hits — itself a weak novelty signal; the remaining 9 timed out or 429'd at the arxiv search UI and were re-run via broader web search. Tabulation + relevance in R269 §3 citation table. Closest hits: VSG [2210.11698] for PRIMARY; FIM-LoRA [2605.16800], ALoRA [2403.16187], MoLA, LayerRoute [2606.01838], LoRA-Switch [2405.17741], LoRA-Drop [2601.02569] for SECONDARY.)*
- [x] **T2** Read top 3 closest papers from T1 in full (via `https://r.jina.ai/https://arxiv.org/pdf/{ID}`). *(Full reads: VSG [2210.11698], LayerRoute [2606.01838], FIM-LoRA [2605.16800]. Assessments in R269 §3.)*
- [x] **T3** Grep `HydraBudgetConfig` call sites; confirm no adapter-aware variant exists. *(Confirmed: struct at `crates/katgpt-core/src/types.rs:4203` has only `{ skip_threshold, cumulative_threshold, modelless: bool, skip_erasure_draft: bool }`; `modelless` means "lookup vs logit-lens," NOT "base vs adapter." Call sites: `src/pruners/hydra_budget.rs:11,95`, `tests/bench_165_hydra_budget_goat.rs`. No adapter-aware variant anywhere.)*
- [x] **T4** Read `SnapshotMeta` serialization; confirm forward-compat. *(Cross-repo file IS accessible from this workspace: `riir-ai/crates/riir-engine/src/snapshot.rs:300-310`. Finding: `SnapshotMeta` has NO `#[serde(default)]` on any field — NOT forward-compatible as-is. R269 §5 assumption was wrong. Any extension requires a migration prerequisite.)*
- [x] **T5** Write Q1.a–Q1.d verdict into R269 §3 and close this issue with the resolution action. *(Done — R269 §3 rewritten with PRIMARY/SECONDARY/Q1.c/Q1.d verdicts + citation table; resolution action = downgrade to Gain per table row 3.)*

## Estimated effort

T1+T3+T4: ~30 min. T2: ~1 hr. T5: ~15 min. Total: ~2 hr.
