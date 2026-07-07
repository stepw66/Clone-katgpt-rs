# Issue 049: CHaRS × CommittedFieldBlend × latent_functor — Per-NPC Archetype-Routing Steering Super-GOAT Evaluation

> **Opened:** 2026-07-07
> **Source research:** [katgpt-rs/.research/389_CHaRS_Cluster_Aware_Representation_Steering.md](../.research/389_CHaRS_Cluster_Aware_Representation_Steering.md) §2.4 F1
> **Status:** ✅ CLOSED (2026-07-07) — NOT Super-GOAT. Full Q1–Q4 evaluation complete. Q2 fails (refinement of soft-MoE-steering class, not new operation); Q3 weak (refines R158's committed-personality selling point). The bare CHaRS primitive (Plan 409) ships as GOAT regardless. R389 §2.4 F1 updated.
> **Owner:** unassigned
> **Verdict tier at opening:** GOAT (the bare CHaRS primitive). F1 fusion is the unevaluated Super-GOAT candidate.
> **Final verdict:** GOAT (confirmed). F1 fusion = NOT Super-GOAT.

---

## Context

Research 389 (CHaRS, arXiv:2603.02237) verdict was **GOAT** for the bare primitive. The F1 fusion candidate — **CHaRS × CommittedFieldBlend × latent_functor re-estimation = per-NPC archetype-routing steering that adapts when the NPC's latent region shifts** — was flagged as a Super-GOAT *candidate* but **NOT committed** in that session. Per §1.5 of the research skill, "candidate" is not a deferred-commitment escape hatch: it either triggers the full mandatory outputs (open primitive + private guide + plan) NOW, or it gets downgraded to an issue for full evaluation. This is the issue.

## The candidate fusion

**Mechanism:** CommittedFieldBlend (Plan 321, R302) commits an NPC's personality as a fixed K=3 archetype blend `π`, computed once from a trajectory summary. CHaRS gives per-input soft routing over a K-anchor bank via RBF(x, anchor) × OT-coupling. Fuse: the anchor bank `{a_i, b_j, v_ij, P⋆, σ}` IS the NPC's committed archetype library (the K frozen operator fields), and CHaRS computes the per-tick routing weight from the NPC's *current HLA position*. When the NPC drifts into a new latent region (e.g., social → combat), the RBF gate shifts weight to the combat archetype's translation automatically. `ReestimationScheduler` (latent_functor/reestimation.rs) triggers bank re-commit on coherence drop.

**Hypothesized selling point:** *"Our NPCs are steered by their current affective region, not a single global personality vector — a wolf in hunt-mode is steered by the hunt archetype's translation, the same wolf in pack-mode by the social archetype's, with smooth sigmoid-gated transitions as its HLA state moves between regions. No competitor ships per-input region-aware steering at crowd scale."*

## Q1–Q4 evaluation checklist (must complete before any Super-GOAT commitment)

### Q1: No prior art? — **YES (genuinely unshipped)**

Must verify across all 5 repos, both layers (notes+plans+docs AND src+crates), with vocabulary translation:

- [x] **Q1.1** Grep `CommittedFieldBlend|apply_blended|ArchetypeFieldSource` consumers — is anyone already doing per-input routing on top of the committed blend? **NO.** `CommittedFieldBlend` ships in `katgpt-core/src/committed_field_blend/` with `apply_blended()` as the hot path. The runtime bridge (`committed_blend/functor_bridge.rs`) confirms: "The blend multiplies the functor — it does not replace it" with `sigmoid(pi_k/tau)` as a **FIXED** weight. `apply_blended_functor(...) = sum_k sigmoid(pi_k/tau) · g_k(e_target)` — the `pi` vector is committed ONCE per NPC and never varies per-input. The bench + example use a fixed blend per entity.
- [x] **Q1.2** Grep `chars|cluster.*steer|input.position.*steer|region.awware.*steer|per.tick.*archetype` — any partial implementation? **NO.** Zero hits for CHaRS-specific vocabulary. No partial per-input archetype routing exists.
- [x] **Q1.3** Grep `ReestimationScheduler|reestimation_trigger|coherence.*tau` — does the scheduler already do region-aware steering implicitly? **NO — this is the load-bearing check, and it PASSES (not covered).** The `ReestimationScheduler` (`latent_functor/reestimation.rs`) does **periodic re-estimation** of direction vectors when `coherence < tau_reest`. It fires at most `max_reestimations_per_tick=4` times per tick per NPC (warm tier, budget-capped). It re-fits the GLOBAL direction vector from recent observations — it does NOT compute per-input routing weights. This is a **slow-timescale correction** (re-fit on drift), not **per-tick per-input routing** (route by current HLA position). The detect-then-correct loop is implicit; the detect-then-ROUTE-per-input loop is NOT. CHaRS's `w_i(x)` is computed EVERY tick from the current HLA state — a fundamentally different timescale and mechanism.
- [x] **Q1.4** Grep `latent_functor.*zone_gating|zone_gating.*steer` — does zone gating already provide region-aware routing that CHaRS would duplicate? **NO (partial overlap, different axis).** `zone_gating.rs` ships zone-density dynamic gating: it adjusts `(tau, beta, reest_budget)` by zone **interaction density** (a SPATIAL/raw scalar `I_d`, synced via SyncBlock). This gates the **trust threshold** for functor coherence, not the **archetype blend weights**. It answers "how much should we trust the functor in this zone?", not "which archetype should dominate this NPC's steering given its current affective region?". Different axis (spatial density vs affective HLA position), different target (trust gate vs blend weights).
- [x] **Q1.5** Read `riir-engine/src/committed_blend/` and `riir-engine/src/latent_functor/reestimation.rs` — what's the actual integration surface? **DONE.** The `committed_blend/` runtime has 9 files: `mod.rs` (HLA dynamics blend), `functor_bridge.rs` (relational stance blend, D=32), `karc_bridge.rs` (Bi-NCDE forecast), `curiosity_bridge.rs`, `recommit.rs`, `freeze.rs`, `shard_view.rs`, `chain_bridge.rs`, `archetypes.rs`. The integration surface for CHaRS would be `functor_bridge.rs` (replace fixed `sigmoid(pi_k/tau)` with per-input `w_k(x)` from RBF on current HLA) + a new `chars_anchor_bank` field in the committed state. The `reestimation.rs` scheduler would trigger bank re-commit on coherence drop (already the pattern it uses for direction re-estimation).
- [x] **Q1.6** Read `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md` — does the committed-personality story already include region-aware routing? **NO.** R158's selling point is "committed personalities that survive observation gaps" via a FIXED `pi` computed once from trajectory summary. §2.1 explicitly: "Compute blend weights ONCE: `pi_k = sigmoid(dot(s, dir_k) / tau)` ... Freeze `pi` — never mutate until a major personality event." Per-tick evolution uses `evolve_hla_pi(z) = sum_k pi_k · evolve_hla_k(z)` with CONSTANT `pi_k`. No region-aware routing.

**Q1 verdict: YES (genuinely unshipped).** Per-input RBF routing over the committed archetype library, computed from the NPC's current HLA position every tick, is not shipped in any of the 5 repos. The closest cousin (CommittedFieldBlend) is entity-fixed; the scheduler is periodic re-fit; zone-gating is spatial-density trust adjustment. The fusion IS novel at the mechanism level.

### Q2: New class of behavior? — **NO (PARTIAL — fails the Super-GOAT bar)**

- [x] **Q2.1** Is "per-NPC region-aware steering" a new capability class, or a refinement of CommittedFieldBlend's "per-NPC committed archetype blend"? **REFINEMENT.** The OPERATION is identical: `blend K archetype translations by soft weights` → `sum_k w_k · f_k(z)`. CommittedFieldBlend uses `w_k = sigmoid(pi_k/tau)` (fixed per NPC); CHaRS fusion uses `w_k = RBF(hla, anchor_k)` (per-input). The **weight source** changes; the **operation class** does not.
- [x] **Q2.2** Compare against the bar R382 Spherical Steering was measured against — Plan 322 established "norm-preserving rotation as a new latent operation class"; does CHaRS × CommittedFieldBlend establish an analogous new operation class, or is it a new routing criterion on the existing soft-MoE-steering class? **NEW ROUTING CRITERION, not new operation class.** R382 (Spherical Steering) was GOAT not Super-GOAT for the same reason: its norm-preserving Slerp was a refinement of Plan 322's rotation class, not a new class. CHaRS's per-input routing is a refinement of CommittedFieldBlend's soft-MoE-steering class. The R382 precedent applies directly.
- [x] **Q2.3** Identify the closest shipped cousin (likely CommittedFieldBlend + `ReestimationScheduler`) — does the combination produce a behavior neither has alone? **YES at the individual-NPC level (a single NPC's steering adapts to its current affective region), but NOT at the operation-class level.** The combination produces richer per-NPC behavior, but the operation is still "blend K fields by soft weights" — the same class CommittedFieldBlend already ships.

**Q2 verdict: NO (PARTIAL).** Fails the "new capability class" bar. It's a new routing criterion on the existing soft-MoE-steering class. **This is the failing criterion for Super-GOAT.**

### Q3: Product selling point? — **WEAK (refines R158)**

- [x] **Q3.1** Finish the sentence: "Our NPCs do X that no competitor can". Is X concrete and demoable? **PARTIALLY.** "Our NPCs are steered by their current affective region — a wolf in hunt-mode is steered by the hunt archetype's translation, the same wolf in pack-mode by the social archetype's, with smooth sigmoid-gated transitions as its HLA state moves between regions." Concrete and demoable, but...
- [x] **Q3.2** Does this selling point already exist in `riir-ai/.docs/pillars/` or `supergoat_candidates/`? **YES (overlaps Pillar 8 / R158).** Pillar 8 (Reasoning Pack) covers "personality-driven NPC reasoning at scale." R158's existing selling point is "committed personalities that survive observation gaps... emergent per-NPC personality at crowd scale." The CHaRS fusion REFINES this: R158 already sells "emergent per-NPC personality via blend variation" (different NPCs have different `pi`); CHaRS adds "a SINGLE NPC's blend varies over time by affective region." The product-level selling point ("emergent personality at crowd scale") is already covered.
- [x] **Q3.3** Is the wolf hunt-mode vs pack-mode example actually a *new* behavior, or is it what `PersonalityWeightedComposition`'s per-layer drift + CommittedFieldBlend's commitment already produce? **It IS new at the individual-NPC timescale (no shipped primitive makes a single NPC's blend adapt per-tick to its HLA position), but the crowd-scale product story is already sold by R158.** The gap is: R158's crowd-scale emergence comes from NPC-to-NPC `pi` variation; CHaRS would add within-NPC temporal variation. Both are "emergent personality" — different axes of the same selling point.

**Q3 verdict: WEAK.** The selling point refines R158's committed-personality story. It adds within-NPC temporal adaptivity to the existing cross-NPC blend variation. Concrete and demoable at the individual level, but the product moat sentence is already owned by Pillar 8 / R158.

### Q4: Force multiplier? — **YES**

- [x] **Q4.1** Count pillars touched. **YES (≥5).** HLA (latent substrate), CommittedFieldBlend/Pillar 8 (archetype library cousin), latent_functor (ReestimationScheduler as re-commit trigger), EmotionDirections (anchor-bank construction recipe), freeze/thaw (CharsAnchorShard). Plus LatCal (F3 commitment) and ItemEmbedIndex (F4 steered retrieval) as speculative connections.
- [x] **Q4.2** Are the connections *new capabilities* (force multiplier), or refinements of existing connections? **MIXED.** The HLA→CHaRS routing connection is new (per-input affective-region routing is unshipped). The CommittedFieldBlend connection is a refinement (new weight source on existing operation). The ReestimationScheduler connection is new (bank re-commit trigger, not just direction re-fit).

**Q4 verdict: YES.** Strong force multiplier — ≥5 pillars touched, with genuinely new connections on the HLA routing + scheduler re-commit axes.

## Final verdict: **NOT Super-GOAT**

| Q | Criterion | Verdict | Evidence |
|---|---|---|---|
| Q1 | No prior art? | **YES** | Per-input RBF routing over committed archetype library is genuinely unshipped. CommittedFieldBlend's `pi` is fixed (functor_bridge.rs); scheduler is periodic re-fit; zone-gating is spatial-density trust adjustment. |
| Q2 | New class of behavior? | **NO (PARTIAL)** | Operation "blend K fields by soft weights" is the CommittedFieldBlend class. CHaRS changes the weight source (fixed `pi` → per-input `w_k(x)`), which is a new routing criterion, not a new operation. R382 precedent applies. **Failing criterion.** |
| Q3 | Product selling point? | **WEAK** | Refines R158's committed-personality story (Pillar 8). Adds within-NPC temporal adaptivity to existing cross-NPC blend variation. Crowd-scale product sentence already owned by R158. |
| Q4 | Force multiplier? | **YES** | ≥5 pillars touched (HLA, CommittedFieldBlend, latent_functor, EmotionDirections, freeze/thaw). Genuinely new connections on HLA routing + scheduler re-commit axes. |

**Super-GOAT requires YES on ALL four. Q2 fails, Q3 is weak. Verdict: NOT Super-GOAT.**

This confirms R389 §4's own GOAT verdict for the bare primitive. The F1 fusion (CHaRS × CommittedFieldBlend × latent_functor re-estimation) does NOT elevate to Super-GOAT even with the full composition — it remains a strong GOAT-tier refinement of the committed-personality moat.

## If Q1–Q4 all YES: mandatory outputs (in the evaluation session, not now)

**N/A — Q2 fails.** No Super-GOAT commitment. The mandatory outputs do not apply.

## If any Q is NO: close this issue

- [x] **DONE** — Update R389 §2.4 F1 to "evaluated, not Super-GOAT, reason: Q2 fails (refinement of soft-MoE-steering class, not new operation; R382 precedent); Q3 weak (refines R158 committed-personality selling point)".
- [x] The bare primitive (Plan 409) ships regardless as a GOAT. (Plan 409 number was taken by `409_jlens_concept_readout_prefilter_poc.md`; the CHaRS primitive needs the next free plan number if/when implemented.)

## Pre-evaluation reading list (mandatory before Q1)

- [ ] `katgpt-rs/.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md` — CommittedFieldBlend Super-GOAT precedent (Q1–Q4 evidence matrix)
- [ ] `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md` — existing committed-personality selling point
- [ ] `riir-ai/.research/153_latent_field_steering_game_runtime_guide.md` — existing steering selling point
- [ ] `riir-ai/.docs/pillars/README.md` + `riir-ai/.docs/supergoat_candidates/README.md` — moat map (Q3)
- [ ] `riir-ai/crates/riir-engine/src/committed_blend/` — actual CommittedFieldBlend runtime
- [ ] `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` — actual scheduler
- [ ] `katgpt-rs/.research/382_Spherical_Steering_Geodesic_Slerp.md` — the GOAT-not-Super-GOAT precedent for Q2

## Defend-wrong PoC (§3.6, if Super-GOAT candidate promotes)

If Q1–Q4 all YES, the parity claim "per-NPC region-aware steering matches or beats naive global steering on a heterogeneous-cluster toy benchmark" MUST be defended in `riir-ai/crates/riir-poc/` before the guide is treated as canonical. Three competitors minimum:

1. CHaRS × CommittedFieldBlend (the fusion)
2. CommittedFieldBlend alone (no per-input routing — committed π fixed)
3. Plan 309 Latent Field Steering (single global direction, no archetypes)

Toy benchmark: synthetic HLA trajectories with planted region heterogeneity (3 distinct affect regions, each requiring a different steering direction). Measure: steering accuracy per region, latency per tick at crowd scale (1000 NPCs), sampling invariance under fog-of-war gaps.

## Risks / known traps

1. **CommittedFieldBlend's contract is "commit once, never mutate".** Adding per-input routing may violate this contract. The fusion must specify whether the routing weight `w_i(x)` is a *read* (computed per-input from frozen bank + current HLA, never persisted) or a *write* (updates the committed π). Read is safe; write violates FAME Prop. 3 sampling invariance.

2. **Compositional ordering is non-commutative.** `π-blend then CHaRS-steer` ≠ `CHaRS-steer then π-blend`. Must pick one and verify it preserves the properties of both.

3. **CommittedFieldBlend's π is K=3 floats; CHaRS's routing weight is K floats per input.** If both are committed, the per-NPC sync artifact doubles. May not be worth the cost.

4. **The R382 precedent is a real risk.** Spherical Steering's F1 fusion (Slerp × CommittedFieldBlend × HLA divergence = "personality drift auto-correction") was evaluated (Issue 039, 2026-07-06) and REJECTED as not-Super-GOAT on Q1 (heavily covered: "stable long-horizon affect" is R159 / Plan 322; detect-then-correct loop is shipped `ReestimationScheduler`) and Q3 (weak/duplicated selling point — CommittedFieldBlend doesn't drift by design, PersonalityWeightedComposition's drift IS the personality). **CHaRS × CommittedFieldBlend may hit the same Q1 trap** — the detect-then-route pattern may be implicit in the shipped scheduler. Q1.3 above is the load-bearing check.
