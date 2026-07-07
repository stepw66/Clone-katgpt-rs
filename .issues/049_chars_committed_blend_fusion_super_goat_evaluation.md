# Issue 049: CHaRS × CommittedFieldBlend × latent_functor — Per-NPC Archetype-Routing Steering Super-GOAT Evaluation

> **Opened:** 2026-07-07
> **Source research:** [katgpt-rs/.research/389_CHaRS_Cluster_Aware_Representation_Steering.md](../.research/389_CHaRS_Cluster_Aware_Representation_Steering.md) §2.4 F1
> **Status:** Open — Super-GOAT candidate, NOT committed. Needs full Q1–Q4 evaluation before any guide/plan.
> **Owner:** unassigned
> **Verdict tier at opening:** GOAT (the bare CHaRS primitive). F1 fusion is the unevaluated Super-GOAT candidate.

---

## Context

Research 389 (CHaRS, arXiv:2603.02237) verdict was **GOAT** for the bare primitive. The F1 fusion candidate — **CHaRS × CommittedFieldBlend × latent_functor re-estimation = per-NPC archetype-routing steering that adapts when the NPC's latent region shifts** — was flagged as a Super-GOAT *candidate* but **NOT committed** in that session. Per §1.5 of the research skill, "candidate" is not a deferred-commitment escape hatch: it either triggers the full mandatory outputs (open primitive + private guide + plan) NOW, or it gets downgraded to an issue for full evaluation. This is the issue.

## The candidate fusion

**Mechanism:** CommittedFieldBlend (Plan 321, R302) commits an NPC's personality as a fixed K=3 archetype blend `π`, computed once from a trajectory summary. CHaRS gives per-input soft routing over a K-anchor bank via RBF(x, anchor) × OT-coupling. Fuse: the anchor bank `{a_i, b_j, v_ij, P⋆, σ}` IS the NPC's committed archetype library (the K frozen operator fields), and CHaRS computes the per-tick routing weight from the NPC's *current HLA position*. When the NPC drifts into a new latent region (e.g., social → combat), the RBF gate shifts weight to the combat archetype's translation automatically. `ReestimationScheduler` (latent_functor/reestimation.rs) triggers bank re-commit on coherence drop.

**Hypothesized selling point:** *"Our NPCs are steered by their current affective region, not a single global personality vector — a wolf in hunt-mode is steered by the hunt archetype's translation, the same wolf in pack-mode by the social archetype's, with smooth sigmoid-gated transitions as its HLA state moves between regions. No competitor ships per-input region-aware steering at crowd scale."*

## Q1–Q4 evaluation checklist (must complete before any Super-GOAT commitment)

### Q1: No prior art? — UNEVALUATED

Must verify across all 5 repos, both layers (notes+plans+docs AND src+crates), with vocabulary translation:

- [ ] **Q1.1** Grep `CommittedFieldBlend|apply_blended|ArchetypeFieldSource` consumers — is anyone already doing per-input routing on top of the committed blend?
- [ ] **Q1.2** Grep `chars|cluster.*steer|input.position.*steer|region.aware.*steer|per.tick.*archetype` — any partial implementation?
- [ ] **Q1.3** Grep `ReestimationScheduler|reestimation_trigger|coherence.*tau` — does the scheduler already do region-aware steering implicitly?
- [ ] **Q1.4** Grep `latent_functor.*zone_gating|zone_gating.*steer` — does zone gating already provide region-aware routing that CHaRS would duplicate?
- [ ] **Q1.5** Read `riir-engine/src/committed_blend/` and `riir-engine/src/latent_functor/reestimation.rs` — what's the actual integration surface?
- [ ] **Q1.6** Read `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md` — does the committed-personality story already include region-aware routing?

**Q1 verdict:** ___ (YES/NO with evidence)

### Q2: New class of behavior? — UNEVALUATED

- [ ] **Q2.1** Is "per-NPC region-aware steering" a new capability class, or a refinement of CommittedFieldBlend's "per-NPC committed archetype blend"?
- [ ] **Q2.2** Compare against the bar R382 Spherical Steering was measured against — Plan 322 established "norm-preserving rotation as a new latent operation class"; does CHaRS × CommittedFieldBlend establish an analogous new operation class, or is it a new routing criterion on the existing soft-MoE-steering class?
- [ ] **Q2.3** Identify the closest shipped cousin (likely CommittedFieldBlend + `ReestimationScheduler`) — does the combination produce a behavior neither has alone?

**Q2 verdict:** ___

### Q3: Product selling point? — UNEVALUATED

- [ ] **Q3.1** Finish the sentence: "Our NPCs do X that no competitor can". Is X concrete and demoable?
- [ ] **Q3.2** Does this selling point already exist in `riir-ai/.docs/pillars/` or `supergoat_candidates/`? (Must `read_file` both indexes first.)
- [ ] **Q3.3** Is the wolf hunt-mode vs pack-mode example actually a *new* behavior, or is it what `PersonalityWeightedComposition`'s per-layer drift + CommittedFieldBlend's commitment already produce?

**Q3 verdict:** ___

### Q4: Force multiplier? — UNEVALUATED

- [ ] **Q4.1** Count pillars touched. Must be ≥2 for Super-GOAT. Candidates: P2 (neuron-db substrate), P8 (reasoning pack), self-learn NPCs, latent_functor, HLA, EmotionDirections, freeze/thaw.
- [ ] **Q4.2** Are the connections *new capabilities* (force multiplier), or refinements of existing connections?

**Q4 verdict:** ___

## If Q1–Q4 all YES: mandatory outputs (in the evaluation session, not now)

1. **Open primitive** → `katgpt-rs/.plans/409_chars_cluster_aware_steering_primitive.md` (the bare primitive ships regardless — Plan 409 is in R389's routing already).
2. **Private guide** → `riir-ai/.research/NNN_per_npc_archetype_routing_steering_guide.md` (selling point: per-NPC region-aware steering).
3. **Private plan** → `riir-ai/.plans/NNN_chars_committed_blend_runtime_integration.md` (runtime wiring: HLA hook, latent_functor interop, archetype library loader, CharsAnchorShard freeze integration).
4. **Cross-ref guides** → riir-neuron-db (CharsAnchorShard layout), riir-chain (LatCal commitment of OT plan) — if those fusions are pursued.

## If any Q is NO: close this issue

- Update R389 §2.4 F1 to "evaluated, not Super-GOAT, reason: ___".
- The bare primitive (Plan 409) ships regardless as a GOAT.

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
