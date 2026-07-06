# Issue 039: Plan 405 F1 Fusion — Q1–Q4 Novelty Gate FAILED (not Super-GOAT)

**Date:** 2026-07-06
**Status:** Closed — verdict recorded, no action
**Origin:** Plan 405 §Phase 5 T5.1 (deferred fusion evaluation)
**Parent research:** [katgpt-rs/.research/382_Spherical_Steering_Geodesic_Slerp.md](../.research/382_Spherical_Steering_Geodesic_Slerp.md) §2.4 F1

---

## TL;DR

The F1 fusion candidate from Research 382 §2.4 — **Slerp (Plan 405) × CommittedFieldBlend (Plan 321) × HLA divergence detection (vMF gate) = "personality drift auto-correction"** — was evaluated against the Q1–Q4 novelty gate (skill `research` §1.5) and **fails on Q1, Q2, Q3** (Q4 partial). **Verdict: not Super-GOAT. No guide, no plan, no primitive.** The selling point is already covered by shipped prior art, and the fusion's core premise (personality drift is a failure mode to auto-correct) contradicts the existing NPC-cognition design philosophy.

---

## The F1 candidate (recap)

> `CommittedFieldBlend` (Plan 321) commits an NPC's personality as a BLAKE3-committed blend of K=3 archetype direction vectors. The committed blend defines the NPC's "home" on the affect manifold. At runtime, if the NPC's HLA state drifts away from its committed home (measured by the vMF confidence gate — `s_t = μ_home · ĥ_hla` falling below threshold), Slerp-steer it back toward `μ_home` with strength proportional to drift. **Claimed novel capability:** NPCs that auto-correct personality drift at runtime without re-training — emotion regulation by construction.
> — Research 382 §2.4 F1

## Q1–Q4 verdict

### Q1 — No prior art? **NO (heavily covered)**

Three-layer check (notes + code + vocabulary translation):

| F1 element | Prior art (shipped) | Where |
|---|---|---|
| "Stable long-horizon affect via norm-preserving HLA rotation" selling point | **R159 / Plan 322 (Phase-Rotation Subspace Gate)** — TL;DR: *"Our NPCs rotate their affect smoothly between combat and social subspaces over thousands of ticks without magnitude drift — emotional stability by construction"* | `riir-ai/.research/159_*.md` (Super-GOAT guide, the cousin primitive's private selling-point doc) |
| "Detect divergence → apply correction" loop architecture | `latent_functor::reestimation::ReestimationScheduler` ("coherence < tau_reest → re-estimate") + CCE twin `cce_runtime::reestimation_trigger` | `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs`, `riir-ai/crates/riir-engine/src/cce_runtime/reestimation_trigger.rs` |
| "Steer toward committed archetype direction" | `CommittedFieldBlend::apply_blended` already produces `f_π(z) = Σ_k sigmoid(π_k/τ) · f_k(z)` — the committed-blend-shaped dynamics update, every tick | `katgpt-rs/crates/katgpt-core/src/committed_field_blend.rs:295` |
| "Drift detector" (vMF gate as trigger) | `coherence < tau_reest` is the existing drift-detector signal in two schedulers | same reestimation files |
| "Personality drift is a failure mode" premise | **Contradicted** by R146 (drift IS the personality, bounded by clamps + EMA + freeze/thaw), R158 (committed personality doesn't drift by design), R311 (cross-node drift is "correct emergent behavior"), R151 (explicitly distinguishes magnitude drift [bug] from personality drift [behavior]) | `riir-ai/.research/{146,158,311,151}_*.md` |

**Vocabulary translation performed** (mandatory per skill §1.5):
- "auto-correction" → "drift correction loop" → **`reestimation`** ✓ (hit)
- "stable affect" → "norm-preserving affect rotation" → **`phase_rotation` / Plan 322** ✓ (hit)
- "committed archetype direction" → **`CommittedFieldBlend` / `apply_blended`** ✓ (hit)
- "drift detector" → "coherence threshold" → **`tau_reest`** ✓ (hit)

### Q2 — New class of behavior? **NO**

"Detect HLA drift → norm-preserving rotation toward a target" is the **same operation class** as R159 (phase rotation for stable affect) and the **same architecture** as reestimation (coherence-triggered correction). Different math (Slerp vs phase rotation; vMF gate vs coherence threshold), same capability.

### Q3 — Product selling point? **WEAK / DUPLICATED**

"Our NPCs auto-correct personality drift toward a committed archetype" fails the sentence-completion test against existing selling points:
- CommittedFieldBlend's committed personality *doesn't drift* (R158 thesis) — "auto-correct drift" is correcting a failure mode that the commitment was designed to prevent.
- PersonalityWeightedComposition's drift *is the personality* (R146 thesis) — auto-correcting it would destroy the emergent character that is the system's core selling point.
- R159 already sells "stable long-horizon affect."

The only non-duplicated slice is "vMF-gated Slerp toward the blend's implicit home direction when HLA cosine drops" — a refinement of three existing mechanisms, not a new selling point.

### Q4 — Force multiplier? **PARTIAL (already wired)**

The pillars F1 connects (CommittedFieldBlend, HLA, Plan 322 cousin, reestimation loop) are **already connected** by R159's connection map (8 rows) and by CommittedFieldBlend's own connection map (R158, 9 rows). F1 doesn't add new pillar connections; it adds a third rotation variant (Slerp) to a connection map that already has phase rotation (R159) and additive steering (Plan 309).

---

## Verdict

**Not Super-GOAT. Fails Q1, Q2, Q3; Q4 partial.** Per skill §1.5, this triggers no Super-GOAT outputs (no open primitive, no private guide, no plan). The fusion idea is **closed as not-novel** — recorded here to prevent re-evaluation from scratch.

## What this does NOT change

- **Plan 405 (Slerp primitive) stays DEFAULT-ON** in katgpt-rs. The primitive itself passed its own GOAT gate (G1–G6, commit `86e3b915`); this issue is only about the *fusion* with CommittedFieldBlend + HLA divergence, not the primitive.
- **The Slerp primitive remains available** for any consumer that needs single-target geodesic rotation toward a unit-norm vector. The F1 fusion was one hypothesized consumer; it not being Super-GOAT doesn't demote the primitive.
- **F2 fusion** (Slerp × Plan 322 phase rotation = "rotate-within-and-toward") from Research 382 §2.4 is **untouched** by this verdict — it's a different fusion candidate not evaluated here. If F2 is ever pursued, it needs its own Q1–Q4 gate.

## Re-evaluation triggers

Re-open this issue only if:
1. A new primitive ships that makes the "drift is correctable toward committed home" premise non-contradictory (e.g., a *transient* HLA excursion mode distinct from the *permanent* personality-drift mode that R146/R158 govern).
2. Empirical evidence emerges that R159's phase rotation fails to stabilize HLA on a workload where Slerp-toward-archetype succeeds (then F1 becomes a per-stack demotion candidate against R159, not a Super-GOAT).

---

## TL;DR

F1 fusion (Slerp × CommittedFieldBlend × HLA divergence = "personality drift auto-correction") fails the Q1–Q4 novelty gate. The selling point is already shipped as R159 (Phase-Rotation Subspace Gate, Super-GOAT), the detect-then-correct loop architecture is already shipped as `reestimation`, and the core premise (personality drift is a failure mode) contradicts R146/R158/R311 where drift is intentional behavior or committed-away by design. **No Super-GOAT outputs. Plan 405's primitive stays DEFAULT-ON. Closed.**
