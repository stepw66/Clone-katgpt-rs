# Issue 004: Per-NPC Selective Forgetting on Personality Swap — Super-GOAT Fusion Validation

> **Origin:** Research 320 (Red Queen Gödel Machine — Selective Erasure & Best-Belief Selection, arXiv:2606.26294)
> **Date:** 2026-06-28
> **Status:** Open — novelty-gate validation pending
> **Blocks:** Promotion of Plan 336 (`controlled_utility` feature) beyond GOAT-tier open primitive; potential Super-GOAT guide in `riir-ai/.research/`

---

## Context

Research 320 distilled RQGM's training-side consistency contract into two modelless inference primitives (`CriterionVersionedRecords` with selective erasure + `BestBeliefSelector` ε-quantile Beta). The verdict was **GOAT** — provable unification of scattered shipped instances (Plan 279 Gram cache invalidation, Plan 315 cascade `invalidate_zone_on_collapse`, Issue 001 HLA eigenbasis BLAKE3-check).

The Super-GOAT angle (§2.3 of Research 320) is **not committed**: it requires all four novelty-gate questions to be answered YES with confidence, and the author of Research 320 was not confident enough on Q1 (no prior art) and Q3 (selling point depends on riir-train) to commit during the research session. Per the research skill rule ("If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'. Write 'fusion idea — novelty TBD' and create an issue"), this issue tracks the follow-up.

## The Candidate Selling Point

> Our NPCs co-evolve their personality direction vectors and their theory-of-mind models of other NPCs. When an NPC's snapshot is swapped (personality divergence, tame event, faction change, trauma), only the memories that depended on the OLD personality are erased — position, HP, wallet balance (raw, synced) survive bit-identical; affect projections, KG triples, cached policies (latent, local) are selectively invalidated. This enables emergent social dynamics where personality drift triggers targeted forgetting, not full amnesia — at MMO scale (thousands of NPCs, 20Hz tick).

## Open Questions to Resolve Before Verdict

### Q1 — No prior art? (NEEDS DEEPER CODE AUDIT)

Research 320 grepped `.research/` + `.plans/` + surface-level code. The unified `CriterionVersionedRecords` primitive does not appear as a named abstraction, BUT the following modules need a deeper read before claiming "no prior art" with confidence:

- [ ] `riir-ai/crates/riir-engine/src/policy_cache/` — does the cascade invalidation already generalize to multi-slot dep tracking?
- [ ] `riir-ai/crates/riir-engine/src/adapters/` — does `LoRAHotSwap` / `dispatch_lora_merge` already invalidate dependent cached projections?
- [ ] `riir-neuron-db/src/consolidation.rs` — does Raven/δ-Mem already implement selective erasure on consolidation boundaries?
- [ ] `riir-neuron-db/src/mape_k.rs` — does the MAPE-K loop already tag records with criterion versions?
- [ ] `riir-ai/crates/riir-engine/src/committed_blend/` — does `CommittedFieldBlend` already track per-slot dependency?

**Resolution rule:** if any of the above already ships a generic multi-slot dep-tracked record store, this fusion is **not novel** → downgrade to GOAT (the unification is still valuable, just not Super-GOAT).

### Q2 — New class of behavior? (PLAUSIBLE BUT UNVERIFIED)

The *mechanism* (dependency tracking + cache invalidation) is well-known CS. The novelty claim is in the *application* (per-NPC selective forgetting at MMO scale) + the *latent-space reframing* (evaluator = HLA affect direction vector) + the *constraint* (raw/latent sync boundary, deterministic replay, anti-cheat).

- [ ] Identify a concrete competitor / prior art that does per-NPC selective memory erasure on personality swap at MMO scale. If none found → strengthens Q2.
- [ ] Verify the raw/latent boundary holds: position/HP/wallet MUST survive personality swap bit-identical; only affect/KG/policy are selectively erased.

### Q3 — Product selling point? (DEPENDS ON RIIR-TRAIN)

The modelless side (Plan 336) handles *consistency* only. The actual personality-direction co-evolution requires the riir-train side to produce evolving evaluators (per-NPC HLA direction vectors that drift over time, KG triple templates that specialize, quest graders that co-evolve with quest generators).

- [ ] Confirm riir-train has a path to produce per-NPC co-evolving personality direction vectors. If not → the selling point is aspirational, not finishable → downgrade.
- [ ] Alternatively, scope the selling point to a *deterministic* direction-vector swap (e.g., tame event → COMPANIONS direction flips sign, no training needed) and check whether that alone is a strong enough selling point.

### Q4 — Force multiplier? (STRONG — ≥5 PILLARS)

Already documented in Research 320 §2.3. HLA, latent_functor, NeuronShard, MAPE-K, KG triples, Plan 315, Plan 279 all touched. No further validation needed unless Q1/Q2/Q3 downgrade.

## Decision Protocol

```
After Plan 336 ships (Phase 1 GOAT gate passes):
  → Re-audit Q1 (deeper code read of the 5 modules above)
    → If prior art found: downgrade to GOAT, close this issue as "resolved — not Super-GOAT".
    → If no prior art found: proceed to Q2.
  → Re-audit Q2 (competitor scan + raw/latent boundary verification)
    → If competitor exists or boundary breaks: downgrade to GOAT, close.
    → If novel + boundary holds: proceed to Q3.
  → Re-audit Q3 (riir-train path or deterministic-swap scoping)
    → If riir-train has no path AND deterministic-swap is too weak: downgrade to GOAT, close.
    → If path exists OR deterministic-swap is strong enough: proceed to verdict.
  → If all four YES: write Super-GOAT guide in riir-ai/.research/, create riir-ai plan.
```

## Non-Goals

- This issue does NOT block Plan 336. The GOAT-tier open primitives (`CriterionVersionedRecords`, `BestBeliefSelector`) ship regardless of the Super-GOAT verdict — they unify scattered instances and that gain holds independently.
- This issue does NOT track the riir-train co-evolution training work. That is a separate concern in `riir-train/.research/` if/when it lands.

## References

- [Research 320](../.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md) — the originating distillation.
- [Plan 336](../.plans/336_controlled_utility_primitives.md) — the GOAT-tier implementation.
- RQGM paper: [arXiv:2606.26294](https://arxiv.org/pdf/2606.26294).
- Research 080 (VPD co-evolution), 021 (G-Zero), 074-riir-ai (NS-RL survey) — prior co-evolution framings.
- Research 098 (PrudentBanker) — closest cousin on safe-phased aggression + lower-bound selection.
