# Research 320: Red Queen Gödel Machine — Selective Erasure & Best-Belief Selection

> **Source:** [The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/pdf/2606.26294) — Iacob, Jovanović, Shen, Burkhardt, Kurmanji, Tastan, Sani, Venanzi, Odonnat, Cao, Marino, Qiu, Lane (Cambridge / Flower Labs / NVIDIA / MBZUAI / Inria), arXiv:2606.26294v1, 24 Jun 2026
> **Date:** 2026-06-28
> **Status:** Active
> **Related Research:** 080 (VPD co-evolution), 021 (G-Zero Proposer×Generator), 074-riir-ai (NS-RL three-mode co-evolution taxonomy), 098 (PrudentBanker safe-phased bandit), 300 (SelfCompact closed-unit compaction gate), 301 (Misalignment Indicator Probe Bank)
> **Related Plans:** 137 (PrudentBanker safe-phased bandit ✅), 279 (Manifold Power Iteration MoE Router — Gram cache invalidation on snapshot bump ✅), 315-riir-ai (Policy Cache Cascade — `invalidate_zone_on_collapse` ✅), 253-riir-neuron-db (Merkle-Octree Curator Consensus), 002-riir-neuron-db (`can_freeze` consolidation gate ✅)
> **Cross-ref (riir-ai / riir-neuron-db):** Issue 004 (katgpt-rs/.issues/) for Super-GOAT fusion follow-up; Plan 336 (katgpt-rs/.plans/) for GOAT implementation; potential riir-ai guide if fusion verdict promotes
> **Classification:** Public

---

## TL;DR

RQGM's headline is recursive self-improvement under **non-stationary utilities** via co-evolved learned evaluators — a *training* contribution that routes to riir-train. But the paper ships **two transferable modelless inference primitives** that we do not have as unified abstractions:

1. **Criterion-versioned record store with selective erasure** — every utility record carries a `dep_set` of evaluator slots whose criterion affected it; when a slot's criterion changes (snapshot swap / adapter hot-swap / direction-vector flip), only records in that slot's dep set are erased, all others remain valid. Prop. 2 of the paper proves this preserves criterion consistency. We have *scattered instances* (Plan 279 Gram cache version bump, Plan 315 cascade `invalidate_zone_on_collapse`, Issue 001 HLA eigenbasis BLAKE3-check on reload) but no unified primitive.
2. **ε-best-belief Beta quantile selector** — `BB_ε(a) = I⁻¹_ε(1 + S_a, 1 + F_a)`, the ε-quantile of the Beta posterior, is a conservative lower bound the candidate's true utility exceeds with probability 1 − ε. Used for *selection* (which evaluator/snapshot to promote), complementing our Thompson sampling (used for *exploration*).

The combination yields a **controlled-utility-evolution freeze/thaw pattern**: epoch-local stationarity (criterion frozen within epoch) + boundary replacement (challenger promoted only if it strictly raises ε-best-belief on an anchor) + selective erasure (only dependent records invalidated). This reframes our existing MAPE-K self-healing + `MerkleFrozenEnvelope` + Raven/δ-Mem consolidation under a single epoch-local-stationarity + boundary-replacement contract.

**Distilled for katgpt-rs (modelless, inference-time):**
- The *training loop* (LLM-based evaluator improvement) → riir-train. Not distilled here.
- The *consistency contract* (selective erasure on criterion change + ε-best-belief promotion gate) → modelless. Ships as two open primitives in `katgpt-core`:
  - `CriterionVersionedRecords` — generic record store with per-slot `dep_set` tracking + `erase_slot(slot)` that removes only dependent records. BLAKE3-tagged criterion versions. Zero-allocation hot path on no-op erasure.
  - `BestBeliefSelector` — ε-quantile Beta lower bound over `(successes, failures)` accumulators, used for conservative selection of frozen snapshots / adapters / direction vectors. Complements `BanditStrategy::ThompsonSampling` (exploration) with a conservative-exploitation counterpart.

---

## 1. Paper Core Findings

### 1.1 Controlled Utility Evolution (the headline mechanism)

The paper's central contribution is making **evaluation part of the self-improvement loop**. Prior Gödel Machine variants (Darwin, Huxley, HyperAgents) assume a stationary external evaluator. RQGM co-evolves the evaluator alongside the task agent, but does so under a disciplined schedule:

- **Epoch-local stationarity (Asm. 2):** within an epoch, the evaluator, artifact-generation protocol, and binary scoring rule are frozen. Outcomes are time-homogeneous Bernoulli with fixed success probability `p_{r,d,j}(a)`.
- **Boundary-only replacement (§3.5):** the evaluator can change only at exponentially-spaced checkpoints. A challenger is promoted only if it strictly raises the ε-best-belief score on an evaluator-independent *anchor* (a fixed ground-truth dataset).
- **Selective erasure (Def. F.2):** when a slot's evaluator is replaced, only utility records whose `dep_set` contains that slot are erased. Records depending on other slots, or on no slot (evaluator-independent), remain. Prop. 2 proves this preserves criterion consistency.

Prop. 6 shows that with exponential checkpoint spacing (ratio ρ > 1), the cumulative reprocessing cost over a budget of B evaluations is **O(B)**, not O(B²). The paper uses ρ = 2.

### 1.2 ε-Best-Belief Selection (the lower-bound criterion)

The paper uses a single rule for both evaluator replacement and final agent selection:

```
BB_ε(a) = I⁻¹_ε(1 + S_a, 1 + F_a)
```

where `I⁻¹_ε` is the inverse regularized incomplete Beta function (the ε-quantile of `Beta(1 + S, 1 + F)`). Prop. 4 proves: under the working posterior, `Pr(U(a*) ≥ BB_ε(a*) | Q_a*) = 1 − ε`. For a sequence of L selections, with probability ≥ 1 − Lε every selected candidate's utility exceeds its reported lower bound (union bound).

This is a **conservative selection rule** — it prefers candidates with high lower-bound evidence, not just high point estimates. Ties favor the incumbent (to avoid unnecessary erasure).

### 1.3 Three-Level Sampling Hierarchy

- **Node level:** Thompson sampling over clade metaproductivity (CMP) — pooled success rate over a node's subtree.
- **Role level:** least-evaluated eligible role.
- **Task level:** least-evaluated eligible task within the chosen role.

Prop. 1 proves the pooled Beta accumulator's posterior mean converges to the role-task-balanced utility almost surely under balanced sampling.

### 1.4 Co-Evolution Curriculum + Adversarial Objectives

- **Curriculum effect (RQ3):** evaluator replacements permanently re-rank the archive (Spearman ρ settles well below 1 and never recovers). A resilient backbone lineage survives; the remainder is re-ranked. Selective erasure is necessary — the no-erasure control stays pinned to the displaced order.
- **Adversarial objectives at boundaries (§5.4):** to correct LLM self-preference bias in paper reviewers, after each replacement the AI-generated papers the displaced reviewer accepted form an adversarial pool; the next epoch additionally rewards rejecting them. The reviewer's raw APReS accuracy drops slightly but its human/AI calibration equalizes — a harder-to-hack signal for the co-evolving writer.

### 1.5 Empirical Headlines

- Polyglot coding: 71.7% held-out pass rate vs prior SOTA 69.9%, at 1.35×–1.72× fewer tokens (co-evolved code reviewer complements test execution).
- Paper writing: co-evolved writers reach 1.78×–1.86× higher reviewer-panel acceptance.
- Proof grading: co-evolved grader 3× more token-efficient than HGM-H at matched accuracy.
- Theoretical guarantees are **epoch-local only** (Prop. 3, 5) — the paper explicitly disclaims global convergence (App. G).

---

## 2. Distillation

### 2.1 What routes to riir-train (NOT distilled here)

- The LLM-based evaluator fine-tuning loop (Gradient descent through evaluator weights).
- The meta-agent's self-modification of its own codebase (requires foundation-model training).
- The DPO/GRPO-style writer improvement from reviewer signal.
- The adversarial-pool curriculum that hardens the writer over epochs.

These are genuine training contributions. Per §3.5 of the research skill, the modelless-unblock protocol was checked: none of these can be realized via freeze/thaw, raw/lora hot-swap, or latent-space correction, because the *evaluator improvement itself* requires gradient-based learning on new data. → riir-train.

### 2.2 What stays modelless (distilled here)

#### 2.2.1 Criterion-Versioned Record Store with Selective Erasure

**The primitive.** A generic record store where each record `z` carries:

```rust
struct CriterionVersionedRecord<D> {
    data: D,
    dep_set: BitSet,           // evaluator slots whose criterion affected this record
    criterion_tags: Vec<Hash>, // per-slot BLAKE3 of the active criterion when generated
    epoch_vector: Vec<u32>,    // epoch index per slot at generation time
}
```

The store exposes `erase_slot(slot: usize)` which retains records `z` where `slot ∉ z.dep_set OR z.criterion_tags[slot] == current_criterion[slot]`. After erasure, all derived statistics (success/failure counts, CMP, Thompson stats) are recomputed from retained records only.

**Why this is modelless.** The store operates on `&mut` records and a versioned criterion vector. No gradient descent, no LLM calls. The erasure is a pure filter; the recomputation is a pure fold. Zero-allocation on the no-op path (no slot changed).

**Why this unifies scattered instances.** Today we have:
- Plan 279 (`manifold_power_iter_router`): Gram cache with `gram_cache_version: u64`, invalidated on snapshot version bump. Cache entry stores `(M[i], blake3_tag)`.
- Plan 315-riir-ai (`policy_cache_cascade`): `invalidate_zone_on_collapse()` walks per-NPC hot tier, removes entries whose `spatial_zone` matches the collapsing zone.
- Issue 001 (`hla_windowed_eigenbasis_recovery`): "When NPC's frozen snapshot is reloaded, the cached eigenbasis must be BLAKE3-checked against the activation window that produced it. Re-derive if mismatch."
- Plan 253 (`merkle_octree_curator_consensus`): curator reputation update with EMA decay on alpha/beta to handle concept drift.

Each is a single-slot special case of the general primitive. The general primitive lets us express all four as `store.erase_slot(SNAPSHOT_SLOT)` / `store.erase_slot(ZONE_SLOT)` / `store.erase_slot(PERSONALITY_SLOT)`, with the dep_set populated at insert time.

**Consistency guarantee (modelless analog of Prop. 2).** If the store is criterion-consistent before a transition on slot `m`, then after replacing `criterion[m]` and calling `erase_slot(m)`, the store is criterion-consistent under the new criterion vector. This is a pure data-structure invariant — no probability, no convergence assumption.

#### 2.2.2 Best-Belief ε-Quantile Beta Selector

**The primitive.**

```rust
/// Conservative lower-bound selection over Beta(1 + successes, 1 + failures).
/// BB_ε(a) = I⁻¹_ε(1 + S_a, 1 + F_a) — the ε-quantile of the Beta posterior.
/// The candidate's true utility exceeds BB_ε(a) with probability 1 − ε
/// under the working posterior (Prop. 4 of RQGM).
pub fn best_belief_score(successes: u32, failures: u32, epsilon: f32) -> f32;

/// Select the candidate with the highest best-belief score.
/// Ties favor the incumbent (passed as `incumbent_idx`) to avoid unnecessary
/// slot transitions and the erasure they trigger.
pub fn select_best_belief(
    candidates: &[(u32 /* S */, u32 /* F */)],
    epsilon: f32,
    incumbent_idx: Option<usize>,
) -> usize;
```

**Why this complements our existing bandits.** We have `BanditStrategy::ThompsonSampling` (sample from Beta posterior → exploration), `BanditStrategy::Ucb1` (upper confidence bound → exploration), and `BanditStrategy::SafePhased` (PrudentBanker → safe baseline mixture). We do **not** have the ε-quantile **lower** bound for **conservative selection** of frozen artifacts. The closest is PrudentBanker's geometric α escalation, but that controls the *exploration/exploitation mix*, not a *promotion gate*.

The best-belief selector is the right primitive for: "which frozen snapshot do we promote to active?", "which direction vector do we commit to the shard?", "which adapter do we hot-swap in?". These are **selection** decisions under bounded evidence, where conservatism (lower bound, not point estimate) is the right risk posture. Thompson sampling is for "which arm do we pull next?" (exploration).

**Beta quantile implementation.** `I⁻¹_ε(a, b)` (inverse regularized incomplete Beta) is a standard special function. We already ship `Beta` for Thompson sampling via Jöhnk's algorithm (Plan 030). The quantile needs a different algorithm — either Newton iteration on the incomplete Beta, or a continued-fraction inverse. Both are O(1) and allocation-free. Implementation note: reuse `cfg`-gated math; do not pull a new dep.

#### 2.2.3 Controlled-Utility-Evolution Freeze/Thaw Pattern (architectural)

Combining 2.2.1 + 2.2.2 yields a reframe of our existing freeze/thaw + MAPE-K + consolidation stack:

```
epoch j:  criterion vector κ_j frozen (BLAKE3-tagged per slot)
          ↓
          records accumulate in CriterionVersionedRecords, tagged with κ_j
          ↓
checkpoint c (exponential schedule, ratio ρ = 2):
          for each slot m:
              challengers ← candidate criteria for slot m
              e* ← select_best_belief(challengers ∪ {incumbent}, ε=0.05, incumbent_idx=Some(incumbent))
              if e* ≠ incumbent:
                  criterion[m] ← e*
                  store.erase_slot(m)              // selective erasure
                  recompute_derived_stats(store)   // CMP, Thompson, etc.
```

This is the modelless analog of RQGM Algorithm 1 lines 22–31. The "anchor" in our framing is whatever evaluator-independent ground truth the slot has access to: deterministic-replay verifier (game anti-cheat), arena outcome, formal prover (Lean4), or quorum-committed raw scalars.

**Map to existing modules:**
- `riir-neuron-db/src/freeze.rs` `MerkleFrozenEnvelope` = the epoch freeze (BLAKE3-tagged criterion).
- `riir-neuron-db/src/mape_k.rs` MAPE-K loop = the checkpoint schedule + replan.
- `riir-neuron-db/src/consolidation.rs` Raven/δ-Mem = selective erasure on consolidation.
- `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` = the drift-triggered re-estimation scheduler (analog of "evaluator replacement triggered by drift").
- `riir-ai/crates/riir-engine/src/committed_blend/` `CommittedFieldBlend` = the BLAKE3-committed archetype field (analog of "frozen evaluator with anchor lower bound").

None of these today implement the full 4-part contract (epoch freeze + boundary replacement + anchor validation + selective erasure) as a unified abstraction. They each implement a slice.

### 2.3 Fusion (the Super-GOAT angle — tracked as issue, not committed here)

Fusing RQGM's consistency contract with our existing substrate yields a candidate Super-GOAT: **per-NPC co-evolution with selective forgetting on personality swap at MMO scale.**

The reframing across the seven Super-GOAT factory modules:

| Module | RQGM mechanism reframed |
|---|---|
| `katgpt-core/src/sense/` (HLA) | "Evaluator" = HLA affect direction vector. Hot-swap on personality divergence (tame event, faction change, trauma). Selective erasure invalidates only cached affect projections that depended on the old direction. |
| `latent_functor/` | Coherence-driven re-estimation = the drift-triggered "evaluator replacement". Quality gate = anchor validation. The functor's `coherence < tau_reest` is the analog of "challenger strictly raises ε-best-belief on anchor". |
| `cgsp_runtime/` | Curiosity signal = Thompson sampling over CMP. Collapse bridge = selective erasure on collapse-triggered snapshot swap. |
| `riir-neuron-db/src/` (shards) | `MerkleFrozenEnvelope` = epoch freeze. `can_freeze` gate (Plan 002) = anchor lower-bound check. Raven/δ-Mem consolidation = selective erasure + re-scoring. |
| `riir-chain/src/encoding/latcal*.rs` | LatCal fixed-point commitment = deterministic anchor (raw scalar bridge for the ε-best-belief score). |
| DEC operators | (Not directly relevant — this paper has no Stokes/divergence/Hodge angle.) |

**Candidate selling point:** "Our NPCs co-evolve their personality direction vectors and their theory-of-mind models of other NPCs. When an NPC's snapshot is swapped (personality divergence, tame event, faction change), only the memories that depended on the OLD personality are erased — position, HP, wallet balance (raw, synced) survive bit-identical; affect projections, KG triples, cached policies (latent, local) are selectively invalidated. This enables emergent social dynamics where personality drift triggers targeted forgetting, not full amnesia — at MMO scale (thousands of NPCs, 20Hz tick)."

**Why this is a fusion, not a direct map.** RQGM provides the *consistency contract* (selective erasure + best-belief + epoch-local stationarity). Our substrate provides the *latents* (HLA affect vectors, KG triple directions, NeuronShard style_weights) and the *commitment layer* (MerkleFrozenEnvelope, LatCal). Neither alone produces per-NPC selective forgetting at MMO scale.

**Why this is NOT committed as Super-GOAT in this session.** The four novelty-gate questions need deeper validation than this session can provide:
- **Q1 (no prior art):** The unified primitive doesn't ship, but scattered instances do (Plan 279, 315, Issue 001). The *unification* is the novelty claim — needs verification that no existing trait/type already abstracts this. The author of this note grepped `.research/` + `.plans/` + code; the unified primitive does not appear, but a deeper code audit (especially `riir-engine/policy_cache/`, `riir-engine/adapters/`, `riir-neuron-db/consolidation.rs`) is warranted before claiming "no prior art" with confidence.
- **Q2 (new class):** Per-NPC selective forgetting on personality swap is plausibly a new capability class, but the *mechanism* (dependency tracking + cache invalidation) is well-known CS. The novelty is in the application + the latent-space reframing + the MMO-scale constraint.
- **Q3 (selling point):** Concrete and finishable, but depends on riir-train producing the actual personality-direction co-evolution (the modelless side handles consistency only).
- **Q4 (force multiplier):** ≥5 pillars touched (HLA, latent_functor, NeuronShard, MAPE-K, KG triples, Plan 315, Plan 279). Strong.

Per the research skill: "If you are NOT confident enough to commit all 4 YES right now, do not write 'Super-GOAT candidate'. Write 'fusion idea — novelty TBD, needs Q1–Q4 check before verdict' and create an issue." → Issue created at `katgpt-rs/.issues/`.

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | Open primitive → katgpt-rs + private guide → riir-ai/riir-chain/riir-neuron-db |
| **GOAT** | Provable gain over existing approach, but not a new class of capability | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement | Plan only, behind feature flag. |
| **Pass** | Not relevant, OR training-only | One-line note. |

**Verdict: GOAT.**

**One-line reasoning:** Two modelless primitives (criterion-versioned record store with selective erasure + ε-best-belief Beta quantile selector) are unifications of scattered shipped instances with a provable consistency guarantee and a complementary role to Thompson sampling — provable gain, but not a new capability class on their own. The co-evolution training loop routes to riir-train. The Super-GOAT fusion angle (per-NPC selective forgetting on personality swap at MMO scale) is documented in §2.3 and tracked as an issue for follow-up novelty-gate validation.

**Routing:**
- `katgpt-rs/crates/katgpt-core/src/` — two open primitives:
  - `selective_erasure.rs` (or `criterion_store.rs`) — `CriterionVersionedRecords<D>` generic store.
  - `best_belief.rs` — `best_belief_score()`, `select_best_belief()`.
- Feature gate: `controlled_utility` (opt-in until GOAT gate passes).
- Plan: [`katgpt-rs/.plans/336_controlled_utility_primitives.md`](../.plans/336_controlled_utility_primitives.md).
- Issue (Super-GOAT fusion follow-up): [`katgpt-rs/.issues/004_per_npc_selective_forgetting_super_goat_fusion.md`](../.issues/004_per_npc_selective_forgetting_super_goat_fusion.md).

**GOAT gate (must pass before promotion to default):**
- **G1 (correctness):** `CriterionVersionedRecords::erase_slot(m)` preserves criterion consistency under arbitrary transition sequences (modelless analog of Prop. 2). Tested with random transition sequences + fuzz.
- **G2 (perf):** `erase_slot` on a store with N records and k slots is O(N) worst case, O(0) on no-op (slot unchanged). `best_belief_score` is O(1) (closed-form Beta quantile). Both ≤ 100 ns.
- **G3 (no regression):** Wiring `erase_slot` into Plan 279 Gram cache + Plan 315 cascade invalidation + Issue 001 HLA eigenbasis produces bit-identical behavior to the existing hand-rolled invalidation.
- **G4 (alloc-free):** Hot path (no-op erasure, best-belief lookup) is allocation-free. `erase_slot` reuses `Vec::retain` semantics (in-place).
- **G5 (sigmoid, not softmax):** N/A — this primitive operates on discrete records, not probability distributions. Documented for completeness.

**What stays open after the GOAT plan ships:**
- The Super-GOAT fusion (per-NPC selective forgetting) requires the riir-train side to produce actual personality-direction co-evolution. Once that lands, the modelless consistency contract here becomes the runtime substrate, and the Super-GOAT guide can be written in `riir-ai/.research/`.
- Multi-slot commutative transitions (Rem. 1 of the paper) — the primitive should support batch `erase_slots(&[m1, m2, ...])` with order-independence, but this is a Phase 2 refinement, not a Phase 1 blocker.

---

## TL;DR

RQGM (arXiv:2606.26294) is a *training* paper at its headline (co-evolved LLM evaluators under non-stationary utilities) → the training loop routes to riir-train. But it ships two **modelless inference primitives** we lack as unified abstractions: (1) a **criterion-versioned record store with selective erasure** — every record carries a `dep_set` of evaluator slots, and replacing a slot's criterion erases only dependent records (Prop. 2 proves criterion consistency); (2) an **ε-best-belief Beta quantile selector** `BB_ε = I⁻¹_ε(1+S, 1+F)` — a conservative lower bound for *selection* (which snapshot to promote), complementing our Thompson sampling (for *exploration*). Together they reframe our scattered freeze/thaw consistency instances (Plan 279 Gram cache invalidation, Plan 315 cascade `invalidate_zone_on_collapse`, Issue 001 HLA eigenbasis BLAKE3-check) under one epoch-local-stationarity + boundary-replacement + selective-erasure contract. **Verdict: GOAT** — provable unification gain, two new open primitives in `katgpt-core` behind `controlled_utility` feature, GOAT gate G1–G4 defined. The Super-GOAT fusion angle (per-NPC co-evolution with selective forgetting on personality swap at MMO scale) is documented in §2.3 but **not committed** — needs deeper Q1–Q4 novelty validation, tracked as an issue. The four-pillar mapping (HLA affect direction = evaluator; `latent_functor/reestimation` = drift-triggered replacement; `MerkleFrozenEnvelope` = epoch freeze; Raven/δ-Mem = consolidation erasure) shows the latent reframing is strong, but the application novelty (MMO-scale selective forgetting) depends on riir-train delivering the personality co-evolution that the modelless side only keeps consistent.
