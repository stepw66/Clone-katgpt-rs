# Research 217: TRD — Trajectory-Refined Distillation for Modelless Inference

**Date:** 2026-06-11
**Source:** arXiv:2606.08432 — "Trajectory-Refined Distillation" (Jiang, Xu, Ding, Zhang — McGill/Mila/UT Austin)
**Code:** https://github.com/louieworth/trd
**Verdict:** ⭐ GAIN — High priority, novel modelless fusion via TRDraft
**Target:** katgpt-rs (modelless inference engine, MIT open source)
**Relates To:** Research 038 (SDAR), 036 (ROPD), 044 (ELF), 040 (BT Rank), 175 (ThoughtFold), 151 (GDSD), 160 (SDPG)

---

## Executive Summary

TRD identifies **prefix failure** as the structural root cause of OPD instability: when a student's rollout takes a wrong reasoning path, per-token teacher supervision becomes a bimodal mixture (continue failure vs. pivot to correction), producing fragmented gradients that no token-level fix (clipping, reweighting, top-K) can resolve. TRD fixes this at the **trajectory level** — asking the teacher to produce a refined trajectory `yr` conditioned on the raw rollout `yo`, then training on `yr` instead.

**Why we care (modelless):** We don't do training in katgpt-rs. But we DO have an analogous problem: DDTree speculative decoding builds candidate branches from marginal log-probs (token-level), and when a branch enters a "prefix failure" state (wrong reasoning path), the verifier rejects it. Currently we just discard failed branches. TRD's insight — **fix the trajectory, not the loss** — maps to a novel modelless technique: **TRDraft** (Trajectory-Refined Draft), where the ConstraintPruner + BanditPruner act as a "modelless teacher" that refines failed DDTree branches.

**Key paper results:**
- +4.6% AIME24 on Qwen3-1.7B, +12.8% AMOBench on Qwen3-4B (OPD)
- ~50% relative improvement on hardest benchmark (AMOBench Pass@16, 8B model)
- 9× trajectory compression (median 7.7K → 0.88K tokens)
- 81.4% verifier accuracy on yr vs 65.8% on yo
- Solves 9/23 base-unreachable AMOBench questions (frontier expansion)

**GOAT Verdict:** ⭐ GAIN — TRDraft is a novel modelless fusion that no one else has. The paper trains with a teacher model; we refine at inference with ConstraintPruner + BanditPruner + BT Rank. This is modelless TRD. Feature-gate as `trd_refined_draft`, promote to default if GOAT proof passes.

---

## Paper Core Analysis

### The Prefix Failure Mechanism (Section 4)

TRD's contribution is formalizing *why* token-level OPD fixes fail:

1. **Bimodal teacher mixture:** Under prefix failure, the teacher distribution has two modes — one continuing the wrong prefix, one pivoting to correction. Forward KL (mode-covering) gets dominated by the correction onset, causing instability. Reverse KL (mode-seeking) gets dominated by the wrong continuation, making the correction signal invisible.

2. **Fragmented gradient:** Even a perfect teacher can't recover. At position t, the teacher recommends correction token `ȳ*_t`. But at position t+1, the context is `(yo,<t, yo,t)` — the *original wrong continuation*, not `(yo,<t, ȳ*_t)` — the correction path. So the teacher keeps recommending the *same* correction onset token on ever-deepening wrong contexts. The gradient pair sets share only their first element.

3. **Epistemic token trap:** The teacher places 6-8‰ of mass on 16 epistemic onset tokens (Wait, Actually, However...) throughout training — the `ȳ*_t`-repeat signature. This mass collapses to <2‰ on refined trajectories, confirming it's a prefix-failure artifact.

### The TRD Algorithm (Section 5)

```
For each problem x:
  1. yo ~ πθ(·|x)           # Raw on-policy rollout
  2. yr ~ πT(·|x, yo)       # Teacher-refined trajectory (conditioned on yo)
  3. Train on yr via forward KL with full vocabulary
```

**Key design choices:**
- Forward KL on yr (mode-covering) — opposite of what SDAR/reverse-KL papers recommend on yo. The bimodal mixture problem is resolved by trajectory refinement, so forward KL's stability advantage on clean yr dominates.
- Full-vocabulary KL (no top-K truncation) — reduced variance, stabilized gradients
- yr is within on-policy support because it's conditioned on yo (student's own reasoning patterns)
- Even correct yo benefits: yr surfaces alternative valid derivations, expanding exploration

### Trajectory Analysis (Section 6.5)

| Metric | yo (raw) | yr (refined) |
|--------|----------|--------------|
| Verifier accuracy | 65.8% | 81.4% (+15.6%) |
| Median length | 7.7K tokens | 0.88K tokens (9× compression) |
| Correct-only length | ~2.2K | ~0.85K |
| fail→pass rate | — | 16.0% of training set |
| pass→fail leakage | — | 0.4% (negligible) |

**Critical finding:** TRD's gains come from the *breadth* of the refined corpus, not any privileged subset. Filtering to only fail→succ trajectories *hurts* performance (Appendix B.3). Both correct and incorrect yo contribute.

---

## Cross-Reference: Existing Features

| TRD Concept | Our Feature | Status | Gap |
|-------------|-------------|--------|-----|
| Raw rollout yo | DDTree speculative draft | ✅ Production | — |
| Teacher-refined yr | **MISSING** — no trajectory refinement | ❌ Gap | TRDraft fills this |
| Prefix failure detection | Collapse-aware thinking (Plan 212) | 🟡 Planned | TRD formalizes detection |
| Per-token KL on yr | SDAR sigmoid gate (Research 038) | ✅ Planned | SDAR on yr instead of yo |
| Trajectory compression | ThoughtFold chain folding (Research 175) | ✅ Planned | TRD achieves 9× without folding |
| Verifier accuracy signal | ConstraintPruner | ✅ Production | Already rejects invalid branches |
| Bimodal distribution | ELF SDE noise for diversity | ✅ Planned | SDE helps re-drafting |
| Branch ranking | BT Rank (Research 040) | ✅ Planned | Ranks yr vs yo branches |
| Rubric credit assignment | ROPD rubric vectors (Research 036) | ✅ Planned | Credit for which prefix failed |
| Advantage-guided pruning | GDSDPruner (Research 151) | ✅ Planned | Advantage guides re-draft |
| Adaptive budget | BanditPruner | ✅ Production | Controls refinement budget |
| Self-distilled policy | SDPG (Research 160) | ✅ Planned | SDPG on yr trajectories |

### Complementarity: TRD vs SDAR vs ROPD

| Dimension | TRD | SDAR (Research 038) | ROPD (Research 036) |
|-----------|-----|--------------------|--------------------|
| Intervention level | Trajectory | Token | Token (multi-criterion) |
| Problem addressed | Prefix failure | Multi-turn instability | Inter-dimension interference |
| Mechanism | Fix input to loss | Gated loss on raw rollout | Rubric-weighted loss |
| KL direction | Forward KL on yr | Reverse KL on yo | Reverse KL on yo |
| Relationship | **Orthogonal** — TRD fixes trajectories, then SDAR/ROPD can gate per-token loss on yr | **Complementary** — apply SDAR gate on yr instead of yo | **Complementary** — rubric credit on yr is more meaningful |

---

## Novel Fusion: TRDraft — Trajectory-Refined Draft for Speculative Decoding

### Core Idea

TRD uses a teacher model to refine trajectories. We have no teacher model in katgpt-rs (modelless). But we have something equivalent: **ConstraintPruner + BanditPruner + BT Rank** as a "modelless teacher" that refines failed DDTree branches.

```
TRD (paper):        yo → πT(yo) → yr → train on yr
TRDraft (ours):     DDTree draft → ConstraintPruner rejects → re-draft from failure point → yr → use yr
```

### Algorithm: TRDraft

```
1. DDTree generates initial candidate tree (yo equivalent)
2. LeviathanVerifier evaluates each branch
3. For rejected branches (prefix failure detected):
   a. Locate failure point via BanditPruner's Q-value drop
   b. Rollback DDTree to failure point
   c. Re-draft from failure point with ELF SDE noise for diversity
   d. ConstraintPruner constrains re-draft to valid continuations
   e. Verify re-drafted branch
4. BT Rank ranks original + re-drafted branches
5. Select best branch for output
```

### Why This Is Novel

1. **No teacher model.** The "teacher" is ConstraintPruner (rules) + BanditPruner (learned relevance) + ELF SDE (diversity). This is modelless TRD.

2. **Trajectory-level fix at inference time.** The paper fixes trajectories for training. We fix trajectories *live* during speculative decoding. This is a novel application of TRD's insight.

3. **Adaptive refinement depth.** The paper always does 1-step refinement (yo → yr). We can make this adaptive via BanditPruner:
   - Easy problems: Skip refinement (bandit learns this is waste)
   - Medium: 1-step re-draft (standard TRD)
   - Hard: Multi-step re-draft with alternating draft→verify→refine cycles

4. **ThoughtFold synergy.** Before re-drafting, apply ThoughtFold chain folding to identify redundant reasoning in the prefix. This gives the re-draft a cleaner starting point.

### Integration Points

```rust
// In SpeculativeGenerator trait
trait TrajectoryRefinedDraft: SpeculativeGenerator {
    /// Detect prefix failure via verifier rejection + entropy spike
    fn detect_prefix_failure(&self, branch: &DdTreeBranch) -> Option<FailurePoint>;

    /// Re-draft from failure point with constraints
    fn refine_branch(
        &self,
        tree: &mut DdTree,
        failure: FailurePoint,
        constraints: &dyn ConstraintPruner,
        noise: &ElfSdeNoise,
    ) -> DdTreeBranch;

    /// BT Rank comparison: raw vs refined
    fn rank_branches(&self, raw: &[DdTreeBranch], refined: &[DdTreeBranch]) -> DdTreeBranch;
}
```

### Fusion with Existing Features

| Feature | TRDraft Integration |
|---------|-------------------|
| **ELF SDE** | Controlled noise in re-drafting (yr diversity) |
| **BanditPruner** | Learns which refinement strategies work; controls refinement budget |
| **BT Rank** | Ranks refined vs raw trajectories; pairwise > pointwise |
| **ThoughtFold** | Folds redundant branches before refinement (cleaner starting point) |
| **Collapse-aware** | Detects when DDTree has entered "prefix failure" state (triggers TRDraft) |
| **ROPD rubric** | Per-criterion credit assignment for which part of the prefix failed |
| **GDSDPruner** | Advantage-guided relevance for re-drafted branches |

---

## CPU/SIMD/GPU/ANE Routing

| Component | Target | Rationale |
|-----------|--------|-----------|
| Prefix failure detection | CPU | Entropy computation on logits — fast, scalar ops |
| ConstraintPruner check | CPU/SIMD | Rule-based, fixed-size vocab scan — SIMD for top-k |
| Re-drafting (DDTree expansion) | GPU | Logit computation for tree expansion — batched matmul |
| BT Rank pairwise comparison | SIMD | Small N candidates, pairwise σ(si - sj) — vectorizable |
| ThoughtFold chain analysis | CPU | Binary search on reasoning steps — sequential, low compute |

**Plasma tier:** TRDraft operates at inference time — Hot/Warm tier. No Cold tier involvement.

---

## Performance Analysis

### Expected Gains

| Metric | Baseline | TRDraft (expected) | Rationale |
|--------|----------|-------------------|-----------|
| Speculative acceptance rate | ~65% | ~75-80% | Trajectory refinement removes prefix failures |
| Avg tokens per query | baseline | -10-20% | Shorter refined trajectories (paper: 9× compression, but inference is less dramatic) |
| Hard-query accuracy | baseline | +5-15% | Frontier expansion: solving previously unreachable queries |
| Verification cost | baseline | +5-10% | Extra re-draft pass, partially offset by shorter trajectories |
| Latency (P50) | baseline | ±0% | Easy queries skip refinement (BanditPruner) |
| Latency (P99) | baseline | +10-20% | Hard queries trigger multi-step refinement |

### Wall-Clock Budget

TRD paper shows total wall-clock nearly matches baselines because shorter yr offsets extra sampling (Qwen3-8B: 9:20 TRD vs 9:40 vanilla). For modelless inference:
- Re-drafting adds ~1 extra DDTree expansion at failure point
- But shorter branches mean less verification work downstream
- Net: near-zero latency impact on average, positive on hard queries

---

## Risk Assessment

### Honest Risks

1. **Re-drafting may not help if constraints are already tight.** If ConstraintPruner already produces narrow valid sets, re-drafting from failure point may produce the same branch. Mitigation: ELF SDE noise injects diversity.

2. **BanditPruner cold start.** Early in deployment, the bandit has no data on which refinement strategies work. Mitigation: default to 1-step refinement (paper's default), let bandit learn adaptively.

3. **Multi-step refinement could loop.** Re-drafting a branch that fails again, then re-drafting again... Mitigation: cap at 2 refinement steps, fallback to raw branch.

4. **Latency tail.** Hard queries that trigger multi-step refinement add latency. Mitigation: budget cap via BanditPruner, abort refinement if budget exceeded.

5. **Domain gap.** TRD paper tests math/code reasoning. Our domain is game AI + general inference. Prefix failure patterns may differ. Mitigation: GOAT proof on our benchmarks.

### What NOT to Apply

| TRD Aspect | Why Not for katgpt-rs |
|------------|----------------------|
| Teacher model forward pass | Modelless — we have no teacher model |
| Training on yr | No training in katgpt-rs |
| Forward KL loss computation | No loss computation in katgpt-rs |
| OPSD privileged conditioning | No privileged context at inference |
| 8×H100 training pipeline | Modelless — runs on CPU/GPU inference |

---

## GOAT Verdict

### Assessment

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐ HIGH | Modelless TRD (TRDraft) is novel — no paper applies TRD's trajectory refinement to inference-time speculative decoding |
| Applicability | ⭐⭐⭐ HIGH | DDTree + ConstraintPruner + BanditPruner already exist; TRDraft extends naturally |
| Expected gain | ⭐⭐ MEDIUM-HIGH | Paper shows +4.6-12.8% on hard benchmarks; inference-time gain likely 5-15% on hard queries |
| Implementation cost | ⭐⭐ LOW | ~200-300 lines: `TrajectoryRefinedDraft` trait + `FailurePoint` struct + integration |
| Risk | ⭐⭐ MEDIUM-LOW | Feature-gated, BanditPruner controls budget, graceful fallback to raw branch |

### Feature Gate

```toml
[features]
default = []
trd_refined_draft = ["elf_sde", "bandit_pruner", "bt_rank"]
```

GOAT-gate: `trd_refined_draft` — OFF by default. Promote to default if GOAT proof shows:
1. >5% improvement on hard-query speculative acceptance rate
2. No P50 latency regression
3. P99 latency regression <15%

### Commercial Strategy (per Research 003)

- **TRDraft** → MIT katgpt-rs engine. Trajectory refinement at inference is "plumbing" — open, attracts adoption.
- **Training on yr** → Private riir-ai SaaS. The *model* that learns from refined trajectories is "fuel" — closed, monetizable.
- Engine/fuel split intact. ✅

### Implementation Priority

**Phase 1:** `TrajectoryRefinedDraft` trait + `FailurePoint` detection (via Collapse-aware + verifier rejection)
**Phase 2:** Single-step re-draft with ConstraintPruner + ELF SDE
**Phase 3:** BT Rank integration for raw vs refined comparison
**Phase 4:** BanditPruner adaptive budget (skip/1-step/multi-step)

---

## Tasks

- [ ] Implement `FailurePoint` struct with entropy-spike + verifier-rejection detection
- [ ] Implement `TrajectoryRefinedDraft` trait extending `SpeculativeGenerator`
- [ ] Add single-step re-draft with ConstraintPruner constraints
- [ ] Integrate ELF SDE noise for re-draft diversity
- [ ] Add BT Rank pairwise comparison for raw vs refined branches
- [ ] BanditPruner adaptive refinement budget (skip/1-step/2-step)
- [ ] GOAT proof: speculative acceptance rate on hard queries
- [ ] GOAT proof: latency P50/P99 regression check
- [ ] Feature gate `trd_refined_draft`, promote to default if GOAT passes

---

## References

- TRD paper: https://arxiv.org/abs/2606.08432
- TRD code: https://github.com/louieworth/trd
- SDAR (our Research 038): Token-level gated distillation — complementary to TRD
- ROPD (our Research 036): Rubric-based credit assignment — rubrics on yr
- ThoughtFold (our Research 175): Chain folding — cleaner starting point for re-draft
- BT Rank (our Research 040): Pairwise ranking — raw vs refined branch comparison
- ELF (our Research 044): SDE noise for re-draft diversity
- GDSD (our Research 151): Advantage-guided pruner — advantage on re-drafted branches
- SDPG (our Research 160): Self-distilled policy gradient — on yr trajectories
- Collapse-Aware (Plan 212): Entropy collapse detection — prefix failure trigger

---

## TL;DR

TRD proves that prefix failure — not token-level noise — is the root cause of OPD instability. The fix is trajectory-level (refine yo → yr), not loss-level (clip/reweight). For katgpt-rs, this maps to **TRDraft**: when DDTree's verifier rejects a branch (prefix failure), re-draft from the failure point using ConstraintPruner + ELF SDE noise + BT Rank. This is modelless TRD — no teacher model needed, the verifier IS the teacher. Expected gain: 5-15% on hard queries, near-zero P50 latency impact. Feature-gate as `trd_refined_draft`, promote to default if GOAT proof passes. Orthogonal to SDAR/ROPD (token-level) — both can run together (TRDraft fixes trajectory, SDAR gates per-token loss on the fixed trajectory). Engine/fuel split intact.
