# Plan 215: Regime-Transition Inference (Self-Revising Discovery)

**Research**: R190 (Self-Revising Discovery Regime Transition)
**Status**: ✅ COMPLETE — GOAT PROVED (8/8 benchmarks) | DEMOTE: 19× overhead vs baseline, kept opt-in
**Feature Gate**: `regime_transition` (DEMOTE — kept opt-in, 19× overhead, heavy deps)
**Depends On**: Plan 209 (FOL), Plan 210 (INSIGHT), Plan 211 (Three-Mode Router)

---

## Overview

Implement the paper's "discovery as vocabulary change" concept as inference-time regime transitions in the pruner system. When the current set of pruners cannot express the answer (regime collapse), detect it, propose new pruner types, and admit them through an MDL gate.

---

## Architecture

```
DDTree Exploration
    │
    ├── Failure Pattern Detection (papaya HashMap)
    │       │
    │       ├── Search Collapse → keep exploring (existing behavior)
    │       └── Regime Collapse → trigger RegimeTransitionGate
    │
    ├── RegimeTransitionGate
    │       ├── Correctness gate: WASM sandbox test
    │       └── Information gate: Epiplexity DL reduction check
    │
    ├── New Pruner Extraction
    │       ├── Plan 209 FolPruner: extract from failure patterns
    │       └── Plan 210 ExpressionPruner: fit symbolic boundary
    │
    └── ProvenancePreservingAbsorb
            ├── blake3 hash of pruner provenance chain
            └── Kan-transport replay on regime change
```

---

## Tasks

### T1: CollapseClassifier Trait

- [x] Define `CollapseType` enum: `Search`, `Regime`
- [x] Define `CollapseClassifier` trait with `fn classify(&self, ddtree_stats: &DDTreeStats) -> CollapseType`
- [x] Implement `RegimeCollapseClassifier` that checks: all branches fail at same depth → Regime, otherwise → Search
- [x] Unit test: mock DDTreeStats with uniform failure depth → Regime
- [x] Unit test: mock DDTreeStats with scattered failures → Search
- [x] Feature gate behind `regime_transition`

### T2: RegimeTransitionGate

- [x] Define `RegimeTransitionGate` struct wrapping `ConstraintPruner` + `DecisionTrace`
- [x] Implement `fn evaluate(candidate: &dyn ConstraintPruner, trace: &DecisionTrace) -> GateResult`
- [x] Correctness check: run candidate through WASM sandbox (reuse existing validator infra)
- [x] Information check: compute `description_length(trace, current_pruuners)` vs `description_length(trace, current_pruners + candidate)`
- [x] Admission cost: configurable bits per new pruner type (default: 32 bits)
- [x] Accept iff: `DL_new < DL_old - AdmissionCost`
- [x] Unit test: candidate that reduces DL by > AdmissionCost → Accept
- [x] Unit test: candidate that reduces DL by < AdmissionCost → Reject
- [x] Integration test: full pipeline with mock DDTree producing regime collapse

### T3: ProvenanceChain for AbsorbCompress

- [x] Define `ProvenanceChain` struct: `Vec<ProvenanceStep>` where each step has episode_id, reward, bandit_pull, blake3_hash
- [x] Extend `AbsorbCompress` to record ProvenanceChain on each absorb
- [x] Implement `fn transport(&self, new_schema: &[PrunerType]) -> TransportResult`
- [x] Transport replays provenance steps in new vocabulary, verifies parameters still valid
- [x] blake3 hash of ProvenanceChain = commitment hash
- [x] Unit test: absorb → transport with same schema → all parameters valid
- [x] Unit test: absorb → transport with different schema → some parameters invalid → flag

### T4: AdversarialBreaker Wrapper

- [x] Define `AdversarialBreaker<P: ConstraintPruner>` wrapping any pruner
- [x] Track failure patterns via `Mutex<HashMap<FailurePattern, u32>>` (no papaya dep needed)
- [x] When pattern count exceeds threshold (configurable, default 5) → generate synthetic edge case
- [x] Synthetic edge case: perturb the failing token sequence by ±1 token in each position
- [x] Feed synthetic through DDTree to verify it exposes a genuine failure
- [x] If genuine → extract as new rule via Plan 209 T2 (RuleExtractor)
- [x] Integration test: mock failure pattern → synthetic generation → rule extraction

### T5: Four-Regime Router Extension

- [x] Extend Plan 211 (Three-Mode Router) to Four-Regime Router
- [x] Add `Discovery` regime: entered only when RegimeCollapseClassifier returns Regime
- [x] Add `Consolidation` regime: entered after any successful regime transition, runs AbsorbCompress + ProvenanceChain
- [x] Bandit arms: 3 regimes × 2 heaviness options = 6 arms (UCB1 with sigmoid confidence)
- [x] Sigmoid-gated mixing (NOT softmax) per user constraint
- [x] Integration test: mock scenario triggering discovery → regime transition → consolidation cycle

### T6: GOAT Proof — Before/After Benchmark

- [x] Create `tests/bench_regime_transition.rs`
- [x] Before: Run DDTree on known-hard constraint problem WITHOUT regime transition (fixed pruners)
- [x] After: Run same problem WITH regime transition enabled
- [x] Measure: accuracy (valid branches found), token efficiency, total compute time
- [x] Expected: regime transition finds valid branches that fixed pruners miss
- [x] Also measure: overhead of regime transition detection (< 5% of total compute)
- [x] Run with `--features regime_transition` and `--nocapture`
- [x] Document results in `.benchmarks/` folder

### T7: CPU/GPU Auto-Route Integration

- [x] When `Discovery` regime is active and load is low (CPU idle) → run regime transition on CPU
- [x] When load is high → defer regime transition to background thread
- [x] `RegimeTransitionScheduler` with configurable concurrency limit (default: 1 concurrent transition)
- [x] Integration test: concurrent decode + regime transition → no perf regression on decode path

---

## Execution Order

1. T3 (ProvenanceChain) — zero risk, extends existing AbsorbCompress
2. T1 (CollapseClassifier) — foundation, simple trait
3. T2 (RegimeTransitionGate) — core gate logic
4. T4 (AdversarialBreaker) — extends Plan 209
5. T5 (Four-Regime Router) — extends Plan 211
6. T6 (GOAT Proof) — validate
7. T7 (CPU/GPU Auto-Route) — production hardening

---

## Expected Outcomes

| Metric | Before | After | Notes |
|--------|--------|-------|-------|
| Valid branches on novel constraints | Low (fixed pruners) | High (discovered pruners) | Core value proposition |
| Regime collapse detection | None | Binary classifier | Enables discovery mode |
| Provenance audit trail | None | blake3-hashed chain | Trust in absorbed parameters |
| Overhead | 0% | < 5% | Feature-gated, zero cost when off |
| Self-improvement rate | Linear (AbsorbCompress only) | Super-linear (vocabulary growth) | The key improvement |
