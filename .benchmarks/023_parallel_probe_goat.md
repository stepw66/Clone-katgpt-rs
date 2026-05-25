# GOAT Proof 023: Parallel-Probe 2D — Consensus-Based Early Stopping & Branch Pruning (Plan 133)

> **Date:** 2026-05-25
> **Feature Gate:** `parallel_probe`
> **Depends on:** Plan 133 T1 (controller + matrix), T2 (answer extraction), T3 (verifier integration)

## Summary

GOAT proofs for the Parallel-Probe 2D speculative decoding system. Tests consensus-based early stopping, deviation-based branch pruning, answer extraction accuracy, and integration with the speculative pipeline at micro scale.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Config | `micro_config()` |
| N branches | 4 |
| Probe interval | 100 tokens |
| Stability patience | 3 |
| Prune patience | 10 |
| Warmup steps | 12 |
| Min active branches | 3 |
| Prune vote ratio | 0.5 |
| Seed | 42 |

## GOAT Results

### Proof 1: Consensus Early Stopping

**Hypothesis:** When all active branches converge to the same answer, the controller stops early within `stability_patience` probe steps.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Steps to stop (all agree) | ≤ stability_patience | — | ⏳ Pending |
| Consensus answer correct | Yes | — | ⏳ Pending |
| No false stops during disagreement | Yes | — | ⏳ Pending |

**Gate:** ⏳ Pending — Consensus detection with configurable patience.

### Proof 2: Deviation-Based Branch Pruning

**Hypothesis:** Branches that consistently disagree with the majority are pruned after `prune_patience` consecutive probe steps, but not below `min_active_branches`.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Deviant branch pruned after patience | Yes | — | ⏳ Pending |
| Min active branches respected | ≥ min_active | — | ⏳ Pending |
| No pruning during warmup | Yes | — | ⏳ Pending |
| Active count after prune | Correct | — | ⏳ Pending |

**Gate:** ⏳ Pending — Pruning respects patience, warmup, and minimum active thresholds.

### Proof 3: Answer Extraction Accuracy (Regex)

**Hypothesis:** `RegexAnswerExtractor` correctly identifies answers from LaTeX boxed, "the answer is", and numeric patterns with zero false positives on non-answer text.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| `\boxed{...}` extraction | 100% | — | ⏳ Pending |
| "The answer is ..." extraction | 100% | — | ⏳ Pending |
| Numeric extraction | 100% | — | ⏳ Pending |
| No false positives on plain text | 0% | — | ⏳ Pending |
| Priority order correct | Yes | — | ⏳ Pending |

**Gate:** ⏳ Pending — All supported patterns extracted correctly with proper priority.

### Proof 4: Think Token Extraction

**Hypothesis:** `ThinkTokenExtractor` returns the answer after `</think` boundary, handling multiple think blocks and empty post-think content.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Basic extraction after `</think` | Correct | — | ⏳ Pending |
| Last tag used when multiple | Correct | — | ⏳ Pending |
| Returns None for no tag | None | — | ⏳ Pending |
| Returns None for empty after tag | None | — | ⏳ Pending |

**Gate:** ⏳ Pending — Think token boundary correctly handled.

### Proof 5: Discrete Action Extraction

**Hypothesis:** `DiscreteActionExtractor` extracts valid action indices from game-domain text, respecting the `max_actions` bound.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Explicit "action: N" | Correct | — | ⏳ Pending |
| Last valid integer fallback | Correct | — | ⏳ Pending |
| Out-of-range values rejected | None | — | ⏳ Pending |
| Zero-index boundary | Correct | — | ⏳ Pending |

**Gate:** ⏳ Pending — Action extraction correct for game domains.

### Proof 6: ParallelProbeVerifier Integration

**Hypothesis:** `ParallelProbeVerifier` correctly wraps an inner verifier, extracts answers from branch texts, and delegates to the controller for probe decisions.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Branch text accumulation | Correct | — | ⏳ Pending |
| Probe triggers at correct interval | Yes | — | ⏳ Pending |
| Stop decision cached | Yes | — | ⏳ Pending |
| Active branches tracked correctly | Yes | — | ⏳ Pending |
| Inner verifier accessible | Yes | — | ⏳ Pending |

**Gate:** ⏳ Pending — Verifier integration works end-to-end with answer extraction.

### Proof 7: ProbingMatrix Correctness

**Hypothesis:** `ProbingMatrix` stores and retrieves per-branch answer histories correctly, respecting max_probes limit.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Push and get operations | Correct | — | ⏳ Pending |
| Max probes limit enforced | Yes | — | ⏳ Pending |
| Row and column access | Correct | — | ⏳ Pending |
| Empty matrix handling | Correct | — | ⏳ Pending |

**Gate:** ⏳ Pending — Matrix operations are correct and bounded.

---

## Overall Status: ⏳ Pending

All 7 GOAT proof targets are pending implementation of integration tests with a trained mini model.
