# Plan 232: DynamicRankPruner Acceptance Rate Benchmark

## Status: ✅ DONE

## Setup
- **Test file**: `tests/bench_232_dynamic_rank_acceptance.rs`
- **Feature gate**: `bandit` (baseline), `dynamic_rank+bandit` (wrapped)
- **Methodology**: DDTree build + simulated context-dependent verification over 1000 episodes

## What It Measures
- **Acceptance rate**: how many tokens in the extracted best path survive simulated verification
- **Context-dependent correctness**: parent context determines which tokens are "correct" (simulates real verifier rejecting context-inappropriate tokens)
- **Reward**: cumulative reward signal from verification

## Results (2026-06-09)

| Metric | BanditPruner Baseline | DynamicRankPruner + BanditPruner | Delta |
|--------|----------------------|----------------------------------|-------|
| Tree nodes | 16,000 | 16,000 | +0 |
| Accepted | 1,212 | 1,211 | -1 |
| Acceptance rate | 7.58% | 7.57% | -0.01pp |
| Total reward | 680.1 | 678.2 | -1.9 |
| Rate delta | — | -0.08% | — |

## Analysis

**Near-zero impact**: DynamicRankPruner wrapping BanditPruner produces essentially identical acceptance rates.

Root causes:
1. **Static corrections have limited effect on tree structure**: Both trees have identical node counts (16,000), meaning the relevance score corrections from DynamicRankPruner don't change which branches get explored in the DDTree. The marginals dominate the tree structure, not the pruner scores.
2. **Learning rate too small**: DynamicRankPruner uses `delta * 0.01` for corrections — the accumulated corrections are dwarfed by marginal probabilities.
3. **Acceptance is verification-limited**: The simulated verifier accepts/rejects based on whether tokens match the context's "hot zone". The pruner can't change which tokens are objectively correct — it can only influence which tokens the tree prefers to explore.

## GOAT Gate
✅ **Passed**: DynamicRankPruner does not degrade acceptance rate (within -0.01pp, tolerance: -10pp).

## Recommendation
DynamicRankPruner is **safe to merge** (no regression), but the acceptance rate benchmark shows it's **not yet impactful** as a default-ON feature. The static ranking detection works correctly, but the correction mechanism needs:
- Larger correction learning rate or adaptive rate
- Corrections that actually alter which branches the DDTree explores (currently marginals dominate)
- Potentially a feedback loop where corrections influence tree construction, not just relevance scoring
