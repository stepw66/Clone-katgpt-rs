# Issue 044: State-Action Cache G2 Budget-Expansion Gap (Plan 390 opt-in-forever)

**Date:** 2026-07-07
**Plan:** 390 (State-Action Pair Cache for MCTS over Deterministic Inference Actions)
**Research:** 386 (UnMaskFork)
**Status:** Open — opt-in-forever, re-gate contingent on a real dLLM PoC

## Problem

The Plan 390 Phase 3 GOAT gate **FAILED on G2** (effective-budget expansion).
The primitive is correct (G1/G3/G5 PASS) but the synthetic benchmark domain
is too small to show the budget-expansion benefit that motivates the primitive.

## GOAT Gate Results

| Gate | Target | Result | Verdict |
|---|---|---|---|
| G1 — cache hit rate | ≥30% at NFE≥1024 | **42.1%** at NFE=1024 | ✅ PASS |
| G2 — budget expansion | ≥1.4× | **1.00×** | ❌ FAIL |
| G3 — no-regression | cached ≥ no-cache | identical at every NFE | ✅ PASS |
| G4 — zero-alloc | 0 allocs/Expand | DEFERRED | ⏸️ |
| G5 — cache bounded | O(NFE × actions) | 107 entries at NFE=8192 | ✅ PASS |

## Why G2 Failed (Root Cause)

The synthetic domain (16-token sequences, 7 depth levels, 3 actions) is too
small: the search converges to the optimal trajectory (0.875 reward) at the
**minimum NFE (256)** in both the cached and no-cache arms. There is no
reward-convergence-speedup signal because the domain's state space is fully
explored before the cache's cumulative savings matter.

This is a **benchmark-methodology limitation**, not a primitive defect:
- G1 proves the cache IS being used (42% hit rate — the cache converts
  deterministic-transition revisits into zero-NFE hits).
- G3 proves the cache never hurts (identical reward at every NFE).
- G5 proves the cache is memory-bounded (107 entries, 0.013× NFE).

The cache works; the benchmark just can't show the budget-expansion benefit
on a domain this small.

## What Would Fix G2

### Option A: Larger synthetic domain
Scale the domain to a size where the search does NOT converge at minimum NFE.
For example: 128-token sequences, 20+ depth levels, 6+ actions. This would
make the state space large enough that the cache's cumulative savings manifest
as faster reward convergence.

### Option B: Real dLLM PoC (Plan 5, deferred)
Wire the cache into the D2F inference pipeline (Plan 066) and measure on a
real (small) dLLM. UMF reports 47.8% hit rate at NFE=3072 on coding tasks —
a real dLLM's state space is large enough that the budget-expansion benefit
is measurable. This is the Plan 5 path, contingent on a concrete consumer.

### Option C: Direct NFE-savings metric (methodology change)
Instead of measuring reward-convergence speedup (G2's current definition),
measure the raw NFE-equivalent saved by cache hits: `cache_hits × avg_rollout_depth`.
This directly quantifies the cache's value without requiring a domain large
enough for reward-convergence differences. This is the most honest metric but
deviates from the plan's G2 definition.

## Decision

**Opt-in-forever.** The primitive ships as a correct, bounded, diagnostic
caching layer behind the `mcts_state_action_cache` feature flag. It is NOT
promoted to default-on (the GOAT gate's headline G2 metric failed).

The verdict in Research 386 is revised from **GOAT (pending benchmark)** to
**Gain** — the primitive is a genuine modelless caching improvement (G1/G3/G5
prove it works), but the ≥1.4× budget-expansion GOAT threshold was not met on
the synthetic domain and cannot be re-tested without a larger domain or real
dLLM PoC.

## Re-gate Conditions

Re-open this issue (and potentially re-run Plan 390 Phase 3) when ANY of:
1. A larger synthetic domain is constructed (Option A above) that shows the
   search does not converge at minimum NFE.
2. A real dLLM PoC (Plan 5) is unblocked and confirms the cache hit rate
   transfers to a real dLLM-scale state space.
3. The G2 metric is redefined to direct NFE-savings (Option C) and the
   re-gate shows ≥1.4× NFE-equivalent savings.

Until then, the primitive remains opt-in and the plan is closed.
