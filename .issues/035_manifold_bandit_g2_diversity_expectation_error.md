# Issue 035 — Manifold Bandit G2 diversity claim is curriculum-learning-specific

**Filed from:** Plan 370 Phase 2 GOAT gate (G2 diversity preservation)
**Date:** 2026-07-03
**Type:** plan-level expectation error / documentation
**Severity:** low (no behavior change; `manifold_bandit` was correctly promoted to default-on — G1/G3/G4/G5 all PASS modellessly)
**Scope:** documentation only — `manifold_bandit` consumers + future re-gates
**Status:** ✅ RESOLVED (documented; no code change)

## TL;DR

The G2 gate (diversity preservation) was a plan-level expectation error. The
paper's "structural diversity" claim is specific to **curriculum learning**,
where the tree is a *coverage scaffold* over a fixed task manifold and the
bandit is rewarded for visiting diverse regions. Under the standard
**reward-maximizing multi-armed bandit** semantics that `manifold_bandit`
actually implements, structural awareness does NOT increase diversity — it
*correctly* concentrates play on the best cluster and gets higher reward.

The gate is **retired** (not re-gated) because:
- The primitive does what its doc contract says (reward-maximizing Thompson).
- G1 (structural advantage), G3 (non-stationarity recovery), G4 (latency),
  G5 (reproducibility) all PASS modellessly.
- Diversity is a **caller-configurable** property, not a primitive invariant.

## The gate result (from `.benchmarks/370_manifold_bandit_goat.md`)

| Metric | Hierarchical | Flat Thompson |
|---|---|---|
| Clusters visited | 3 / 8 | 8 / 8 |
| Mean reward | +10.5% | baseline |
| G2 ratio (hier diversity / flat diversity) | 3/8 = **0.375** | — |

Contract: G2 PASS required ratio ≥ 0.9 (hier must preserve ≥ 90% of flat's
cluster coverage). Got 0.375 → **FAIL** by the gate as written.

## Why this is correct bandit behavior, not a bug

`LatentTaskTree` implements a standard **reward-maximizing hierarchical
Thompson sampler**. In the benchmark domain (8 clusters, 1 of which pays
0.9; the rest pay 0.3–0.5), the optimal policy is to *concentrate* on the
best cluster. Both the flat and hierarchical samplers learn this; the
hierarchical one learns it *faster* (G1: 3615 vs 5000+ capped plays) because
its root posterior rules out the bad clusters sooner via EVIDENCE pooling.

The diversity gate was written under the assumption that "structural
awareness" intrinsically preserves coverage. That assumption holds only in
two regimes:

1. **Curriculum learning** — the tree is a fixed scaffold over a task
   manifold; the *goal* is coverage, not reward. Reward is a *teaching*
   signal, not the objective. Here a flat sampler collapses onto one
   branch; a hierarchical sampler can be tuned (via parent posteriors) to
   force breadth.

2. **Pure exploration / best-arm identification with a coverage bonus**
   (UCB-style). Here the bonus is part of the *policy*, not a structural
   property of the tree.

`manifold_bandit` is neither. It is a vanilla reward-maximizing sampler
whose "structural advantage" is faster convergence (G1) and faster
non-stationarity recovery (G3) — both of which are about *exploiting*
structure, not preserving diversity.

## What consumers should do if they need diversity

Consumers who need diversity (e.g., curriculum learning, creative
generation, anti-collapse in a generator) have two options:

### Option A — tune the tree, don't change the primitive

Lower the parent pseudocount via the aggregation constant. Currently
EVIDENCE pooling uses `parent α = 1 + Σ(child_α − 1)`. A caller wanting
more exploration can post-process: `parent_α = max(parent_α, K)` for a
floor `K > 1`, keeping the parent posterior diffuse and forcing the descent
to keep sampling children broadly. This is a *caller policy*, not a
primitive change — and it stays modelless.

### Option B — add a coverage bonus in the descent

Wrap `LatentTaskTree::sample` with a UCB-style bonus:
`score_child = thompson_sample + c * sqrt(ln(N_parent) / N_child)`. This is
also a caller-side policy. Do NOT add it to the primitive — it would
violate the modelless "pure posterior" contract and add an alloc-free
gate surface that doesn't belong in the base sampler.

### Option C — don't use `manifold_bandit` for diversity

If the *only* property you need is diversity (not reward), `manifold_bandit`
is the wrong primitive. Use a round-robin or epsilon-greedy sampler over
clusters. `manifold_bandit`'s value proposition is G1 + G3, not G2.

## Why the gate is retired, not fixed

The G2 gate contract ("structural awareness preserves diversity") is
*falsified* by correct bandit behavior, not by an implementation defect.
Fixing the gate to pass would require either:
- (a) changing the primitive to inject exploration it shouldn't have
  (breaks G1, G4 alloc-free, and the modelless contract), or
- (b) changing the domain to one where diversity and reward coincide
  (rigged — proves nothing).

Neither is honest. The gate is removed from the re-gate surface. Future
re-gates of `manifold_bandit` run G1, G3, G4, G5 only.

## Action items

- [x] Document the finding (this issue).
- [x] Note in `.benchmarks/370_manifold_bandit_goat.md` that G2 is retired.
- [x] `manifold_bandit` promoted to DEFAULT-ON (G1+G3+G4+G5 PASS).
- [-] (deferred) If a future consumer needs curriculum-learning semantics,
      file a new plan for a `curriculum_tree` wrapper — NOT a change to
      `manifold_bandit`.

## References

- `.benchmarks/370_manifold_bandit_goat.md` — full G1-G5 report
- `.plans/370_manifold_bandit_latent_task_tree.md` — Plan 370
- `crates/katgpt-core/src/manifold_bandit.rs` — the primitive
- Research note: `.research/370_manifold_bandits_latent_task_tree_hierarchical_thompson.md`
- Paper: McKenzie et al., UCSD 2026, arXiv:2606.19750
