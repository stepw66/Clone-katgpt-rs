# Plan 424 Phase 6 — DDTree Deep-Argmax Acceptance Benchmark

**Date:** 2026-07-10
**Plan:** [katgpt-rs/.plans/424_gdn_tree_verification_primitive.md](../.plans/424_gdn_tree_verification_primitive.md) (Phase 6, T6.2)
**Paper:** arXiv:2607.06763 §3.5 / Figure 6 — Oda et al., "Trees from Marginals"
**Bench:** `crates/katgpt-speculative/benches/bench_424_dd_tree_deep_argmax.rs`

## What was tested

The paper §3.5 / Figure 6 claims that at deep draft positions (draft length > ~4),
the factorized marginal is diluted by averaging over many possible prefixes, and
**argmax-of-marginal outperforms full-marginal branching** on mean accepted
prefix length. The crossover occurs around draft length 2–4.

Plan 424 Phase 6 added `deep_argmax_threshold: Option<usize>` to `TreeBuilder`
(T6.1, sibling agent): when `Some(t)`, tree expansion beyond depth `t` uses
argmax-of-marginal (single greedy child per node) instead of pushing all valid
tokens. This benchmark (T6.2) tests whether `Some(4)` improves mean accepted
prefix length.

## Method

Synthetic acceptance-length proxy (katgpt-rs has no target model):

1. Generate a ground-truth token sequence `gt` (the target's greedy decode).
2. Generate draft marginals: at depth `d`, `p[gt[d]] = signal(d)` decays linearly
   (`base - slope·d`, clamped to `floor`); remaining mass spread over 3 decoys.
3. Build DDTree with `None` vs `Some(2)` vs `Some(4)`.
4. "Accepted prefix length" = longest prefix of `gt` that appears as a root→leaf
   path in the built tree.

Two regimes:
- **slow-decay** (base=0.85, slope=0.05, floor=0.40): argmax reliably correct at
  all depths.
- **fast-decay** (base=0.70, slope=0.09, floor=0.05): deep argmax barely above
  decoys.

Config: vocab=32, draft_lookahead=8, tree_budget=12 (tight — budget < vocab so
full branching can't reach deep). 200 seeds per cell.

## Results

### slow-decay (base=0.85, slope=0.05, floor=0.40)

| threshold | mean accept len | mean tree depth | tree size |
|-----------|-----------------|-----------------|-----------|
| None      | 7.000           | 3.366           | 12.0      |
| Some(2)   | 7.000           | 3.333           | 12.0      |
| Some(4)   | 7.000           | 3.417           | 12.0      |

### fast-decay (base=0.70, slope=0.09, floor=0.05)

| threshold | mean accept len | mean tree depth | tree size |
|-----------|-----------------|-----------------|-----------|
| None      | 4.000           | 2.083           | 12.0      |
| Some(2)   | 4.000           | 2.083           | 12.0      |
| Some(4)   | 4.000           | 2.083           | 12.0      |

## Verdict: NO GAIN (threshold is redundant on best-first DDTree)

**The threshold does not improve acceptance length in any tested regime.** All
thresholds produce identical mean acceptance (7.0 slow, 4.0 fast).

### Root cause

The DDTree builder uses **best-first expansion** (max-heap on cumulative
log-prob). The argmax token at each depth has the highest marginal probability
→ highest cumulative log-prob → gets popped first → extends the chain. Siblings
are also pushed to the heap but never win the pop race for the best path because
their log-prob is lower.

The threshold restricts WHICH siblings get pushed, but since siblings never win
the pop anyway, the restriction has no effect on the accepted path. The best-first
heap already follows the argmax chain as deep as budget allows.

### Why the paper's crossover doesn't apply here

The paper's §3.5 / Figure 6 argmax-vs-marginal crossover applies to tree builders
that **sample** from the marginal (stochastic expansion). In a sampling-based
builder, full-marginal branching explores diverse tokens (some of which diverge
from the target), while argmax stays on the most-likely path. At deep positions
where the marginal is noisy, the sampled branches waste budget → argmax wins.

The katgpt DDTree does **deterministic best-first expansion**, not sampling. The
heap's log-prob scoring is already a form of soft-argmax — it deterministically
follows the highest-probability path. There is no stochastic divergence for the
threshold to correct.

### Implication for the flag

- `deep_argmax_threshold` is **correct** (verified by 4 unit tests: identity,
  restriction, boundary, builder-setter parity).
- It is **harmless** (default `None` = byte-identical to no-flag behavior).
- It provides **no gain** on the current best-first DDTree.
- A **future sampling-based builder** (if added) could benefit from the flag.

**Decision:** Keep the flag (opt-in via `None` default, no default-on promotion
path). Do NOT promote to any config default — there is no modelless gain to
justify it. Document the redundancy honestly.

## Reproduce

```bash
CARGO_TARGET_DIR=/tmp/424_dd_argmax \
  cargo bench -p katgpt-speculative --bench bench_424_dd_tree_deep_argmax -- --nocapture
```

## TL;DR

Phase 6 T6.1 (deep_argmax_threshold flag) is implemented and tested. Phase 6 T6.2
benchmark shows **no acceptance-length gain** on the best-first DDTree — the
heap's log-prob scoring already follows the argmax chain. The flag is kept as an
opt-in no-op default for future sampling-based builders; not promoted.
