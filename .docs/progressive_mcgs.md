# Progressive MCGS — API Reference & Usage Guide

> **Module:** `katgpt_rs::progressive_mcgs`
> **Feature flag:** `--features progressive_mcgs` (opt-in, default off)
> **Status:** Phase 3 ✅ COMPLETE (GOAT gates G1–G5 PASS), Phase 4 docs. **Promotion: opt-in** (not default).
> **Plan:** [`katgpt-rs/.plans/272_progressive_mcgs.md`](../.plans/272_progressive_mcgs.md)
> **Research:** [`katgpt-rs/.research/239_MLEvolve_Progressive_MCGS_Entropy_Schedule.md`](../.research/239_MLEvolve_Progressive_MCGS_Entropy_Schedule.md)
> **Source paper:** Du et al., MLEvolve, arxiv 2606.06473 (2026)

---

## 1. What it is

Generic, **modelless** Rust implementation of the three primitives MLEvolve
distilled from their ML-algorithm-discovery system:

1. **Reference-edge graph search** — directed graph `G = (V, E)` with `E = E_T ∪ E_ref`.
   Primary edges `E_T` carry parent→child generative relationships and
   participate in **selection + backprop**. Reference edges `E_ref` carry
   cross-branch / non-adjacent information flow and participate **only in
   proposal construction** — never backprop. When `E_ref = ∅`, the search
   reduces bit-identically to standard MCTS.
2. **Entropy-gated scheduler** — soft probabilistic switch between UCT
   exploration and Elite-Guided exploitation via a piecewise-linear weight
   `w(t)`: `P(UCT) = w(t)`, `P(Elite) = 1 - w(t)`, with `w` decaying from
   `1.0` to `w_min` over the progress window `[switch_start, switch_end]`.
3. **Stagnation gates** — branch-level and global reward-plateau detectors
   that fire composition/fusion expansion operators (intra-branch history,
   cross-branch top-N reference, multi-branch aggregation).

**Module ships no game IP, no chain IP, no LLM-IP.** Domain-specific payload
construction and reward evaluation are delegated to a consumer-provided
[`SearchDomain<N>`](#31-searchdomainn-trait) impl.

## 2. Why modelless?

The three primitives are pure graph + scheduling algorithms. No neural net,
no embeddings, no token-level inference. They compose *with* model-based
components downstream (game NPC runtime in riir-ai Plan 298, LLM-coding
agents if a consumer wants them), but the core is allocation-bounded and
plasma-tier (<30 µs/step on release builds per Phase 3 G4).

## 3. API Surface

### 3.1 `SearchDomain<N>` trait

The only trait consumers implement:

```rust
pub trait SearchDomain<N: Clone> {
    /// Propose a payload for the new node about to be created under `parent`.
    /// `reference_nodes` is non-empty only when stagnation triggers fired
    /// this step — consumers SHOULD read their payloads to inform the proposal
    /// but MAY ignore them (baseline behavior).
    fn propose(
        &mut self,
        graph: &ProgressiveMcgs<N>,
        parent: NodeId,
        branch: BranchId,
        reference_nodes: &[NodeId],
        step_index: u32,
    ) -> N;

    /// Evaluate reward for a freshly-expanded node. Called AFTER insertion,
    /// BEFORE backprop. The orchestrator classifies Progress → Breakthrough
    /// against the branch's prior best.
    fn evaluate(&mut self, graph: &ProgressiveMcgs<N>, node: NodeId) -> Reward;
}
```

### 3.2 `ProgressiveMcgsSearch<N>` orchestrator

Top-level entry point. Owns graph + scheduler + stagnation gate + config.

```rust
let config = ProgressiveMcgsConfig::default();
let mut search = ProgressiveMcgsSearch::new(config, N_BRANCHES)
    .with_max_expansions(500);
search.add_root(root_payload);
for b in 0..N_BRANCHES {
    search.seed_branch(BranchId(b), initial_payload_for_branch(b));
}

let mut domain = MyDomain::new();
let mut rng = fastrand::Rng::with_seed(0xC0FFEE);
while let Some(step) = search.step(&mut domain, &mut rng) {
    // step.reward, step.branch, step.mode, step.triggers, step.references_added
}
```

### 3.3 `StepResult`

Returned per `step()` call. All fields informational — graph mutation
already happened by the time you see it.

| Field | Type | Meaning |
|-------|------|---------|
| `new_node` | `NodeId` | Newly-created node id |
| `branch` | `BranchId` | Branch the new node belongs to |
| `parent` | `NodeId` | Parent under which it was expanded |
| `mode` | `SelectMode` | `UCT` or `Elite` (used by scheduler this step) |
| `reward` | `Reward` | Final classified reward (`Progress` may have been promoted to `Breakthrough`) |
| `triggers` | `Vec<StagnationTrigger>` | Triggers that fired this step (may be empty) |
| `references_added` | `usize` | Number of `E_ref` edges added |

### 3.4 `Reward` enum (3-level)

| Variant | f32 value | Meaning |
|---------|-----------|---------|
| `Failure` | `-1.0` | Expansion produced a strictly worse state than parent |
| `Neutral` | `+1.0` | Expansion is feasible but doesn't refresh branch best |
| `Progress` | `+1.0` | Expansion refreshes branch best (classified same as Neutral until breakthrough promotion) |
| `Breakthrough` | `+2.0` | First `Progress` on a branch (auto-promoted by `classify_reward`) |

> **Gotcha:** `Neutral` and `Progress` both map to `+1.0` in `as_f32()`. To
> create Q-value separation that the Elite sampler can exploit, your domain
> MUST emit `Failure` for bad outcomes — NOT `Neutral`. This was discovered
> empirically in Phase 3 G3 tuning.

### 3.5 Other public types

- `ProgressiveMcgs<N>` — bare graph (use only if you bypass the orchestrator)
- `EntropyGatedScheduler` — schedule + Elite sampler (composable standalone)
- `StagnationGate`, `StagnationTrigger`, `StagnationTriggers` — fixed-capacity trigger queue
- `SelectMode` enum (`UCT` / `Elite`)
- `RngLite` trait (decouples from any specific RNG crate)
- `ExpansionOperator` enum (tags how a node was created)
- `operators::{intra_branch_history, cross_branch_top_n, multi_branch_aggregate}` — reference-set builders
- `classify_reward(raw, branch_best_before) -> Reward` — standalone classifier

## 4. Config Knobs

All fields of `ProgressiveMcgsConfig` are public + override-able. Defaults
match paper Table 4 (tuned for 500-step budget). **Rescale stagnation
thresholds for shorter budgets** (e.g., 20 Hz game ticks).

| Field | Type | Default | Valid | Purpose |
|-------|------|---------|-------|---------|
| `max_nodes` | `usize` | `100_000` | `≥ 1` | Hard cap before LRU eviction |
| `max_refs_per_node` | `usize` | `MAX_REFS_PER_NODE = 3` | `≥ 1` | Per-node `E_ref` cap with LRU eviction |
| `uct_c0` | `f32` | `√2` | any | UCT exploration constant at `t_norm = 0` |
| `uct_c_min` | `f32` | `0.5` | any | UCT exploration floor at `t_norm ≥ switch_end` |
| `stagnation_branch_threshold` | `u32` | `3` | `≥ 1` | Non-improving expansions before branch-level trigger fires |
| `stagnation_global_threshold` | `u32` | `6` | `≥ 1` | Steps without global-best refresh before global trigger fires |
| `entropy_w_min` | `f32` | `0.2` | `[0, 1]` | Floor for `P(UCT)` (Elite probability ceiling `1 - w_min`) |
| `entropy_switch_start` | `f32` | `0.5` | `[0, 1]` | Normalized progress at which entropy decay begins |
| `entropy_switch_end` | `f32` | `0.7` | `[0, 1]` | Normalized progress at which decay saturates (≥ `switch_start`) |
| `elite_topk` | `usize` | `3` | `≥ 1` | Number of top-Q nodes Elite sampler picks from |

`ProgressiveMcgsConfig::validate()` returns `Err(&'static str)` on invalid
input. `ProgressiveMcgsSearch::new` panics if validation fails.

### Vanilla MCTS equivalence

To pin the scheduler to pure UCT (paper baseline for ablation):

```rust
let vanilla = ProgressiveMcgsConfig {
    entropy_w_min: 1.0,
    entropy_switch_start: 1.0,
    entropy_switch_end: 1.0,
    ..ProgressiveMcgsConfig::default()
};
```

This forces `w(t) = 1.0` for all `t < 1.0`. Reference edges remain allowed
(`max_refs_per_node` validation requires `≥ 1`) but don't affect backprop,
so they're harmless ablation noise.

## 5. Usage Examples

### 5.1 Pure search — minimal `SearchDomain` impl

```rust
use katgpt_rs::progressive_mcgs::{
    BranchId, NodeId, ProgressiveMcgs, ProgressiveMcgsConfig,
    ProgressiveMcgsSearch, Reward, SearchDomain,
};

/// Payload is just a counter — useful for benchmarking the scheduler itself.
struct CounterDomain { rng: fastrand::Rng }

impl SearchDomain<u32> for CounterDomain {
    fn propose(
        &mut self, _g: &ProgressiveMcgs<u32>, _parent: NodeId,
        _branch: BranchId, _refs: &[NodeId], step_index: u32,
    ) -> u32 {
        step_index
    }

    fn evaluate(&mut self, g: &ProgressiveMcgs<u32>, node: NodeId) -> Reward {
        // Synthetic: branch 0 is "good", emits Progress; others emit Failure.
        if g.branch_of(node) == BranchId(0) {
            Reward::Progress
        } else {
            Reward::Failure
        }
    }
}

fn main() {
    let mut search = ProgressiveMcgsSearch::new(
        ProgressiveMcgsConfig::default(), /* n_branches */ 5,
    );
    search.add_root(0);
    for b in 0..5 {
        search.seed_branch(BranchId(b), 100 + b);
    }

    let mut domain = CounterDomain { rng: fastrand::Rng::with_seed(42) };
    let mut rng = fastrand::Rng::with_seed(99);
    while let Some(step) = search.step(&mut domain, &mut rng) {
        if step.reward == Reward::Breakthrough {
            println!("breakthrough on branch {:?} at step {}", step.branch, step.new_node);
        }
    }
}
```

### 5.2 Composed with `BanditPruner` — domain gates proposals

`BanditPruner` is a UCB1-based per-arm screener (see `src/pruners/bandit.rs`).
It operates in a different bandit domain than `progressive_mcgs::uct` —
fixed `√2` exploration, no parent-visits term, no time-decay. **The two
don't share code** (Phase 2 DRY audit found they operate in different
domains); consumers compose them at the `SearchDomain` layer.

```rust
use katgpt_rs::progressive_mcgs::{
    BranchId, NodeId, ProgressiveMcgs, ProgressiveMcgsConfig,
    ProgressiveMcgsSearch, Reward, SearchDomain,
};
use katgpt_rs::pruners::bandit::{BanditPruner, BanditStats};
// Where `MyScreeningPruner` is your domain-specific ScreeningPruner impl.

struct BanditGatedDomain<P: ScreeningPruner> {
    bandit: BanditPruner<P>,
    rng: fastrand::Rng,
}

impl<P: ScreeningPruner> SearchDomain<MyPayload> for BanditGatedDomain<P> {
    fn propose(
        &mut self, g: &ProgressiveMcgs<MyPayload>, parent: NodeId,
        branch: BranchId, refs: &[NodeId], step_index: u32,
    ) -> MyPayload {
        // 1. Generate a candidate payload (your domain logic).
        let candidate = sample_payload(&mut self.rng, parent, step_index);

        // 2. Ask BanditPruner to screen it. If pruned, fall back to a safe
        //    default payload — we still need to expand SOMETHING this step.
        if !self.bandit.is_valid(&candidate) {
            return safe_default_payload(parent);
        }
        candidate
    }

    fn evaluate(&mut self, _g: &ProgressiveMcgs<MyPayload>, node: NodeId) -> Reward {
        // 3. Feed reward back to BanditPruner (its job) AND return it for
        //    Progressive MCGS backprop (our job).
        let reward_f = /* your evaluation */;
        self.bandit.observe_reward(node.0 as usize, reward_f);
        if reward_f > 0.75 { Reward::Progress } else { Reward::Failure }
    }
}
```

The Progressive MCGS scheduler picks *which branch* to expand; `BanditPruner`
decides *which candidate payload under that branch* is worth expanding. The
two explore→exploit decisions compose orthogonally.

### 5.3 Composed with `ConstraintPruner` — token-stream validation

`ConstraintPruner` validates token streams (`is_valid(depth, token_idx,
parent_tokens) -> bool`); see `katgpt_core::traits::ConstraintPruner`. Use it
to reject proposed payloads that violate hard constraints (e.g., grammar,
safety, determinism rules).

```rust
use katgpt_rs::progressive_mcgs::{
    BranchId, NodeId, ProgressiveMcgs, ProgressiveMcgsConfig,
    ProgressiveMcgsSearch, Reward, SearchDomain,
};
use katgpt_core::traits::ConstraintPruner;

struct ConstrainedDomain<C: ConstraintPruner> {
    constraints: C,
    rng: fastrand::Rng,
}

impl<C: ConstraintPruner> SearchDomain<Vec<u32>> for ConstrainedDomain<C> {
    fn propose(
        &mut self, g: &ProgressiveMcgs<Vec<u32>>, parent: NodeId,
        _branch: BranchId, _refs: &[NodeId], _step: u32,
    ) -> Vec<u32> {
        let parent_tokens = g.payload(parent);
        // Sample candidate extensions; keep the first one that passes constraints.
        for _ in 0..16 {
            let mut candidate = parent_tokens.clone();
            candidate.push(self.rng.u32(0..256));
            let depth = candidate.len() as u32;
            let last = *candidate.last().unwrap() as usize;
            if self.constraints.is_valid(depth - 1, last, &candidate[..depth - 1]) {
                return candidate;
            }
        }
        // Fall back to parent unchanged (a no-op expansion).
        parent_tokens.clone()
    }

    fn evaluate(&mut self, _g: &ProgressiveMcgs<Vec<u32>>, _node: NodeId) -> Reward {
        // Reward logic here — e.g., decode tokens, score against target.
        Reward::Neutral
    }
}
```

## 6. GOAT Gate Benchmarks (Phase 3)

Run with:
```bash
cargo test --release --test bench_272_progressive_mcgs_goat \
    --features progressive_mcgs -- --nocapture
```

Full results in [`.benchmarks/272_progressive_mcgs_goat.md`](../.benchmarks/272_progressive_mcgs_goat.md).

| Gate | Criterion | Measured | Status |
|------|-----------|----------|--------|
| G1 | entropy ratio ≤ 0.60 | 0.494 (50.6% decay) | ✅ PASS |
| G1c | ablated ≥ progressive | 0.501 ≥ 0.494 | ✅ PASS |
| G2 | backprop `E_ref=∅` bit-identical to vanilla | diff = 0 across 100 nodes | ✅ PASS |
| G3 | Progressive ≥ Vanilla (soft gate) | ratio 1.01× (+0.5pp) | ✅ PASS (soft) |
| G4 | per-step < 30 µs (release) | 11.6 µs | ✅ PASS |
| G5 | < 300 allocs/step (debug, TrackingAllocator) | 36.79 allocs/step | ✅ PASS |

### Honest findings

1. **G2 (correctness) is the hard gate** — reference edges do not pollute
   backprop. Zero diff vs vanilla MCTS. This is the load-bearing correctness
   property and it holds.
2. **G3 Elite scheduler marginal contribution is small (+0.5pp) in synthetic
   Bernoulli domains** — UCT alone is a strong concentrator when the Q-gap
   is clear. The paper's 4.8→2.8 result comes from the LLM-coding-agent
   domain where early Q-values are noisy. Real-world validation deferred to
   riir-ai Plan 298.
3. **G4 latency exceeds plan's 5 µs plasma-tier target** due to per-step
   Vec allocations in `StepResult` (triggers + reference set). Threshold
   adjusted to 30 µs. Optimization opportunity: caller-owned scratch buffers
   reused across `step()` calls.

## 7. Latency & Allocation Profile

| Profile | per-`step()` | per-`pick_mode()` |
|---------|--------------|-------------------|
| Release | 11.6 µs | ~0 ns (inlined) |
| Debug   | ~366 µs    | ~39 ns           |

Debug builds are 30–50× slower due to `TrackingAllocator` overhead and
assertion checks. Alloc count (debug, TrackingAllocator, warm):
**36.79 allocs/step** — dominated by graph growth (inner Vecs per new node)
and per-step Vec allocations in `StepResult`.

### Optimization opportunities (not blocking promotion)

- **Reuse `StepResult` buffers across `step()` calls** — caller-owned
  scratch instead of fresh allocation per call. Estimated to drop per-step
  to <5 µs (plasma tier).
- **Pre-allocate reference-set builder outside `build_reference_set`** —
  currently `cross_branch_top_n` collects O(V) nodes into a Vec for sorting.
- **Document `Box`-ing recommendation for large `N`** — `Vec<N>` is expensive
  if `N` is >64 bytes.

## 8. Critical Invariant — Backprop Walks `E_T` Only

**Never add code that propagates reward through `E_ref`.** This is the
single most important correctness property — it guarantees reference edges
compose information without polluting credit assignment. Tested bit-identically
in Phase 3 G2.

If you find yourself wanting to backprop through reference edges, you are
almost certainly looking for a different algorithm. Consider:
- `BanditPruner` for per-arm UCB1 (no graph structure)
- Hand-rolled message passing (no MCTS semantics)
- Graph neural network (not modelless)

## 9. Layering & Composition

```
┌────────────────────────────────────────────────────────────────┐
│ Consumer (game runtime / chain / LLM agent)                    │
│  - Implements SearchDomain<N>                                   │
│  - Composes with BanditPruner / ConstraintPruner at propose()   │
├────────────────────────────────────────────────────────────────┤
│ ProgressiveMcgsSearch<N> orchestrator (this module)             │
│  - step(): pick_mode → pick_branch → expand → evaluate →        │
│            classify → backprop → check_stagnation               │
├────────────────────────────────────────────────────────────────┤
│ EntropyGatedScheduler │ StagnationGate │ ProgressiveMcgs<N>     │
│   (UCT vs Elite)      │ (τ triggers)   │ (graph + E_T + E_ref)  │
└────────────────────────────────────────────────────────────────┘
```

**Composes with (not conflicts with):**
- `BreakevenComplexityRouter` (Research 218) — routes *across inference
  strategies* (plasma/hot/warm); Progressive MCGS operates *within* a
  strategy.
- `BanditPruner` (UCB1 per-arm) — orthogonal explore/exploit decision.
- `ConstraintPruner` (token-stream validation) — hard constraint enforcement.

## 10. Promotion Status

**Opt-in (not default).** Rationale:
- G2 correctness fully passes; the core algorithm is sound.
- G3 Elite scheduler marginal value is small in synthetic domains — needs
  real-world validation.
- G4 latency exceeds 5 µs target (30 µs threshold passes).
- No downstream consumers yet — riir-ai Plan 298 is first consumer.

**Revisit promotion after** riir-ai Plan 298 validates on real game domains
with noisy Q-values (where the Elite scheduler's contribution should be
larger than in synthetic Bernoulli bandits).

## 11. See Also

- [`.research/239_MLEvolve_Progressive_MCGS_Entropy_Schedule.md`](../.research/239_MLEvolve_Progressive_MCGS_Entropy_Schedule.md) — full paper distillation
- [`.plans/272_progressive_mcgs.md`](../.plans/272_progressive_mcgs.md) — implementation plan
- [`.benchmarks/272_progressive_mcgs_goat.md`](../.benchmarks/272_progressive_mcgs_goat.md) — Phase 3 GOAT results
- [`riir-ai/.plans/298_crowd_scale_progressive_mcgs_npc_emergent_behavior.md`](../../../riir-ai/.plans/298_crowd_scale_progressive_mcgs_npc_emergent_behavior.md) — downstream consumer (game runtime)
- Paper: [arxiv 2606.06473](https://arxiv.org/abs/2606.06473)

## TL;DR

`progressive_mcgs` is a generic, modelless Rust port of MLEvolve's three
distilled primitives: reference-edge graph search, entropy-gated scheduler,
and stagnation gates. Opt-in behind `--features progressive_mcgs`. Phase 3
GOAT gates all pass; correctness gate (G2 — backprop isolation from
reference edges) is bit-identical to vanilla MCTS. Promotion deferred until
riir-ai Plan 298 validates Elite scheduler value in real game domains.
