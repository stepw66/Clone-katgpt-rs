# Plan 170: LDT Phase 2 — Lattice State Fusion

> **Research:** [152_LDT_Phase2_Lattice_State_Fusion.md](../.research/152_LDT_Phase2_Lattice_State_Fusion.md)
> **Source:** [Lattice Deduction Transformers](https://arxiv.org/pdf/2605.08605) — Davis, Haller, Alfarano, Santolucito (2026)
> **Phase 1:** Plan 088 (GOAT 7/7, default-on)
> **Feature Gate:** `lattice_deduction` (existing, already default-on)
> **Type:** Modelless (zero training for F1-F4)
> **Priority:** P1 (F1, F3) / P2 (F2, F4)

## Tasks

- [x] **F1: AlphaScreeningPruner** — α-operator as ScreeningPruner impl (sound by construction)
- [x] **F2: ConflictClauseDB** — CDCL-style clause learning from failed branches
- [x] **F3: Cached ŷ_prev** — Target stabilization when no solution remains consistent
- [x] **F4: Depth-Escalating Conflict Threshold** — Adaptive max_prune_rate by search depth
- [x] **F5: GOAT Proof** — Extended proofs for all four fusions
- [x] **F6: Feature Gate Audit** — Zero impact on default build

---

## Context

Phase 1 (Plan 088) distilled three direct paper contributions: asymmetric threshold (T1), conflict detection (T2), α-operator (T3). Phase 2 creates **fusion ideas** that combine LDT's abstract interpretation framework with our existing architecture in novel ways:

1. AlphaTarget (T3) → ScreeningPruner trait impl → sound multi-solution pruning
2. ConflictDetector (T2) + CDCL clause learning → search acceleration
3. AlphaTarget edge case fix → cached previous valid target
4. EntropyConflictDetector (T2) → depth-adaptive threshold escalation

All behind existing `lattice_deduction` feature gate (already default-on).

---

## F1: AlphaScreeningPruner

### What

Implement `ScreeningPruner` on top of `AlphaTarget`. When the α-target says a token is allowed at a position, relevance = 1.0. When it's not allowed, relevance = 0.0. This makes multi-solution pruning **sound by construction** — the pruner never eliminates a token that appears in any surviving solution.

### Where

`src/speculative/alpha.rs` — new struct behind `lattice_deduction`

### Implementation

```rust
/// α-operator as ScreeningPruner — sound multi-solution pruning.
///
/// Uses AlphaTarget's candidate sets as a binary pruning signal:
/// - Token in α-target → relevance 1.0 (allowed)
/// - Token not in α-target → relevance 0.0 (pruned)
///
/// This makes the pruner sound by construction: it never prunes
/// a token that appears in any solution still consistent with
/// the current search state.
pub struct AlphaScreeningPruner {
    target: RefCell<AlphaTarget>,
}

impl ScreeningPruner for AlphaScreeningPruner {
    fn relevance(&self, depth: usize, token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        if self.target.borrow_mut().is_allowed(depth, token_idx) {
            1.0
        } else {
            0.0
        }
    }
}
```

### Performance

- `RefCell::borrow_mut` — zero-cost in single-threaded (no runtime check in release)
- `HashSet::contains` — O(1) amortized
- Total: ~50-100ns per relevance call (well under 1µs budget)

### Proof

Benchmark: Multi-solution maze DDTree
- Baseline: NoScreeningPruner
- F1: AlphaScreeningPruner with K=4 solutions
- Measure: tokens pruned, false prune rate, solve rate

---

## F2: ConflictClauseDB

### What

When a DDTree branch is flagged conflicted by `ConflictDetector`, extract the commitment pattern and store as a "learned clause." Future expansions skip branches whose commitments are a superset of any learned clause. This is CDCL-style clause learning for DDTree.

### Where

`src/speculative/alpha.rs` — new struct behind `lattice_deduction`

### Implementation

```rust
/// A learned conflict clause — a set of commitments known to cause conflict.
///
/// A state violates this clause if ALL its commitments are present.
/// This is the unit propagation rule from CDCL SAT solvers.
pub struct ConflictClause {
    commitments: HashSet<(usize, usize)>,
}

/// Database of learned conflict clauses for DDTree search acceleration.
///
/// When a branch fails, extract its commitments and learn a clause.
/// Future expansions check against all clauses before exploring.
pub struct ConflictClauseDB {
    clauses: Vec<ConflictClause>,
    max_clauses: usize,
}
```

### Performance

- Clause check: O(k × c) where k = clauses, c = avg clause size
- With max_clauses = 64 and avg clause size = 4: ~256 comparisons ≈ 200ns
- Learn operation: O(1) amortized (Vec push)
- Bounded by max_clauses to prevent unbounded growth

### Proof

Benchmark: Sudoku DDTree with conflict clause learning
- Baseline: ConflictDetector only (no clause learning)
- F2: ConflictDetector + ConflictClauseDB (max_clauses = 64)
- Measure: branches explored, clauses learned, solve rate

---

## F3: Cached ŷ_prev Target Stabilization

### What

When `AlphaTarget.remaining_solutions() == 0`, the α-target currently returns empty sets. LDT caches the last non-empty target and uses it instead. This provides stable signal even in conflict states.

### Where

`src/speculative/alpha.rs` — modify `AlphaTarget`

### Implementation

Add `cached_prev: Option<Vec<HashSet<usize>>>` to `AlphaTarget`. When `alpha_intersect` returns non-empty, cache it. When it returns empty, use the cached previous target.

```rust
pub struct AlphaTarget {
    current: Vec<Option<usize>>,
    solutions: Vec<Vec<usize>>,
    cached_target: Option<Vec<HashSet<usize>>>,
    cached_prev: Option<Vec<HashSet<usize>>>,  // NEW
}
```

### Proof

Unit test: After committing past all solutions, target returns last valid target (not empty).

---

## F4: Depth-Escalating Conflict Threshold

### What

Make `EntropyConflictDetector.max_prune_rate` depth-adaptive. At shallow depths (early search), be more lenient. At deeper depths (committed search), be more aggressive. This mirrors LDT's θ_eval_CLS > θ_train_CLS insight.

### Where

`src/speculative/types.rs` — extend `EntropyConflictDetector`

### Implementation

Add `depth_escalation: f32` field. In `is_conflicted`, compute effective max_prune_rate as `base - depth * escalation`, clamped to [0.1, base].

```rust
pub struct EntropyConflictDetector {
    pub max_prune_rate: f32,       // base rate (0.6)
    pub entropy_floor: f32,        // minimum entropy (0.01)
    pub depth_escalation: f32,     // rate decrease per depth (0.02)
}

// In is_conflicted (with depth parameter):
let effective_max = (self.max_prune_rate - depth as f32 * self.depth_escalation)
    .max(0.1)
    .min(self.max_prune_rate);
```

Note: Requires adding `depth: usize` parameter to `ConflictDetector::is_conflicted`.

### Proof

Unit test: At depth 0, threshold is max_prune_rate. At depth 20, threshold is tighter.

---

## F5: GOAT Proof

### Setup

Extended GOAT proof in `tests/bench_ldt_lattice_deduction.rs`:

| Proof | Config | Expected |
|-------|--------|----------|
| F1 | AlphaScreeningPruner + K=4 solutions | Zero false prunes |
| F2 | ConflictClauseDB + max_clauses=64 | Fewer branches vs baseline |
| F3 | AlphaTarget with ŷ_prev cache | Non-empty target after conflict |
| F4 | Depth-escalating threshold | Tighter at depth 20 vs depth 0 |

### Metrics

1. **F1:** False prune rate = 0% (sound by construction)
2. **F2:** Branches explored ≤ baseline
3. **F3:** Target non-empty after committing past all solutions
4. **F4:** Conflict threshold at depth 20 < threshold at depth 0

### Benchmark File

`.benchmarks/019_ldt_phase2_lattice_fusion.md`

---

## F6: Feature Gate Audit

### Checklist

- [x] All new code behind `#[cfg(feature = "lattice_deduction")]`
- [x] `cargo build` (no features) succeeds with zero warnings
- [x] `cargo build --features lattice_deduction` succeeds
- [x] `cargo test --features lattice_deduction` passes
- [x] No performance regression on default build

---

## Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │  Phase 1 (Plan 088)           │
                    │  T1: θ_elim ≈ 0.111          │
                    │  T2: EntropyConflictDetector   │
                    │  T3: AlphaTarget + α-operator  │
                    └──────────────┬────────────────┘
                                   │
                 ┌─────────────────┼──────────────────┐
                 │                 │                   │
    ┌────────────▼──────┐  ┌──────▼──────────┐  ┌───▼────────────────┐
    │ F1: AlphaScreen   │  │ F2: Conflict     │  │ F3: Cached ŷ_prev  │
    │ Pruner            │  │ ClauseDB         │  │                    │
    │ α → Screening     │  │ Failed branch    │  │ Stable target      │
    │ Pruner trait      │  │ → learned clause │  │ when solutions = 0 │
    │ (sound by const.) │  │ (CDCL for DDTree)│  │                    │
    └───────────────────┘  └─────────────────┘  └────────────────────┘
                 │                 │                   │
                 └─────────────────┼──────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │ F4: Depth-Escalating Threshold  │
                    │ Conflict detection tightens     │
                    │ as search deepens               │
                    └───────────────────────────────┘
                                   │
                    ┌──────────────▼────────────────┐
                    │  DDTree + MCTS                  │
                    │  Enhanced search pipeline       │
                    │  Feature: lattice_deduction     │
                    └───────────────────────────────┘
```

---

## File Changes Summary

| File | Change | Type |
|------|--------|------|
| `src/speculative/alpha.rs` | `AlphaScreeningPruner`, `ConflictClauseDB`, `cached_prev` on AlphaTarget | Enhancement |
| `src/speculative/types.rs` | `depth_escalation` on `EntropyConflictDetector`, depth param on `ConflictDetector` | Enhancement |
| `src/speculative/mod.rs` | New re-exports | Enhancement |
| `tests/bench_ldt_lattice_deduction.rs` | Extended GOAT proofs for F1-F4 | Test |

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | F1 + F3 (AlphaScreeningPruner + cached ŷ_prev) | Code + unit tests |
| 2 | F2 (ConflictClauseDB) | Code + unit tests |
| 3 | F4 (Depth-escalating threshold) | Code + trait change |
| 4 | F5 + F6 (GOAT proofs + audit) | Benchmark results |

---

## GOAT Proof Results

```
═══════════════════════════════════════════════════════════
  LDT Phase 2 — GOAT Proof (Plan 170)
═══════════════════════════════════════════════════════════

  F1: AlphaScreeningPruner: sound pruning by construction ✓
  F2: ConflictClauseDB: clause learning + FIFO eviction ✓
  F3: Cached ŷ_prev: stable target after conflict ✓
  F4: Depth-escalating threshold: tighter at depth 20 vs depth 0 ✓

═══════════════════════════════════════════════════════════
  GOAT PROOF COMPLETE — All 4 fusions verified
═══════════════════════════════════════════════════════════
```

Unit tests: 25/25 pass (including all Phase 1 + Phase 2 tests)
Feature gate: `lattice_deduction` (already default-on)

---

## References

- Paper: https://arxiv.org/pdf/2605.08605
- Phase 1: Research 050, Plan 088 (GOAT 7/7)
- Research: `.research/152_LDT_Phase2_Lattice_State_Fusion.md`
- Related: Plan 049 (G-Zero), Plan 057 (HLA), Plan 061 (Entropy), Plan 067 (NFSP/MCTS)
- riir-ai companion: Research 040, Plan 186 (DEFERRED)
