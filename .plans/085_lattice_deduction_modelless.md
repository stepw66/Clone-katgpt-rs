# Plan 085: Lattice Deduction Modelless Distillation

> **Research:** [50_LDT_Lattice_Deduction_Transformer.md](../.research/50_LDT_Deduction_Transformer.md)
> **Source:** [Lattice Deduction Transformers](https://arxiv.org/pdf/2605.08605) — Davis, Haller, Alfarano, Santolucito (2026)
> **Feature Gate:** `lattice_deduction`
> **Type:** Modelless (zero training for T1-T3)
> **Priority:** P1 (T1, T2) / P2 (T3)

## Tasks

- [ ] **T1: Asymmetric Pruning Threshold** — LDT θ_elim = 1/(1+w+/w−) ≈ 0.111
- [ ] **T2: ConflictDetector Trait** — Entropy-based conflict signal for early backtracking
- [ ] **T3: α-Operator for Multi-Solution** — Progressive intersection target for maze/Go
- [ ] **T4: Sudoku GOAT Proof** — DDTree + asymmetric threshold vs baseline
- [ ] **T5: Maze GOAT Proof** — α-operator multi-path vs single-path
- [ ] **T6: MCTS Conflict Cutoff Proof** — Early backtracking in Go arena
- [ ] **T7: Feature Gate Audit** — Zero impact on default build

---

## Context

LDT (800K params, 15min training on B200) achieves 100% on Sudoku-Extreme where frontier LLMs score 0%. The key insight: **operate on an interpretable lattice (not latent space) so deduction is structurally sound**.

Our `ConstraintPruner` + `ScreeningPruner` + `DDTree` is already this lattice. LDT adds:
1. **Asymmetric loss** — penalize false elimination 8× harder than false retention
2. **Conflict head** — separate signal for "this state is unsatisfiable"
3. **α-operator** — progressive multi-solution supervision

All three distill to modelless enhancements behind `lattice_deduction` feature gate.

---

## T1: Asymmetric Pruning Threshold

### What

LDT uses w+/w− = 8 in BCE loss, making the model conservative: only eliminate candidates when very confident. The equivalent modelless translation is adjusting the pruning threshold.

The natural threshold from asymmetric loss: `θ_elim = 1/(1 + w+/w−) = 1/9 ≈ 0.111`

### Where

`src/speculative/types.rs` — add constant and config option
`src/speculative/dd_tree.rs` — use threshold in expansion

### Implementation

```rust
// src/speculative/types.rs

/// LDT-derived asymmetric elimination threshold.
/// From w+/w− = 8: penalize false elimination 8× harder.
/// θ_elim = 1/(1 + w+/w−) ≈ 0.111
/// Only eliminate candidates when confidence is very high.
#[cfg(feature = "lattice_deduction")]
pub const LDT_THETA_ELIM: f32 = 1.0 / (1.0 + 8.0); // ≈ 0.111

/// Configuration for LDT-style asymmetric pruning.
#[cfg(feature = "lattice_deduction")]
#[derive(Debug, Clone)]
pub struct LdtPruneConfig {
    /// Elimination threshold (default: LDT_THETA_ELIM ≈ 0.111).
    pub theta_elim: f32,
    /// Whether to use asymmetric threshold (default: true).
    pub enabled: bool,
}

#[cfg(feature = "lattice_deduction")]
impl Default for LdtPruneConfig {
    fn default() -> Self {
        Self {
            theta_elim: LDT_THETA_ELIM,
            enabled: true,
        }
    }
}
```

In DDTree expansion: when `lattice_deduction` is enabled and `LdtPruneConfig.enabled`, use `theta_elim` instead of default pruning threshold.

### Proof

Benchmark: Sudoku speculative solve (examples/sudoku_03_tui.rs)
- Baseline: default DDTree threshold
- LDT: θ_elim ≈ 0.111
- Measure: solve rate, false prune count, total forward passes

---

## T2: ConflictDetector Trait

### What

LDT has a separate CLS sigmoid that fires on unsatisfiable states. We don't need a neural head for this — we can detect conflicts via entropy analysis.

When the model has eliminated too many candidates (entropy drops anomalously low), or when `ConstraintPruner` rejects all remaining candidates, the state is conflicted.

### Where

`src/speculative/types.rs` — new trait
`src/speculative/dd_tree.rs` — wire into expansion loop
`src/speculative/step.rs` — early termination signal

### Implementation

```rust
// src/speculative/types.rs

/// LDT-inspired conflict detection for early backtracking.
///
/// LDT uses a separate CLS sigmoid for conflict detection.
/// Our modelless translation: detect conflict via entropy/marginal analysis.
///
/// Returns true when the current search state is likely unsatisfiable,
/// triggering early backtracking instead of continued exploration.
#[cfg(feature = "lattice_deduction")]
pub trait ConflictDetector: Send + Sync {
    /// Check if the current state shows conflict signals.
    ///
    /// `marginals` — per-depth token probability distributions
    /// `pruned_count` — how many candidates were eliminated this step
    /// `total_candidates` — total candidates before pruning
    fn is_conflicted(
        &self,
        marginals: &[&[f32]],
        pruned_count: usize,
        total_candidates: usize,
    ) -> bool;
}

/// Entropy-based conflict detector.
///
/// Flags conflict when:
/// 1. Any position has zero valid candidates (hard conflict = ⊥)
/// 2. Pruning rate exceeds threshold (too aggressive = likely wrong path)
/// 3. Entropy drops below floor (overconfident = probably hallucinating)
#[cfg(feature = "lattice_deduction")]
pub struct EntropyConflictDetector {
    /// Maximum fraction of candidates that can be pruned in one step.
    /// LDT's conflict threshold θ_cls = 0.6 → analogous to 60% max prune rate.
    pub max_prune_rate: f32,
    /// Minimum entropy per position (below = conflict).
    pub entropy_floor: f32,
}

#[cfg(feature = "lattice_deduction")]
impl Default for EntropyConflictDetector {
    fn default() -> Self {
        Self {
            max_prune_rate: 0.6,  // LDT θ_cls = 0.6 analog
            entropy_floor: 0.01,  // Near-deterministic = suspicious
        }
    }
}
```

### Integration with DDTree

In DDTree expansion: after computing marginals and applying `ScreeningPruner`, check `ConflictDetector::is_conflicted()`. If true, don't expand this branch — treat as dead end and explore alternatives.

This gives **early backtracking** without needing to run the full forward pass to terminal state.

### Proof

Benchmark: MCTS Go arena (examples/go_*)
- Baseline: standard MCTS with random rollouts
- LDT: MCTS with entropy conflict cutoff
- Measure: average search depth, win rate vs random, forward pass count

---

## T3: α-Operator for Multi-Solution

### What

LDT's α-operator: `ŷ = x ⊓ α({y ∈ Y | y consistent with x})`

This intersects the current state with the union of all valid solutions still consistent with it. As search commits, the target tightens progressively.

### Where

`src/speculative/` — new module `alpha.rs` behind feature gate
`src/speculative/mod.rs` — conditionally include

### Implementation

```rust
// src/speculative/alpha.rs

//! LDT α-operator: progressive multi-solution supervision target.
//!
//! ŷ = x ⊓ α({y ∈ Y | y consistent with x})
//!
//! For domains with multiple valid solutions (maze shortest paths,
//! Go joseki variations), this provides a tightening target as
//! search commits to particular branches.

/// Check if solution y is consistent with current state x.
/// Consistent = every committed position in x matches y.
fn is_consistent(current: &[Option<usize>], solution: &[usize]) -> bool {
    current.iter().enumerate().all(|(i, opt)| {
        opt.map_or(true, |v| v == solution[i])
    })
}

/// LDT α-operator: intersect current state with union of consistent solutions.
///
/// `current` — current candidate state (Some = committed, None = open)
/// `solutions` — K pre-computed valid solutions
///
/// Returns: per-position candidate sets (bitfield) representing
/// the tightest sound target given current commitments.
pub fn alpha_intersect(
    current: &[Option<usize>],
    solutions: &[Vec<usize>],
) -> Vec<HashSet<usize>> {
    let consistent: Vec<&Vec<usize>> = solutions.iter()
        .filter(|sol| is_consistent(current, sol))
        .collect();

    // α: union of values at each position across consistent solutions
    let mut alpha: Vec<HashSet<usize>> = vec![HashSet::new(); current.len()];
    for sol in &consistent {
        for (i, &val) in sol.iter().enumerate() {
            alpha[i].insert(val);
        }
    }

    // ⊓: intersect with current commitments
    for (i, opt) in current.iter().enumerate() {
        if let Some(v) = opt {
            alpha[i].clear();
            alpha[i].insert(*v);
        }
    }

    alpha
}
```

### Application

1. **Maze**: Pre-compute K=16 shortest paths. At each DDTree step, compute α-target. Use as screening signal — only expand tokens consistent with at least one remaining path.

2. **Go**: Pre-compute K professional games from same opening. At each MCTS node, compute α-target. Use as heuristic bonus for moves consistent with professional play.

### Proof

Benchmark: 15×15 maze shortest path
- Baseline: DDTree with single target
- LDT: DDTree with α-target (K=16 paths)
- Measure: convergence speed, solve rate, forward pass count

---

## T4: Sudoku GOAT Proof

### Setup

Use existing Sudoku speculative solver (`examples/sudoku_03_tui.rs`).

| Config | Threshold | Conflict Detect | Expected |
|--------|-----------|-----------------|----------|
| Baseline | Default (0.5) | Off | Current performance |
| LDT-T1 | θ_elim ≈ 0.111 | Off | Fewer false prunes |
| LDT-T1+T2 | θ_elim ≈ 0.111 | EntropyConflict | Early backtracking |
| LDT-Full | θ_elim + conflict + α | All | Best performance |

### Metrics

1. **Solve rate** — % of puzzles solved correctly
2. **False prune rate** — % of correct tokens incorrectly eliminated
3. **Forward passes** — total model calls per puzzle
4. **Backtrack count** — number of dead-end detections

### Benchmark File

`.benchmarks/012_ldt_sudoku_goat.md`

---

## T5: Maze GOAT Proof

### Setup

Use maze infrastructure from STRATEGA (Plan 056).

| Config | K paths | α-target | Expected |
|--------|---------|----------|----------|
| Baseline | 1 | Off | Single path target |
| LDT-K4 | 4 | On | Moderate improvement |
| LDT-K16 | 16 | On | Diminishing returns |

### Metrics

1. **Path optimality** — ratio to shortest path length
2. **Convergence speed** — steps to solution
3. **Forward passes** — total computation

### Benchmark File

`.benchmarks/013_ldt_maze_goat.md`

---

## T6: MCTS Conflict Cutoff Proof

### Setup

Use Go arena from Plan 065 (AutoGo Distillation).

Run tournament: MCTS + conflict cutoff vs MCTS baseline, 20 games, 9×9.

| Config | Conflict Cutoff | Max Prune Rate |
|--------|----------------|----------------|
| Baseline | Off | — |
| LDT-c06 | On | 0.6 |
| LDT-c04 | On | 0.4 |

### Metrics

1. **Win rate** vs random baseline
2. **Average MCTS depth** before cutoff
3. **Total rollouts** per game
4. **Forward model calls** per game

### Benchmark File

`.benchmarks/014_ldt_mcts_goat.md`

---

## T7: Feature Gate Audit

### Checklist

- [ ] `lattice_deduction` feature added to `Cargo.toml`
- [ ] All new code behind `#[cfg(feature = "lattice_deduction")]`
- [ ] `cargo build` (no features) succeeds with zero warnings
- [ ] `cargo build --features lattice_deduction` succeeds
- [ ] `cargo test` (no features) passes all existing tests
- [ ] `cargo test --features lattice_deduction` passes all tests including new ones
- [ ] `cargo clippy --features lattice_deduction` passes
- [ ] No performance regression on default build (bench before/after)

---

## Architecture Diagram

```
                    ┌─────────────────────────────┐
                    │  ConstraintPruner (existing) │
                    │  is_valid() → bool           │
                    │  Sound binary pruning         │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  ScreeningPruner (existing)   │
                    │  relevance() → f32            │
                    │  Graded semantic relevance    │
                    └──────────────┬───────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 │                 │                  │
    ┌────────────▼──────┐  ┌──────▼──────────┐  ┌───▼────────────────┐
    │ T1: LDT θ_elim    │  │ T2: Conflict     │  │ T3: α-operator     │
    │ (asymmetric       │  │ Detector         │  │ (multi-solution    │
    │  threshold)        │  │ (entropy-based   │  │  intersection)     │
    │                    │  │  early cutoff)   │  │                    │
    │ θ = 1/(1+8)       │  │ max_prune=0.6    │  │ ŷ = x ⊓ α(Y|x)    │
    │ ≈ 0.111            │  │ entropy_floor    │  │                    │
    └────────────────────┘  └─────────────────┘  └────────────────────┘
                 │                 │                  │
                 └─────────────────┼─────────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │  DDTree + MCTS (existing)     │
                    │  Enhanced with LDT techniques │
                    │  Feature-gated, zero default  │
                    └──────────────────────────────┘
```

---

## File Changes Summary

| File | Change | Feature Gate |
|------|--------|-------------|
| `Cargo.toml` | Add `lattice_deduction` feature | — |
| `src/lib.rs` | Conditional `mod alpha` | `lattice_deduction` |
| `src/speculative/types.rs` | `LdtPruneConfig`, `ConflictDetector`, `EntropyConflictDetector` | `lattice_deduction` |
| `src/speculative/alpha.rs` | New file: `alpha_intersect`, `is_consistent` | `lattice_deduction` |
| `src/speculative/dd_tree.rs` | Wire θ_elim and conflict detection into expansion | `lattice_deduction` |
| `src/speculative/mod.rs` | Conditional `mod alpha` | `lattice_deduction` |
| `tests/test_ldt_deduction.rs` | New test file: threshold, conflict, α-operator proofs | `lattice_deduction` |

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | T1 + T7 (feature gate + threshold) | Config + threshold + build passes |
| 2 | T2 (ConflictDetector) | Trait + entropy impl + DDTree wiring |
| 3 | T4 (Sudoku proof) | Benchmark results |
| 4 | T3 (α-operator) | alpha_intersect + maze integration |
| 5 | T5 + T6 (Maze + MCTS proofs) | Benchmark results |
| 5 | T7 final (audit) | Clean build, all tests pass |

---

## References

- Paper: https://arxiv.org/pdf/2605.08605
- Research: `.research/50_LDT_Deduction_Transformer.md`
- Related: Plan 049 (G-Zero self-play), Plan 057 (HLA recurrent), Plan 061 (entropy anomaly), Plan 066 (D2F), Plan 067 (NFSP/MCTS)
- Benchmarks: `.benchmarks/012_ldt_sudoku_goat.md`, `013_ldt_maze_goat.md`, `014_ldt_mcts_goat.md`
