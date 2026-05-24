# Plan 130: Epiplexity — Structural Information Scoring for Modelless Distillation

**Research**: 090_Epiplexity_Structural_Information_Computationally_Bounded_Observers.md
**Status**: 📋 Planned
**Feature Gate**: `epiplexity_scoring = []`

---

## Motivation

From epiplexity paper (arXiv:2601.03220): Structural information extractable by computationally bounded observers — measured as area under loss curve above final loss — correlates with OOD generalization, validates AlphaZero self-play, and explains why data ordering/factorization matters.

**Current gaps**:
1. `ScreeningPruner::relevance()` uses domain-specific heuristics, not structural information content
2. No loss-curve-based scoring during distillation (we compute losses but discard the shape)
3. SR²AM bandit uses entropy only — ignores structural vs random information distinction
4. No way to rank training data by "how much structure a bounded observer can extract"

**Highest-value distillation**: Prequential epiplexity estimator is nearly free (we already have loss curves), directly improves data selection for modelless distillation, and provides theoretical backing for G-Zero self-play.

## Scope

- [x] **In scope**: EpiplexityEstimator, prequential coding, loss-curve tracking, ScreeningPruner integration, SR²AM context extension, GOAT proofs on game arenas
- [ ] **Out of scope**: Requential coding (requires teacher-student KL at every step), scaling law estimation, cryptographic proofs, full MDL program search

## Tasks

### T1: EpiplexityEstimator Core
- [ ] Create `src/pruners/epiplexity/mod.rs`
- [ ] Implement `EpiplexityEstimator` struct
  - `fn new(capacity: usize) -> Self` — ring buffer for loss history
  - `fn record_step(&mut self, step_loss: f32)` — append per-step loss
  - `fn compute_epiplexity(&self, final_loss: f32) -> f32` — Σ(loss_i - final_loss) above final
  - `fn compute_per_sample(&self, final_losses: &[f32]) -> Vec<f32>` — per-position estimate
- [ ] Implement `TimeBoundedEntropy` companion
  - `fn compute_entropy(&self, final_loss: f32, n_tokens: usize) -> f32` — H_T estimate
- [ ] Unit tests: constant data → S≈0, random data → S≈0, structured data → S>0

### T2: EpiplexityScreeningPruner
- [ ] Create `src/pruners/epiplexity/screening.rs`
- [ ] Implement `EpiplexityScreeningPruner<P: ScreeningPruner>`
  - Wraps inner pruner, weights relevance by epiplexity signal
  - `fn relevance(&self, token: TokenId, context: &Context) -> f32`
  - Blend: `inner.relevance() * (1.0 - α) + epiplexity_weight * α` where α ∈ [0, 1]
- [ ] Implement `EpiplexityWeight` enum
  - `Uniform` — no weighting (baseline)
  - `LossDrop` — weight by |loss_before - loss_after| at position
  - `CumulativeArea` — weight by running epiplexity contribution
- [ ] Feature gate: `#[cfg(feature = "epiplexity_scoring")]`
- [ ] Unit tests: wrapper preserves inner pruner behavior when α=0

### T3: Loss Curve Tracker Integration
- [ ] Create `src/pruners/epiplexity/loss_curve.rs`
- [ ] Implement `LossCurveTracker` — hooks into training loop
  - `fn on_batch_end(&mut self, batch_idx: usize, avg_loss: f32)`
  - `fn on_epoch_end(&mut self, epoch: usize, val_loss: f32)`
  - `fn epiplexity_estimate(&self) -> f32` — prequential estimate
- [ ] Implement `PerPositionLossTracker` — for fine-grained scoring
  - Track loss at each token position across training
  - Compute per-position epiplexity contribution
- [ ] Integration point: hook into existing `masked_loss()` in `src/dllm.rs`
- [ ] Feature gate: `#[cfg(feature = "epiplexity_scoring")]`

### T4: SR²AM Context Extension
- [ ] Extend `ConfiguratorContext` in Plan 112 with epiplexity bin
  - Add `epiplexity_bin: u8` — discretize S_T into 10 bins (like entropy)
  - `fn from_entropy_epiplexity(domain: &str, entropy: f32, epiplexity: f32) -> Self`
- [ ] Update `ConfiguratorBandit` arm selection
  - High S_T + low H_T → `PlanExtend` (structure-rich, predictable)
  - Low S_T + high H_T → `PlanSkip` (random, unpredictable)
  - High S_T + high H_T → `PlanNew` (complex, needs fresh plan)
- [ ] Feature gate: `#[cfg(feature = "epiplexity_bandit")]` depends on `["epiplexity_scoring", "bandit"]`
- [ ] Backward compatible: existing entropy-only path preserved when feature off

### T5: Factorization Scoring for Game Traces
- [ ] Create `src/pruners/epiplexity/factorization.rs`
- [ ] Implement `FactorizationScorer`
  - `fn score_forward(&self, trace: &GameTrace) -> f32` — actions→state order
  - `fn score_reverse(&self, trace: &GameTrace) -> f32` — state→actions order
  - `fn preferred_order(&self, trace: &GameTrace) -> FactorizationOrder`
- [ ] Implement `FactorizationOrder` enum
  - `Forward` — easy to compute (moves→board)
  - `Reverse` — requires inference (board→moves, higher epiplexity per paper)
  - `Adaptive` — choose per-trace based on estimated epiplexity gap
- [ ] Integration with Event Log (Plan 124) trace format
- [ ] Feature gate: `#[cfg(feature = "epiplexity_scoring")]`

### T6: GOAT Proofs — Epiplexity on Game Arenas
- [ ] Bomber Arena: measure epiplexity of training data
  - Compare random play vs heuristic play vs self-play traces
  - Hypothesis: self-play traces have highest S_T (most structural info)
  - Run 1000 rounds, report S_T and H_T per data source
- [ ] Go Arena: measure epiplexity of game traces
  - Compare random vs MCTS vs AutoGo distilled traces
  - Validate: higher S_T → better downstream move accuracy
- [ ] Chess: reproduce paper's forward vs reverse result
  - Measure S_T for moves→board vs board→moves ordering
  - Validate: reverse order has higher S_T AND better OOD accuracy
- [ ] Report: `.benchmarks/013_epiplexity_game_arenas.md`

### T7: Benchmarks — Epiplexity vs Baseline Screening
- [ ] Benchmark: EpiplexityScreeningPruner vs NoScreeningPruner
  - Measure: downstream accuracy, data efficiency, OOD transfer
  - Domains: Go (9×9), Bomber (10×10), Chess (puzzles)
  - 1000 rounds per configuration, 3 seeds
- [ ] Benchmark: SR²AM with epiplexity context vs entropy-only
  - Measure: planning quality, token efficiency, bandit regret
  - Compare context dimensions: (domain, entropy) vs (domain, entropy, epiplexity)
- [ ] Benchmark: factorization scoring on game traces
  - Measure: training loss convergence, downstream accuracy
  - Forward vs reverse vs adaptive ordering
- [ ] Report: `.benchmarks/014_epiplexity_screening_bench.md`

### T8: Documentation & Cleanup
- [ ] Update `README.md` — add Epiplexity section under Heuristic Learning Infrastructure
- [ ] Update `.docs/` if applicable
- [ ] Update feature flags table in README
- [ ] Clippy pass: `cargo clippy --fix --allow-dirty`
- [ ] Ensure all tests pass: `cargo test --features epiplexity_scoring,epiplexity_bandit`

## Architecture

```
src/pruners/epiplexity/
├── mod.rs              # EpiplexityEstimator, feature gate re-exports
├── screening.rs        # EpiplexityScreeningPruner<P>
├── loss_curve.rs       # LossCurveTracker, PerPositionLossTracker
└── factorization.rs    # FactorizationScorer, FactorizationOrder
```

## Key Design Decisions

1. **Prequential over Requential**: Area-under-loss-curve is nearly free; requential requires teacher-student KL at every step (2-10× overhead). Prequential is sufficient for ranking data sources.

2. **Opt-in feature gate**: Epiplexity scoring adds minimal overhead but changes screening behavior. Feature gate allows A/B comparison.

3. **Composable wrapper**: `EpiplexityScreeningPruner<P>` wraps any existing `ScreeningPruner`, preserving backward compatibility. Blend factor α controls epiplexity influence.

4. **Batch-level estimation**: Per-sample epiplexity is noisy; batch/epoch-level is more reliable. Per-position used only for fine-grained analysis.

5. **Game arena validation**: Paper validates on chess; we extend to Go, Bomber, and our full game stack.

## Success Criteria

- [ ] EpiplexityEstimator correctly identifies structured vs random data (unit tests)
- [ ] Self-play game traces have measurably higher S_T than random play (T6)
- [ ] EpiplexityScreeningPruner improves downstream accuracy over baseline (T7)
- [ ] SR²AM with epiplexity context outperforms entropy-only (T7)
- [ ] All GOAT proofs pass (T6)
- [ ] Zero regressions on existing benchmarks