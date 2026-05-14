# Plan 052: GFlowNet Modelless Distillation — Flow-Balanced DDTree

**Branch:** `develop/feature/052_gflownet_modelless`
**Depends on:** Plan 021 (ScreeningPruner), Plan 030 (Bandit), Plan 049 (G-Zero Phase 1)
**Research:** `.research/23_GFlowNet_Shortest_Paths.md`
**Goal:** Distill the GFlowNet shortest-path theorem (minimize flow = shortest paths) into our existing ScreeningPruner + BanditPruner + DDTree stack without any new neural network training. Benchmark-first: measure before, implement, measure after, revert if no gain.

## Objective

The GFlowNet paper proves that minimizing expected trajectory length forces the policy to concentrate on shortest paths. Our existing stack already computes:
- **Forward marginals** (LoRA model logits) — the P_F equivalent
- **Backward relevance** (WASM `Validator::relevance()`) — the P_B equivalent  
- **Flow proxy** (BanditPruner Q-values) — the F(s) equivalent

But we use them independently. The paper's insight: **harmonizing forward and backward signals** produces shorter, higher-quality solutions with less search budget. We distill this into four additive, independently measurable, revertible changes.

## Baseline (before optimization)

Measured via existing benchmarks. To be filled during Task 1:

| Metric | Before |
|--------|--------|
| DDTree screened (tree nodes) | TBD |
| DDTree screened (time μs) | TBD |
| Tactical solve (17×16, nodes) | TBD |
| Tactical solve (17×16, time μs) | TBD |
| Bandit DDTree (1000 ep, reward) | TBD |
| ScreeningPruner relevance calls | TBD |

## Tasks

### Phase 0: Benchmark Baseline (MUST DO FIRST)

- [ ] **T1: Create benchmark test** — `tests/bench_gflownet_modelless.rs`
  - DDTree `build_screened` with `NoScreeningPruner` vs `BinaryScreeningPruner` (existing)
  - Tactical 17×16 strategic solve (nodes + time)
  - `bandit_02_ddtree` 1000-episode reward convergence
  - Record baseline numbers in this plan

### Phase 1: FlowPruner — Stop-Probability Regularization (D1)

The GFlowNet adds `λ / P_F(s_f | s, θ)` as flow regularization. In our model, `P_F(s_f | s)` is the LoRA model's probability of the EOS token at depth d. Low EOS prob = model thinks solution needs more tokens = high flow = boost exploration.

- [ ] **T2: Implement `FlowPruner<P: ScreeningPruner>`** — `src/speculative/flow_pruner.rs`
  ```rust
  pub struct FlowPruner<P: ScreeningPruner> {
      inner: P,
      lambda: f32,           // flow regularization strength
      stop_probs: Vec<f32>,  // per-depth EOS probability from marginals
  }
  
  impl<P: ScreeningPruner> ScreeningPruner for FlowPruner<P> {
      fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32 {
          let inner = self.inner.relevance(depth, token_idx, parent_tokens);
          if inner <= 0.0 { return 0.0; }
          
          // Flow bonus: boost relevance where model thinks solution continues
          // High stop_prob = model wants to stop = low flow = no bonus needed
          // Low stop_prob = model wants to continue = high flow = boost exploration
          let flow_bonus = 1.0 + self.lambda * (1.0 - self.stop_prob(depth));
          (inner * flow_bonus).clamp(0.0, 1.0)
      }
  }
  ```
  - `stop_probs` populated from marginals before DDTree build
  - Default `lambda = 0.3` (tunable, paper uses 10^-3 to 10^-2 for λ but that's loss-scale)

- [ ] **T3: Benchmark FlowPruner** — Add to `tests/bench_gflownet_modelless.rs`
  - DDTree with `FlowPruner<NoScreeningPruner>` vs `NoScreeningPruner` alone
  - Measure: tree nodes used, solution quality (path length), time
  - **Gate: FlowPruner must use ≤10% more nodes AND produce equal or shorter paths OR revert T2**

### Phase 2: Balanced DDTree — Harmonize Forward + Backward (D2)

The paper proves P_F and P_B should agree. Currently DDTree blends `ln(P_llm) + ln(R)` where R comes from ScreeningPruner. But R is typically binary (0 or 1). The GFlowNet insight: use R as a continuous signal that represents "how likely is this token on a shortest path to a valid solution?"

- [ ] **T4: Add `build_balanced` method to `TreeBuilder`** — `src/speculative/dd_tree.rs`
  - New method: `build_balanced(marginals, config, screener, stop_probs, lambda, chain_seed)`
  - Score formula changes from `ln(P_llm) + ln(R)` to `ln(P_llm) + lambda × ln(R) + flow_bonus`
  - Where `flow_bonus = lambda_flow × (1.0 - stop_probs[depth])`
  - This is a **generalization** of `build_screened` — when `lambda=1.0` and `lambda_flow=0.0`, it's identical
  - Keep `build_screened` unchanged for backward compat

- [ ] **T5: Benchmark balanced DDTree** — Add to `tests/bench_gflownet_modelless.rs`
  - `build_balanced` with various lambda values (0.5, 1.0, 1.5) vs `build_screened`
  - Measure: tree nodes, solution length, time
  - **Gate: balanced must produce ≤5% shorter paths with ≤10% more nodes OR revert T4**

### Phase 3: Flow-Weighted Bandit Reward (D3)

The paper minimizes trajectory length via flow regularization. We add a trajectory length bonus to the existing `DeltaBanditPruner` reward.

- [ ] **T6: Add `observe_delta_with_flow` to `DeltaBanditPruner`** — `src/pruners/g_zero/delta_bandit.rs`
  ```rust
  pub fn observe_delta_with_flow(&mut self, arm: usize, delta: f32, prefix_len: usize) {
      let flow_bonus = self.lambda_length / prefix_len.max(1) as f32;
      self.observe_delta(arm, delta + flow_bonus);
  }
  ```
  - New field: `lambda_length: f32` (default: 0.1)
  - Shorter solutions (small prefix_len) get higher bonus
  - This is the GFlowNet flow regularization applied to bandit rewards

- [ ] **T7: Benchmark flow-weighted bandit** — Add to `tests/bench_gflownet_modelless.rs`
  - 1000-episode bandit with flow bonus vs without
  - Measure: reward convergence speed, average solution length
  - **Gate: flow-weighted must converge in ≤ same episodes with shorter solutions OR revert T6**

### Phase 4: Goal-State Replay Sampling (D4)

The paper constructs the backward policy by reversing graph edges. We walk winning game replays backward through the WASM validator to learn which actions are on shortest paths.

- [ ] **T8: Implement `ReplayBackwardWalker`** — `src/pruners/bomber/replay_backward.rs`
  - Takes a winning replay (JSONL from `bomber_04_replay_gen`)
  - Walks backward from final tick to first tick
  - At each tick, tests alternative actions via `is_safe_action()` (WASM validator)
  - Records: (state, chosen_action, safe_alternatives) → backward policy data
  - Output: JSONL with backward policy samples

- [ ] **T9: Benchmark backward replay quality** — Add to `tests/bench_gflownet_modelless.rs`
  - Compare: forward-only replay data vs forward+backward replay data
  - Measure: how many safe alternatives found, uniqueness of backward samples
  - **Gate: backward walker must find ≥2 safe alternatives per tick on average OR revert T8**

### Phase 5: Integration & Final Benchmark

- [ ] **T10: Run full benchmark suite** — `tests/bench_gflownet_modelless.rs`
  - All phases combined
  - Compare against baseline from T1
  - Record final numbers in this plan

- [ ] **T11: Update `src/speculative/mod.rs`** — Export `FlowPruner`, `build_dd_tree_balanced`
- [ ] **T12: Update `README.md`** — Add GFlowNet distillation section
- [ ] **T13: Update `.research/23_GFlowNet_Shortest_Paths.md`** — Add actual benchmark results
- [ ] **T14: Commit** — `feat: GFlowNet modelless distillation (Plan 052)`

## Files Modified

| File | Changes |
|------|---------|
| `src/speculative/flow_pruner.rs` | **New:** FlowPruner<P> wrapper |
| `src/speculative/dd_tree.rs` | **New:** `build_balanced` method on TreeBuilder |
| `src/pruners/g_zero/delta_bandit.rs` | **New:** `observe_delta_with_flow` method + `lambda_length` field |
| `src/pruners/bomber/replay_backward.rs` | **New:** ReplayBackwardWalker |
| `src/speculative/mod.rs` | Export FlowPruner, build_dd_tree_balanced |
| `tests/bench_gflownet_modelless.rs` | **New:** Full benchmark suite |
| `.research/23_GFlowNet_Shortest_Paths.md` | Update with benchmark results |
| `.plans/052_gflownet_modelless_distillation.md` | This file |

## Feature Gate

All new code behind `#[cfg(feature = "bandit")]` (reuses existing gate — FlowPruner and build_balanced extend the bandit/HL infrastructure).

## Key Design Decisions

1. **Benchmark-first, revert-friendly** — Each phase has a quality/performance gate. If it doesn't help, we revert that phase only. Other phases are independent.
2. **Additive, not replacing** — `build_balanced` is a new method, `build_screened` stays unchanged. Callers opt-in.
3. **FlowPruner is a wrapper** — Wraps any ScreeningPruner, adds flow bonus. Zero-alloc: just a multiplication.
4. **Flow bonus from stop-probs** — Uses LoRA model's EOS token probability, which is already computed in marginals. No new forward pass needed.
5. **Backward replay is offline** — ReplayBackwardWalker processes JSONL files, doesn't run during game ticks. Zero runtime overhead.

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| FlowPruner adds noise, hurts quality | Medium | Quality gate: revert if paths aren't shorter |
| build_balanced is slower due to extra scoring | Low | Performance gate: revert if >10% slower |
| Backward replay finds no useful alternatives | Medium | Quality gate: revert if <2 alternatives/tick |
| Flow bonus clashes with existing AbsorbCompress | Low | FlowPruner and AbsorbCompress compose (wrapper chain) |

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| DDTree solution length | ≤5% shorter | T3, T5 |
| DDTree tree nodes | ≤10% more | T3, T5 |
| Bandit convergence | ≤ same episodes | T7 |
| Backward alternatives | ≥2 per tick | T9 |
| Latency impact | ≤5% increase | T10 |