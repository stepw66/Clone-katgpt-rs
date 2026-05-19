# Plan 078: RePlaid Variance-Minimized Schedules — Modelless Path

**Branch:** `develop/feature/078_replaid_variance_schedules_modelless`
**Depends on:** Plan 066 (D2F), Plan 030 (Bandit), Plan 072 (SDAR Modelless)
**Research:** `.research/41_RePlaid_Continuous_Diffusion_Scaling.md`
**Model-Based Twin:** `riir-ai/.plans/079_replaid_elbo_self_condition_model_based.md`
**Source:** arXiv:2605.18530 — RePlaid (Prop 1, Lemma 3, Sec 5.2)
**Goal:** Port RePlaid's variance-minimized schedule optimization to our modelless stack. Three targets: D2F noise schedule, bandit exploration rate, SDAR sigmoid β. All self-supervised — no teacher, no gradients.

**Key Insight:** RePlaid Prop 1 proves there exists a unique noise schedule γ* that makes per-timestep diffusion loss **constant** — evenly distributing difficulty across steps. The schedule is found by minimizing Monte-Carlo variance of the loss. We adapt this principle to any process with sequential per-step costs: D2F denoising steps, bandit episodes, SDAR gating intensity.

**Why modelless first:** Validates the variance-minimization pattern cheaply across three independent subsystems. If flattening per-step variance improves convergence in at least one subsystem, the pattern is worth porting to the gradient-based path (Plan 079).

**Honest Scope:** We do NOT implement continuous diffusion. We port the **schedule optimization principle** (Prop 1) to three existing discrete subsystems. The theoretical guarantee (constant loss → optimal schedule) assumes Bayes-optimal conditions; our models are far from Bayes-optimal, so empirical validation is required.

---

## Tasks

### Phase 0: Core Primitive — VarianceMinimizer

- [ ] **T1: Implement `VarianceMinimizer` struct** — `src/pruners/variance_minimizer.rs`
  - Tracks running mean and variance of a signal across steps/episodes
  - Adapts a scalar schedule parameter to minimize variance (RePlaid Fig 8 simplified)
  - Exponential moving average (EMA) for online updates — no history storage
  ```rust
  //! Variance-minimized schedule optimizer (RePlaid Prop 1 adaptation).
  //!
  //! RePlaid proves that minimizing Monte-Carlo variance of per-timestep loss
  //! yields a constant-difficulty schedule (Prop 1). This struct adapts that
  //! principle to any sequential process with scalar per-step costs.
  //!
  //! Usage: Track per-step cost → adapt schedule parameter → flatten variance.
  //! No teacher, no gradients — purely online statistics.

  /// Configuration for variance minimization.
  #[derive(Debug, Clone)]
  pub struct VarianceMinimizerConfig {
      /// EMA decay for running mean (0.99 = slow adaptation).
      pub mean_decay: f32,
      /// EMA decay for running variance (0.99 = slow adaptation).
      pub var_decay: f32,
      /// Learning rate for schedule parameter update.
      pub lr: f32,
      /// Minimum schedule parameter value.
      pub min_param: f32,
      /// Maximum schedule parameter value.
      pub max_param: f32,
  }

  impl Default for VarianceMinimizerConfig {
      fn default() -> Self {
          Self {
              mean_decay: 0.99,
              var_decay: 0.99,
              lr: 0.01,
              min_param: 0.01,
              max_param: 1.0,
          }
      }
  }

  /// Online variance-minimized schedule optimizer.
  ///
  /// Tracks per-step cost and adapts a schedule parameter to minimize
  /// the variance of costs across steps. Inspired by RePlaid Prop 1:
  /// "there exists a unique noise schedule γ* such that ℓ(t) ≡ κ for all t."
  #[derive(Debug, Clone)]
  pub struct VarianceMinimizer {
      config: VarianceMinimizerConfig,
      /// Running mean of per-step costs.
      running_mean: f32,
      /// Running variance of per-step costs.
      running_var: f32,
      /// Current schedule parameter being optimized.
      param: f32,
      /// Number of observations seen.
      n_observations: u32,
  }
  ```
  - `observe(cost: f32) -> ()` — update running mean/var with new cost
  - `adapt() -> f32` — adjust `param` to minimize variance, return new param
  - `param() -> f32` — current schedule parameter
  - `variance() -> f32` — current running variance (for logging/diagnostics)
  - `mean() -> f32` — current running mean (for logging/diagnostics)
  - `reset() -> ()` — clear statistics (for domain switches)
  - [x] **T1.1:** Unit tests — `test_variance_minimizer_converges` (synthetic: costs decrease variance as param adapts), `test_variance_minimizer_clamps` (param stays in [min, max])

### Phase 1: D2F Variance-Minimized Noise Schedule

- [ ] **T2: Add `AdaptiveNoiseSchedule` to `src/dllm.rs`**
  - Wraps existing `NoiseSchedule` with per-step loss tracking
  - During training, track reconstruction loss at each denoising step
  - After each training epoch, adapt mask ratios to equalize per-step difficulty
  - Key difference from fixed `monotonic_ratios()`: ratios are **learned** to flatten loss curve
  ```rust
  /// Adaptive noise schedule that equalizes per-step denoising difficulty.
  ///
  /// RePlaid Prop 1: "there exists a unique noise schedule γ* such that
  /// ℓ_θ,γ*(t) ≡ κ for all t, and consequently Var_t[ℓ] = 0."
  ///
  /// We adapt this to discrete D2F: track per-step reconstruction accuracy,
  /// then adjust mask ratios so each step contributes equal difficulty.
  /// Steps that are too easy (high accuracy) get harder masks.
  /// Steps that are too hard (low accuracy) get easier masks.
  pub struct AdaptiveNoiseSchedule {
      /// Base schedule parameters.
      min_ratio: f32,
      max_ratio: f32,
      n_blocks: usize,
      /// Per-step loss tracker (one VarianceMinimizer per block).
      step_trackers: Vec<VarianceMinimizer>,
      /// Current adapted ratios.
      current_ratios: Vec<f32>,
      /// Number of adaptation steps performed.
      adaptations: u32,
  }
  ```
  - `new(min_ratio, max_ratio, n_blocks) -> Self`
  - `record_step_loss(block_idx: usize, loss: f32) -> ()` — called during training
  - `adapt_ratios() -> Vec<f32>` — adjust ratios to flatten per-step loss, return new ratios
  - `ratios() -> &[f32]` — current ratios (fallback to monotonic before first adapt)
  - `reset() -> ()` — clear trackers (new training run)
  - Backward-compatible: if `AdaptiveNoiseSchedule` is never `record_step_loss`'d, falls back to `NoiseSchedule::monotonic_ratios()` behavior

- [ ] **T3: Integrate into `train_mini_dllm` training loop**
  - Replace fixed `NoiseSchedule` with `AdaptiveNoiseSchedule`
  - After each epoch, call `adapt_ratios()` to update mask ratios
  - Log per-step loss and variance for diagnostics
  - [x] **T3.1:** Unit test — `test_adaptive_schedule_reduces_variance` (train on pattern dataset, verify variance of per-step losses decreases over epochs vs fixed schedule)
  - [x] **T3.2:** Unit test — `test_adaptive_schedule_preserves_accuracy` (final accuracy ≥ fixed schedule accuracy)

- [ ] **T4: Integrate into GPU D2F training (riir-ai `riir-gpu/src/dllm.rs`)**
  - Port `AdaptiveNoiseSchedule` to WGSL-side mask ratio computation
  - `GpuDllmTrainer::train()` calls `adapt_ratios()` between epochs
  - Feature-gated behind `dllm` feature (already exists)
  - [x] **T4.1:** Integration test — `test_gpu_adaptive_d2f_training` (verify GPU training completes with adaptive schedule)

### Phase 2: Bandit Variance-Minimized Exploration

- [ ] **T5: Add `VarianceEpsilon` strategy to `BanditStrategy` enum** — `src/pruners/bandit.rs`
  - New variant that adapts ε based on per-episode reward variance
  - High variance → increase exploration (haven't converged)
  - Low variance → decrease exploration (exploit learned Q-values)
  - Inspired by RePlaid's principle: minimize variance of reward signal across episodes
  ```rust
  pub enum BanditStrategy {
      // ... existing variants ...
      Ucb1,
      EpsilonGreedy { epsilon: f32, decay: f32 },
      ThompsonSampling,
      /// Variance-minimized epsilon (RePlaid-inspired).
      ///
      /// Adapts exploration rate to equalize per-episode reward variance.
      /// When reward variance is high, exploration increases.
      /// When reward variance is low, exploration decreases.
      /// Self-supervised — no hyperparameter tuning needed beyond initial ε.
      VarianceEpsilon {
          /// Initial epsilon.
          epsilon: f32,
          /// EMA decay for variance tracking (0.99 = slow).
          var_decay: f32,
          /// Learning rate for epsilon adaptation.
          lr: f32,
      },
  }
  ```
  - `prepare_episode()` for `VarianceEpsilon`: adapt ε based on accumulated variance
  - `update_arm()` for `VarianceEpsilon`: track reward variance alongside Q-values
  - [x] **T5.1:** Unit test — `test_variance_epsilon_adapts` (synthetic: rewards converge → ε decreases; rewards diverge → ε increases)

- [ ] **T6: Add `BanditStats::reward_variance()` method**
  - Track per-arm reward variance alongside Q-values
  - Returns variance for logging/diagnostics
  - Uses Welford's online algorithm for numerically stable variance
  ```rust
  impl BanditStats {
      /// Running variance of rewards per arm (Welford's algorithm).
      /// Only computed when `BanditStrategy::VarianceEpsilon` is active.
      fn reward_variance(&self, arm: usize) -> f32;

      /// Mean reward variance across all visited arms.
      fn mean_reward_variance(&self) -> f32;
  }
  ```

- [ ] **T7: Benchmark VarianceEpsilon vs EpsilonGreedy vs UCB1**
  - Run on Bomber arena (1000 episodes, seed=42)
  - Metrics: win rate, regret convergence, ε evolution over episodes
  - Compare against existing EpsilonGreedy (ε=0.3) and UCB1 baselines
  - **Gate:** If VarianceEpsilon doesn't beat EpsilonGreedy on at least one metric (win rate or regret), document why and keep feature-gated

### Phase 3: SDAR Learned β

- [ ] **T8: Add `SdarLearnedBeta` to `src/pruners/sdar_gate.rs`**
  - Replace fixed `SDAR_BETA = 5.0` with learned β that minimizes gated-signal variance
  - Track variance of `sdar_gate(gap, beta) * signal` across episodes
  - Adapt β to flatten this variance (same principle as Prop 1)
  ```rust
  /// SDAR gate with learned β (RePlaid variance-minimized).
  ///
  /// Instead of fixed β=5.0, learns β that minimizes the variance
  /// of gated reward signals across episodes. High variance means
  /// the gate is inconsistently applied → adjust β.
  pub struct SdarLearnedBeta {
      /// Current β value.
      beta: f32,
      /// Variance minimizer for gated signal.
      minimizer: VarianceMinimizer,
  }

  impl SdarLearnedBeta {
      /// Create with initial β (paper default: 5.0).
      pub fn new(initial_beta: f32) -> Self;

      /// Record a gated signal observation and adapt β.
      /// Call after each episode with the mean gated reward.
      pub fn observe_and_adapt(&mut self, gated_signal: f32) -> f32;

      /// Current β value.
      pub fn beta(&self) -> f32;
  }
  ```

- [ ] **T9: Integrate `SdarLearnedBeta` into `SdarBanditPruner`**
  - Add optional `learned_beta: Option<SdarLearnedBeta>` field
  - When present, use `learned_beta.beta()` instead of `SDAR_BETA`
  - Call `observe_and_adapt()` at end of each episode
  - Feature-gated behind existing `sdar` feature

- [ ] **T10: Benchmark Learned β vs Fixed β**
  - Run on Go 9×9 arena (1000 episodes, seed=42)
  - Compare: fixed β=5.0, fixed β=3.0, fixed β=10.0, learned β
  - Metrics: win rate, DDTree nodes, β evolution over episodes
  - **Gate:** If learned β doesn't beat fixed β=5.0, document why and keep feature-gated

### Phase 4: Unified Benchmark + Feature Gate

- [ ] **T11: Create comprehensive benchmark** — `tests/bench_replaid_variance_schedules.rs`
  - Three benchmarks in one file (D2F, Bandit, SDAR)
  - Each with before/after comparison
  - Record in `.benchmarks/012_replaid_variance_schedules.md`
  - **Decision gate:**
    - If ≥2/3 subsystems improve → ship behind `replaid_schedules` feature gate
    - If 1/3 improves → ship improving subsystem only, document others
    - If 0/3 improves → stop, document negative result, keep code feature-gated

- [ ] **T12: Feature gate `replaid_schedules`** — `Cargo.toml`
  - Default: off (experimental until benchmarks prove value)
  - Gated in: `src/pruners/variance_minimizer.rs`, `AdaptiveNoiseSchedule` in `src/dllm.rs`, `VarianceEpsilon` in `bandit.rs`, `SdarLearnedBeta` in `sdar_gate.rs`
  - Add to `full` feature set

- [ ] **T13: Update documentation**
  - `README.md` — add RePlaid Variance Schedules entry under Heuristic Learning
  - `.docs/09_heuristic-learning.md` — add VarianceMinimized section
  - `.research/41_RePlaid_Continuous_Diffusion_Scaling.md` — add Phase 4 results reference
  - Cross-reference `riir-ai/.plans/079_replaid_elbo_self_condition_model_based.md`

---

## Design Decisions

### Why EMA instead of history buffer?
RePlaid uses a monotone neural net γ̃(t) with gradient hooks (Fig 8). We can't use gradients in modelless mode. EMA provides:
- O(1) memory (no history buffer)
- Online updates (no batch processing)
- Smooth adaptation (decay controls responsiveness)
- Proven stable in our existing `BanditStats::update()` pattern

### Why three separate targets?
Each target validates the variance-minimization principle in a different regime:
- **D2F noise schedule**: Per-step loss within a single denoising pass (micro scale)
- **Bandit exploration**: Per-episode reward across many episodes (macro scale)
- **SDAR β**: Per-gate signal intensity across episodes (intermediate scale)

If the principle works across all three scales, it's robust. If it only works at one scale, we know the boundary conditions.

### Why not just add noise to the schedule?
Random perturbation (jitter) is not the same as variance-minimized adaptation. Jitter adds noise to explore; variance minimization removes noise to converge. These are complementary: jitter for exploration, variance minimization for exploitation.

### Feature Gate Strategy
Same pattern as `g_zero`, `delta_mem`, `sdar`. Experimental until benchmarks prove value. If Phase 4 benchmarks show clear gain, promote to `default` in future plan.

---

## Relationship to Model-Based Plan (riir-ai Plan 079)

| Aspect | This Plan (078, modelless) | Plan 079 (model-based) |
|--------|---------------------------|----------------------|
| Target | D2F noise, bandit ε, SDAR β | wgpu LoRA loss, self-conditioning |
| Signal source | Per-step loss / reward (observed) | ELBO variance (computed) |
| Training | None (online statistics only) | wgpu kernel backward passes |
| Shared types | `VarianceMinimizer`, `VarianceMinimizerConfig` | Same types, plus ELBO loss |
| Shared insight | Prop 1 (constant per-step loss) | Prop 1 + self-conditioning + ELBO |

**Shared types** (`VarianceMinimizer`, `VarianceMinimizerConfig`) are defined here (modelless) and re-exported by riir-ai's training infrastructure. The model-based plan adds ELBO loss computation and self-conditioning loops on top of the same variance-minimization core.

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Variance minimization doesn't help D2F (T3) | Minor — keep fixed schedule | Document negative result, keep `AdaptiveNoiseSchedule` as optional |
| VarianceEpsilon performs worse than UCB1 (T7) | Minor — keep existing strategies | UCB1 is already default, VarianceEpsilon is additive option |
| Learned β diverges (T10) | Medium — gate signals collapse | Clamp β to [SDAR_BETA_MIN, SDAR_BETA_MAX] = [0.1, 50.0] |
| EMA too slow to adapt (all) | Medium — no benefit within episode budget | Tune `var_decay` per target (faster for D2F, slower for bandit) |
| All three targets fail (T11) | Low — plan stops | Document why Prop 1 doesn't transfer to discrete settings |

---

## References

- Research 41: `.research/41_RePlaid_Continuous_Diffusion_Scaling.md` (this paper)
- Research 10: `.research/10_ColaDLM_Continuous_Latent_Diffusion.md` (previous continuous diffusion, rejected)
- Research 34: `.research/34_D2F_Discrete_Diffusion_Forcing.md` (our discrete diffusion)
- Research 38: `.research/38_SDAR_Self_Distilled_Agentic_RL.md` (sigmoid gating)
- Plan 066: D2F Discrete Diffusion Forcing (existing D2F infrastructure)
- Plan 030: Multi-Armed Bandit (existing bandit infrastructure)
- Plan 072: SDAR Gated Distillation Modelless (existing SDAR infrastructure)
- `riir-ai/.plans/079_replaid_elbo_self_condition_model_based.md` — model-based twin plan