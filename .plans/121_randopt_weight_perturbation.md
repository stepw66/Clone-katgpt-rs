# Plan 120: RandOpt — Weight-Space Perturbation Ensembling

> **Status:** ✅ Complete (10/10 tasks done)
> **Branch:** `develop/feature/120_randopt_weight`
> **Depends on:** Plan 030 (BanditPruner ✅), Plan 049 (G-Zero ✅), Plan 079 (ELF SDE ✅)
> **Research:** `.research/080_RandOpt_Neural_Thickets_Random_Weight_Perturbation.md`
> **Source:** arXiv:2603.12228 — Neural Thickets (Gan & Isola, MIT CSAIL)
> **Feature gate:** `randopt_weight` (opt-in, depends on `bandit`)
> **Goal:** Implement RandOpt weight-space random perturbation + top-K ensembling as a `BanditPruner`-compatible protocol, plus solution-density/spectral-discordance diagnostics for existing modelless bandits.

## Summary

RandOpt proves that random Gaussian perturbations of pretrained weights, evaluated on a small training set, then ensembled via majority vote, achieve accuracy competitive with PPO/GRPO/ES — in O(1) training steps, fully parallel.

For our stack, RandOpt maps directly to `BanditPruner` in weight-space:
- N arms = N random weight perturbations θ' = θ + σ·ε(seed)
- Reward = validation score on D_train
- Top-K selection = ensemble members
- Majority vote = inference aggregation

Additionally, the paper's **solution density** and **spectral discordance** metrics provide diagnostic tools for our existing modelless bandits.

---

## Tasks

- [x] **T1: `RandOptConfig` + `RandOptWeightSampler`** — Core perturbation types
  - `RandOptConfig { population_size, ensemble_size, sigma_set, base_seed }`
  - `RandOptWeightSampler` generates θ' = θ + σ·ε(seed) for given base weights
  - Seed-based reproducibility (deterministic from `base_seed + arm_index`)
  - Multiple σ support: assign σ from `sigma_set` round-robin or random
  - File: `src/pruners/randopt.rs`

- [x] **T2: `RandOptScorer` trait** — Validation scoring interface
  - `pub trait RandOptScorer: Send + Sync { fn score(&self, weights: &[f32]) -> f32; }`
  - Implement `AccuracyScorer` for discrete-answer tasks (majority vote match)
  - Implement `WinRateScorer` for game arenas (win rate over N rounds)
  - File: `src/pruners/randopt.rs`

- [x] **T3: `RandOptEnsemble`** — Majority-vote + mean aggregation
  - `RandOptEnsemble::new(ensemble_size)`
  - `fn aggregate(&self, predictions: &[DiscreteAnswer]) -> DiscreteAnswer` (majority vote)
  - `fn aggregate_continuous(&self, predictions: &[f32]) -> f32` (mean)
  - File: `src/pruners/randopt.rs`

- [x] **T4: `RandOptSession`** — Orchestrate full RandOpt pipeline
  - Wraps `BanditSession` protocol: N perturbations → score → top-K → ensemble
  - `fn run(&mut self, base_weights: &[f32], scorer: &dyn RandOptScorer) -> RandOptResult`
  - `RandOptResult { best_seeds, best_sigmas, scores, top_k_indices }`
  - Reuses `BanditStrategy` for selection (UCB1 default)
  - File: `src/pruners/randopt.rs`

- [x] **T5: `BanditStrategy::RandOptAdaptive`** — Density-aware exploration
  - New enum variant: `RandOptAdaptive { density_threshold, decay }`
  - Measures local solution density δ = fraction of recent arms with positive reward
  - High δ (≥ threshold) → exploit (use Q-values directly)
  - Low δ (< threshold) → explore (use UCB1 or Thompson)
  - EMA tracking of density per episode
  - File: `src/pruners/bandit.rs`

- [x] **T6: `spectral_discordance()` diagnostic** — Specialist detection
  - `fn spectral_discordance(performance_matrix: &[Vec<f32>]) -> f32`
  - Input: N arms × M tasks percentile-rank matrix
  - Output: D ∈ [0, M/(M-1)], D→1 means specialists, D→0 means generalists
  - Exposed via `BanditSession` as `session.spectral_discordance()`
  - File: `src/pruners/bandit.rs`

- [x] **T7: `solution_density()` diagnostic** — Thicket regime detection
  - `fn solution_density(scores: &[f32], base_score: f32, margin: f32) -> f32`
  - Returns δ(m) = fraction of scores ≥ base_score + margin
  - Useful for both weight-space RandOpt and modelless bandit diagnostics
  - Exposed via `BanditSession` as `session.solution_density(margin)`
  - File: `src/pruners/bandit.rs`

- [x] **T8: Feature gate + module wiring**
  - Add `randopt_weight = ["bandit"]` to `Cargo.toml`
  - Add `#[cfg(feature = "randopt_weight")] pub mod randopt;` to `src/pruners/mod.rs`
  - Add to `full` feature list
  - Add example registration `[[example]] name = "randopt_01_basic"`

- [x] **T9: Example `randopt_01_basic`** — Synthetic weight perturbation demo
  - Create synthetic "model" weights (small MLP, ~1000 params)
  - Define synthetic task: predict parity of binary input
  - Run RandOpt with N=100, K=10, σ ∈ {0.01, 0.02, 0.03}
  - Show: base accuracy → top-1 → ensemble K=10 → improvement
  - Print solution density and spectral discordance
  - File: `examples/randopt_01_basic.rs`

- [x] **T10: GOAT proofs** — 21 GOAT proofs passing (config defaults, deterministic perturbation, ensemble improvement, solution density, spectral discordance, sigma round-robin, session result, etc.)
  - G1: Population scaling (sweep N, measure accuracy)
  - G2: Ensemble benefit (K=50 vs K=1)
  - G3: Sigma sensitivity (sweep σ)
  - G4: Specialist detection (D > 0.5 across 3+ tasks)
  - G5: O(1) wall-clock (compare N parallel vs sequential)
  - G6: Distillation recovery (if time permits, SFT on hard examples)
  - File: `examples/randopt_01_basic.rs` (integrated benchmarks)

---

## Architecture

```text
src/pruners/
  randopt.rs          # T1-T4: RandOptWeightSampler, RandOptScorer, RandOptEnsemble, RandOptSession
  bandit.rs           # T5-T7: RandOptAdaptive strategy, spectral_discordance(), solution_density()

examples/
  randopt_01_basic.rs # T9-T10: Demo + GOAT proofs
```

### Data Flow

```text
┌─────────────────────────────────────────────────┐
│ RandOptSession::run(base_weights, scorer)       │
│                                                 │
│  1. RandOptWeightSampler                        │
│     FOR i in 0..N:                              │
│       θ_i = θ + σ_i · ε(seed_i)                │
│                                                 │
│  2. RandOptScorer (parallel)                    │
│     v_i = scorer.score(θ_i)                     │
│                                                 │
│  3. Top-K Selection (BanditStrategy)            │
│     I_top = topk(scores, K)                     │
│                                                 │
│  4. RandOptEnsemble (inference)                 │
│     ŷ = majority_vote(predictions[I_top])       │
│                                                 │
│  5. Diagnostics                                 │
│     δ = solution_density(scores, base, margin)  │
│     D = spectral_discordance(perf_matrix)       │
└─────────────────────────────────────────────────┘
```

### Trait Integration

```text
ScreeningPruner (existing)
  └── BanditPruner<P: ScreeningPruner> (existing)
        └── RandOptSession (new, uses BanditPruner internally)
              ├── RandOptWeightSampler (generates perturbations)
              ├── RandOptScorer (scores perturbations)
              └── RandOptEnsemble (aggregates predictions)
```

---

## Key Design Decisions

1. **`RandOptSession` wraps `BanditSession`, not replaces it**: RandOpt IS a bandit protocol. We reuse `BanditStrategy` for selection, `BanditStats` for tracking, and `BanditSession` for episode management.

2. **`RandOptScorer` is a trait, not concrete**: Different domains (math reasoning, game arenas, code generation) have different scoring functions. The trait allows domain-specific implementations.

3. **Diagnostics are standalone functions, not methods**: `spectral_discordance()` and `solution_density()` are pure functions that take data arrays. They can be used with any bandit, not just RandOpt.

4. **`RandOptAdaptive` is a `BanditStrategy` variant**: This makes density-aware exploration available to ALL bandit users, not just RandOpt sessions. Any `BanditPruner` can benefit from thicket detection.

5. **Synthetic example first, LoRA integration deferred**: `randopt_01_basic` uses synthetic weights to prove the concept. LoRA weight perturbation (`randopt_02_lora`) requires `riir-ai` integration and is out of scope for this plan.

---

## GOAT Proof Targets

| # | Property | Metric | Target |
|---|----------|--------|--------|
| G1 | Population scaling | Accuracy vs N | Log-linear improvement, N=1000 > N=10 |
| G2 | Ensemble benefit | K=50 vs K=1 accuracy | K=50 ≥ K=1 + 5% |
| G3 | Sigma sensitivity | Best σ from sweep | Clear optimum in {1e-3, 2e-3, 3e-3} |
| G4 | Specialist detection | Spectral discordance D | D > 0.5 across 3 synthetic tasks |
| G5 | O(1) wall-clock | Time vs sequential | Parallel N=1000 ≈ single evaluation time |
| G6 | Distillation recovery | Distilled vs ensemble | ≥ 85% ensemble accuracy (stretch goal) |

---

## Out of Scope

- LoRA weight perturbation (requires `riir-ai` wgpu integration)
- VLM (vision-language model) support
- Distillation into single model (future plan)
- Integration with `riir-ai` training pipeline
- Large-scale LLM experiments (requires GPU cluster)

---

## Relationship to Existing Plans

| Plan | Relationship |
|------|-------------|
| 030 BanditPruner | RandOpt IS BanditPruner in weight-space |
| 049 G-Zero | RandOpt confirms why modelless Phase 1 works (solution density) |
| 053 δ-Mem | δ signal ≈ RandOpt validation score |
| 079 ELF SDE | ELF noise injection ≈ RandOpt σ·ε generation |
| 086 SimpleTES | TES loop + RandOpt = RPUCG with weight perturbation |
| 112 SR²AM | SR²AM could auto-tune RandOpt hyperparams (N, K, σ) |

---

## References

- Paper: https://arxiv.org/pdf/2603.12228
- Research: `.research/080_RandOpt_Neural_Thickets_Random_Weight_Perturbation.md`
- Upstream: `.raw/RandOpt/`
