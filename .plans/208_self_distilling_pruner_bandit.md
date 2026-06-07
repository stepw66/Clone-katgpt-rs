# Plan 208: Self-Distilling Pruner Bandit — Episode-Guided Arm Selection

**Date:** 2026-06-07
**Status:** ✅ Done
**Research:** `.research/182_STV_Self_Trained_Verification.md` (F3: Self-Distilling Pruner Bandit)
**Depends On:** Plan 206 (EpisodePruner, EpisodeLookup, MemoryEpisodeLookup), existing `BanditPruner`
**Feature Gate:** `self_distilling_bandit` (default-OFF until GOAT proof)
**GOAT Criteria:** ≥10% pruner accuracy improvement with episodes vs without; zero regression without episodes

---

## Problem

`BanditPruner` uses reward from DDTree acceptance (binary: accepted/rejected). When an episode DB has reference solutions, we can provide a richer reward signal: "did the chosen pruner configuration produce output that matches the known-good reference?" This is the modelless analog of STV's On-Policy Distillation (OPD).

**Current:** BanditPruner learns from acceptance rate alone.
**Proposed:** BanditPruner learns from episode-guided reward: match quality against reference solutions.
**Result:** 10-20% improvement in pruner accuracy over time, compounding as episode DB grows.

---

## Architecture

```
DDTree Generation
    │
    ├── BanditPruner (existing) — arm selection via UCB1/Thompson
    │
    ├── NEW: SelfDistillingBandit<P, L>
    │   ├── inner: BanditPruner<P> — delegates arm selection
    │   ├── lookup: L — EpisodeLookup for reference solutions
    │   ├── reward_signal: EpisodeRewardComputer — computes match reward
    │   └── config: SelfDistillingConfig — thresholds, decay
    │
    └── Reward Loop:
        1. SelfDistillingBandit delegates to inner BanditPruner for arm selection
        2. After generation, lookup episode by prompt_hash
        3. Compare generated output to reference
        4. Compute reward = episode_match_reward + alpha * acceptance_reward
        5. Update inner bandit with combined reward
```

### Reward Computation

```text
episode_reward = sigmoid(k * (match_ratio - 0.5))  where match_ratio = matching_tokens / total_tokens
combined_reward = (1 - alpha) * episode_reward + alpha * acceptance_reward
```

- `alpha` controls blend between episode signal and acceptance signal (default: 0.3)
- When no episode exists → pure acceptance reward (alpha = 1.0, zero regression)
- `k` controls reward steepness (default: 4.0)

---

## Tasks

### Phase 1: Core Implementation

- [x] **T1: Create `src/pruners/self_distilling_bandit.rs`**
  - `SelfDistillingConfig` struct with configurable alpha, k, match_threshold
  - `EpisodeRewardComputer` struct with `compute_reward(generated, reference, acceptance) -> f32`
  - `SelfDistillingBandit<P, L>` generic over inner `ScreeningPruner` P and `EpisodeLookup` L
  - Feature-gated behind `#[cfg(feature = "self_distilling_bandit")]`

- [x] **T2: Implement `ScreeningPruner` for `SelfDistillingBandit`**
  - `relevance()` delegates to inner `BanditPruner`
  - Zero cost on miss path (no episode → delegate directly to inner)

- [x] **T3: Implement episode-guided reward update**
  - `episode_update(&mut self, prompt_hash, arm, generated, acceptance_reward, domain_hash)`
  - Looks up episode, computes match reward, blends with acceptance signal
  - Updates inner bandit with combined reward via `update()`
  - Records reward history for convergence tracking

### Phase 2: Domain-Keyed Bandit

- [x] **T4: Add domain-keyed arm selection**
  - Per-domain Q-value tracking: `DomainQTable` with per-bucket `BanditStats`
  - `domain_hash` parameter routes to per-domain Q-table
  - Falls back to global Q-values when domain has < `min_domain_samples` (default: 10)

- [x] **T5: Tests for domain-keyed selection**
  - Test: different domains converge to different best arms
  - Test: cold domain falls back to global Q-values

### Phase 3: Convergence Tracking

- [x] **T6: Add convergence metrics**
  - `ConvergenceMetrics` struct: avg_reward, episode_hit_rate, arm_entropy, total_updates, warm_domains
  - `convergence_metrics(&self) -> ConvergenceMetrics` method

- [x] **T7: Tests for convergence**
  - Test: convergence metrics improve over time with episodes
  - Test: cold start metrics show no artificial inflation

### Phase 4: Module Wiring + Feature Gate

- [x] **T8: Wire module into `src/pruners/mod.rs`**
  - Add `pub mod self_distilling_bandit;` behind `#[cfg(feature = "self_distilling_bandit")]`
  - Re-export key types

- [x] **T9: Add feature gate to `Cargo.toml`**
  - `self_distilling_bandit = ["egcs"]` — depends on episode_pruner infrastructure

### Phase 5: Example + GOAT Proof

- [x] **T10: Create `examples/self_distilling_demo.rs`**
  - 3-section demo: bandit with vs without episode reward, convergence plot, domain routing

- [x] **T11: Create GOAT proof `.benchmarks/208_self_distilling_goat.md`**
  - G1: Accuracy ≥ 10% better with episodes vs without
  - G2: Zero regression on problems without episodes
  - G3: Latency overhead ≤ 2% on miss path
  - G4: All tests pass with/without feature

### Phase 6: Plan Update

- [x] **T12: Update plan status to Done**

---

## Feature Gate Configuration

```toml
[features]
self_distilling_bandit = ["egcs"]  # Depends on EpisodeLookup + EpisodePruner infra
# Default-OFF until GOAT proof
```

## Files to Create/Modify

| File | Action | Phase |
|------|--------|-------|
| `src/pruners/self_distilling_bandit.rs` | NEW | 1-3 |
| `src/pruners/mod.rs` | EXTEND | 4 |
| `Cargo.toml` | EXTEND | 4 |
| `examples/self_distilling_demo.rs` | NEW | 5 |
| `.benchmarks/208_self_distilling_goat.md` | NEW | 5 |
| `.plans/208_self_distilling_pruner_bandit.md` | THIS | 6 |

## SOLID Compliance

- **S:** `SelfDistillingBandit` only adds episode reward signal. Arm selection stays in `BanditPruner`.
- **O:** New reward signal extends existing bandit without modifying arm selection logic.
- **L:** Implements `ScreeningPruner`, can replace `BanditPruner` anywhere it's used.
- **I:** Thin public API: `episode_update()`, `convergence_metrics()`.
- **D:** Depends on `EpisodeLookup` trait, not concrete storage.

## Expected Performance

| Metric | Without SD-Bandit | With SD-Bandit | Delta |
|--------|-------------------|----------------|-------|
| Pruner accuracy (with episodes) | Baseline | +10-20% | Episode-guided reward |
| Pruner accuracy (no episodes) | Baseline | Same | Zero regression |
| Per-query overhead | 0 | <1% | Episode lookup + reward compute |
| Memory per domain | 0 | ~num_arms × 12 bytes | Q-values + visit counts |

---

## TL;DR

Plan 208 = **Self-Distilling Pruner Bandit** — wraps `BanditPruner` with episode-guided reward signal from Research 182 F3. After generation, compares output to episode reference, computes match reward, blends with acceptance reward, updates bandit. Domain-keyed Q-values for per-problem-type arm selection. Feature-gated behind `self_distilling_bandit` (depends on `egcs`), default-OFF until GOAT proof. ~400 lines new code.
