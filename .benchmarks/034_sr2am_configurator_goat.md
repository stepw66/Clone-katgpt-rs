# GOAT Proof 034: SR²AM Configurator Bandit (Plan 112)

> **Date:** 2025-06-29
> **Feature Gate:** `sr2am_configurator` (implies `bandit`)
> **Depends on:** Plan 112 (ConfiguratorBandit, ConfiguratorContext, PlanningDecision)

## Summary

GOAT proof for SR²AM Configurator Bandit — an adaptive planning depth regulator that uses entropy-based context binning with UCB1 arm selection. Core result: **Bandit learns domain-specific policies via context isolation, achieving 33% PlanSkip savings with correct entropy-conditional arm selection (low→Skip, high→New).**

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Rounds | 1000 per proof (500 for G2/G3, 200 for G5) |
| β (token cost weight) | 0.1 |
| Entropy bins | 10 (0–9) |
| Build | Debug (unoptimized + debuginfo) |
| Platform | macOS |

## GOAT Proof Results

### G1: Arm Selection Learns Multiple Contexts

| Metric | Value |
|--------|-------|
| Contexts learned | 10 |
| Threshold | ≥ 3 |

**Result: ✅ PASS** — Bandit learned all 10 entropy bins across the spectrum.

### G2: Low Entropy Prefers PlanSkip

| Arm | Q-value (entropy_bin=0) |
|-----|-------------------------|
| PlanSkip | 0.700 |
| PlanExtend | 0.450 |
| PlanNew | 0.200 |

**Result: ✅ PASS** — PlanSkip Q=0.700 is highest at low entropy.

### G3: High Entropy Prefers PlanNew

| Arm | Q-value (entropy_bin=9) |
|-----|-------------------------|
| PlanNew | 0.700 |
| PlanExtend | 0.450 |
| PlanSkip | 0.200 |

**Result: ✅ PASS** — PlanNew Q=0.700 is highest at high entropy.

### G4: Reward Signal Tradeoff

| Quality | Token Cost | Reward | Expected |
|---------|------------|--------|----------|
| 0.8 | 0.1 | 0.790 | Positive |
| 0.01 | 1.0 | -0.090 | Negative |
| 0.5 | 0.0 | 0.500 | = quality |

**Result: ✅ PASS** — Reward = quality_gain − β × token_cost verified.

### G5: Context Isolation

| Domain | Best Arm | Q-best | Q-worst | Gap |
|--------|----------|--------|---------|-----|
| Game (domain=0) | PlanSkip | 1.000 | 0.000 | 1.000 |
| Code (domain=1) | PlanNew | 1.000 | 0.000 | 1.000 |

**Result: ✅ PASS** — Same entropy bin, different domains → different policies.

### G6: Decision Distribution & PlanSkip Savings

| Decision | Count | Percentage |
|----------|-------|------------|
| PlanSkip | 330 | 33.0% |
| PlanNew | 251 | 25.1% |
| PlanExtend | 419 | 41.9% |

**Result: ✅ PASS** — PlanSkip savings 33.0% ≥ 20% threshold.

## GOAT Gate Summary

| # | Proof | Gate | Result |
|---|-------|------|--------|
| G1 | Multi-context learning | ≥3 contexts | ✅ PASS |
| G2 | Low entropy → PlanSkip | Q_skip highest | ✅ PASS |
| G3 | High entropy → PlanNew | Q_new highest | ✅ PASS |
| G4 | Reward signal tradeoff | quality − β×cost | ✅ PASS |
| G5 | Context isolation | domain-specific policies | ✅ PASS |
| G6 | PlanSkip savings | ≥20% skip rate | ✅ PASS |

**Overall: 6/6 gates PASS**

## Key Finding

**Bandit learns domain-specific policies via context isolation.** The ConfiguratorBandit correctly:

1. **Bins entropy** into 10 levels for context-aware decisions
2. **Selects arms** via UCB1 from existing bandit infrastructure
3. **Isolates contexts** — same entropy, different domain → different best arm
4. **Saves tokens** — 33% PlanSkip means 1/3 of turns skip planning entirely
5. **Trades off correctly** — reward = quality_gain − β × token_cost penalizes waste

The three arms cover the full planning spectrum:
- **PlanNew**: Fresh tree for uncertain/novel situations (25.1%)
- **PlanExtend**: Keep tree, +1 depth for moderate situations (41.9%)
- **PlanSkip**: Early exit for confident/routine situations (33.0%)

## Files Changed

| File | Change |
|------|--------|
| `tests/bench_112_sr2am_configurator_goat.rs` | NEW: 6 GOAT proof tests |
| `.benchmarks/034_sr2am_configurator_goat.md` | NEW: This file |

## Related

- Plan 112: `.plans/112_sr2am_configurator.md`
- Bandit infrastructure: `.docs/09_heuristic-learning.md`
- δ-Mem: `.benchmarks/033_lt2_looped_goat.md`
