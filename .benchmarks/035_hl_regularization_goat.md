# Benchmark 035: HL Regularization GOAT Proof

> **Status:** ⏳ Pending
> **Plan:** 135 (HL Regularization Principles)
> **Requires:** `--features bandit,bomber`

## Summary

GOAT proof for patch regularization in Heuristic Learning. Validates that the
current `AbsorbCompress` configuration does not exhibit train-only degradation
— the pattern where compressed arms improve training metrics but hurt held-out
performance (the overfitting signal identified in the HL-ImageNet experiment).

## Target

Run Bomber arena **1000 rounds** with the current `AbsorbCompress` config
(`CompressConfig::default()`):

- `min_visits`: 200
- `q_threshold`: 0.05
- `promote_count`: 3
- `check_interval`: 100
- `min_benefit_ratio`: 2.0

## GOAT Gates (Proposed)

### G1: No Train-Only Degradation

| Metric | Condition |
|--------|-----------|
| Win rate (last 200 rounds) | ≥ win rate (first 200 rounds) |

Compression should not degrade late-game performance relative to early-game.
A drop would indicate that promoted hard blocks overfit to early patterns.

### G2: Support Adequacy

| Metric | Condition |
|--------|-----------|
| Arms compressed | All have `arm_visits` ≥ `min_visits` (200) |

Verify that no arm is compressed without meeting the support threshold.

### G3: Precision Gate Active

| Metric | Condition |
|--------|-----------|
| Mean Q-value of compressed arms | < `q_threshold` (0.05) |

Verify that compressed arms genuinely have low reward, not just low variance.

## How to Run

```bash
cargo test --features bandit,bomber -- bench_035_hl_regularization_goat
```

## Related

- Plan 135: `.docs/09_heuristic-learning.md` → Patch Regularization Principles
- Research 096 D1: Six regularization criteria
- `AbsorbCompressLayer`: `src/pruners/absorb_compress.rs`
- Benchmark 034: `.benchmarks/034_sr2am_configurator_goat.md`
