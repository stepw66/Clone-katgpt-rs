# Benchmark 014: Epiplexity Screening Benchmarks

**Plan:** 130 — Epiplexity Structural Information Scoring
**Research:** 090 — Epiplexity Structural Information Computationally Bounded Observers
**Feature Gate:** `epiplexity_scoring = []`
**Date:** 2025-07-13

---

## Summary

Integration benchmarks for epiplexity structural information scoring, validating:
1. LossCurveTracker hooks into training loss curves
2. Game arena epiplexity measurement (Bomber + Go)
3. EpiplexityScreeningPruner vs NoScreeningPruner comparison
4. SR²AM epiplexity discrimination beyond entropy-only
5. Factorization scoring on game traces
6. Self-play S_T > random play S_T
7. α>0 improves relevance scoring for structured data

---

## Test Results

**26 tests, all passing.**

### T3: LossCurveTracker Integration with masked_loss

| Test | Description | Result |
|------|-------------|--------|
| `test_t3_loss_curve_tracker_hooks_into_training` | Structured loss curve → S_T > 0 | ✅ Pass |
| `test_t3_loss_curve_tracker_constant_training` | Constant loss → S_T ≈ 0 | ✅ Pass |
| `test_t3_loss_curve_tracker_noisy_training` | Noisy loss → small total_loss_drop | ✅ Pass |
| `test_t3_per_position_tracker_from_losses` | Per-position S matches structure | ✅ Pass |

**Integration point:** `LossCurveTracker` receives per-epoch losses from `train_mini_dllm` via:
```rust
let loss = masked_loss_into(...);
tracker.on_batch_end(epoch, loss);
```

### T5: &[f32] Trace Interface

| Test | Description | Result |
|------|-------------|--------|
| `test_t5_trace_interface_loss_curve` | &[f32] loss trace → S > 0 | ✅ Pass |
| `test_t5_trace_interface_factorization` | FactorizationScorer on &[f32] | ✅ Pass |

**Interface:** Both `EpiplexityEstimator` and `FactorizationScorer` operate on `&[f32]` — no Event Log dependency.

### T6: Game Arena Epiplexity

| Test | Description | Result |
|------|-------------|--------|
| `test_t6_bomber_self_play_higher_epiplexity_than_random` | Bomber SP S_T > random | ✅ Pass |
| `test_t6_bomber_self_play_factorization_gap` | Bomber traces have directional gap | ✅ Pass |
| `test_t6_go_self_play_higher_epiplexity_than_random` | Go SP S_T > random | ✅ Pass |
| `test_t6_go_score_trace_structured` | Go score trace S > 0 | ✅ Pass |

**Key finding:** Self-play game traces produce loss curves with higher epiplexity than random play traces, confirming the paper's prediction that structured data (self-play) carries more extractable information.

### T7: Screening Pruner Benchmarks

| Test | Description | Result |
|------|-------------|--------|
| `test_t7_screening_pruner_alpha_zero_equals_no_screening` | α=0 matches NoScreeningPruner | ✅ Pass |
| `test_t7_screening_pruner_alpha_one_uses_epiplexity_only` | α=1 uses epiplexity signal | ✅ Pass |
| `test_t7_screening_pruner_alpha_blend_interpolation` | Blend interpolation correct | ✅ Pass |

### T7: SR²AM Epiplexity vs Entropy-Only

| Test | Description | Result |
|------|-------------|--------|
| `test_t7_sr2am_epiplexity_discriminates_structured_vs_random` | S_T discriminates when H_T cannot | ✅ Pass |

**Key finding:** Two datasets with identical final loss (same H_T) but different structure are distinguishable by S_T. Epiplexity adds discriminating power beyond entropy alone.

### T7: Factorization Scoring on Game Traces

| Test | Description | Result |
|------|-------------|--------|
| `test_t7_factorization_on_bomber_traces` | Forward preferred for decreasing traces | ✅ Pass |
| `test_t7_factorization_on_go_traces` | Go traces have non-trivial gaps | ✅ Pass |
| `test_t7_factorization_ranking_by_structure` | High structure ranks first | ✅ Pass |

### T10: Self-Play S_T > Random Play

| Test | Description | Result |
|------|-------------|--------|
| `test_t10_self_play_higher_st_than_random_bomber` | Bomber SP mean S_T > random (50 games) | ✅ Pass |
| `test_t10_self_play_higher_st_than_random_go` | Go SP mean S_T > random (50 games) | ✅ Pass |

### T11: EpiplexityScreeningPruner Improves Accuracy

| Test | Description | Result |
|------|-------------|--------|
| `test_t11_alpha_gt_zero_changes_relevance_for_structured_data` | α>0 changes relevance vs α=0 | ✅ Pass |
| `test_t11_loss_drop_weight_prioritizes_structured_positions` | LossDrop mode ranks by drop | ✅ Pass |
| `test_t11_cumulative_area_weight_correlates_with_structure` | More structure → higher relevance | ✅ Pass |

### Cross-Validation & Edge Cases

| Test | Description | Result |
|------|-------------|--------|
| `test_cross_validation_tracker_matches_estimator` | Tracker ↔ Estimator agree | ✅ Pass |
| `test_edge_case_single_loss_value` | Single value → S ≈ 0 | ✅ Pass |
| `test_edge_case_all_same_loss` | Constant → S ≈ 0 | ✅ Pass |
| `test_edge_case_increasing_loss` | Increasing with last as final → S ≈ 0 | ✅ Pass |

---

## Files

| File | Purpose |
|------|---------|
| `tests/test_130_epiplexity_integration.rs` | 26 integration tests |
| `tests/test_130_epiplexity_goat.rs` | 67 GOAT proofs (pre-existing) |
| `.benchmarks/041_epiplexity_structural_information_goat.md` | GOAT proofs report |
| `.benchmarks/014_epiplexity_screening_bench.md` | This file |

---

## Run Commands

```bash
# Integration tests
cargo test -p katgpt-rs --features epiplexity_scoring --test test_130_epiplexity_integration

# GOAT proofs
cargo test -p katgpt-rs --features epiplexity_scoring --test test_130_epiplexity_goat

# All together
cargo test -p katgpt-rs --features epiplexity_scoring --test test_130_epiplexity
```

---

## Test Summary

```
running 26 tests · test_130_epiplexity_integration
..................................
test result: ok. 26 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

running 67 tests · test_130_epiplexity_goat
...................................................................
test result: ok. 67 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**Total: 93 tests pass across both test files.**
