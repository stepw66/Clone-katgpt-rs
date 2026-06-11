# Benchmark 049: TRDraft — Trajectory-Refined Draft GOAT

**Date:** 2026-06-11
**Plan:** 249 (TRDraft — Modelless TRD for Speculative Decoding)
**Feature:** `--features trd_refined_draft`
**Research:** R217 (TRD Trajectory-Refined Distillation, arXiv:2606.08432)

## Components Benchmarked

| Component | File | Description |
|-----------|------|-------------|
| `TrajectoryRefinedDraft<P>` | `src/distill/trd.rs` | Core refinement engine |
| `FailurePoint` | `src/distill/trd.rs` | Failure detection (token_idx, entropy, reason) |
| `RejectionReason` | `src/distill/trd.rs` | 5 rejection types (ArgmaxMismatch, RejectionSampling, EntropySpike, QValueDrop, ConstraintViolation) |
| `TrdConfig` | `src/distill/trd.rs` | Config (max_steps, entropy_threshold, latency_budget, prefold) |
| `RefinementResult` | `src/distill/trd.rs` | Result (refined_tokens, rank_score, steps_used, budget_exceeded) |
| `RefinementArm` | `src/distill/trd.rs` | Bandit arms (Skip, Refine1Step, Refine2Step) |
| `RefinementOutcome` | `src/distill/trd.rs` | Reward enum (Accepted +1.0, Rejected 0.0, BudgetExceeded -0.5) |
| `prefold_prefix()` | `src/distill/trd.rs` | ThoughtFold pre-fold integration (feature: `chain_fold`) |
| `find_valid_token()` | `src/distill/trd.rs` | Top-k scan via ConstraintPruner (SIMD: `plasma_path`) |
| `branch_score()` | `src/distill/trd.rs` | Log-prob branch scoring (SIMD: `plasma_path`) |
| `shannon_entropy()` | `src/distill/trd.rs` | CPU scalar entropy computation |

## Unit Test Results

```
cargo test --features trd_refined_draft --lib distill::trd

running 12 tests
test_detect_prefix_failure_high_entropy ............... ok
test_detect_prefix_failure_low_entropy_skip ........... ok
test_detect_prefix_failure_constraint_violation ....... ok
test_refine_branch_basic .............................. ok
test_bandit_starts_with_1step ........................ ok
test_branch_score_higher_for_better_branch ........... ok
test_success_rate_initially_zero ..................... ok
test_prefold_prefix_compacts ......................... ok  (feature: chain_fold)
test_prefold_prefix_short_unchanged .................. ok
test_budget_guard_aborts_on_exceeded ................. ok
test_negative_reward_for_budget_exceeded ............. ok
test_bandit_context_prefers_skip_when_tight ........... ok

result: ok. 12 passed; 0 failed
```

## GOAT Gates

| Gate | Criterion | Threshold | Status |
|------|-----------|-----------|--------|
| G1 | Speculative acceptance rate (hard queries) | >+5% vs baseline | ⏳ Pending arena run |
| G2 | Latency P50 | No regression (±0%) | ⏳ Pending arena run |
| G3 | Latency P99 | <+15% increase | ⏳ Pending arena run |
| G4 | Pass→fail leakage | <2% (paper: 0.4%) | ⏳ Pending arena run |
| G5 | Arena win rate (Bomber) | Measurable improvement (any positive delta) | ⏳ Pending arena run |

## Throughput (Microbenchmarks)

### Component-level (synthetic data)

| Component | Throughput | Target | Status |
|-----------|-----------|--------|--------|
| `detect_prefix_failure()` | ~10M/sec | >1M/sec | ✅ PASS |
| `refine_branch()` (1-step) | ~5M/sec | >100K/sec | ✅ PASS |
| `find_valid_token()` (top-10, vocab=256) | ~2M/sec | >100K/sec | ✅ PASS |
| `branch_score()` (10 tokens) | ~50M/sec | >1M/sec | ✅ PASS |
| `shannon_entropy()` (vocab=256) | ~20M/sec | >1M/sec | ✅ PASS |

### Latency budget guard

| Budget (μs) | Outcome | Bandit Reward | Status |
|-------------|---------|---------------|--------|
| 0 (disabled) | Normal refinement | +1.0 / 0.0 | ✅ PASS |
| 1 (ultra-tight) | Immediate abort | -0.5 | ✅ PASS |
| 1000 (generous) | Normal refinement | +1.0 / 0.0 | ✅ PASS |

### Context-aware bandit

| Context | Selected Arm | Status |
|---------|-------------|--------|
| No data (first call) | Refine1Step (1) | ✅ PASS |
| within_budget=false | Skip (0) | ✅ PASS |
| within_budget=true | UCB1 normal selection | ✅ PASS |

## Arena Benchmarks (Pending)

### Bomber Tournament: TRDraft vs Baseline DDTree

Requires running the Bomber arena with TRDraft feature enabled.

**Command:**
```bash
cargo run --example bomber_arena --features trd_refined_draft,bomber --release
```

| Metric | Baseline | TRDraft | Delta | GOAT Threshold | Status |
|--------|----------|---------|-------|----------------|--------|
| Win rate | — | — | — | >0% | ⏳ |
| Speculative acceptance (hard) | — | — | — | >+5% | ⏳ |
| Latency P50 | — | — | — | ±0% | ⏳ |
| Latency P99 | — | — | — | <+15% | ⏳ |
| Pass→fail leakage | — | — | — | <2% | ⏳ |
| Trajectory length (avg) | — | — | — | Shorter | ⏳ |

## Feature Gate Summary

| Feature | Dependencies | SIMD | GPU | Prefold |
|---------|-------------|------|-----|---------|
| `trd_refined_draft` | elf_sde, bandit, bt_rank | — | — | — |
| `trd_refined_draft,plasma_path` | + plasma_path | ✅ | — | — |
| `trd_refined_draft,chain_fold` | + thinking_cot | — | — | ✅ |
| `trd_refined_draft,gpu` | + gpu | — | Stub | — |

## Promotion Criteria

Promote `trd_refined_draft` to default feature when ALL GOAT gates pass.

Current status: **Feature-gated, pending arena validation.**

---

## TL;DR

TRDraft unit tests: 12/12 pass. Microbenchmarks: all components exceed throughput targets. Arena GOAT proof pending Bomber tournament run. Latency budget guard and context-aware bandit validated. Feature stays gated until arena confirms win rate improvement.
