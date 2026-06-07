# Benchmark Report: INSIGHT Explain Pipeline (Plan 210)

**Date**: 2026-06-07
**Branch**: develop
**Features tested**: `insight_explain` (symbolic_distill, concept_grounding, decision_explain, reward_calibrator)

## 1. F4 Calibration Overhead (per-relevance call)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Per-call overhead | <100ns | `Instant::now()` around `relevance()` with/without calibrator |
| 10K token evals | <1ms total | Batch evaluation timing |

### Methodology
- Wrap `NoPruner` in `RewardGatedCalibrator`
- Compare: calibrated vs uncalibrated `relevance()` calls
- Measure: papaya HashMap lookup overhead

**Status**: Pending benchmark run

## 2. F1 Expression Fitting Time (per-1000 traces)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Fitting 1000 traces, 8 features | <1ms | `Instant::now()` around `fitter.fit()` |
| Per-relevance evaluation | <50ns | `Instant::now()` around `expression.evaluate()` |

### Methodology
- Generate synthetic TraceDataset with 1000 records, 8 features
- Time `SymbolicExpressionFitter::fit()`
- Time `SymbolicExpression::evaluate()` on single feature vectors

**Status**: Pending benchmark run

## 3. F1 Evaluation Overhead (per-relevance call)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Per-call overhead | <50ns | `Instant::now()` around `ExpressionPruner::relevance()` |
| Baseline comparison | - | Compare with inner pruner overhead |

**Status**: Pending benchmark run

## 4. F2 Grounding Overhead (per-explanation)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Per-grounding call | <10μs | `Instant::now()` around `TemplateGrounding::ground()` |
| Chain-of-thought | <50μs | `Instant::now()` around `explain_chain()` |

### Methodology
- Create PrunerState snapshots
- Time grounding + chain-of-thought generation
- Not on hot path — post-inference only

**Status**: Pending benchmark run

## 5. F3 Sensitivity Analysis (per-100-token trace)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| 100 tokens × 4 pruners | <5ms | `Instant::now()` around `PerturbationExplainer::explain()` |
| Per-token cost | <50μs | Derived from total |

**Status**: Pending benchmark run

## 6. Memory Overhead

| Component | Additional allocations | Notes |
|-----------|----------------------|-------|
| TraceRecorder | Vec<TraceRecord> per episode | Pre-allocated with capacity 1024 |
| SymbolicExpression | Vec<Term> (4-8 terms) | Fixed after fitting |
| SensitivityCache | HashMap<[u8;32], Vec<f32>> | Grows with unique traces |
| RewardGatedCalibrator | HashMap<ParameterKey, ParameterStats> | papaya lock-free |

**Status**: Estimated (pending memory profiling)

## 7. Hot-Path Overhead (feature disabled)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency delta | <1% | Compare with/without features compiled |
| Code size delta | 0 bytes | Features not compiled → no codegen |

### Verification
- `cargo check` without features: clean ✓
- `cargo check --features insight_explain`: clean ✓
- 2474/2474 tests passing ✓

## GOAT Gate Criteria

| Gate | Criteria | Status |
|------|----------|--------|
| G1 | Expression accuracy ≥80% on known DDTree boundaries | ⏳ Pending |
| G2 | Grounding coverage ≥90% of pruner state variables | ⏳ Pending |
| G3 | Attribution correctness ≥85% vs manual analysis | ⏳ Pending |
| G4 | Calibration convergence ≤500 episodes | ⏳ Pending |
| G5 | Hot-path overhead <1% when feature disabled | ✅ Verified (zero codegen) |
| G6 | All tests pass with/without feature gates | ✅ 2474/2474 passing |

## Promotion Decision

**Verdict**: ⏳ PENDING — awaiting benchmark runs and GOAT criteria validation

If all GOAT criteria pass → add `insight_explain` to default features.
If any criteria fail → document failure, keep opt-in, create follow-up plan.
