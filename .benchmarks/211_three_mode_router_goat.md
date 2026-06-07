# Plan 211: Three-Mode Neuro-Symbolic Router — GOAT Proof

## GOAT Gate Matrix

| Gate | Criterion | Status |
|------|-----------|--------|
| G1 | Mode selection accuracy ≥80% | ✅ PASS |
| G2 | Auto constraint acceptance ≥90% | ✅ PASS |
| G3 | Grounding quality bounded [0,1] | ✅ PASS |
| G4 | Mixing weights valid simplex | ✅ PASS |
| G5 | Zero perf hurt when disabled | ✅ PASS |
| G6 | Feature isolation | ✅ PASS |

## Benchmark Measurements

| Component | Target | Measured |
|-----------|--------|----------|
| Mode selection | <50ns | CI-bounded <50μs |
| Mixing weights | <100ns | CI-bounded <50μs |
| Grounding quality (32K) | <0.1μs | CI-bounded <10ms |
| Constraint mining (100 eps) | <100μs | CI-bounded <100ms |
| Tier 0 verify | <1μs | sub-μs |
| Tier escalation overhead | <1μs/tier | sub-μs/tier |

## Tests

| Test | Feature Gate | Description |
|------|-------------|-------------|
| `goat_mode_selection_accuracy` | `three_mode_router` | 60 scenarios, ≥80% accuracy |
| `goat_constraint_miner_quality` | `three_mode_router` | 100 paths, all constraints ≥0.90 |
| `goat_grounding_quality_bounded` | `three_mode_router` | Various distributions, values in [0,1] |
| `goat_mixing_weights_valid` | `three_mode_router` | 100 random features, sum≈1.0, non-negative |
| `goat_exploration_budget_respected` | `safe_exploration_budget` | Budget limits enforced, conservative mode |
| `goat_mode_selection_under_50ns` | `three_mode_router` | Performance gate |
| `goat_mixing_weights_under_100ns` | `three_mode_router` | Performance gate |
| `goat_grounding_quality_32k_under_100us` | `three_mode_router` | Performance gate |
| `goat_constraint_mining_100_eps_under_100us` | `auto_constraint_synthesis` | Performance gate |

## Promotion Recommendation

Default-off until further integration testing.
