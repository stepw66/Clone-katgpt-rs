# Plan 278: FaithfulnessProbe — GOAT Gate Results (Phase 1 + 2 + 3)

**Date:** 2026-06-16
**Plan:** [katgpt-rs/.plans/278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
**Research:** [katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
**Private guide (riir-ai):** [129_Cognitive_Integrity_Layer_Guide.md](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md)
**Hardware:** Apple Silicon arm64 (M-series), release build.

---

## Phase 1 — Unblocking Skeleton

### Unit Tests (T1.8): 24/24 PASS

`cargo test --features faithfulness_probe,triggered_injection --lib faithfulness::`

| Module | Tests | Result |
|---|---|---|
| `types.rs` | `test_intervention_enum_repr_u8` (size=1), `test_profile_pod_size` (16 bytes), `test_is_faithfully_used_strict_all_conditions`, `test_vec_implements_memory_slice` | ✅ 4/4 |
| `perturb.rs` | empty/shuffle/corrupt/irrelevant/filler + edge cases | ✅ 7/7 |
| `probe.rs` | **`test_faithful_consumer_detected` (G1)**, **`test_unfaithful_consumer_detected` (G1b)** | ✅ 2/2 |
| `attribution.rs` | linear-consumer gradient match, empty/zero-ε, **ranking consistency (G2 simplified)** | ✅ 4/4 |
| `gate.rs` | inject/skip/boundary/custom/extreme/copy/sigmoid | ✅ 7/7 |

### G1 (faithful consumer detected): ✅ PASS
### G1b (unfaithful consumer detected): ✅ PASS

---

## Phase 2 — AttributionProbe + TriggeredInjectionGate

### G2 (attribution ranking, simplified): ✅ PASS (linear consumer)
- Full Spearman ρ ≥ 0.8 vs reference IG deferred to Phase 3.

### T2.8: TriggeredInjectionGate latency — ✅ PASS

`cargo bench --bench triggered_injection_bench --features faithfulness_probe,triggered_injection`

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| `should_inject` mean | <10ns | **0.132 ns/call** | ✅ PASS |
| `should_inject` p99 batch | <10ns | **0.177 ns/call** | ✅ PASS |

**Hot-path optimization applied:** since `sigmoid(x) > 0.5 ⟺ x > 0` and `λ > 0`, the boolean decision collapses to `u > τ` — one compare, no `exp()`. The full sigmoid value remains available via `EntropyThresholdGate::sigmoid_value(u)` for opt-in soft-gating.

### T2.9: DefaultFaithfulnessProbe audit-cadence cost — ✅ PASS

`cargo bench --bench faithfulness_probe_bench --features faithfulness_probe`

| n_dim | Target | Measured | Verdict |
|---|---|---|---|
| 16 | <1ms | 0.26µs | ✅ |
| 64 | <1ms | 0.67µs | ✅ |
| 256 | <1ms | 2.38µs | ✅ |
| 1024 | <1ms | 9.18µs | ✅ |
| 4096 | <1ms | 36.83µs | ✅ |

All well under the 1ms audit-cadence target (this is NOT hot-path — runs every N ticks).

---

## Phase 3 — Full GOAT Gate (G1/G1b/G2/G3/G8)

`cargo test --features faithfulness_probe,triggered_injection --lib faithfulness::goat_gate -- --nocapture`

### G1 + G1b (extended) — randomized detection rate ✅ PASS

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| **G1** faithful detection | ≥99% | **100.0%** (200/200) | ✅ PASS |
| **G1b** unfaithful detection | ≥99% | **100.0%** (200/200) | ✅ PASS |
| Combined overall | ≥99% | **100.0%** (400/400) | ✅ PASS |

Property test (hand-rolled with `fastrand` — `proptest`/`quickcheck` are not katgpt-rs dev-deps per repo convention; see `crates/katgpt-core/src/micro_belief/tests.rs:137`). 400 randomized trials: 200 faithful consumers (positive weights in [0.3, 2.0], distinct memory values) + 200 unfaithful consumers (constant output, ignores memory). All correctly classified.

### G2 — IG surrogate Spearman ρ ✅ PASS

| Sub-test | Threshold | Measured | Verdict |
|---|---|---|---|
| **G2** (64 segments, non-linear consumer, ρ ≥ 0.8) | ≥0.8 | **ρ = 1.0000** | ✅ PASS |
| G2 monotonic sanity (50 segments, ρ ≥ 0.95) | ≥0.95 | **ρ = 1.0000** | ✅ PASS |

Non-linear consumer: `behavior = Σ w_i·m_i + ½·Σ m_i²`. Exact gradient norm = `√(Σ (w_i + m_i)²)` — computable analytically. `FiniteDifferenceAttributionProbe` with ε=1e-3 ranks segments identically to the exact gradient norm.

### G3 — Triggered-injection gain ✅ PASS

| Sub-test | Threshold | Measured | Verdict |
|---|---|---|---|
| **G3a** skip rate (saturated regime) | ≥50% | **50.0%** (1000/2000) | ✅ PASS |
| **G3b** quality parity (cosine delta) | ≤2% | **0.63%** | ✅ PASS |
| G3 quality floor (min cosine) | ≥0.98 | **0.9963** | ✅ PASS |

Saturated-regime simulation: consumer behavior = `prior + α·memory` with α=0.05 (5% memory contribution). Bimodal uncertainty distribution (half low/saturated, half high/needs-memory). `EntropyThresholdGate` (tau=0.5, lambda=8.0) correctly skips the saturated half with <1% quality loss.

### G8 — Zero-overhead when off ✅ PASS

| Check | Threshold | Measured | Verdict |
|---|---|---|---|
| `cargo build --no-default-features --features sparse_mlp` | clean compile | ✅ clean | ✅ PASS |
| `faithfulness`/`triggered_injection` symbols in default-off build | 0 | **0 matches** (`nm` on `libkatgpt_rs.rlib`) | ✅ PASS |
| Default test suite regression | 0% | **0 failures** (3628 tests pass) | ✅ PASS |
| `lib.rs` gate coverage | `#[cfg(feature)]` on module | ✅ `#[cfg(any(feature="faithfulness_probe", feature="triggered_injection"))]` | ✅ PASS |

---

## Phase 3 Exit: ✅ ALL GATES PASS

### GOAT Gate Decision (T3.6)

| Gate | Result | Action |
|---|---|---|
| G1/G1b | ✅ 100% detection | — |
| G2 | ✅ ρ=1.0000 | — |
| G3 | ✅ 50% skips, 0.63% quality delta | **Promote `triggered_injection` to default-ON** |
| G8 | ✅ 0% regression | — |

**Decision:**
- **`triggered_injection` → DEFAULT-ON.** G3 proved the gate saves compute (50% injection skips) with negligible quality loss (0.63% << 2% threshold). Promoted in `Cargo.toml` default features. The "always-inject" baseline is demoted.
- **`faithfulness_probe` → OPT-IN (unchanged).** It's a diagnostic running at audit cadence (every N ticks), not a hot-path component. Stays opt-in per ADR-2.

### Feature Structure (post-promotion)

- `triggered_injection` (default-ON): gates `src/faithfulness/{gate,types}.rs` — the hot-path gate + core types.
- `faithfulness_probe` (opt-in): additionally gates `src/faithfulness/{probe,attribution,perturb,goat_gate}.rs` — the full diagnostic suite.

Module compiled when EITHER feature is on; submodules individually gated in `mod.rs`.

---

## Cross-References

- **Plan:** [278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
- **Research:** [244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
- **Private guide (riir-ai):** [129_Cognitive_Integrity_Layer_Guide.md](../../riir-ai/.research/129_Cognitive_Integrity_Layer_Guide.md)
- **Source paper:** [arxiv 2601.22436](https://arxiv.org/pdf/2601.22436) — Zhao et al. 2026 (ICML)
- **Implementation:** `crates/katgpt-core/src/faithfulness/{mod,types,probe,attribution,gate,perturb,goat_gate}.rs` (moved from `katgpt-rs/src/faithfulness/` so riir-engine Plan 308 can consume via katgpt-core)
- **Benches:** `katgpt-rs/benches/{triggered_injection_bench,faithfulness_probe_bench}.rs` (import via `katgpt_core::faithfulness::*`)
- **API docs:** [`.docs/04_calibration/faithfulness_probe.md`](../.docs/04_calibration/faithfulness_probe.md)

## TL;DR

**All GOAT gates pass.** G1/G1b (100% faithful/unfaithful detection over 400 trials) ✅. G2 (Spearman ρ=1.0000 on non-linear consumer, 64 segments) ✅. G3 (50% injection skips with 0.63% quality delta in saturated regime) ✅. G8 (0 symbols in default-off build, 0% test regression) ✅. **`triggered_injection` promoted to default-ON; `faithfulness_probe` kept opt-in (diagnostic).** riir-ai Plan 308 unblocked.
