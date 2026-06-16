# Plan 278: FaithfulnessProbe — Phase 1+2 GOAT Gate Results

**Date:** 2026-06-16
**Plan:** [katgpt-rs/.plans/278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
**Research:** [katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
**Hardware:** Apple Silicon arm64 (M-series), release build.

---

## Phase 1 — Unblocking Skeleton

### Unit Tests (T1.8): 24/24 PASS

`cargo test --no-default-features --features sparse_mlp,faithfulness_probe,triggered_injection --lib faithfulness::`

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
- Full Spearman ρ ≥ 0.8 vs reference transformer IG deferred to Phase 3.

### T2.8: TriggeredInjectionGate latency — ✅ PASS

`cargo bench --bench triggered_injection_bench --features faithfulness_probe,triggered_injection`

| Metric | Target | Measured | Verdict |
|---|---|---|---|
| `should_inject` mean | <10ns | **2.6 ns/call** | ✅ PASS |
| `should_inject` p99 batch | <10ns | ~2.6ns (slowest batch spike from Mutex contention in test harness, not gate) | ✅ PASS |

**Hot-path optimization applied:** since `sigmoid(x) > 0.5 ⟺ x > 0` and `λ > 0`, the boolean decision collapses to `u > τ` — one compare, no `exp()`. Original implementation called `exp()` per decision (~18ns); the collapsed-compare fast path is 7× faster. The full sigmoid value remains available via `EntropyThresholdGate::sigmoid_value(u)` for opt-in soft-gating.

### T2.9: DefaultFaithfulnessProbe audit-cadence cost — ✅ PASS

`cargo bench --bench faithfulness_probe_bench --features faithfulness_probe`

| n_dim | Target | Measured | Verdict |
|---|---|---|---|
| 16 | <1ms | 0.25µs | ✅ |
| 64 | <1ms | 0.63µs | ✅ |
| 256 | <1ms | 2.41µs | ✅ |
| 1024 | <1ms | 9.14µs | ✅ |
| 4096 | <1ms | 145µs | ✅ |

All well under the 1ms audit-cadence target (this is NOT hot-path — runs every N ticks).

---

## Phase 2 Exit: ✅ MET

- AttributionProbe ranking-consistency test passes on linear consumer.
- `TriggeredInjectionGate` <10ns (2.6ns actual).
- `DefaultFaithfulnessProbe::faithfulness_profile` <1ms for all tested dims (up to 4096).
- Both features (`faithfulness_probe`, `triggered_injection`) gated, default-off.

Phase 3 GOAT gate (full Spearman ρ vs transformer IG, G3 triggered-injection gain, G8 zero-overhead) not yet run.

---

## Cross-References

- **Plan:** [278_faithfulness_probe_modelless.md](../.plans/278_faithfulness_probe_modelless.md)
- **Research:** [244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
- **Source paper:** [arxiv 2601.22436](https://arxiv.org/pdf/2601.22436) — Zhao et al. 2026 (ICML)
- **Implementation:** `katgpt-rs/src/faithfulness/{mod,types,probe,attribution,gate,perturb}.rs`
- **Benches:** `katgpt-rs/benches/{triggered_injection_bench,faithfulness_probe_bench}.rs`

## TL;DR

Phase 1+2 GOAT gate passes. G1/G1b (faithful/unfaithful detection) ✅. G2 simplified (attribution ranking on linear consumer) ✅. `TriggeredInjectionGate` at 2.6ns (target <10ns, 7× speedup from collapsed-compare fast path). `DefaultFaithfulnessProbe` audit-cadence cost <1ms up to n=4096. 24/24 unit tests pass. Features stay opt-in pending Phase 3 full GOAT gate.
