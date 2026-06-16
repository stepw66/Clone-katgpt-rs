# Plan 278: FaithfulnessProbe — Causal Intervention Diagnostic for Injected Memory (Modelless)

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md](../.research/244_Self_Evolver_Faithfulness_Cognitive_Integrity.md)
**Source paper:** [arxiv 2601.22436](https://arxiv.org/pdf/2601.22436) — Zhao et al. 2026 (ICML), "Large Language Model Agents Are Not Always Faithful Self-Evolvers"
**Target:** `katgpt-rs/src/faithfulness/` (new module) + Cargo features `faithfulness_probe`, `triggered_injection`
**Status:** Active — Phase 1 (unblocking skeleton)

---

## Goal

Ship the open, generic half of the Cognitive Integrity Layer (private half: `riir-ai/.plans/308`): a **`FaithfulnessProbe`** trait + intervention suite that runs the paper's causal-intervention methodology on injected memory segments, and a **`TriggeredInjectionGate`** trait that decides whether to inject at all based on consumer uncertainty. Both are modelless (zero training, zero backprop through base weights), zero-allocation, hot-path-safe. Feature-gated; default off until GOAT gate (Research 129 G1–G9) passes.

**Unblocks:** Plan 308 (riir-ai runtime integration), Plan 054 (output-side path-hacking fusion), verification of HLA `evolve_hla` injection binding.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/src/faithfulness/mod.rs` with module doc + feature gate `#![cfg(feature = "faithfulness_probe")]`.
- [ ] **T1.2** Define `Intervention` enum (`Empty`, `Shuffle`, `Corrupt`, `Irrelevant`, `Filler`) — `#[repr(u8)]`, `#[derive(Clone, Copy, Debug, PartialEq, Eq)]`. Zero-alloc.
- [ ] **T1.3** Define `FaithfulnessProfile<D>` struct (`empty_delta`, `shuffle_or_corrupt_delta`, `irrelevant_delta`, `filler_delta`) — POD, `D: PartialOrd + Copy + Default`. Implement `is_faithfully_used(threshold)` per Research 244 §4.
- [ ] **T1.4** Define `ConsumerContext<B>` trait — minimal interface for a consumer to expose: `baseline_behavior(&self) -> B`, `behavior_with_memory(&self, memory: &M) -> B`, `behavior_delta(&self, a: &B, b: &B) -> D`.
- [ ] **T1.5** Define `FaithfulnessProbe` trait per Research 244 §4 (associated types `Memory`, `Behavior`, `Delta`; methods `probe_intervention`, `faithfulness_profile`).
- [ ] **T1.6** Implement `DefaultFaithfulnessProbe<M, B, D>` — generic over `ConsumerContext`. Runs the full intervention suite and aggregates to `FaithfulnessProfile`.
- [ ] **T1.7** Default perturbation strategies: `Empty` (zero-fill or truncate), `Shuffle` (Fisher-Yates on slice), `Corrupt` (random byte/token replacement), `Irrelevant` (caller-provided pool), `Filler` (constant placeholder). Each as a small `fn perturb_<variant>(memory: &mut M, rng: &mut impl Rng)` — no allocation where possible.
- [ ] **T1.8** Unit tests:
  - `test_faithful_consumer_detected` — synthetic consumer where memory deterministically drives behavior; probe returns `is_faithfully_used = true`. (Research 129 G1)
  - `test_unfaithful_consumer_detected` — synthetic consumer where memory is ignored (action from prior only); probe returns `is_faithfully_used = false`. (Research 129 G1b)
  - `test_intervention_enum_repr_u8` — size is 1 byte.
  - `test_profile_pod_size` — `FaithfulnessProfile<f32>` is 16 bytes.
- [ ] **T1.9** Wire Cargo feature `faithfulness_probe` in `katgpt-rs/Cargo.toml`; ensure default-off; ensure zero overhead when off (grep `cfg(feature)` coverage).

**Phase 1 exit:** tests pass; `cargo build --features faithfulness_probe` compiles; `cargo build` (no features) has no regression.

---

## Phase 2 — AttributionProbe + TriggeredInjectionGate

### Tasks

- [ ] **T2.1** Define `AttributionProbe` trait per Research 244 §4 (`attribution_norm(&self, memory: &Self::Memory, epsilon: f32) -> f32`). Finite-difference central surrogate for IG (paper App D.7).
- [ ] **T2.2** Implement `FiniteDifferenceAttributionProbe` — central differences: `(f(M + εδ) − f(M − εδ)) / (2ε)`, L2-norm the result. Zero backprop. Takes `&mut` scratch buffer.
- [ ] **T2.3** Validation: on a small reference consumer (e.g., a 2-layer linear model with known IG), verify `FiniteDifferenceAttributionProbe` ranks segments consistently with reference IG. Spearman ρ ≥ 0.8. (Research 129 G2)
- [ ] **T2.4** Define `TriggeredInjectionGate` trait: `fn should_inject(&self, uncertainty: f32) -> bool`. Sigmoid-thresholded: `should_inject := sigmoid(λ · (u − τ)) > 0.5`. **Sigmoid, not softmax** (per AGENTS.md constraint).
- [ ] **T2.5** Implement `EntropyThresholdGate { tau: f32, lambda: f32 }` — default impl. Zero-allocation.
- [ ] **T2.6** Define `UncertaintySignal` trait — unifies entropy / collapse signal / curiosity pulse into a single `f32` in `[0, 1]`. Allows Plan 212 collapse detector and Research 041 curiosity pulse to feed the same gate.
- [ ] **T2.7** Feature flag `triggered_injection` (separate from `faithfulness_probe`, also default off).
- [ ] **T2.8** Bench: `criterion` bench for `TriggeredInjectionGate::should_inject` — must be <10ns p99 (it's a sigmoid + compare). Document in `benches/triggered_injection_bench.rs`.
- [ ] **T2.9** Bench: `criterion` bench for `DefaultFaithfulnessProbe::faithfulness_profile` on a synthetic consumer — establish the audit-cadence cost. This is NOT hot-path; runs at audit cadence (e.g., every N ticks).

**Phase 2 exit:** AttributionProbe validated against reference IG; `TriggeredInjectionGate` <10ns; both feature-gated.

---

## Phase 3 — GOAT Gate (Research 129 G1, G1b, G2, G3, G8)

### Tasks

- [ ] **T3.1** **G1 + G1b** — extend Phase 1 unit tests to a property test: `proptest` over random faithful/unfaithful synthetic consumers; `is_faithfully_used` returns correct verdict ≥99% of the time.
- [ ] **T3.2** **G2** — IG surrogate validation: pick a small transformer (or a synthetic non-linear consumer with computable IG); compute reference IG; compute `FiniteDifferenceAttributionProbe` ranking; assert Spearman ρ ≥ 0.8 across ≥50 segments.
- [ ] **T3.3** **G3** — triggered-injection gain: on a saturated-regime benchmark (synthetic: consumer where prior suffices, so memory is redundant), `EntropyThresholdGate` skips ≥50% of injections with quality parity ±2% vs always-inject.
- [ ] **T3.4** **G8** — default-off zero-overhead: run existing katgpt-rs benchmark suite (HLA reconstruction bench, DDTree bench) with both features OFF; assert 0% regression.
- [ ] **T3.5** Record gate results in `katgpt-rs/.benchmarks/278_faithfulness_probe_goat.md`.
- [ ] **T3.6** GOAT gate decision:
  - If G1/G1b/G2/G3/G8 all pass → promote `triggered_injection` to default-on (saves compute + matches quality). Keep `faithfulness_probe` opt-in (diagnostic). Demote the "always-inject" loser.
  - If any fails → create `katgpt-rs/.issues/NNN_*.md`, demote, do not promote.

**Phase 3 exit:** GOAT gate recorded; promotion decision made with evidence.

---

## Phase 4 — Docs + Unblocks Plan 308

### Tasks

- [ ] **T4.1** Add `faithfulness/` module to `katgpt-rs/README.md` Feature Showcase section (between DenseMesh and KV Compression): brief description + feature flags + link to Research 244.
- [ ] **T4.2** Add `katgpt-rs/.docs/faithfulness_probe.md` — API reference + usage guide (canonical example: probing HLA `evolve_hla` injection binding).
- [ ] **T4.3** Cross-link Research 244 ↔ Plan 278 ↔ Research 129 ↔ Plan 308 in all four files' headers.
- [ ] **T4.4** Tag release per AGENTS.md commit convention: `feat(faithfulness): causal intervention probe + triggered injection gate (Plan 278, Research 244)`.

**Phase 4 exit:** docs land; Plan 308 unblocked.

---

## Architecture Decision Records

### ADR-1: Why Not Gradients Through Base Weights?

The paper uses Integrated Gradients at the attention level (requires backprop). We **cannot** — modelless-first constraint (AGENTS.md, Research skill constraint #1). The finite-difference surrogate (App D.7) is the modelless-friendly form: `ε`-ball probing, no backprop, no gradient graph. Validated by the paper's own ablation (App D.7 shows embedding-gradient L2 norm correlates strongly with attention-level IG).

### ADR-2: Why Separate `faithfulness_probe` and `triggered_injection` Features?

`faithfulness_probe` is a **diagnostic** — runs at audit cadence (every N ticks), not every tick. Expensive (full intervention suite). Stays opt-in even after GOAT gate.

`triggered_injection` is a **hot-path gate** — runs every injection event, <10ns. Cheap. Promoted to default-on if G3 passes (saves compute + matches quality).

Coupling them would either make the diagnostic too cheap (skip the full intervention suite) or the hot-path too expensive (run the full suite every tick). Separate concerns, separate features.

### ADR-3: Why Sigmoid, Not Softmax, for `TriggeredInjectionGate`?

AGENTS.md hard constraint. Softmax over a single scalar is meaningless (always 1.0). Sigmoid gives a proper inject/skip probability; threshold at 0.5 for the boolean decision. The continuous form is preserved for soft-gating (multiply memory contribution by the sigmoid value rather than hard skip) in future work.

---

## Expected Performance

| Metric | Target | Basis |
|---|---|---|
| `TriggeredInjectionGate::should_inject` latency | <10ns p99 | One sigmoid + one compare. Plasma-tier. |
| `FiniteDifferenceAttributionProbe` per segment | <100µs | 2 forward passes (M±εδ) + L2 norm. Audit cadence, not hot path. |
| `DefaultFaithfulnessProbe::faithfulness_profile` per segment | <1ms | 4 interventions × (perturb + forward + delta). Audit cadence. |
| Code size | <500 LOC | Trait defs + default impls + perturbation strategies. Well under 2048-line .rs limit. |
| Default-off overhead | 0% | Feature-gated; no codegen when off. |

---

## File Map

```
katgpt-rs/
├── Cargo.toml                          ← MODIFIED: add `faithfulness_probe`, `triggered_injection` features
├── src/
│   ├── faithfulness/
│   │   ├── mod.rs                      ← NEW: module doc, re-exports, feature gate
│   │   ├── types.rs                    ← NEW: Intervention, FaithfulnessProfile, ConsumerContext trait
│   │   ├── probe.rs                    ← NEW: FaithfulnessProbe trait, DefaultFaithfulnessProbe
│   │   ├── attribution.rs              ← NEW: AttributionProbe trait, FiniteDifferenceAttributionProbe
│   │   ├── gate.rs                     ← NEW: TriggeredInjectionGate trait, EntropyThresholdGate, UncertaintySignal
│   │   └── perturb.rs                  ← NEW: perturb_empty / _shuffle / _corrupt / _irrelevant / _filler
│   └── lib.rs                          ← MODIFIED: pub mod faithfulness (feature-gated)
├── benches/
│   ├── triggered_injection_bench.rs    ← NEW: criterion bench for gate
│   └── faithfulness_probe_bench.rs     ← NEW: criterion bench for audit-cadence probe
└── .benchmarks/
    └── 278_faithfulness_probe_goat.md  ← NEW: G1/G1b/G2/G3/G8 results
```

---

## TL;DR

Open half of the Cognitive Integrity Layer. Ships `FaithfulnessProbe` (causal intervention suite from the paper) + `AttributionProbe` (finite-difference IG surrogate) + `TriggeredInjectionGate` (entropy-thresholded inject/skip). All modelless, zero-alloc, feature-gated. GOAT gate G1/G1b/G2/G3/G8 — promote `triggered_injection` to default if it passes. Unblocks Plan 308 (riir-ai runtime integration with HLA `evolve_hla`, NeuronShard, KG Octree, dMoE).
