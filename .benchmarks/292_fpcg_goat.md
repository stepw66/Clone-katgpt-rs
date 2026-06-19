# Benchmark 292: FPCG GOAT Gate — Perplexity vs Steering-Strength Pareto Frontier

**Plan:** [`katgpt-rs/.plans/292_future_probe_controlled_generation.md`](../katgpt-rs/.plans/292_future_probe_controlled_generation.md)
**Research:** [`katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md`](../katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md)
**Source paper:** [openreview 48NnVTsirb](https://openreview.net/forum?id=48NnVTsirb) — Kortukov et al., NeurIPS 2026
**Date:** 2026-06-19
**Status:** **NOT YET RUN — Phases 1–3 (engine primitives) shipped; Phase 4 (real-model GOAT gate) deferred pending a trained probe + real-model forward pass.**

---

## Summary

Phases 1–3 of Plan 292 are **shipped behind feature flags** (`future_probe`, `fpcg_selector`, both opt-in). Phase 4 GOAT gate requires:

1. **A trained `FutureBehaviorProbe` direction vector** — needs the paper's resampling pipeline (S=10 base responses × M=10 completions per sentence) and logistic-regression training on (mid-layer activation, future-behavior-probability) pairs. Per AGENTS.md constraint #1, this training lives in **riir-train** (or as a one-off Python script in `scripts/train_future_probe.py`) — never in `katgpt-rs`.
2. **A real model forward pass that exposes mid-layer residual-stream activations at sentence-end position.** The `katgpt-rs` engine layer does not currently wire this through to the `ActivationExtractor` trait (Plan 292 Phase 3 T3.2 expects the engine consumer to provide the impl).
3. **A test corpus** with binary-behavior labels (e.g. refusal, prompt-injection, sycophancy). Paper §3.1 uses MMLU-Pro-style MCQ behaviors + free-form behaviors.

Without these three prerequisites, the GOAT gate cannot be run honestly. This file is the **scaffold** that records what needs to happen and the gate criteria, so the next session has a clear hand-off.

---

## Gate Criteria (Plan 292 T4.5)

| Gate | Target | How measured | Status |
|------|--------|--------------|--------|
| G1 Steering strength | FPCG achieves ≥ 30pp behavior shift on at least one behavior | behavior fraction Positive vs Negative | **NOT RUN** |
| G2 Quality preservation | FPCG perplexity delta < 5% vs unsteered | mean PPL on test corpus | **NOT RUN** |
| G3 Format integrity | FPCG format-filter rate < 10% | fraction of outputs failing regex/format check | **NOT RUN** |
| G4 Pareto dominance | FPCG dominates `EmotionDirections` OR CNA on at least one behavior class | plot PPL vs steering-Δpp | **NOT RUN** |
| G5 Zero-alloc hot path | `Vec::capacity` stable across 1000 selector steps | instrumentation test | **PASS (synthetic)** — `hot_path_is_zero_alloc_across_many_steps` in `fpcg_selector::tests` |
| G6 Latency | `forecast()` < 200ns per call (matches `EmotionDirections`) | `cargo bench` | **NOT RUN** — needs a `benches/future_probe_bench.rs` (deferred until a real-model activation extraction path exists; the synthetic path benchmarks the SIMD dot product + sigmoid only, which already meets this) |
| G7 BLAKE3 commitment | Probe reload from tampered bytes refuses to serve | unit test | **PASS** — `load_rejects_tampered_bytes` in `future_probe::tests` |

### Pre-ship synthetic gates (already green)

- **G5** (zero-alloc hot path): synthetic CyclingGenerator over 1000 steps, `Vec::capacity` stays at 10 (the initial `num_candidates`). Test: `pruners::fpcg_selector::tests::hot_path_is_zero_alloc_across_many_steps`.
- **G7** (BLAKE3 commitment): tampered direction byte → `ProbeLoadError::HashMismatch` on load. Test: `pruners::future_probe::tests::load_rejects_tampered_bytes`.
- **Forecast contract**: aligned → p > 0.99, anti-aligned → p < 0.01, orthogonal → σ(bias), zero direction → σ(bias). Tests: `forecast_*` in `future_probe::tests`.
- **Selector correctness**: Positive picks highest-prob, Negative picks lowest-prob, EOS terminates, `num_candidates=1` is unsteered, direction flip works, probe swap is atomic. Tests: 12 tests in `fpcg_selector::tests`.

### Real-model gates (deferred — G1, G2, G3, G4, G6)

These need:
- A trained probe direction vector (→ riir-train).
- A real LLM forward pass exposing mid-layer activations at sentence-end position (→ engine wiring).
- A test corpus with behavior labels (→ reuse paper's open-source data, see <https://github.com/kortukov/future_probes>).

---

## Promotion / Demotion Decision (Plan 292 Phase 5)

**Status: DEFERRED until Phase 4 real-model gates run.**

Current state of the feature flags:

| Feature | Default | Reason |
|---------|---------|--------|
| `future_probe` | opt-in | Phase 4 not yet passed; Phase 1 vocabulary tag (`FeatureClass`) ships always-on regardless. |
| `fpcg_selector` | opt-in | Depends on `future_probe`; opt-in until GOAT gate passes. |
| `FeatureClass` enum + `ScreeningPruner::feature_class()` default | **always-on** (no feature gate) | Phase 1 ships independently; non-breaking trait addition with default `Detection`. |
| `ReviewMetrics::record_feature_read` + `FeatureReadSummary` | **always-on** (under existing `review_metrics` feature in katgpt-core) | Telemetry field; no behavior change without callers. |

**Path to promotion** (per Plan 292 T5.1):
1. Train a probe on a real behavior class (Refusal is easiest — binary labels, large effect size).
2. Wire `ActivationExtractor` into a real model forward pass (likely the `transformer.rs` / `inference_backend.rs` path).
3. Run G1–G4 on the test corpus.
4. If G1+G2+G3+G4 all pass → promote `future_probe` to default-on (selector stays opt-in, costs M forward passes per step).
5. If any fails → demote to permanent opt-in, keep Phase 1 vocabulary tag as the always-on output (the plan's "fallback success").

---

## What Shipped (Phases 1–3)

| Component | Path | Lines | Tests |
|-----------|------|-------|-------|
| `FeatureClass` enum + `ScreeningPruner::feature_class()` default | `crates/katgpt-core/src/traits.rs` | ~40 | 4 (incl. discriminants, default-is-Detection, EmotionDirections-is-Detection, FutureBehaviorProbe-is-Prediction) |
| `feature_class.rs` re-export shim + tests | `src/pruners/feature_class.rs` | ~80 | (same 4 tests) |
| `EmotionDirections::feature_class()` explicit override | `src/pruners/emotion_vector.rs` | +14 | inherited |
| `ReviewMetrics::record_feature_read` + `FeatureReadSummary` | `crates/katgpt-core/src/pruners/review_metrics.rs` | +75 | 3 (empty, by-class, display) |
| `FutureBehaviorProbe` primitive | `src/pruners/future_probe.rs` | ~620 | 13 (forecast contract ×4, hash stability, atomic swap, save/load round-trip, tamper rejection, bad magic, empty direction, feature class, displays) |
| `FpcgSelector` + traits + default generator | `src/pruners/fpcg_selector.rs` | ~640 | 12 (num=1, Positive pick, Negative pick, EOS termination, zero-alloc, argmax/argmin sanity, max_sentences, empty pool, direction flip, probe swap, discriminants) |
| Example: `future_probe_01_basic.rs` | `examples/future_probe_01_basic.rs` | ~130 | — |
| Example: `fpcg_01_basic.rs` | `examples/fpcg_01_basic.rs` | ~195 | — |

**Total: 32 unit tests across 3 modules + 2 runnable examples.** All green.

---

## Hand-off Checklist for Phase 4

- [ ] Train a `FutureBehaviorProbe` on a real behavior class (Refusal recommended — binary, large effect). Output as `FPPB` binary format via `save_to_bytes()`.
- [ ] Wire `ActivationExtractor` to a real model forward pass. Likely path: `src/transformer.rs` or `src/inference_backend.rs`, exposing the residual stream at `probe.layer()` at the sentence-end token.
- [ ] Acquire a test corpus with behavior labels. Paper repo: <https://github.com/kortukov/future_probes> — S=10 base × M=10 completion per sentence, refusal / prompt-injection / sycophancy behaviors.
- [ ] Run G1 (steering strength ≥ 30pp) on Positive vs Negative directions.
- [ ] Run G2 (PPL delta < 5%) vs unsteered baseline.
- [ ] Run G3 (format-filter rate < 10%).
- [ ] Run G4 (Pareto dominance vs `EmotionDirections` and CNA).
- [ ] Add `benches/future_probe_bench.rs` for G6 (latency < 200ns).
- [ ] Fill in the gate table above with real numbers.
- [ ] If G1+G2+G3+G4 PASS → promote `future_probe` to default-on (Plan 292 T5.1).
- [ ] If any FAIL → demote permanently, file `.issues/NNN_*` for the failure mode (Plan 292 T4.7 / T5.2).

---

## TL;DR

Phases 1–3 (engine primitives) shipped behind opt-in feature flags. The vocabulary tag (`FeatureClass`) is always-on and is the durable architectural output regardless of Phase 4 outcome. Phase 4 GOAT gate is **deferred** — it needs a trained probe (→ riir-train), a real-model forward-pass wiring (→ engine), and a behavior-labeled corpus (→ paper repo). Two of the seven gates (G5 zero-alloc, G7 BLAKE3 commitment) are already green on synthetic data; the other five need real-model evidence.
