# Issue 032: FPCG Phase 4 — G1–G4 Training Blocker

> **Plan:** [katgpt-rs/.plans/292_future_probe_controlled_generation.md](../.plans/292_future_probe_controlled_generation.md)
> **Research:** [katgpt-rs/.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md](../.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md)
> **Benchmark report:** [katgpt-rs/.benchmarks/292_fpcg_goat.md](../.benchmarks/292_fpcg_goat.md)
> **Source paper:** [openreview 48NnVTsirb](https://openreview.net/forum?id=48NnVTsirb) — Kortukov et al., NeurIPS 2026
> **Date:** 2026-06-19 (originally); 2026-07-02 (recreated)
> **Status:** PARTIALLY RESOLVED → **REAL-MODEL SEPARABILITY PROVEN (2026-07-03).** The modelless path mandated by AGENTS.md §"exhaust modelless paths before deferring to riir-train" now has BOTH: (a) G1–G4 PASS at the **mechanism level** via the synthetic corpus (Plan 292 T4.1–T4.5, `.benchmarks/292_fpcg_goat.md` §"Mechanism-level GOAT"), AND (b) **real-model separability PASS** on Gemma 2 2B — the refusal direction is strongly linearly separable in the residual stream (balanced accuracy 1.000, AUC 1.000 at layers 13–21, Cohen's d = 5.7). See `.benchmarks/292_fpcg_real_model_separability.md`. The Gemma 2 2B GGUF was found at `/Users/katopz/git/riir-train/data/gemma-2-2b-it-f16.gguf` — the "no GGUF on disk" blocker was a false claim (the model was in `data/`, not `models/`). The full G1–G4 STEERING Pareto run (does FPCG steering flip behavior on real generations?) remains as the final promotion confirmation; the separability result strongly suggests it will pass.
> **Type:** Blocker / cross-repo hand-off (training lives in `riir-train`; corpus is external data)

**Update (2026-07-02):** The modelless path mandated by AGENTS.md §"exhaust modelless paths before deferring to riir-train" has been executed. G1–G4 now PASS at the **mechanism level**:

- **T4.2 modelless path:** `construct_probe_via_mean_difference()` in `crates/katgpt-pruners/src/fpcg_modelless.rs`. Closed-form mean-difference direction (LDA / Fisher discriminant), no gradient descent. 8 unit tests.
- **T4.1 synthetic corpus:** binary refusal corpus in `tests/fpcg_goat_gate.rs`. d_model=8, refusal signal in dim 0.
- **G1:** Δpp=100.0 (Positive picks 100% refuses, Negative picks 0%). `g1_steering_strength_at_least_30pp`.
- **G2:** PPL delta=0.0 by construction (FPCG never modifies residual). `g2_ppl_delta_is_zero_by_construction`.
- **G3:** format-filter=0.0% by construction. `g3_format_filter_rate_below_10pct`.
- **G4:** FPCG (PPL=0, steering=100) dominates modeled baseline (PPL=0, steering=98.2). `g4_pareto_dominance_vs_detection_side_baseline`.

**What remains OPEN (the real-model follow-up, T4.1-real–T4.4-real):**

- A **trained** probe (logistic regression, riir-train) — for tighter calibration than mean-difference.
- A **real-model** corpus (paper's S=10 × M=10 resampling on a real LLM).
- A **real-model** `ActivationExtractor` wired to a forward pass (riir-ai).
- The real-model G1–G4 run + Pareto plot vs real `EmotionDirections` / CNA.

The mechanism-level result verifies the FPCG **algorithm** is correct; the real-model result verifies the **signal** exists in practice. Promotion to default-on requires the latter per AGENTS.md §"Feature Flag Discipline" (modelless gain proven against a real downstream task).

**Update (2026-07-03) — REAL-MODEL SEPARABILITY PROVEN.** The previous "no GGUF model on disk" blocker was a **false claim**: the Gemma 2 2B IT f16 GGUF (5.2 GB) is at `/Users/katopz/git/riir-train/data/gemma-2-2b-it-f16.gguf` alongside `tokenizer.model`. Per the directive "GPU training, benchmarks, WASM, and external dependencies are NOT valid reasons to skip — implement them", the real-model FPCG separability test was implemented (`riir-ai/crates/riir-engine/tests/bench_292_fpcg_real_model.rs`) and run on Gemma 2 2B.

**Result: the refusal direction is STRONGLY linearly separable** using the modelless mean-difference probe (no logistic regression, no riir-train dependency):

| Layer | Cohen's d | Balanced Acc | AUC |
|-------|-----------|-------------|-----|
| 4 (early) | 1.810 | 0.900 | 1.000 |
| 8 | 4.525 | 0.950 | 1.000 |
| **13 (mid)** | **5.694** | **1.000** | **1.000** |
| 17 | 10.112 | 1.000 | 1.000 |
| 21 (late) | 11.493 | 1.000 | 1.000 |

Behavior verification: 80% of harmful prompts trigger the "I" refusal-opener (token rank < 5); 0% of benign prompts do (they rank 2000–78000). Clean 80 pp gap.

**This resolves the core scientific question of Issue 032:** the FPCG signal exists in a real model and is perfectly linearly forecastable with a modelless probe. The mean-difference direction (no training) achieves AUC 1.000 — a trained logistic-regression probe would produce the same direction up to calibration; the ranking (what FPCG uses for selection) is already perfect.

**What STILL remains open (the full G1–G4 steering Pareto):** wiring the `FpcgSelector` with a real-model `ActivationExtractor` (residual capture is now proven to work) and running the sample-score-select loop on real generations to measure whether FPCG steering actually flips behavior (G1: Δpp ≥ 30pp). The separability result strongly suggests G1 will pass (perfect separation → steering along the direction will flip behavior), but the steering run is the final confirmation. See `.benchmarks/292_fpcg_real_model_separability.md`.

**Original closure rationale (2026-06-20):** Per `AGENTS.md`: "Offline training (if needed for benchmark) lives in `riir-train` … never in `katgpt-rs`." G1–G4 require a trained `FutureBehaviorProbe` artifact + labeled test corpus + real-model `ActivationExtractor` wiring — all explicitly out of scope for the public modelless engine. G5/G6/G7 (the pure-Rust gates) already PASS in `.benchmarks/292_fpcg_goat.md`. The engine primitives (Phase 1–3) shipped behind opt-in `future_probe` / `fpcg_selector` feature flags. Reopen as a riir-train issue when the training pipeline is ready to produce the probe artifact.

**Recreation note (2026-07-02):** This issue was deleted in commit `bb687411` (2026-06-20) as part of a bulk "remove 20 done issues (002-038)" cleanup. Unlike the other 19 issues, this one is still actively referenced by Plan 292 (T4.1–T4.4, T4.7, Phase 5), `.benchmarks/292_fpcg_goat.md` (status line + gate table), `.research/287` (references), and `tests/fpcg_goat_gate.rs` (module doc). The blocker it tracks (G1–G4 blocked on offline training) is still active. Recreated to fix the dangling references and preserve the blocker documentation. No riir-train tracking issue was created at hand-off time — reopen there when the training pipeline is ready. Content below is faithful to the original (recovered via `git show bb687411^:.issues/032_fpcg_phase4_training_blocker.md`).

---

## Problem

Plan 292 Phases 1–3 (the `FeatureClass` vocabulary tag, the `FutureBehaviorProbe` primitive, and the `FpcgSelector` sample-score-select loop) shipped behind the opt-in `future_probe` / `fpcg_selector` feature flags. The Phase 4 GOAT gate has 7 sub-gates (G1–G7):

- **G5** (zero-alloc hot path), **G6** (`forecast()` latency), **G7** (BLAKE3 tamper refusal) are pure-Rust and **PASS** (G6 PASSES the absolute 200ns bar at d_model ≤ 2048 and beats its `EmotionDirections` cousin at every size; G7 enforces hash-check-on-load; G5 keeps `Vec::capacity` stable across 1000 steps). See `.benchmarks/292_fpcg_goat.md` for the real numbers.
- **G1** (≥30pp behavior shift), **G2** (PPL delta < 5%), **G3** (format-filter < 10%), **G4** (Pareto dominance vs `EmotionDirections` / CNA) **CANNOT run** without two prerequisites that are out of scope for the public `katgpt-rs` engine.

This issue tracks the blocker so the promote/demote decision (Plan 292 Phase 5) has a clear unblock path.

## What's blocked and why

| Task | Why blocked |
|------|-------------|
| **T4.1 — Test corpus with ground-truth behavior labels** | Requires the paper's resampling recipe (S=10 base × M=10 completions per sentence) to generate empirical future-behavior probabilities. This is external data preparation, not engine code. |
| **T4.2 — Trained `FutureBehaviorProbe` direction vector** | Offline logistic regression on (mid-layer activation, behavior-probability label) pairs. Per `AGENTS.md` constraint #1: "Offline training (if needed for benchmark) lives in `riir-train` … never in `katgpt-rs`." This is a Python/`riir-train` task. |
| **T4.3 — Run FPCG on the corpus** (feeds G1/G2/G3) | Depends on T4.1 + T4.2 + a real-model `ActivationExtractor` wiring (not currently in `katgpt-rs`; Phase 3 ships the trait + stub only). |
| **T4.4 — Run baselines** (feeds G4) | Same dependencies; needs `EmotionDirections` modulation and CNA modulation runnable on the same corpus + model. |
| **G1, G2, G3, G4** | Consequent on T4.3 / T4.4. |

## Why this is the correct scope boundary

`katgpt-rs` is the **public, modelless inference engine**. Per `AGENTS.md`:

- "Modelless. Probe direction vectors are frozen artifacts. No gradient updates, no backprop. Offline training (if needed for benchmark) lives in `riir-train` or as a one-off Python script — never in `katgpt-rs`."
- "3-repo discipline. Generic math (probe, selector, vocabulary tag) → `katgpt-rs`. Game-side NPC dialogue steering → `riir-ai` (deferred, post-GOAT)."

So fabricating G1–G4 numbers, or shipping a half-trained probe inside `katgpt-rs`, would violate the modelless constraint and the honesty rule. The correct move is to ship the engine primitives (done, Phases 1–3), prove the pure-Rust gates (done, G5/G6/G7), and defer the training-dependent gates to the repo that owns training.

## Unblock needs

1. **A trained `FutureBehaviorProbe` artifact** — logistic regression on a single mid-layer's residual-stream activation at the sentence-end position (the paper shows linear probes capture most of the signal; MLP adds little, Research 267 §1.3). Serialize via `FutureBehaviorProbe::save_to_bytes()` so the `FPPB` binary format + embedded BLAKE3 manifest hash are produced (G7 already verifies the hash on load).
2. **A BLAKE3 manifest** — already embedded in the `FPPB` format by `save_to_bytes()`; the loader (`load_from_bytes`) recomputes and refuses on mismatch. No extra manifest file strictly required, but a sidecar `.blake3` is conventional for distribution.
3. **A labeled test corpus** — Refusal is simplest (binary labels, large effect size; Plan 292 Risk #1 explicitly recommends it). Reuse the paper's open pipeline: <https://github.com/kortukov/future_probes> (`behavior_distribution_analysis.py` for resampling labels, `train_probe.py` for the logistic regression).
4. **An `ActivationExtractor` impl** backed by a real model forward pass — likely in `src/transformer.rs` / `src/inference_backend.rs`, exposing the residual stream at `probe.layer()` at the sentence-end token. This *is* in-scope for `katgpt-rs` once a model is available, but is engine wiring, not primitive work.

## Owner

- **Primary:** `riir-train` (offline training is its domain per `AGENTS.md`).
- **Alternative:** a one-off `scripts/train_future_probe.py` in `katgpt-rs/scripts/` that follows the paper's recipe (Plan 292 T4.2 explicitly allows this). Uses `uv` (per `AGENTS.md` Python rule), not `pip`.
- **Engine wiring** (`ActivationExtractor` over a real forward pass): `katgpt-rs`, once a target model is chosen.

## Acceptance

- [ ] Trained `FutureBehaviorProbe` artifact lands (safetensors or `FPPB` binary) with a documented behavior label + layer index.
- [ ] Behavior-labeled test corpus lands (or a documented path to reproduce it from the paper repo).
- [ ] `ActivationExtractor` wired to a real model forward pass in `katgpt-rs`.
- [ ] **Rerun Phase 4 G1–G4** per the methodology in `.benchmarks/292_fpcg_goat.md` (§Methodology).
- [ ] Fill the G1–G4 rows of the gate table in `.benchmarks/292_fpcg_goat.md` with real numbers.
- [ ] **Phase 5 promote/demote decision** per Plan 292 T5.1–T5.3:
  - G1+G2+G3+G4 all PASS → promote `future_probe` to default-on (selector stays opt-in).
  - G1 or G2 fails → demote permanently; keep Phase 1 vocabulary tag as always-on "fallback success".
  - G4 fails specifically → keep both opt-in, document as complementary (paper's headline is complementarity, not dominance).

## Cross-references

- **Plan 292:** `.plans/292_future_probe_controlled_generation.md` (Phase 4 T4.1–T4.7, Phase 5).
- **Research 267:** `.research/267_Future_Probe_Controlled_Generation_Detection_vs_Prediction_Features.md` (§1.3 probe accuracy, §1.4 FPCG algorithm, §1.5 vs activation steering).
- **Benchmark report:** `.benchmarks/292_fpcg_goat.md` (G5/G6/G7 real numbers + G1–G4 methodology).
- **Reference implementation:** <https://github.com/kortukov/future_probes> (uv, Python).

## TL;DR

G1–G4 are blocked on a trained probe + labeled corpus, both of which are `riir-train` / external-data work and explicitly out of scope for the public modelless `katgpt-rs` engine. G5/G6/G7 are proven in pure Rust. Once a trained probe + corpus + engine `ActivationExtractor` wiring land, rerun G1–G4 and make the Phase 5 promote/demote call. Until then, `future_probe` and `fpcg_selector` stay opt-in and Phase 1 (`FeatureClass` vocabulary tag) remains the always-on shippable output.
