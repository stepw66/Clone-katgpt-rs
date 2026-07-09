# Plan 420: Modelless KV Cache Consolidation — IB-Gated Value Mean-Shift

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/401_Bottlenecked_Transformer_KV_Cache_Consolidation.md](../.research/401_Bottlenecked_Transformer_KV_Cache_Consolidation.md)
**Source paper:** [arXiv:2505.16950](https://arxiv.org/abs/2505.16950) — Bottlenecked Transformers (Oomerjee et al., 2025/2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/kv_consolidation/` (new module) + Cargo feature `kv_consolidation`
**Status:** Phase 1 COMPLETE — quality gain REFUTED by §3.6 PoC (2026-07-09). Phases 2–4 PERMANENTLY SHELVED — riir-train Plan 313 confirmed the refutation on a TRAINED model (31% accuracy, 0.00pp gain). The paper's quality benefit is inseparable from its TRAINED Cache Processor; the modelless mean-shift does not capture it.

---

## Goal

Ship a modelless KV cache consolidation primitive that periodically rewrites KV value vectors in-place at surprise-triggered step boundaries, using a deterministic sigmoid-gated mean-shift toward the recent step's value centroid. This fills the cache-operator gap: all existing KV work (213 Still, 233 AM, 109 ShardDrop) is compression/selection; none rewrites for quality without reducing footprint.

The primitive is justified by Information Bottleneck theory (Research 401 §1.2): autoregressive training makes the KV cache minimally compressive of input (high I(X;Z)); periodic consolidation reduces I(X;Z) while preserving predictive information I(Z;Y), improving generalization.

**The GOAT gate is PoC-gated per §3.6:** architectural coverage is confirmed, but quality parity with the paper's trained Cache Processor has been REFUTED on both untrained (Phase 1) and trained (Plan 313) models. The primitive is permanently Gain (architectural only) — no feature flag will ship.

**KV cache stack slot:** consolidation (new slot — distinct from compression/selection/quantization). No demote-loser interaction (different operation class).

---

## Phase 1 — §3.6 Defend-Wrong PoC (MANDATORY before any feature flag)

The PoC lives in `katgpt-rs/crates/katgpt-core/benches/bench_420_kv_consolidation_poc.rs`. It must defend OR refute the quality-parity claim. Use `CARGO_TARGET_DIR=/tmp/kv_consolidation_poc` and clean up when done.

### Tasks

- [x] **T1.1** Build a controlled toy reasoning task: few-shot in-context addition in a micro-GPT (single-layer, d_model=64, 8 heads). 200 problems × 5 few-shot examples. Metric: exact-match, token-accuracy, NLL under teacher forcing (greedy decode is degenerate on untrained models — diverges immediately).
- [x] **T1.2** Implement the three competitors in `bench_420_kv_consolidation_poc.rs` (bench-only, no feature flag):
  1. **Baseline** — vanilla KV cache, no consolidation.
  2. **Modelless consolidation** — sigmoid-gated value mean-shift at step boundaries (every newline token), layer-decaying gate `g_c^(ℓ) = g_max · sigmoid(−λ · ℓ/L)` (degenerate: sigmoid(0)=0.5 for single-layer), top-k=32 reconsolidation by attention mass, recent step window R=64.
  3. **Random-rewrite control** — same selection, same per-dimension magnitude, random sign. Tests whether the mean-shift direction matters.
- [x] **T1.3** Run the PoC: 200 problems × 3 competitors × 3 seeds = 1800 evaluations. Self-test confirms consolidation code is correct (keys unchanged, values modified, variance reduced 74.6%).
- [x] **T1.4** Verdict gate: **QUALITY GAIN REFUTED.** Consolidation ≈ baseline (Δtoken_acc = -0.06pp, ΔNLL = +0.0001). Consolidation ≈ random-rewrite (ΔNLL = +0.0005). The modelless mean-shift has no effect on an untrained model — as expected (the IB argument requires a trained model whose KV cache carries learned extraneous detail). Phase 2 NOT proceeding.
- [x] **T1.5** Sweep key hyperparameters: `g_max ∈ {0.1, 0.3, 0.5}`, `k ∈ {16, 32, 64}`. λ is degenerate for single-layer (sigmoid(0)=0.5 regardless). R fixed at 64 (steps are ~13 tokens, never capped). Result: **zero sensitivity** to any hyperparameter — all configs produce identical token_acc (0.0133) and near-identical NLL (7.8627–7.8630). Confirms consolidation is inert on untrained models.

---

## Phase 2 — Primitive Skeleton (feature flag, only if Phase 1 passes)

### Tasks

- [ ] **T2.1** Create `crates/katgpt-core/src/kv_consolidation/mod.rs` with:
  - `KvConsolidationConfig` struct: `g_max`, `lambda`, `k`, `rsw_len`, `trigger: ConsolidationTrigger` enum (`NewlineToken | SurpriseGate { threshold }`)
  - `KvConsolidator` struct: holds config + scratch buffers (pre-allocated, zero-alloc hot path per AGENTS.md)
  - `consolidate(&mut self, kv_cache: &mut KvCache, attention_weights: &[f32], layer: usize) -> ConsolidationReport`
- [ ] **T2.2** Implement the consolidation update in `kv_consolidation/ops.rs`:
  - `consolidate_recent(values: &mut [f32], step_indices: &[usize], gate: f32)` — mean-shift recent step values toward their centroid
  - `reconsolidate_recalled(values: &mut [f32], recalled_indices: &[usize], attention_mass: &[f32], step_centroid: &[f32], gate: f32)` — mean-shift recalled values toward step centroid, attention-weighted
  - Both: sigmoid gate, only value vectors (keys untouched), zero-allocation (use scratch buffers)
- [ ] **T2.3** Implement the selection in `kv_consolidation/select.rs`:
  - `select_topk_by_attention_mass(attention: &[f32], step_indices: &[usize], k: usize) -> Vec<usize>` — top-k prior positions by mean attention from recent step
  - Use a fixed-size max-heap (pre-allocated `[usize; K_MAX]`) — no Vec allocation in hot path
- [ ] **T2.4** Implement the trigger in `kv_consolidation/trigger.rs`:
  - `NewlineToken` — fires on `\n` token ID
  - `SurpriseGate` — fires when entropy exceeds threshold (reuse SwiR's block-relative entropy if available, else δ-Mem surprise gate)
- [ ] **T2.5** Add Cargo feature `kv_consolidation = []` to `crates/katgpt-core/Cargo.toml`. Off by default.
- [ ] **T2.6** Wire into the KV cache update path: after each token decode, check trigger; if fired, call `consolidate()`. Feature-gated.

---

## Phase 3 — GOAT Gate (benchmark + promote/demote)

### Tasks

- [ ] **T3.1** Write `crates/katgpt-core/benches/kv_consolidation_bench.rs`:
  - G1 (correctness): consolidated cache produces valid attention distribution (no NaN, sums to 1 after softmax)
  - G2 (quality): exact-match accuracy on the Phase 1 toy task with the shipped primitive (must match Phase 1 PoC results within noise)
  - G3 (no-regression): consolidation overhead < 5% of decode time (consolidation is invoked infrequently — once per step boundary, not per token)
  - G4 (alloc-free): consolidation hot path does zero heap allocations (use scratch buffers, verified by counting allocations in a debug build)
- [ ] **T3.2** Run the GOAT gate. Record results in `.benchmarks/420_kv_consolidation_goat.md`.
- [ ] **T3.3** Verdict:
  - All gates pass + Phase 1 showed ≥2pp quality gain → **promote to default** (`kv_consolidation` added to `default` features).
  - G1/G3/G4 pass but G2 (quality) fails or Phase 1 gain < 2pp → **keep opt-in**. Record honest result.
  - G1 fails → **fix or shelve**. Consolidation must not corrupt the cache.

---

## Phase 4 — Fusion Extensions (optional, after Phase 3 promotion)

### Tasks

- [ ] **T4.1** F2 (subspace projection): replace mean-shift with PCA-subspace projection of recalled values onto the recent step's principal components. Bench against mean-shift version. Promote if better.
- [ ] **T4.2** F3 (conformal trigger): replace entropy trigger with conformal-interval-width trigger (Plan 340). Consolidate only when calibrated uncertainty is high. Bench.
- [ ] **T4.3** Compose with SegmentCheckpoint (Plan 226): checkpoint AFTER consolidation, so checkpoints store the consolidated state. Reduces checkpoint storage redundancy.
- [ ] **T4.4** Compose with δ-Mem (Plan 053): use δ-Mem's delta-rule as the consolidation update mechanism instead of mean-shift. Tests the revival hypothesis (Research 401 §2.5).

---

## Notes

- **No training anywhere.** The entire primitive is deterministic: attention scores for selection, mean value for the shift direction, sigmoid for gating, entropy for triggering. No learned parameters.
- **KV cache stack slot:** consolidation (new slot). Does not compete with compression (213/233) or quantization (039) — can compose with them (consolidate first, then compress).
- **The PoC is the gate.** Phase 1's verdict determines whether this primitive ships at all. If the modelless mean-shift doesn't beat baseline by ≥2pp, the consolidation concept is architectural-only and gets shelved honestly.
- **Latency budget:** consolidation runs once per step boundary (every ~64 tokens with R=64), not per token. Overhead should be negligible relative to 64 forward passes. G3 target: < 5% of decode time.
- **The paper's §6.5 finding guides the design:** values only (not keys), early layers (decaying gate), moderate magnitude (sigmoid gate with small g_max). The modelless version should mirror these empirical constraints.
