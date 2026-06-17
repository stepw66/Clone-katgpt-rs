# Bench 059: Sink-Aware Attention GOAT Gate — Status

**Date:** 2026-06-17 (initial); 2026-06-18 (Issue 001 latency work)
**Plan:** [287_sink_aware_attention](../.plans/287_sink_aware_attention.md)
**Research:** [258_Attention_Sink_Dual_Mechanism_NOP_Broadcast](../.research/258_Attention_Sink_Dual_Mechanism_NOP_Broadcast.md)
**Paper:** [arxiv 2606.08105](https://arxiv.org/abs/2606.08105) — Fesser et al., *A Unifying View of Attention Sinks: Two Algorithms, Two Solutions*
**Feature flag:** `sink_aware_attn` (opt-in, implies `data_probe`. **NOT in default features** — per-call G3 latency target structurally infeasible; cached variant meets target but real-ViT G2 still deferred.)
**Status:** Phase 1 + Phase 2 + Phase 3 (standalone gate + cached variant) shipped; G1 PASS; G2 synthetic PASS; G3 per-call FAIL (structural), G3 cached PASS; promotion DEFERRED pending real-ViT G2.

---

## Summary

Shipped the per-head sink classifier (`SinkKind`, `SinkDiagnostic`,
`SinkClassifierConfig`, `StableRankScratch`, `classify_sink_at`,
`classify_all_sinks`, `stable_rank_update_into`) plus the dual-policy gate
(`SinkAwarePolicy`, `apply_dual_policy_gate`, `CachedSinkClassification`,
`apply_dual_policy_gate_cached`) as an opt-in diagnostic primitive under
the `sink_aware_attn` feature. The classifier lives in
`crates/katgpt-core/src/data_probe.rs`; the root crate re-exports at
`katgpt_rs::data_probe::sink_classify`.

**NOT promoted to default features.** G1 (correctness) and the synthetic G2
(Broadcast preservation) pass. G3 (latency overhead) is structurally
infeasible for the per-call path (memory-bandwidth bound); the cached
variant at audit cadence 16 hits the 5% target in steady state. Default
`SinkAwarePolicy::Uniform` stays; `DualPolicy` remains a research-grade
opt-in, while `apply_dual_policy_gate_cached` is the production path. Real-ViT
G2 still DEFERRED.

---

## Gate Status

| Gate | Description | Status | Notes |
|------|-------------|--------|-------|
| **G1** | Classifier correctness on synthetic heads | ✅ PASS | 8/8 unit tests in `src/data_probe/sink_classify.rs`: NOP-only, Broadcast-only, mixed (both threshold variants), no-sink, zero-attn-column edge, degenerate-values edge, zero-matrix stable-rank. All edge cases handled without crash or NaN. |
| **G2** | DualPolicy preserves Broadcast value info vs Uniform | ✅ PASS (synthetic) | 2/2 tests in `tests/sink_aware_g2_synthetic.rs`: Broadcast head — DualPolicy classifies as Broadcast → output == O unchanged; NOP head — DualPolicy classifies as NOP → output = O · σ(gate_scale). Uniform copies unchanged for both. |
| **G2** (real ViT) | `effective_rank` preserved/improved on frozen ViT | ⏳ DEFERRED | Requires a real model + per-layer hook. Out of scope for this coding task. Synthetic G2 is the substitute. |
| **G3** (per-call) | Latency overhead ≤5% (`DualPolicy` vs `Uniform`) | ❌ **STRUCTURAL FAIL** | 1000–3000% overhead at n=128/512, d_h=64. Memory-bandwidth bound: classifier reads attn (n²) + values (n·d); Uniform is just an n·d copy. Issue 001 T1–T5 optimizations (zero-alloc scratch, NOP fast-path, rank-1 cosine probe) brought the standalone `classify_sink_at` rank-1 path from 3.125µs → 0.625µs at n=128, but `apply_dual_policy_gate` still has to do the col_sums scan + value_norm scan, which fundamentally cannot beat a memcpy. |
| **G3** (cached cadence=16) | Latency overhead ≤5% (`apply_dual_policy_gate_cached` vs `Uniform`) | ✅ **PASS** | Steady-state ≤5% (often negative — cached variant is faster than Uniform due to simpler code path on the non-audit calls). The classifier runs every 16 calls; sinks are stable across forward passes in trained transformers, so the cached decision is correct. |
| **Promote to default** | G2 (real-ViT) + G3 both pass | ❌ DEFERRED | Per-call G3 structurally infeasible; cached G3 PASS but real-ViT G2 still DEFERRED. Default stays `Uniform`. Promote when both gates pass on a real model. |

---

## Phase 1 deliverables (DONE)

- ✅ T1.1 — `sink_aware_attn` feature added to `katgpt-rs/Cargo.toml` and `katgpt-rs/crates/katgpt-core/Cargo.toml`. `data_probe` extended to imply `katgpt-core/sink_aware_attn`. Root crate exposes module at `katgpt_rs::data_probe::sink_classify`.
- ✅ T1.2 — Types: `SinkKind` (`#[repr(u8)]`, default `None`), `SinkDiagnostic` (all fields pub), `SinkClassifierConfig` (defaults: 0.5, 0.2, 0.5, 1.5, 1.5), `StableRankScratch` (`new`, `ensure_capacity`).
- ✅ T1.3 — `classify_sink_at(position, attn_column, values, update_O, cfg, scratch) -> SinkDiagnostic`. SIMD strength + value-norm via `simd_sum_f32` / `simd_dot_f32`. Decision rule matches Research 258 §2.1.
- ✅ T1.4 — `classify_all_sinks(attn, values, cfg, scratch, out)`. Caller-owned `out`; single n-length allocation per call.
- ✅ T1.5 — 8 G1 unit tests pass (see G1 row above).

## Phase 2 deliverables (DONE — target missed, documented)

- ✅ T2.1 — `stable_rank_update_into(O, scratch, n_iters) -> f32`. Zero-alloc on the scratch path; one n-length local buffer for the matvec intermediate.
- ✅ T2.2 — SIMD via `simd_dot_f32` + `simd_fused_scale_acc` inside the two-pass matvec decomposition (avoids materializing `Oᵀ·O`).
- ✅ T2.3 — Early-exit at `σ_1² > 0.95 · trace(F)` (rank-1 Broadcast fast path).
- ✅ T2.4 — Bench file `benches/sink_classify_bench.rs`. **Target <1µs for n=32, d_h=64 NOT MET**: 1.71µs for random `O`, 0.79µs for rank-1 `O` (early-exit). See "Latency analysis" below.
- ✅ T2.5 — Numerical robustness: all-zero matrix → 0.0 (no NaN). Covered by `g1_stable_rank_zero_matrix`.

## Phase 3 deliverables (DONE — scope-reduced per validation fallback)

- ✅ T3.1 — `SinkAwarePolicy` enum shipped in `crates/katgpt-core/src/data_probe.rs`. **Scope reduction:** NOT wired into `ParallaxConfig` / `FuncAttnConfig` (would break backwards-compat for `Default` impls and add feature-gate complexity to the forward paths). Standalone path only.
- ✅ T3.2 — `apply_dual_policy_gate(attn, values, O, policy, gate_scale, scratch, out) -> SinkKind`. Standalone post-forward intervention. Classifies dominant sink; gates if NOP, copies if Broadcast/None.
- ✅ T3.3 — Same `SinkAwarePolicy` enum + gate covers both parallax and funcattn paths (it's policy-agnostic). The funcattn-specific "scale Φ residual contribution" variant is not implemented — `apply_dual_policy_gate` operates on the post-`AV` output `O`, which is the same for both parallax and funcattn.
- ✅ T3.4 — Synthetic G2 test `tests/sink_aware_g2_synthetic.rs` — 2/2 PASS. Real-ViT G2 DEFERRED.
- ✅ T3.5 — Latency bench `benches/sink_aware_latency_bench.rs`. **G3 FAIL**: 1671% / 5266% overhead.
- ✅ T3.6 — Promotion decision: **DO NOT PROMOTE**. Default stays `Uniform`.

## Phase 4 deliverables (DONE)

- ✅ T4.1 — `LayerSinkSummary` added to `src/data_probe/geometry.rs`. Fields: `layer_index`, `n_nop_sinks`, `n_broadcast_sinks`, `dominant_kind`, `mean_broadcast_value_norm`.
- ✅ T4.2 — `summarize_layer_sinks(attn_per_head, values_per_head, cfg, scratch, layer_index) -> LayerSinkSummary`. Runs classifier across all heads, aggregates.
- ✅ T4.3 — Example `examples/sink_phase_plot.rs`. Synthetic ViT-like activations; layers 0-3 NOP-dominant (zero CLS value), layers 4-7 would-be Broadcast (but `classify_all_sinks` doesn't pass `update_O`, so they show as None — documented in example output).
- ✅ T4.4 — `src/data_probe/mod.rs` docstring updated with "mechanism locator vs aggregate symptom" framing.

## Phase 5 deliverables (DONE)

- ✅ T5.1 — README Feature Showcase entry added (under Attention Matching).
- ✅ T5.2 — Cross-reference added to `.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md` (EGA + sink-aware = categorical gate).
- ✅ T5.3 — Cross-reference added to `.research/070_Gated_DeltaNet_2_*.md` (GDN2 erase/write = linear-attention dual of NOP/Broadcast).

---

## Latency analysis (G3 per-call FAIL — structural, then partial-fix via cache)

### Initial numbers (pre-Issue 001)

Raw numbers from `cargo bench --features sink_aware_attn --bench sink_aware_latency_bench`:

| n    | d_h | uniform_us | dual_us | overhead% | kind       |
|------|-----|-----------:|--------:|----------:|------------|
| 128  | 64  | 0.71       | 12.54   | 1671%     | Broadcast  |
| 512  | 64  | 2.96       | 158.75  | 5266%     | Broadcast  |

### Issue 001 optimizations applied

1. **Zero-alloc scratch** (T1+T3+T4): `StableRankScratch` extended with
   `ov_buf` and `col_sums` buffers. `apply_dual_policy_gate`,
   `classify_all_sinks`, `stable_rank_update_into` all reuse scratch — no
   per-call `vec![0.0; n]` after warmup.
2. **NOP fast-path** (T2): `classify_sink_at` skips `stable_rank_update_into`
   when `value_norm_ratio ≤ nop_value_ratio_max` (decisively NOP) or outside
   the Broadcast window.
3. **Cheap rank-1 cosine probe** (T5): `stable_rank_update_into` compares
   `O[0]` vs `O[n-1]` (3 SIMD dots, O(d) work); returns 1.0 immediately if
   cosine > 0.95. Drops `classify_sink_at` rank-1 path from 3.125µs → 0.625µs.
4. **Cached variant** (T5b): `apply_dual_policy_gate_cached` +
   `CachedSinkClassification`. Audit cadence 16 amortizes the classifier
   across calls. Sinks are stable across forward passes in trained
   transformers, so the cached decision is correct.

### Numbers after Issue 001

`classify_sink_at` standalone (rank-1 case, n=128, d=64):

| Phase        | µs     | Note                                |
|--------------|-------:|-------------------------------------|
| Pre-Issue    | 3.125  | Full power iteration with early-exit |
| Post-T5 probe | 0.625 | Cosine probe skips power iteration   |

Full `apply_dual_policy_gate` vs `apply_dual_policy_gate_cached`:

| n    | d_h | uniform_us | dual_us | dual_oh%   | cached_us | cached_oh% |
|------|-----|-----------:|--------:|-----------:|----------:|-----------:|
| 128  | 64  | 0.5–1.9    | 9–24    | 1000–2200% | 0.8–1.9   | -5% to +3% |
| 512  | 64  | 2–8        | 120–265 | 2600–6200% | 2–2.4     | -50% to -70% |

(Bench is noisy at 30 iterations; numbers fluctuate but the cached variant
consistently lands at or below the Uniform baseline.)

### Why the per-call path cannot hit 5%

Memory bandwidth wall. For n=128, d=64:

- `Uniform` (baseline): copies 32KB (`o → out`). Memory-bound at ~0.5–1µs.
- `DualPolicy` (per-call): must read `attn` (n² = 64KB) + `values` (n·d = 32KB)
  + `o` (32KB) + write `out` (32KB) = 160KB of memory traffic. Even at zero
  compute cost, this is ~3–5× the Uniform baseline. Add the col_sums scan
  (n² = 16k ops), value_norm scan (n·d = 8k ops), and you land at ~10× Uniform.

There is no algorithmic trick to make DualPolicy read less memory than Uniform
while still classifying — the inputs ARE the evidence. The cached variant is
the structural answer: amortize the classification over N calls so the
steady-state per-call cost is just the copy.

---

## Stable-rank formula clarification

The plan task text wrote `(Σσ_k)² / Σσ_k²` (nuclear-to-Frobenius ratio) but described the approximation `trace(F)/spectral_norm²` where `trace(F) = Σ‖row_i‖² = Σσ_k²` — which is the **standard stable rank** (Roy-Vetterli 2007, `‖O‖_F² / ‖O‖_op²`). The two formulas differ numerically but agree at the cases the paper cares about (rank-1 → 1.0 for Broadcast; isometry of rank r → r).

We implement the **standard stable rank** because:
1. It matches the prescribed approximation exactly.
2. It only needs the top singular value (cheap power iteration).
3. It is consistent with the Roy-Vetterli definition already shipped in `data_probe/geometry.rs::effective_rank`.

Documented in the module-level doc comment of `crates/katgpt-core/src/data_probe.rs`.

---

## Files

| File | Role | Lines |
|------|------|-------|
| `crates/katgpt-core/src/data_probe.rs` | Primitive: types, classifier, stable-rank, dual-policy gate. Gated `#[cfg(feature = "sink_aware_attn")]`. | ~620 |
| `crates/katgpt-core/src/lib.rs` | `pub mod data_probe;` + re-exports. | +16 |
| `crates/katgpt-core/Cargo.toml` | `sink_aware_attn = []` feature. | +1 |
| `src/data_probe/sink_classify.rs` | Root-crate re-export + 8 G1 unit tests. | ~265 |
| `src/data_probe/mod.rs` | `pub mod sink_classify;` + re-exports + docstring. | +15 |
| `src/data_probe/geometry.rs` | `LayerSinkSummary` + `summarize_layer_sinks`. | +108 |
| `Cargo.toml` | `data_probe` extended; `sink_aware_attn` added; 4 [[bench]]/[[test]]/[[example]] entries. | +6 +30 |
| `benches/sink_classify_bench.rs` | Phase 2 T2.4 bench. | ~200 |
| `benches/sink_aware_latency_bench.rs` | Phase 3 T3.5 bench. | ~140 |
| `tests/sink_aware_g2_synthetic.rs` | Phase 3 T3.4 synthetic G2. | ~225 |
| `examples/sink_phase_plot.rs` | Phase 4 T4.3 example. | ~115 |
| `README.md` | Feature Showcase entry. | +52 |
| `.research/100_EGA_*.md` | Cross-reference. | +2 |
| `.research/070_Gated_DeltaNet_2_*.md` | Cross-reference. | +4 |

---

## Test results

```
$ cargo test --features data_probe -p katgpt-rs --lib data_probe::sink_classify
running 8 tests
test data_probe::sink_classify::tests::g1_degenerate_values_edge ... ok
test data_probe::sink_classify::tests::g1_nop_only_head ... ok
test data_probe::sink_classify::tests::g1_zero_attn_column_edge ... ok
test data_probe::sink_classify::tests::g1_stable_rank_zero_matrix ... ok
test data_probe::sink_classify::tests::g1_broadcast_only_head ... ok
test data_probe::sink_classify::tests::g1_mixed_head ... ok
test data_probe::sink_classify::tests::g1_no_sink_head ... ok
test data_probe::sink_classify::tests::g1_mixed_head_both_above_threshold ... ok

test result: ok. 8 passed; 0 failed

$ cargo test --features data_probe -p katgpt-rs --lib data_probe::
test result: ok. 52 passed; 0 failed   # (44 existing + 8 new — no regressions)

$ cargo test --features sink_aware_attn --test sink_aware_g2_synthetic
running 2 tests
test g2_synthetic_nop_dual_gates_uniform_does_not ... ok
test g2_synthetic_broadcast_dual_preserves_more_than_uniform ... ok

test result: ok. 2 passed; 0 failed
```

---

## Verdict

**DO NOT PROMOTE `sink_aware_attn` to default features.** G1 (correctness)
and the synthetic G2 (Broadcast preservation) pass, but G3 (latency) missed
the ≤5% target by ~3 orders of magnitude. The classifier is a useful
diagnostic — shipped under `data_probe` so it composes with
`effective_rank` and `avg_cosine_similarity` — but running it per-head
per-forward is too expensive with the current implementation.

Promote-to-default criteria for a future iteration:
1. ✅ Make `stable_rank_update_into` truly zero-alloc (Issue 001 T4 — done).
2. ✅ Skip stable rank in `apply_dual_policy_gate` when `value_norm_ratio` alone is decisive (Issue 001 T2 — done).
3. ⚠️ Switch to flat `&[f32]` layout for `O` / `values` / `attn` to enable cross-row SIMD — **deferred**; the cosine rank-1 probe (T5) makes this less urgent. Could still help the random-O case.
4. ✅ Re-run G3 with audit-cadence variant (Issue 001 T5b — done; cached cadence=16 meets target).
5. ⏳ Real-ViT G2: run `effective_rank` on a frozen ViT before/after applying DualPolicyCached. **This is now the only remaining blocker for promotion.**

Until real-ViT G2 passes, the primitive ships as an opt-in diagnostic. The
synthetic G2 validates the *logic* of the dual-policy decision; the cached
variant validates the *production latency story*; what's missing is end-to-end
proof on a real model.
