# Benchmark 283: Self-Advantage Recursion Gate — GOAT Gate

**Plan:** [279 → 283 Self-Advantage Recursion Gate](../.plans/283_self_advantage_recursion_gate.md)
**Research:** [250 Latent Recursion Policy Improvement](../.research/250_Latent_Recursion_Policy_Improvement_Advantage_Margin.md)
**Source:** [arxiv:2511.16886](https://arxiv.org/abs/2511.16886) — "Latent Reasoning in TRMs is Secretly a Policy Improvement Operator"
**Date:** 2026-06-16
**Verdict:** ❌ **NOT GOAT** — keep `self_advantage_gate` opt-in. G1/G2/G4 pass; G3 (latency <1µs) fails at vocab=1024.

---

## GOAT Gate Results

| Gate | Criterion | Target | Result | Status |
|------|-----------|--------|--------|--------|
| **G1** | Forward-pass reduction (no-gate vs gated) | ≥ 2× | 2.68×–6.76× across vocab {8,32,128,1024} | ✅ PASS |
| **G2** | Argmax quality preservation | ≥ 95% | 100.0% at all vocab sizes | ✅ PASS |
| **G3** | `self_advantage()` per-call latency | < 1 µs | 0.041µs (V=8) … 3.958µs (V=1024) | ❌ FAIL |
| **G4** | Robustness across vocab sizes {8,32,128,1024} | all pass G1+G2 | 4/4 pass G1+G2 | ✅ PASS |

**Overall: 3/4 gates pass → NOT GOAT → feature stays opt-in.**

---

## Detailed Results

### G1/G2: Forward-pass reduction + quality (geometric blend α=0.5, max_steps=20, 200 cases/vocab)

| Vocab | Baseline steps | Gated steps | Reduction | Argmax match |
|-------|---------------|-------------|-----------|-------------|
| 8 | 4000 | 592 | **6.76×** | 100.0% ✓✓ |
| 32 | 4000 | 759 | **5.27×** | 100.0% ✓✓ |
| 128 | 4000 | 1026 | **3.90×** | 100.0% ✓✓ |
| 1024 | 4000 | 1493 | **2.68×** | 100.0% ✓✓ |

### Threshold sensitivity (vocab=32)

| Threshold | Gated steps | Reduction | Argmax% | G1 | G2 |
|-----------|------------|-----------|---------|-----|-----|
| 0.000 | 4000 | 1.00× | 100.0% | ✗ | ✓ |
| 0.005 | 888 | 4.50× | 100.0% | ✓ | ✓ |
| **0.010** | **759** | **5.27×** | **100.0%** | ✓ | ✓ |
| 0.050 | 583 | 6.86× | 100.0% | ✓ | ✓ |
| 0.100 | 486 | 8.23× | 100.0% | ✓ | ✓ |
| 0.500 | 400 | 10.00× | 100.0% | ✓ | ✓ |
| 1.000 | 345 | 11.59× | 100.0% | ✓ | ✓ |

**Default threshold = 0.01** (practical sweet spot — 5.27× reduction with zero quality loss).

### G3: Latency

| Vocab | Latency (µs) | G3 (<1µs) |
|-------|------------|-----------|
| 8 | 0.041 | ✓ PASS |
| 32 | 0.083 | ✓ PASS |
| 128 | 0.458 | ✓ PASS |
| **1024** | **3.958** | **✗ FAIL** |

**Root cause:** `self_advantage()` is an O(vocab) loop computing log-softmax subtraction. At vocab=1024, the loop takes ~4µs (~3.9ns/element). The 1µs target is achievable for vocab ≤256 but not for full LLM-scale vocabs (32000+).

**Mitigation path:** SIMD log-softmax (process 4 or 8 elements per cycle via NEON/AVX2) would bring vocab=1024 to ~1µs. For full LLM vocabs, a chunked or approximate variant is needed. This is a **future optimization**, not a blocker for the opt-in feature.

---

## Structural Note: EarlyStopGate Comparison

The plan (T4.1) called for A/B against `EarlyStopGate`. This is **structurally impossible** as a drop-in comparison:
- `EarlyStopGate<P>` is a `ScreeningPruner` consuming `(depth, token_idx, parent_tokens)` for tree-path expansion screening — it has **no logits access** and does not gate recursion loops.
- `AdvantageMarginGate` consumes `(pre_logits, post_logits, candidate)` and gates recursion-loop continuation.

They operate on **different gate points** (tree expansion vs loop continuation). The baseline used instead is **no-gate** (always run max_steps recursion), which is the correct apples-to-apples comparison for dead-compute detection.

---

## Decision

**`self_advantage_gate` stays opt-in** (not promoted to default).

**Rationale:**
1. G3 fails at vocab ≥1024 — the latency target (<1µs) is too aggressive for O(vocab) at LLM scale.
2. The feature is still **highly valuable as opt-in** — 2.68×–6.76× forward-pass reduction with 100% quality preservation is a strong result for latency-sensitive recursion loops.
3. The `product_policy_sharpen` primitive (Phase 3) ships alongside as opt-in — it's useful independently for controllable reasoning trust weight.
4. Future SIMD optimization could bring G3 within target, at which point re-evaluation for default-on is warranted.

**No demotion needed** — there is no incumbent default-on feature being replaced. `EarlyStopGate` operates at a different gate point (tree expansion, not loop continuation).

---

## Reproduction

```bash
cargo bench --bench self_advantage_gate_bench --features self_advantage_gate
```

---

## TL;DR

Self-Advantage Recursion Gate GOAT gate: **3/4 pass (G1 2.68–6.76× reduction, G2 100% quality, G4 robust), G3 fails at vocab=1024 (3.96µs > 1µs target)**. Feature stays **opt-in** (`self_advantage_gate` + `product_policy_sharpen`). Strong dead-compute detector for latency-sensitive recursion loops — 5.27× forward-pass reduction at default threshold 0.01 with zero quality loss. SIMD optimization could bring G3 within target for future re-evaluation.
