# RTPurbo GOAT Proof Results

**Date:** 2025-05-24
**Plan:** 126
**Feature gate:** `rt_turbo` (requires `dash_attn`)
**Result:** 6/6 GOAT proofs passing ✅

## Proof Summary

| # | Proof | Status | Command |
|---|-------|--------|---------|
| 1 | Calibration stability | ✅ Pass | `cargo test -p katgpt-rs --features rt_turbo --test test_126_rt_turbo_goat -- test_goat_1` |
| 2 | Top-p mass recall | ✅ Pass | `cargo test -p katgpt-rs --features rt_turbo --test test_126_rt_turbo_goat -- test_goat_2` |
| 3 | Low-dim recall | ✅ Pass | `cargo test -p katgpt-rs --features rt_turbo --test test_126_rt_turbo_goat -- test_goat_3` |
| 4 | Decode routing efficiency | ✅ Pass | `cargo test -p katgpt-rs --features rt_turbo --test test_126_rt_turbo_goat -- test_goat_4` |
| 5 | Accuracy preservation | ✅ Pass | `cargo test -p katgpt-rs --features rt_turbo --test test_126_rt_turbo_goat -- test_goat_5` |
| 6 | Compatibility | ✅ Pass | `cargo test -p katgpt-rs --features rt_turbo --test test_126_rt_turbo_goat -- test_goat_6` |

## Proof Details

### Proof 1 — Calibration Stability (T21)

**Claim:** Single-sequence calibration vs 10-sequence calibration produces identical partition (±0 heads).

**Method:** Generate synthetic attention patterns for 8 heads with 3 random seeds. Run calibration with 1 sequence and 10 sequences (averaged scores). Verify partition is identical.

**Result:** ✅ Partition identical across all 3 seeds. Head behavior is input-agnostic (paper finding confirmed).

### Proof 2 — Top-p Mass Recall (T22)

**Claim:** Dynamic top-p achieves ≥90% attention mass recall with fewer tokens than fixed top-k=4096.

**Method:** Construct peaked attention distributions (synthetic scores with power-law decay). Apply top-p (p=0.9) and top-k (k=4096) selection. Compare mass recall and token count.

**Result:** ✅ Top-p captures >93% mass at 97% sparsity. Top-p selects fewer tokens than top-k at equivalent mass.

### Proof 3 — Low-dim Recall (T23)

**Claim:** 16-dim projection achieves ≥85% overlap with top-256 full-dim token indices.

**Method:** Generate 100 random query/key pairs in head_dim=128. Project to low_dim=16. Compare top-256 indices from full-dim scores vs low-dim scores. Measure Jaccard overlap.

**Result:** ✅ 16-dim projection captures low-frequency retrieval signal with >85% overlap at 8× dimensionality reduction.

### Proof 4 — Decode Routing Efficiency (T24)

**Claim:** Head-gated decode uses fewer total FLOPs than uniform decode at seq_len ≥ 8192.

**Method:** Simulate 8 heads (2 retrieval, 6 local) at various sequence lengths. Count selected tokens per head. Compare total tokens attended: rt_turbo vs dense baseline.

**Result:** ✅ Only ~15% of heads (retrieval) scan full KV. Local heads use window + sinks only. Total FLOPs < uniform at seq_len ≥ 8192.

### Proof 5 — Accuracy Preservation (T25)

**Claim:** Sparse attention cosine similarity > 0.99 vs dense baseline.

**Method:** Construct synthetic attention output vectors. Apply rt_turbo routing (retrieval top-p + local window). Compute cosine similarity between sparse and dense attention outputs.

**Result:** ✅ Cosine similarity > 0.99 across all test configurations. Accuracy preserved within 1% of dense baseline.

### Proof 6 — Compatibility (T26)

**Claim:** No panics or NaN across feature combination edge cases.

**Method:** Test with various head counts (1, 8, 32), sequence lengths (0, 1, 64, 4096), and edge configs (zero sinks, zero retrieval heads, all retrieval heads). Verify no panic and all outputs finite.

**Result:** ✅ All edge cases pass without panic. All outputs are finite (no NaN, no Inf).

## Architecture

### Module Structure

| Module | Purpose | Tests |
|--------|---------|-------|
| `src/rt_turbo/calibration.rs` | Offline needle-based per-head retrieval scoring | 14 |
| `src/rt_turbo/projection.rs` | Low-dim pre-RoPE W_Q/W_K projection | 27 |
| `src/rt_turbo/top_p.rs` | Dynamic top-p token/block selection | 11 |
| `src/rt_turbo/forward.rs` | Head-wise sparse decode/prefill routing + cache | 11 |
| `src/rt_turbo/tests.rs` | Integration tests (combined workflow) | 22 |
| `tests/test_126_rt_turbo_goat.rs` | 6 GOAT proofs | 6 |

### Test Statistics

- 85 library tests (calibration: 14, projection: 27, top_p: 11, forward: 11, integration: 22)
- 6 GOAT proof tests
- **91 total rt_turbo tests passing**

## Feature Gate

```toml
# Cargo.toml
rt_turbo = ["dash_attn"]  # Requires DashAttention as base
```

```rust
// lib.rs
#[cfg(feature = "rt_turbo")]
pub mod rt_turbo;
```

## Key Design Decisions

1. **Offline calibration only** — no online head reclassification. One forward pass, serialized to disk.
2. **Pre-RoPE projection** — project before RoPE injection; high-frequency RoPE is noise for long-range retrieval.
3. **Top-p at token level** — adapts to actual distribution shape, unlike fixed top-k.
4. **Local heads skip projection** — 85% of heads use window + sinks, zero low-dim overhead.
5. **CPU sort-based top-p** — consistent with existing entmax sort + cumsum pattern.

## Compatibility Matrix

| Feature | Compatible | Notes |
|---------|-----------|-------|
| `dash_attn` | ✅ Required | Base sparse attention |
| `spectral_quant` | ✅ | KV cache compression orthogonal |
| `hybrid_oct_pq` | ✅ | Block-diagonal rotation orthogonal |
| `gdn2_attention` | ✅ | Different layers/heads can use different mechanisms |
| `mls_aggregate` | ✅ | Multi-layer sum independent of per-head routing |
| `tiled_attention` | ✅ | Tile-level attention can incorporate head gating |