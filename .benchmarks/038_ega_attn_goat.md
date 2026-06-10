# Benchmark 038: Energy-Gated Attention (EGA) Spectral Salience — GOAT Proofs

**Plan:** 139 — Energy-Gated Attention
**Feature Gate:** `ega_attn = []` (opt-in, NOT default-on)
**Date:** 2026-05-25

---

## Architecture

EGA gates value aggregation by the spectral energy of key token embeddings:

```
X (input embeddings)
 │
 ├──→ e = X · w_proj              [seq_len] energy scores
 │    └──→ ẽ = z_normalize(e)     z-normalized energy
 │         └──→ g = σ(α · (ẽ - τ))  sigmoid gate vector
 │
 A (softmax attention weights) [seq_len × seq_len]
 │
 └──→ Âᵢⱼ = Aᵢⱼ · gⱼ             gate each key position
      └──→ Âᵢⱼ /= Σₖ(Âᵢₖ + ε)    renormalize (sum-to-one)
           └──→ Y = Â · V          value aggregation
```

### Key Types

| Type | Purpose |
|------|---------|
| `EgaGate` | Per-head EGA parameters: w_proj (d), alpha (1), tau (1) |
| `sigmoid(x)` | Standard sigmoid function |
| `z_normalize(scores)` | In-place z-normalization |
| `compute_energy_gate(energy, α, τ)` | Full gate computation from energy scores |

### Parameters per Head

| Parameter | Size | Default | Role |
|-----------|------|---------|------|
| `w_proj` | d | 1/d | Energy projection vector |
| `alpha` | 1 | 2.2 | Gate sharpness (paper converged) |
| `tau` | 1 | 0.35 | Energy threshold (paper converged) |
| **Total** | **d + 2** | | |

---

## GOAT Proofs (6/6 ✅)

Test file: `tests/test_139_ega_attn.rs`

| # | Test | Assertion | Status |
|---|------|-----------|--------|
| P1 | `proof_ega_energy_finite` | Energy scores are all finite for random input | ✅ |
| P2 | `proof_ega_gate_sums_to_one` | Gated attention weights sum to 1.0 per row | ✅ |
| P3 | `proof_ega_low_energy_suppressed` | Low-energy positions receive less weight than high-energy | ✅ |
| P4 | `proof_ega_high_energy_preserved` | Uniform energy → uniform attention (no distortion) | ✅ |
| P5 | `proof_ega_parameter_count` | EgaGate has exactly head_dim + 2 parameters | ✅ |
| P6 | `proof_ega_zero_wproj_uniform` | Zero w_proj produces uniform gate → no positional bias | ✅ |

---

## Throughput

| Operation | Scale | Time | Notes |
|-----------|-------|------|-------|
| Energy scores | seq_len=64, dim=128 | <1μs | O(seq_len × dim) dot products |
| Z-normalize | seq_len=64 | <1μs | Single pass |
| Gate computation | seq_len=64 | <1μs | sigmoid per position |
| Gate attention (in-place) | 64×64 | <10μs | Gate + renormalize |
| Full pipeline | seq_len=64, dim=128 | <15μs | Energy → gate → renormalize |

---

## Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `w_proj` init | 1/d | — | Uniform energy prior |
| `alpha` | 2.2 | 0.1–10.0 | Gate sharpness; higher → sharper transition |
| `tau` | 0.35 | -3.0–3.0 | Energy threshold; above → preserved, below → suppressed |

---

## Module Structure

```
src/ega_attn.rs              # ~220 lines — EgaGate + helpers + unit tests
tests/test_139_ega_attn.rs   # ~200 lines — 6 GOAT proofs
```

---

## Feature Gate

```toml
[features]
ega_attn = []  # Energy-Gated Attention (Plan 139, opt-in)
```

No dependencies. Pure Rust.

---

## Files Modified

| File | Change |
|------|--------|
| `Cargo.toml` | Added `ega_attn = []` feature |
| `src/lib.rs` | Added `#[cfg(feature = "ega_attn")] pub mod ega_attn;` |
| `src/ega_attn.rs` | **NEW** — Core EGA types and helpers |
| `tests/test_139_ega_attn.rs` | **NEW** — 6 GOAT proofs |
