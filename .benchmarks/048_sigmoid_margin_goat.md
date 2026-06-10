# GOAT Proof 048: Sigmoid Margin Loss + Retrieval Margin Diagnostic (Plan 157)

> **Date:** 2026-05-29
> **Feature Gate:** `sigmoid_margin` (root) → `katgpt-core/sigmoid_margin` (requires `maxsim`)
> **Depends on:** Plan 080 (MaxSim late-interaction scoring), Plan 157 (`sigmoid_margin_loss`, `compute_retrieval_margin`, `dim_sufficiency_bound`)
> **Research:** 123 — "Is Dimensionality a Barrier for Retrieval Models?" (arXiv 2605.23556, Bangachev–Bresler–Kogan–Polyanskiy, MIT, May 2026)

## Summary

GOAT proof for the sigmoid-margin retrieval toolkit. Research 123 proves that near-optimal
retrieval margin is achievable in dimension **d = Θ(k · log n)** (k = query sparsity, n = corpus
size) — tight upper *and* lower bound — and that a **SigLIP-style sigmoid loss** reaches positive
margin at far lower dimension than InfoNCE (d ≈ 5–9, near-constant in n, vs InfoNCE d ≈ Θ(n^⅓)).

Core result: **7/7 GOAT proofs passing (12 unit tests).** `sigmoid_margin_loss` matches the manual
softplus reference, `compute_retrieval_margin` separates clean vs mixed embeddings, the
`dim_sufficiency_bound` is monotone and scales as k·log n, the loss gradient pushes embeddings
toward positive margin, the margin correlates with MaxSim score, and **MaxSim scoring shows no
regression** when the feature is enabled.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Loss | SigLIP-style `softplus(t · (score − b) · sign)` |
| Margin | `compute_retrieval_margin` (min positive − max negative dot product) |
| Dim bound | `dim_sufficiency_bound(k, n)` = ⌈C · k · log n⌉ |
| Build | Release (`--release`, core crate) |
| Platform | macOS (aarch64) |

## GOAT Proof Results

### G1: Loss Correctness (proof1 ×3)

**Claim:** `sigmoid_margin_loss` equals the manual softplus formula, including bias/temperature, and → 0 under perfect separation.

| Test | Result |
|------|--------|
| `proof1_loss_matches_manual` | ✅ |
| `proof1_loss_with_bias_and_temperature` | ✅ |
| `proof1_loss_perfect_separation` | ✅ (loss → 0) |

### G2: Retrieval Margin Sign (proof2 ×2)

**Claim:** Margin is positive for separated embeddings, negative when positives/negatives overlap.

| Test | Result |
|------|--------|
| `proof2_margin_positive_for_separated_embeddings` | ✅ |
| `proof2_margin_negative_for_mixed_embeddings` | ✅ |

### G3: Dimension Bound (proof3 ×3)

**Claim:** `dim_sufficiency_bound` is monotone in (k, n), handles edge cases, and scales as **O(k · log n)** — matching Theorem 1.4/1.5 (tight).

| Test | Result |
|------|--------|
| `proof3_bound_monotonic` | ✅ |
| `proof3_bound_edge_cases` | ✅ |
| `proof3_bound_scales_as_k_log_n` | ✅ |

> **Research anchor (k=2, raw `.raw/TopK/`):** sigmoid loss needs d ≈ 6→9 as n grows 20→240
> (O(log n) ✓); InfoNCE needs d ≈ 10→23 over the same range (Θ(n^⅓)). Max margin for k-sparse
> queries is Θ(1/√k).

### G4: Gradient Direction (proof4)

**Claim:** The loss gradient moves embeddings toward a positive margin.

| Test | Result |
|------|--------|
| `proof4_loss_gradient_pushes_to_positive_margin` | ✅ |

### G5: MaxSim Correlation + No Regression (proof5, proof6)

**Claim:** Retrieval margin correlates with MaxSim late-interaction score, and enabling
`sigmoid_margin` does not regress existing MaxSim scoring.

| Test | Result |
|------|--------|
| `proof5_margin_correlates_with_maxsim` | ✅ |
| `proof6_no_maxsim_regression` | ✅ |

### G6: Feature Isolation (proof7)

| Test | Result |
|------|--------|
| `proof7_feature_gate_functions_exist` | ✅ (functions compile + export under the gate) |

## GOAT Gate Summary

| # | Proof | Gate | Result |
|---|-------|------|--------|
| G1 | Loss correctness | matches softplus reference, →0 on separation | ✅ PASS |
| G2 | Margin sign | + separated / − mixed | ✅ PASS |
| G3 | Dim bound | monotone, O(k·log n) scaling | ✅ PASS |
| G4 | Gradient | pushes toward positive margin | ✅ PASS |
| G5 | MaxSim correlation | margin ↔ MaxSim, no regression | ✅ PASS |
| G6 | Feature isolation | gated functions exist | ✅ PASS |

**Overall: 7/7 gates PASS (12 unit tests).**

## Commands to Reproduce

```bash
# Run all sigmoid-margin proof tests
cargo test --release -p katgpt-core --features sigmoid_margin sigmoid_margin -- --nocapture

# Verify builds without feature
cargo check -p katgpt-core
```

## Key Findings

1. **Sigmoid ≫ InfoNCE for margin** — sigmoid loss reaches positive margin at d ≈ log n; InfoNCE
   needs polynomial dimension. This is the practical lever: small embedding dims suffice.
2. **Dimension bound is tight** — `dim_sufficiency_bound` encodes the Θ(k·log n) upper = lower
   bound, usable to size embedding dims for a target corpus.
3. **Composes with MaxSim** — the margin diagnostic correlates with MaxSim and leaves it
   unregressed, so it can run as an always-on quality signal.

## Feature Gate

```toml
# katgpt-core/Cargo.toml
sigmoid_margin = ["maxsim"]  # Sigmoid margin loss + retrieval margin diagnostic (Research 123, Plan 157)

# katgpt-rs/Cargo.toml
sigmoid_margin = ["katgpt-core/sigmoid_margin"]
```

**Status:** 7/7 GOAT passed — **default-on**.

## Files

| File | Role |
|------|------|
| `crates/katgpt-core/src/simd.rs` | `sigmoid_margin_loss`, `compute_retrieval_margin`, `dim_sufficiency_bound` + `sigmoid_margin_tests` |
| `crates/katgpt-core/src/lib.rs` | Re-exports under the `sigmoid_margin` gate |
| `.benchmarks/048_sigmoid_margin_goat.md` | NEW: this file |

## Related

- `.research/123_TopK_Dimensionality_Barrier_Retrieval.md`
- `.benchmarks/013_turboquant_vs_spectralquant_maxsim.md` (MaxSim baseline)
