# Plan 415: Within-Class Effective Rank — Class-Conditioned Collapse Diagnostic

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/394_GNN_Survey_Within_Class_Effective_Rank_Fusion.md](../.research/394_GNN_Survey_Within_Class_Effective_Rank_Fusion.md)
**Source paper:** [arXiv:2412.19419](https://arxiv.org/abs/2412.19419) §5.3.1 + Supplementary S.1.2 — Tanis et al., MITRE/NCI GNN survey
**Target:** `crates/katgpt-core/src/data_probe/geometry.rs` (extend) — inherits the existing `sink_aware_attn` feature gate (same as the sibling `effective_rank`)
**Status:** Active — Phase 1 COMPLETE (all gates PASS)

## Goal

Ship `within_class_effective_rank(states, dim, class_labels) -> f32` — the entropy-based effective rank of the **within-class residual** covariance matrix (paper eq. S.1.2). This is the GOAT-distilled primitive from Research 394: a **fusion of two shipped halves** that have never been combined:

1. `effective_rank` (`data_probe/geometry.rs`, Roy & Vetterli 2007) — entropy-based effective rank, but **class-agnostic** (centers by global mean).
2. `within_class_adjacency` / `between_class_adjacency` (`riir-engine/src/latent_functor/quality_gate.rs`, Plan 303 T5.1) — the class-conditioning machinery, currently used only for Dirichlet-energy separation scoring.

The paper (§5.3.1) claims this specific combination — effective rank applied to the within-class residual covariance — is novel. Modelless, ~40 lines, zero new dependencies (reuses the private `jacobi_eigenvalues` eigensolver).

**Why it matters:** the shipped class-agnostic `effective_rank` cannot distinguish "between-class variance dominates, within-class collapsed" (a failure mode for committed-personality / HLA populations) from "all variance is healthy and isotropic". The two cases produce the same global rank but different within-class ranks. The existing Dirichlet-energy quality gate in `latent_functor` measures *separation* (between > within) but not *within-class subspace health*. This primitive fills the gap.

**Feature flag:** inherits `sink_aware_attn` (the gate for all of `geometry.rs`). Stays opt-in — same status as the sibling `effective_rank`. No Cargo.toml change.

## Phase 1 — Primitive (CORE)

### Tasks

- [x] **T1.1** Add `within_class_effective_rank(states: &[f32], dim: usize, class_labels: &[usize]) -> f32` to `data_probe/geometry.rs`. Reuse `jacobi_eigenvalues`; replace global-mean centering with class-mean centering. Compute pooled within-class covariance `Σ_w = (1/Σ(n_c−1)) Σ_c Σ_{i∈S_c} (x_i − μ_c)(x_i − μ_c)^T`, then `r_WC = exp(−Σ p_i log p_i)` over normalized eigenvalues.
- [x] **T1.2** Add `within_class_effective_rank_owned` convenience wrapper taking `&[Vec<f32>]` + `&[usize]` (mirrors `effective_rank`'s signature for ergonomic callers).
- [x] **T1.3** Add `WithinClassGeometryReport { within_class_erank, global_erank_for_contrast, n_classes, n_states, dim }` + `within_class_geometry_report` convenience function (mirrors `representation_geometry_report`).
- [x] **T1.4** Unit tests in `geometry.rs::tests`:
  - (a) identical vectors within a class → within-class rank ≈ 0;
  - (b) two well-separated isotropic classes → within-class rank ≈ dim;
  - (c) two collapsed classes (each class rank-1) → within-class rank ≈ 1;
  - (d) degenerate single-class (all labels identical) → matches shipped `effective_rank`;
  - (e) empty / single-state guards return 0.

## Phase 2 — GOAT gate

- [x] **T2.1 G1 (correctness):** synthetic two-class case; verify `r_WC ∈ [1, min(d, n−C)]` and monotonicity in within-class variance (as within-class noise shrinks, `r_WC` decreases). ✅ `p415_isotropic_within_class_is_high`, `p415_rank1_within_class_collapses_to_one`, `p415_monotone_in_within_class_variance`.
- [x] **T2.2 G2 (non-redundancy vs shipped `effective_rank`):** construct a case where global `effective_rank` is high but `within_class_effective_rank` is low (between-class variance dominates, within-class collapsed) — prove the two metrics disagree. This is the load-bearing gate: it proves the primitive adds information the shipped class-agnostic metric lacks. ✅ `p415_g2_nonredundancy_vs_global` — 4 orthogonal class centroids, each class collapsed to a single point; within ≈ 0, global ≈ 3, disagreement > 1.5. **Key insight discovered during test construction:** effective rank is scale-invariant — tiny-but-isotropic within-class variance still gives high rank; the low-rank signal requires rank-deficient within-class structure, not just small-magnitude variance.
- [x] **T2.3 G3 (no-regression):** `cargo test -p katgpt-core --features sink_aware_attn --lib` passes (1385 passed, 0 failed); `cargo check -p katgpt-core --all-features` clean.
- [x] **T2.4 G4 (latency):** `criterion` micro-bench at dim=64, n=256, C=4: within-class = 232.2 µs/call, global = 478.6 µs/call (ratio 0.485x — within-class is FASTER than global, both dominated by the O(dim³) Jacobi eigensolver; the class-mean O(n·d) pass is negligible). ✅ bench_415_within_class_erank_goat.

## Notes

- **Not UQ-bearing:** this measures representation geometry (effective rank), not a probability distribution / coverage / quantile. The "Report the Floor" conformal-naive baseline (Plan 340) does NOT apply. G1–G4 above are sufficient.
- **Not Super-GOAT** (per Research 394 §3 novelty gate): Q2 fails — it is a better diagnostic for an existing capability class (representation collapse), not a new class of behavior. No private guide created.
- **No promotion target:** `geometry.rs` is opt-in (`sink_aware_attn`) and stays opt-in pending the parent Plan 287 G2/G3 gate. This primitive ships alongside its siblings under the same gate.
