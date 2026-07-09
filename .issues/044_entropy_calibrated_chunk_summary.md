# Issue 044: Entropy-Calibrated Chunk Summary (HiLS Prop 3.1)

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/399_Hierarchical_Landmark_Sparse_Attention.md](../.research/399_Hierarchical_Landmark_Sparse_Attention.md)
**Source:** [arXiv:2607.02980](https://arxiv.org/abs/2607.02980) — HiLS-Attention (Hu et al., Tencent Hunyuan)
**Target:** `katgpt-rs/crates/katgpt-attn/src/dash_attn/chunk_summary.rs` + `routing.rs`
**Status:** Open — optimization of existing opt-in primitive

---

## Problem

Our shipped `summarize_chunk_into` computes `k̄_c = softmax(q̄·K/√d)·K` (HiLS
Eq 8) but **does not** compute the entropy bias `b'_c = -Σ p_j log p_j` that
HiLS's Proposition 3.1 proves is necessary for the chunk score to be a faithful
first-order Taylor approximation of the LogSumExp chunk mass.

Without `b'_c`, the routing score `dot(query, k̄_c)` captures only the
"mean-logit" regime. The entropy term interpolates adaptively: `→ log S` when
logits are uniform (many mildly-relevant tokens), `→ 0` when one logit
dominates (concentrated retrieval). This is the exact gap the paper identifies
between mean-pooling (NSA/MoBA/InfLLM v2) and faithful chunk mass estimation.

## Constraint

**Backward-compatible / dormant at zero-init.** When `head_cls` is zero
(default), `softmax(0)` is uniform → `p_j = 1/S` → `b'_c = log S` (constant
across all chunks) → no ranking change. The entropy bias only changes rankings
when `head_cls` is non-trivial (trained or deterministically seeded). So this
is a **zero-regression** change at the current default and activates the moment
riir-train provides learned landmark queries.

## Tasks

- [ ] **T1** Add `entropy_bias` computation to `summarize_chunk_into`: after
      `softmax_inplace(&mut scores_buf[..chunk_size])`, compute
      `b'_c = -Σ p_t log p_t` as one reduction over the already-L1-resident
      softmax weights. Zero allocation.
- [ ] **T2** Decide API shape: either (a) extend `summarize_chunk_into` to also
      write `entropy_bias` into an out-param, or (b) add a new
      `summarize_chunk_with_entropy` that returns `(summary_key, entropy_bias)`.
      Prefer (a) on the hot path (avoids a second pass); (b) for ergonomics on
      cold paths.
- [ ] **T3** Update `routing.rs` chunk scorer to use `ŝ_{i,c} = q^T k'_c / √d + b'_c`
      when the entropy is available. Gate behind the existing `dash_attn` feature.
- [ ] **T4** Unit test: zero-init `head_cls` → `b'_c == log(S)` exactly (uniform
      distribution) → routing ranking unchanged (bit-identical to current behavior).
- [ ] **T5** Unit test: non-trivial `head_cls` (e.g. the existing
      `test_summarize_chunk_with_learned_query` setup with `[0, 100, 0, 0]`) →
      `b'_c ≈ 0` (near-degenerate distribution) → low-entropy chunk gets a
      smaller bias than a uniform-logit chunk.
- [ ] **T6** Sanity: confirm `dash_attn` still compiles + existing tests pass
      with no behavior change at zero-init.

## Not in scope

- **No GOAT gate now.** The gain is dormant at zero-init; a meaningful gate
  requires riir-train-provided learned `head_cls`. When those land, re-gate
  with before/after on chunk-selection accuracy (NIAH-style) at fixed budget.
- **No hierarchical softmax factorization.** That is pure algebra whose purpose
  is gradient flow for training; dormant for modelless inference.
- **No landmark tokens / Q-Cal / HoPE.** All training recipe → riir-train.

## Notes

- The entropy `b'_c` is computed from the **same softmax weights** already
  produced for `k̄_c` — it is a free byproduct, not a second attention pass.
- Paper Tab 6 ablation "w/o Prop 3.1" (use raw landmark key without Taylor
  rectification) shows the entropy term contributes real PPL/extrapolation
  gains *in the trained setting*. The modelless benefit at zero-init is nil
  (constant); the benefit appears only with non-trivial queries.
