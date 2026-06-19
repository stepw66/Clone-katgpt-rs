# Issue 033: FUNCATTN T5.1 eigenbasis composition — no benefit on anisotropic regression

**Date:** 2026-06-19
**Plan:** [286_functional_attention_spectral_transport.md](../.plans/286_functional_attention_spectral_transport.md) Phase 5 T5.1
**Status:** Documented null result. Primitive + composition ship as opt-in; no promotion.

---

## Summary

The Plan 286 T5.1 hypothesis was that pre-rotating FUNCATTN's basis weights
`w_basis` onto a SpectralQuant-calibrated eigenbasis would make FUNCATTN
"more expressive per parameter" on anisotropic input distributions, because
the `k` basis rows would align with the `k` highest-variance directions of
the input stream.

**The hypothesis is FALSE for FUNCATTN's adaptive-basis architecture.** On a
synthetic anisotropic regression task (target depends only on PC1, the top
eigen-direction), eigen-aligned FUNCATTN is **17–25% worse** than vanilla
FUNCATTN with random orthogonal init, at matched parameter budget:

| Variant        | Init MSE | Final MSE (80 steps) | Final MSE (200 steps) |
|----------------|----------|----------------------|-----------------------|
| Vanilla        | 0.678    | 0.113                | 0.110                 |
| Eigen-aligned  | 0.642    | 0.132                | 0.138                 |
| Ratio          | —        | 1.17                 | 1.25                  |
| Trivial (var)  | —        | 0.338                | 0.338                 |

Both variants learn well below the trivial-predictor baseline, so this is not
a correctness issue — it's a sample-efficiency / optimization-trajectory issue.

---

## Root cause

FUNCATTN's adaptive basis computes scores `s = x · w_basis^T / τ`, then applies
`sigmoid(s)` (or softmax) and **row-normalizes** to produce the partition-of-
unity `Φ[n, :]`. The row-normalization is the key: it makes `Φ` invariant to
**additive shifts** of the score vector, and the activation makes it invariant
to **monotone reordering**.

An orthogonal rotation of the `w_basis` rows is a *lossless* transformation
of the function class the basis can represent (the rotation is invertible).
But it is **not** an information-concentrating transformation, because:

1. The rotation `V` mixes the score components; it doesn't reduce their number.
2. Row-normalization strips the magnitude information that would otherwise
   let the high-eigenvalue directions dominate the partition.
3. The FD-SGD trajectory on the random orthogonal init happens to find a
   better local minimum on this particular dataset/seed.

This is consistent with the theoretical guarantee: orthogonal rotation
preserves expressivity but does not improve it. The "more expressive per
parameter" intuition borrowed from SpectralQuant (where eigenbasis alignment
*does* help because the downstream quantizer is *not* invariant to rotation)
does not transfer to FUNCATTN's row-normalized adaptive basis.

---

## What shipped anyway

The primitive and composition are correct and potentially useful for other
reasons, so they ship as opt-in:

- **`katgpt-core::funcattn::pre_rotate_basis_weights_into`** — the lossless
  rotation primitive. 4 unit tests verify identity-noop, row-norm preservation,
  orthogonality preservation, and partition-of-unity after rotation.
- **`src/funcattn_compose/spectral_pre_rotate.rs`** — composition glue:
  `calibrate_and_pre_rotate_basis` (one-call SpectralQuant + rotation) and
  `effective_basis_rank` (diagnostic that recommends a smaller `k` based on
  the cumulative-variance thresholds).
- **Cargo feature `funcattn_spectral_pre_rotate`** — opt-in, NOT in `default`
  and NOT in `full`.

The `effective_basis_rank` diagnostic is the residual value of the composition:
it tells a caller whether their `k` is over-provisioned relative to the input
spectrum, which is a useful compression-time hint even if pre-rotation itself
doesn't help the forward pass.

---

## G6 test

`tests/funcattn_t5_1_eigenbasis_compose.rs` — runs vanilla vs eigen-aligned
FUNCATTN on the anisotropic regression task, asserts only mechanics (both
variants learn below trivial baseline; rotation is deterministic). The
PASS/TIE/FAIL verdict on the MSE ratio is reported via eprintln, not asserted
(research question, not correctness question — matches the G2 pattern).

Run:
```bash
cargo test --features funcattn,spectral_quant --release \
  --test funcattn_t5_1_eigenbasis_compose -- --nocapture
```

---

## Follow-ups (optional, not blocking)

1. **Try a non-row-normalized basis variant.** If FUNCATTN were modified to
   skip the row-normalization (using raw sigmoid outputs weighted by an
   inverse-temperature), the eigenbasis alignment might concentrate information
   the way SpectralQuant does. This would be a new primitive variant, not a
   composition — out of scope for Plan 286.
2. **Try a different task.** The regression target here depends only on PC1.
   A target that depends on the *full* spectrum (e.g., a function of all 8
   principal components with different weights) might show a different
   pattern. Low priority — the theoretical argument above suggests the
   result would be similar.
3. **Try learned (not FD-SGD) training.** AdamW with proper learning-rate
   scheduling might erase the trajectory difference. Out of scope — Plan 286
   uses FD-SGD deliberately (no autodiff dep, matches G3/G2 convention).

## TL;DR

Eigenbasis pre-rotation is lossless but doesn't help FUNCATTN's row-normalized
adaptive basis. Ships as opt-in for callers who want eigen-aligned scores for
downstream reasons; documented null result here. The `effective_basis_rank`
diagnostic is the residual value.
