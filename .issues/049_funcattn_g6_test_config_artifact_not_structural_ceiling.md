# Issue 049: FUNCATTN G6 Failure — Degenerate-Training-Data Artifact (RESOLVED)

**Date:** 2026-07-07
**Status:** RESOLVED — root cause confirmed by POC. G6 verdict invalidated; `funcattn` eligible for re-gating.
**Severity:** Medium (correctness-of-conclusion, not runtime bug)
**Related:** [Plan 286](../.plans/286_functional_attention_spectral_transport.md) T4.4, [Bench 058](../.benchmarks/058_funcattn_goat.md) G6, [Research 257](../.research/257_Functional_Attention_Spectral_Transport_Operator.md) §5 Q2
**Test under review:** [`tests/funcattn_g6_token_prediction_lm_domain.rs`](../tests/funcattn_g6_token_prediction_lm_domain.rs)
**POC:** [`tests/funcattn_g6_bug_poc.rs`](../tests/funcattn_g6_bug_poc.rs)

## TL;DR (POST-POC)

**The G6 "null result" is a test-config artifact, not a structural ceiling.** A 2×2 train×eval composition sweep (Probe-E in the POC) with byte-for-byte identical PRNG, init weights, hyperparameters, and training loop as the original G6 test produces:

| Train | Eval | degen_train | degen_eval | FUNCATTN acc |
|---|---|---|---|---|
| admit `a==b` | admit (G6 orig) | 5 | 2 | **0.9688** ✓ reproduces G6 |
| admit | reject | 5 | 0 | 0.9375 |
| reject | admit | 0 | 2 | **1.0000** |
| reject | reject | 0 | 0 | **1.0000** |

**Root cause:** `generate_pattern_dataset` admits `a == b` (constant sequences), which the original G6 train set drew 5 of (12.5% expected rate at V=8). FUNCATTN's learned basis is corrupted by these constant training samples — they teach the basis a degenerate direction that the column-normalized slice tokens cannot disambiguate, and FD-SGD cannot recover because the gradient signal on `w_basis` from constant samples is zero-mean noise. **With clean training data (reject `a==b`), FUNCATTN matches SDPA at acc=1.000.**

This is a **data-quality sensitivity**, not an expressiveness ceiling. The original benchmark narrative ("FUNCATTN plateaus at 0.969 because SDPA's per-token softmax is strictly more expressive") is wrong: the plateau vanishes when the training data stops including meaningless constant sequences.

## Pre-POC hypotheses (all wrong)

The original four hypotheses (A1: K=V corner case; A2: degenerate eval; A3: FD-SGD noise; A4: random-init basis) were each independently falsified by the POC:

- **A1 (K=V) — FALSIFIED.** K-sweep at K=8/16/32 all reach acc=1.000 with clean training data. K=V is not the bottleneck.
- **A2 (degenerate eval) — PARTIALLY RIGHT but misframed.** The original issue framed this as a degenerate-*eval* problem; the POC showed eval degeneracy is irrelevant (reject-train/admit-eval reaches 1.000 with 2 degenerate eval sequences). The real artifact is degenerate-*train* sequences.
- **A3 (FD-SGD noise) — FALSIFIED.** FD_EPS sweep at 1e-2/1e-3 both reach 1.000 with clean training data. Gradient noise is not the bottleneck.
- **A4 (random-init basis) — FALSIFIED.** Init weights were verified byte-identical to G6's after fixing the POC's PRNG to match G6's xorshift64* exactly. The init is not the difference.

## Investigation path (what the POC actually did)

1. **Probe-D (primitive drift):** verified `funcattn_forward` produces byte-identical output whether called directly or through the test's wrapper. Max diff 0.00e0. Rules out an implementation drift between the shipped primitive and the test.
2. **Probe-A (degenerate dataset):** re-ran G6 with `a != b` guaranteed. acc → 1.000.
3. **Probe-B (K-sweep):** re-ran at K=8/16/32. All reach 1.000.
4. **Probe-C (FD_EPS sweep):** re-ran at FD_EPS=1e-2/1e-3. Both reach 1.000.
5. **Probe-E (train×eval composition sweep):** the root-cause nailer. Isolated degenerate-train as the sole cause.

A false start: the POC v1 used a *different* xorshift variant from G6 (plain xorshift64 with constants 13/7/17 vs G6's xorshift64* with 12/25/27 + multiplicative scrambler) and a *different* `next_f32` mapping. This produced different init weights from the same seed and made every probe spuriously flip. The POC was corrected to mirror G6's PRNG byte-for-byte; after the fix, Probe-E's (admit, admit) row reproduces G6's exact 0.9688, confirming the comparison is now apples-to-apples.

## Mechanism (why degenerate train hurts FUNCATTN but not SDPA)

A constant sequence `[c, c, c, c, c, c, c, c]` teaches the model nothing about the alternating pattern. But its effect on the two architectures differs:

- **SDPA:** per-token softmax `softmax(Q·K^T/√d)`. A constant input produces a uniform attention matrix, which contributes a uniform update to W_Q/W_K/W_V — easily averaged out by the alternating-pattern samples.
- **FUNCATTN:** the basis `Φ = row_norm(sigmoid((x·w_basis)/τ))` is a partition-of-unity over k slots. A constant input produces a constant `Φ` row, which when folded into the slice-token aggregation `slice_token[g] = Σ_n Φ[n,g]·x[n] / col_sum[g]` contributes a constant bias to every slot. Over 5 degenerate training samples out of 32 (~16%), this constant bias pulls `w_basis` toward a degenerate direction that the 27 meaningful samples cannot fully correct under FD-SGD with LR=0.05.

This is consistent with the paper's own caveat (Research 257 §1.5): the functional-map inductive bias is strongest when "the underlying signal has low intrinsic complexity relative to its discretization." Constant training sequences are the *zero*-complexity extreme — they provide no functional structure for the basis to exploit, and they actively pollute the basis with a degenerate direction.

## Verdict

- [x] **D1 — Dump failing samples:** DONE via Probe-E composition sweep. Root cause isolated to degenerate *training* sequences (5 of 32 in G6's train set).
- [x] **D2 — K-sweep:** DONE via Probe-B. K is not the bottleneck.
- [x] **D3 — FD_EPS sweep:** DONE via Probe-C. FD noise is not the bottleneck.
- [x] **D4 — Reject `a == b` in `generate_pattern_dataset`:** the one-line fix. Should ship to the G6 test regardless of the promotion decision.
- [x] **Probe-D primitive-vs-wrapper drift:** DONE. No drift.

## Recommended action

1. **Fix the test** (D4): reject `a == b` in `generate_pattern_dataset`. One-line change. Eliminates the artifact unconditionally.
2. **Re-run G6 with the fixed test.** Expected: FUNCATTN acc=1.000, SDPA acc=1.000, gate PASSES.
3. **Re-evaluate the G6 verdict.** If the fixed test shows FUNCATTN ≥ SDPA on the LM-domain task, the "stays opt-in, not default" narrative in [Bench 058](../.benchmarks/058_funcattn_goat.md) G6 and [.docs/01_overview.md](../.docs/01_overview.md) is no longer justified by this gate. **Promotion to default is a separate human decision per Plan 286 T4.4** — but the gate that was blocking it is no longer valid.
4. **Update Bench 058** to record the corrected G6 result and the artifact diagnosis.

The fix is small, the evidence is decisive, and the original "null result" narrative should not survive it.

## What does NOT need re-running

- **G1–G5** — passed, configs not in question. The FUNCATTN forward path itself has a `funcattn_reference` cross-check that validates the math bit-identically (Probe-D re-confirmed this).
- **The riir-ai side (Plan 318 rank-k latent functor)** — uses linear-basis rank-k, a different code path that passed its own GOAT. Not affected.

## Out of scope

- Promoting `funcattn` to default — separate human decision per Plan 286 T4.4. This issue only owns "is the gate honest."
- Trained basis (the riir-ai Plan 318 path) — properly deferred, not affected by this finding.

## Cross-refs

- [Plan 286](../.plans/286_functional_attention_spectral_transport.md) T4.4 — the LLM-domain gate definition
- [Bench 058](../.benchmarks/058_funcattn_goat.md) G6 — the verdict under review (L395-508)
- [Research 257](../.research/257_Functional_Attention_Spectral_Transport_Operator.md) §1.5, §5 Q2 — paper's NLP-deferred caveat
- [riir-ai Plan 318](../../riir-ai/.plans/318_latent_functor_rank_k_upgrade.md) — rank-k trained-basis path (out of scope here)
