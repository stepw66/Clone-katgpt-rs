# Issue 049: FUNCATTN G6 Failure — Test-Config Artifact, Not Proven Structural Ceiling

**Date:** 2026-07-07
**Status:** OPEN — POC ran 2026-07-07, all probes reported acc=1.0000 BUT result is INVALID (seed_offset=1 confound, see "POC run results" section below). G6 verdict stands. A1/A2/A3 remain untested pending a faithful-reproduction POC.
**Severity:** Medium (correctness-of-conclusion, not runtime bug)
**Related:** [Plan 286](../.plans/286_functional_attention_spectral_transport.md) T4.4, [Bench 058](../.benchmarks/058_funcattn_goat.md) G6, [Research 257](../.research/257_Functional_Attention_Spectral_Transport_Operator.md) §5 Q2
**Test under review:** [`tests/funcattn_g6_token_prediction_lm_domain.rs`](../tests/funcattn_g6_token_prediction_lm_domain.rs)

## TL;DR

G6's verdict (FUNCATTN acc=0.969 < SDPA acc=1.000, "stays opt-in, not default") was recorded as an "honest null result" matching the paper's NLP-deferred prediction. **That conclusion is not justified by the test as written.** The test has at least four independent config artifacts that each *independently* could produce the 3/128 miss — none of which have been ruled out. Before treating G6 as a real ceiling, the cheap diagnostics below must run. If any single one flips the verdict to PASS, the "null result" narrative is wrong and `funcattn` is **prematurely demoted**.

This is not a defense of FUNCATTN — it's a defense of the GOAT gate's integrity. A gate that fails for the wrong reason is worse than no gate.

## Why this might be a test-config artifact, not a real ceiling

Reading [`tests/funcattn_g6_token_prediction_lm_domain.rs`](../tests/funcattn_g6_token_prediction_lm_domain.rs) line-by-line surfaces four concerns:

### A1. K = V = 8 — basis at vocab size, zero spare capacity (L102-108)

```rust
const V: usize = 8;  // vocab
const D: usize = 8;  // model dim
const K: usize = 8;  // FUNCATTN basis dim
```

With `K = V`, the basis has **exactly** enough capacity to represent the vocab partitions and nothing more. Any near-degenerate `Φ` row becomes a hard failure. The paper's reference config is `K=64` (Research 257 §2.3 footnote) — three orders more capacity than this test gives. The benchmark doc itself lists `K=16`/`K=32` as untested in §"What would need to change" #1, but never ran it.

**This is the single most suspicious config choice.** A capacity ceiling at exactly `K=V` is not evidence of an algorithmic ceiling at realistic `K`.

### A2. Dataset admits `a == b` degenerate sequences (L230-244)

```rust
let a = (rng.next_u64() as usize) % effective_vocab.max(1);
let b = (rng.next_u64() as usize) % effective_vocab.max(1);
let seq: Vec<usize> = (0..seq_len).map(|i| if i % 2 == 0 { a } else { b }).collect();
```

Nothing rejects `a == b`. When `a == b`, the sequence is constant `[a,a,a,a,...]` — there is no "alternating pattern" to learn, and the masked position has no context-derived answer. With `V=8`, `P(a==b) = 1/8 = 12.5%` per sequence. Across 16 eval sequences, the expected number of degenerate eval sequences is **2.0**, and across `16 × 8 = 128` eval samples the expected number of degenerate samples is **16**.

The 3 misclassified samples (3/128 = 2.3%) are *smaller* than the expected degenerate-sample count — meaning **the test may be measuring how each architecture handles ambiguous targets, not how they handle the alternating pattern**. SDPA's per-token softmax happens to win this tiebreak; that is not a structural statement about FUNCATTN.

### A3. FD-SGD with FD_EPS=1e-2 on a 456-param model (L132, L488-504)

```rust
const FD_EPS: f32 = 1e-2;
// ...
let grad = (lp - lm) * inv_2eps;  // central FD, O(eps²) error
self.field_set(*field, i, orig - LR * grad);
```

Central finite-difference gradient with `ε=1e-2` has `O(ε²) = 1e-4` truncation error per gradient component, but on a softmax-cross-entropy surface with near-saturated rows (the LR comment at L125-131 admits `|∂L/∂logit| ≈ 0.125` at init and warns of saturation), the **noise-to-signal ratio** on the 64 `W_basis` entries can easily exceed the gradient signal. FUNCATTN has 64 extra params (`K·D = 8·8`) over SDPA; those are exactly the params most exposed to FD noise because they sit furthest from the loss.

A real autograd path (or even `FD_EPS=1e-3`) has not been tried. If the 3 misses vanish under analytic gradients, the failure was an optimizer artifact.

### A4. Random-init basis tested against paper's trained-basis headline (L399, L425)

```rust
let w_basis = orthogonal_init(K, D, rng);  // random orthogonal
```

FUNCATTN's paper headline numbers (Research 257 §1.4, Table: -22% elasticity, -26% Darcy, etc.) are all with **backprop-trained** `W_basis`. The G6 test uses random orthogonal init and declares "null result" — this is testing FUNCATTN in its weakest regime and calling it a structural ceiling. Research 257 §1.5 explicitly warns: *"Training is required for basis matrices W_Φ, W_Ψ. The closed-form C solve is inference-time; the basis is not."*

## What would prove it's NOT a bug (the cheap diagnostics)

Three diagnostics, in increasing order of effort. Each is independently decisive for the verdict.

- [ ] **D1 — Dump the 3 failing samples.** Add instrumentation to print `(seq, masked_pos, true_token, predicted_token, probs_row)` for every misclassified eval sample. **Decision rule:**
  - If ≥1 of the 3 has `a == b` (degenerate sequence) → **verdict invalid, A2 confirmed, test must reject `a==b`**.
  - If all 3 share the same `masked_pos` parity → structural pattern, real ceiling likely.
  - If scattered with distinct `(a,b)` pairs → inconclusive, run D2.

- [ ] **D2 — K-sweep (K=8 → 16 → 32, V held at 8).** Re-run G6 at each K. **Decision rule:**
  - If acc → 1.000 at K=16 or K=32 → **A1 confirmed, K=V was a corner case**, G6 should be re-gated at realistic K.
  - If acc stays at 0.969 across all K → real capacity-independent ceiling, original verdict stands.

- [ ] **D3 — FD_EPS sweep (1e-2 → 1e-3 → 1e-4) and/or analytic-gradient POC.** **Decision rule:**
  - If acc → 1.000 at smaller FD_EPS → **A3 confirmed, FD noise artifact**, original verdict invalid.
  - If acc stays at 0.969 → optimizer was not the bottleneck.

- [ ] **D4 — Reject `a == b` in `generate_pattern_dataset`.** One-line fix; eliminates A2 unconditionally. Should ship regardless of D1's finding.

## What does NOT need re-running

- **G1–G5** — these passed and the test configs are not in question. The FUNCATTN forward path itself (`funcattn_forward` in `funcattn.rs`) has a `funcattn_reference` cross-check in its own `mod tests` that validates the math bit-identically; the math is not the suspect.
- **The riir-ai side (Plan 318 rank-k latent functor)** — uses *linear-basis rank-k*, a different code path that passed its own GOAT (G1 100/100, G2 delta=+0.75). Not affected by this issue.

## POC

A runnable POC lives at [`tests/funcattn_g6_bug_poc.rs`](../tests/funcattn_g6_bug_poc.rs) (ships with this issue). It runs four probes on the same dataset/seed as G6:

1. **Probe-A (A2):** counts degenerate `a==b` sequences in the eval set, then re-generates the eval set with `a != b` guaranteed and re-runs G6 at the same seed. Reports the new acc.
2. **Probe-B (A1):** K-sweep at K=8, 16, 32. Reports acc at each.
3. **Probe-C (A3):** FD_EPS sweep at 1e-2, 1e-3. Reports acc at each. (1e-4 omitted by default — ~16× slower.)
4. **Probe-D (A4 sanity):** confirms whether `funcattn_forward` matches `funcattn_reference` on the exact G6 eval samples (rules out an implementation drift between the test's predictor wrapper and the primitive).

Run with:
```bash
cargo test --features funcattn --release --test funcattn_g6_bug_poc -- --nocapture
```

**If any single probe flips acc to 1.000, the original G6 verdict is invalid and `funcattn` should be re-gated before any promotion/demotion narrative is treated as settled.**

## Decision matrix after POC runs

| Probe outcome | Verdict |
|---|---|
| All probes: acc stays 0.969 | Original G6 stands. Close issue as "confirmed real ceiling." |
| D1 finds `a==b` in failing samples | G6 invalid. Fix dataset (D4), re-gate. |
| D2 K=16 or K=32 → 1.000 | G6 invalid at K=V. Re-gate at realistic K. |
| D3 FD_EPS=1e-3 → 1.000 | G6 invalid (FD noise). Re-gate with analytic or smaller FD. |
| Multiple probes flip | G6 invalid on multiple independent counts — strongest case for promotion re-evaluation. |

## POC run results (2026-07-07) — CONFOUND DETECTED, VERDICT NOT FLIPPED

The POC was run in release (`cargo test --features funcattn --release --test funcattn_g6_bug_poc -- --nocapture`). **All 5 probes reported acc=1.0000 and printed `*** FLIPS ***`. This conclusion is INVALID — the POC has a seed confound that accounts for the entire accuracy gain, independent of the A1/A2/A3 hypotheses.**

### The confound: `seed_offset=1`

Every probe calls `train_and_eval_fa(...)` with `seed_offset: 1` as the last argument:

```rust
// POC line 513 (Probe-A), 543 (Probe-B), 561 (Probe-C), 580+ (Probe-E)
let acc = train_and_eval_fa(&train_nd, &eval_samples_nd, 8, steps, 0.05, 1e-2, 1);
//                                                                                ^
//                                                                          seed_offset = 1
```

Inside `train_and_eval_fa` (POC line 475):
```rust
let mut rng = Rng::new(SEED_U64 + seed_offset);  // SEED_U64 + 1, NOT SEED_U64
let mut fa = FuncattnPredictor::new(&mut rng, k);
```

The POC initializes the FUNCATTN predictor from a **completely independent RNG stream** (`SEED_U64 + 1`), while G6 initializes the predictor from the **continuation of the same stream** that generated the dataset (`SEED_U64` → 48 sequences of draws → predictor init). These are different random initializations of the 456 weights.

### Proof the seed alone explains the flip

| Run | Seed (predictor init) | Result |
|---|---|---|
| Original G6 test (`funcattn_g6_token_prediction_lm_domain`), run in release | `0x00C0_FFEE_42AA` (G6's stream, post-data) | **0.9688** — reproduces the recorded verdict exactly |
| POC all probes | `0x00C0_FFEE_42AA + 1` (independent stream) | **1.0000** everywhere |

The original G6 test was re-run in the same release build and still produces 0.9688 (3/128 miss, loss → 0.0000, SDPA → 1.0000). The ONLY material difference between G6 and the POC's probes is the predictor's random initialization. The accuracy improvement is attributable to the seed change, **not** to reject-degenerate (A2), K-sweep (A1), or FD_EPS (A3).

### Why Probe-E's `train=admit eval=admit (G6 ORIGINAL)` cell is misleading

Probe-E's first cell is labeled "G6 ORIGINAL" and also reports 1.0000. It is **not** G6's original config: it uses the same `seed_offset=1` predictor init as every other cell. It reproduces G6's *dataset* (admit-degenerate train+eval) but not G6's *predictor initialization*. The label is a false equivalence.

### What the POC must do to validly test A1/A2/A3

The POC must reproduce G6's exact RNG draw order and vary only one factor at a time:

1. **`seed_offset=0` is necessary but NOT sufficient.** G6 generates data and inits the predictor from one continuous stream (`rng = Rng::new(SEED_U64)` → `generate_pattern_dataset(&mut rng, 32, ...)` → `generate_pattern_dataset(&mut rng, 16, ...)` → `FuncattnPredictor::new(&mut rng)`). The POC's current architecture separates data-gen and predictor-init into two independent streams, so even with `seed_offset=0` the predictor init differs from G6.
2. **Structural fix required:** `train_and_eval_fa` must accept a pre-initialized `&mut FuncattnPredictor` (or take the data-generation RNG and continue drawing from it for init), matching G6's stream interleaving. Then vary only the A1/A2/A3 factor while holding init constant.
3. **Minimum decisive experiment:** G6-exact init + reject-degenerate eval (A2 only). If acc → 1.000 with identical predictor init, A2 is confirmed. If acc stays 0.969, A2 is ruled out and the 3 misses are structural.

### Verdict

**G6's null result (FUNCATTN acc=0.969 < SDPA acc=1.000) STANDS.** The POC's "all probes flip" finding is an artifact of the `seed_offset=1` confound and does not invalidate G6. The A1/A2/A3 hypotheses remain **untested** — they require a POC that reproduces G6's exact RNG stream. Until that POC exists and flips the verdict, do not treat `funcattn` as prematurely demoted.

This is not a defense of G6 — it's a defense of the GOAT gate's integrity in the other direction: a POC that "flips" a verdict for the wrong reason (seed change) is just as invalid as a gate that fails for the wrong reason (test artifact). The issue's own TL;DR standard applies symmetrically.

## Out of scope (do NOT pursue in this issue)

- Trained basis (A4) — that's the riir-ai Plan 318 path, properly deferred. This issue only owns the *katgpt-rs-side* question of whether G6's null-result narrative is justified by the test as written.
- Promoting `funcattn` to default — even if the POC flips the verdict, promotion is a separate human decision per Plan 286 T4.4. This issue only owns "is the gate honest."

## Cross-refs

- [Plan 286](../.plans/286_functional_attention_spectral_transport.md) T4.4 — the LLM-domain gate definition
- [Bench 058](../.benchmarks/058_funcattn_goat.md) G6 — the verdict under review (L395-508)
- [Research 257](../.research/257_Functional_Attention_Spectral_Transport_Operator.md) §1.5, §5 Q2 — paper's NLP-deferred caveat
- [Research 100](../.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md) §"Plan 332 Followup" — fixed-vs-learned basis nuance precedent
- [riir-ai Plan 318](../../riir-ai/.plans/318_latent_functor_rank_k_upgrade.md) — rank-k trained-basis path (out of scope here)
