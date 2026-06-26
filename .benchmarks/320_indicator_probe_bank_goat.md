# Plan 320 — Indicator Probe Bank GOAT Gate Results

**Date:** 2026-06-25
**Bench:** `crates/katgpt-core/benches/bench_320_indicator_probe_bank_goat.rs`
**Plan:** [`.plans/320_misalignment_indicator_probe_bank.md`](../.plans/320_misalignment_indicator_probe_bank.md)
**Research:** [`.research/301_Misalignment_Indicator_Probe_Bank.md`](../.research/301_Misalignment_Indicator_Probe_Bank.md)
**Source paper:** [Zhou et al. 2026 — Probing the Misaligned Thinking Process of Language Models](https://arxiv.org/pdf/2606.24251) (ICML 2026 Mech Interp Workshop)
**Verdict:** **ALL GATES PASS — G1–G5, G6, G7 green.** Eligible for Phase 5 promotion of `indicator_probe_bank` + `indicator_similarity` to default-on; `indicator_cascade` stays opt-in (consumer-crate verifier territory).

---

## Synthetic bank

All gates run on a synthetic planted-structure bank (no game semantics, no real
direction-vector training):

- **8 indicators** (the paper's strongest 8: Action Concealment, Malicious Action
  Planning, Observer Suspicion Modeling, Misalignment Cover Story, Strategic
  Omission, Recognized-Problem Suppression, Error Calibration, Strategic
  Underperformance). Generic label discriminants only — the private 18-indicator
  NPC taxonomy lives in riir-ai.
- **D = 72** state dim (HLA[8] + style_weights[64]).
- Each direction is a sparse scaled unit vector (`DIR_SCALE = 4.0`) on a disjoint
  coordinate axis (every 9th coord), so within-block and cross-block cosines are
  ~0 and per-indicator AU-ROC isolates the single-feature separability the paper
  hypothesizes.
- Threshold = `DIR_SCALE / 2` so a clean positive (raw ≈ DIR_SCALE) fires and a
  noise-only negative (raw ≈ N(0, DIR_SCALE·NOISE_STD)) does not.
- `NOISE_STD = 0.5`. Deterministic PRNG (xorshift32) so results are reproducible.

**Transcript positives are 2-indicator clusters** (a planted pair `(i, i^1)` both
fire): this mirrors the paper's premise that a genuine misalignment event shows
**correlated** indicator activation, not a single isolated spike. Negatives are
pure noise (no direction component), so any firing is a spurious single spike
that the stage-2 label-coherence verifier can reject.

---

## Gate results

| Gate | Contract | Target | Result | Verdict |
|------|----------|--------|--------|---------|
| G1 | Per-indicator AU-ROC | all 8 ≥ 0.85 | all 8 = 1.000 | ✅ PASS |
| G2 | OR-fusion transcript-TPR / turn-FPR | TPR ≥ 0.85 at FPR ≤ 0.05 | TPR 1.000 at FPR 0.041 (tau=0.96) | ✅ PASS |
| G3 | Cascade FPR reduction | ≥ 5× at ≤ 10pp TPR cost | 100× (FPR 0.071→0.000, 0pp cost) at tau=0.94 | ✅ PASS |
| G4 | Hot-path latency + alloc-free | < 200ns/call, 0 allocs/100 calls | 53.9 ns/call, 0 allocs | ✅ PASS |
| G5 | Similarity block recovery (ARI) | ARI ≥ 0.9 | ARI 1.000 | ✅ PASS |
| G6 | Feature-off zero-overhead | no regression | `cargo check --no-default-features` clean | ✅ PASS |
| G7 | Wire-format integrity (tamper-evident) | reject tampered bytes | `BankLoadError::HashMismatch` | ✅ PASS |

**Run command:**
```bash
cargo bench -p katgpt-core --features "indicator_probe_bank indicator_similarity indicator_cascade" \
  --bench bench_320_indicator_probe_bank_goat -- --nocapture
```

(If the intermittent macOS dyld/amfid launch stall hits, run the compiled release
binary directly with `DYLD_PRINT_STATISTICS=1`; the stall is non-deterministic and
does not affect result correctness. See Plan 327 session notes for the workaround.)

---

## G1 — per-indicator AU-ROC

For each of the 8 indicators, generate 500 positive states (`direction_i + noise`)
+ 500 negative states (pure noise), project onto the indicator, compute AU-ROC via
the rank-counting identity.

```
ind0=1.000  ind1=1.000  ind2=1.000  ind3=1.000
ind4=1.000  ind5=1.000  ind6=1.000  ind7=1.000
worst = 1.000  (≥ 0.85 ✓)
```

The synthetic setup gives perfect separability because the direction magnitude
(DIR_SCALE=4.0) dominates the noise (NOISE_STD=0.5). This confirms the primitive
correctly projects + sigmoid-gates; real-world AU-ROC on the 18-indicator NPC
taxonomy will be measured in riir-ai Phase 2 (the paper's hypothesis is that
low-dim state is *more* separable, so this is a best-case sanity check).

## G2 — OR-fusion transcript-TPR / turn-FPR

1000-state transcript (50 positive 2-indicator clusters, 950 negative noise).
Sweep `tau_fire` from 0.50 to 0.99; report the largest transcript-TPR achievable
at turn-FPR ≤ 0.05.

```
best: transcript-TPR = 1.000 at turn-FPR = 0.041, tau = 0.96  (TPR ≥ 0.85 ✓, FPR ≤ 0.05 ✓)
```

## G3 — cascade FPR reduction

Same transcript shape (200 positives for a denser FPR baseline). Stage-1 = bank
OR-fusion; stage-2 = `LabelCoherenceVerifier` (confirm only if a 2nd indicator
also fires above the same `tau_fire`). Sweep `tau_fire`; find the operating point
with the highest FPR-reduction ratio at ≤ 10pp TPR cost.

```
at tau=0.78: s1 FPR=0.333 → s2 FPR=0.064 (5.2× reduction, 0pp cost)   ← first ≥5× point
at tau=0.88: s1 FPR=0.179 → s2 FPR=0.009 (20.4× reduction, 0pp cost)
at tau=0.94: s1 FPR=0.071 → s2 FPR=0.000 (stage-2 drives FPR to 0; sentinel ratio=100, 0pp cost)  ← selected
```

The cascade's payoff holds across a wide tau band: the 2-indicator-cluster design
(true positives) is always corroborated by a second spike, while single-spike
false positives are rejected because their 2nd-highest sits at the noise floor.
The stub verifier is intentionally simple (not a real LLM judge); real-world FPR
reduction on the NPC taxonomy is measured in riir-ai Phase 2.

## G4 — hot-path latency + alloc-free

`IndicatorProbeBank::project_all_into` + `or_fused_fire` over 1,000,000 calls
after warmup (N=8 indicators, D=72 state dim), then alloc-count over 100 calls
via `CountingAllocator`.

```
53.9 ns/call  (< 200ns ✓ — 3.7× under target)
0 allocs / 100 calls  (= 0 ✓)
```

The hot path is caller-owned-scratch throughout: `project_all_into` writes into a
caller-provided `&mut [f32; N]`, `or_fused_fire` reads it back. No logging, no
trait-object boxing in the hot loop.

## G5 — similarity block recovery (ARI)

Construct a block-structured bank: 4 blocks of 2 indicators, each block on a
disjoint 2-dim subspace with within-block cosine ≈ 0.69 (directions `[1, 0.4]`
and `[0.4, 1]`). Run `IndicatorSimilarityMatrix::cluster(0.6, 0.6)`; compute
Adjusted Rand Index vs the planted 4×2 partition.

```
ARI = 1.000  (≥ 0.9 ✓ — perfect recovery)
```

The complete-linkage cluster algorithm (merge `Ga, Gb` iff every cross-pair ≥
`tau_intra`, densest-merge-first for determinism) recovers the planted blocks
exactly.

## G6 — feature-off zero-overhead

```bash
cargo check -p katgpt-core --no-default-features    # clean
cargo check -p katgpt-core                           # clean (default, indicator_* off)
cargo check -p katgpt-core --all-features            # clean
```

No `indicator_probe_bank` / `indicator_similarity` / `indicator_cascade` code is
compiled when the features are off. The `pruners` parent module is now always
compiled (decoupled from `review_metrics` in Phase 1) but the indicator
submodules gate themselves.

## G7 — wire-format integrity

Round-trip `to_frozen_bytes` → `from_frozen_bytes` succeeds; flipping one byte in
the directions body yields `BankLoadError::HashMismatch` (the recomputed BLAKE3
over `directions ++ thresholds` no longer matches the embedded hash).

```
clean round-trip: OK
tampered byte → BankLoadError::HashMismatch  ✓ (tamper-evident)
```

---

## Phase 5 promotion decision

**Promote `indicator_probe_bank` + `indicator_similarity` to DEFAULT-ON.**
Rationale (per AGENTS.md GOAT rule):

- **All gates G1–G7 pass** on the synthetic bank.
- **Pure modelless gain:** the bank holds pre-computed, BLAKE3-committed
  direction vectors loaded at init; runtime is dot-product + sigmoid + argmax.
  No training, no gradient descent, no learned weights. The only weight
  mutation is freeze/thaw (snapshot swap). This satisfies the modelless-first
  mandate.
- **Zero overhead when no bank is loaded:** the feature compiles in types +
  traits, but does nothing unless a caller constructs an `IndicatorProbeBank`
  and calls `project`. Downstream impact is pure additive.
- **Read-side primitive:** `indicator_probe_bank` and `indicator_similarity`
  are pure reads (project onto frozen directions, compute pairwise cosines).
  This matches the `FutureBehaviorProbe` (Plan 292) precedent: pure read-side
  primitives promote to default-on; the gain is structural correctness (OR-fused
  multi-direction monitoring, inspectable similarity structure).

**Keep `indicator_cascade` OPT-IN.** The cascade implies a stage-2 verifier impl,
which is consumer-crate territory (riir-ai supplies the LLM-judge impl). The open
half ships the `IndicatorVerifier<L>` trait + stubs; the cascade driver itself is
correct and zero-alloc, but promoting it to default-on would imply a default
verifier that doesn't exist. Opt-in is the honest default.

---

## TL;DR

All 7 GOAT gates pass (G1 AU-ROC 1.000, G2 TPR 1.000/FPR 0.041, G3 100× FPR
reduction at 0pp cost, G4 53.9ns/0 allocs, G5 ARI 1.000, G6 feature-off clean,
G7 tamper-evident). `indicator_probe_bank` + `indicator_similarity` promote to
DEFAULT-ON (pure modelless read-side primitives, zero overhead when unloaded);
`indicator_cascade` stays opt-in (consumer-crate verifier territory). Private
selling-point moat (18-indicator NPC taxonomy, bidirectional cognitive monitoring,
KG-triple audit trail) lives in riir-ai `.research/157_*.md` + downstream plans.
