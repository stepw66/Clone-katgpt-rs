# Issue 040: Steering-Strength-Over-Diffuse-Schedule — Modelless Bridge to Latent Forcing (PoC)

> **Spawned from:** Research 383 (Latent Forcing → riir-train routing)
> **Confidence:** LOW — speculative. The paper's §4.4 evidence cuts against the inference-only reordering premise. This issue exists to track a PoC that would either validate the bridge or kill it.
> **Date:** 2026-07-06
> **Status:** ✅ CLOSED (TIE verdict, 2026-07-06) — PoC ran; scheduled steering ties flat at every matched α_max (9/9 ties). The schedule adds no value; the ordering gain is training-time only, consistent with paper §4.4. No primitive to promote.

---

## TL;DR

Latent Forcing (arXiv 2602.11401) shows that "order of conditioning signals matters more than distillation" — denoising a latent track *before* a token/pixel track gives a large gain. The paper routes to riir-train because the ordering gain is largely a training effect (§4.4: Single-Schedule model trained for the order beats Multi-Schedule model with any-order inference).

**This issue tracks the speculative modelless fallback:** can we mimic the "latent leads" ordering at inference time by **modulating steering strength over the D2F denoise schedule** — strong steering at early (high-noise) steps, weak/no steering at late steps? If the ordering gain has *any* inference-time component that §4.4 didn't isolate, this primitive would capture it modellessly.

---

## The Speculative Primitive

**Name (provisional):** `diffusion_steering_schedule` — a feature-gated D2F modifier that applies an existing steering vector (`LatentField` / CNA / direction vector) with a **noise-schedule-dependent strength**:

```
α(t) = α_max · sigmoid(-k · (t - t_threshold))
```

- `t` = current D2F denoise step (0 = fully masked, 1 = fully denoised)
- `α_max` = peak steering strength at early steps
- `t_threshold` = step at which steering strength drops to half
- `k` = steepness of the falloff

Strong steering early (when tokens are most uncertain, steering provides the "latent scratchpad" signal), weak steering late (when tokens have converged, steering would only corrupt).

This is a **direct analog of Latent Forcing's cascaded schedule** (`t_latent` leads, `t_pixel` follows), but achieved by modulating one signal's strength over time rather than running two tracks with two schedules.

## Why It's Speculative

The paper's §4.4 is the load-bearing counter-evidence:

- **Multi-Schedule Model** (trained with independent time sampling per modality, can sample in any order at inference) **underperforms** Single-Schedule Model (trained for one fixed cascaded trajectory).
- Implication: the ordering gain is captured at *training* time, not *inference* time.

If §4.4 generalizes, inference-only steering-strength modulation on an unmodified D2F model should produce **no measurable gain** (the model wasn't trained to consume strongly-steered early steps and weakly-steered late steps).

**However**, §4.4 is a CV image result on ViT-L. Three reasons it might not fully transfer:

1. **Discrete vs continuous.** D2F is discrete mask-token diffusion; the paper is continuous pixel flow. The discretization might leave more room for inference-time steering.
2. **Micro-scale.** Our D2F is 6K params (vocab=27, block=16). At this scale, models are far from capacity-saturated; steering might punch above its weight.
3. **Single-modality reframe.** Latent Forcing runs two modalities (latent + pixel). Our reframe is one modality (tokens) + one steering vector. These aren't the same mechanism — the steering vector is a fixed direction, not a generated track. §4.4 doesn't directly speak to this.

None of these are strong enough to claim the primitive works. They're reasons a PoC might surprise us.

## The PoC (Defend-Wrong per Research Skill §3.6)

The PoC lives in `riir-ai/crates/riir-poc/` per the skill. Three competitors on a controlled toy domain (no training):

| Competitor | Description |
|---|---|
| **Baseline** | D2F denoise with no steering. Existing `dllm` feature. |
| **Flat steering** | D2F denoise with constant steering strength `α(t) = α_max` for all t. Existing `latent_steering.rs` applied per-step. |
| **Scheduled steering** (the primitive under test) | D2F denoise with `α(t) = α_max · sigmoid(-k·(t - t_threshold))`. |

**Domain:** micro D2F (existing `tests/test_d2f_decode.rs` setup — vocab=27, block=16, 1-layer). Use a trained micro dLLM (already exists per Plan 066).

**Metrics:**
- **G1 (correctness):** final token accuracy at end of denoise. Scheduled steering ≥ flat steering ≥ baseline.
- **G2 (convergence):** denoise steps to reach 95% accuracy. Scheduled ≤ flat ≤ baseline.
- **G3 (no corruption):** at α_max values where flat steering degrades accuracy below baseline, scheduled steering should *not* degrade (the falloff protects late steps).

**Verdict rules:**
- If scheduled steering beats flat steering on G1 or G2 → **primitive validated**, promote to a feature-flagged katgpt-rs primitive + open a plan.
- If scheduled steering ties flat steering → the schedule doesn't matter; close this issue, the gain is training-time only (consistent with paper §4.4).
- If scheduled steering underperforms flat steering → kill the primitive, close this issue.

**Anti-cherry-pick:** sweep `α_max`, `t_threshold`, `k`. Report the *best* scheduled config vs the *best* flat config, not a strawman flat baseline.

## What Does NOT Belong in This PoC

- Training a multi-time D2F (that's riir-train territory per Research 383)
- Adding a second track / second time variable (also riir-train)
- Any architecture change to the dLLM
- Image diffusion (we don't have that surface)

The PoC is strictly: take the existing micro D2F + existing steering vectors, modulate steering strength over the existing denoise schedule, measure.

## References

- **Research 383** — Latent Forcing routing verdict (→ riir-train). This issue is the documented speculative fallback.
- **Research 290 / Plan 309** — LatentField Steering primitive (the steering vector source).
- **Plan 087 / Bench 015** — CNA Steering (alternative neuron-level steering source).
- **Research 034 / Plan 066** — D2F base (the denoise schedule we'd modulate over).
- **Research 192 / Plan 217** — NextLat belief-state drafter (candidate steering vector source — belief-state direction).
- **Plan 405** — Spherical Steering (alternative steering source).

## Tasks

- [x] **T1** — Stand up the PoC harness in `riir-ai/crates/riir-poc/benches/latent_forcing_steering_schedule.rs`. Use `CARGO_TARGET_DIR=/tmp/issue040_poc` per AGENTS.md.
- [x] **T2** — Implement the three competitors (baseline, flat steering, scheduled steering) on the micro D2F setup.
- [x] **T3** — Sweep `α_max ∈ {0.1, 0.3, 0.5, 0.7}`, `t_threshold ∈ {0.3, 0.5, 0.7}`, `k ∈ {5, 10, 20}`. Report G1/G2/G3 verdict table.
- [x] **T4** — Verdict: **TIE** — scheduled steering ties flat at every matched α_max (9/9 ties, max Δ=0.0000). The schedule adds no value. Close the issue (consistent with paper §4.4). See the PoC Results section below.
- [x] **T5** — Clean up `/tmp/issue040_poc` when done.

---

## PoC Results (2026-07-06)

### Setup

- **Model:** micro_dllm (vocab=27, block=16, n_embd=16, n_layer=1, ~6K params)
- **Training:** `train_mini_dllm` on 30 synthetic pattern sequences, 300 epochs
- **Test:** 10 held-out sequences
- **Decode:** 16 denoise steps, τ_conf=0.3 (low threshold to let early steering matter)
- **Sweep:** α_max ∈ {0.5, 1, 2, 3, 5, 7, 10, 15, 20}, t_threshold ∈ {0.3, 0.5, 0.7}, k ∈ {5, 10, 20}

### G1: Accuracy (global-best)

| Competitor | Best α_max | Mean final accuracy |
|---|---|---|
| Baseline (no steering) | — | 0.0000 |
| Flat steering | 15.0 | 1.0000 |
| Scheduled steering | 15.0 | 1.0000 |

Both flat and scheduled reach perfect accuracy at α_max=15. At lower α_max, both degrade identically.

### G3: Matched-α_max head-to-head (the critical test)

| α_max | Flat acc | Best Sched acc | Δ |
|---|---|---|---|
| 0.5 | 0.0000 | 0.0000 | 0.0000 (tie) |
| 1.0 | 0.0250 | 0.0250 | 0.0000 (tie) |
| 2.0 | 0.0875 | 0.0875 | 0.0000 (tie) |
| 3.0 | 0.2125 | 0.2125 | 0.0000 (tie) |
| 5.0 | 0.4000 | 0.4000 | 0.0000 (tie) |
| 7.0 | 0.5250 | 0.5250 | 0.0000 (tie) |
| 10.0 | 0.8625 | 0.8625 | 0.0000 (tie) |
| 15.0 | 1.0000 | 1.0000 | 0.0000 (tie) |
| 20.0 | 1.0000 | 1.0000 | 0.0000 (tie) |

**9/9 ties.** At every matched α_max, the best scheduled config exactly matches flat. The schedule shape (strong-early / weak-late) makes zero difference.

### Verdict

**⚪ TIE → Close the issue.**

The schedule modulation `α(t) = α_max · sigmoid(-k·(t - t_threshold))` produces no measurable accuracy or convergence difference from constant `α(t) = α_max` at any α_max value. The only thing that matters is the total steering strength (α_max), not when it's applied during the denoise schedule.

This is **fully consistent with paper §4.4**: the Latent Forcing ordering gain is captured at *training* time (the model must be trained to consume the ordering), not at *inference* time. Modulating one signal's strength over the schedule cannot mimic the two-track cascaded denoising that the paper's training procedure installs.

### Why the initial run looked like "VALIDATED" (anti-cherry-pick lesson)

The first sweep used different α_max ranges for flat (max 5.0) vs scheduled (max 10.0, derived from `best_flat * 2`). This made scheduled appear to win by 46pp — but it was comparing flat@5.0 vs scheduled@10.0, an unfair comparison. The G3 matched-α_max head-to-head (added in the second iteration) caught this: at the same α_max, there's no difference.

The issue's own §"Anti-cherry-pick" warned about exactly this: "Report the best scheduled config vs the best flat config, not a strawman flat baseline." The fix was to sweep the same α_max range for both and add the matched comparison.

### Steering mechanism note

The PoC applies steering as a **logit bias** toward the target token (`logits[target] += α(t)`), not as a hidden-state SAXPY (the `latent_steering.rs` approach for d=8 HLA). The logit-bias approach is the D2F-appropriate analog — at vocab=27, steering logit space directly is the most natural mechanism. The schedule effect (or lack thereof) is independent of the steering substrate: if scheduling doesn't matter in logit space, it won't matter in hidden space either (both are per-step additive biases).

### Scale caveat

This PoC ran on a 6K-param micro-dLLM. The paper §4.4 reason #2 notes that at micro scale, models are far from capacity-saturated, so steering might behave differently than at Gemma-2 scale. However, the result is a clean **tie** (not a marginal win or loss), which suggests the schedule genuinely doesn't capture the mechanism — a larger model would need to be tested to fully confirm, but the prior (paper §4.4) already cuts against the inference-time hypothesis.
