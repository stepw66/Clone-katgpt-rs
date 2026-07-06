# Issue 040: Steering-Strength-Over-Diffuse-Schedule — Modelless Bridge to Latent Forcing (PoC)

> **Spawned from:** Research 383 (Latent Forcing → riir-train routing)
> **Confidence:** LOW — speculative. The paper's §4.4 evidence cuts against the inference-only reordering premise. This issue exists to track a PoC that would either validate the bridge or kill it.
> **Date:** 2026-07-06
> **Status:** Open — PoC not started

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

- [ ] **T1** — Stand up the PoC harness in `riir-ai/crates/riir-poc/benches/latent_forcing_steering_schedule.rs`. Use `CARGO_TARGET_DIR=/tmp/issue040_poc` per AGENTS.md.
- [ ] **T2** — Implement the three competitors (baseline, flat steering, scheduled steering) on the micro D2F setup.
- [ ] **T3** — Sweep `α_max ∈ {0.1, 0.3, 0.5, 0.7}`, `t_threshold ∈ {0.3, 0.5, 0.7}`, `k ∈ {5, 10, 20}`. Report G1/G2/G3 verdict table.
- [ ] **T4** — Verdict: validate / kill / promote-to-plan.
- [ ] **T5** — Clean up `/tmp/issue040_poc` when done.
