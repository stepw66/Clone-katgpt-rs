# Issue 131 — Weaver Runtime Integration (inference-only SpeculativeGenerator adapter)

> **Spawned from:** `riir-train/.plans/314_weaver_adapter_training.md` Phase 6
>   ("Open as a katgpt-rs issue when Phase 6 passes" — Phase 6 passed 2026-07-10)
> **Date:** 2026-07-10
> **Status:** BLOCKED — awaiting trained Weaver weights from the riir-train 300k
>   completion training run (which itself requires a frozen trained DFlash
>   drafter + verifier — see "Blocker chain" below)
> **Priority:** Low (deferred until weights exist)
> **Feature gate (proposed):** `weaver_runtime` (opt-in)

## TL;DR

riir-train Plan 314 shipped the **Weaver adapter** — a 56.7M-param
autoregressive residual transformer that corrects the DFlash drafter's
top-K=512 marginal distributions toward the verifier's distribution. Training
loop, data pipeline, backward pass, and synthetic GOAT gate are all DONE in
riir-train. The paper reports **+77% mean acceptance length over chain DFlash,
+32% over DDTree**.

This issue tracks the **katgpt-rs runtime half**: an inference-only adapter
that loads trained Weaver weights (via freeze/thaw) and applies the residual
correction to DFlash draft logits at decode time. **Blocked until trained
weights exist** — no code work can start until the 300k completion training
run produces a `weaver_v1.safetensors` checkpoint with non-trivial gain.

## Blocker chain (why this is blocked)

```
Trained DFlash drafter weights ──────────────────────────┐
Trained verifier (base LLM) weights ─────────────────────┤
                                                         ▼
                              300k-completion precompute (riir-train T4.1 real)
                                         │
                                         ▼
                              Weaver training run (riir-train Phase 5)
                                         │
                                         ▼
                              weaver_v1.safetensors (BLAKE3-checked)
                                         │
                                         ▼
                              THIS ISSUE (katgpt-rs runtime integration)
```

**No trained DFlash/verifier transformer weights exist in any of the 5 repos**
(audited 2026-07-10: only LoRA artifacts + bandit states found, no base model
weights). The precompute and training produce garbage without real frozen
targets. This is the fundamental blocker — not a code problem.

The Weaver adapter's value proposition (distilling the verifier's top-K
distribution into the drafter's marginal) requires a real verifier→drafter
quality gap to close. At the repo quintet's current scale (game-domain
micro-GPT), this gap may not exist. The paper's gain is measured on
production LLMs (Nemotron V2 verifier, DFlash/DDTree drafters).

## What the integration looks like (when unblocked)

### Architecture (from riir-train Plan 314 / Research 402)

```
                    ┌──────────────────────────────────────┐
                    │   Frozen DFlash / DDTree drafter      │
                    └──────────────┬───────────────────────┘
                                   │ top-K=512 token ids + draft logits
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   Weaver adapter (THIS ISSUE)         │
                    │                                       │
                    │   uᵢ = W_c · RMSNorm(hᵢ_dflash) + pᵢ  │  # conditioning
                    │   causal self-attn over draft path    │
                    │   SwiGLU MLP                          │
                    │   ℓ_weaver = top-K projection         │
                    │                                       │
                    │   ℓ_final[topk] = ℓ_dflash[topk]      │  # residual add
                    │                     + ℓ_weaver[topk]  │
                    │   renormalize over K candidates       │
                    └──────────────┬───────────────────────┘
                                   │ corrected top-K marginals
                                   ▼
                    ┌──────────────────────────────────────┐
                    │   Verifier (base LLM)                 │
                    └──────────────────────────────────────┘
```

### Integration point in katgpt-rs

The Weaver correction slots into the **DFlash predict pipeline** as a
post-draft logit corrector, between `dflash_predict_with` producing
`DraftResult.marginals` and the verifier's acceptance check.

**Two integration options (decide when unblocked):**

1. **Marginal corrector (lighter):** Modify `DraftResult.marginals` in-place
   after `dflash_predict_with` returns. The Weaver forward reads the DFlash
   hidden states + top-K token ids, produces the residual, and the marginals
   are renormalized. This is a post-processing step — no DFlash internal
   changes.

2. **Logit corrector (heavier, matches paper):** Intercept the DFlash draft
   logits before marginalization, apply the Weaver residual, then
   renormalize. This requires exposing the raw draft logits from the DFlash
   forward context.

**Recommendation:** option 1 (marginal corrector) for the initial integration —
it's non-invasive and the marginal is what the verifier ultimately consumes.

### Weight loading (freeze/thaw)

Trained weights ship as `weaver_v1.safetensors` with a BLAKE3 manifest
(riir-train Plan 314 T5.2). The runtime loads via the freeze/thaw envelope
(consistent with riir-neuron-db's `MerkleFrozenEnvelope`):

- `WeaverWeights` struct (from riir-train `weaver.rs`) is `#[repr(C)]` Pod.
- Load: `load_checkpoint(path) -> Result<WeaverWeights, Blake3Mismatch>`.
- Feature-gated as `weaver_runtime` (opt-in) — the weights are a trained
  artifact, not modelless-promotable.

### Feature gate

`weaver_runtime = []` (opt-in, default-OFF). The feature:
- Adds the `WeaverCorrector` adapter struct.
- Gates the DFlash post-processing hook.
- Depends on `katgpt-core` types (`DraftResult`) but NOT on riir-train
  (weights load via safetensors, no training dependency at runtime).

**Promotion:** stays opt-in permanently (trained-weight dependency, not
modelless). Unlike modelless primitives, a trained adapter cannot be
default-on because it requires a checkpoint file to exist on disk.

## Acceptance criteria (when unblocked)

- [ ] `WeaverCorrector` struct: holds `WeaverWeights`, implements the forward
      pass (conditioning → causal attn → SwiGLU → top-K projection → residual
      add → renormalize).
- [ ] Load path: `WeaverCorrector::from_checkpoint(path)` reads
      `weaver_v1.safetensors`, verifies BLAKE3, returns the corrector.
- [ ] Integration hook: DFlash `DraftResult.marginals` are corrected when the
      `weaver_runtime` feature is on and a corrector is registered.
- [ ] G1 (correctness): corrected marginals sum to 1.0 over top-K, no NaN/Inf.
- [ ] G2 (gain): mean acceptance length(corrected) > mean acceptance length(raw)
      on the real verifier (not synthetic). This is the real acceptance
      benchmark — the synthetic +134% from riir-train Phase 6 is not
      transferable without real weights.
- [ ] G3 (no-regression): when `weaver_runtime` is OFF, DFlash behavior is
      bit-identical to the current default (zero-cost abstraction).
- [ ] G4 (latency): Weaver forward adds < X µs per draft step (TBD — the
      single-layer model is lightweight, but the top-K=512 projection reads
      4 MiB of weights; needs measurement).

## Why this is NOT modelless-promotable

The Weaver adapter is a **trained** artifact by construction:
- Its 56.7M parameters encode the verifier→drafter distillation.
- Zero-init weights produce zero residual (no-op) — the value IS the trained
  weights.
- No freeze/thaw, raw/lora hot-swap, or latent projection can substitute —
  the correction is a learned nonlinear mapping from drafter context to
  logit residuals.

This is a legitimate riir-train dependency. The modelless mandate
(AGENTS.md §3.5) does not apply — the modelless path was never the question
for Weaver (unlike Research 400 / Issue 428, where the modelless path was
prematurely declared exhausted).

## Cross-references

- **riir-train Plan 314** — the training plan (DONE, synthetic validation)
- **riir-train Research 402** — the paper distillation (Weaver = residual
  over top-K marginals)
- **katgpt-core `DraftResult`** (`crates/katgpt-core/src/speculative/types.rs`)
  — the integration point (marginals field)
- **katgpt-speculative `dflash_predict_with`** (`crates/katgpt-speculative/src/dflash.rs`)
  — where the correction hooks in
- **riir-neuron-db `MerkleFrozenEnvelope`** — the freeze/thaw weight-loading
  pattern to mirror
