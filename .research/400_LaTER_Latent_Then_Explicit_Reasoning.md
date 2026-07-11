# Research 400: LaTER — Latent-Then-Explicit Reasoning

> **Source:** [LaTER: Efficient Test-Time Reasoning via Latent Exploration and Explicit Verification](https://arxiv.org/abs/2605.07315) — Li et al., arXiv:2605.07315v1, May 2026
> **Date:** 2026-07-09
> **Status:** Done
> **Related Research:** 241 (SwiR — the superset), 325 (Latent Reasoning Survey — §7.2 G1), 275/313 (SwiR plan + real-model GOAT)
> **Classification:** Public

---

## TL;DR

LaTER (Latent-Then-Explicit Reasoning) is a two-stage test-time reasoning paradigm: bounded exploration in continuous latent space, then switch to explicit CoT for verification. Its **training-free** instantiation projects final-layer hidden states back to the input embedding space, preserves the latent KV cache, and uses entropy + model-native stop-token probes to decide when to switch. Paper reports 16–32% token reduction on Qwen3-14B with matching/improved accuracy.

**Distilled for katgpt-rs:** **fully subsumed by SwiR (Research 241, Plan 275, DEFAULT-ON since 2026-06-27).** LaTER's "latent-then-explicit" is the *special case* (single latent→explicit transition) of SwiR's general alternating latent↔explicit controller with asymmetric dwell windows. Every LaTER mechanism maps 1:1 to a shipped SwiR primitive:

| LaTER mechanism | SwiR (shipped, DEFAULT-ON) |
|---|---|
| Latent→explicit two-stage | Alternating latent↔explicit (strict generalization) |
| Entropy-based switch | Block-relative entropy switch `H_t < H̄` |
| Stop-token probe | Switch Count Controller convergence/termination triggers |
| Project hidden state → embedding space | Soft embedding `ẽ_t = Σ_v p_t[v]·e(v)` (identical primitive) |
| Preserve latent KV cache | Soft embedding preserves first-order distribution |
| Training-free | SwiR is training-free end-to-end |

**Verdict: Pass** — 0/4 novelty gate. The modelless mechanism is shipped and *more general* than LaTER. No new primitive, no plan, no guide. The training-side contribution (Latent-Switch-69K corpus, fine-tuning recipe) → riir-train.

---

## 1. Paper Core Findings

### 1.1 The two-stage paradigm

LaTER (§3) proposes: (1) bounded **latent** exploration — the model reasons in continuous hidden-state space without decoding tokens; (2) **explicit** CoT — switch to standard token-by-token generation for verification and answer emission. The switch is the core decision.

### 1.2 Training-free instantiation (the modelless subset)

The paper's training-free path (§4) — the part potentially modelless-distillable:

- **Project final-layer hidden state → input embedding space.** The last hidden state is projected back through the input embedding matrix so it can be re-fed as a "latent token." This preserves the latent KV cache (the latent reasoning work isn't discarded on switch).
- **Entropy switch.** Compute next-token entropy; when it drops below a threshold (confidence rising), switch latent → explicit.
- **Model-native stop-token probe.** Detect when the model's own stop tokens become probable as a secondary switch trigger.
- **Preserve latent KV cache** across the latent→explicit boundary so explicit CoT builds on the latent work.

### 1.3 Trained instantiation

A supervised corpus (Latent-Switch-69K, pairing condensed intuitions with shortened derivations) + latent-rollout fine-tuning yields further gains (Qwen3-14B: AIME 2025 70.0% → 80.0%, −33% tokens). This is training-side → riir-train.

### 1.4 Empirical headlines

| Config | Tokens | Accuracy |
|---|---|---|
| CoT baseline (Qwen3-14B, AIME 2025) | 15,730 | 70.0% |
| Training-free LaTER | 10,661 (−32%) | 73.3% (+3.3pp) |
| Trained LaTER | −33% | 80.0% (+10.0pp) |

Training-free: 16–32% token reduction across several benchmarks with matched/improved accuracy.

---

## 2. Distillation

### 2.1 Why this is subsumed by SwiR (Research 241 / Plan 275)

LaTER's mechanism is a **single-transition special case** of SwiR's alternating controller. SwiR (DEFAULT-ON since 2026-06-27, promoted in Plan 313 T6.2) ships:

1. **Block-Relative Entropy Switch** — `mode_{t+1} = Explicit if (H_t < H̄) else Latent if (H_t > H̄ ∧ Δt ≥ W_E→L)`. This is LaTER's entropy switch + the missing dwell-window discipline. SwiR *resets* H̄ on every switch; LaTER's single-transition design never needs a reset because it switches once.

2. **Asymmetric Dwell Windows** — Explicit→Latent requires `W_E→L` steps of sustained uncertainty (default 512, tuned to 32 for short-response models); Latent→Explicit fires immediately. LaTER has no dwell concept — it switches latent→explicit on the first entropy drop, which SwiR's ablation (Research 241 §1.2) shows causes oscillation without the dwell guard.

3. **Switch Count Controller** — caps transitions at `C_max`, with convergence trigger at `½C_max` (enqueue `</think>`) and termination trigger at `C_max` (inject answer prefix). This is a strict superset of LaTER's stop-token probe: LaTER probes for stop tokens; SwiR *injects* control tokens on a bounded schedule, which is strictly more controllable.

4. **Soft Embedding** — `ẽ_t = Σ_v p_t[v] · e(v)` (probability-weighted mixture over the vocabulary embedding matrix). This is **identical** to LaTER's "project final-layer hidden state back to input embedding space." SwiR's G4 gate (`min_v e(v) ≤ ẽ_t ≤ max_v e(v)` componentwise) proves the soft embedding stays in the vocab convex hull — a correctness property LaTER does not establish.

5. **Signal Mixing at Switch Instants** — `ẽ ← α·ẽ + (1−α)·e_<think>` on latent entry. LaTER has no equivalent; SwiR shows it contributes +0.6pp (paper Tab. 9).

6. **Kurtosis Escape Hatch** — rigid-constraint tasks (the paper's 3D-surface-shortest-path failure) auto-fall-back to explicit-only mode. LaTER has no failure-mode handling.

SwiR additionally reports real-model GOAT: G2 token-efficiency 1.32×/1.37×/1.43× at n=3/5/10 on Gemma 2 2B + MATH-500 (`.benchmarks/313_swir_real_model_goat.md`). LaTER's training-free 16–32% token reduction on Qwen3-14B is in the same regime on a larger model — consistent with, not beyond, the shipped SwiR results.

### 2.2 Vocabulary crosswalk (LaTER ↔ shipped code)

| LaTER term | Shipped equivalent | Where |
|---|---|---|
| latent-then-explicit | `ThinkMode::{Latent, Explicit}` alternating | `src/swir/controller.rs` (Plan 275) |
| project hidden state → embedding space | `soft_embedding(probs, embedding_matrix, ...)` | `src/swir/soft_embedding.rs` |
| entropy switch | `SwiRController::step(entropy, ...)` block-relative H̄ | `src/swir/controller.rs` |
| stop-token probe | Switch Count Controller convergence/termination triggers | `src/swir/controller.rs` |
| preserve latent KV cache | soft embedding preserves first-order distribution (G4 hull invariant) | `src/swir/soft_embedding.rs` + `convex_hull_check.rs` |
| training-free | SwiR is training-free; promoted to default-on | Plan 275, Plan 313 T6.2 |

### 2.3 Latent-space reframing

The research skill requires a latent-space reframe before verdict. The reframe is trivial here because LaTER *is* a latent-space mechanism: the "latent exploration" stage operates on soft embeddings (continuous mixtures in the vocab convex hull), and the switch is a latent-state-derived scalar (entropy) gating a discrete-vs-continuous step decision. SwiR already implements this exact latent-to-latent operation on the soft-embedding state with sigmoid-compatible gating (the entropy comparison is a step function; a sigmoid blend `ẽ = σ(λ(H̄−H_t))·ẽ_latent + (1−σ)·e_argmax` is the Research 253 fusion extension). No new latent-space reframing is possible that SwiR doesn't already cover.

---

## 3. Verdict

**Tier: Pass**

**One-line reasoning:** LaTER's modelless mechanism is a single-transition special case of SwiR (Research 241, Plan 275, DEFAULT-ON), which ships a strict superset (alternating controller + asymmetric dwell + switch-count cap + signal mixing + kurtosis escape). Every LaTER primitive maps 1:1 to shipped SwiR code; SwiR additionally ships what LaTER lacks. The training-side corpus → riir-train.

### Novelty gate (Q1–Q4)

| Q | Answer | Notes |
|---|--------|-------|
| **Q1 No prior art?** | ❌ NO | SwiR (Research 241 / Plan 275, DEFAULT-ON) ships the exact mechanism, more general. Research 325 §2.1 vocabulary crosswalk maps "explicit↔latent switch" → SwiR. |
| **Q2 New capability class?** | ❌ NO | Same capability (latent↔explicit reasoning switch), strictly less general (single transition vs alternating). |
| **Q3 Product selling point?** | ❌ NO | "Latent-then-explicit" is a weaker story than SwiR's "adaptive alternating with overthinking bounds." |
| **Q4 Force multiplier?** | ❌ NO | Single reasoning primitive; SwiR already multiplies the reasoning pack. |

All NO → NOT Super-GOAT, NOT GOAT, NOT Gain. Pass.

### MOAT gate (katgpt-rs domain)

- **In scope:** yes — test-time reasoning primitive.
- **Strengthens moat:** NO — the shipped SwiR is strictly stronger; LaTER adds nothing the moat doesn't already have.
- **Promote/demote:** N/A — no primitive to ship.

### Routing

| Component | Destination | Rationale |
|---|---|---|
| Latent↔explicit switch, entropy gate, soft embedding | (already shipped) SwiR, Plan 275, DEFAULT-ON | LaTER is subsumed; no action. |
| Latent-Switch-69K corpus, latent-rollout fine-tuning | → riir-train | Training-side. |

---

## 4. Cross-References

- **Research 241** (`katgpt-rs/.research/241_SwiReasoning_Explicit_Latent_Switch.md`) — **the superset.** LaTER's mechanism is the single-transition special case.
- **Plan 275** (`katgpt-rs/.plans/275_swir_switch_thinking.md`) — SwiR implementation, DEFAULT-ON (2026-06-27).
- **Plan 313** — SwiR real-model GOAT (G2 token-efficiency 1.32×+ on Gemma 2 2B + MATH-500).
- **Research 325** (`katgpt-rs/.research/325_Survey_Latent_Reasoning_Taxonomy_Unifying_Map.md`) — the latent-reasoning survey; §2.1 vocabulary crosswalk maps "explicit↔latent switch" → SwiR; §7.2 G1 lists System-1.5 as the highest-priority modelless gap (not LaTER).
- **Research 253** (`katgpt-rs/.research/253_SwiR_DMax_Continuous_Router_Fusion.md`) — the sigmoid-blend fusion extension (sub-token continuous router); strictly beyond both SwiR and LaTER.
- **Research 266** (FPRM), **Research 282** (LoopCoder-V2) — halting primitives that compose with the switch controller.

---

## TL;DR

LaTER (arXiv:2605.07315, May 2026) proposes latent-then-explicit test-time reasoning: explore in latent space, switch to explicit CoT when entropy drops / stop tokens appear. Its training-free instantiation is **fully subsumed by SwiR (Research 241 / Plan 275, DEFAULT-ON since 2026-06-27)**, which ships a strict superset — an *alternating* latent↔explicit controller with asymmetric dwell windows, a switch-count cap with convergence/termination triggers, signal mixing at switch instants, and a kurtosis escape hatch. LaTER's single-transition design is the special case; SwiR is the general case with real-model GOAT (G2 token-efficiency 1.32×+ on Gemma 2 2B + MATH-500). Every LaTER mechanism maps 1:1 to shipped SwiR code; SwiR additionally ships what LaTER lacks. **Verdict: Pass** — 0/4 novelty gate; no new primitive, no plan, no guide. The training-side corpus (Latent-Switch-69K) → riir-train.
