# Research Verdict 383: Latent Forcing — Multi-Time Multi-Tokenizer Scheduling (→ riir-train)

> **Paper:** [arXiv 2602.11401](https://arxiv.org/abs/2602.11401) — Baade, Chan, Sargent, Chen, Johnson, Adeli, Fei-Fei (Stanford), Feb 11 2026, ICML 2026
> **Code:** https://github.com/AlanBaade/LatentForcing
> **Domain:** Pixel-space image diffusion — BUT the core mechanism (multi-time multi-tokenizer scheduling) is domain-general and applies to our D2F diffusion LLM stack
> **Status:** → **riir-train** (training method for discrete diffusion LLMs). NOT a modelless katgpt-rs primitive — §3.5 paths exhausted.
> **Classification:** Public
> **Date:** 2026-07-06
> **Revision:** v2 — initial verdict (PASS, "out of scope / domain mismatch") was **wrong**. The user correctly flagged that we ship a diffusion LLM (D2F, Plan 066 / Research 034) and the multi-time scheduling mechanism is genuinely uncovered in our stack. This revision corrects the routing.

---

## TL;DR

Latent Forcing trains a single diffusion model with **two time variables** — one for a latent track (DINOv2/Data2Vec2), one for a pixel track — and schedules the latent track to denoise first ("scratchpad"). The core finding: **"order of conditioning signals matters more than distillation"** (1.9× FID reduction vs JiT+REPA, distinct from one-time REPA gains). The mechanism is a **multi-time multi-tokenizer flow-matching training recipe**, not an inference-time trick.

**Status for the quintet:** → **riir-train**. Applied to our D2F discrete diffusion LLM (Plan 066, Research 034 — Wang et al. 2508.09192), the analog is: add a continuous latent track alongside the discrete token track with a leading noise schedule, so denoised latent structure conditions later token denoising. This requires **retraining** the dLLM with two tracks + two time variables. §3.5 modelless paths (freeze/thaw, raw/lora, latent correction) do not unblock it — it is an architectural + training-recipe change, not a systematic-bias correction. The paper itself confirms this: §4.4 shows the "Single-Schedule Model" (trained for the specific cascaded order) beats the "Multi-Schedule Model" (trained for any order) — inference-only reordering is insufficient.

**Distilled for katgpt-rs (modelless, inference-time):** Nothing load-bearing. A weaker modelless fallback exists (use a frozen belief-state drafter — NextLat MLP / HLA — to generate latent conditioning in one shot before D2F token denoising) but the dLLM was not trained to consume this extra conditioning, so it would be out-of-distribution. Not a GOAT candidate without riir-train support.

---

## 1. Paper Core Findings (verified by reading the full HTML)

1. **Multi-tokenizer flow matching.** A single DiT jointly diffuses k=2 modalities (pixels + DINOv2 latents), each with its own time variable `t_latent`, `t_pixel`. Loss is a weighted sum of per-modality v-losses (Eq 1). Generalizes to k>2.
2. **Order matters more than distillation.** Ablations (Table 9): latent-before-pixel ordering drives the gain (1.9× FID reduction vs JiT+REPA, 2.5× vs JiT), distinct from REPA distillation which is a one-time training speedup that loses effectiveness late in training (Wang et al. 2025).
3. **Generation order = SNR trajectory.** Order formalized as per-modality SNR over global time (Eq 3). Cascaded (latent fully denoised first) or variance-shifted (`α=9`, Eq 4) both win; concurrent/parallel schedules lose (Fig 3).
4. **Minimal architecture change.** Add per-modality time-embedding MLP (+0.5% params), add per-patch latent+pixel embeddings (token count unchanged), optional 2-expert output head (last 4 layers split into latent-expert + pixel-expert, zero extra params/FLOPs).
5. **Lossless & end-to-end at inference.** No VAE/RAE bottleneck — latent discarded at output, pixels are the output. PSNR ∞.
6. **Cascaded-error mitigation (training-only).** Small noise (`β ≤ 0.25`) on latents during pixel training steps prevents overfitting to high-freq latent detail (Table 6). Harmful at inference (Table 7).
7. **Scaling-as-scheduling identity.** Scaling a modality's variance by `α` is informationally equivalent to time-shift `f_α(t) = tα/(1+(α-1)t)` (Eq 4) — confirms Simple Diffusion (Hoogeboom 2023) for multi-modality.
8. **Multi-Schedule vs Single-Schedule training.** §4.4: Multi-Schedule (independent time sampling per modality, any-order inference) underperforms Single-Schedule (one fixed trajectory, trained in-distribution). **Implication: the ordering gain requires training for that specific ordering.**

---

## 2. Why This IS Relevant to Our Stack (correcting the initial PASS)

### 2.1 We ship a diffusion LLM — D2F (Plan 066, Research 034)

D2F is our discrete diffusion forcing implementation: block-wise AR + intra-block parallel denoising + inter-block causal KV cache, behind the `dllm` feature. It is a real diffusion LLM research surface, not a toy.

**Critical clarification — two different "Diffusion Forcing" papers:**

| Paper | Authors | Mechanism | Our status |
|-------|---------|-----------|-----------|
| **D2F** (our Plan 066) | Wang et al. 2025 (arXiv 2508.09192, SJTU/UCSD) | Block-causal, **single monotonic noise schedule** across blocks (t₁<t₂<...<tₙ) | ✅ Shipped (`dllm` feature) |
| **Diffusion Forcing** (Latent Forcing's base) | Chen et al. 2025a (NeurIPS) | **Multi-time schedules** for AR+diffusion joint modeling | ❌ Not shipped |
| **Latent Forcing** (this paper) | Baade et al. 2026 | Multi-time multi-tokenizer, latent track leads pixel track | ❌ Not shipped |

Our D2F is the Wang line (single-schedule, block-causal). Latent Forcing builds on the Chen line (multi-time). **The multi-time axis is genuinely uncovered in our stack.** My initial "domain mismatch" claim conflated these two and was wrong.

### 2.2 The transferable primitive for D2F

Apply Latent Forcing's mechanism to D2F: **add a continuous latent track alongside the discrete token track, with the latent track on a leading (faster) noise schedule.** The denoised latent acts as a scratchpad that conditions later token denoising. Concretely:
- Token track: discrete mask-token diffusion (existing D2F)
- Latent track: continuous, e.g., a frozen encoder's output (DINOv2 analog for text = a frozen embedding model, or NextLat's belief-state MLP output, or HLA state)
- Two time variables, cascaded or variance-shifted schedule
- Joint training with weighted v-loss per modality

This is a **training recipe for discrete diffusion LLMs** — exactly riir-train's scope.

### 2.3 Why §3.5 modelless paths do NOT unblock this

Per the mandatory modelless-unblock protocol (research skill §3.5), checking all three paths before routing to riir-train:

1. **Freeze/thaw (path 1):** No. Freeze/thaw swaps a frozen snapshot; it cannot add a second diffusion track with its own time variable. The dLLM architecture itself must change (extra time-embedding MLP, extra token positions or additive embeddings, optional 2-expert output head).
2. **Raw/lora reader-writer hot-swap (path 2):** No. A deterministically constructed LoRA can correct systematic biases in existing weights, but cannot teach the model to consume a new modality track with a new time variable. The model has never seen two-time-variable conditioning; no closed-form weight correction introduces that capability.
3. **Latent-space correction (path 3):** No. Latent projection + sigmoid gating can steer existing outputs, but the multi-time mechanism requires the model to *generate* (denoise) the latent track jointly with tokens, not just project onto it. The paper's §4.4 confirms: inference-only trajectory reordering on a Multi-Schedule model underperforms a Single-Schedule model trained for that trajectory.

**Documentation of §3.5 failure:** All three paths fail because Latent Forcing's contribution is not a bias correction — it is a new joint training objective over two modalities with two time variables. The model must be trained to (a) denoise the latent track, (b) consume partially-denoised latent as conditioning for the token track, (c) handle two time variables in adaLN. No deterministic construction provides this; gradient descent through the joint loss is required.

**→ Genuine riir-train dependency.** Routing to riir-train.

### 2.4 Why the initial PASS was wrong (lesson recorded)

The initial verdict pattern-matched to Research 119 (PiD Pixel Diffusion Decoder) — a CV pixel-diffusion paper that was correctly PASS. But:
- **PiD** was about latent→pixel *decoding* (VAE decode + super-resolution). Purely CV.
- **Latent Forcing** is about *reordering the diffusion trajectory via multi-time scheduling* — the mechanism is domain-general. The paper itself cites Coconut (Hao et al. 2024, continuous latent reasoning for LLMs) as a direct analog.

The failure mode: I anchored on "pixel-space image generation" in the title and treated the paper as CV-only, missing that the *mechanism* (multi-time multi-tokenizer scheduling) transfers to any diffusion model including our D2F. The skill's mandatory latent-reframing step (§1.5) exists to prevent exactly this — I performed it but reached for weak reframings (HLA evolution, QGF fusion) instead of the obvious one: **D2F + latent track = multi-time discrete diffusion LLM**.

**Lesson:** When a paper's title says "image" but the mechanism is "diffusion trajectory reordering," and we ship a diffusion LLM, the first reframe to try is "apply the trajectory reordering to our diffusion LLM" — not "find a non-diffusion latent analog." The D2F stack is the obvious target; NextLat/HLA are weaker analogs.

---

## 3. Routing

| Repo | Role here | Action |
|------|-----------|--------|
| **riir-train** | **Primary** — training recipe for multi-time multi-tokenizer discrete diffusion LLMs | Create `.research/` note + plan. Adapt Latent Forcing's cascaded/variance-shift schedule to D2F's block-causal discrete setting. Two tracks: discrete tokens (existing D2F) + continuous latent (frozen text encoder or NextLat belief-state MLP output). Train jointly with per-modality v-loss. Benchmark: denoising quality / convergence steps vs single-schedule D2F baseline. |
| katgpt-rs | Secondary — weaker modelless fallback only | No primitive to ship. A speculative modelless angle (frozen belief-state drafter generates latent conditioning in one shot, fed as additive embedding to D2F) is out-of-distribution for an unmodified dLLM and not a GOAT candidate without riir-train support. Do NOT implement in katgpt-rs. |
| riir-ai / riir-chain / riir-neuron-db | — | No direct angle. The HLA/functor/shard runtimes already do "latent state conditions output" modellessly in richer forms (NextLat Plan 217, MUX-Latent Plan 238, latent_functor staging); the paper adds nothing to those non-diffusion paths. |

---

## 4. What riir-train Should Actually Try (concrete recipe)

For the riir-train note (not created in this session — this is a katgpt-rs-side routing verdict):

1. **Start from D2F (Wang 2025) block-causal dLLM** — our existing `dllm` feature base.
2. **Add a continuous latent track.** Options for the latent source:
   - Frozen text encoder (BERT-like, DINOv2 analog) — paper's default approach
   - NextLat belief-state MLP output (Research 192) — already in our stack
   - HLA per-position state — already in our stack, modelless
3. **Two time variables.** Add a second time-embedding MLP to the dLLM's adaLN (+0.5% params, per paper).
4. **Schedule: cascaded** (latent fully denoised first, then tokens) — paper's best single-schedule config.
5. **Loss: per-modality weighted** — discrete CE for tokens, continuous v-loss (or MSE on denoised latent) for latent track. Equal loss magnitude (paper's `λ` tuning).
6. **Cascaded-error mitigation:** small noise on latent during token steps (β ≤ 0.25), training only.
7. **Benchmark vs single-schedule D2F:** measure (a) denoising convergence steps, (b) final token quality (perplexity / task accuracy), (c) throughput. The paper's "order matters more than distillation" finding predicts the multi-time version should beat single-schedule D2F at equal compute.

**Risk:** D2F is micro-scale (6K params, vocab=27, block=16). The paper's gains are at ViT-L scale (459M params, ImageNet-256). Whether multi-time scheduling helps at micro-scale is the open research question — exactly what riir-train exists to answer.

---

## 5. GOAT / MOAT Assessment (revised)

| Criterion | Score | Reason |
|-----------|-------|--------|
| Relevant to our diffusion LLM stack | ✅ | D2F is our diffusion LLM; multi-time scheduling is genuinely uncovered |
| Modelless (katgpt-rs) | ❌ | Requires retraining; §3.5 paths exhausted |
| → riir-train routing | ✅ | Training recipe for discrete diffusion LLMs — squarely in riir-train scope |
| Super-GOAT | ❌ (deferred) | If riir-train proves the gain on D2F, THEN revisit for Super-GOAT (would connect D2F + NextLat + HLA pillars). Not claimable now. |

---

## 6. Verdict

**→ riir-train. Training method for multi-time multi-tokenizer discrete diffusion LLMs.**

**One-line reasoning:** Latent Forcing's core mechanism (multi-time scheduling with a leading latent scratchpad track) is domain-general, genuinely uncovered in our D2F (which is single-schedule Wang 2025, not multi-time Chen 2025a), and requires retraining the dLLM with two tracks + two time variables. §3.5 modelless paths all fail (not a bias correction — a new joint training objective). This is exactly riir-train's scope: a training recipe for the LLM stack.

**Honest correction:** The initial PASS verdict was wrong. It pattern-matched to Research 119 (PiD, a genuine CV-only PASS) on the basis of "pixel-space image generation" in the title, missing that the *mechanism* (diffusion trajectory reordering) transfers to our D2F diffusion LLM. The skill's mandatory latent-reframing step exists to prevent this; I performed it but reached for weak analogs (HLA, QGF) instead of the obvious D2F target. Lesson recorded in §2.4.

---

## 7. Paper Metadata

- **arXiv:** 2602.11401
- **Date:** Feb 11, 2026 (ICML 2026)
- **Authors:** Alan Baade, Eric Ryan Chan, Kyle Sargent, Changan Chen, Justin Johnson, Ehsan Adeli, Li Fei-Fei
- **Affiliation:** Stanford University
- **Code:** https://github.com/AlanBaade/LatentForcing
- **License:** CC-BY-SA 4.0
- **Backbone:** JiT (x-prediction, v-loss) + DINOv2 / Data2Vec2 latent track
- **Benchmark:** ImageNet-256, FID-50K 7.2 (unguided) / 2.48 (guided) — SOTA pixel-space DiT at compute scale
- **Key citation for our purposes:** Chen et al. 2025a "Diffusion Forcing: next-token prediction meets full-sequence diffusion" (NeurIPS) — the multi-time base that Latent Forcing extends, and the actual gap vs our Wang-2025-based D2F.
