---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, and latent-space operations across the katgpt-rs / riir-ai / riir-train repo trio. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the 3-repo commercial strategy, modelless-first constraint, latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

> **Pivot (issue 004, 2026-06-14):** Training-method research lives in `riir-train`. This repo (`katgpt-rs`) and `riir-ai` ship **freeze/thaw runtime + self-learn/adaptive NPCs + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop → `riir-train/.research`. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, or a modelless inference primitive → distill here.

## When to use this skill

Activate when the user (or you) are doing any of:

- Reading / fetching / summarizing an ML, AI, or systems paper (arxiv, PDF, blog).
- Deciding which of the 3 repos a paper or idea belongs in.
- Creating a new `.research/NNN_*.md` note or `.plans/NNN_*.md` plan.
- Implementing a modelless inference primitive (pruner, bandit, router, speculative decode, KV-cache op, sparse attention, quantization-aware inference).
- Designing freeze/thaw snapshot cycles, adapter hot-swap, or runtime adapter routing.
- Designing latent-to-latent operations (dot-product projection, sigmoid gating, manifold geometry, spectral methods on activations).
- Designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, zone attention, emergent social/economic behavior).

Do NOT activate for: pure refactor tasks, bug fixes with no research angle, or ordinary feature work that doesn't touch the research/plans folders.

## Repos (siblings under the same parent)

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game IP, no chain IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, chain, game systems.
- `riir-train/` — private training vault. Adapter training, optimizers, loss functions. Out of scope for this workflow — just note "→ riir-train" and stop.

Always reference files with project-relative paths (e.g. `katgpt-rs/.research/238_*.md`, `riir-ai/.plans/NNN_*.md`). The agent can `read_file` these directly.

## Primary focus (distill HERE in katgpt-rs / riir-ai)

- **Latent-to-latent operations** — anything that stays in embedding/latent space: dot-product projections, cosine similarity retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decoding to tokens then re-encoding.
- **Freeze/thaw patterns** — versioned weight snapshots, atomic hot-swap, lock-free read paths, BLAKE3/commitment-checked adapter reload, per-entity personality divergence via snapshot versioning.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — all inference-time, zero training).
- **Self-learn / adaptive CoT** — runtime curiosity, entropy-driven exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No LLM training, no backprop through weights — runtime self-improvement via latent-space updates is welcome.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quantization-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs each with independent latent state, real-time latency budgets (20Hz tick, plasma/hot tier), spatial partitioning + fog-of-war, emergent social/economic behavior (factions, trade routes, reputation), zone-level attention routing, crowd-scale curiosity/exploration signals. Latent ops must batch across many entities; raw sync must stay bit-identical for deterministic replay/anti-cheat.

## Redirect to riir-train (do NOT distill here)

- LoRA / OFT / SPEFT / IA3 / QLoRA / ManifoldE / BAKE / GPart / MSA / Dendritic and all adapter-**training** methods.
- Training optimizers (Muon, Adam variants, symmetry-compatible optimizers).
- Training loss functions, curricula, distillation recipes.
- Quantization-aware **training** (quantization-aware **inference** stays here).
- DPO / GRPO / SFT / RL **training** pipelines (runtime GRPO self-play stays in `riir-ai` — it updates latent state, not weights).
- Anything that requires backpropagation through base weights.

If a paper is training-only → note "→ riir-train" in one line and stop. Do not create files in this session for it.

## Distillation targets (3-repo strategy)

Per `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md`:

| Repo | Role | What lands here |
|------|------|-----------------|
| `katgpt-rs` (public, MIT) | Engine — modelless inference framework | Generic primitives: ConstraintPruner traits, bandits, DDTree, speculative decode, sparse attention kernels. **No game IP, no chain IP.** |
| `riir-ai` (private) | Game product — freeze/thaw runtime, self-learn, chain | Runtime IP: `LoRAWeightVersion`, `LoRAHotSwap`, `dispatch_lora_merge`, `TrainingProvider` trait, routing, game systems, neuro-symbolic chain. |
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** Out of scope for this workflow — just note "→ riir-train" and move on. |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/src/` (or `katgpt-rs/crates/katgpt-rs-core/`)
- **Runtime/game/chain** → `riir-ai/.research/` + `riir-ai/crates/`
- **Training-only** → note the redirect, do not create files in this session

## Workflow

### 0. Read & classify the paper

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}` (per AGENTS.md). Ask: *is the value in the training loop, or in a latent-space / inference / routing insight?* If training-only → note "→ riir-train", stop.

### 1. Distill fundamentally

Don't direct-map the paper. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup.

- `grep` `katgpt-rs/.research/` and `katgpt-rs/.plans/` for related prior work (keyword, paper title, author).
- `grep` `riir-ai/.research/` and `riir-ai/.plans/` likewise.
- Verdict by the commercial strategy doc (`003_*.md`): **Super-GOAT** > GOAT > Gain > Pass (see §Verdict tiers below).
- Create research `.md` at the right repo (see table above).

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits, e.g. `239_`, `240_`). Check the folder first — numbers may be non-contiguous; pick the next free slot.

**Research note format** (see `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` for a canonical example):

```markdown
# Research NNN: <Title>

> **Source:** <paper title + arxiv link + authors + date>
> **Date:** YYYY-MM-DD
> **Status:** Active | Done | Shelved
> **Related Research:** NNN (short note), ...
> **Related Plans:** NNN (short note), ...
> **Cross-ref (riir-ai):** Research NNN, Plan NNN   ← only if cross-repo
> **Classification:** Public | Private   ← katgpt-rs notes are always Public

---

## TL;DR

<2-4 sentences: the distilled primitive, why it matters here, what it unblocks>

**Distilled for katgpt-rs (modelless, inference-time):**
<the transferable insight, stripped of training setup>

---

## 1. Paper Core Findings
...
## 2. Distillation
...
## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/**. Plans → both repos as needed. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | One-line note. No files created in this session. |

**One-line reasoning required for each verdict.** For Super-GOAT: state the selling point explicitly.
```

### 1.5. Novelty gate — is this Super-GOAT?

Before planning, score novelty. Ask all four:

1. **No prior art?** Grep `.research/` across all repos — does any existing note already cover this mechanism? If yes → not novel, it's a Gain at best.
2. **New class of behavior?** Not better numbers, but something no incumbent can do (a new capability, not an optimization).
3. **Product selling point?** Can you finish the sentence: "Our NPCs/systems do X that no competitor can"? If you can't → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems (check connection map in `.research/`). Solo novelty without integration = GOAT, not Super-GOAT.

**If YES to all 4 → verdict = Super-GOAT.** Mandatory outputs:
1. **Open primitive** → `katgpt-rs` (generic math, no game semantics).
2. **Architectural GUIDE** → `riir-ai/.research/NNN_*.md` (the private selling-point doc). The guide MUST include:
   - TL;DR with commercial value (the selling point in one sentence)
   - Distilled primitive (how the mechanism works modellessly)
   - Connection map (which existing systems it multiplies)
   - Latent vs raw boundary (what crosses sync, what stays local)
   - What stays private vs open
   - Validation protocol (how to prove it's Super-GOAT, not just hype)
   - Implementation priority table (P0–P3)
3. **Plan(s)** → `katgpt-rs/.plans/` (open) and/or `riir-ai/.plans/` (private runtime).

**If NO to any → proceed to GOAT/Gain verdict.** Plan only, no guide.

> **Rule:** Super-GOAT ideas are the private IP moat. The open primitive is the adoption hook; the riir-ai guide is the selling point. Never ship the guide publicly. Never skip the guide for a Super-GOAT — that's losing the knowledge.

### 2. If gain (or GOAT), plan it

Add plan `.md` to `katgpt-rs/.plans/` (modelless) and/or `riir-ai/.plans/` (runtime/game/chain). Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done). **Never** plan into `riir-train` from this workflow.

> Super-GOAT plans should be created AFTER the riir-ai guide. The guide is the strategy; the plan is the execution.

**Plan format** (see `katgpt-rs/.plans/271_attention_matching_compaction.md` for a canonical example):

```markdown
# Plan NNN: <Title>

**Date:** YYYY-MM-DD
**Research:** [katgpt-rs/.research/NNN_*.md](../.research/NNN_*.md)
**Source paper:** [arxiv ID.NNN](https://arxiv.org/abs/ID) — <short cite>
**Target:** `katgpt-rs/src/<module>/` (new module) + Cargo feature `<feature_name>`
**Status:** Active — Phase N <state>

---

## Goal

<one paragraph: what ships, what it enables, GOAT gate>

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** <concrete, verifiable task>
- [ ] **T1.2** ...
```

**GOAT gate rule** (AGENTS.md): every plan that introduces a new technique must have a feature flag and a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 4. Search if curious

Keyword search arxiv:

```
https://r.jina.ai/https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={KEYWORD}&terms-0-field=abstract&classification-computer_science=y&classification-mathematics=y&classification-physics_archives=all&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&size=50&order=-announced_date_first
```

Good keywords: `latent space routing`, `adapter hot-swap`, `inference-time composition`, `spectral pruning`, `sigmoid gating`, `snapshot consistency`, `lock-free weight swap`.

## Constraints (non-negotiable)

1. **Modelless first** — inference-time only. No LLM training, no backprop through base weights. Closest to "training" allowed: freeze/thaw snapshot cycles and latent-space direction-vector updates at runtime.
2. **Latent-to-latent preferred** — operate in embedding/latent space as long as possible. Decode to tokens or project to raw scalars only at the boundary. Use dot-product + **sigmoid** (never softmax) for projections onto learned direction vectors. Semantic domain (emotion, mood, curiosity, style) → latent. Physical domain (position, HP, wallet balance) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — the only weight mutation allowed at runtime is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked). Never mutate weights in-place during inference. If a paper needs gradient updates, redirect to riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. These update latent state / direction vectors / routing tables, NOT base weights.
5. **3-repo discipline** — katgpt-rs (public engine) → riir-ai (private runtime/game/chain) → riir-train (private training). Keep the commercial strategy intact. Training know-how never leaks to katgpt-rs.
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-allocation hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain (latency, quality, or security). For latent ops: show the projection preserves ranking. For freeze/thaw: show readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive dispatch. Plasma (µs, CPU/SIMD) → Hot (sub-ms, GPU) → Warm/Cold (ms+, GPU/ANE). Latent ops that fit in L1 cache stay on SIMD; manifold ops that need batched matmul go to GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — aim for perf on game side (plasma/hot latency budget) AND security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state that crosses the sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding vector.

## Latent vs raw space rules (critical for game AI)

Reinforce these when designing game systems or chain state:

- **Physical domain** (position, velocity, HP, wallet balance): MUST remain raw exact values. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic domain** (emotion, mood, curiosity, style, habit): SHOULD operate in latent space via dot-product + sigmoid onto learned direction vectors.
- **Social domain** (encounters, relationships, factions): SHOULD produce KG triples from proximity in latent/embedding space, not from raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum commit → Cold tier, it MUST be raw and deterministic. If data is consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-allocation, gateable by feature flag, and not introduce sync dependency.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat validation.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone-level KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent behavior, not a bug.

## Anti-patterns (redirect to riir-train, do not implement here)

- "Train a LoRA adapter to do X" → riir-train
- "Fine-tune with method Y" → riir-train
- "Optimizer Z improves convergence" → riir-train
- "Distillation recipe from teacher to student" → riir-train
- "Quantization-aware training" → riir-train (quantization-aware **inference** stays here)
- "DPO/GRPO/SFT/RL training pipeline" → riir-train (runtime GRPO self-play stays in riir-ai — it updates latent state, not weights)

## Cross-references (read on demand)

- `katgpt-rs/.contexts/optimization.md` — perf rules (zero-alloc, SIMD, rayon, caching)
- `katgpt-rs/.contexts/ibraheemdev-papaya-v0.2.3-examples.md` — papaya lock-free hashmap usage
- `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md` — 3-repo strategy source of truth
- `katgpt-rs/.research/004_LoRA_Architecture_Verdict.md` — LoRA / validator terminology
- `katgpt-rs/.research/005_Artifact_Definition.md` — artifact terminology
- `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` — canonical research-note example
- `katgpt-rs/.plans/271_attention_matching_compaction.md` — canonical plan example

## TL;DR

This skill packages the katgpt-rs research workflow: read paper → classify (training? → riir-train, stop) → distill the latent/inference/routing primitive → **novelty gate** (Super-GOAT? → open primitive + private riir-ai guide; else GOAT/Gain → plan only) → implement behind feature flag → benchmark → promote GOAT or demote loser. Hard constraints: modelless-first, latent-to-latent with sigmoid (never softmax), freeze/thaw over fine-tuning, 3-repo commercial discipline, raw scalars at the sync boundary. **Super-GOAT = private moat; never skip the riir-ai guide.**
