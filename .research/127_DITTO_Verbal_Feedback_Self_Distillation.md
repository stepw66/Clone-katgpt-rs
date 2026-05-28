# Research 127: DITTO — Reinforcing Human Behavior Simulation via Verbal Feedback

> **Paper:** [Reinforcing Human Behavior Simulation via Verbal Feedback](https://arxiv.org/abs/2605.20506) — Sun et al., CMU + Microsoft, May 2026 (32 pages)
> **Date:** 2025-05-28
> **Related:** SDAR (R038), ROPD (R036), NPC Dialog Engine (riir-ai R006), Bradley-Terry (R040)
> **Verdict:** ⚠️ NO NEW PLAN — existing SDAR + ROPD already covers DITTO's mechanism

---

## Executive Summary

DITTO uses verbal feedback (textual critiques, not scalar rewards) as first-class signal in GRPO training. After each rollout, a judge provides verbal feedback → policy generates feedback-conditioned improved rollout → both optimized jointly via GRPO. At test time, no feedback needed — the policy has internalized it.

**Why we care:** Validates our existing SDAR + ROPD architecture. DITTO = SDAR (teacher-student self-distillation) + ROPD (rubric/verbal judge feedback) combined. No new algorithm needed.

**Key results (Qwen3-8B):**
- 36% improvement over base model across 10 human simulation tasks
- Exceeds GPT-5.4 on 6/10 tasks
- Verbal feedback > standard GRPO especially on subjective/behavioral tasks (+11-14 points)
- Preserves safety dimensions (secret-keeping) that scalar GRPO degrades

---

## Paper Core

### Problem

Standard RL for LLMs (RLHF, DPO, GRPO) reduces all feedback to scalar rewards. This works for verifiable domains (code, math) but fails for human behavior simulation where feedback is verbal, subjective, and multi-dimensional.

### Solution: DITTO

For each prompt x:

1. **Student rollout**: `y0 ~ πθ(·|x)` → scored `r0`
2. **Verbal feedback**: `(r0, h) = J(x, y0)` — judge returns scalar + structured text critique
3. **Teacher rollout**: `y1 ~ πθ(·|x, h)` → scored `r1` — conditioned on verbal feedback
4. **Joint GRPO**: Optimize over group `G(x) = {y0_i, y1_i}` with group-relative advantages
5. **Feedback loss**: Additional GRPO on feedback-conditioned rollouts only (`L_fb`)
6. **Total**: `L = L_group + L_fb`

### Key Insight: Feedback as Privileged Information

Verbal feedback h is available during training but NOT at test time. The policy internalizes the feedback-driven improvements into its base behavior. This is exactly the LUPI (Learning Using Privileged Information) paradigm from Vapnik.

---

## Mapping to Our Architecture

| DITTO Component | Our Equivalent | Status |
|-----------------|---------------|--------|
| Student rollout y0 | GRPO on-policy sampling | ✅ `loss_grpo.rs` |
| Verbal feedback judge J | ROPD Rubricator + LLM-as-Judge | ✅ ROPD (R036) + BT (R040) |
| Teacher rollout y1 (feedback-conditioned) | SDAR teacher branch (privileged context) | ✅ SDAR (R038) |
| Joint GRPO over {y0, y1} | SDAR `L_GRPO + λ·L_SDAR` | ✅ SDAR sigmoid-gated |
| Feedback loss L_fb | SDAR auxiliary loss | ✅ Covered |
| Multi-dimensional scoring | ROPD rubric scoring | ✅ Per-criterion pass/fail |
| LLM-as-Judge evaluation | Bradley-Terry pairwise ranking | ✅ R040 |

### Key Difference: SDAR > DITTO (for our case)

SDAR's sigmoid-gated token-level distillation is **strictly more general** than DITTO's joint GRPO:
- DITTO: uniform weighting over all tokens in joint group
- SDAR: per-token sigmoid gate `σ(β·Δt)` — trusts positive endorsements, softly attenuates negative rejections
- DITTO's advantage only shows on subjective tasks where verbal feedback provides richer signal than scalar rewards
- SDAR already handles this via ROPD rubrics as the "verbal" signal source

---

## Validation of Existing Choices

### 1. ROPD Rubric Feedback = DITTO Verbal Feedback ✅

DITTO proves verbal/structured feedback > scalar rewards for behavioral tasks. ROPD already provides this via rubric-based scoring (multi-criteria pass/fail → weighted reward). The DITTO paper's Table 2 shows feedback is adaptive to failure modes — exactly what ROPD's per-criterion gap targeting does.

### 2. SDAR Teacher-Student = DITTO Feedback-Conditioned Rollout ✅

DITTO's feedback-conditioned teacher rollout y1 is architecturally identical to SDAR's privileged-context teacher branch. The "privileged context" in DITTO is verbal feedback; in SDAR it's retrieved skills. Both are LUPI — available at training, not inference.

### 3. Safety Preservation Validates ConstraintPruner Design ✅

DITTO's finding that verbal feedback preserves safety dimensions (secret-keeping) while scalar GRPO degrades them validates our ConstraintPruner + WASM validator approach. Scalar rewards lose information about why something is wrong; structured feedback preserves it.

### 4. NPC Dialog Engine Direction Validated ✅

DITTO's core thesis — making AI more human-like through structured feedback rather than scalar rewards — validates Pillar 3 (NPC Dialog Engine). Our approach (WASM FSM guardrails + semantic retrieval + LoRA personality) follows the same principle: structured constraints > scalar scoring for behavioral fidelity.

---

## What DITTO Adds (Minor)

| Insight | Relevance | Action |
|---------|-----------|--------|
| Verbal feedback > scalar for subjective tasks | Validates ROPD over vanilla GRPO | None — already implemented |
| Feedback accelerates learning (faster convergence) | SDAR + ROPD should show same effect | Monitor in training curves |
| L_fb (feedback-only GRPO loss) | SDAR's L_SDAR already covers this | None — already implemented |
| SOUL benchmark (10 tasks, 6 categories) | Domain-specific to social simulation, not games | Not applicable |
| Secret-keeping dimension preserved | Validates WASM validator constraint enforcement | None — already implemented |

---

## SOUL Benchmark Assessment

The SOUL benchmark targets social simulation domains (Theory of Mind, persona, user simulation) that are **not directly applicable** to our game AI domain. Our game arenas (Bomber, Go, FFT, Monopoly) already serve as domain-specific evaluation environments with GOAT proofs.

| SOUL Category | Our Equivalent | Overlap |
|---------------|---------------|---------|
| Theory of Mind | Game opponent modeling (Fourier MCTS) | Low — games ≠ social ToM |
| Character Role Play | NPC Dialog Engine (Pillar 3) | Medium — NPC character fidelity |
| Social Skill | Arena self-play evaluation | Low — social norms ≠ game strategy |
| Learner Simulation | None | None |
| User Simulation | Frame-Sampling player behavior | Low |
| Persona Simulation | NPC personality LoRA | Medium — persona consistency |

---

## Verdict

**⚠️ NO NEW PLAN.** DITTO validates existing architecture choices (SDAR + ROPD + ConstraintPruner) without introducing new algorithms or mechanisms we don't already have.

**Why no plan:**
1. SDAR's sigmoid-gated distillation is strictly more general than DITTO's joint GRPO
2. ROPD's rubric scoring already provides multi-dimensional verbal feedback
3. Bradley-Terry pairwise ranking already covers LLM-as-Judge evaluation
4. SOUL benchmark is social simulation, not game AI
5. The "verbal feedback > scalar" insight is already operationalized in our ROPD pipeline

**Cross-reference updates:**
- riir-ai R006 (NPC Dialog Engine): DITTO validates structured feedback approach for NPC fidelity
- Decision Matrix (27): Add DITTO as "What Is NOT a Pillar" entry — validates existing SDAR+ROPD, no new mechanism
