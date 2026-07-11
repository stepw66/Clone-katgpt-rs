# Research 343: System-1.5 Reasoning — Depth + Step Shortcuts

> **Source:** [System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts](https://arxiv.org/pdf/2505.18962) — Xiaoqiang Wang, Suyuchen Wang, Yun Zhu, Bang Liu (Université de Montréal / Milia / Canada CIFAR), arXiv:2505.18962v3, May 2025.
> **Date:** 2026-06-29
> **Status:** Done
> **Related Research:** 194 (DDTree), 218 (Breakeven Router), 241 (SwiR switch), 266 (FPRM damped halting), 282 (LoopCoder-V2 gain/cost halting), 286 (depth-invariance diagnostic), 325 §7.2 row G1 (survey gap candidate — **misclassified**, see §4 below)
> **Related Plans:** none (training-only, no plan created)
> **Classification:** Public

---

## TL;DR

**Verdict: Pass — training-only.** The headline "20× faster GSM8K with CoT accuracy preserved, no backbone change" is **misleading without qualification**: "no backbone change" means the vanilla Transformer weights are frozen during *stage 2*, but stage 2 still **trains a per-layer router-adapter `ϕ` via gradient descent** (early-exit loss + NLL, eq 10–13), and stage 1 **fine-tunes the student Transformer `θ_student` via hidden-state MSE distillation** from a CoT-trained teacher (eq 6–9). Both stages require backpropagation. The 20× speedup is a property of the *trained* model's inference, not a modelless inference-time technique applied to a frozen checkpoint. **→ riir-train** (out of scope for the modelless workflow per skill §"Redirect to riir-train").

**Distilled for katgpt-rs (modelless, inference-time):** nothing new ships from this paper. The modelless *cousin* — analytical depth+step routing on a frozen checkpoint — is **already covered by our corpus** under different vocabulary: FPRM damped fixed-point halting (266), LoopCoder-V2 gain/cost halting (282), depth-invariance diagnostic (286), Breakeven Complexity Router (218/250), SwiR switch-thinking (241). Each of those provides a *modelless* depth/step routing signal (residual decay, gain/cost scissors, magnitude accumulation, breakeven N*, entropy switch) that does not require the paper's atomic-thought DAG criticality training labels.

---

## 1. Paper Core Findings

### 1.1 The two shortcuts (the architecture)

System-1.5 introduces two inference-time shortcuts over a vanilla Transformer whose weights are inherited from a stage-1 distilled student:

**Depth shortcut (DS, vertical).** Each Transformer layer `l` is augmented with a router-adapter module `(R_l, g_l)`. At inference, for token `t`:

```
R_l(h_{l-1,t}) → sigmoid → w        (router confidence, eq 2)
if w > λ_depth:   h_{l,t} = g_{l-1}(h_{l-1,t})      (early exit via adapter, eq 3)
else:             h_{l,t} = f_l(h_{l-1,t})           (continue through full layer)
```

`R_l` is an FFN + sigmoid; `g_l` is a lightweight adapter branch. Non-critical tokens exit shallow; critical tokens continue deep.

**Step shortcut (SS, horizontal).** If a decoding step halts at intermediate layer `l`, its hidden state `h_{l,t}` is copied directly to step `t+1` at the *same* layer (eq 4–5), skipping the reprocessing from layer 0 that a standard Transformer would do. This only works because stage 1 trains the student to operate in latent space (continuous thought) — the step shortcut is structurally modelless but depends on the stage-1 training product.

### 1.2 The two-stage training (the actual contribution)

**Stage 1 — Language-to-latent alignment (eq 6–9).** Trains `θ_student` via:
- NLL on final-answer generation (eq 8),
- MSE between student's last-layer hidden states and a CoT-trained teacher's stop-gradient'd last-layer hidden states (eq 6, the consistency loss),
- with `α` weighting (eq 9).

This is **gradient descent on the student Transformer's parameters**. Without it, the student has no latent-space reasoning capability and the step shortcut is meaningless.

**Stage 2 — Shortcut learning (eq 10–13).** Freezes `θ_student`, inserts router-adapter `ϕ`, trains `ϕ` via:
- **Early-exit loss** (eq 10): MSE between `ϕ`'s composed hidden states and the student's hidden states, weighted per `(layer, step)` by `e_{l,t}` (eq 11),
- **Criticality weighting** (eq 11): atomic-thought DAG decomposition labels each CoT step `t` as critical (`c_t = 1`) or non-critical (`c_t = 0`); non-critical steps get an early-exit weight that *grows with depth*, critical steps get one that *shrinks with depth*. The router thus learns to exit non-critical steps early and push critical steps deep.
- NLL on final-answer generation (eq 12),
- with `β` weighting (eq 13).

This is **gradient descent on the router-adapter parameters**, using a discrete labeling scheme (atomic-thought DAG criticality) as the supervision signal.

### 1.3 Empirical headlines

| Method | GSM8K Acc | #Steps | FLOPs reduction | Wall-clock speedup |
|---|---|---|---|---|
| CoT | 46.94 | 26 | — | — |
| Coconut | 36.75 | 2 | 11.98× | 11.98× |
| CODI | 43.78 | 2 | 13.37× | 13.37× |
| **System-1.5** | **46.66** | **2** | **1.95×** per step | **20.27×** |

System-1.5 matches CoT accuracy while compressing to 2 decoding steps and exiting most tokens at shallow layers. On StrategyQA the speedup reaches 55.65×.

### 1.4 The "no backbone change" claim, qualified

The paper's framing — "no backbone change, 20× faster" — is technically true in the narrow sense that the vanilla Transformer weights are frozen during stage 2. But:
1. **Stage 1 changes the backbone** (fine-tunes `θ_student` via MSE distillation).
2. **Stage 2 adds new trainable parameters** (the per-layer router-adapter `ϕ`) whose weights are derived via gradient descent from atomic-thought DAG labels.

The 20× speedup is a property of the *combined* trained artifact (`θ_student` + `ϕ`), not of any inference-time operation on a generic frozen checkpoint.

---

## 2. Distillation — §3.5 modelless-unblock protocol (MANDATORY before deferral)

Per skill §3.5, before redirecting any mechanism to riir-train, exhaust the three modelless-unblock paths.

### 2.1 Path 1 — Freeze/thaw snapshot correction

**Question:** can a frozen snapshot state, thawed at inference, fix the issue?

**Answer: NO.** The router-adapter `ϕ` *is* the trained artifact. There is no "corrected snapshot" to thaw — the router weights must be learned from the atomic-thought DAG criticality labels. Freeze/thaw is a *deployment* mechanism for already-trained weights; it does not produce them.

### 2.2 Path 2 — Raw/lora reader-writer hot-swap (deterministic construction)

**Question:** can a deterministically-constructed (not trained) reader/writer adapter reproduce the paper's behavior?

**Answer: NO — not as the paper specifies.** The router `R_l = FFN + sigmoid` is a generic function approximator; its weights encode the mapping `hidden_state → criticality_score`. The atomic-thought DAG labels are discrete (`c_t ∈ {0,1}` per step), and the paper provides **no closed-form construction** mapping a hidden state to a criticality score. Any deterministic construction (e.g., "exit early when `‖h_l − h_{l-1}‖ < ε`") would be a *new invention* not validated by the paper.

**Important:** this is exactly the situation the AC-Prefix G1 lesson (AGENTS.md) warns about — "systematic, characterizable biases are modelless-correctable candidates." But here the bias is not systematic/characterizable in closed form: the paper's criticality signal comes from an external decomposition tool (atomic-thought DAG), not from a measurable property of the hidden state. There is no closed-form `f(h) → criticality` the reader-LoRA could encode without learning.

### 2.3 Path 3 — Latent-space correction (dot-product projection + sigmoid gate)

**Question:** can the bias be corrected by projecting the latent state onto a correction direction and gating the output?

**Answer: NO — not as the paper specifies, for the same reason as Path 2.** A latent projection `σ(⟨h, d_criticality⟩)` requires a *learned* direction vector `d_criticality`. The paper does not provide one in closed form; it learns the equivalent (the FFN weights of `R_l`) via gradient descent on the DAG labels.

### 2.4 Verdict of §3.5 check

All three modelless-unblock paths fail *for the paper as written*. The router-adapter weights require gradient descent on a discrete labeling scheme that has no closed-form hidden-state mapping. **Genuine riir-train dependency.**

### 2.5 The modelless cousin (NOT from this paper)

A *different*, modelless depth+step routing primitive can be constructed from analytical hidden-state signals — but this is a **separate invention**, not a distillation of System-1.5, and is **already covered by our corpus**:

| Modelless depth/step routing signal | Where it ships | Paper analog |
|---|---|---|
| Damped fixed-point residual halt (`‖f(z)−z‖ < ε` with patience-decay η) | FPRM (266), `fpopt_halt` feature | "exit when router confident" — but FPRM uses residual, not a trained FFN |
| Gain/cost loop halt (marginal gain < marginal cost) | LoopCoder-V2 (282), `GainCostLoopHalter` | "exit non-critical, continue critical" — but LoopCoder-V2 derives criticality from output-shift + effective-rank, not a DAG |
| Magnitude-accumulation diagnostic (`d‖h_t‖/dt`) | Depth-Invariance (286), `DepthInvarianceDiagnostic` | explains *why* deep chains drift — root-cause for any depth-routing decision |
| Breakeven N* cost-amortization routing | Breakeven Router (218/250), `BreakevenBandit` | "which compute tier" — orthogonal axis (tier vs depth) |
| Entropy-driven explicit↔latent switch | SwiR (241), `SwiRController` | "fast vs slow mode" — orthogonal axis (modality vs depth) |

**None of these requires the atomic-thought DAG training signal.** Each provides a modelless depth/step routing criterion. If a future plan wants to fuse them into a unified "analytical depth+step shortcut" primitive, it should cite FPRM + LoopCoder-V2 + depth-invariance + Breakeven as the ancestors — **not System-1.5**, which contributes only the training recipe.

---

## 3. Latent-space reframing (mandatory per skill)

Per skill §1.4, re-cast the paper's mechanism as a latent-to-latent op on the seven Super-GOAT factory modules.

### 3.1 The reframing that would qualify

The *step shortcut* (copy `h_{l,t}` → `h_{l,t+1}`) is genuinely a latent-to-latent operation: it operates entirely in hidden-state space, never decoding to tokens. Re-cast on:
- **HLA** (`katgpt-core/src/sense/`): "carry-forward" of per-NPC belief state across ticks — already shipped as the recurrent update in `evolve_hla`. The step shortcut is structurally identical to HLA's leaky-integrator persistence, just at finer granularity (per-step vs per-tick).
- **latent_functor** (`riir-ai/crates/riir-engine/src/latent_functor/`): the step shortcut is a *zero-cost functor application* (identity carry) gated by the router. Maps to `zone_gating.rs` (router = zone gate, step shortcut = bypass functor body when gate closed).
- **cgsp_runtime**: the depth shortcut maps to per-NPC curiosity-driven compute allocation — spend deeper compute on curious NPCs, shallow on bored ones. But our curiosity signal is runtime-derived (entropy, prediction error), not trained.

### 3.2 Why this does not produce a Super-GOAT

The latent reframing lands cleanly on **existing machinery**: HLA persistence, functor gating, runtime curiosity. The paper's *novelty* (the trained router-adapter + atomic-thought criticality) does not survive the modelless constraint. What remains after stripping the training is the step-shortcut-as-identity-carry, which is **already shipped** under the HLA / functor / micro_belief vocabulary.

Per skill §1.4: *"If your fusion idea only touches adapter routing / KV compression / speculative decode without a latent-state reframing, you are likely in GOAT territory."* Here the inverse holds: the latent-state reframing is *so clean* that it maps 1:1 onto already-shipped kernels, leaving no novelty.

---

## 4. Correction to Research 325 §7.2 row G1

**The survey misclassified this paper.** Research 325 §7.2 row G1 lists System-1.5 as a "MODELLESS-CANDIDATE GAP" with the rationale: *"Pure inference-time routing of vertical (depth) + horizontal (step) compute."*

This is incorrect. The same survey's §3.1 (the taxonomy table) correctly classifies System-1.5 in the **"Training-induced recurrence"** sub-family alongside Coconut, CODI, CCOT, PCCOT, Pause/Filler/Planning tokens, and Lightthinker. And §7.3 of the same survey correctly routes Coconut / CODI / CCOT to riir-train with the note: *"Compressed reasoning training — VQ-VAE / self-distillation / gist-token training objectives."*

System-1.5's training recipe (two-stage distillation: hidden-state MSE + early-exit loss with DAG-derived criticality labels) is in the **same training-only family** as Coconut/CODI/CCOT. The §7.2 row G1 classification appears to have been taken from the paper's abstract framing ("no backbone change") without verifying that stage 2 still trains the router-adapter via gradient descent.

**Recommended correction (for a future Research 325 follow-up):**
- Move System-1.5 from §7.2 row G1 to §7.3 (training-time gaps → riir-train).
- Update §7.5 action item G1 from *"highest-priority gap, recommend standalone distillation session"* to *"resolved 2026-06-29 (Research 343): training-only, redirected to riir-train; modelless cousin already shipped under FPRM/LoopCoder-V2/depth-invariance vocabulary."*

This correction does **not** invalidate the rest of Research 325 — the other seven §7.2 gap candidates (G2–G8) remain plausible modelless candidates pending their own distillation sessions.

---

## 5. Verdict

**Pass.**

**One-line reasoning:** Both training stages require gradient descent (stage 1: student transformer MSE distillation; stage 2: router-adapter early-exit loss with atomic-thought DAG criticality labels); the §3.5 modelless-unblock protocol fails for all three paths because the criticality signal has no closed-form hidden-state mapping; the modelless cousin (analytical depth+step routing) is already shipped under FPRM (266) / LoopCoder-V2 (282) / depth-invariance (286) / Breakeven (218) vocabulary.

**Why not Super-GOAT:**
- Q1 (no prior art?): **FAIL** — modelless depth/step routing is shipped (FPRM, LoopCoder-V2, depth-invariance, Breakeven, SwiR).
- Q2 (new class of behavior?): **FAIL** — paper's class (adaptive depth+step compute) is already a shipped capability class; the paper adds a training recipe.
- Q3 (product selling point?): **N/A** — paper is training-time; any selling point belongs to riir-train.
- Q4 (force multiplier?): **N/A** — training-only.

**Why not GOAT / Gain:** no modelless primitive to benchmark; the paper's value is its training method, which is → riir-train.

**Why not silent drop:** Research 325 §7.2 row G1 explicitly flagged this as the highest-priority modelless-candidate gap and §7.5 recommended this distillation session. This note documents the verification result and the §7.2 misclassification so future sessions do not re-evaluate.

---

## 6. Action items

- [x] **T1:** Verify paper is training-time (this note §1.2). DONE.
- [x] **T2:** Run §3.5 modelless-unblock protocol (this note §2). DONE — all three paths fail.
- [x] **T3:** Vocabulary-translate paper terms to codebase equivalents and grep for prior art (this note §2.5 table). DONE — five cousins found.
- [x] **T4:** Latent-space reframing (this note §3). DONE — lands on shipped HLA/functor/cgsp machinery.
- [-] **T5 (deferred):** Research 325 §7.2 row G1 correction. File a one-line follow-up issue or addendum to Research 325 §7.5 noting the reclassification. Not blocking; tracked here.
- [ ] **T6 (optional, NOT this paper):** If a future plan wants a unified "analytical depth+step shortcut" primitive fusing FPRM + LoopCoder-V2 + depth-invariance + Breakeven, file it as a fresh `.research/NNN_*.md` note citing those four ancestors (NOT System-1.5). Out of scope for this session.

---

## 7. Cross-references

- **Training recipe →** `riir-train/.research/` (out of scope for this workflow; if pursued, the two-stage distillation + atomic-thought DAG criticality labeling is the transferable training know-how).
- **Modelless depth-routing cousins →** Research 266 (FPRM `fpopt_halt`), 282 (LoopCoder-V2 `GainCostLoopHalter`), 286 (depth-invariance `DepthInvarianceDiagnostic`), 218/250 (Breakeven `BreakevenBandit`).
- **Modelless mode-switching cousin →** Research 241 (SwiR `SwiRController`).
- **Survey context →** Research 325 §7.2 row G1 (misclassified — see §4 above), §7.3 (correct sibling classification), §7.5 action item G1 (resolved by this note).

---

## TL;DR

**Verdict: Pass — training-only, → riir-train.** System-1.5's headline "20× faster, no backbone change" is technically true but misleading: stage 1 fine-tunes the student Transformer via hidden-state MSE distillation, and stage 2 trains a per-layer router-adapter via early-exit loss with atomic-thought DAG criticality labels. Both stages use gradient descent. The §3.5 modelless-unblock protocol fails for all three paths (freeze/thaw, raw/lora hot-swap, latent projection) because the criticality signal has no closed-form hidden-state mapping. The modelless *cousin* (analytical depth+step routing) is already shipped under FPRM (266) / LoopCoder-V2 (282) / depth-invariance (286) / Breakeven (218) vocabulary. Research 325 §7.2 row G1 misclassified this paper as a modelless candidate; §3.1 of the same survey correctly classifies it as "Training-induced recurrence" alongside Coconut/CODI/CCOT, which §7.3 correctly routes to riir-train. Recommended follow-up: move G1 from §7.2 to §7.3 in a future Research 325 addendum. No files created other than this note; no plans; no open primitive; no private guide.
