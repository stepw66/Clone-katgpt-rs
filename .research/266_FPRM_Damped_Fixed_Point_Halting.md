# Research 266: FPRM — Damped Fixed-Point Halting for Looped Transformers

> **Source:** "Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers" — Movahedi, Milovanović, Feigin, Theus, Hofmann, Boeva, Rusch, Orvieto (ELLIS Institute Tübingen / ETH Zurich / Liquid AI). [arXiv:2606.18206](https://arxiv.org/abs/2606.18206). 16 Jun 2026.
> **Reference impl:** https://github.com/sajad-movahedi/fprm (linked from paper)
> **Date:** 2026-06-18
> **Status:** Done
> **Related Research:** 035 (Attractor / DEQ comparison), 073 (LT2 Looped), 079 (EqR Equilibrium Reasoners), 097 (Training-Free Looped — failed-strategy table), 148 (Hydra adaptive depth), 228 (TwinProp dendritic adaptive compute), 255 (CLR test-time scaling), **265 (CoFRe/FP-MGM — sibling paper, written same day)**
> **Related Plans:** 108 (LT2 Looped — ships `LoopMode::WeightShared`), 119 (EqR Convergence Selector — breadth scaling, deferred), 136 (TF-Loop — ships `LoopMode::TrainingFree` with K-stage RK β=0.5), 152 (Newton-Schulz cubic fixed-point), 165 (Hydra adaptive budget), 276 (MicroRecurrentBeliefState — `AttractorKernel` null result on random init), 283 (Self-Advantage Gate on HLA — closest residual-halt cousin), 284 (Runtime CLR depth-tier scaling)
> **Classification:** Public

---

## TL;DR

FPRM trains a **non-hierarchical** looped Transformer (no HRM/TRM H-step + L-step machinery) that iterates a weight-tied block `z_{i+1} = f_θ(z_i; x)` until the hidden state converges to a fixed-point `z⋆ = f_θ(z⋆; x)`. The convergence itself is the halting signal — no external ACT head, no learned halting module. Two architectural moves make the loop trainable at deep effective layers: (a) **pre-norm + residual scaling** (α₁ inside the layer, α₂ for iteration-wise input re-injection) replaces the conventional post-norm that prior looped models (HRM/TRM) used only to keep activations bounded, and (b) a **damped fixed-point optimizer** (Algorithm 1, FPOpt) that decays step-size η by geometric factor γ whenever the residual stops improving for P consecutive steps (patience), breaking oscillation around the fixed-point.

**Distilled for katgpt-rs (modelless, inference-time):** ~80% of the inference surface is **already shipped** (per Research 265's sibling analysis). `LoopMode::WeightShared { loop_count }` (Plan 108) *is* the weight-tied looped block; `LoopMode::TrainingFree` (Plan 136) is a strictly more sophisticated ODE-motivated variant with K-stage RK β=0.5 sub-stepping (functionally a damping factor); Plan 284 ships runtime adaptive depth-tier scaling; Plan 085 ships L2/KL residual scoring; Plan 152 ships Newton-Schulz cubic fixed-point. The narrow delta is a specific **runtime primitive — damped fixed-point optimizer with patience-based geometric η decay (FPOpt Algorithm 1)** — that fuses our shipped residual scoring + shipped damping + shipped patience (currently only wired for tree search in `dd_tree.rs::early_exit_patience`) into a single adaptive-halt rule for looped transformers. A secondary architectural insight — **pre-norm + residual scaling** as a stable loop configuration — applies to our LT2/TF-Loop config choice.

**Verdict: Gain.** Provable latency win on easy inputs (paper Fig. 1: ~27% fewer effective layers on Sudoku-Extreme easy bucket, +10pp accuracy), but not a new capability class — same outputs as fixed-K looping, less compute, on the opt-in looped-transformer path. The architectural modifications (pre-norm, residual scaling) are also not novel relative to our existing TF-Loop K-stage RK design (β acts as a residual scale). Plan only, behind a new `fpopt_halt` feature flag, NOT default until GOAT-gated.

**→ riir-train (out of scope, noted for completeness):** deep-supervision training loop (Alg. 2), truncated BPTT with Neumann-series gradient (Prop. 1), Adam-Atan2 optimizer (already distilled via Plan 082b), EMA rate 0.999, learnable α₁/α₂ init at (0.75, 0.25). Training-side material → redirect.

---

## 1. Paper Core Findings

### 1.1 Architecture — non-hierarchical looped fixed-point Transformer

The base update is the classical looped-Transformer recurrence:

```
z_{i+1} = f_θ(z_i ; x),                    (Eq. 1)
```

where `f_θ` is a stack of `L` Transformer layers applied weight-tied across iterations, and `x` is re-injected between iterations. FPRM's contribution is *not* the recurrence itself — it is (a) the architectural modifications that make this recurrence trainable at large depth without hierarchy, and (b) using convergence of the recurrence as the halting signal.

### 1.2 Pre-norm + residual scaling (the architectural move)

Prior looped models (HRM, TRM, Universal Transformer, RecurrentGemma) use **post-norm** because it keeps activations bounded under iteration — a hard requirement, since unbounded activations diverge over many loops. But post-norm introduces signal-propagation pathology at depth (curse of depth, rank collapse) that limits the *effective* layer the loop can utilise. FPRM switches to **pre-norm** and recovers boundedness a different way:

**Layer-wise residual scaling** (tied scalars α₁, β₁ shared across all `L` layers in one application of `f_θ`):

```
z_ℓ = α₁ · z_{ℓ−1} + β₁ · f^ℓ_θ(Norm_pre(z_{ℓ−1}))        (Eq. 2)
```

**Iteration-wise input mixing** (tied scalars α₂, β₂ shared across all iterations):

```
z^0_{i+1} = α₂ · z_{2L_i} + β₂ · x                         (Eq. 3)
```

**Theorem 1 (boundedness).** With `0 ≤ α₁, α₂ < 1`, `β₂ = 1 − α₂·α₁^{2L}`, `β₁ = β₂·(1−α₁)/(1−α₁^{2L})`, and per-layer Lipschitz bound `‖f^ℓ(u)‖ ≤ c_f`, the iterates `{z^0_i}` are bounded: `‖z^0_∞‖ ≤ ‖x‖ + α₂·c_f`. (Proof in paper §A.1 — straightforward unrolling + geometric-series convergence.)

**Theorem 2 (small α₂ ⇒ contraction).** If `α₂·λ_f < 1` where `λ_f` is the Lipschitz constant of the L-layer map, then `f_θ(·; x)` is a contraction with unique fixed-point `z⋆` and residual decays linearly: `‖f_θ(z_i;x) − z_i‖ ≤ (α₂·λ_f)^i · ‖f_θ(z_0;x) − z_0‖`. (Proof §A.2 — Banach fixed-point theorem.)

Empirical init: α₁=0.75, α₂=0.25 (more contractive at init → easier convergence; α₁ high keeps residual stream dominant, matching standard signal-propagation fixes).

### 1.3 Damped fixed-point optimizer (FPOpt, Algorithm 1) — the runtime primitive

Even when the iteration is locally contractive in the sense of Thm. 2, in practice the Jacobian `J = ∂f_θ/∂z|_{z⋆}` can have eigenvalues with `|λ_i| ≥ 1` but `ℜ(λ_i) < 1` — the iteration then **spirals around** `z⋆` instead of contracting toward it (oscillation). The paper proves (Thm. 3) that a damped update `g_{η,θ}(z;x) := η·f_θ(z;x) + (1−η)·z` eliminates the oscillation while preserving the fixed-points (since `g_{η,θ}(z;x)=z ⇔ f_θ(z;x)=z` for any η > 0).

**Algorithm 1 — FPOpt** (one damped step with patience-based decay):

```
state: η ← η₀, p ← P, r⋆ ← ∞
STEP(z, z̃):
    r ← ‖z − z̃‖_∞ / (‖z̃‖_∞ + ε)                # relative L∞ residual
    z ← η·z̃ + (1−η)·z                            # damped update
    if r < r⋆:
        r⋆ ← r, p ← P                            # progress: reset patience
    else:
        p ← p − 1
        if p ≤ 0 and r > τ:
            η ← γ·η, p ← P                       # geometric η decay
    return z, r
HALT when r < τ (paper default τ = 0.1).
```

Hyperparameters from §G: η₀ = 1.0, γ ∈ [0.95, 0.997] (higher γ = more compute = better accuracy, monotone), P ∈ {5, 10} (minor effect once γ is high). Hard caps: max iter (35k eval / 12–24 train) + min-η floor.

### 1.4 Adaptivity — the headline empirical result

The halting signal `r_i < τ` fires *per-sample*, so compute scales with difficulty:
- **Sudoku-Extreme** (Fig. 1, Fig. 5): FPRM matches or beats TRM at +10 accuracy points while using **~27% fewer effective layers** on easy puzzles. Default TRM (no ACT at inference) exhausts the full budget on every input.
- **State-tracking A5 / S5** (Fig. 4): FPRM reaches 98.1% / 98.8% at length 128 (trained at length 32), with effective layers scaling smoothly with sequence length. TRM+conv+ACT only adapts on a *few* seeds — most either drop accuracy or burn the full budget.
- **Depth-utilisation** (Fig. 6, Fig. 7): FPRM with pre-norm+scaling saturates at ~2× the effective layer of the post-norm variant. Hierarchy (HRM/TRM's H+L steps) appears to *alleviate* signal-prop issues — FPRM matches hierarchical baselines without the hierarchy by fixing signal propagation directly.

### 1.5 Hierarchy-removal insight

HRM and TRM distribute compute between fast-looping (L-steps) and slow-looping (H-steps). FPRM achieves the same or better without hierarchy. The paper's hypothesis: **hierarchy's benefit is mostly signal-propagation**, not the biological "System 1 / System 2" motivation. Fig. 13: TRM performs best when compute is reallocated from inner L/H loops to outer deep-supervision steps — consistent with post-norm TRM being signal-propagation-limited at deep effective layer. We never adopted HRM/TRM hierarchy, so this is confirmatory, not actionable.

### 1.6 Training-side (→ riir-train)

- **Truncated BPTT** through the latest K iterations, with implicit-function-theorem gradient at the fixed-point: `dz⋆/dθ = (I−J)^{-1} P`, approximated by Neumann series truncated at depth K. Prop. 1 bounds the truncation error by `O(σ^K)` under contractivity (σ = ‖J‖₂).
- **Deep supervision** every T_sup = K iterations.
- **Adam-Atan2** optimizer (β₁=0.9, β₂=0.95) — already distilled via Plan 082b.
- **EMA** rate 0.999.
- **Learnable α₁, α₂** initialised at (0.75, 0.25); after training the distribution widens but the median stays near init.

All of this is training-side and out of scope for the modelless workflow.

---

## 2. Distillation

### 2.1 What we already ship (the prior-art surface)

| Paper mechanism | Shipped cousin | File / Plan |
|---|---|---|
| Weight-tied looped block (Eq. 1) | `LoopMode::WeightShared { loop_count }` — default-on, GOAT 8/8 | Plan 108, `crates/katgpt-core/src/types.rs:314`, `forward_looped` |
| ODE-motivated damped iterated block (more general than Eq. 1) | `LoopMode::TrainingFree` + `TrainingFreeLoopConfig` (K-stage RK β=0.5, window, cache strategy) — default-on, GOAT 4/4 | Plan 136, `tf_loop` |
| Adaptive depth (halt when converged) | Hydra cumulative-DE convergence gate; runtime depth-tier cap | Plans 165, 284; `InferenceOverrides.depth_tier` |
| Fixed-point residual scoring | `ManifoldResidual` trait + `L2ResidualScorer` / `KlResidualScorer` + `ResidualRelevanceScorer`, GOAT 6/6 | Plan 085, `src/pruners/manifold_residual.rs` |
| Cubic fixed-point iteration (different sense — orthogonalisation) | Newton-Schulz 5-iteration cubic fixed-point, GOAT 25/25 | Plan 152, `src/newton_schulz.rs` |
| Residual-based early-halt | Self-Advantage Gate on HLA reconstruction — halts HLA iterative refinement when residual collapses, GOAT 2/3 + 0ns overhead | Plan 283, Bench 057 |
| Patience-based early halt (different substrate — tree search) | `early_exit_patience` in `dd_tree.rs::TreeBuilder::build` and `build_screened`, and in `distill/ilc.rs` — patience on consecutive-dominant with gap threshold | `src/speculative/dd_tree.rs:2234`, `src/distill/ilc.rs:635` |
| Attractor fixed-point basins (different substrate — NPC belief) | `AttractorKernel` (Family A) and `LatentThoughtKernel` (Family B, K iterations of A per tick) | Plan 276, `crates/katgpt-core/src/micro_belief/{attractor,latent_thought,types}.rs` |
| DEQ / Anderson acceleration comparison | **Disproved** — Research 035 (Attractor) and Research 097 (Training-Free Looped) both found Anderson acceleration fails on transformer blocks (not contractive) | Research 035, 079, 097 |
| Test-time scaling (sample-level difficulty adaptation) | Runtime CLR claim-level reliability + depth tier; SimpleTES RPUCG; FreqBandit; BanditPruner | Plans 284, 086, 189, default-on |

### 2.2 The FPRM delta vs shipped prior art

The honest read: Research 265 (CoFRe/FP-MGM, written today) already mapped the same prior-art surface for a sibling paper and reached a Gain verdict. FPRM's specific delta against our shipped stack is narrow:

**(a) Damped FPOpt as a single primitive (Alg. 1).** We have all the *ingredients* — residual scoring (Plan 085), damped sub-stepping (Plan 136 K-stage RK), patience (in tree search), adaptive tier (Plan 284) — but they are not fused into FPOpt's specific rule: "geometric η decay after P steps of no residual improvement, halt when relative residual < τ". The fusion is novel; the ingredients are not.

**(b) Pre-norm + residual scaling as a loop-architecture choice.** Our TF-Loop K-stage RK with β=0.5 is functionally a fixed damping factor; FPRM generalises this with learnable per-layer α₁ and per-iteration α₂. Whether our LT2/TF-Loop config already uses pre-norm is worth verifying (it should, since pre-norm is the modern default), but the explicit residual-scaling parameterisation is a refinement.

**(c) Hierarchy-removal.** Confirmatory only — we never adopted HRM/TRM hierarchy.

### 2.3 Fusion

**FPRM × Plan 136 (TF-Loop) × Plan 085 (Deep Manifold residual) × `early_exit_patience` pattern:** TF-Loop currently runs a *fixed* K iterations of K-stage RK sub-stepping. The fusion: replace fixed-K with FPOpt — compute the relative L∞ residual `r = ‖z − f(z)‖_∞ / (‖f(z)‖_∞ + ε)` after each sub-step (reusing `ManifoldResidual::residual`), halt when `r < τ`, and apply geometric η decay (γ=0.99 default) after P=10 steps of no improvement. This composes the paper's FPOpt with our shipped K-stage RK (which remains the *inner* damping strategy) and our shipped residual scorer. The early_exit_patience pattern from `dd_tree.rs` is a structural analogue — same "consecutive progress count + decay" idiom, different substrate.

**FPRM × Plan 276 (AttractorKernel null result):** Plan 276's G2.1 found that random-init attractor kernels do *not* exhibit fixed-point-basin hysteresis — the basins require trained weights. FPRM's Thm. 3 + Alg. 1 provide a *runtime* alternative: instead of waiting for trained basins to emerge, force convergence via damping. This does not rescue Plan 276 (the attractor's *basin structure* is still random), but it suggests that for any iterative kernel, **damped FPOpt can substitute for trained convergence** when only the *halting* behaviour (not the basin semantics) matters. Worth noting for any future attractor-style runtime work.

**FPRM × Plan 283 (Self-Advantage Gate on HLA):** Plan 283 already implements residual-based early-halt for HLA — `Bench 057` shows 100% argmax preservation at 3-step-early halt with 0ns overhead. The fusion: add FPOpt's patience-based η decay to Plan 283's halt rule. Currently Plan 283 is binary halt/no-halt; FPOpt would make it a *damped* halt — when residual stalls but is still above threshold, decay the step-size for a few more tries before giving up. Expected gain: recover a few additional cases that currently exhaust the budget, at near-zero overhead.

None of these fusions are a new capability class. All three are **refinements on opt-in paths**.

### 2.4 → riir-train (out of scope, noted for completeness)

- Deep-supervision + truncated BPTT loop (Alg. 2).
- Neumann-series implicit gradient at the fixed-point (Prop. 1) — relevant only if training a fixed-point model.
- Learnable α₁, α₂ init at (0.75, 0.25); Adam-Atan2; EMA 0.999.
- Pre-norm + residual scaling as a *training* architectural modification for a looped checkpoint.

All training-side. Redirect.

---

## 3. Verdict

**Gain.**

**One-line reasoning:** FPRM's narrow novel delta — the FPOpt damped fixed-point optimizer with patience-based geometric η decay (Alg. 1) — is a runtime primitive that fuses ingredients we already ship (residual scoring, damped sub-stepping, adaptive depth-tier, patience-based halt in tree search) into a single adaptive-halt rule for looped transformers; provable ~27% latency reduction on easy inputs in the paper, but on opt-in looped paths in our stack and not a new capability class.

**Novelty gate (§1.5 of the workflow):**

| Q | Answer |
|---|---|
| 1. No prior art? | **Partial.** All ingredients shipped (Plans 085, 108, 136, 152, 165, 276, 283, 284 + `early_exit_patience` in tree search). The *specific fusion* (FPOpt Alg. 1) is novel; the *mechanism class* is not. |
| 2. New class of behaviour? | **No.** Same outputs as fixed-K looping, less compute. Plan 283 already ships residual-based early-halt on HLA; Plan 284 already ships adaptive depth-tier. FPOpt is a refinement, not a new capability. |
| 3. Product selling point? | **No.** Cannot finish "our NPCs/systems do X that no competitor can" with FPRM alone — we already have multiple adaptive-compute primitives. |
| 4. Force multiplier? | **Yes** — connects LT2 (108), TF-Loop (136), Deep Manifold residual (085), Self-Advantage Gate (283), CLR depth-tier (284), attractor kernel (276). |

1.5/4 YES → not Super-GOAT. Not GOAT either (no provable headline gain over our *existing* adaptive-compute stack — Plan 283 already achieves 0ns-overhead residual halt; the marginal cases FPOpt would recover are narrow). **Gain**: plan only, opt-in feature flag, GOAT-gate before any default promotion.

**Tier comparison vs sibling papers:**

| Paper | Verdict | Why same tier |
|---|---|---|
| Research 265 (CoFRe/FP-MGM) | Gain | Sibling fixed-point-masked-gen paper; ~80% inference surface already shipped; narrow delta (3SR carry rule) |
| Research 079 (EqR) | (deferred to Plan 119) | Same fixed-point view; breadth scaling, no new capability |
| Research 035 (Attractor) | (already distilled) | Direct ancestor; Anderson acceleration disproved; attractor view as mental model only |
| **Research 266 (FPRM, this note)** | **Gain** | Sibling fixed-point-reasoning paper; ~80% inference surface already shipped; narrow delta (FPOpt Alg. 1) |

---

## 4. Plan Sketch (katgpt-rs/.plans/267_*)

**Target:** `crates/katgpt-core/src/tf_loop/fpopt.rs` (new file, behind `fpopt_halt` feature) + integration point in `forward_looped`.

**GOAT gate (must pass before default promotion):**

- **G1 — Mechanics.** FPOpt produces bounded output for any input; relative residual monotonically non-increasing over a window of P steps before decay; no NaN/Inf.
- **G2 — Latency win on easy inputs.** On a synthetic suite with bimodal difficulty (easy: 2-loop convergence; hard: 20-loop convergence), FPOpt uses ≥30% fewer effective layers than fixed-K=20 baseline at matched argmax accuracy.
- **G3 — No regression on hard inputs.** Hard-input accuracy within 1pp of fixed-K=K_max baseline.
- **G4 — Zero allocation on the hot path.** State (η, p, r⋆) is 3 f32/u32 fields; residual computed in-place via existing `ManifoldResidual::residual`.
- **G5 — Feature isolation.** Compiles with/without `fpopt_halt`; zero overhead when disabled.

**Phases:**

- **Phase 1 — FPOpt kernel (CORE).** Implement `FPOptState` + `fpopt_step` in `tf_loop/fpopt.rs`. Unit tests G1.
- **Phase 2 — TF-Loop integration.** Wire FPOpt into `LoopMode::TrainingFree` as an optional halt rule. Bench G2/G3 on a synthetic bimodal suite.
- **Phase 3 — HLA integration (optional stretch).** Wire FPOpt's η decay into Plan 283's Self-Advantage Gate for the "stalled-but-above-threshold" cases.
- **Phase 4 — Documentation.** Update `tf_loop` README and `.docs/07_adaptation.md`.

**Demotion rule (AGENTS.md):** if FPOpt does not meet G2 (≥30% latency win on easy inputs) AND G3 (no hard-input regression), the feature stays opt-in and is documented as a null result (like Plan 276's attractor kernel).

---

## 5. Open Questions / Verification Needed

1. **Does our LT2/TF-Loop config use pre-norm or post-norm?** Pre-norm is the modern default, but this should be verified in the layer code before claiming FPRM's architectural move is "already shipped". If we use post-norm, switching to pre-norm + explicit residual scaling (α₁, α₂) is a *separate* GOAT-gated plan with its own benchmarks (signal-propagation stability at large K).
2. **Is `LoopMode::TrainingFree` K-stage RK β=0.5 strictly more general than FPRM's residual scaling?** Intuitively yes (β acts as a residual scale, K-stage RK is a more principled ODE integrator), but a head-to-head on the same task would settle it. Paper §F shows TRM with post-norm benefits from *fewer* inner loops + more outer deep-supervision steps — consistent with our K-stage RK finding from Research 097 (only K-stage RK survived, all higher-order methods failed).
3. **Does FPOpt's η decay actually help on our workloads, or does K-stage RK β=0.5 already cover the damping niche?** The paper applies FPOpt to *weight-tied iteration of a single block* (DEQ-style); our K-stage RK applies to *training-free mid-block looping of frozen checkpoints*. Different substrates — FPOpt may be redundant on ours. G2 will tell.

---

## TL;DR

FPRM is a **non-hierarchical looped Transformer with fixed-point halting**, stabilised by pre-norm + residual scaling (α₁, α₂) and damped by a patience-based geometric η decay (FPOpt Alg. 1). Training-side contributions → riir-train. Inference-side: **~80% already shipped** per Research 265's sibling analysis (LT2 weight-shared loop = Plan 108; TF-Loop K-stage RK = Plan 136; Deep Manifold residual = Plan 085; Self-Advantage Gate residual halt = Plan 283; CLR depth-tier = Plan 284; `early_exit_patience` = tree search). **Verdict: Gain** — the narrow novel delta is FPOpt as a single fused primitive for looped-transformer adaptive halting; plan only, opt-in `fpopt_halt` feature flag, GOAT-gated (G1 mechanics + G2 ≥30% easy-input latency win + G3 no hard-input regression + G4 zero-alloc + G5 feature isolation) before any default promotion. Three fusions identified (TF-Loop + Deep Manifold + tree-search patience; AttractorKernel runtime-convergence substitute; Self-Advantage Gate damped-halt extension) — all refinements on opt-in paths, none a new capability class. Open Q1 (do we already use pre-norm in LT2/TF-Loop?) blocks the architectural claim and must be verified before planning.
