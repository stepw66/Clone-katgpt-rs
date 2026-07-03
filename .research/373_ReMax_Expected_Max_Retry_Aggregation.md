# Research 373: ReMax — Expected-Max-Over-m Retry Aggregation (Bonus-Free Exploration Operator)

> **Source:** [Emergence of Exploration in Policy Gradient Reinforcement Learning via Retrying](https://arxiv.org/pdf/2606.00151) — Nishimori, Parmas, Koyamada, Kozuno, Kitamura, Ishii, Matsuo (UTokyo + Kobe + Kyoto + ATR + Alberta + Isara Labs), ICML 2026, PMLR 306
> **Code:** [github.com/nissymori/remax-rl](https://github.com/nissymori/remax-rl)
> **Date:** 2026-07-03
> **Status:** Active
> **Related Research:** 250 (Self-Advantage / AdvantageMarginGate — closest math cousin, different app), 248 (BoM single-pass sampling — sampling-based best-of-K cousin), 284 (Simplicity Bias Sampler), 098 (PrudentBanker bandit), 370 (Manifold Bandit — Thompson-sampling tree), 363 (realtime RL budget gate — PASS, training-only), 288 (KARC forecaster)
> **Related Plans:** 374 (this paper's plan — ReMax aggregation primitive)
> **Cross-ref (riir-train):** RePPO training algorithm (PPO variant + EI-based advantage + Q-replacement) → `riir-train/.research/` one-line note. The training loop is out of scope for this workflow.
> **Classification:** Public

---

## TL;DR

ReMax scores a policy by the **expected maximum return over M samples** ("best of M retries"), accounting for return uncertainty. Exploration emerges as an *emergent property* of this objective — no explicit bonus term (no entropy bonus, no count bonus, no RND). The paper derives a **closed-form policy gradient** (RePPO) for this objective and generalizes the integer retry count M to a **continuous parameter m > 0**, enabling fine-grained control of the exploration–exploitation trade-off via objective curvature.

**Distilled for katgpt-rs (modelless, inference-time):**

The paper's headline contribution is a **training algorithm** (RePPO = PPO variant with EI-based advantage) — that part → riir-train. What survives as a *modelless primitive* is the **closed-form aggregation operator** behind ReMax:

1. **Expected-max-over-m** (Eq 4): given a discrete policy π and Q-values q for K actions, compute `J_m(π, q) = q₍₁₎ + Σⱼ (q₍ⱼ₊₁₎ − q₍ⱼ₎)(1 − Cⱼ)ᵐ` — the expected best-of-m value, in O(K log K), no sampling. For m=1 it degenerates to the mean (standard RL); for m>1 it biases toward upside; for m<1 it accelerates convergence.
2. **Expected Improvement** (Eq 10): `EI_m(R; π, q) = v₍₁₎ + Σⱼ (v₍ⱼ₊₁₎ − v₍ⱼ₎)(1 − Cⱼ)ᵐ⁻¹` where `vᵢ = (R − qᵢ)₊` — a Bayesian-optimization-style acquisition weight per action, O(K log K).
3. **Continuous-m curvature control**: the *same* closed form works for any real m > 0, interpolating exploration (m>1 flattens gradient, sustains stochastic policy) and exploitation (m<1 sharpens, accelerates). Even with *deterministic* rewards, m reshapes objective geometry — it is not just an uncertainty model.

**Verdict: GOAT (pending benchmark).** The closed-form expected-max-over-m is a novel modelless operator with **no exact prior art** in the codebase (checked all 5 repos, both paper-vocab and codebase-vocab; see §2.2). The closest cousins — `best_of_k_rollouts` (sampling-based, O(K×rollout)), `AdvantageMarginGate` (related advantage math, different application: recursion-loop gating), BoM single-pass (generative diversity, different domain) — are *structurally distinct*. **However**, the codebase carries a strong **negative prior**: SDAR (NO GOAT, ELO 954≈955), RMSD (NO GOAT), FFO dual-cutoff (Harmful) all found that reward/objective modulation does *not* improve action selection at our scale. ReMax is a different mechanism (objective curvature via max-over-m, not a reward bonus), but the risk is real. The GOAT gate (Plan 374) must benchmark against UCB1 / softmax / best_of_k on a controlled bandit regret domain. Promote-to-default only if it beats UCB1's sublinear regret with fewer hyperparameters (single m vs bonus coefficient c). **Not a Super-GOAT** because Q3 (selling point) is unproven and Q4 (force multiplier) is speculative pending the benchmark.

---

## 1. Paper Core Findings

### 1.1 The ReMax objective — "score by expected best-of-M, not by mean"

Standard RL maximizes `J_RL(π) = E_{A~π}[μ_A]` (expected return of one draw). ReMax maximizes:

```
J_M^ReMax(π) = E_{μ~Π}[ E_{A[M]~π}[ max_{m∈[M]} μ_{A_m} | μ ] ]
```

— the expected **best** return over M i.i.d. draws from π, averaged over reward uncertainty Π.

**Key properties:**
- **M=1, fixed rewards** → degenerates to standard RL → optimal policy is deterministic.
- **M≥2, uncertain rewards** → optimal policy is **stochastic** (hedging which arm is rewarding).
- **M→∞** → the best arm dominates; policy concentrates on the arm with highest upside.
- Exploration is not a bonus added to the reward; it is the *shape of the objective itself*.

### 1.2 Closed-form computation (Proposition 3.2) — the distillable core

Given Q-values `q = (q₁,...,q_K)` and policy π, sort q descending: `q₍₁₎ ≥ ... ≥ q₍K₎`, with aligned masses `π₍ⱼ₎`. Define `Cⱼ = Σ_{u≤j} π₍ᵤ₎` (cumulative mass of top-j actions). Then:

```
J_M^ReMax(π, q) = q₍₁₎ + Σ_{j=1}^{K-1} (q₍ⱼ₊₁₎ − q₍ⱼ₎) · (1 − Cⱼ)^M       (Eq 4)
```

Cost: **O(K log K)** (one sort + one cumulative-sum pass). No dependence on M. This is the closed-form expected maximum over M i.i.d. draws — derived by conditioning on the best sampled rank.

**Generalization to real m > 0**: replace `(1−Cⱼ)^M` with `(1−Cⱼ)ᵐ`. This is well-defined for full-support policies (all πᵢ > 0). The paper clips `1−Cⱼ` from below for numerical stability when m < 1.

### 1.3 Expected Improvement (Proposition 4.3) — the acquisition function

For a reference return R, policy π, and Q-values q, the **Expected Improvement** is:

```
EI_M(R; π, q) = v₍₁₎ + Σ_{j=1}^{K-1} (v₍ⱼ₊₁₎ − v₍ⱼ₎) · (1 − Cⱼ)^{M−1}     (Eq 10)
```

where `vᵢ = (R − qᵢ)₊ = max(R − qᵢ, 0)`. This measures: *how much does a single additional draw improve over the best of M−1 other draws?* It is the per-action weight in the ReMax policy gradient (Def 4.2). Generalizes to `EI_m` via `(1−Cⱼ)^{m−1}`.

This is structurally identical to the **Expected Improvement acquisition function** from Bayesian optimization (Jones et al. 1998), but computed in closed form from the policy's own action probabilities rather than from a Gaussian process posterior.

### 1.4 Continuous-m curvature control (the deterministic-reward insight)

Even with **fixed** rewards `μ = (0, 1)` and `p := π(a=1)`:

```
J_m^ReMax(p) = 1 − (1−p)^m
```

The maximizer is always `p*=1`, but the **local geometry** near p=1 depends strongly on m:
- **m > 1**: objective flattens near the optimum → gradients vanish → convergence slows → **sustains exploration**.
- **m < 1**: objective sharpens → gradients amplified → **accelerates convergence**.
- **m = 1**: standard RL curvature.

This means m is a *single continuous knob* that controls exploration-exploitation even without reward uncertainty — by reshaping the objective landscape. Peak empirical performance on MinAtar was at **m ∈ [1.2, 1.4]**; on Atari's hard-exploration games, **m ∈ [0.9, 1.0]** (those games need less exploration).

### 1.5 Empirical results

- **Bandits (posterior)**: ReMax with M∈{2,3} achieves empirically sublinear cumulative regret, comparable to UCB and Thompson sampling; Softmax incurs higher regret (Fig 2).
- **MinAtar**: RePPO(m∈[1.2,1.4]) without entropy bonus outperforms PPO-V/Q with entropy bonus and PPO-V+RND. Maintains higher policy entropy than PPO+entropy (Fig 3). Ablation: both the action-independent baseline and Q-replacement are necessary (Fig 5).
- **Craftax** (1B steps): RePPO(1.2) without bonuses is competitive with PPO+entropy+RND (11.87% vs 11.68%), clearly outperforms PPO without bonuses (9.31%).
- **Speed**: RePPO overhead vs PPO-Q is comparable to the V→Q critic upgrade — negligible vs the performance gain.

### 1.6 Q-replacement (the practical trick)

When computing EI for the return R, the term `v_a = (R − q_a)₊` should be zero (R can't beat itself). But an underspecified critic may yield `R > Q(s,a)`, overestimating EI and causing policy overfit. Fix: **replace the a-th element of q with R** before computing EI, enforcing `v_a = 0` by construction.

---

## 2. Distillation

### 2.1 What is modelless vs what is training

| Component | Modelless? | Where |
|---|---|---|
| ReMax objective closed form (Eq 4) | **YES** — O(K log K) aggregation operator | → katgpt-rs primitive |
| Expected Improvement (Eq 10) | **YES** — O(K log K) acquisition weight | → katgpt-rs primitive |
| Continuous-m parameterization | **YES** — single scalar, no training | → feature config |
| RePPO policy gradient (Eq 9, 12) | **NO** — requires gradient ascent on θ | → riir-train |
| Q-critic training (PPO critic fit) | **NO** — requires TD learning | → riir-train |
| Q-replacement trick | **YES** — inference-time substitution | → katgpt-rs (part of EI impl) |

The **distillable core** is two closed-form functions: `expected_max_over_m(π, q, m)` and `expected_improvement(R, π, q, m)`. Both take a discrete policy (probabilities) and Q-values (or a reference return), and return a scalar / per-action weights. Zero allocation, O(K log K), feature-gated.

### 2.2 Prior-art check (5-repo grep, both vocabularies)

**Paper vocabulary** (`remax|reppo|expected_max|expected_maximum|max@k|pass@k|retry|m_draws`): **ZERO hits** across all repos.

**Codebase vocabulary** (`expected_improvement|EI|advantage_margin|best_of_k|exploration_bonus`): hits exist but are **structurally distinct**:

| Existing | What it does | Why it's NOT ReMax |
|---|---|---|
| `best_of_k_rollouts` (Plan 083, `dd_tree.rs`) | Runs K full SDE tree rollouts, selects best by cumulative Q | **Sampling-based** (O(K×tree)), not closed-form. Operates on tree *paths*, not single-decision Q-values. Selection modes: BestQ / MostFrequent / Top1Converged. |
| `AdvantageMarginGate` (Plan 283, R250) | `A(candidate) − E[A(a)]` from pre/post-recursion logits | **Related math** (advantage-like signal), but **different source** (logit ratio from one model's two passes vs Q-values+policy) and **different application** (recursion-loop dead-compute gating vs policy-gradient advantage). Gates loop *continuation*; doesn't aggregate over M draws. |
| UCB1 (`BanditPruner`, Plan 030) | `Q_a + c·√(ln(N)/n_a)` | **Explicit exploration bonus** — exactly what ReMax claims to *replace*. UCB1 adds a count-based term; ReMax reshapes the objective. |
| RPUCG (Bench 016) | `λ·√((1+|S|)/(1+n_i))` | Explicit pseudo-count bonus. Same class as UCB1. |
| BoM single-pass (Plan 281, R248) | K noise queries → K futures → supervise best | **Generative diversity** (K forward passes on a diffusion/world model), not discrete-action Q-value aggregation. Different domain. |
| `WidthSelectionMode::MostFrequent` (mode@K) | Most common path across K rollouts | Voting/consensus, not expected-max. |

**Conclusion**: no shipped code computes the *closed-form expected maximum over m i.i.d. draws from a discrete policy*, parameterized by continuous m. The ReMax aggregation operator is genuinely novel in this codebase.

### 2.3 Negative prior — the risk

Three shipped experiments found that **reward/objective modulation does not improve action selection** at our scale:

| Experiment | Verdict | Finding |
|---|---|---|
| SDAR Arena (Plan 072) | **NO GOAT** | ELO 954 ≈ Rubric 955. "Reward modulation ≠ selection improvement." 28% higher bandit regret. |
| RMSD (Plan 125) | **NO GOAT** | 46/46 structural proofs pass, but RMSD within 10% of SDAR over 1000 bomber games. "Reward signal modulation does not improve action selection." |
| FFO dual-cutoff (Plan 062) | **Harmful** | UCB1 exploration bonus inflates low-Q scores. Cutoff masks useful arms. |

**Why ReMax might still differ**: SDAR/RMSD modulated the *reward signal* (adding bonuses, masking arms). FFO added hard cutoffs on top of UCB1. ReMax modulates the *objective aggregation* (max-over-m instead of mean) — it doesn't add a bonus, it changes how multiple draws are combined. The m parameter controls objective *curvature*, not reward magnitude. Whether this structural distinction translates to a real gain is exactly what the GOAT gate (Plan 374) must test.

### 2.4 Latent-space reframing (mandatory per workflow)

How does ReMax look on the codebase's latent-state kernels?

**(a) HLA → action bridge (strongest):** The HLA 8-dim belief state projects to Q-values for K NPC actions via dot-product + sigmoid onto learned direction vectors. ReMax's expected-max-over-m could serve as the *action selection transform*: instead of `argmax(Q)` (greedy) or `softmax(Q/τ)` (temperature), select from the ReMax-optimal policy induced by `(π, q, m)`. The continuous m becomes a **per-NPC "adventurousness" knob** — high-m NPCs sustain diverse action distributions (curious, risk-seeking), low-m NPCs converge fast (exploitative, cautious). This is a **latent-to-raw bridge** (HLA latent → discrete action) parameterized by m. The m parameter could itself be *driven by* the HLA curiosity/arousal channels — high arousal → high m → more exploration.

**(b) latent_functor / zone gating:** Zone attention selection could use expected-max-over-m instead of top-1 dot-product: "which zone has the highest expected best-of-m projection?" This would make zone attention *stochastic under uncertainty* rather than deterministic. Speculative — needs the primitive first.

**(c) DEC / Stokes:** Not directly relevant. ReMax is about discrete action selection under uncertainty, not manifold geometry.

**(d) NeuronShard / LatCal:** Not directly relevant. Shards are storage; LatCal is commitment.

**The latent reframing lands on (a) — HLA→action selection with curiosity-driven m.** This is a riir-ai runtime angle (per-NPC action selection), but the *open primitive* (the closed-form math) belongs in katgpt-rs.

### 2.5 Fusion opportunities

| Fusion | What it produces | Status |
|---|---|---|
| ReMax × **BanditPruner** (UCB1 replacement) | Bonus-free exploration via m-parameter instead of count-based `c·√(ln(N)/n)`. Single knob replaces bonus coefficient + count tracking. | Primary GOAT gate target |
| ReMax × **BoMSampler** (R248) | BoM generates K diverse futures; ReMax computes expected-best analytically from their value distribution instead of sampling. Could reduce K forward passes to 1 + closed form. | Speculative — needs BoM's value distribution available |
| ReMax × **AdvantageMarginGate** (R250) | Use EI as the advantage signal for recursion-loop gating: "is the expected improvement of one more recursion step positive?" Unifies the two advantage-like signals. | Speculative |
| ReMax × **Manifold Bandit** (R370) | Thompson-sampling tree + expected-max-over-m aggregation at tree nodes. Latent-space bandit with retry-aware value. | Speculative |

---

## 3. Verdict

**Tier: CORRECT PRIMITIVE, NO MODELLESS GOAT (resolved 2026-07-03, Plan 374 Phase 5).**

| Gate | Status | Reasoning |
|---|---|---|
| Novel mechanism (Q1) | ✅ | No exact prior art (5-repo grep, both vocabularies). Closed-form expected-max-over-m with continuous m is genuinely new. |
| New capability class (Q2) | ❌ **FAIL** | **Theorem (Plan 374 Phase 3):** argmax_a EI_m(q_a; π, q) = argmax_a q_a. The ReMax EI, used as a per-arm deterministic selection score, is **provably equivalent to greedy**. No new modelless capability for action selection. |
| Product selling point (Q3) | ❌ **FAIL** | "NPCs explore without bonuses" does not hold modellessly. ReMax's exploration is training-time (policy gradient on J_m, m > 1 flattens gradient). Correctly deferred to riir-train (RePPO). |
| Force multiplier (Q4) | ⚠️ Deferred | Connections to bandits, BoM, AdvantageMarginGate, Manifold Bandit exist but are all training-context. None confirmed modellessly. |

**One-line reasoning:** The closed-form operators are correct (G1: MC + analytic recurrence, max err 3.87e-7) and fast (G4: 603ns for K=128). But the **No Modelless Exploration theorem** proves that deterministic argmax EI = argmax q — ReMax provides no inference-time exploration bonus. Its exploration mechanism lives in policy gradient training (RePPO) → riir-train. **Keep opt-in.**

**Per-stack ledger:**
- Stack slot: action-selection/bandit (modelless) → **opt-in, no modelless gain**.
- Stack slot: RePPO-advantage (training) → **redirected to riir-train**.

**MOAT gate (katgpt-rs domain):** ✅ In scope as a building block. The operators ship behind `remax_aggregation` (opt-in, NOT promoted to default — no modelless gain per AGENTS.md §"Promotion requires modelless gain"). Their value is as RePPO advantage computation primitives for riir-train, not as a standalone modelless exploration mechanism.

**Riir-train redirect:** The RePPO training algorithm (PPO variant + EI-based advantage + Q-critic + Q-replacement) → `riir-train/.research/` one-line note. The PG derivation (Eq 9, Prop 4.1), the PPO surrogate modification (Eq 12), and the λ-return Q-critic fitting are all training-bound. Out of scope for this workflow. **This is where ReMax's exploration mechanism actually lives** — the modelless distillation hypothesis (inference-time selection bonus) is disproven by the theorem.

---

## 4. The Primitive (distilled API)

```rust
/// Expected maximum over m i.i.d. draws from a discrete policy.
///
/// Given Q-values `q` for K actions and a policy `pi` (probabilities summing to 1),
/// computes the closed-form expected best-of-m value in O(K log K).
///
/// - m = 1.0 → degenerates to the mean `E_{A~pi}[q_A]` (standard RL).
/// - m > 1.0 → biases toward upside (exploration-flavored).
/// - m < 1.0 → accelerates convergence (exploitation-flavored).
///
/// Source: Nishimori et al. ICML 2026, Proposition 3.2 (Eq 4).
/// Feature gate: `remax_aggregation`.
pub fn expected_max_over_m(pi: &[f32], q: &[f32], m: f32) -> f32;

/// Expected Improvement: how much does one more draw improve over the best of (m-1) others?
///
/// Given a reference return `R`, policy `pi`, and Q-values `q`, returns the per-action
/// EI weight. Used as an acquisition/ranking signal (Bayesian-optimization analog).
///
/// Source: Nishimori et al. ICML 2026, Proposition 4.3 (Eq 10).
/// Feature gate: `remax_aggregation`.
pub fn expected_improvement(R: f32, pi: &[f32], q: &[f32], m: f32) -> Vec<f32>;
```

**Properties:**
- O(K log K) — one sort + one cumulative-sum pass. No allocation beyond the output (EI returns K weights; expected_max returns a scalar).
- Continuous m > 0. Clip `(1−Cⱼ)` from below (ε = 1e-8) for m < 1 numerical stability.
- No softmax (per AGENTS.md constraint #2 — sigmoid preferred). The input π is already a probability vector; the operator transforms it via power-of-cumulative-mass, not softmax.
- Deterministic, no RNG — the "exploration" is in the *objective shape*, not in sampling noise.

---

## 5. GOAT Gate Design (Plan 374)

| Gate | Metric | Baseline | Pass threshold |
|---|---|---|---|
| G1 (correctness) | Closed form matches brute-force Monte-Carlo expected-max over M draws | Monte-Carlo with 10⁶ samples | Max abs error < 1e-3 for m ∈ {0.5, 1.0, 1.5, 2.0, 3.0} |
| G2 (bandit regret) | Cumulative regret on K=10 Beta-Bernoulli bandit, T=1000, 256 seeds | UCB1 (sublinear), Thompson sampling (sublinear), Softmax | ReMax(m∈[1.2,1.4]) within 1 std-error of UCB1 OR better |
| G3 (no-regression) | Action-selection quality on bomber arena (1000 games) | Greedy argmax, UCB1 BanditPruner | Within 5% of best baseline |
| G4 (latency) | Per-call latency for K ≤ 128 actions | UCB1 score computation | < 500 ns (sub-µs, plasma tier) |
| G5 (feature-isolation) | `cargo check` with/without `remax_aggregation` | — | 0 warnings, clean compile both ways |

**Promotion rule:** If G2 shows ReMax within UCB1's std-error AND G4 < 500ns → promote `remax_aggregation` to default-on as an *alternative* selection strategy (not replacing UCB1 — coexist behind config). If G2 fails (ReMax worse than Softmax) → keep opt-in, document as negative result alongside SDAR/RMSD.

**UQ-bearing check (the "Report the Floor" rule, §Feature Flag Discipline):** ReMax's expected-max-over-m produces a scalar value estimate, not a probability distribution / interval / quantile. It is NOT a UQ-bearing primitive in the conformal-prediction sense. The conformal-naive floor rule does not apply. (If a future variant produces predictive intervals from the max-over-m distribution, then the floor rule applies.)

---

## 6. Honest Risk Assessment

**Risk 1 — Negative prior (HIGH):** SDAR, RMSD, and FFO all failed. The codebase's empirical finding is that reward modulation doesn't improve action selection. ReMax is structurally different (objective curvature, not reward bonus), but this is a hypothesis, not a proof. The GOAT gate exists precisely to test this.

**Risk 2 — Scale mismatch (MEDIUM):** The paper benchmarks on MinAtar/Craftax with deep RL training (10M–1B steps). Our use case is inference-time action selection on small discrete action spaces (K ≤ 128 for NPC actions). The m-tuning sweet spot (1.2–1.4 on MinAtar, 0.9–1.0 on Atari) may not transfer. The bandit regret gate (G2) is the closest analog to our use case.

**Risk 3 — Already-covered-by-UCB1 (LOW-MEDIUM):** UCB1 already achieves sublinear regret with a single bonus coefficient. ReMax's m parameter replaces the bonus coefficient — but if both achieve sublinear regret, the gain is marginal (one knob vs one knob). The differentiator would be: ReMax adapts to reward *uncertainty* (Fig 1 center: ReMax increases exploration as variance grows, Softmax doesn't) — UCB1 also adapts to uncertainty via counts. The test is whether ReMax's *non-count-based* uncertainty adaptation (via the Q-distribution shape) outperforms count-based UCB1.

**Risk 4 — Training-paper-only (MITIGATION):** The paper provides no inference-time experiments. Our distillation (using ReMax as an inference-time selection transform) is *our idea*, not the paper's contribution. If the GOAT gate fails, it's not a failure of the paper — it's a failure of our distillation hypothesis.

---

## 7. Implementation Notes

- **Sorting**: use `sort_by(|a, b| b.partial_cmp(a).unwrap())` on (q, π) pairs. For K ≤ 128, insertion sort or `sort_unstable_by` is cache-friendly.
- **Cumulative sum**: single pass, `C[0] = π_sorted[0]; C[j] = C[j-1] + π_sorted[j]`.
- **Power**: `powf(1.0 - C[j], m)` — use `libm` for `no_std` / WASM compatibility.
- **Numerical stability**: clip `1.0 - C[j]` to `[1e-8, 1.0]` before `powf`. The paper uses this exact clip (App D, line 19 of the JAX code).
- **Q-replacement**: when computing EI for a sampled action's return, replace `q[action]` with R before sorting. This enforces `v_action = 0` by construction.
- **No softmax**: the operator takes π as input (already a probability vector from sigmoid/normalization). It does not apply softmax internally. Per AGENTS.md constraint #2.

---

## TL;DR

ReMax is a training paper (RePPO = PPO variant) whose *training loop* → riir-train. The modelless distillable core is a **closed-form expected-max-over-m aggregation operator** (Eq 4) + **Expected Improvement** (Eq 10), both O(K log K), parameterized by continuous m > 0. No exact prior art in the codebase (5-repo grep, both vocabularies). Closest cousins: `best_of_k_rollouts` (sampling-based), `AdvantageMarginGate` (related math, different app), BoM (generative). **Verdict: GOAT (pending benchmark).** The codebase's negative prior (SDAR/RMSD/FFO — reward modulation doesn't help) creates real risk. The GOAT gate must prove ReMax matches or beats UCB1 on bandit regret with a single m parameter. If it fails → negative result alongside SDAR/RMSD. If it passes → promote as alternative selection strategy. Plan 374 implements the primitive behind `remax_aggregation` feature flag.
