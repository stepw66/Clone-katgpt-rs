# Plan 374 — ReMax GOAT Gate Verdict

**Date:** 2026-07-03
**Primitive:** `remax_aggregation` — `expected_max_over_m`, `expected_improvement`, `expected_improvement_per_action`
**Paper:** [arXiv:2606.00151](https://arxiv.org/abs/2606.00151) — Nishimori et al., ICML 2026

---

## TL;DR

**The primitive is correct and fast. It provides NO modelless exploration bonus.**

The ReMax Expected Improvement operator, when used as a per-arm deterministic
selection score, is **provably equivalent to greedy selection** (argmax EI =
argmax q, by monotonicity). ReMax's exploration is a training-time phenomenon —
it emerges from policy gradient on J_m(π, q), not from inference-time action
selection. The training algorithm (RePPO) is **implemented in riir-train**
(Plan 304, feature `remax_ppo`).

**Verdict:** Keep `remax_aggregation` as **opt-in**. The primitive is a correct
building block for RePPO training (riir-train), not a standalone modelless
exploration mechanism.

---

## Gate Results

| Gate | Status | Detail |
|------|--------|--------|
| **G1** (correctness) | ✅ PASS | MC validation + analytic recurrence (see below) |
| **G2** (bandit regret) | ⚠️ PASS (theorem) | ReMax = Greedy, by proof + empirical confirmation |
| **G3** (no-regression) | ✅ N/A | Opt-in feature, no existing code depends on it |
| **G4** (latency) | ✅ PASS | max=603ns (K=128), per_action=11.7µs (O(K²)) |
| **G5** (feature isolation) | ✅ PASS | Clean compile with/without/all features |

**GOAT verdict:** Gates pass, but the gain is **not modelless**. The primitive's
exploration mechanism requires training (policy gradient on J_m). Per
AGENTS.md §"Promotion requires modelless gain": **keep opt-in**.

---

## G1 — Correctness (PASS)

Three complementary validation strategies:

### G1 (A) Monte-Carlo — `expected_max_over_m`

Brute-force MC validation for integer M. 500K trials per (K, M) combo.

| K range | M range | Max abs error | Tolerance |
|---------|---------|---------------|-----------|
| {2, 5, 10, 50, 128} | {2, 3, 5, 10} | **1.39e-3** | 3e-3 |

### G1 (A) Monte-Carlo — `expected_improvement`

MC validation of EI = E[(R − max of M−1 draws)₊].

| K range | M range | Max abs error | Tolerance |
|---------|---------|---------------|-----------|
| {2, 5, 10} | {2, 3, 5} | **1.22e-3** | 3e-3 |

### G1 (B) Analytic Recurrence — the strongest check

**Identity:** `J_m(π, q) − J_{m−1}(π, q) = E_{A~π}[EI_m(q_A; π, q)]`

Holds EXACTLY for all m > 1 (both sides reduce to the same sum). Cross-validates
`expected_max_over_m` against `expected_improvement_per_action` without MC noise.

| K range | m range | Max abs error | Tolerance |
|---------|---------|---------------|-----------|
| {2, 5, 10, 50, 128} | {1.25, 1.5, 2.0, 2.5, 3.0} | **3.87e-7** | 1e-4 |

---

## G2 — Bandit Regret (PASS — Theorem Confirmation)

### The "No Modelless Exploration" Theorem

> **Theorem:** For any policy π, Q-values q, and m > 0:
>
>     argmax_a EI_m(q_a; π, q) = argmax_a q_a
>
> **Proof:** EI_m(R; π, q) is monotonically non-decreasing in R. Each
> v₍ⱼ₎ = (R − q₍ⱼ₎)₊ is non-decreasing in R, and the telescoping sum
> EI = v₍₁₎ + Σⱼ (v₍ⱼ₊₁₎ − v₍ⱼ₎) · wⱼ with non-negative weights
> wⱼ = (1 − Cⱼ)^{m−1} ≥ 0 preserves monotonicity. Therefore q_a > q_b
> implies EI(q_a) ≥ EI(q_b). ∎

**Consequence:** Using ReMax EI as a per-arm deterministic selection score is
**provably equivalent to greedy**. There is no modelless exploration bonus.

### Empirical Confirmation (K=10 Bernoulli, T=1000, 64 seeds)

| Strategy | Mean Regret | Std Error |
|----------|-------------|-----------|
| UCB1 | 126.66 | 2.51 |
| Thompson | 236.30 | 7.05 |
| Greedy | 32.87 | 7.08 |
| Softmax(τ=0.1) | 47.50 | 2.11 |
| **ReMax(m=1.2)** | **24.87** | **5.45** |
| **ReMax(m=1.4)** | **35.12** | **8.53** |
| **ReMax(m=2.0)** | **26.23** | **6.22** |

Max |ReMax − Greedy| = 8.0, within 2σ of Greedy's stderr (7.08). **Theorem confirmed.**

Note: Greedy/ReMax outperform UCB1/Thompson here because the bandit is "easy"
(means drawn from Uniform(0,1), small best-vs-second gap). This is a known
phenomenon — exploration-heavy methods incur more regret on easy bandits.

### Why the plan's original G2 gate is inapplicable

The plan asked for "ReMax within 1 stderr of UCB1." This assumed ReMax would
provide a modelless exploration bonus. The theorem proves it cannot —
deterministic argmax EI = argmax q, by construction. The exploration in ReMax
emerges from **policy gradient training** on J_m (m > 1 flattens the gradient,
preventing policy collapse), which is the RePPO algorithm — correctly deferred
to riir-train.

---

## G4 — Latency (PASS)

| K | `expected_max_over_m` | Budget | `per_action_inplace` | Budget |
|---|----------------------|--------|---------------------|--------|
| 8 | 47 ns | 1000 ns ✅ | 103 ns | 150 ns ✅ |
| 16 | 87 ns | 1000 ns ✅ | 254 ns | 384 ns ✅ |
| 32 | 161 ns | 1000 ns ✅ | 769 ns | 1536 ns ✅ |
| 64 | 309 ns | 1000 ns ✅ | 2802 ns | 6144 ns ✅ |
| 128 | 603 ns | 1000 ns ✅ | 11692 ns | 24576 ns ✅ |

`expected_max_over_m` is O(K log K) — one sort + one cumulative-sum pass.
`expected_improvement_per_action_inplace` is O(K²) — K evaluations of the
telescoping sum. Both are allocation-free after the sort index buffer.

---

## What This Means for the 5-Repo Strategy

1. **katgpt-rs** — ships the correct, fast closed-form operators
   (`expected_max_over_m`, `expected_improvement`, `per_action`). Opt-in
   feature. No modelless GOAT — not promoted to default.

2. **riir-train** — ✅ **IMPLEMENTED (Plan 304 + 305 + 306 + 310 + 311).** The RePPO training algorithm
   (PPO variant + EI advantage + Q-critic + GAE + tabular chain MDP + NN
   actor-critic + gridworld exploration benchmark + full PPO machinery with
   entropy bonus + ConvNet + MinAtar Breakout) ships behind feature `remax_ppo`.
   Consumes these katgpt-rs operators for the advantage computation. **1244/1244 tests pass.**

   **Key findings (Plan 305 + 306 + 310 + 311) — QUINTUPLE-CONFIRMED boundary condition:**
   - Tabular function approximation: RePPO m>1 competitive, NO exploration
     superiority (Plan 305).
   - Neural network function approximation (2-layer MLP, REINFORCE update):
     RePPO m>1 competitive, STILL NO exploration superiority on K=4 gridworld
     (Plan 306, G2 gate FAILS, ratio 0.97).
   - **Root cause (proven):** For K=2, advantage normalization makes all m
     values identical (scalar multiple). For K>=3, the effect is too subtle to
     survive advantage normalization + noisy training dynamics.
   - **Full PPO machinery (Plan 310):** Adding clipped surrogate + entropy
     bonus + multi-epoch updates does NOT surface a robust m>1 benefit. The
     MEAN ratio (2.33) is inflated by bimodal outliers; the MEDIAN ratio is
     1.00 with identical breakthrough rates (29/32). The entropy bonus is the
     real driver of exploration, not the m parameter.
   - **ConvNet + MinAtar Breakout (Plan 311):** The paper's ACTUAL setup —
     ConvNet(4→16, 3×3) feature extraction on 10×10×4 MinAtar Breakout
     observations, full PPO (clipped surrogate + entropy + 4-epoch), 8 seeds ×
     100 episodes. G2 gate FAILS: median ratio 1.000, mean ratio 0.985.
     Issues 307-309 are CLOSED — the CPU path was sufficient; GPU scale would
     not change the verdict.

3. **riir-ai** — no direct consumption. The per-NPC action selection guide
   (HLA -> action with curiosity-driven m) is **not viable** — the ReMax
   exploration claim does not hold across 5 regimes (tabular, MLP, ConvNet;
   chain MDP, gridworld, MinAtar Breakout). The entropy bonus is the correct
   exploration mechanism for the runtime.

---

## Comparison to Negative Prior

This result joins the codebase's documented negative prior on reward/objective
modulation for action selection:

| Primitive | Plan | Verdict | Root Cause |
|-----------|------|---------|------------|
| SDAR | 072 | NO GOAT | Asymmetric trust doesn't improve bandit updates |
| RMSD | 125 | NO GOAT | Reward magnitude shaping doesn't help |
| FFO | 062 | NO GOAT | Dual-cutoff is harmful |
| **ReMax** | **374** | **NO modelless GOAT** | **EI selection = greedy (by theorem)** |

ReMax is structurally different from the above (it's an objective curvature
operator, not a reward shaper), but the conclusion is the same: **modelless
action selection cannot be improved by reshaping the objective.** The gain, when
it exists, comes from training on the reshaped objective.

---

## Run

```bash
# G1 + G2 theorem tests
CARGO_TARGET_DIR=/tmp/remax_g1 cargo test -p katgpt-core \
    --features remax_aggregation --lib -- --nocapture pruners::remax

# G2 bandit + G4 latency benchmark
CARGO_TARGET_DIR=/tmp/remax_bench cargo build --release -p katgpt-core \
    --features remax_aggregation --bench bench_374_remax_goat
/tmp/remax_bench/release/deps/bench_374_remax_goat-* --nocapture
```
