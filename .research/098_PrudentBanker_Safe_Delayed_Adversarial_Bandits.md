# Research 98: PrudentBanker — Safe Delayed Adversarial Bandits

**Paper:** [arXiv:2605.23351](https://arxiv.org/abs/2605.23351) — Prudent-Banker: No Extra Fees for Baseline Safety in Adversarial Bandits With and Without Delays
**Authors:** Ting Hu, Luanda Cai, Emmanouil-Vasileios Vlatakis-Gkaragkounis (UW–Madison, 2026)
**Date:** 2026-05-25
**Verdict:** 🟡 **Conditional Adopt — phased aggression mechanism is a significant upgrade to our BanditPruner safety, but adversarial-delayed regime is narrower than our production use (stochastic rewards, known baselines). Distill the safe-mixture + phased-restart into our existing bandit stack as an opt-in strategy. Game-specific delayed-feedback guardrails (MMO tick lag, frame-sampling) stay in riir-ai.**

---

## TL;DR

PrudentBanker solves the SOLID problem: achieve minimax-optimal worst-case regret Õ(√T + √D) **while** keeping O(1) regret against a designated safe baseline, even with arbitrary feedback delays. It combines Banker-OMD (delay-robust Online Mirror Descent) with phased aggression (geometric α escalation when evidence certifies baseline suboptimality). The key technical innovation is a **delay-calibrated restart threshold** ξ(D) ≈ √(D/δ) that prevents false transitions from safe-to-aggressive mode when feedback hasn't arrived yet.

**Lower bound proved:** The Õ(δ⁻¹/²(√T + √D)) rate is optimal — safety is fundamentally not free.

---

## Core Mechanism

```
PRUDENT-BANKER Architecture:

  xt = αₖ · x̂ₜ(active learner) + (1 - αₖ) · xc(safe baseline)

  Three hierarchical adaptation layers:
  1. Stages — adapt to unknown total delay D (doubling trick on D̂ₛ)
  2. Banker-OMD core — handle local feedback latency (step-size credit accounting)
  3. Phases — adapt aggression αₖ for safety (geometric doubling: αₖ = 2ᵏ⁻¹/R̂)

  Restart conditions:
  - HARD restart: D̂ₛ exceeded → reset everything, double delay estimate
  - SOFT restart: phase gap > 2R̂ + ξ(D̂ₛ) → increase α, reset learner only

  Key slack term: ξ(D̂ₛ) = (√(8·D̂ₛ + 1) - 1) / δ
  This is the "hidden debt buffer" that prevents premature aggression under delays.
```

### Guarantees (negative entropy regularizer)

| Metric | Bound |
|--------|-------|
| Worst-case regret | Õ(δ⁻¹/²(√T + √D)) |
| Comparator regret | O(1 + log D) |
| Lower bound | Ω(δ⁻¹/²(√T + √D)) — matching! |

---

## Distillation Map to Our Architecture

### What We Already Have (No Action)

| Paper Component | Our Equivalent | Notes |
|-----------------|---------------|-------|
| Multi-armed bandit core | `BanditPruner<P>` with `BanditStrategy` enum | UCB1, ε-greedy, Thompson Sampling |
| Arm selection + Q-values | `BanditStats` — q_values, visits, total_pulls | Already per-arm tracking |
| Safe baseline concept | `NoScreeningPruner` as conservative fallback | Already the "do nothing" arm |
| Exploration/exploitation | `BanditStrategy::Ucb1` (default) | Deterministic, O(log N) regret |
| Feedback observation | `BanditPruner::update(accepted_token, reward)` | Already delayed by verification |
| Shared cooperative learning | `SharedBanditStats` + `Arc<Mutex<BanditStatsInner>>` | Multi-agent cooperative bandits |
| Δ-gated exploration | `DeltaBanditPruner` — δ-weighted blind spot detection | Already uses bandit for relevance |
| Phased aggression (partial) | `ConfiguratorBandit` — PlanNew/PlanExtend/PlanSkip | SR²AM configurator as bandit arms |
| Context-aware selection | `ConfiguratorBandit` keyed by `(domain, entropy_bin)` | Already context-sensitive |
| Game-specific bandit tuning | Per-game frozen bandit configs (private) | See riir-ai (private) |

### What's Worth Distilling (New)

1. **`SafeBanditStrategy` — phased aggression with safe baseline mixture**
   - Add new variant to `BanditStrategy` enum
   - Mixing: `xₜ = αₖ · active_arm + (1 - αₖ) · baseline_arm`
   - Geometric α escalation: only get aggressive when evidence warrants
   - O(1) regret against baseline — no catastrophic exploration cost
   - **Value:** Production safety net for DDTree — if bandit explores bad arms, the safe baseline caps cumulative damage

2. **Delay-calibrated phase threshold — `ξ(D)` slack term**
   - When feedback is delayed (episode completion, async verification), add buffer to phase-gap statistic
   - Prevents premature "baseline is suboptimal" certification
   - **Value:** Frame-sampling bridge (Plan 070) — AI runs at lower Hz than game tick, creating inherent feedback delay. This ξ(D) buffer prevents the bandit from getting stuck in bad exploration cycles when frame samples are sparse.

3. **Banker-OMD step-size credit accounting**
   - Instead of fixed learning rate, maintain per-round "credit" budget that accounts for pending feedback
   - When feedback is missing, reduce effective learning rate proportionally
   - **Value:** Our `BanditStats` already tracks visits/pulls but doesn't account for in-flight observations. When the game loop runs at a higher Hz than bandit evaluation (per-game ratio, private), some observations are "pending" between bandit evaluations.

### What's NOT Worth Distilling

- **Full adversarial regime** — Our rewards are stochastic (game outcomes, token acceptance), not adversarial. The minimax analysis is overkill for production game AI where opponents don't set losses adversarially per-round.
- **Negative entropy / Tsallis regularizer** — Our `BanditStats` uses simple Q-value averaging (sample mean). Switching to OMD with mirror maps is unnecessary complexity for our reward distributions.
- **Lower bound proofs** — Interesting theory but doesn't change implementation. We prove GOAT empirically.
- **Uniform baseline assumption** — We don't need xc to be full-support. Our baselines are game-specific heuristics that may assign 0 probability to some actions.
- **Batched bandit reduction** — Only relevant for adversarial lower bounds, not for our stochastic game setting.

---

## Connection to MMO GOAT Pillars

**Game application:** Moved to riir-ai (private). The generic delay-adversarial bandit primitive stays here as open math.

We have a private delay-aware bandit application for real-time game frame-sampling. See riir-ai (private).

---

## Open/Close Boundary

```
katgpt-rs (MIT)                           riir-ai (Private)
──────────────────────                     ──────────────────────
BanditStrategy::SafePhased enum variant    Per-game safe baseline xc configs
BanditStats with delay_credit field        Per-game ξ(D) tuning parameters
Phase gap computation (generic)            Frame-sampling delay adaptation
Generic phased aggression logic            MMO tick-delay bandit wrappers

"plug socket"                              "plug"
```

**Rule:** The safe-phased bandit strategy and generic delay-credit accounting go in katgpt-rs. Game-specific baselines, per-game delay estimates, and the tuned ξ parameters stay private in riir-ai.

---

## Verdict Summary

| Aspect | Rating | Rationale |
|--------|--------|-----------|
| Theoretical novelty | ★★★★★ | First algorithm to solve SOLID — O(1) comparator regret + minimax optimal under delays. Matching lower bounds. |
| Practical relevance | ★★★☆☆ | Adversarial regime is narrower than our stochastic game rewards. Delayed feedback is relevant but not dominant. |
| Implementation complexity | ★★☆☆☆ | Simple to add as new `BanditStrategy` variant. Phase-gap + α mixing is ~200 lines. |
| GOAT proof potential | ★★★★☆ | Easy to prove: (1) baseline regret bounded, (2) worst-case competitive with UCB1, (3) delay-awareness doesn't hurt no-delay case |
| MMO pillar support | ★★★★☆ | Directly applicable to Pillar 4 (Frame-Sampling) and indirectly to all pillars via safer exploration |

**Bottom line:** The phased aggression mechanism is a clean, principled upgrade to our bandit stack. It adds a safety net that caps exploration cost against a known-good baseline. The delay-calibrated threshold is the real gem — it prevents exactly the failure mode we'd hit in production when frame-sampling creates sparse feedback. Low implementation cost, high conceptual value, easy GOAT proof. Do it.

**Feature gate:** `safe_bandit` (opt-in, NOT default-on) — new `BanditStrategy::SafePhased` variant.

**Super-GOAT (private):** Game-specific delay-aware frame-sampling bandit → moved to riir-ai/.research/124. The generic bandit math stays here.
