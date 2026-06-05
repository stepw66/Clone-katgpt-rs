# Research 167: Economy of Minds (EoM) — Hayek Market Coordination for Modelless Inference

> **Source:** [arXiv:2606.02859](https://arxiv.org/abs/2606.02859) — Qi et al., Jun 2026
> **Code:** `.raw/EoM/` (HayekMAS Python reference)
> **Date:** 2026-06-05
> **Verdict: HIGH VALUE — Decentralized market-based agent coordination distills into modelless Rust via our existing BanditPruner + AbsorbCompress + DDTree stack. The key insight is NOT the LLM multi-agent system (that's orchestration, stays in Python). The key insight is the AUCTION → PAYMENT → REWARD → BANKRUPTCY → BIRTH cycle as a modelless credit assignment mechanism for our HL (Heuristic Learning) arms.**

---

## TL;DR

EoM shows that a population of weak agents can self-organize into strong collective intelligence through simple economic signals: auctions for action rights, payments between agents, and wealth-based survival. No central orchestrator needed — prices coordinate.

**What we already have (no action needed):**
- Auction mechanism → LatCal DeFi auction program, Monopoly property auctions (Plan 035)
- Bankruptcy cycle → Monopoly FinancialCrisis → Bankrupt FSM state
- Population evolution → RandOpt N-perturbation (R080), AlphaProof Elo population (R088), OpenDeepThink BT ranking (R040)
- Credit assignment → SDPG dense signal (R160), SimpleTES trajectory credit (R052), StepCode bi-level (R025)
- Exploration/exploitation → BanditPruner UCB1/Thompson, AbsorbCompress promotion, ε-decay

**What's worth distilling (NOVEL fusion):**
1. **WealthPruner** — BanditPruner where arm wealth replaces UCB1 confidence bonus. Arms accumulate wealth from rewards, lose wealth from bids. Bankrupt arms (wealth < 0) get "rebirthed" with new strategy from richest arm's mutation. This is a fundamentally different exploration signal than UCB1 — it's economic selection, not statistical optimism.
2. **AuctionDDTree** — DDTree where branch expansion is auctioned among candidate tokens. Wealthy token groups (high cumulative reward) bid higher, winning the right to expand. Creates implicit specialization without explicit routing.
3. **ChainRewardSplitter** — EoM's step-reward chain splitting (window of recent winners share reward) distills into modelless trajectory credit. Already partially in SimpleTES (R052), but EoM's unique-member window is simpler and cheaper.

**What's NOT worth distilling:**
- Full LLM agent system → We're modelless. EoM's agents are LLM wrappers with prompts. Not applicable.
- Prompt mutation (trainable_system_prompt evolution) → This IS LLM training. Violates constraint 1.
- Holland wealth-proportional bidding → Requires maintaining per-arm floating-point wealth state. Our UCB1 is cheaper and already proven.
- Wakeup condition matching → EoM agents have LLM-evaluated wakeup rules. We use entropy-gated activation (cheaper, modelless).

---

## Core Mechanics (from `.raw/EoM/`)

### 1. Auction (Step-Level)
```
Each step:
  1. Filter active agents (wakeup conditions)
  2. Each agent submits bid (fixed / holland wealth-proportional)
  3. Highest bidder wins → pays bid to PREVIOUS winner (P2P transfer)
  4. Winner acts (LLM call)
  5. Environment returns reward
  6. Winner receives reward
```

Key insight: Payment flows PREVIOUS winner → creates temporal credit chain. If Agent A plans, Agent B implements, Agent C evaluates → A gets paid by B's bid, B gets paid by C's bid, C gets the environmental reward. Decentralized credit assignment without a global credit function.

### 2. Population Evolution (Episode-Level)
```
After each episode:
  1. Check bankruptcy (wealth < 0)
  2. Remove bankrupt agents
  3. With probability p_a: birth GOOD agent (mutate richest)
  4. With probability p_b: birth BAD agent (mutate poorest with failure trace)
  5. Periodic births (every N episodes)
  6. Rent charge (every M episodes) — forces exploration by draining idle wealth
```

Key insight: BAD birth uses the DEAD agent's failure trace to create a "repaired" child. This is the exploration analogue — learn from failure, not just exploit success.

### 3. Reward Schemes
- `path_reward_only` — terminal reward split across all agents on the action path
- `path_reward_and_stepwise_reward` — adds per-step environmental reward
- `step_reward_split_chain` — window of recent winners shares step reward (unique members only)

### 4. Bid Schemes
- `fixed` — base_bid + novice premium
- `holland` — VETERAN → TYCOON promotion when wealth ≥ threshold, bid = α × wealth

---

## Creative Fusion Ideas

### Fusion 1: WealthPruner — Economic Bandit Arms (NOVEL, Modelless)

**The idea:** Replace UCB1's `Q(a) + c·sqrt(ln(N)/n(a))` exploration bonus with wealth-based bidding.

```
Standard UCB1:    arm_score = Q(a) + c·sqrt(ln(N)/n(a))
WealthPruner:     arm_score = Q(a) + wealth(a) × bid_alpha

Where:
  wealth(a) accumulates from rewards (positive) and bids (negative)
  bid_alpha is a hyperparameter (default 0.1)
  Bankrupt arms (wealth < 0) get "rebirthed" with Q from richest arm ± noise
```

**Why this is different from UCB1:**
- UCB1 is optimistic in the face of uncertainty (explore arms with few pulls)
- WealthPruner is capitalistic (arms that EARN more get more opportunity)
- Bankruptcy creates forced exploration (dead arms get replaced with mutations of successful arms)
- Rent charge prevents any arm from hoarding wealth indefinitely

**Modelless path:** WealthPruner is pure bookkeeping — `f64` accumulators per arm, no neural network. The "mutation" on rebirth is: take richest arm's Q-value, add Gaussian noise σ=0.1, reset wealth to initial.

**Where:** Extends `BanditPruner` with `WealthBanditPruner` variant.
**Depends on:** `BanditPruner`, `AbsorbCompress` (for the promotion mechanic)

### Fusion 2: AuctionDDTree — Token Auction for Branch Expansion (NOVEL, Modelless)

**The idea:** When DDTree selects the next token to expand, instead of always picking the highest marginal, hold a "sealed-bid auction" among candidate tokens.

```
Standard DDTree:  next_token = argmax(marginal[token])
AuctionDDTree:    1. Candidate tokens = top-K marginals
                  2. Each candidate's "bid" = cumulative_reward_of_similar_tokens × bid_alpha
                  3. Winner = highest bid (not highest marginal)
                  4. Winner pays bid to second-highest bidder (Vickrey auction)
```

**Why this matters:** DDTree currently has no memory of which token groups have been historically successful. AuctionDDTree creates implicit specialization — tokens that have led to good outcomes in the past get more expansion budget. This is the EoM price-signal insight applied to tree search.

**Where:** New `AuctionDDTree` strategy variant alongside standard DDTree.
**Depends on:** `DDTree`, token-level reward history (already tracked in `SampleBuffer`)

### Fusion 3: ChainRewardSplitter — Window-Based Trajectory Credit (SIMPLE, Modelless)

**The idea:** EoM's step-reward chain splitting is simpler than SimpleTES's full trajectory credit bridge.

```
EoM approach:
  Maintain rolling window of last W arm activations
  When reward arrives:
    unique_arms = deduplicate(window)
    share = reward / len(unique_arms)
    Each arm gets: share

SimpleTES approach:
  Full C×L×K budget loop
  Trajectory-level max-score credit
  RPUCG graph propagation
```

EoM's approach is O(W) per reward, SimpleTES is O(C×L×K). EoM is cheaper and sufficient for our modelless path.

**Where:** New `ChainCreditAssigner` struct in `katgpt-rs/src/pruners/`.
**Depends on:** `BanditPruner` (arm tracking)

---

## GOAT Verdict (per Verdict 003)

| Question | Answer |
|----------|--------|
| Does this land in engine (MIT) or fuel (SaaS)? | **Engine** — WealthPruner, AuctionDDTree, ChainRewardSplitter are all modelless inference primitives. MIT. |
| Does this fit the "Ferrari, no gas" model? | Yes — the economic coordination is the Ferrari. The gas is the reward signal (from game environment, verifier, or user feedback). |
| Does this hurt existing UCB1/BanditPruner paths? | No — WealthPruner is a new variant, not a replacement. Feature-gated under `wealth_pruner`. |
| Should it be on by default? | **After GOAT proof** — if WealthPruner converges faster than UCB1 in bomber arena, yes. |
| Modelless? | ✅ All three fusions are pure bookkeeping. No neural forward pass. |
| LoRA-only for training? | ✅ No training involved. Reward signal comes from environment. |
| Self-learning adaptive CoT? | ✅ WealthPruner IS self-learning — arms adapt their bids based on accumulated wealth, which reflects historical reward. |
| CPU/GPU auto-route? | ✅ All operations are CPU-side f64 arithmetic. Sub-μs per arm update. |
| SOLID/DRY? | ✅ WealthPruner extends BanditPruner. AuctionDDTree extends DDTree. ChainCreditAssigner is a new struct. No duplication. |

### Expected Performance

| Component | Overhead | Target |
|-----------|----------|--------|
| WealthPruner::relevance() | +1 f64 multiply per arm | <1% vs UCB1 |
| WealthPruner::update() | +2 f64 adds per arm | <5% vs UCB1 |
| AuctionDDTree branch select | +K comparisions (K=top-K candidates) | <10% vs DDTree |
| ChainCreditAssigner | +W f64 adds per reward | <1% vs direct reward |
| Bankruptcy check | O(arms) scan | Episode-level, not hot path |

---

## Relationship to Existing Research

| Research | Connection |
|----------|-----------|
| **R021 (G-Zero Self-Play)** | G-Zero's δ-signal is a dense credit signal. WealthPruner is an ALTERNATIVE credit signal — economic rather than gradient-based. |
| **R052 (SimpleTES)** | ChainRewardSplitter is the cheap version of SimpleTES's trajectory credit bridge. |
| **R075 (Data Gate)** | EoM's bankruptcy filter is analogous to Data Gate's strict filter — both remove bad data before it contaminates the population. |
| **R076 (SR²AM)** | SR²AM's configurator decides when to plan. WealthPruner decides which arm gets to act. Different problem, same "meta-decision" pattern. |
| **R080 (RandOpt)** | RandOpt's population perturbation is the "good birth" from EoM. WealthPruner adds the "bad birth" (failure-driven mutation). |
| **R088 (AlphaProof)** | AlphaProof's Elo-rated sketch population is analogous to wealth-ranked agent population. |
| **R160 (SDPG)** | SDPG's dense credit assignment is richer than EoM's step-reward chain. But EoM's approach is cheaper. |
| **R035 (Monopoly FSM)** | Monopoly is our EoM sandbox — auctions, bankruptcy, wealth, 4 player archetypes. The closest existing implementation. |

---

## Constraints Compliance

| Constraint | Compliance |
|-----------|-----------|
| **Modelless first** | ✅ All three fusions are inference-time bookkeeping. No LLM training. |
| **Land in riir-ai domain** | ✅ Core traits in katgpt-rs (MIT engine). Game integration via existing arena pipeline. |
| **LoRA only for training** | ✅ No training involved. |
| **Self-learning adaptive CoT** | ✅ WealthPruner arms self-improve via wealth accumulation. Freeze/thaw persistence via existing `ThinkingBanditFrozen` (Plan 194). |
| **SOLID, DRY** | ✅ Extends existing traits, no duplication. |
| **Tests/examples** | ✅ Bomber arena before/after: UCB1 vs WealthPruner. Expected: WealthPruner converges to same arm but with fewer episodes (economic selection > statistical optimism). |
| **CPU/GPU auto-route** | ✅ All CPU-side. GPU reserved for rendering/training. |

---

## References

- EoM paper: [arXiv:2606.02859](https://arxiv.org/abs/2606.02859)
- EoM code: `.raw/EoM/` (HayekMAS reference implementation)
- Our bandit: `katgpt-rs/src/pruners/bandit.rs`
- Our absorb-compress: `katgpt-rs/src/pruners/absorb_compress.rs`
- Our DDTree: `katgpt-rs/src/ddtree.rs`
- Monopoly arena: `katgpt-rs/.plans/035_monopoly_fsm.md`
- RandOpt population: `katgpt-rs/.research/080_RandOpt_Neural_Thickets.md`
- SimpleTES credit: `katgpt-rs/.research/052_SimpleTES_Evaluation_Driven_Scaling.md`

**TL;DR:** EoM's economic coordination (auction → payment → reward → bankruptcy → birth) distills into three modelless Rust components: WealthPruner (economic bandit), AuctionDDTree (token auction), and ChainRewardSplitter (cheap trajectory credit). The novel insight is using WEALTH as the credit signal instead of UCB1's optimism bonus — arms that earn more get more opportunity, bankrupt arms get replaced. This is economic selection applied to inference-time arm management.
