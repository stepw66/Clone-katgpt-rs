# Research 168: Ruliology of Competition — Enumerative Game Theory via Simple Program Strategies

**Date:** 2026-06
**Source:** [Wolfram — Games between Programs: The Ruliology of Competition](https://writings.stephenwolfram.com/2026/06/games-between-programs-the-ruliology-of-competition/)
**Status:** GOAT — Fusion Research
**Modelless:** ✅ (core engine)
**Model-based:** ✅ (riir-ai games)

---

## Distillation

Wolfram systematically enumerates ALL programs in a computational class (FSM, CA, TM), pits every pair against each other in iterated games (matching pennies, prisoner's dilemma), and ranks by average mean payoff. Key findings:

1. **Winning strategies are NOT the most complex** — no correlation between behavior complexity and payoff
2. **Computational irreducibility** — you cannot predict who wins without running the game
3. **"Universal winners"** exist — a 10-state FSM can beat ALL 2-state FSMs
4. **Cross-class competition** — larger programs systematically outperform smaller ones by having "specialized sub-strategies" for each opponent
5. **Adaptive evolution works** — random mutation + keep-if-better converges on winning strategies, often finding simple solutions from complex intermediaries
6. **Grim trigger > tit-for-tat** — when you exhaustively enumerate ALL 2-state FSMs for Prisoner's Dilemma, grim trigger wins, NOT tit-for-tat (contradicting Axelrod tournament results)

---

## Fusion Ideas — Not Direct Mapping

### Fusion 1: `RuliologyBandit` — Bandit Arms ARE Simple Programs

**Insight:** Instead of hand-crafting bandit arms (greedy, cautious, random), **enumerate all FSMs of size N as bandit arms**. Each FSM IS an arm. The bandit selects which FSM to "play as" against the current opponent.

**Why creative:** Current `BanditPruner` selects among fixed heuristic arms. `RuliologyBandit` makes the arm space *discoverable* — the bandit doesn't just pick actions, it picks *strategy programs*. This is modelless — no training needed. The FSM enumeration is deterministic.

**Modelless mapping:**
- `ConstraintPruner` → `is_valid()` filter on FSM space
- `BanditPruner` → arm selection over enumerated FSMs
- `AbsorbCompress` → promote winning FSMs to stable arms
- `ScreeningPruner` → `relevance()` ranks FSMs by payoff history

**GOAT verdict:** This is a **pure modelless enhancement**. No LoRA, no training. The bandit learns at inference time which FSM strategy works best against the current opponent. The FSM space is enumerated once offline.

### Fusion 2: `CrossParadigmArena` — FSM vs CA vs Bandit vs HL

**Insight:** Wolfram showed that cross-class competition reveals dominance hierarchies. In our domain, this means testing FSM strategies against CA strategies against BanditPruner against HL against MCTS against LoRA+WASM — **all in the same arena with the same payoff matrix**.

**Why creative:** Current arenas test same-paradigm tiers (Greedy vs HL vs GZero). Cross-paradigm tests reveal whether simple FSM programs beat learned LoRA weights, or whether BanditPruner adapts fast enough to handle CA opponents. This directly informs the "should we add more model complexity?" question.

**Modelless mapping:**
- `GameState` trait already exists — just need `SimpleProgram: Strategy` implementations
- `EloCalculator` + `Leaderboard` already exist — just feed cross-paradigm results
- `PayoffTable<N>` already computes Nash — use it to analyze the meta-game

### Fusion 3: `ComputationalIrreducibilityGate` — When Simulation IS the Answer

**Insight:** Wolfram's core finding is computational irreducibility — you can't shortcut the game simulation. This is a **diagnostic tool**: if the win matrix of a strategy space has high Kolmogorov complexity (can't be compressed), then no analytical shortcut exists and you MUST simulate.

**Why creative:** This becomes a gate for when to use expensive methods. If the game is reducible (simple pattern in win matrix), use the shortcut. If irreducible, use simulation/rollout. This is the **modelless analog of adaptive CoT** — think when irreducible, act directly when reducible.

**Connection to existing work:**
- Plan 194 (Adaptive CoT) — bandit learns when to think. `IrreducibilityGate` is the *structural reason* why thinking is needed.
- `DataGate` (Plan 111) — strict task-level filtering. `IrreducibilityGate` is the *game-theoretic* version — filter strategies by whether they can be analytically predicted.
- `BanditPruner` — already selects arms. `IrreducibilityGate` adds a meta-arm: "simulate this opponent?" vs "use cached result?"

### Fusion 4: `RuliologyPruner` — Exhaustive Strategy Pruning via Win Matrix Compression

**Insight:** Wolfram's compressed-size-vs-payoff plots show that high-payoff strategies often have SIMPLE behavior. This means we can **prune the strategy space** by computing win matrices offline and keeping only strategies in the "high payoff, simple behavior" quadrant.

**Why creative:** This is the **ConstraintPruner analog for strategy space**. Current `ConstraintPruner` prunes invalid token branches. `RuliologyPruner` prunes unpromising strategy branches *before* the bandit even tries them. Zero runtime cost — the pruning happens offline during enumeration.

**Modelless flow:**
1. Enumerate all FSM(N) strategies offline
2. Run round-robin tournament → win matrix
3. Compress behavior traces → complexity score per strategy
4. Keep only Pareto-front (high payoff, low complexity) → pre-filtered arm set
5. Bandit selects from this pre-filtered set at inference time

### Fusion 5: `AdaptiveStrategyMutation` — Wolfram-Style Co-Evolution as HL Feedback

**Insight:** Wolfram's adaptive evolution (random mutation, keep-if-better) is structurally identical to our existing HL pipeline (`AbsorbCompress` + `TemplateProposer`), but applied at the FSM level instead of the action level. The fusion: **mutate FSM graphs as strategy templates, promote winners via AbsorbCompress**.

**Why creative:** Current `TemplateProposer` generates action templates. Wolfram-style mutation generates **strategy program templates**. A mutated FSM graph IS a new strategy — it's a more powerful mutation operator than action-level mutation.

**Modelless mapping:**
- `TemplateProposer` → add `FsmTemplateProposer` that mutates FSM graphs
- `AbsorbCompress` → promote FSM graphs that achieve stable positive payoff
- `DeltaGatedAbsorbCompress` → δ-gate the FSM mutation acceptance
- `HotSwapPruner` → hot-swap the current FSM strategy at runtime

---

## Verdict: What to Build

| Fusion | Gain | Perf Cost | GOAT? | Default? |
|--------|------|-----------|-------|----------|
| **F1: RuliologyBandit** | Discovery of optimal FSM strategies at inference time | Negligible — FSM enumeration is O(1) offline | ✅ GOAT | ✅ Default |
| **F2: CrossParadigmArena** | Reveals paradigm dominance hierarchy | Test-only, no runtime cost | ✅ GOAT | ✅ Default (test/example) |
| **F3: ComputationalIrreducibilityGate** | Structural reason for adaptive thinking | Cheap — just compression ratio check | ✅ GOAT | ✅ Default (gate) |
| **F4: RuliologyPruner** | Pre-filter strategy space to Pareto-front | Zero runtime — offline pruning | ✅ GOAT | ✅ Default |
| **F5: AdaptiveStrategyMutation** | Strategy-level evolution, not action-level | Same as existing HL pipeline | ✅ GOAT | 🔧 Feature-gated |

### Why Modelless First

All 5 fusions are **inference-time only**:
- FSM enumeration is deterministic combinatorics — no training
- Win matrix computation is offline tournament — no gradient
- Bandit arm selection is existing `BanditPruner` — no new infra
- Complexity scoring is compression ratio — no model
- Strategy mutation is FSM graph edit — no backprop

### Landing in riir-ai Domain

Per the engine/fuel split (Verdict 003):
- **Engine (MIT, katgpt-rs):** `SimpleProgram` trait, `FsmStrategy`, `CaStrategy`, `RuliologyBandit`, `IrreducibilityGate`
- **Fuel (private, riir-ai):** Cross-paradigm arena results, domain-specific payoff matrices, evolved FSM bundles per game
- **SaaS moat:** The pre-computed ruliology databases (win matrices, Pareto-fronts per game) become episode DB fuel

---

## Connection to Existing Research

| Existing | Connection |
|----------|------------|
| R021 (G-Zero Self-Play) | G-Zero discovers strategies via self-play. Ruliology enumerates ALL strategies exhaustively. Complementary: G-Zero for open-ended discovery, Ruliology for guaranteed-optimal within bounded classes |
| R027 (STRATEGA) | Rule-based agents (92% win) beat MCTS. Ruliology explains WHY: simple programs find pockets of reducibility |
| R075 (Data Gate) | Strict task-level filtering prevents collapse. IrreducibilityGate (F3) is the game-theoretic analog |
| R079 (EqR) | Convergence ≠ correctness. Irreducibility quantifies when convergence IS predictable |
| R098 (PrudentBanker) | Safe phased bandit aggression. RuliologyBandit inherits this phased approach |
| R134 (BES) | Bidirectional evolutionary search. F5 is BES applied to FSM graphs |
| R118 (LEO) | All-goals learning. A universal FSM winner is the structural analog — one strategy for all opponents |
| Plan 194 (Adaptive CoT) | Bandit learns when to think. IrreducibilityGate provides the structural signal |
| R026 (RTS Intransitive) | `PayoffTable<N>` already computes Nash. Ruliology enumerates the full strategy space around that Nash |

---

## Key Numbers from Wolfram

| Metric | Value |
|--------|-------|
| Distinct 2-state FSMs | 22 |
| Distinct 3-state FSMs | 956 |
| Best 2-state FSM avg payoff (matching pennies) | ~0.151 |
| Best 3-state FSM vs all 2-state FSMs | ~0.593 |
| 10-state FSM can beat ALL 2-state FSMs | +1.0 (universal winner) |
| Prisoner's Dilemma winner (2-state) | Machine 30 (grim trigger), NOT tit-for-tat |
| Tit-for-tat rank among all 2-state FSMs | Low — overrated by Axelrod tournament |
| Complexity vs Payoff correlation | None — winning strategies are NOT complex |
| Max period (2-state vs 2-state) | 4 steps |
| Max period (3-state vs 3-state) | 9 steps |
| Crossover time for PD rankings to stabilize | >500 steps |

---

## TL;DR

Wolfram's ruliology reveals that **exhaustive enumeration of simple programs** beats hand-designed strategy tournaments. The codebase has 70% of the infrastructure (bandits, payoffs, Nash, tournaments, mutation/selection) but 0% of the enumeration layer. Five fusions bridge this gap — all modelless, all landing in the engine/fuel split. The deepest insight: **winning strategies are simple but can only be found by running everything** — computational irreducibility is why adaptive methods (HL, bandits, self-play) work and analytical shortcuts don't. This validates the entire HL thesis.
