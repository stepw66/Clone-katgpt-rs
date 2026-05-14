# Issue 056: Game Theory Player Design — Tit-for-Tat Composite Player

**Status:** Superseded — See Plan 055 for MMORPG TFT (FFT/GvG)
**Feature gate:** `g_zero`
**Source:** Plan 054 (Player A/B Benchmark) + Game Theory Analysis
**Impact:** None yet — bomber TFT deferred in favor of FFT/GvG TFT (Plan 055)

> **Note:** Bomber provocation is ambiguous (bomb nearby ≠ intentional attack).
> FFT/GameEvents give crystal-clear provocation signals (`DamageDealt { attacker, target }`).
> Plan 055 (`055_mmorpg_tft_party_ai.md`) covers the real MMORPG use case with party-level TFT,
> role-based retaliation, and GvG coordination. This issue remains as bomber-specific reference.

---

## Context

Plan 054 benchmark results showed **Greedy (72.1%)** beats all composable players in isolation.
This matches a well-known game theory result: **Tit-for-Tat wins iterated games** by being
Nice, Retaliatory, Forgiving, and Clear (Axelrod's tournament, 1980).

### Benchmark Results (Plan 054)

```
Config        │ Survival │ Avg Kills │ P50 (μs) │ Game Theory Analog
🐱 Greedy     │   72.1%  │      0.12 │      1.3 │ Closest to TFT (Nice + Clear)
🐶 Validator  │   58.6%  │      0.05 │      1.2 │ "Sucker" (Pure Cooperator)
🐵 HL         │   57.0%  │      0.24 │      0.6 │ "Always Defect" (Envious)
🤖 GZero      │   64.1%  │      0.06 │      0.5 │ Mixed (Nice + noisy adaptation)
```

---

## Game Theory Mapping

### Tit-for-Tat Traits vs Our Players

| TFT Trait | Definition | Greedy | HL | GZero | Validator |
|-----------|-----------|--------|-----|-------|-----------|
| **Nice** | Never provoke first | ✅ Collects PU | ❌ Hunts opponents | ✅ Weak scorer | ✅ Safe moves |
| **Retaliatory** | Strike back if attacked | ❌ Ignores threats | ⚠️ Preemptive hunt | ❌ No retaliation | ❌ Avoids conflict |
| **Forgiving** | Resume cooperation after | N/A (never retaliates) | ❌ Chases across map | N/A | N/A |
| **Clear** | Predictable, opponents adapt | ✅ Simple heuristic | ⚠️ Complex strategy | ❌ Template random | ✅ Deterministic |

### Why Greedy Wins (Game Theory Lens)

Greedy is **Nice + Clear** — it collects resources (cooperates with itself), doesn't initiate
conflict, and has predictable behavior. In a 4-player arena:

1. HL hunts opponents → creates conflict → dies more
2. HL and GZero fight each other → mutual destruction
3. Greedy avoids fights → survives the crossfire
4. This is literally the **Iterated Prisoner's Dilemma** result

### Why HL Loses (Game Theory Lens)

HL violates TFT's core principle: **"Don't be envious"**.

- HL focuses on kills (0.24/round) — it's trying to **beat** opponents, not maximize its own outcome
- HL's hunt/intercept/trap tactics create escalation spirals
- In game theory terms, HL is a **Grim Trigger** player — once it sees an opponent, it never stops attacking
- Grim Trigger loses long-term because it can't recover from accidental conflicts

### The Missing Player: Tit-for-Tat Bomber

No current player implements true TFT. What's missing:

```
Default state: Nice (collect powerups, avoid conflict)
   │
   ├─ Opponent places bomb nearby (within blast range)?
   │    YES → Switch to Retaliatory (use HL attack tactics for N ticks)
   │           After N ticks → Switch to Forgiving (stop chasing)
   │
   └─ No provocation?
        Stay Nice (Greedy heuristic: collect, explore, survive)
```

---

## Proposed Design: `TftPlayer` (Tit-for-Tat)

### Architecture

```text
TftPlayer {
    mode: TftMode,               // Nice | Retaliatory(ticks_left)
    provocation_threshold: f32,   // distance/criteria for "provoked"
    retaliation_duration: u8,     // ticks to stay retaliatory (Forgiving)
    scorer: ScorerKind,           // Greedy (Nice) | HL (Retaliatory)
}

Per tick:
    if mode == Retaliatory && ticks_left == 0:
        mode = Nice  // Forgiving: resume cooperation

    if mode == Nice && opponent_bomb_nearby():
        mode = Retaliatory(retaliation_duration)  // Retaliatory: strike back

    match mode {
        Nice         → score_action (Greedy's heuristic)
        Retaliatory  → score_action + attack_bonus (HL's tactics)
    }

    Safety filter: always active (wall-aware blast, BFS escape)
    Explore: 10% ε-greedy (balanced)
```

### TFT Mode FSM

```
         ┌─────────────────────────────────────┐
         │                                     │
    ┌────▼────┐   bomb nearby   ┌──────────────┴──┐
    │  NICE   │───────────────► │  RETALIATORY     │
    │(Greedy  │                 │  (HL attack      │
    │ scorer) │  ticks expire   │   + hunt bonus)  │
    │         │◄─────────────── │                  │
    └─────────┘   (Forgiving)   └──────────────────┘
```

### Key Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `provocation_radius` | 4 cells | How close opponent bomb must be to trigger retaliation |
| `retaliation_duration` | 10 ticks | How long to stay aggressive (Forgiving window) |
| `escalation_threshold` | 2 bombs | Multiple bombs = stronger provocation |
| `scorer_nice` | `score_action` (Greedy) | Base heuristic when cooperating |
| `scorer_attack` | HL strategy bonus | Attack heuristic when retaliating |

---

## Expected Outcomes

### Hypothesis

| Metric | Greedy | HL | TFT (predicted) | Reasoning |
|--------|--------|-----|-----------------|-----------|
| Survival | 72.1% | 57.0% | **68-75%** | Nice by default (like Greedy) |
| Kills | 0.12 | 0.24 | **0.15-0.20** | Retaliatory when provoked (not envious) |
| P50 Latency | 1.3μs | 0.6μs | **~1.0μs** | Mode switch is cheap |

### Game Theory Prediction

In mixed 4-player tournaments:
- TFT vs Greedy: both cooperate → high mutual survival
- TFT vs HL: TFT retaliates when HL attacks → HL gets punished for aggression
- TFT vs GZero: TFT ignores GZero → GZero's templates irrelevant
- TFT vs Validator: both cooperative → similar performance

**Nash Equilibrium**: TFT should reach equilibrium where no player benefits from
unilaterally changing strategy. Greedy can't improve by attacking. HL can't improve
by being more aggressive (already loses). TFT is stable.

---

## Tasks

### Phase 1: TftPlayer Implementation
- [ ] **T1**: Create `microgpt-rs/src/pruners/bomber/tft_player.rs`
  - `TftMode` enum: `Nice` | `Retaliatory { ticks_left: u8 }`
  - `TftPlayer` struct implementing `BomberPlayer` trait
  - `is_provoked(events, pos, opponents, radius) -> bool` — detect nearby hostile bombs
  - `select_action`:
    - Update mode (Forgiving: decrement ticks, switch to Nice when expired)
    - Check provocation (Retaliatory: switch if bomb nearby)
    - Score actions based on current mode
    - Safety filter (always active)
    - 10% ε-greedy
  - `update_outcome` — track outcomes for stats (no bandit needed for TFT)
  - Gate behind `#[cfg(feature = "g_zero")]` (reuses bomber + g_zero infra)

- [ ] **T2**: Wire `TftPlayer` into `microgpt-rs/src/pruners/bomber/mod.rs`
  - `#[cfg(feature = "g_zero")] pub mod tft_player;`
  - `#[cfg(feature = "g_zero")] pub use tft_player::TftPlayer;`

### Phase 2: Benchmark
- [ ] **T3**: Add `TftPlayer` to `g_zero_04_player_ab_benchmark.rs`
  - Add `PlayerKind::Tft` variant
  - Run 1000 rounds isolated benchmark
  - Compare with existing 4 configs

- [ ] **T4**: Create mixed tournament benchmark
  - `g_zero_05_tft_mixed.rs` — 4-player mixed tournament:
    - Slot 0: Greedy
    - Slot 1: HL
    - Slot 2: GZero
    - Slot 3: TFT
  - 1000 rounds, measure survival + kills + Nash-like equilibrium analysis
  - Print game theory alignment table (Nice/Retaliatory/Forgiving/Clear scores)

### Phase 3: Validation
- [ ] **T5**: Verify TFT survival ≥ Greedy survival (hypothesis: 68-75%)
- [ ] **T6**: Verify TFT kills > Greedy kills (hypothesis: 0.15-0.20)
- [ ] **T7**: `cargo clippy --fix --allow-dirty` — zero warnings
- [ ] **T8**: `cargo test -p microgpt-rs --features g_zero` — all tests pass
- [ ] **T9**: Commit with message: `feat(bomber): TftPlayer — game theory Tit-for-Tat (Issue 056)`

---

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Player name | `TftPlayer` | Matches game theory terminology |
| Feature gate | `g_zero` | Reuses bomber infra, no new feature needed |
| No bandit | Stateless mode FSM | TFT is simple — no learning needed (Clear principle) |
| Provocation trigger | Opponent bomb in blast range | Direct threat detection, unambiguous signal |
| Retaliation scorer | HL attack tactics | Proven kill efficiency (0.24/round) |
| Nice scorer | Greedy `score_action` | Proven survival (72.1%) |
| Safety filter | Always active | Non-negotiable — wall-aware blast + BFS escape |

---

## Game Theory Principles Applied

### 1. Nice (Default to Cooperation)
> "Never be the first to defect" — Axelrod

TFT starts in `Nice` mode, collecting powerups and avoiding conflict.
Uses Greedy's proven `score_action` heuristic as base.

### 2. Retaliatory (Enforce Boundaries)
> "Punish defection immediately" — Axelrod

When opponent places bomb in blast range → switch to HL's attack tactics.
Hunt, intercept, trap the aggressor. Don't be a pushover.

### 3. Forgiving (Don't Hold Grudges)
> "Resume cooperation after retaliation" — Axelrod

After `retaliation_duration` ticks → back to `Nice` mode.
Don't chase across the map. Don't escalate spirals.
The 10% forgiveness rule from "Generous Tit-for-Tat".

### 4. Clear (Be Predictable)
> "Simple logic that opponents can understand" — Axelrod

Mode FSM is 2 states with simple transitions.
Opponents learn: "If I bomb near TFT, it attacks back. If I leave it alone, it collects powerups."
This creates stable Nash Equilibrium — opponents have incentive to cooperate.

### Why This Should Work

The benchmark proved:
- **Greedy's scorer** = best survival (Nice + Clear)
- **HL's attack** = best kills (Retaliatory)
- **Neither alone** = optimal (Greedy lacks retaliation, HL lacks niceness)

TFT combines the best of both through **situational mode switching**:
- 90% of the time: Nice (Greedy scorer) → high survival
- 10% of the time: Retaliatory (HL attack) → punish aggressors
- 100% of the time: Forgiving → prevent escalation spirals

This is the **Generous Tit-for-Tat** strategy that dominates iterated games.

---

## File Map

```
microgpt-rs/
  src/pruners/bomber/
    tft_player.rs                     ← T1: TftPlayer implementation
    mod.rs                            ← T2: add pub mod tft_player
    g_zero_player.rs                  ← (existing, unchanged)
    players.rs                        ← (existing, unchanged)

riir-ai/crates/riir-examples/
  examples/
    g_zero_04_player_ab_benchmark.rs  ← T3: add PlayerKind::Tft
    g_zero_05_tft_mixed.rs            ← T4: mixed tournament
  Cargo.toml                          ← T4: add [[example]] g_zero_05_tft_mixed
```

---

## Open Questions

1. **Provocation threshold**: Should we detect opponent bombs only, or also opponent proximity?
   - Bomb-only is cleaner (Clear principle) but may be too slow to react
   - Proximity-only may trigger false positives (opponent just passing by)
   - Proposed: bomb-in-blast-range as primary, proximity-within-2 as secondary

2. **Retaliation duration**: 10 ticks? 20? Adaptive based on game phase?
   - 10 ticks ≈ 1 bomb fuse cycle (standard timing)
   - Too short → not enough punishment to deter aggression
   - Too long → becomes Grim Trigger (loses forgiveness advantage)

3. **Generous TFT**: Should we add 10% chance to forgive without retaliation?
   - Original Axelrod tournament: pure TFT won
   - Noisy environments: Generous TFT (10% forgive) wins
   - Bomber arena has "noise" (random explosions, collateral damage)
   - Proposed: configurable, default ON with 10% forgiveness

4. **Should TFT track WHO provoked it?**
   - Current design: retaliate against nearest opponent
   - Game theory: should retaliate against the specific defector
   - Simpler: just attack nearest (Clear principle)