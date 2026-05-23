# 104 — Fog of War Tactical Puzzle: Exploration-Based Solver Comparison

## Tasks

- [ ] Vision system (BFS flood-fill, wall-blocking, radius 4)
- [ ] FogState (seen, visible, discovered targets)
- [ ] Frontier computation (tiles adjacent to unknown)
- [ ] Exploration strategies (BF nearest, AI heuristic, Hybrid region+BF)
- [ ] Single-phase iterative solver (explore while solving)
- [ ] Game engine with fog integration (bridge blocks vision)
- [ ] TUI with fog visualization (dim unseen, bright visible, remembered faded)
- [ ] Headless benchmark across seeds
- [ ] Three-round TUI flow with comparison metrics

## Problem Statement

Current `tactical_07_strategic.rs` solvers are **omniscient** — they know all target
positions from the start. This makes BF = AI = Hybrid because:

1. Same information → same optimal path
2. `par_find_shortest_sequence` evaluates ALL valid candidates regardless of marginals
3. Pruner constrains tree structure identically for AI and Hybrid

**Real AI doesn't have perfect information.** In an MMO or real game:
- You only see what's in vision range
- You don't know what's behind doors/walls until you look
- You remember previously seen positions
- Exploration strategy matters

## Architecture

### New File
- `examples/tactical_09_fog.rs` — self-contained fog-of-war tactical puzzle

### Vision System

BFS flood-fill from player position:
- Radius: 4 tiles (BFS distance)
- Walls block vision (can see the wall tile but not through it)
- Bridge tiles when CLOSED block vision (like walls)
- Bridge tiles when OPEN allow vision through

This means you CANNOT see boxes/goal on the other side of a closed bridge.
You must discover and toggle levers to open the bridge, then explore beyond it.

### FogState

```rust
struct FogState {
    seen: HashSet<(usize, usize)>,      // ever seen (memory)
    visible: HashSet<(usize, usize)>,    // currently in vision range
    discovered_keys: Vec<(usize, usize)>,
    discovered_boxes: Vec<(usize, usize)>,
    discovered_levers: Vec<(usize, usize)>,
    discovered_traps: HashSet<(usize, usize)>,
    discovered_bridge: HashSet<(usize, usize)>,
    goal_pos: Option<(usize, usize)>,    // None until seen
}
```

### Single-Phase Iterative Solver

No separate explore/solve phases. The solver IS the explorer:

```
loop {
    1. update_vision()        → reveal nearby tiles
    2. discover_targets()     → record newly seen items
    3. choose_action()        → strategy-specific (THE KEY DIFFERENCE)
    4. execute_action()       → move, interact, boss moves
    5. if solved: break       → at goal, all boxes open, bridge open
}
```

Each action is ONE step (up/down/left/right). The solver picks one step at a time
based on its strategy. Total steps to solve = the score.

### Three Exploration Strategies

**🐻 BF — Greedy Nearest Frontier:**
- Find nearest tile adjacent to unexplored area (BFS from current position)
- Move toward it
- No intelligence about what might be there
- Handles interactions passively (picks up keys it walks over, toggles levers it stands on)
- Key-lock: tries keys in order until one works (brute force)
- Lever: tries levers it discovers, checks bridge after each

**🐰 AI — Heuristic Frontier Scoring:**
- Score each frontier tile by strategic value:
  - +3 near chokepoints (corridor entrances, bridge area)
  - +2 near dead-ends (likely to contain items)
  - +1 per adjacent unseen tile (more unknown = more valuable)
  - -2 if already visited recently (avoid backtracking)
- After discovering some targets, adjust strategy:
  - Has keys but no boxes → boxes probably beyond bridge → prioritize opening bridge
  - Has some levers → try combinations to open bridge
  - Bridge open → prioritize exploring beyond bridge
- Key-lock: reasons about which key fits (tries adjacent key first)
- Lever: uses Occam's razor (fewer toggles first, like current discover_levers)

**🦊 Hybrid — AI Region + BF Path:**
- AI decides WHICH frontier region to explore (macro strategy)
  - Region = cluster of nearby frontiers
  - AI scores regions by strategic value (same heuristics as 🐰)
- BF finds shortest path to best frontier in chosen region (micro tactics)
- Key-lock: AI reasons, but if reasoning fails, BF tries all keys
- Lever: AI discovers which levers matter, BF tries orderings
- When stuck (no good frontier): falls back to BF nearest frontier

### Expected Differentiation

On a 16×16 map with bridge chokepoint:

| Metric | 🐻 BF | 🐰 AI | 🦊 Hybrid |
|--------|-------|-------|-----------|
| Exploration steps | High (random-ish) | Low (directed) | Medium-Low |
| Backtracking | Frequent | Rare | Minimal |
| Bridge discovery | Late (stumbles on it) | Early (seeks chokepoints) | Early (AI targets bridge area) |
| Lever solve | Tries all orderings | Occam's razor | AI prunes + BF orders |
| Total steps | Worst | Best or near-best | Close to AI, sometimes better |

**When Hybrid beats AI**: When AI's frontier scoring misranks a region (heuristic wrong)
but Hybrid's BF fallback still finds a good path. AI commits to wrong direction, Hybrid
adapts.

**When Hybrid beats BF**: Always, because AI-guided region selection avoids BF's
random wandering.

## Key Insight for Research

The fog-of-war approach creates **genuine information asymmetry** between solvers:
- BF explores blindly → high exploration cost
- AI explores intelligently → low exploration cost, but may commit to wrong heuristic
- Hybrid balances → AI guides exploration, BF ensures path quality

This is the first condition where the three solvers should produce **measurably different**
step counts, because each solver discovers targets in a different order and takes
different paths through the unknown map.

## Implementation Notes

- Reuse game engine from `tactical_07_strategic.rs` (StrategicGame, StrategicState)
- Add fog layer on top (FogState, vision computation)
- No DDTree for the iterative solver (too expensive for single-step decisions)
- DDTree only used if we add a "replan with known info" phase after full discovery
- Boss and traps still apply during exploration (adds time pressure)
- TUI shows fog as: hidden=dark gray, seen=dim, visible=bright, discovered items=highlighted