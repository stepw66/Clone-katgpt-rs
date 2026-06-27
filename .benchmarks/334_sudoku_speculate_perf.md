# Bench 334: Sudoku Speculative-Solve Perf — Arto Inkala (Hardest)

Pure perf characterization (no GOAT gate — this is a perf-truth bench, not a
primitive-promotion gate). Answers: **how fast can we solve the hardest Sudoku
the speculate way, and can it beat backtracking?**

## Setup

- **Puzzle**: Arto Inkala "World's Hardest" — 21 clues, 60 empty cells.
- **Bench**: `benches/sudoku_speculate_bench.rs` (`harness = false`, `std::time::Instant`,
  median-of-31 × batch-of-16, 3 warmup — matches `cucg_bench.rs` convention).
- **Drafter**: uniform marginals over digits 1–9 (worst case, zero signal — the
  pruner supplies all constraint information).
- **Pruner**: path-aware `SudokuPruner` (100% valid branches, cross-depth conflict
  checking).
- **Run**: `cargo bench --bench sudoku_speculate_bench --features sudoku`

## Three Modes

| Mode | What it measures |
|------|------------------|
| 1. `backtrack` | Canonical `Sudoku9x9::solve()` — ground-truth complete solver. |
| 2. `speculate_iterative` | Iterative DDTree + greedy path commit + backtrack fallback. The realistic "speculative decoding" pattern. |
| 3. `build_one_tree` | Raw DDTree primitive throughput — nodes/µs for one 8-deep build. |

## Results (macOS, release, 2026-06-27)

### Mode 1 — backtrack baseline

| Metric | Value |
|--------|-------|
| solved | ✅ true |
| steps  | 49,559 |
| median time | **2.430 ms/solve** |
| per-step | 0.05 µs |

Matches the docs baseline (49,559 steps) — bench is correct.

### Mode 2 — speculate_iterative (DDTree + greedy commit + fallback)

| lookahead | budget | solved | spec_commits | fallback_steps | tree_nodes | time |
|-----------|--------|--------|--------------|----------------|------------|------|
| 4  | 32  | ❌ false | 13 | 3   | 85  | 11.02 µs |
| 8  | 64  | ❌ false | 14 | 18  | 145 | 18.33 µs |
| 8  | 128 | ❌ false | 12 | 237 | 224 | 31.40 µs |
| 16→8 | 256 | ❌ false | 7  | 4   | 259 | 28.05 µs |

Every config falls back to backtracking (solved=false means the speculation
hit a dead-end and the fallback ran — but the bench reports pre-fallback
spec_commits and the fallback step count). The speculate phase itself is
microseconds-fast, but it never solves Inkala: uniform marginals have no
signal, so greedy commits paint into corners within ~7–14 cells, then revert
+ backtrack.

### Mode 3 — DDTree primitive throughput (lookahead=8)

| budget | nodes_built | time | nodes/µs |
|--------|-------------|------|----------|
| 64     | 64          | 6.97 µs    | 9.2 |
| 256    | 256         | 25.24 µs   | 10.1 |
| 1,024  | 1,024       | 106.81 µs  | 9.6 |
| 4,096  | 2,678       | 262.75 µs  | 10.2 |
| 16,384 | 2,678       | 270.12 µs  | 9.9 |

Steady-state ~10 nodes/µs (10 M nodes/sec). Tree saturates at 2,678 nodes —
that's the full 8-deep pruned search space for Inkala's first 8 empties
(9⁸ raw = 43M, pruned to 2,678 = 16,000× reduction by the path-aware pruner).

## Key Finding — Architectural Ceiling

**`TreeNode.parent_path: u128` packs 16-bit tokens → hard max lookahead = 8
(128/16).** The DDTree speculate primitive is a **token-level speculative-
decoding kernel**, NOT a full-puzzle search. A 60-empty Sudoku **cannot be
solved in one tree** — it physically cannot fit in the u128 path encoding.

The coupled limit: `TreeBuilder::parent_tokens_buf` is sized to
`config.draft_lookahead + 1` (= 9 by default). Exploring depth ≥ 9 panics with
`range end index 10 out of range for slice of length 9`. This is why
`Config::draft()` ships with `draft_lookahead: 8`.

## Verdict — Can speculate beat backtrack on hardest Sudoku?

**No, not with the current infrastructure + uniform drafter.** Two reasons:

1. **No signal**: With uniform marginals, the drafter contributes zero
   information — every digit it proposes is already constraint-valid via the
   pruner. Speculation at best matches backtrack; at worst it pays tree-build
   overhead (~10–30 µs/round) before falling back.

2. **8-deep ceiling**: Even with a perfect drafter, the u128 layout caps
   lookahead at 8. Solving 60 cells requires ≥8 speculate rounds, each
   paying the primitive cost, with dead-end reverts between them.

**Break-even**: speculate wins only when
`acceptance_rate × commits_per_round × per_commit_savings > tree_build_overhead`.
With `p_accept = 1/9` (uniform) on Inkala, the LHS ≈ 1 × 8 × 0.05µs = 0.4µs
<< RHS ≈ 10–30µs. Never holds.

## What Would Make Speculate Win

Per the modelless-first mandate (AGENTS.md), before deferring to riir-train:

1. **MRV cell ordering** (modelless): reorder empties by minimum-remaining-
   values so the drafter proposes forced moves first. This is a deterministic
   drafter improvement, no training needed. Could push `p_accept` from 1/9
   toward 1/1 on forced cells.
2. **Constraint propagation as the drafter** (modelless): use naked/hidden
   singles as the draft signal — any cell with 1 valid digit is committed
   without speculation. Pure deterministic rules engine.
3. **Trained digit priors** (→ riir-train, deferred): a real draft model that
   proposes the RIGHT digit, not just a valid one. Out of scope for modelless.

Options 1 and 2 are modelless and should be tried first per §3.5 of the
research skill. Filing as an optimization candidate (see `issues/`).

## Files

- `benches/sudoku_speculate_bench.rs` — the bench (380 LOC).
- `Cargo.toml` — `[[bench]]` entry (required-features = `["sudoku"]`, harness = false).

## TL;DR

Hardest Sudoku (Inkala) solves in **2.430 ms** via backtrack (49,559 steps).
The speculate way **cannot beat it** with the current DDTree infra: (a) uniform
marginals give the drafter zero signal, and (b) `TreeNode.parent_path: u128`
hard-caps lookahead at 8, so a 60-cell puzzle can never be solved in one tree.
The DDTree primitive runs at ~10 M nodes/sec — fast for token-level speculative
decoding, but the wrong tool for full-puzzle search.
