# Validator — Constraint Validation + Transformer-VM

> **What we sell here.** Deterministic constraint pruning that keeps a frozen
> decoder honest (Sudoku grids, Rust ASTs), plus Percepta — a transformer
> repurposed as a virtual machine.

## Docs

| Doc | Role |
|---|---|
| [`constraint_validator.md`](constraint_validator.md) | Deterministic constraint validator — Sudoku + Rust-AST pruning, `ConstraintPruner` trait |
| [`percepta.md`](percepta.md) | Percepta — transformer-VM in Rust (2D convex-hull attention, WASM in weights) |

## See also

- [`../06_game_arenas/sudoku.md`](../06_game_arenas/sudoku.md) — the arena that drives the Sudoku constraint path
