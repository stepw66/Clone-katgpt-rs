# Issue 050: Linking-Fold Detector Cold-Path Perf — Budget Recalibration or Optimization

**Date:** 2026-07-07
**Source plan:** [katgpt-rs/.plans/410_linking_fold_primitive.md](../.plans/410_linking_fold_primitive.md) (Phase 4 T4.1/T4.4)
**Type:** Optimization / budget-acceptance decision (blocks default-on promotion of `linking_fold`)
**Status:** OPEN — promotion of `linking_fold` to `default` blocked until resolved

## Summary

The `linking_detector` (paper Algorithm 1, `crates/katgpt-core/src/linking_fold/linking_detector.rs`) fails its original G2 perf budget. Plan 410 specified **≤ 50 ms @ n = 2×1000 point clouds, d = 8**. The brute-force implementation measures **~410 ms @ n = 2×200** and extrapolates to **minutes @ n = 2×1000**. The fold hot-path (`fold_projection_into` / `fold_gelu_into`) passes all gates cleanly (12–17 ns/call, 0 allocs); only the detector is over budget.

`linking_fold` ships **opt-in** until this issue is resolved.

> **Note on numbering:** Issue 049 was briefly used by a sibling agent for a CHaRS×CommittedFieldBlend topic and then removed (commits `2a3aff33` → `cc5c3ab3`). This linking-fold detector issue is renumbered to 050 to avoid the historical collision in `git log` searches.

## Measured evidence (Plan 410 Phase 4 T4.1, 2026-07-07)

Bench: `cargo bench -p katgpt-core --features linking_fold --bench bench_410_linking_fold_goat`

| n (per cloud) | d | median latency | note |
|---|---|---|---|
| 80 | 3 | ~25 ms | lib-test scale (`#[cfg(test)]` fixtures) |
| 200 | 8 | **407 ms** | bench G2 gate (recalibrated audit budget 500 ms ✅) |
| 1000 | 8 | minutes (extrapolated, bench hung) | original plan target — **not measured directly** |

Fold hot-path (the actually-useful per-tick primitive): **12.5 ns @ D=8 (Abs), 16.1 ns @ D=8 (Gelu), 16.8 ns @ D=64 (Abs), 16.9 ns @ D=64 (Gelu)** — all well under the 50 ns / 500 ns budgets.

## Root cause

The detector is `O(β_X · β_Y · L² · N_sub²)` where:
- `β` = cycle rank of the k-NN graph ≈ `E − V + C`. For `k = 8`, `β` grows roughly linearly with `n` (~3 β per node after ε-pruning).
- `L` = mean cycle length (~8–20 vertices).
- `N_sub` = Gauss-quadrature subdivisions per edge (default 4).

At `n = 200`, `β ≈ 400` per cloud → `β_X · β_Y ≈ 160 k` cycle pairs, each a `~2500`-op Gauss integral → ~400 ms. The cost is quadratic-ish in `n` because `β` scales with `n` and the pair count is `β_X · β_Y`.

The paper (§H) presents Algorithm 1 as a reference implementation; it does not claim sub-50ms perf. The 50ms@n=2000 budget in Plan 410 was set from a complexity estimate that underestimated `β`.

## Resolution options (pick one)

### Option A — Accept the recalibrated audit-cadence budget (promote as-is)

The detector is explicitly **audit-cadence** (run once per session / sleep-cycle, not per-tick) — see the `linking_detector.rs` module doc. A **500 ms budget @ n = 2×200** is fit-for-purpose for that cadence. Under this option:
- Update Plan 410's G2 criterion from "≤ 50 ms @ n = 2×1000" to "≤ 500 ms @ n = 2×200 (audit cadence)".
- Promote `linking_fold` to `default` with a comment noting the detector's audit-cadence characterization.
- Document the n-scaling cliff so consumers know not to call `detect_linking` on >n=500 clouds without subsampling.

**Cost:** zero implementation work. **Risk:** consumers who ignore the audit-cadence guidance and call the detector per-tick will tank perf.

### Option B — Optimize the detector (promote after the gate passes at the original budget)

Concrete optimizations, in priority order:
1. **Early-exit on first non-zero link** — already implemented (the `detect_linking_into` loop returns on the first witness). The cost is dominated by the *unlinked* case, which must exhaust all pairs. Add a **batch early-exit**: after every `K` pairs with all-zero integrals, check a cheaper necessary condition (e.g., bounding-box separation in PCA-3D space) and short-circuit.
2. **Cycle-basis pruning** — only the *long* cycles (length ≥ some threshold relative to the inter-cloud distance) can carry linking. Drop short cycles before the pair loop.
3. **Reduce default `k_neighbors`** from 8 to 6 — halves `β` at the cost of missing some thin links. Make it a config knob, document the trade-off.
4. **Spatial index for k-NN** — replace brute-force `O(n²)` k-NN with a kd-tree (`kd-tree` crate, ~2k LoC, no unsafe). This helps the k-NN step but not the dominant Gauss-integral pair loop.

Realistic target after (1)+(2): **≤ 50 ms @ n = 2×500** (still not n=2000, but a 10× improvement). Getting to n=2000 likely requires (4) + algorithmic work on the Gauss integral (FFT acceleration of the double line integral — non-trivial).

**Cost:** 1–2 days. **Risk:** optimization bugs that misclassify linked clouds as unlinked (silent false negatives — the detector's failure mode must remain "false positive on noise," not "false negative on real links").

### Option C — Split the feature

Split `linking_fold` into `linking_fold_fold` (the hot-path fold, all gates pass → promote to default) and `linking_fold_detector` (cold-path, stays opt-in). This lets the valuable primitive ship default-on without waiting on detector optimization.

**Cost:** small (feature-flag split + Cargo.toml rework). **Risk:** consumers who want the detector must enable two features; minor ergonomic cost.

## Recommendation

**Option A** if the detector is only ever called from audit/sleep-cycle paths (verify by grepping call sites once consumers exist — currently zero in-tree consumers, so the risk is hypothetical). **Option C** if the fold is expected to be used independently of the detector (likely — the fold is the per-tick correction, the detector is the rare audit trigger). **Option B** only if a real consumer hits the perf wall at audit cadence.

Default-on promotion is **BLOCKED** until one of A/B/C is chosen. The fold passing all gates is not sufficient on its own because the feature bundles both primitives under one flag.

## Cross-references

- Plan: [katgpt-rs/.plans/410_linking_fold_primitive.md](../.plans/410_linking_fold_primitive.md)
- Research: [katgpt-rs/.research/391_Low_Dimensional_Topology_Linking_Number.md](../.research/391_Low_Dimensional_Topology_Linking_Number.md)
- Bench: `crates/katgpt-core/benches/bench_410_linking_fold_goat.rs`
- Alloc test: `crates/katgpt-core/tests/linking_fold_alloc_check.rs`
- Detector source: `crates/katgpt-core/src/linking_fold/linking_detector.rs`
