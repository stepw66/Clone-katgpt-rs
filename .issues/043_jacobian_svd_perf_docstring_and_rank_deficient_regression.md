# Issue 043: `jacobian_svd_at_into` perf — misleading docstring + rank-deficient regression

> **Opened:** 2026-07-07
> **Source:** Plan 409 Phase 2 latency gate failure (Research 388 Fusion A refuted)
> **Priority:** P2 (correctness-adjacent — the misleading docstring led to a wrong research verdict; the rank-deficient regression affects all real-world consumers)
> **Status:** Open

## Problem

Two perf-related problems in `jacobian_svd_at_into` (`katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs:436-513`) surfaced during Plan 409 Phase 2's latency bench. Both contributed to Research 388's Fusion A being refuted (the latency claim compared a misleading ~455 ns number against an assumed ~1 ms probe cost, getting both wrong).

### (a) Misleading docstring — the ~455 ns claim

The docstring at `subspace_phase_gate.rs:426-429` claims:

```
/// The benchmark breakdown for R^8→R^8:
/// - [`jacobian_svd_at`] (with 17-`Vec` conversion): ~830 ns/call
/// - [`jacobian_svd_at_into`] (this fn, zero alloc): ~455 ns/call
```

**This is misleading.** The ~455 ns was measured with a **trivial `f`** (identity or near-identity), where the SVD of the identity matrix converges in one Jacobi sweep (~417 ns confirmed by diagnostic bench). For a **realistic linear map** `f(x) = W·x` at R^8→R^8:

| `f` type | `jacobian_svd_at_into` latency |
|---|---|
| Identity (`f(x)=x`) | **417 ns** ← matches docstring |
| Flat linear map (full-rank W) | **3.9 µs** ← 8.6× the claim |
| `Vec<Vec<f32>>` linear map (full-rank) | **4.0 µs** |
| Rank-4 linear map (rank-deficient W) | **31 µs** ← 68× the claim |

The claim omits the Jacobian forward-difference cost (n+1 = 9 eval calls at n=8) and the SVD convergence cost on a non-trivial matrix.

### (b) Rank-deficient SVD perf regression

The one-sided Jacobi SVD (`one_sided_jacobi_svd_into`, line 639) is **8× slower** on rank-deficient matrices than full-rank matrices of the same size:

| Matrix type (8×8) | `jacobian_svd_at_into` latency | SVD sweeps (est.) |
|---|---|---|
| Full-rank dense | 3.9 µs | ~10 |
| Rank-4 (4 zero rows) | 31 µs | ~60 (hits `max_sweeps`) |

**Root cause:** null-space column pairs with norms hovering just above the `col_floor_sq` threshold (`frob_sq · tol² ≈ 3e-13` for the test matrix) pass the per-pair convergence check at line 727:

```rust
if apq.abs() <= tol * (app * aqq).sqrt() {
    continue; // Already diagonal in this plane.
}
```

When `aqq` (a null-space column's norm²) is ~1e-12 (above `col_floor_sq = 3e-13` but numerically null), the check `apq.abs() <= tol * sqrt(app * 1e-12)` can FAIL (because `apq` is also ~1e-12 from floating-point noise, but `tol * sqrt(app * 1e-12) ≈ 1e-13`), triggering a spurious noise rotation every sweep. This prevents the `!rotated` convergence break at line 751 and burns all 60 sweeps.

**This affects all real-world consumers:**
- HLA: rank 5 in a 64-dim embedding (59 null-space columns).
- NeuronShard: rank ≪ ambient dimension.
- Plan 312 (Viable Manifold Graph): Jacobians of low-rank belief kernels.
- Plan 301 (Subspace Phase Gate): the `N ≥ d` phase transition explicitly produces rank-deficient Jacobians at the transition boundary.

## Proposed fix

### (a) Docstring correction

Rewrite `subspace_phase_gate.rs:422-429` to:
1. State the ~455 ns figure is for the **SVD-only path** (trivial `f`, e.g. identity).
2. Add a latency table for realistic `f` types (linear full-rank, linear rank-deficient, nonlinear).
3. Note that the Jacobian forward-difference cost is `(n+1) × eval_cost` and dominates for expensive `f`.

### (b) Rank-deficient fast-path

Two options:

**Option 1 (recommended): raise `col_floor_sq`.** The current threshold `frob_sq * tol²` is too aggressive. A column with norm² = 1e-12 relative to frob_sq = 30 is at relative magnitude 3e-14 — well below numerical relevance. Raising the floor to `frob_sq * 1e-10` (or `sigma_max² * 1e-10`) would deflate these borderline null-space pairs without affecting signal column accuracy. The `AND` condition (both columns below floor → skip) at line 724 already prevents signal-null pairs from being skipped.

**Option 2: post-sweep column-norm check.** After each sweep, if a column's norm is below a relative threshold (e.g. `< sigma_max * 1e-5`, matching the `rank_threshold` at line 807), mark it as "converged null" and exclude it from future sweep rotations. This is more robust but requires tracking per-column state.

**Validation:** the existing test suite (`cargo test -p katgpt-core --lib` with `subspace_phase_gate` feature) must pass unchanged. Add a perf regression test: `jacobian_svd_at_into` on a rank-4 8×8 matrix must be ≤ 2× the full-rank 8×8 latency (currently 8×).

## Scope

- `katgpt-rs/crates/katgpt-core/src/subspace_phase_gate.rs` — docstring + `one_sided_jacobi_svd_into` convergence logic.
- No API changes. No feature flag changes. Pure perf + docs.

## Non-goals

- **Does NOT recover Research 388 Fusion A.** Even with a fixed SVD, the prefilter needs n+1 eval calls vs the probe's 5 — structurally slower for n ≥ 4. This issue fixes the shipped primitive's perf and docs; the Fusion A verdict remains refuted.
