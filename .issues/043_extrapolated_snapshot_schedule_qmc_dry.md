# Issue 043 — `extrapolated_snapshot_schedule_qmc` DRYs against the BLAKE3 variant

**Filed:** 2026-07-04
**Priority:** P2 (DRY duplication with maintenance risk — two near-identical 40-line functions; future changes to the assertion/resize/coeff math must be applied twice)
**Origin:** Post-commit survey of Plan 367 Fusion C (commit `162a645c`, 2026-07-04).
**Blocks:** Nothing. **Blocked by:** Nothing.
**Type:** Refactor (behavior-preserving; bit-identical output pre/post).

---

## Context

Plan 367 Fusion C added `extrapolated_snapshot_schedule_qmc` to
`crates/katgpt-core/src/diversity/temp.rs` as the QuasiMoTTo (QMC) variant of
the existing `extrapolated_snapshot_schedule`. The QMC variant swaps the
BLAKE3-per-`j` noise draw for a single `source.draw(k, uniforms_scratch)` and
then maps `u_j ∈ [0,1)` to `[-noise_sigma, +noise_sigma]` via the affine
transform `(u_j * 2 − 1) * noise_sigma`.

## Finding

`crates/katgpt-core/src/diversity/temp.rs`:

| Function | Lines | What it does |
|---|---|---|
| `extrapolated_snapshot_schedule` | 90–128 | BLAKE3-noise variant |
| `extrapolated_snapshot_schedule_qmc` | 148–228 | QMC-noise variant |

The two functions share **identical** structure for ~30 of ~40 lines:

1. **Assertion block** (temp.rs:98–108 vs 168–183) — both assert `s0.len() == s1.len()`, `lambda_schedule.len() == out.len()`, plus a length check on the noise source (`noise_seeds.len() == out.len()` vs `uniforms_scratch.len() >= out.len()`). Identical except for the noise-source length predicate.
2. **Per-`j` resize guard** (temp.rs:112–115 vs 191–194) — `if theta_j.len() != d { theta_j.resize(d, 0.0); }`. Bit-identical.
3. **`coeff` computation** (temp.rs:121 vs 200) — `let coeff = lambda_schedule[j] * (1.0 + xi_j);`. Bit-identical.
4. **Inner `d`-loop** (temp.rs:122–126 vs 201–205) — `theta[i] = s0[i] + coeff * (s1[i] - s0[i]);`. Bit-identical.

The **only** genuine difference is how `xi_j` is computed:

- BLAKE3 (temp.rs:116–120): `blake3_noise(noise_seeds[j], noise_sigma)` per `j`.
- QMC (temp.rs:185–199): one bulk `source.draw(k, uniforms_scratch)` before the loop, then `(uniforms_scratch[j] * 2.0 - 1.0) * noise_sigma` per `j`. The bulk draw is gated by `if noise_sigma != 0.0` (skipped when noise is off).

## Impact

**Maintenance risk.** Any future change to:
- the `coeff = lambda * (1 + xi)` formula,
- the resize-on-mismatch guard,
- the inner `theta[i] = s0[i] + coeff * (s1[i] - s0[i])` hot loop,
- a new dimension assertion,

must be applied in two places. The two are currently kept in sync manually; there is no compile-time guarantee they stay aligned.

**No perf cost today.** Both functions are zero-allocation (caller-provided `out: &mut [Vec<f32>]` and either `noise_seeds: &[u64]` or `uniforms_scratch: &mut [f32]`). The duplication is a correctness/maintenance risk, not a hot-loop bottleneck.

## Proposed fix

Extract a private helper that takes a closure (or trait object) producing the
per-`j` noise scalar:

```rust
/// Shared extrapolation core: writes `theta_j = s0 + lambda_j·(1+xi_j)·(s1−s0)`
/// for j in 0..k, where `xi_j` is produced by `noise_fn(j)`.
///
/// Caller pre-draws noise (BLAKE3 per-j, or QMC bulk-then-indexed) and passes
/// a closure `Fn(usize) -> f32` returning `xi_j` for index `j`.
fn extrapolated_snapshot_schedule_with_noise(
    s0: &[f32],
    s1: &[f32],
    lambda_schedule: &[f32],
    out: &mut [Vec<f32>],
    mut noise_fn: impl FnMut(usize) -> f32,
) {
    assert_eq!(s0.len(), s1.len(), "s0 and s1 must have same dimension");
    assert_eq!(lambda_schedule.len(), out.len(),
        "lambda_schedule and out must have length k");
    let d = s0.len();
    for (j, theta_j) in out.iter_mut().enumerate() {
        if theta_j.len() != d { theta_j.resize(d, 0.0); }
        let xi_j = noise_fn(j);
        let coeff = lambda_schedule[j] * (1.0 + xi_j);
        let theta = theta_j.as_mut_slice();
        for i in 0..d {
            theta[i] = s0[i] + coeff * (s1[i] - s0[i]);
        }
    }
}
```

Then:

```rust
pub fn extrapolated_snapshot_schedule(
    s0: &[f32], s1: &[f32], lambda_schedule: &[f32],
    noise_seeds: &[u64], noise_sigma: f32, out: &mut [Vec<f32>],
) {
    assert_eq!(noise_seeds.len(), out.len(),
        "noise_seeds and out must have length k");
    extrapolated_snapshot_schedule_with_noise(
        s0, s1, lambda_schedule, out,
        |j| if noise_sigma == 0.0 { 0.0 } else { blake3_noise(noise_seeds[j], noise_sigma) },
    );
}

pub fn extrapolated_snapshot_schedule_qmc(
    s0: &[f32], s1: &[f32], lambda_schedule: &[f32],
    source: &mut dyn crate::speculative::QmcSource,
    noise_sigma: f32, out: &mut [Vec<f32>], uniforms_scratch: &mut [f32],
) {
    let k = out.len();
    assert!(uniforms_scratch.len() >= k, /* ... */);
    if noise_sigma != 0.0 { source.draw(k, uniforms_scratch); }
    extrapolated_snapshot_schedule_with_noise(
        s0, s1, lambda_schedule, out,
        |j| if noise_sigma == 0.0 {
            0.0
        } else {
            (uniforms_scratch[j] * 2.0 - 1.0) * noise_sigma
        },
    );
}
```

**Inlining note:** `noise_fn` is a closure passed by value (not `dyn`); with `#[inline]` on the helper, LLVM should monomorphize and inline the closure at each call site, preserving the current per-`j` branch elision when `noise_sigma == 0.0`. Verify with a bench (the BLAKE3 path's G4 budget is `2.46µs` — the refactor must not regress this).

### Why not a trait-based unification

A trait like `trait NoiseSource { fn draw_into(&mut self, k: usize, out: &mut [f32]); }` with BLAKE3 and QMC impls would unify the **bulk draw**, but the BLAKE3 variant doesn't have a bulk draw today — it hashes per-`j`. Forcing one would either (a) allocate a `noise_seeds`-sized buffer to mimic bulk, or (b) require the BLAKE3 path to grow a `draw_into` impl that internally loops. The closure approach is strictly simpler and preserves the existing call shapes (no API break).

## Severity

**P2.** Real DRY violation with a clear maintenance hazard (two 40-line functions differing in ~5 lines), but:
- Not P1: no >2x perf cost, no file-size violation (temp.rs is 1515 lines, under the 2048 guideline).
- Not P0: no correctness bug today (the two are currently synchronized).
- Higher than P3: the duplication is in the hot-loop inner pattern (`theta[i] = s0[i] + coeff * v_i`), which is exactly the kind of code where a future SIMD/simd_dot_f32 optimization should land once, not twice.

## GOAT gate (if the refactor lands)

- **G1:** Bit-identical output pre/post on a sweep of `(s0, s1, lambda, sigma)` fixtures. The existing 6 tests in temp.rs (lines 878–1019 for QMC, plus the BLAKE3 variant's tests) must pass unchanged.
- **G2:** No latency regression on the BLAKE3 path (target ≤ `2.46µs`, the existing G4 budget from the temp_loss_fingerprint GOAT gate).
- **G3:** All existing tests pass.
- **G4:** Zero additional allocations (closure is stack-only, no captures that allocate).
- **G5/G6:** Trivially modelless (no behavior change).

This refactor is **not** a candidate for default-on promotion — it's a DRY refactor inside an already-default-on feature (`temp_loss_fingerprint` + `qmc_sampling`).

## Tasks

- [ ] **T1** Verify the closure-based extraction produces bit-identical output on the existing test fixtures (both BLAKE3 and QMC variants). Run `cargo test -p katgpt-core --features temp_loss_fingerprint,qmc_sampling --lib diversity::temp`.
- [ ] **T2** Land the refactor: extract `extrapolated_snapshot_schedule_with_noise`, rewrite both public functions as thin wrappers. No API break.
- [ ] **G2 re-bench** Run the existing `temp_loss_fingerprint` GOAT bench (`perturbed_loss_vector 2.46µs` baseline) and confirm no regression. If the closure indirection costs anything (it shouldn't — `#[inline]` + monomorphization), document and decide whether to keep the refactor.

## Non-Goals

- ❌ Trait-based `NoiseSource` unification — rejected above (forces BLAKE3 to grow a bulk-draw path it doesn't need).
- ❌ Touching `blake3_noise` itself — it's a leaf helper, correctly scoped.
- ❌ Touching `perturbed_loss_vector` or `select_diverse_subset` — they consume the schedule output but don't share the inner-loop pattern.
- ❌ Promoting any feature or changing defaults — this is a pure internal refactor.

## Cross-References

- **Plan:** `katgpt-rs/.plans/367_quasi_monte_carlo_sampling.md` (Fusion C)
- **Commit:** `162a645c feat(qmc): Plan 367 Fusion C — extrapolated_snapshot_schedule_qmc + QmcSource re-export`
- **Sibling issue:** Issue 044 (Heterogeneous `fit_into` post-accumulate DRY — same audit-first DRY pattern, different file).
- **Audit-first precedent:** Issue 042 (sigmoid gate DRY) — establishes the codebase convention of auditing before unifying. This issue's audit (T1) is the same shape.

## TL;DR

`extrapolated_snapshot_schedule_qmc` (temp.rs:148–228, Plan 367 Fusion C) duplicates ~30 of ~40 lines of `extrapolated_snapshot_schedule` (temp.rs:90–128) — the assertion block, per-`j` resize guard, `coeff` formula, and inner `d`-loop are bit-identical. The only genuine difference is the per-`j` noise scalar `xi_j` (BLAKE3 hash vs QMC affine map). Extract a private `extrapolated_snapshot_schedule_with_noise(s0, s1, lambda, out, noise_fn: impl FnMut(usize) -> f32)` helper; rewrite both public functions as thin wrappers around it. P2, behavior-preserving, no API break, GOAT-gated on bit-identical output + no latency regression.
