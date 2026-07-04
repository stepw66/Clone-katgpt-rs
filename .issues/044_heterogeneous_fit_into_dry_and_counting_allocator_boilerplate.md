# Issue 044 — Heterogeneous `fit_into` post-accumulate solve block DRY + `CountingAllocator` test boilerplate

**Filed:** 2026-07-04
**Priority:** P2 (fit_into DRY) + P3 (test boilerplate) — two related DRY observations from the Plan 376 Phase 4 + Phase 6 survey.
**Origin:** Post-commit survey of Plan 376 (commits `8de9e189` Phase 4 + `b5c3d515` Phase 6, 2026-07-04).
**Blocks:** Nothing. **Blocked by:** Nothing.
**Type:** Refactor (behavior-preserving).

---

## Finding A (P2) — `HeterogeneousEnsemble::fit_into` duplicates the post-accumulate solve block

### Context

Plan 376 Phase 4 added a heterogeneous-D variant of the velocity-field ensemble:
`HeterogeneousEnsemble<P, D>` where each field has its own native dim `d_i`,
transported to the common `D` via Cross-Resolution bases (Plan 310). The fit
math is the same regression-optimal ridge solve as the homogeneous
`VelocityFieldEnsemble<P, D>`; only the per-pair accumulation differs (each
field's output is transported before the dot product).

### Finding

`crates/katgpt-core/src/velocity_field_ensemble.rs`:

| Function | Lines | Variant |
|---|---|---|
| `VelocityFieldEnsemble::fit_into` | 373–447 | Homogeneous (all fields share `D`) |
| `HeterogeneousEnsemble::fit_into` | 1190–1237 | Heterogeneous (per-field native dim + transport) |

The **post-accumulate normalization + ridge-solve block is duplicated verbatim**:

Homogeneous (lines 414–446):
```rust
let inv_n = 1.0 / (n as f32);
for g in scratch.gram.iter_mut() { *g *= inv_n; }
for r in scratch.rhs.iter_mut()   { *r *= inv_n; }

scratch.gram_reg[..P * P].copy_from_slice(&scratch.gram[..P * P]);
for i in 0..P {
    scratch.gram_reg[i * P + i] += lambda;
}

ridge_solve_direct_f32(
    &mut self.eta,
    &mut scratch.chol[..],
    &mut scratch.z_solve,
    &scratch.gram_reg[..],
    &scratch.rhs,
    P,
    1,
);
```

Heterogeneous (lines 1215–1236): **byte-for-byte identical** except for the
scratch type (`HeterogeneousFitScratch` vs `EnsembleFitScratch`). The two
scratches share the `gram`, `rhs`, `gram_reg`, `chol`, `z_solve` field names
(see Finding A.2 below).

The **per-pair accumulation** genuinely differs:
- Homogeneous `accumulate_pair_into` (lines 314–343): direct `fields[i].eval_into(i_t, &mut scratch.b_out_i)` and dot.
- Heterogeneous `accumulate_pair_heterogeneous_into` (lines 1279–1316): native eval → `project_to_spectral_into` → `reconstruct_from_spectral_into` → dot.

So the unification point is **after accumulation, before solve** — the
normalize-by-N, build `K + λI`, call `ridge_solve_direct_f32` sequence.

### Finding A.2 — Scratch structs share 5 of 7 fields

`EnsembleFitScratch<P, D>` (lines 248–264):
```
gram: Vec<f32>, rhs: [f32; P], gram_reg: Vec<f32>, chol: Vec<f32>,
z_solve: [f32; P], b_out_i: [f32; D], b_out_j: [f32; D]
```

`HeterogeneousFitScratch<P, D>` (lines 1330–1354):
```
gram: Vec<f32>, rhs: [f32; P], gram_reg: Vec<f32>, chol: Vec<f32>,
z_solve: [f32; P], b_at_d_i: [f32; D], b_at_d_j: [f32; D],
spectral_buf: Vec<f32>, native_buf_i: Vec<f32>, native_buf_j: Vec<f32>
```

The first 7 fields are **semantically identical** (just renamed `b_out_*` →
`b_at_d_*` in the heterogeneous variant — both hold "field output projected to
D"). The heterogeneous variant adds 3 transport-only buffers.

### Impact

**Maintenance risk.** A future change to:
- the ridge regularization formula (e.g., adaptive `λ`, Tikhonov with `Γ ≠ I`),
- the normalization scheme (e.g., weighted pairs, leave-one-out CV),
- the solver call (e.g., switching to LDLᵀ or QR for rank-deficient Gram),

must be applied in both `fit_into` functions and both scratch structs. Today
they're synchronized by manual inspection; there's no compile-time guarantee.

**No perf cost today.** Both paths are zero-allocation after scratch
construction (verified by `tests/{velocity_field_ensemble,heterogeneous_velocity_field_ensemble}_alloc_check.rs`).

### Proposed fix

**Option 1 (recommended): free function `solve_ridge_eta_into`.** Extract the
post-accumulate solve block into a free function taking the scratch fields by
reference. No struct unification needed — both scratches expose the same field
names, so a function generic over `&mut [f32]` slices works:

```rust
/// Normalize Gram + RHS by N, add ridge λI, solve (K + λI) η = r via
/// Cholesky. Writes the P-dim solution into `eta`.
///
/// Shared by `VelocityFieldEnsemble::fit_into` and
/// `HeterogeneousEnsemble::fit_into` — the per-pair accumulation differs
/// (the heterogeneous variant transports each field to D first), but the
/// solve-after-accumulate math is identical.
#[inline]
fn solve_ridge_eta_into<const P: usize>(
    eta: &mut [f32; P],
    gram: &mut [f32],        // P*P, row-major
    rhs: &mut [f32; P],
    gram_reg: &mut [f32],    // P*P scratch
    chol: &mut [f32],        // P*P scratch
    z_solve: &mut [f32; P],
    n: usize,
    lambda: f32,
) {
    let inv_n = 1.0 / (n as f32);
    for g in gram.iter_mut() { *g *= inv_n; }
    for r in rhs.iter_mut()  { *r *= inv_n; }

    gram_reg[..P * P].copy_from_slice(&gram[..P * P]);
    for i in 0..P { gram_reg[i * P + i] += lambda; }

    ridge_solve_direct_f32(eta, chol, z_solve, gram_reg, rhs, P, 1);
}
```

Both `fit_into` functions call it after their respective accumulation loops.
No struct changes, no API break, no feature-flag changes.

**Option 2 (rejected): unify the scratch structs.** A trait like
`trait FitScratch { fn gram_mut(&mut self) -> &mut [f32]; ... }` would let
`fit_into` be generic over scratch type, but:
- The homogeneous `b_out_i`/`b_out_j` are `[f32; D]` (fixed-size, const-generic) while the heterogeneous ones live inside a struct that also holds `Vec<f32>` transport buffers — the field-access boilerplate would dwarf the savings.
- The two `fit_into` functions have different `accumulate_pair_*` calls, so they can't share a body even with a unified scratch.
- Option 1 captures 100% of the duplication with zero struct churn.

Pick Option 1.

### Severity

**P2.** Real DRY violation (~30 lines of byte-identical solve logic + 5 of 7 shared scratch fields), but:
- Not P1: no perf cost, no file-size violation (velocity_field_ensemble.rs is 1583 lines, under the 2048 guideline).
- Not P0: no correctness bug today.
- Higher than P3: the solve block is exactly where future numerical-methods improvements (adaptive λ, LOO-CV, rank-deficient fallback) would land, and those must be applied once, not twice.

---

## Finding B (P3) — `CountingAllocator` boilerplate duplicated across 7+ test files

### Context

Plan 376 Phase 6 added `tests/heterogeneous_velocity_field_ensemble_alloc_check.rs`,
which follows the existing `CountingAllocator` global-alloc pattern for G3
(zero-alloc) verification.

### Finding

`grep CountingAllocator crates/katgpt-core/tests/**/*.rs` returns 7+ matches:
- `tests/analytic_lattice_alloc_check.rs`
- `tests/conformal_alloc_check.rs`
- `tests/velocity_field_ensemble_alloc_check.rs`
- `tests/heterogeneous_velocity_field_ensemble_alloc_check.rs` (new, Plan 376 Phase 6)
- `tests/bench_331_babel_codec_goat.rs`
- `tests/bench_360_engram_staging_goat.rs`
- (likely more — the grep was paginated)

Each defines a near-identical `struct CountingAllocator; static ALLOC_COUNT: AtomicUsize; static DEALLOC_COUNT: AtomicUsize; unsafe impl GlobalAlloc ...` block (~20 lines).

### Impact

**Maintenance cost.** Each new G3 alloc-check test copies ~20 lines of boilerplate. If the pattern ever needs to change (e.g., per-size counters, thread-local isolation), it must be updated in 7+ places.

**Not a perf concern.** Test-only code; doesn't ship.

### Proposed direction (audit-first, like Issue 042)

**Why this is P3 and not P2:** the duplication is in test infrastructure, not
shipped code. The "fix" is also non-trivial because each test binary needs its
**own** `#[global_allocator]` static, and Rust's `#[global_allocator]`
attribute requires a single static per crate (test binaries are effectively
single-crate). A `tests/common/counting_alloc.rs` helper module that each
test file `macro_rules!`-imports would work, but it's a non-trivial refactor
of working test code for marginal benefit.

**Recommendation:** file as P3 for visibility, defer until the next test-infra
cleanup pass. The pattern is stable and the copies are isolated — they don't
pose a correctness or perf risk.

If/when extracted, the shape would be a `macro_rules! counting_allocator!()`
that emits the struct + statics + `#[global_allocator]` line, called at the
top of each alloc-check test file. ~5 lines of macro, replaces ~20 lines per
file.

### Severity

**P3.** Test-infra DRY; stable pattern; deferred.

---

## GOAT gate (if Finding A refactor lands)

- **G1:** Bit-identical `eta` output pre/post on the existing fit-recovery tests (`test_fit_recovers_known_eta` for homogeneous, `test_heterogeneous_fit_recovers_known_eta` for heterogeneous). Both must recover η to `<1e-4`.
- **G2:** No latency regression on `fit_into` (homogeneous baseline `6.27µs` from the Plan 376 Phase 3 GOAT gate; heterogeneous has no latency gate yet).
- **G3:** Zero allocations on both paths (existing alloc-check tests pass unchanged).
- **G4/G5/G6:** Trivially modelless (no behavior change).

This refactor is **not** a candidate for default-on promotion — both features
are already at their target promotion state (`velocity_field_ensemble`
default-on, `velocity_field_ensemble_heterogeneous` opt-in pending a concrete
consumer).

## Tasks

- [x] **T1 (Finding A)** Extract `solve_ridge_eta_into` free function. Rewrite both `fit_into` functions to call it after their respective accumulation loops. Verify bit-identical `eta` on existing fit-recovery tests.
  - **DONE 2026-07-04.** `solve_ridge_eta_into<const P: usize>` added as a private `#[inline]` free function at `velocity_field_ensemble.rs` (after `accumulate_pair_into`, before the `VelocityFieldEnsemble` impl). Both `VelocityFieldEnsemble::fit_into` and `HeterogeneousEnsemble::fit_into` now call it after their (genuinely different) accumulation loops. Verified: `cargo test -p katgpt-core --features velocity_field_ensemble,velocity_field_ensemble_heterogeneous --lib` → **1031 passed; 0 failed**, including `test_fit_recovers_known_eta` (homogeneous η recovery <1e-4) and `test_heterogeneous_fit_recovers_known_eta` (heterogeneous η recovery <1e-4). G1 bit-identical PASS. Alloc-check tests (`velocity_field_ensemble_alloc_check`, `heterogeneous_velocity_field_ensemble_alloc_check`) both PASS → G3 zero-alloc preserved on both paths.
- [-] **T2 (Finding A, optional)** Consider renaming `EnsembleFitScratch::b_out_{i,j}` → `b_at_d_{i,j}` (or vice versa) to match `HeterogeneousFitScratch`. Cosmetic; only worth it if T1 lands and the naming asymmetry becomes confusing.
  - **SKIPPED 2026-07-04.** T1 landed; naming asymmetry is NOT confusing in practice — the two structs are used in different code paths (`VelocityFieldEnsemble` vs `HeterogeneousEnsemble`) and the field names are local to each struct. A rename would touch ~15 references (struct def, constructor, `accumulate_pair_into`, doc comments, tests) for zero behavior change and marginal clarity improvement. Not worth the churn. The naming difference is documented in Finding A.2 above.
- [-] **T3 (Finding B, DEFERRED)** Audit `CountingAllocator` duplication across test files. If extracted, use a `macro_rules! counting_allocator!()` in a `tests/common/` module. P3, defer to next test-infra cleanup.

## Non-Goals

- ❌ Unifying `VelocityFieldEnsemble` and `HeterogeneousEnsemble` into a single generic struct — the const-generic `D` vs runtime `native_dim` mismatch makes this a major API redesign, not a DRY cleanup.
- ❌ Touching `accumulate_pair_*` — they genuinely differ (transport step); no DRY opportunity.
- ❌ Promoting `velocity_field_ensemble_heterogeneous` to default-on — that's gated on a concrete consumer emerging (per the Cargo.toml comment).

## Cross-References

- **Plan:** `katgpt-rs/.plans/376_velocity_field_ensemble_primitive.md` (Phase 4 + Phase 6)
- **Commits:** `8de9e189 feat(plan-376): Phase 4 — Heterogeneous-D velocity fields via Cross-Resolution transport`, `b5c3d515 feat(plan-376): Phase 6 — UQ conformal floor benchmark (BEATS FLOOR)`
- **Sibling issue:** Issue 043 (`extrapolated_snapshot_schedule_qmc` DRY — same audit-first pattern).
- **Sibling issue:** Issue 038 (`velocity_field_ensemble_uq_conformal_floor` — tracks the UQ-floor follow-up; this issue's Finding A is independent of that).
- **Audit-first precedent:** Issue 042 (sigmoid gate DRY).
- **Sibling ridge path:** `crates/katgpt-core/src/karc.rs::fit_direct` (Plan 308) — uses the same `ridge_solve_direct_f32`. If a third caller ever emerges, a shared `solve_ridge_eta_into` becomes even more valuable.

## TL;DR

Two DRY observations from the Plan 376 Phase 4 + 6 survey:

**Finding A (P2):** `HeterogeneousEnsemble::fit_into` (velocity_field_ensemble.rs:1190–1237) duplicates the post-accumulate normalize-by-N + build-`K+λI` + `ridge_solve_direct_f32` call block from `VelocityFieldEnsemble::fit_into` (lines 414–446) byte-for-byte. The two scratch structs also share 5 of 7 fields. Extract a free function `solve_ridge_eta_into(eta, gram, rhs, gram_reg, chol, z_solve, n, lambda)` and call it from both `fit_into` functions after their (genuinely different) accumulation loops. No struct unification needed.

**Finding B (P3, deferred):** `CountingAllocator` global-alloc boilerplate is copy-pasted across 7+ test files including the new `heterogeneous_velocity_field_ensemble_alloc_check.rs`. Stable test-infra pattern; defer to next cleanup pass; if extracted, use a `macro_rules! counting_allocator!()`.

Files confirmed clean (no findings): `qmc_halter.rs` (excellent — `#[repr(u8)]` on `QmcHaltReason`, zero-alloc `evaluate`, NaN-safe, comprehensive tests), `velocity_field_ensemble_uq_floor.rs`, `stochastic_interpolant_step_into`, `Schedule`, lib.rs feature-flag additions, `extrapolated_snapshot_schedule_qmc` test fixtures.
