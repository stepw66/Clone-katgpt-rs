# Issue 124 — Spectral Rewiring SVD Column Cap (64-col substrate limit)

**Opened:** 2026-07-10
**Discovered by:** Plan 423 Phase 3 GOAT gate (`.benchmarks/423_spectral_rewire_goat.md`)
**Resolved:** 2026-07-10 (same day — substrate upgrade landed)
**Severity:** blocks 128×128 / 512×512 spectral_rewire (and any future consumer needing >64-col SVD)
**Type:** refactor / optimization (substrate upgrade)
**Status:** RESOLVED

## Problem

The one-sided Jacobi SVD in `katgpt-core/src/subspace_phase_gate.rs`
(`thin_svd_into` / `one_sided_jacobi_svd_into`) uses **fixed-size stack arrays**
for the singular-value argsort:

```rust
// subspace_phase_gate.rs ~L812-828
debug_assert!(n <= 64, "one-sided Jacobi result scratch supports n <= 64");
let mut raw_sigma: [f32; 64] = [0.0; 64];   // ← fixed cap
for i in 0..n { ... raw_sigma[i] = ...; }   // OOB when n > 64
let mut perm: [usize; 64] = [0; 64];         // ← fixed cap
```

This caps the SVD at `n_cols ≤ 64`. In release builds the `debug_assert!`
compiles away, so `n_cols > 64` silently writes out of bounds (undefined
behavior) rather than panicking.

## Impact on spectral_rewire

`katgpt_spectral::spectral_rewire` factors `W₀` as `(d_out × d_in)` via
`thin_svd_into`, so `d_in` is the capped dimension. **Supported domain: `d_in ≤ 64`.**
This blocks the Plan 423 GOAT gate targets 128×128 and 512×512 (both have
`d_in > 64`).

The cap does NOT bind `d_out` (the row count) — e.g. 512×64 works. A transpose
fallback (factor `W₀ᵀ` when `d_in > 64` but `d_out ≤ 64`) would help non-square
cases but NOT square matrices >64×64.

## Mitigation already shipped (Plan 423 Phase 3)

`spectral_rewire_into` and `SpectralRewireIndex::new` now guard `d_in > 64` with
a clear panic message referencing this issue + the `SVD_MAX_COLS` constant. The
opaque OOB is now a documented, surfaced contract violation.

## Proposed fix

Heap-allocate the argsort buffers in `one_sided_jacobi_svd_into`:

- `raw_sigma`: `[f32; 64]` → `&mut [f32]` slice from `SvdScratch` (add a
  `sigma_buf: Vec<f32>` field, sized `n_cols`).
- `perm`: `[usize; 64]` → `&mut [usize]` slice from `SvdScratch` (add a
  `perm_buf: Vec<usize>` field).

This removes the cap with zero hot-path cost (the buffers live in the reusable
`SvdScratch`, sized once via `with_capacity`). The insertion sort stays O(n²)
but n is bounded by the matrix dimension (no LLM-scale matrices in this repo).

## Audit scope (do NOT break existing consumers)

`thin_svd_into` / `SvdScratch` / `SvdResultScratch` have **16+ consumers**
(Tucker HOSVD, off_principal, subspace_phase_gate itself, Plan 301, Issue 008,
Issue 043, Plan 312, NeuronShard, HLA eigenbasis, etc.). The fix must:

- [x] Add `sigma_buf` / `perm_buf` to `SvdScratch` (sized in `with_capacity`).
- [x] Update `one_sided_jacobi_svd_into` to use the heap slices.
- [x] Remove the `n <= 64` `debug_assert!` and the `[f32; 64]` / `[usize; 64]`.
- [x] Re-run ALL `katgpt-core` SVD tests (`subspace_phase_gate` test module,
  Tucker HOSVD bench, off_principal tests) — bit-identical output required.
  **2974/2974 tests pass with `--all-features`. Output is bit-identical** (only
  buffer location changed, not the algorithm).
- [x] Add a regression test: SVD of a 128×128 matrix (was: panic/OOB).
  `thin_svd_into_128x128_exceeds_old_64_col_cap` — 128 singular triples,
  σ_max=128, σ_min=1, full rank.
- [x] Re-run the Plan 423 GOAT gate at 128×128 / 512×512 and record results.
  **G1a 128×128 r=16 PASS** (fraction=1.000019, rel err=1.948e-5). G1b 128×128
  random-delta fraction=0.1340 (not concentrated, as expected). The
  `SVD_MAX_COLS` cap in `katgpt-spectral` has been REMOVED; the cap-guard test
  replaced by `spectral_rewire_works_for_d_in_above_old_64_cap` (d_in=80, PASS).
  512×512 was NOT added to the gate (the one-sided Jacobi SVD at 512 cols would
  take ~minutes per call due to O(n²) per sweep × 60 sweeps — the cold-tier
  latency would be unacceptable; 128×128 is the practical bound for the
  cold-tier SVD path).

## Out of scope for Plan 423

This is a `katgpt-core` substrate change, not a `katgpt-spectral` primitive
change. It belongs in its own plan/commit. Plan 423's GOAT gate passed on the
supported domain (`d_in ≤ 64`); this issue is about EXPANDING that domain.

## Cross-references

- **Plan 423** — spectral_rewire primitive (the consumer that hit the cap).
- **`.benchmarks/423`** — the GOAT gate run that discovered the cap.
- `katgpt-core/src/subspace_phase_gate.rs` ~L808-828 — the fixed arrays.
