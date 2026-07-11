# Plan 413: Multi-scale V-cycle on cell complexes

## TL;DR

Ship a modelless multi-scale V-cycle primitive (`htno_v_cycle`) in
`katgpt-dec`, behind the `htno_v_cycle` Cargo feature. It fills the
multi-scale composition gap the shipped single-complex DEC operators
(`exterior_derivative`, `codifferential`, `hodge_laplacian`,
`hodge_decompose`) always had: those handle one resolution level; this
composes two (fine → coarse → fine).

- [x] A1. Create `katgpt-dec/src/htno.rs` with `htno_v_cycle` skeleton. Gate
      behind `htno_v_cycle` feature in `katgpt-dec/Cargo.toml`.
- [x] A2. Implement restrict step: fine vertex cochain → coarse via selector
      gather (O(1) per coarse vertex, no dense matvec — same gather-scatter
      pattern as `SheafMaps::selector`).
- [x] A3. Implement prolongate step: coarse solution → fine complex (adjoint
      scatter back along the same selector indices).
- [x] A4. Compose: restrict → `coarse_op` (caller-supplied) → prolongate.
      `htno_v_cycle_into` reuses pre-allocated scratch for the alloc-free
      steady-state path.
- [x] A5. **GOAT gate G1 (commutativity):** `dₖKc ∘ Rₖ = Rₖ₊₁ ∘ dₖK` verified
      on (a) a regular grid induced sub-complex and (b) an irregular (path)
      induced sub-complex. Both pass. The 2×2 aggregation coarsening is
      documented as non-commuting (its coarse edges are long-range, not fine
      edges) — see the `aggregation_coarsening_does_not_commute_documented`
      test.
- [x] A6. GOAT gate G2 (perf): micro-V-cycle is O(n_coarse · dim) restrict +
      O(n_coarse · dim) prolongate + caller coarse_op. The restrict/prolongate
      pair is strictly cheaper than rebuilding the complex at coarse
      resolution (which is O(n_coarse²) for the boundary matrices).
- [x] A7. GOAT gate G3 (no-regression): `cargo check -p katgpt-core` passes
      with and without `--features htno_v_cycle`.
- [x] A8. GOAT gate G4 (alloc-free hot path): `htno_v_cycle_into` allocates
      zero bytes beyond the two pre-allocated scratch cochains (restrict +
      coarse_solved), which are reused across calls. The only allocation is
      the output cochain when using the allocating `htno_v_cycle` entry point.
- [x] A9. Doc-comments describe the primitive as generic DEC math only.
- [x] A10. Forwarded through `katgpt-core` as `katgpt_core::dec::htno_v_cycle`.

## What ships

| Item | Where |
|---|---|
| `htno_v_cycle`, `htno_v_cycle_into` | `katgpt-dec/src/htno.rs` |
| `VCycleRestriction` (selector restriction map) | same |
| `VCycleScratch` (reusable scratch buffers) | same |
| `grid_coarsen_2x2` (2×2-block grid coarsening helper) | same |
| Cargo feature `htno_v_cycle` | `katgpt-dec/Cargo.toml`, forwarded in `katgpt-core/Cargo.toml` |
| Re-export | `katgpt_core::dec::{htno_v_cycle, VCycleRestriction, ...}` |

## Commutativity note (G1)

The identity `dₖKc ∘ Rₖ = Rₖ₊₁ ∘ dₖK` holds **by construction** when the coarse
complex is an induced sub-complex of the fine complex (coarse vertices = a
subset of fine vertices; coarse edges = the fine edges between them). For
aggregation coarsenings (2×2 blocks), the coarse edges connect representatives
that are multiple fine cells apart — these are long-range edges, not fine
edges, so the edge-level commutativity does not hold. The V-cycle still
provides coarse smoothing in that case; it is a smoother, not a d-commuting
transfer. This is documented in the `htno.rs` module docs and in the
`aggregation_coarsening_does_not_commute_documented` test.
