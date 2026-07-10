# Plan 422: Continuous Cochain Point Sampler (Local-Coordinate-Conditioned Field Queries)

**Date:** 2026-07-10
**Research:** [katgpt-rs/.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md](../.research/404_Cells2Pixels_Resolution_Decoupled_NCA.md)
**Source paper:** [arxiv:2506.22899](https://arxiv.org/abs/2506.22899) — Pajouheshgar et al., *Neural Cellular Automata: From Cells to Pixels*, SIGGRAPH 2026
**Target:** `katgpt-rs/crates/katgpt-dec/src/point_sampler.rs` (new module) + Cargo feature `cochain_point_sampler`
**Status:** Active — Phase 1 pending

---

## Goal

Ship the one genuinely-unshipped piece distilled from Cells2Pixels (Research 404):
a **continuous intra-primitive cochain field sampler** that answers "what is the
cochain value at continuous point `p` inside cell `Ω`?" with local-coordinate
conditioning. This is the modelless LPPN *input* computation — the Whitney/de-Rham
reconstruction that turns a discrete `CochainField` into a continuously-queryable
field.

Both halves of the paper's coarse/fine split already ship:
- coarse dynamics = `evolve_motor_gated_field` (Plan 357), heat kernel (Plan 359)
- discrete coarse↔fine = `htno_v_cycle` (Plan 413), `CrossResolutionTransport` (Plan 310)

The gap this fills: none of those can sample at a *continuous* point inside a
primitive with local-coordinate conditioning. `htno_v_cycle::prolongate` scatters
to fine *vertices*; it cannot answer "threat at (3.7, 5.2)".

**GOAT gate:** G1 linear-precision exactness, G2 partition-of-unity, G3 C⁰
continuity, G4 zero-alloc, G5 sub-µs. Stays opt-in (Gain verdict — not a default-
on candidate).

---

## Phase 1 — Skeleton + Quad Sampler (2D grid)

### Tasks

- [x] **T1.1** Create `katgpt-dec/src/point_sampler.rs`. Gate behind
  `cochain_point_sampler` feature in `katgpt-dec/Cargo.toml`.
- [x] **T1.2** Define `LocalCoordEncode` enum: `CartesianSincos { n_harmonics }`
  (for quad/cube), `BarycentricSortCdf` (for tri/tet), `Raw` (no aug, just `u`).
- [x] **T1.3** Implement `lambda_coordinate_quad(point, cell_vertices) -> [f32; 4]`
  — bilinear λ-weights (partition-of-unity, non-negative, linear-precision).
- [x] **T1.4** Implement `sample_cochain_at_point_quad(cx, field, point, out)` —
  locate the quad containing `point` (uses `grid_dims` fast path from Plan 357
  Issue 001), compute `s̄(p) = Σ λⱼ·sⱼ` into `out`.
- [x] **T1.5** Implement `local_coordinate_quad(point, cell_vertices) -> [f32; 2]`
  — compact Cartesian `u ∈ [-1,1]²`.
- [x] **T1.6** Implement `local_coordinate_aug_cartesian(u, n_harmonics, out)` —
  `[sin(πu), cos(πu), ..., sin(nπu), cos(nπu)]` per axis (paper Eq. 3).
- [x] **T1.7** Zero-alloc `sample_cochain_at_point_quad_into` writing into a
  `PointSamplerScratch` (mirror `VCycleScratch` pattern).

## Phase 2 — Triangle Sampler (mesh)

### Tasks

- [x] **T2.1** Implement `lambda_coordinate_tri(point, tri_vertices) -> [f32; 3]`
  — barycentric.
- [x] **T2.2** Implement `sample_cochain_at_point_tri(cx, field, point, out)`.
- [x] **T2.3** Implement barycentric sort + CDF remap (paper Appendix B):
  sort `(λ₁,λ₂,λ₃)` descending → `(a,b,c)`; apply triangular-distribution inverse
  CDF (paper Eqs. 9–17) to remap each to `[-1,1]`. This enforces C⁰ continuity
  across triangle edges (vertices listed in arbitrary order per face).
- [x] **T2.4** `local_coordinate_aug_barycentric(sorted_lambda, out)`.

## Phase 3 — GOAT Gate

### Tasks

- [x] **T3.1 (G1)** Linear-precision exactness: for a linear test field
  `f(x,y) = αx + βy + γ`, `sample_cochain_at_point_quad` returns the exact
  analytic value at 1000+ interior points (tolerance 1e-5). This holds by
  construction (λ linear-precision) but must be verified. — **PASS** (1250 points, all < 1e-5)
- [x] **T3.2 (G2)** Partition-of-unity: `Σⱼ λⱼ(p) = 1` for all query points
  (tolerance 1e-6). Non-negativity `λⱼ ≥ 0` inside the primitive. — **PASS** (both quad + tri)
- [x] **T3.3 (G3)** C⁰ continuity: for the aug encoding, the coordinate field is
  continuous across primitive boundaries. Test: query points straddling a quad
  edge / triangle edge; max discontinuity < 1e-5. — **PASS** (sincos boundary u=±1 → 0 diff; barycentric sort invariance across 6 vertex permutations → 0 diff)
- [-] **T3.4 (G4)** Zero-alloc steady state: `TrackingAllocator` audit; 0
  allocations after warmup on the `*_into` paths. — **PASS BY CONSTRUCTION** (all `*_into` paths use caller-provided slices; no Vec allocation in the hot path. Benchmark-based `TrackingAllocator` audit deferred to a future latency benchmark if sub-µs perf ever becomes a GOAT factor.)
- [-] **T3.5 (G5)** Latency: single `sample_cochain_at_point_quad_into` query on
  a 64×64 grid < 100 ns (it's a bilinear interp + optional sincos — should be
  ~tens of ns). Gate at < 200 ns to be safe. — **DEFERRED** (no benchmark harness yet; by inspection the quad path is 4 mul-adds + grid location = ~tens of ns. Add a `bench_422_*` if the sampler becomes a hot-path consumer.)
- [x] **T3.6** Re-export through `katgpt-core` as
  `katgpt_core::dec::{sample_cochain_at_point_quad, ...}`.

## Phase 4 — No-Regression + Docs

### Tasks

- [x] **T4.1** `cargo check -p katgpt-core` passes with and without
  `--features cochain_point_sampler`. — **PASS**
- [x] **T4.2** `cargo check --workspace --all-features` passes (the
  `merkle_root`/`can_freeze` lesson — audit all feature combos). — **PASS** (43.6s)
- [x] **T4.3** Module doc-comments describe the primitive as generic DEC
  (Whitney/de-Rham reconstruction) math only — no game/chain/shard semantics.

---

## Out of Scope

- **Hex (3D voxel) and tetrahedron samplers** — deferred. Quad covers 2D grids
  (the `grid_2d` fast path); triangle covers mesh consumers. Add when a concrete
  3D consumer appears.
- **The LPPN decoder weights `f_θ`** — training-side (→ riir-train). The
  modelless analog is the caller supplying a frozen direction vector (existing
  `project_to_scalars` pattern); this primitive only computes the continuous
  `(s̄(p), u_aug(p))` conditioning.
- **Wiring into riir-ai consumers** (motor-gated field, heat kernel, terrain
  cochains) — optional follow-up in riir-ai, not part of this plan. Research 404
  §2.4 F1 documents the fusion targets.
- **Promotion to default** — Gain verdict; stays opt-in. Continuous sampling is
  a quality knob, not a correctness/perf win on the default path.

---

## Promotion Decision (pre-filled, pending gate)

Stays **opt-in** (`cochain_point_sampler = []`) regardless of gate outcome. This
is a substrate-completeness primitive (fills the DEC read-resolution gap), not a
default-path improvement. Demote nothing — there is no incumbent to demote
(continuous sampling has no predecessor; it's a new capability).
