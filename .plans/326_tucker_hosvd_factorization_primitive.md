# Plan 326 — Tucker / HOSVD Tensor Factorization Primitive

## Context

Research 307 §3 (`.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md`)
identified three narrow Gain-tier gaps from the FNO practical-perspective paper.
Plans 323 (Fourier Continuation) and 325 (Spectral Differentiation) closed gaps
#1 and #2. This plan closes **gap #3: Tucker / HOSVD tensor factorization for
`NeuronShard` compaction**.

The FNO paper's TFNO variant (§6.1) applies Tucker compression to weight tensors.
Modellessly, Tucker decomposition (Higher-Order SVD) is a **deterministic**
factorization — no training, no gradient descent — so it satisfies the
modelless-first mandate.

### Dimensional correction vs Research 307

Research 307 §2.3/§3 says reshape `style_weights[64]` as `(K=8, I=8, O=8)`. That
is **incorrect** — `(8,8,8) = 512 ≠ 64`. The research note conflated the TFNO
paper's generic `(K,I,O)` weight-tensor shape with our 64-element shard vector.

The actual layout options for a 64-element flat buffer as a true 3-tensor:
- `(4, 4, 4)` — cube, 4³ = 64. **Natural Tucker reshape.**
- `(8, 8, 1)` — degenerate (mode-3 is rank-1); reduces to the 2D SVD already
  shipped via `subspace_phase_gate::thin_svd_into` with `STYLE_WEIGHTS_RESHAPE_N=8`.

This plan ships the **generic N-mode HOSVD primitive** (works for any
`(I₁, I₂, I₃)` with `I₁·I₂·I₃ = len`), then uses `(4,4,4)` as the shard
integration shape. The primitive is not shard-specific — it is the 3-mode
generalization of `thin_svd_into`.

### Where it lives

`katgpt-rs/crates/katgpt-core/src/linalg/tucker.rs` (new), feature
`tucker_factorization`. Reuses `thin_svd_into` from `subspace_phase_gate` for
the per-mode SVDs. Re-exported from `linalg/mod.rs`.

(NOT under `spectral/` despite the `spectral/mod.rs` TODO comment — Tucker/HOSVD
is an SVD generalization, not a Fourier operation. The TODO comment will be
updated to point at `linalg/tucker.rs`.)

## Phase 1 — Primitive (behind feature flag)

- [x] 1.1 Add feature `tucker_factorization = ["subspace_phase_gate"]` to
      `crates/katgpt-core/Cargo.toml`. Promoted to DEFAULT-ON after Phase 3.
- [x] 1.2 Create `crates/katgpt-core/src/linalg/tucker.rs`. **API evolved during
      implementation** to a more general N≤4-mode design (vs the planned fixed
      3-mode): `TuckerConfig` (validated shape+ranks), `TuckerScratch`,
      `TuckerResultScratch` (SOA, reusable), `TuckerResult` (owned convenience),
      `tucker_decompose_into` (zero-alloc hot path), `tucker_reconstruct_into`,
      `tucker_decompose` (allocating wrapper). Uses `thin_svd_into` for per-mode
      SVDs with a tall-skinny transpose trick.
- [x] 1.3 Result storage shipped as `TuckerResultScratch` (SOA: flat `core` + flat
      `factors` with per-mode offsets) + `TuckerResult` (owned). Mirrors the
      `SvdResultScratch` pattern.
- [x] 1.4 Mode-n unfolding (`unfold_into`) + inverse (`fold_into`) + row-major/
      column-stride helpers. Kolda & Bader convention.
- [x] 1.5 Wired into `linalg/mod.rs`: `pub mod tucker;` + re-exports.
- [x] 1.6 Wired into `lib.rs`: `pub mod linalg;` gated on
      `any(karc_forecaster, geometric_product, tucker_factorization)`.
- [x] 1.7 `SVD_MAX_RANK = 16` constraint documented + enforced in `TuckerConfig::new`
      (returns `ShapeExceedsSvdLimit` if any mode-n unfolding's min dim > 16).

## Phase 2 — GOAT Bench

- [x] 2.1 `benches/bench_326_tucker_hosvd_goat.rs` with `harness = false`,
      registered in `Cargo.toml`.
- [x] 2.2 **G1 — correctness:** rank-(2,2,2) recovery rel err **4.096e-8** < 1e-4.
- [x] 2.3 **G2 — perf:** (8,8,8) ranks (4,4,4) mean **71.38µs** < 500µs.
- [x] 2.4 **G3 — no-regression:** full-rank (4,4,4) max abs err **1.013e-6** < 1e-4.
- [x] 2.5 **G4 — alloc-free:** **0** allocations / 100 steady-state calls.

## Phase 3 — Promotion Decision

- [x] 3.1 GOAT bench run: all 4 gates PASS.
- [x] 3.2 Modelless gain confirmed (closed-form HOSVD, no training) →
      `tucker_factorization` **promoted to `default`**.
- [x] 3.3 Results documented in `.benchmarks/326_tucker_hosvd_goat.md`.

## Phase 4 — Validation & Commit

- [x] 4.1 `cargo test -p katgpt-core --features tucker_factorization --lib linalg::tucker`
      → 28 passed, 0 failed (debug + release).
- [x] 4.2 `cargo check --all-features` → clean.
- [x] 4.3 `cargo check` (default, post-promotion) → clean.
- [x] 4.4 Commit on `develop`.

## Non-goals

- **riir-neuron-db integration** (`shard_compactor.rs`) — out of scope for this
  primitive plan. The primitive ships standalone; wiring it into the cold-tier
  compaction is a separate plan in riir-neuron-db.
- **`semantic_axes` fusion** — the existing 2D SVD path in `subspace_phase_gate`
  stays as-is. Tucker is additive, not a replacement.
- **N > 4 modes** — `MAX_MODES = 4` covers the TFNO `(K,I,O)` weight-tensor
  shape and the shard `(4,4,4)` / batch `(N,8,8)` cases.
- **SVD_MAX_RANK > 16** — the underlying one-sided-Jacobi SVD has a stack-
  allocated `[f32; 16]` raw-sigma buffer. Larger tensors need chunking or a
  heap-backed SVD backend (future work).

## Validation Summary

| Check | Result |
|-------|--------|
| Unit tests (debug) | 28 passed, 0 failed |
| Unit tests (release) | 28 passed, 0 failed |
| G1 rank recovery | 4.096e-8 < 1e-4 ✅ |
| G2 perf (8,8,8) | 71.38µs < 500µs ✅ |
| G3 full-rank lossless | 1.013e-6 < 1e-4 ✅ |
| G4 alloc-free | 0 / 100 calls ✅ |
| cargo check --all-features | clean |
| cargo check (default, post-promotion) | clean |
| Promotion | DEFAULT-ON (pure modelless gain) |

## TL;DR

Shipped a generic N≤4-mode Tucker/HOSVD factorization primitive
(`linalg/tucker.rs`, feature `tucker_factorization`, **DEFAULT-ON**) that
generalizes the existing 2D `thin_svd_into` to N-tensors. Deterministic,
modelless, GOAT-gated (G1-G4 all PASS). Closes the third and final FNO gap
from Research 307 §3. The shard integration shape is `(4,4,4)` (correcting
the research note's dimensional error of `(8,8,8)=512`).
