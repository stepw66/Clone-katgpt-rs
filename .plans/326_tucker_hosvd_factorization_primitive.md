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

- [ ] 1.1 Add feature `tucker_factorization = []` to `crates/katgpt-core/Cargo.toml`.
      Opt-in until GOAT gate passes.
- [ ] 1.2 Create `crates/katgpt-core/src/linalg/tucker.rs` with:
  - `TuckerError` (empty/mismatched dims, truncation rank > mode size)
  - `TuckerScratch` — pre-allocated work buffers (unfolded matrix, SVD scratch,
    core tensor, reconstruction buffer). Reusable across calls.
  - `tucker_hosvd_into(t_flat, dims, result, scratch)` — zero-alloc hot path.
    `dims = (I₁, I₂, I₃)`. Writes factor matrices `U₁, U₂, U₃` and core `S`
    into `result`.
  - `tucker_hosvd(t_flat, dims)` — convenience wrapper (allocates).
  - `tucker_hosvd_truncated_into(t_flat, dims, ranks, result, scratch)` —
    truncated variant: `R_k ≤ I_k` factor matrices. The TFNO compaction angle.
  - `reconstruct_into(core, factors, out, scratch)` — rebuild the tensor from
    `(S, U₁, U₂, U₃)`. Inverse of decompose; needed for G1 correctness gate.
- [ ] 1.3 `TuckerResult` — owns `core: Vec<f32>` (length `I₁·I₂·I₃`) and
      `factors: [Vec<f32>; 3]` (lengths `I₁·I₁, I₂·I₂, I₃·I₃`). Plus
      `TuckerResultScratch` (SOA, reusable) mirroring the `SvdResultScratch`
      pattern from `subspace_phase_gate`.
- [ ] 1.4 Mode-k unfolding helpers (matricization) and inverse (fold-back).
      Mode-k unfolding maps the 3-tensor to a matrix `T_(k) ∈ R^(I_k × (I·J/I_k))`.
      Standard lexicographic unfolding (Kolda & Bader 2009 convention).
- [ ] 1.5 Wire into `linalg/mod.rs`:
      `#[cfg(feature = "tucker_factorization")] pub mod tucker;` + re-exports.
- [ ] 1.6 Wire into `lib.rs` if needed (linalg is already pub; check).

## Phase 2 — GOAT Bench

- [ ] 2.1 Create `benches/bench_326_tucker_hosvd_goat.rs` with `harness = false`.
      Register in `Cargo.toml` with `required-features = ["tucker_factorization"]`.
- [ ] 2.2 **G1 — correctness (decompose → reconstruct round-trip).**
      Build a known low-rank 3-tensor (e.g. outer product of 3 known vectors →
      rank-1 core), HOSVD-decompose, reconstruct, assert max|err| < 1e-4.
      Also: reconstruct a random `(4,4,4)` tensor with full ranks → max|err| < 1e-3
      (one-sided Jacobi has ~1e-6 residuals on small matrices; accumulated over
      3 modes + reconstruction the bound loosens).
- [ ] 2.3 **G2 — perf.** `tucker_hosvd_into` on `(4,4,4)` and `(8,8,8)` tensors.
      Target: ≤ 50µs (mirrors Plans 323/325). One-sided Jacobi on a 4×16 and
      8×64 unfolded matrix should be sub-µs.
- [ ] 2.4 **G3 — no-regression.** Truncated Tucker with `ranks = dims` (full)
      must equal the untruncated result (bit-identical or within 1e-6).
- [ ] 2.5 **G4 — alloc-free hot path.** CountingAllocator: 0 allocations across
      100 steady-state `tucker_hosvd_into` calls (after warmup).

## Phase 3 — Promotion Decision

- [ ] 3.1 Run the GOAT bench. All four gates must PASS.
- [ ] 3.2 If all PASS AND the gain is modelless (it is — HOSVD is a closed-form
      deterministic decomposition, no training) → promote `tucker_factorization`
      to `default` in `Cargo.toml`.
- [ ] 3.3 Document results in `.benchmarks/326_tucker_hosvd_goat.md`.

## Phase 4 — Validation & Commit

- [ ] 4.1 `cargo test -p katgpt-core --features tucker_factorization --lib linalg::`
      (unit tests in tucker.rs).
- [ ] 4.2 `cargo check --all-features` (catches combo regressions per the
      `merkle_root` lesson).
- [ ] 4.3 `cargo check` (default features, post-promotion).
- [ ] 4.4 Commit on `develop` with `feat(core):` prefix.

## Non-goals

- **riir-neuron-db integration** (`shard_compactor.rs`) — out of scope for this
  primitive plan. The primitive ships standalone; wiring it into the cold-tier
  compaction is a separate plan in riir-neuron-db.
- **`semantic_axes` fusion** — the existing 2D SVD path in `subspace_phase_gate`
  stays as-is. Tucker is additive, not a replacement.
- **N-mode generalization for N > 3** — 3-mode covers the shard `(4,4,4)` case
  and the general TFNO `(K,I,O)` case. Higher modes add complexity with no
  current consumer.

## TL;DR

Ship a generic 3-mode Tucker/HOSVD factorization primitive
(`linalg/tucker.rs`, feature `tucker_factorization`) that generalizes the
existing 2D `thin_svd_into` to 3-tensors. Deterministic, modelless, GOAT-gated.
Closes the third and final FNO gap from Research 307 §3. The shard integration
shape is `(4,4,4)` (correcting the research note's dimensional error of
`(8,8,8)=512`).
