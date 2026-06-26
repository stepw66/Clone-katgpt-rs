# Plan 330: Analytic Lattice Encoder/Decoder + Chain Composer Primitive

**Date:** 2026-06-26
**Research:** [katgpt-rs/.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md](../.research/311_Analytic_Lattice_Encoder_Decoder_Primitive.md)
**Source paper:** Synthesis (R311 §2) — Functional Attention × PJ-RoPE × Gyrocalculus fusion
**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/` (new module) + Cargo feature `analytic_lattice_encoder`
**Status:** Active — Phase 0 <scaffold>

---

## Goal

Ship the three missing open primitives identified in R311:

1. **`AnalyticLatticeEncoder`** trait — closed-form `encode(&self, entity) -> [f32; N]` for arbitrary domain entities, typed per lattice slot.
2. **`direction_vector_decode`** — SIMD projection of a latent state onto a direction vector, producing an action-score scalar (generalization of riir-games `scalar_projection.rs`, but generic, not HLA-specific).
3. **`compose_chain`** — operator product `C[n-1] × ... × C[1] × C[0]` for an arbitrary-length chain of `f32` transport operators (k×k).

All three: zero-alloc, SIMD-first, ARM64/x86_64/wasm32-portable, behind ONE feature flag. GOAT gate G1–G6 (per R311 §5) must pass before promotion to `default`.

---

## Phase 0 — Module skeleton + types

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/mod.rs` (new)

### Tasks

- [ ] **T0.1** Add Cargo feature `analytic_lattice_encoder = []` to `katgpt-rs/crates/katgpt-core/Cargo.toml`. NOT default-on.
- [ ] **T0.2** Create `analytic_lattice/mod.rs` with module doc + 3 sub-module declarations (`encoder.rs`, `decoder.rs`, `chain.rs`).
- [ ] **T0.3** Define the typed-slot lattice vector:

```rust
/// Typed per-slot lattice vector — 8 lanes matching Plan 335 eggshell.
/// Slot semantics are CALLER-defined (game IP); this primitive is slot-agnostic.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct LatticeVector<const N: usize>(pub [f32; N]);

/// A k×k transport operator (output of FuncAttn or extract_functor_rank_k).
#[derive(Clone, Debug)]
pub struct TransportOperator {
    pub k: usize,
    pub data: Vec<f32>, // row-major k×k
}
```

- [ ] **T0.4** Wire into `katgpt-core/src/lib.rs` behind the feature flag.

---

## Phase 1 — `AnalyticLatticeEncoder` trait + reference impls

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/encoder.rs`

### Tasks

- [ ] **T1.1** Define the trait:

```rust
/// Closed-form encoder: domain entity → typed lattice vector.
///
/// Contract:
/// - Pure function of `entity` (no I/O, no allocation in `encode_into`).
/// - Bit-identical across ARM64 / x86_64 / wasm32 for logically equal input (G1).
/// - Output is bounded; caller decides normalization.
pub trait AnalyticLatticeEncoder<E, const N: usize> {
    fn encode_into(&self, entity: &E, out: &mut LatticeVector<N>);
    #[inline]
    fn encode(&self, entity: &E) -> LatticeVector<N> {
        let mut v = LatticeVector([0.0; N]);
        self.encode_into(entity, &mut v);
        v
    }
}
```

- [ ] **T1.2** Ship 3 reference implementations (proving the trait is generic):
  - `LinearWeightsEncoder` — `out[i] = sum_j(w[i][j] * feat[j])` (linear map; closes the loop with `compute_linear_basis_into` in `latent_functor/arithmetic.rs`).
  - `SigmoidGateEncoder` — `out[i] = sigmoid(w[i]·feat)` (matches AGENTS.md "never softmax" rule).
  - `TypedSlotEncoder` — per-slot closures `[Fn(&E)->f32; N]` (the game-side extension point — lets riir-ai define "x = player.level, y = boss.hp, ...").
- [ ] **T1.3** G1 test: encode the same entity 3 times across simulated targets, assert byte-identical output.

---

## Phase 2 — `direction_vector_decode` SIMD primitive

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/decoder.rs`

### Tasks

- [ ] **T2.1** Implement the decoder as a zero-alloc SIMD dot-product + sigmoid:

```rust
/// Project `state` onto `direction`, return scalar action score in (0,1).
///
/// This is the GENERALIZED version of riir-games `scalar_projection::project_to_scalars`,
/// lifted out of HLA-specific 5-scalar semantics into a generic single-direction primitive.
/// The 5-scalar HLA bridge in riir-games becomes a thin wrapper that calls this 5 times.
#[inline]
pub fn direction_vector_decode<const N: usize>(
    state: &LatticeVector<N>,
    direction: &LatticeVector<N>,
    temperature: f32,
) -> f32 {
    let z = dot(state.0.as_slice(), direction.0.as_slice()) / N as f32;
    sigmoid(z * temperature)
}
```

- [ ] **T2.2** Add `direction_vector_decode_into` variant for batch decode (multiple directions, single state).
- [ ] **T2.3** G2 test: 100 random states × fixed direction, verify ranking matches brute-force reference within cos ≥ 0.95.
- [ ] **T2.4** Audit: riir-games `scalar_projection.rs` SHOULD be refactored to call this (out of scope here — note as cleanup follow-up).

---

## Phase 3 — `compose_chain` operator product

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/chain.rs`

### Tasks

- [ ] **T3.1** Implement the chain composer:

```rust
/// Compose a chain of k×k transport operators: out = C[n-1] × ... × C[1] × C[0].
///
/// All operators MUST have the same k. Returns the composite operator.
/// This is the cross-entity analog of `funcattn_compose` (which is token-level).
pub fn compose_chain(ops: &[TransportOperator]) -> Result<TransportOperator, ChainError> {
    // Validate same-k, then row-major matmul reduction.
    // Reuse one scratch buffer; zero alloc after first call if caller reuses.
}

/// In-place variant for hot paths.
pub fn compose_chain_into(
    ops: &[TransportOperator],
    scratch: &mut Vec<f32>,
    out: &mut TransportOperator,
) -> Result<(), ChainError> { /* ... */ }
```

- [ ] **T3.2** G3 test: associativity `(A×B)×C ≈ A×(B×C)` within Frobenius ≤ 1e-5.
- [ ] **T3.3** G4 test: full pipeline (encode 3 entities → extract_functor → compose_chain → decode) < 1µs release.
- [ ] **T3.4** G5 test: `TrackingAllocator` audit shows 0 allocs after warmup.

---

## Phase 4 — Spectral audit verifier (G6)

**Target:** `katgpt-rs/crates/katgpt-core/src/analytic_lattice/audit.rs`

### Tasks

- [ ] **T4.1** Implement `spectral_audit(operator, fourier_modes) -> AuditReport` per arxiv 2606.02427:
  - Compute tangent operator (numerical Jacobian at identity).
  - Project onto Fourier modes (DCT-II for real symmetric, 8 modes default).
  - Return per-mode gain + spurious-coupling matrix.
- [ ] **T4.2** G6 test: known-good composite operator returns max spurious coupling ≤ 5%; known-bad (random operator) returns > 5%.
- [ ] **T4.3** Document: this is the GOAT-gate verifier for chain composition — fails loudly if the chain produces nonsense.

---

## Phase 5 — GOAT gate + promotion/demotion

### Tasks

- [ ] **T5.1** Run all 6 gates in `katgpt-rs/crates/katgpt-core/tests/analytic_lattice_goat.rs`.
- [ ] **T5.2** Write benchmark to `katgpt-rs/.benchmarks/330_analytic_lattice_goat.md`.
- [ ] **T5.3** If all 6 gates pass: promote `analytic_lattice_encoder` to `default` in katgpt-core Cargo.toml.
- [ ] **T5.4** If any gate fails: keep opt-in, document the failure in `.issues/`, decide modelless unblock path per workflow §3.5 (check freeze/thaw, raw/lora, latent correction before any riir-train deferral).

---

## Risks

| Risk | Mitigation |
|---|---|
| `compose_chain` numerically unstable for long chains | Normalize each operator before multiplication (operator norm ≤ 1); cap chain length at 16 in v1 |
| Spectral audit G6 too strict (false positives) | Calibrate threshold on known-good composites from Plan 335 eggshell lanes; document baseline |
| Encoder determinism across targets (G1) fails on wasm32 | Use `floor` / `round` consistently; avoid `libm` calls that differ across targets |
| Decoder G2 ranking fails on adversarial direction vectors | Use temperature annealing during validation; document the failure envelope |

## Out of scope

- Game-specific encoding schemas (quest/zone/boss/player) — those live in riir-ai (R162 guide, P339 demo).
- Bevy demo — lives in riir-ai/.plans/339.
- Chain length > 16 — defer until G3 holds at length 16.
- Cross-resolution transport (Plan 310) composition — separate primitive, may fuse later.

## TL;DR

3 open primitives (encoder trait, SIMD decoder, chain composer) + 1 verifier (spectral audit) behind `analytic_lattice_encoder` feature flag. 6-gate GOAT (determinism, ranking, associativity, latency, zero-alloc, spectral audit). Promotes to default if all pass. Game-side schemas live in riir-ai (Plan 339 demo, R162 guide). Math is generic — no game IP leaks to katgpt-rs.
