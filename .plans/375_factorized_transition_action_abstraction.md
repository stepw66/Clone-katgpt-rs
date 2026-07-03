# Plan 375: Factorized Transition Action Abstraction — Open Primitive

**Date:** 2026-07-03
**Research:** [katgpt-rs/.research/374_OTF_LAM_Factorized_Transition_Primitives.md](../.research/374_OTF_LAM_Factorized_Transition_Primitives.md)
**Source paper:** [arXiv:2606.30544](https://arxiv.org/abs/2606.30544) — Nam et al., *Latent Actions from Factorized Transition Effects under Agent Ambiguity*, Brown, 2026-06-30
**Target:** `katgpt-rs/crates/katgpt-core/src/factorized_action/` (new module) + Cargo feature `factorized_action`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship a modelless, inference-time **factorized action abstraction** primitive: given a frozen codebook of K effect primitives, decompose an observation transition into a sparse set of active primitives, score each via a state-aware sigmoid relevance gate, and aggregate via normalized weighted average into a compact action latent. This is the **factorized/compositional cousin** of the shipped monolithic `latent_functor` (`extract_functor`/`apply_functor`, Plan 273) — it enriches the action representation from "one displacement vector" to "a mixture of K reusable effect primitives gated by current state".

The codebook is constructed modellessly via **k-means clustering** on observed transition patches (deterministic Lloyd's algorithm, no gradient descent). The full inference path (patchify → assign → gate → aggregate) is zero-allocation, sigmoid-gated (never softmax), feature-flagged.

**GOAT gate:** the factorized primitive must provably outperform the monolithic baseline on (G1) in-distribution reconstruction, (G2) distractor-suppression reconstruction, (G3) cross-carrier transfer degradation. If G2 fails (no distractor suppression gain), keep opt-in and defer to riir-train for trained VQ-VAE.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create module `katgpt-rs/crates/katgpt-core/src/factorized_action/mod.rs` with:
  - `pub struct EffectCodebook<const K: usize, const D: usize>` — frozen codebook of K D-dim effect vectors, `#[repr(C)]` Pod-compatible. **Paper defaults: K=128, D=32** (verified from `configs/otf_vqvae/default_config.yaml`).
  - `pub struct TransitionFactors { assignments: [u16; MAX_PATCHES], weights: [f32; K], n_active: usize }` — per-transition factorization output (occupancy + activation strength).
  - `pub struct FactorizedActionLatent<const D: usize>([f32; D])` — the aggregated action latent.
  - `pub enum AggregatorType { Gate, Mean }` — **verified from `otf_lam/model.py`**: `"gate"` (default, sigmoid relevance gate) vs `"mean"` (uniform `α_k = 1` ablation). The `Mean` mode is the G2 ablation baseline.
  - Feature gate: `#[cfg(feature = "factorized_action")]`.

- [ ] **T1.2** Implement `EffectCodebook::assign_patch_into(&self, patch: &[f32], out: &mut TransitionFactors, patch_idx: usize)`:
  - Top-1 nearest-neighbor quantization: `k* = argmin_k ||patch - c(k)||²`.
  - Updates `assignments[patch_idx] = k*`, increments `weights[k*]`.
  - Zero-allocation (writes into pre-sized `TransitionFactors`).

- [ ] **T1.3** Implement `EffectCodebook::finalize_factors(&self, factors: &mut TransitionFactors, n_patches: usize)`:
  - Normalize `weights[k] /= n_patches` → activation strength `w(k)`.
  - Set `n_active` = count of nonzero weights.
  - Build occupancy mask `M(k)` from `assignments` (bit-set or sparse list).

- [ ] **T1.4** Implement `factor_token_into(codebook: &EffectCodebook<K,D>, k: usize, factors: &TransitionFactors, state: &[f32], out: &mut [f32])`:
  - `r_k = Γ(c(k), M(k), w(k), x_t)` — state-aware factor token.
  - **Modelless Γ via simplified FiLM** (verified from `otf_lam/model.py::FactorEmbedding` which uses FiLM `(1+γ)*x+β` pervasively):
    - `γ_k = dot(state, g_proj_k)` — state-conditioned scale.
    - `β_k = dot(state, b_proj_k)` — state-conditioned shift.
    - `r_k = (1 + γ_k) * c(k) + β_k` — FiLM-modulated codebook vector.
  - The projection vectors `g_proj_k`, `b_proj_k` are fixed (random orthonormal init, frozen — not learned).
  - No learned MLP.

- [ ] **T1.5** Implement `aggregate_action_latent_into<K,D>(codebook: &EffectCodebook<K,D>, factors: &TransitionFactors, state: &[f32], gate_beta: f32, gate_tau: f32, aggregator: AggregatorType, out: &mut FactorizedActionLatent<D>)`:
  - **Verified aggregation** from `otf_lam/model.py::OTFLAM.forward()` step 6:
    ```python
    alpha_sum = alpha.sum(dim=1).clamp_min(self.eps)
    z_factor = (alpha * factor_embedding).sum(dim=1) / alpha_sum
    ```
  - For each active code k:
    - Compute factor token `r_k` (T1.4).
    - If `aggregator == Gate`: sigmoid relevance gate `α_k = sigmoid(gate_beta * (relevance_score(r_k) - gate_tau))`.
    - If `aggregator == Mean`: `α_k = 1.0` (uniform — the G2 ablation).
    - Accumulate: `numerator += α_k * r_k`, `denominator += α_k`.
  - Normalized gated average: `z = numerator / (denominator + ε)`.
  - Write into `out.0[..D]`.
  - Zero-allocation (pre-sized scratch buffer).

- [ ] **T1.6** Add `factorized_action` feature to `katgpt-rs/crates/katgpt-core/Cargo.toml`:
  ```toml
  [features]
  factorized_action = []
  ```

- [ ] **T1.7** Wire module into `katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "factorized_action")]
  pub mod factorized_action;
  ```

- [ ] **T1.8** Smoke test: `assign_patch_into` + `aggregate_action_latent_into` on a hand-crafted K=4, D=8 codebook + 16-patch transition. Verify output is finite, in reasonable range, deterministic.

**Exit criteria for Phase 1:** module compiles under `cargo check -p katgpt-core --features factorized_action`, smoke test passes, no allocation in the hot path.

---

## Phase 2 — Modelless Codebook Construction (k-means)

### Tasks

- [ ] **T2.1** Implement `fit_codebook_kmeans_into<K,D>(patches: &[&[f32]], k: usize, seed: u64, max_iters: usize, out: &mut EffectCodebook<K,D>)`:
  - Lloyd's algorithm: init via k-means++ (deterministic from `seed`), iterate assign + update until convergence or `max_iters`.
  - Writes centroids into `out.centroids`.
  - Deterministic (fixed seed → fixed codebook). No gradient descent.
  - Isolated target dir: `CARGO_TARGET_DIR=/tmp/katgpt-plan-375` per AGENTS.md.

- [ ] **T2.2** Implement `EffectCodebook::from_observed_transitions<K,D>(transitions: &[(Vec<f32>, Vec<f32>)], patch_size: usize, k: usize, seed: u64) -> Self`:
  - For each `(x_t, x_{t+1})`: compute motion input `o_t = x_{t+1} - x_t` (or Sobel-transformed diff).
  - Patchify `o_t` into `patch_size` blocks.
  - Collect all patches → k-means fit (T2.1).
  - Returns frozen codebook.

- [ ] **T2.3** Test: k-means on synthetic 2D transitions (100 transitions, K=8, D=4). Verify:
  - All K centroids are distinct (no collapse).
  - Codebook is deterministic (same seed → same centroids).
  - Reconstruction MSE < identity baseline (predict `o_t = 0`).

- [ ] **T2.4** Test: cross-carrier transfer. Fit codebook on "digit-A" transitions, evaluate reconstruction MSE on "digit-B" transitions. Verify transfer degradation < 60% (the paper's monolithic baseline is 58–72%; our modelless k-means should be at least competitive).

---

## Phase 3 — GOAT Gate (the promote/demote decision)

### Tasks

- [ ] **T3.1** Create benchmark `katgpt-rs/benches/bench_375_factorized_action_goat.rs` with the **four** competitors (per Research 374 §9 + §10 code verification):
  1. **Monolithic baseline** — `extract_functor` + `apply_functor` (single mean displacement).
  2. **Factorized OTF (modelless, Gate mode)** — k-means codebook (**K=128, D=32** — paper defaults) + sigmoid gate + normalized weighted average.
  3. **Factorized OTF (modelless, Mean mode)** — same codebook, `α_k = 1` uniform (the ablation from `aggregator_type: "mean"`).
  4. **Identity baseline** — predict `x_{t+1} = x_t`.

- [ ] **T3.2** **G1 — Correctness.** Reconstruction MSE on in-distribution transitions (Moving-MNIST-style: 2D digits moving at constant velocity, 1000 transitions). Gate: `factorized_mse ≤ monolithic_mse`.

- [ ] **T3.3** **G2 — Distractor suppression + gate ablation.** Reconstruction MSE on transitions WITH distractor motion (background dot moving independently). **Two sub-gates:**
  - G2a (factorization gain): `factorized_gate_mse < 0.7 × monolithic_mse` (≥30% relative improvement — the paper's key claim).
  - G2b (gate adds value): `factorized_gate_mse < factorized_mean_mse` (the sigmoid relevance gate beats uniform aggregation — verified ablation from `otf_lam/model.py::aggregator_type`).
  If G2a passes but G2b fails → the factorization helps but the modelless sigmoid gate adds no value over uniform mean → note that the trained gate (`GateNetwork` with 4 FiLM layers) is needed → riir-train.

- [ ] **T3.4** **G3 — Cross-carrier transfer.** Codebook fit on digit-{0–4} transitions, evaluated on digit-{5–9}. Transfer degradation `Drop = (E_target - E_source) / E_source`. Gate: `factorized_drop < monolithic_drop`.

- [ ] **T3.5** **G4 — Latency.** Factorized aggregation (**K=128, D=32** — paper defaults, 16 patches) < 1 µs per transition. Zero-allocation after warmup (TrackingAllocator audit). Bench with criterion.

- [ ] **T3.6** **G5 — Sigmoid never softmax.** Static check (grep: no `softmax` in `factorized_action/`) + canary test (sigmoid at logit=0 gives 0.5, softmax of single value gives 1.0 — assert the former).

- [ ] **T3.7** **G6 — Feature isolation.** `cargo check -p katgpt-core --features factorized_action` passes. `cargo check -p katgpt-core --no-default-features` passes. `cargo check --workspace --all-features` passes (no combo regression).

**Promote/demote decision:**
- If G1 + G2 + G3 all PASS → promote `factorized_action` to default-on. Demote nothing (it enriches, doesn't replace `latent_functor`).
- If G2 FAILS (no distractor suppression gain) → keep opt-in. Note in the benchmark that the modelless k-means codebook is insufficient for distractor suppression; trained VQ-VAE needed → riir-train follow-up.
- If G1 FAILS → the primitive is broken; debug before any promotion.

---

## Phase 4 — Cross-Ref Wiring (future, deferred)

### Tasks (deferred — not blocking Phase 3 promotion)

- [-] **T4.1** riir-ai runtime wiring: HLA state → gate conditioner for `aggregate_action_latent_into`. Each NPC gates the same codebook differently → per-NPC compositional action understanding. File as riir-ai plan when prioritized.
- [-] **T4.2** riir-neuron-db `EffectCodebookShard` Pod subtype: store K×D codebook as `#[repr(C)]` Pod, BLAKE3-committed, atomic hot-swap via `MerkleFrozenEnvelope`. File as riir-neuron-db plan when prioritized.
- [-] **T4.3** riir-train VQ-VAE codebook learning: trained codebook as alternative to runtime k-means. One-line redirect per Research 374 §8.

---

## Implementation Notes

### SOLID / DRY compliance
- `EffectCodebook` is a pure data structure (no behavior beyond lookup).
- `assign_patch_into`, `aggregate_action_latent_into`, `fit_codebook_kmeans_into` are free functions operating on references — composable, testable.
- No game semantics in katgpt-core (the "digit" / "transition" vocabulary is benchmark-only).

### Perf rules (per AGENTS.md optimization guidelines)
- Fixed-size arrays `[f32; K]`, `[f32; D]` where K, D are const generics.
- Pre-allocated scratch buffers passed as `&mut [T]`.
- No allocation in the hot path (assign + gate + aggregate).
- Chunked inner loops (4 or 8 elements) for SIMD auto-vectorization in the k-means distance computation.
- k-means is offline (run once per codebook fit), not in the inference hot path.

### Latent vs raw boundary
- The codebook centroids are latent (D-dim vectors).
- The action latent `z^act` is latent.
- Only the final scalar projections (if consumed by HLA emotion extraction or sync) cross the sync boundary — same discipline as existing `latent_functor`.

### Sigmoid mandate
- The relevance gate `α_k = sigmoid(...)` uses sigmoid, never softmax. Verified by G5.
- This is consistent with AGENTS.md constraint #2 and the paper's own design (sigmoid gating throughout).

---

## References

- **Paper:** Nam et al., *Latent Actions from Factorized Transition Effects under Agent Ambiguity*, arXiv:2606.30544, 2026-06-30.
- **Research note:** [katgpt-rs/.research/374_OTF_LAM_Factorized_Transition_Primitives.md](../.research/374_OTF_LAM_Factorized_Transition_Primitives.md)
- **Monolithic baseline:** Plan 273 (`latent_functor/arithmetic.rs`), Research 123 (Latent Functor Runtime — Super-GOAT).
- **Codebook mechanism cousin:** `katgpt-kv` Lloyd-Max VQ (KV compression, not transition factorization).
- **Aggregation pattern cousin:** Plan 297 (`PersonalityWeightedComposition` — weighted layer composition).
- **Motion input cousin:** Plan 277 (Temporal Deriv Kernel — DEFAULT-ON, the `o_t` analog).
