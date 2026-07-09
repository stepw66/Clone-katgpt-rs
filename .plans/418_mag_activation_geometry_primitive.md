# Plan 418: MAG Activation Geometry Primitive — Unsupervised Direction Mining + Modelless Transfer Prediction

**Date:** 2026-07-09
**Research:** [katgpt-rs/.research/397_Mining_via_Activation_Geometry.md](../.research/397_Mining_via_Activation_Geometry.md)
**Source paper:** [arXiv:2607.04222](https://arxiv.org/abs/2607.04222) — LeVi, David, Fomin (ICML 2026 FAGEN)
**Target:** `katgpt-rs/crates/katgpt-core/src/mag/` (new module) + Cargo feature `mag_mining`
**Private guide:** [riir-ai/.research/316_mag_unsupervised_direction_mining_guide.md](../../riir-ai/.research/316_mag_unsupervised_direction_mining_guide.md)
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship a generic, modelless MAG (Mining via Activation Geometry) primitive: unsupervised direction mining from prefix-induced activation shifts, a linearity diagnostic (ϵ_Q), calibrated steering strength, and a modelless transfer-prediction scorer. This is the **missing acquisition step** for the direction-vector ecosystem (Latent Field Steering P309, EmotionDirections P162, PersonalityWeightedComposition P297, CommittedFieldBlend P321) — today all directions are designer-authored or supervised-extracted; MAG mines them unsupervised from the host's own verdicts.

The primitive is generic over the host's readout function, transform, and verdict source — no game/chain/shard semantics. The verdict `y_M` is a host-supplied `&[bool]` (the model/runtime's own binary observable), NOT a human label.

**GOAT gate:** G1 (mining correctness), G2 (contrast separability — the headline gate), G3 (reconstruction error sanity), G4 (transfer beats raw cosine), G5 (zero-alloc), G6 (latency).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create `crates/katgpt-core/src/mag/` module with `mod.rs` + `types.rs` + `mining.rs` + `transfer.rs`.
- [x] **T1.2** Add `mag_mining` feature to `crates/katgpt-core/Cargo.toml` (opt-in; `blake3` is already non-optional — no `dep:` prefix needed, same as `latent_field_steering`).
- [x] **T1.3** Define core types in `types.rs`:
  - `MagDirection` — `{ direction: Box<[f32]>, recon_error: f32, cosine: f32, blake3: [u8; 32] }`
  - `MagOperator` — `#[repr(u8)]` enum: `Direct`, `Prefixed`, `Answered`, `InputDelta`, `QuestionDelta`, `Interaction`, `Verdict`, `FewShot`
  - `TransferMetric` — `#[repr(u8)]` enum: `CentroidCosine`, `Euclidean`, `Correlation`, `RbfMmd`, `Wasserstein1d`, `CkaLinear`, `ClassConditionalCosineMalicious`, `ClassConditionalCosineBenign`
  - Plus: `MagError` enum (DimMismatch/Empty/ZeroNorm/EmptyClass), `DataSet<'a, S>` view, math helpers (norm/normalize/dot/cosine), BLAKE3 commitment, `check_dim` validator.
- [x] **T1.4** Implement `mine_direction` in `mining.rs`:
  - Input: `with_prefix: &[impl AsRef<[f32]>]`, `without_prefix: &[impl AsRef<[f32]>]`
  - Compute `v_Q = mean(with) − mean(without)`, unit-normalize (handles unequal sample counts via per-set denominators)
  - Compute BLAKE3 of the normalized direction bytes
  - Return `MagDirection` (recon_error + cosine set to NaN; populated by separate `reconstruction_error` call + `with_diagnostics` builder)
- [x] **T1.5** Implement `mine_contrast_direction` in `mining.rs`:
  - Input: `positive: &[impl AsRef<[f32]>]`, `negative: &[impl AsRef<[f32]>]` (partitioned by host-supplied `y_M`)
  - Compute `u_Q = mean(negative) − mean(positive)`, unit-normalize (delegates to `mine_direction(negative, positive)` — DRY)
  - Return `MagDirection`
- [x] **T1.6** Implement `reconstruction_error` in `mining.rs`:
  - Input: `with_prefix`, `without_prefix`, `direction: &[f32]`, `alpha: f32`
  - `m̂(p) = m(p) + α · direction`; `ϵ_Q = E[‖Δ(p) − α·direction‖²] / E[‖Δ(p)‖²]` where `Δ(p) = m(Q‖p) − m(p)`
  - Also compute mean cosine of `(α·direction)` vs `(Δ(p))` per-sample
  - Returns `(recon_error, cosine)` tuple
- [x] **T1.7** Implement `calibrate_alpha` in `mining.rs`:
  - `α(τ) = τ · ‖mean(with_prefix)‖ / ‖direction‖` — strength as fraction of prefix activation norm
- [x] **T1.8** Implement operator application helper `apply_operator` / `apply_operator_into`:
  - `apply_operator_into` (zero-alloc): given `MagOperator` + 7 readout slices (`A_p`, `A_Q`, `A_Qp`, `A_Qpy`, `A_y`, `A_empty`, `A_EQp`) → writes operator's vector summary into `&mut [f32]`
  - `apply_operator` (allocating convenience wrapper)
  - All 8 operators implemented; unused readouts accept `&[]`
- [x] **T1.9** Implement `transfer_score` in `transfer.rs`:
  - Input: `candidate: &DataSet`, `target: &DataSet`, `metric: TransferMetric` (DataSet wraps `&[impl AsRef<[f32]>]` + `&[bool]` labels)
  - All 8 metrics implemented: CentroidCosine, Euclidean, Correlation, RbfMmd, Wasserstein1d, CkaLinear, ClassConditionalCosineMalicious/Benign
  - For class-conditional: partition by labels, compute per-class centroids, cosine between corresponding classes
  - CKA uses feature-space (d×d Gram) formulation so candidate/target may have different sample counts
- [x] **T1.10** Implement `rank_candidates` in `transfer.rs`:
  - Input: `candidates: &[DataSet]`, `target: &DataSet`, `metrics: &[TransferMetric]`
  - Returns `Vec<RankEntry>` sorted by mean percentile rank (the paper's §4 protocol)
- [x] **T1.11** Wire `mag` module into `crates/katgpt-core/src/lib.rs` behind `mag_mining` feature. Re-export public API.
  - NOTE: `apply_operator_into` is NOT re-exported at crate root (collides with `analytic_lattice::apply_operator_into` when both features are on); accessible via `katgpt_core::mag::apply_operator_into`.
- [x] **T1.12** `cargo check -p katgpt-core --features mag_mining` passes. Also verified: `--no-default-features --features mag_mining` clean, `--all-features` clean, 35 unit tests + 3 doctests pass.

**Phase 1 Result:** 35 unit tests + 3 doctests pass. Build clean on `--features mag_mining`, `--no-default-features --features mag_mining`, and `--all-features`. No external deps (blake3 already non-optional). Ready for Phase 2 GOAT gate.

---

## Phase 2 — GOAT Gate (G1–G6)

### Tasks

- [ ] **T2.1 (G1 — mining correctness)** Unit tests in `tests/mag_g1.rs`:
  - Synthetic: N=100 pairs, known shift `v ∈ ℝ^64`. Assert `mine_direction` recovers `v` to cos ≥ 0.99.
  - Synthetic 2-cluster: 50 samples from N(μ₁, I), 50 from N(μ₂, I) in ℝ^64. Assert `mine_contrast_direction` recovers `(μ₁−μ₂)/‖·‖` to cos ≥ 0.95.
  - **PASS** required.
- [ ] **T2.2 (G2 — contrast separability, THE headline gate)** Tests in `tests/mag_g2.rs`:
  - Synthetic: 200 samples, 2 overlapping Gaussians (μ₁ = [2,0,...], μ₂ = [−2,0,...], σ=1.5) in ℝ^64. Random 50/50 partition as `y_M` (simulating model-self-labels with noise).
  - Mine `u_Q` via `mine_contrast_direction`. LOO logistic regression on `u_Q` projection (scalar dot-product + threshold).
  - **Gate:** LOO accuracy ≥ 0.75 (well above 0.5 chance). **If FAIL → abandon primitive, demote to research-only Gain.**
  - Also test on harder overlap (σ=3.0): gate ≥ 0.60 (still above chance, documents the separability ceiling).
- [ ] **T2.3 (G3 — reconstruction error sanity)** Tests in `tests/mag_g3.rs`:
  - Perfectly linear shift (`m(Q‖p) = m(p) + v`): assert `ϵ_Q = 0.0` (exact reconstruction).
  - Zero shift (`m(Q‖p) = m(p)`): assert `ϵ_Q = 1.0`.
  - Constructed overshoot (`m̂ = m(p) + 2·v` but true shift is `v`): assert `ϵ_Q > 1.0`.
  - **PASS** required.
- [ ] **T2.4 (G4 — transfer beats raw cosine)** Tests in `tests/mag_g4.rs`:
  - Synthetic transfer task: 6 candidate sets, 1 target. Construct candidates with KNOWN transfer gains (candidate 1 = high overlap with target, candidate 6 = orthogonal). Embed in ℝ^64.
  - Raw centroid cosine Top-1 over 50 random shuffles (random floor = 1/6 ≈ 16.7%).
  - MAG class-conditional triple (InputDelta + Answered + FewShot operators, cos_ben + cos_ben + cos_mal).
  - **Gate:** MAG Top-1 ≥ 0.50 (3× random). Raw cosine should be ≈ random (confirming the paper's ρ≈0 finding on synthetic data).
- [ ] **T2.5 (G5 — zero-alloc)** `TrackingAllocator` audit in `tests/mag_g5.rs`:
  - Warmup: mine direction once.
  - Measure: mine direction + compute transfer score 1000 times.
  - **Gate:** 0 allocations after warmup. Use `Vec::with_capacity` + `clear()` reuse; `Box<[f32]>` for direction storage.
- [ ] **T2.6 (G6 — latency)** Criterion bench in `benches/mag_g6.rs`:
  - `mine_direction` on 500×64 (500 prompts × 64-dim): target < 100µs.
  - `mine_contrast_direction` on 250+250×64: target < 100µs.
  - `transfer_score` (6 candidates × 64-dim): target < 10µs.
  - `reconstruction_error` on 100×64: target < 50µs.
- [ ] **T2.7** SIMD optimization: the inner mean/difference loops over `[f32]` slices should auto-vectorize. Add `#[inline(always)]` to hot helpers. Verify via `cargo asm` or godbolt spot-check that the accumulation loop vectorizes.
- [ ] **T2.8** Run full gate suite. Record results in `.benchmarks/418_mag_goat.md`. Decide: promote to default if G1–G6 all pass; demote to opt-in Gain if G2 fails.

---

## Phase 3 — Documentation + Integration Hooks

### Tasks

- [ ] **T3.1** Add `mag` section to `katgpt-rs/README.md` Feature Showcase with the GOAT gate summary.
- [ ] **T3.2** Add doc examples to `crates/katgpt-core/src/mag/mod.rs` showing: (a) mine a direction from paired activations, (b) compute ϵ_Q linearity diagnostic, (c) rank candidate datasets by transfer score.
- [ ] **T3.3** Cross-reference from Research 397 + riir-ai guide 316 once Phase 2 passes.
- [ ] **T3.4** Note the modelless-unblock relevance: MAG direction mining is a §3.5 path-3 (latent-space correction) tool — a systematically biased verdict can be corrected by mining the bias direction and projecting it out. Document this in the module doc.

---

## Open Questions (tracked, not blocking)

1. **Does G2 separability hold on low-dim HLA (d=8)?** The paper validates on d=4096. If HLA's 8 dims are too low-rank for separable contrast directions, the riir-ai integration (G7) may need a higher-dim readout (latent_functor state, NeuronShard style_weights[64]). The open primitive is dim-agnostic; this is a host-side concern.
2. **Transfer-prediction generalization.** The paper's 94.7% is on their 18-dataset PI corpus. G4 uses synthetic data with known transfer structure. Independent validation on a real game-experience corpus is a riir-ai follow-up (G8), not a katgpt-rs gate.
3. **Operator necessity.** The paper shows Interaction (Y6) and Verdict (Y7) are near-zero on average. The open primitive ships all 8 (completeness); the GOAT gate focuses on the load-bearing operators (Prefixed/InputDelta/Answered/FewShot).

---

## Cross-references

- **Research:** [397_Mining_via_Activation_Geometry.md](../.research/397_Mining_via_Activation_Geometry.md)
- **Private guide:** [riir-ai/.research/316_mag_unsupervised_direction_mining_guide.md](../../riir-ai/.research/316_mag_unsupervised_direction_mining_guide.md)
- **Closest cousins:** Plan 162 (EmotionDirections — supervised), Plan 309 (Latent Field Steering — injection), Plan 297 (PersonalityWeightedComposition — consumer), Plan 321 (CommittedFieldBlend — consumer), Plan 405 (Spherical Steering — consumer)
- **Source paper:** [arXiv:2607.04222](https://arxiv.org/abs/2607.04222)
