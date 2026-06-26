# Plan 332: Structured Basis Selection for FUNCATTN — Open Primitive

**Date:** 2026-06-26
**Origin:** [Issue 001](../.issues/001_apollonian_sphere_manifold_exploration.md) use case #4 — VALIDATED by probe (`tests/apollonian_basis_probe.rs`, 2026-06-26)
**Related Research:** [257 (FUNCATTN)](../.research/257_Functional_Attention_Spectral_Transport_Operator.md), [291 (cross-resolution)](../.research/291_cross_resolution_spectral_transport_open_primitive.md), [100 (EGA — fixed<learned)](../.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md)
**Target:** `katgpt-rs/crates/katgpt-core/src/funcattn.rs` (extend `FuncAttnBasis` enum + add basis constructors) + Cargo feature `funcattn_structured_basis`
**Status:** Phase 0 (probe) COMPLETE — T5.1 "invariance" premise falsified, structured basis beats random by +0.11 cos. Phase 1+ pending.

---

## Goal

Ship a principled multi-scale basis-selection mechanism for FUNCATTN. The probe (Issue 001, 2026-06-26) proved that a HAND-CRAFTED signal-aligned basis beats random-orthogonal by +0.11 cos on transport quality. The open question: can a PRINCIPLED fixed basis (no a-priori signal knowledge) capture most of that gain?

**Proves the idea:** G1 principled-basis cos ≥ random-orthogonal cos + 0.05 on multi-scale transport (the probe's +0.11 is the upper bound; +0.05 is the "worth shipping" threshold) · G2 principled-basis cos ≥ 0.85 × hand-crafted-basis cos (captures most of the achievable gain) · G3 zero-alloc steady state (G5 from Plan 286 preserved) · G4 no regression on existing FUNCATTN tests.

**Kills the idea:** G1 principled-basis cos < random-orthogonal cos (fixed basis loses to no-information baseline) **OR** G2 principled-basis cos < 0.5 × hand-crafted (captures less than half the achievable gain — not worth the complexity).

---

## Background — the T5.1 correction (load-bearing)

Plan 286 T5.1 documented a null result: PCA eigenbasis pre-rotation was 17-25% worse than vanilla, with the explanation *"the adaptive basis's row-normalization is invariant to basis direction"*. **The probe falsified this explanation empirically:**

```
cos(Φ_rand1, Φ_rand2)  = 0.8613  ← noise floor (two random orthogonal bases)
cos(Φ_rand1, Φ_struct) = 0.7779  ← structured basis
Δ = 0.0834 > 0.05  → H_structure HOLDS, H_invariance REJECTED
```

The real explanation for T5.1's null result: PCA pre-rotation of a random-orthogonal `w_basis` by an orthogonal eigenvector matrix `V` produces `W·V^T`, which is **also random-orthogonal** (product of two orthogonal matrices). T5.1 was comparing random-vs-random, not random-vs-structured. The "invariance" was an artifact of the PCA-rotation experimental design, not a property of the basis normalization.

This means: **structured bases CAN help FUNCATTN.** The door is open. This plan walks through it.

---

## Phase 0 — Probe (COMPLETE)

- [x] T0.1 Probe tests written: `crates/katgpt-core/tests/apollonian_basis_probe.rs` (3 tests)
- [x] T0.2 Φ sensitivity test: H_invariance rejected (Δ=0.0834 > 0.05)
- [x] T0.3 Temperature probe: basis choice matters more at τ=0.1 (gap 0.1407 vs 0.0834)
- [x] T0.4 Transport quality test: structured basis beats random by +0.11 cos (τ=0.5)
- [x] T0.5 Issue 001 updated with corrected verdict

---

## Phase 1 — Principled Basis Constructors (CORE)

**Target:** add basis-family constructors to `funcattn.rs` that produce structured `w_basis` WITHOUT a-priori signal knowledge. Three candidates, each addressing a different "principled" prior:

### T1.1 Multi-scale cosine/DCT basis (`funcattn_structured_basis` feature)

The simplest principled multi-scale basis: DCT-II basis vectors at logarithmically-spaced frequencies. This is the "poor man's wavelet packet" — it captures multi-scale structure without requiring signal knowledge, and it's O(d·k) to construct.

- [ ] Add `pub fn make_dct_log_basis(k: usize, d: usize) -> Vec<f32>` to `funcattn.rs`:
  - Frequencies `f_i = round(2^((i / (k-1)) * log2(d/2)))` for i in 0..k (log-spaced from 1 to d/2)
  - Basis row `i`: `w[i, j] = cos(π · f_i · (j + 0.5) / d)`, then L2-normalize + Gram-Schmidt
  - Returns (k, d) row-orthonormal matrix
- [ ] Unit test: `dct_log_basis_is_row_orthonormal` (verify `W·W^T ≈ I_k` to 1e-5)
- [ ] Unit test: `dct_log_basis_covers_log_spaced_frequencies` (verify frequency distribution)

### T1.2 Haar wavelet packet basis (`funcattn_structured_basis` feature)

The next step up: a genuine multi-resolution basis (Haar wavelet packet at log-spaced scales). This shares the multi-scale hierarchical property that Apollonian packings offer, but with a well-understood construction. **This is the Apollonian surrogate** — if Haar wavelet packets (which share Apollonian's multi-scale property) fail, Apollonian would also fail. If they succeed, we justify the harder Apollonian implementation.

- [ ] Add `pub fn make_haar_packet_basis(k: usize, d: usize) -> Vec<f32>` to `funcattn.rs`:
  - Build Haar wavelet packet tree to depth `log2(d)`
  - Select k nodes spanning log-spaced scales (1 coarse + finer scales)
  - Returns (k, d) row-orthonormal matrix
- [ ] Unit test: `haar_packet_basis_is_row_orthonormal`
- [ ] Unit test: `haar_packet_basis_spans_multiple_scales`

### T1.3 Feature gate wiring

- [ ] Add `funcattn_structured_basis = ["funcattn"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml`
- [ ] Add `funcattn_structured_basis = ["katgpt-core/funcattn_structured_basis"]` to `katgpt-rs/Cargo.toml`
- [ ] Gate the two constructors behind `#[cfg(feature = "funcattn_structured_basis")]`

---

## Phase 2 — GOAT Gate (the real test)

**Target:** run the GOAT gate from the probe's multi-scale transport task, but with the PRINCIPLED bases (DCT-log and Haar-packet) instead of the hand-crafted signal-aligned basis.

### T2.1 G1 — principled basis vs random on transport quality

- [ ] Extend `apollonian_basis_probe.rs` (or new `funcattn_structured_basis_g1.rs`) with:
  - Same multi-scale input (d=64, n=20, k=8, 4 sinusoidal scales)
  - Same linear smoothing target
  - Four `w_basis` variants: random-orthogonal, hand-crafted signal-aligned (upper bound), DCT-log, Haar-packet
  - Sweep τ ∈ {0.5, 0.1}
  - Metric: cos(out, target) for each variant
- [ ] **G1 PASS:** DCT-log cos ≥ random cos + 0.05 **AND** Haar-packet cos ≥ random cos + 0.05
- [ ] **G1 KILL:** either principled basis cos < random cos (loses to no-information baseline)

### T2.2 G2 — principled basis captures most of the achievable gain

- [ ] Compute "achievable gain" = hand-crafted cos − random cos (the probe showed +0.11)
- [ ] Compute "principled gain" = principled cos − random cos
- [ ] **G2 PASS:** principled gain ≥ 0.5 × achievable gain (captures at least half)
- [ ] **G2 KILL:** principled gain < 0.5 × achievable gain (not worth the complexity)

### T2.3 G3 — no regression on existing FUNCATTN tests

- [ ] Run existing 17 funcattn tests with each principled basis swapped in as default `w_basis`
- [ ] **G3 PASS:** all 17 tests still pass (the basis is a drop-in replacement)
- [ ] Document any test that needs a basis-specific expected value

### T2.4 G4 — zero-alloc steady state preserved

- [ ] The constructors run ONCE at init (not in the hot path). The hot path (`funcattn_forward`) is unchanged — it consumes `w_basis: &[f32]` regardless of how it was constructed.
- [ ] **G4 PASS:** `funcattn_g5_zero_alloc` test still passes (no change to hot path)
- [ ] Document: basis construction is O(d·k) once at init, amortized over all forward passes

---

## Phase 3 — k-Sweep (fills Research 257 §5 item 5 gap)

The probe used k=8. Research 257 §5 item 5 explicitly flags k=4..16 for the NPC regime as an open sweep that has never been run. Now that we have principled bases, we can fill this gap.

- [ ] T3.1 Sweep k ∈ {4, 8, 16, 32} for each basis variant (random, DCT-log, Haar-packet, hand-crafted)
- [ ] T3.2 Plot cos(out, target) vs k for each basis family
- [ ] T3.3 Identify the elbow: at what k does random-orthogonal catch up to principled? (Hypothesis: principled bases help most at small k where random is rank-starved)
- [ ] T3.4 Document the finding in `.benchmarks/332_structured_basis_k_sweep.md`

---

## Phase 4 — Decision & Promotion

- [ ] T4.1 If G1+G2 PASS: promote `funcattn_structured_basis` to include the winning constructor in the FUNCATTN documentation as the "recommended default basis for multi-scale transport tasks"
- [ ] T4.2 If G1+G2 PASS AND the k-sweep shows principled bases help at k ≤ 16 (the NPC regime): consider promoting the winning constructor to be the default `w_basis` when `k ≤ 16` (auto-select structured basis for small k, random for large k)
- [ ] T4.3 If G1+G2 KILL: document the negative result, close the plan, note that the hand-crafted basis's advantage came from a-priori signal knowledge that fixed bases can't replicate (confirming the EGA fixed<learned concern)
- [ ] T4.4 Update Issue 001 with the final verdict

---

## Apollonian specifically — deferred to Phase 5 (only if Phase 2 passes)

If DCT-log or Haar-packet passes G1+G2, THEN we justify the harder work of implementing true Apollonian harmonics. The logic:

1. **Haar-packet is the Apollonian surrogate.** Both are fixed, multi-scale, hierarchical bases. If Haar-packet wins, Apollonian might win too (and might win more, since its geometry is richer). If Haar-packet loses, Apollonian would also lose (same fixed-basis failure mode).
2. **True d-dim Apollonian harmonics are a research project.** No off-the-shelf implementation exists for d=64. Implementing them requires either (a) 2D/3D projection (lossy), (b) construction from scratch via the Apollonian group, or (c) a different non-Euclidean embedding.
3. **Phase 5 is gated on Phase 2.** Only pursue Apollonian harmonics if a simpler principled multi-scale basis (Haar-packet) already proves the concept.

- [ ] T5.1 (GATED on Phase 2 PASS) Literature search: Apollonian harmonic decompositions in d > 3 dimensions (arxiv search via jina)
- [ ] T5.2 (GATED) Prototype Apollonian harmonic basis constructor (likely via 2D projection or surrogate)
- [ ] T5.3 (GATED) Benchmark Apollonian vs Haar-packet vs DCT-log — does Apollonian's richer geometry beat the simpler multi-scale bases?

---

## Anti-goals (explicitly out of scope)

- **NOT changing `funcattn_forward` hot path.** The basis is constructed once at init; the forward pass consumes `&[f32]` regardless. Zero hot-path changes.
- **NOT adding learned basis selection at runtime.** That's a freeze/thaw concern (Plan 286 T5.3), not this plan. This plan tests FIXED principled bases only.
- **NOT rescuing FUNCATTN's G6 LLM failure.** The probe showed basis choice matters for transport-quality tasks, not LLM token prediction. This plan targets the transport niche, not the LLM niche.
- **NOT implementing block-structured transport operator C.** That's a different primitive (hierarchical transport). This plan is basis selection only.

---

## Cross-Refs

- `katgpt-rs/.issues/001_apollonian_sphere_manifold_exploration.md` — origin, probe results
- `katgpt-rs/crates/katgpt-core/tests/apollonian_basis_probe.rs` — Phase 0 probe (COMPLETE)
- `katgpt-rs/.plans/286_functional_attention_spectral_transport.md` — FUNCATTN parent plan, T5.1 null result (now explained)
- `katgpt-rs/.plans/310_cross_resolution_spectral_transport_primitive.md` — cross-resolution (DEFAULT-ON, handles multi-scale via learned bases)
- `katgpt-rs/.research/257_Functional_Attention_Spectral_Transport_Operator.md` — §5 item 5 (k-sweep gap this plan fills)
- `katgpt-rs/.research/100_EGA_Energy_Gated_Attention_Spectral_Salience.md` — fixed<learned precedent (softer concern now)

## TL;DR

Probe (Issue 001, 2026-06-26) falsified the T5.1 "basis invariance" claim: structured bases DO change Φ materially (Δ=0.0834 > noise floor) and produce +0.11 cos better transport output than random. The T5.1 null result was a random-vs-random artifact (PCA rotation of random-orthogonal = another random-orthogonal). This plan ships principled multi-scale basis constructors (DCT-log, Haar-packet as Apollonian surrogate), runs the GOAT gate (can a fixed principled basis capture ≥50% of the hand-crafted gain without a-priori signal knowledge?), fills the Research 257 §5 k-sweep gap, and gates true Apollonian harmonics (Phase 5) on the simpler bases passing first.
