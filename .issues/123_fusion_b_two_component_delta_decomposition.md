# Issue 123: Fusion B — Two-Component Weight Delta Decomposition (SAR × SOPTV)

> **Spawned from:** Research 406 (Spectral Rewiring — GOAT) × Research 231 / Plan 264 (SOPTV — GOAT)
> **Confidence:** MEDIUM — the decomposition is mathematically clean (orthogonal complement), but the Super-GOAT claim depends on both components being independently useful at NPC scale, which is unvalidated.
> **Date:** 2026-07-10
> **Status:** OPEN

---

## TL;DR

Research 406 (SAR) and Research 231 (SOPTV) study the **same geometric object**
(a weight delta ΔW projected onto a base matrix's SVD subspace) from **opposite
directions**, and reach **complementary** conclusions:

| Paper | Training regime | Finding | Component |
|---|---|---|---|
| **SOPTV** (R231) | Distillation (OPD) | Deltas are **off-principal** (p₁₀ ≤ 1% projection onto top-10% singular directions) | ΔW_off |
| **SAR** (R406) | RL (reward optimization) | Reasoning component is **on-principal** (projects onto base SVD subspace) | ΔW_on |

These are **not contradictory** — they study different training regimes. But
together they suggest a **two-component decomposition** that neither paper alone
provides:

```
ΔW = ΔW_on_principal (SAR)  +  ΔW_off_principal (SOPTV)
```

where `ΔW_on = U_r M V_rᵀ` (spectral rewire, Plan 423) and
`ΔW_off = ΔW − ΔW_on` (the residual, which SOPTV stores sparsely).

**Why this is a Super-GOAT candidate:** if both components are independently
useful (on-principal = capability rewiring, off-principal = task adaptation),
then ANY weight delta can be decomposed into two compact, semantically distinct
representations. This would unify the two existing spectral-projection primitives
(`spectral_rewire` + `off_principal`) into a single decomposition API.

**Why this is an issue, not a plan:** the decomposition is trivial to write
(SAR already returns the residual `ΔW⊥` — that IS the off-principal component).
The question is whether it's *useful*: whether the two components have distinct
semantic roles at our scale. That's a validation task, not an implementation task.

---

## The Decomposition (trivial to construct)

```rust
// Already produced by spectral_rewire_into (Plan 423):
let result = spectral_rewire_into(w0, delta, d_out, d_in, rank, &mut scratch);

// The two components:
let delta_on  = result.delta_star;   // SAR: U_r · M · V_rᵀ  (on-manifold)
let delta_off = result.residual;     // SOPTV: ΔW − ΔW*      (off-manifold)

// Verification (should hold to machine precision):
// delta_on + delta_off == delta
// ‖delta_on‖² + ‖delta_off‖² ≈ ‖delta‖²  (orthogonal decomposition)
```

The orthogonality holds by construction: U_r M V_rᵀ lives in the column space
of U_r and row space of V_rᵀ, while the residual lives in the orthogonal
complement. No new math needed.

---

## The Validation Gate (the real work)

Before promoting Fusion B from "trivial identity" to "shipped Super-GOAT
primitive", a PoC must show the two components are **semantically distinct and
independently useful**:

1. **Distinctness.** For real freeze/thaw deltas and LoRA overlays, the
   on-manifold and off-manifold components should have different statistical
   signatures (different spectral profiles, different sparsity, different
   effect on outputs). If they look identical, the decomposition adds no value.

2. **Independent utility.**
   - Applying only ΔW_on to W₀ should preserve "capability" (the base model's
     top-k singular directions are preserved — already tested in Plan 423 G2).
   - Applying only ΔW_off should capture "task-specific adaptation" (changes
     behavior without rewiring the base capability structure).

3. **Compactness.** Both components should be storable compactly:
   - ΔW_on → rewiring matrix M (r×r) — already compact.
   - ΔW_off → sparse storage via `SparseTaskVector::from_dense` (Plan 264
     Phase 1) — compact if the off-manifold component is sparse.

4. **Recomposition fidelity.** ΔW_on + ΔW_off reconstructs ΔW to machine
   precision (< 1e-6 relative). This holds by construction but must be verified
   empirically (catches SVD truncation bugs).

---

## Dependency Chain

| Dependency | Status | Blocking? |
|---|---|---|
| `spectral_rewire` primitive (Plan 423) | ✅ LANDED (opt-in; mechanism gates pass) | No — on-manifold projection exists |
| `off_principal` / `SparseTaskVector` (Plan 264) | ✅ COMPLETE, default-on | No — already shipped |
| Real weight deltas to decompose | ❓ No consumer produces freeze/thaw deltas yet | **YES for validation** — need a real delta source |
| **Spectral concentration on real deltas** | ❓ UNVALIDATED — G1b shows random deltas are NOT concentrated (0.12–0.18) | **YES — the make-or-break** |

The validation cannot proceed until Plan 423 lands AND a real delta source exists
(a freeze/thaw pipeline that produces ΔW, or a LoRA overlay that produces BA).

---

## Tasks (tracking only — no impl until dependencies land)

- [-] **T1** (deferred) ~~When Plan 423 (`spectral_rewire`) lands and passes its
  GOAT gate~~ — **Plan 423 LANDED (2026-07-10), mechanism gates pass.** The
  remaining blocker is concentration on REAL deltas (G1b shows random deltas
  are NOT concentrated at 0.12–0.18). PoC is blocked on a real delta source.
  When one exists, decompose it and measure distinctness (spectral profile
  divergence between ΔW_on and ΔW_off).
  
  **Modelless diagnostic landed (2026-07-10):** two tests validate the
  MEASUREMENT and SEPARATION mechanism without real deltas:
  - `concentration_measurement_calibrated_for_mixed_deltas` — constructs deltas
    with known on/off mixing ratios (α=0.25/0.5/0.75) and verifies the measured
    `on_manifold_fraction` matches theory within 3%. Self-calibrates against the
    incidental alignment floor (random delta at d=16, r=4 has fraction ≈ 0.28,
    NOT 0, due to r²/(d·d) alignment).
  - `two_component_separation_recovers_known_components` — constructs a delta
    with orthogonal on/off components, verifies delta_star matches the
    on-manifold component (rel err < 5%) and recomposition holds (< 1e-4).
  
  **Verdict:** the primitive correctly MEASURES concentration and SEPARATES
  components. The make-or-break is whether REAL TRAINED deltas are concentrated.
  This is genuinely blocked on riir-train — no trained weight files exist in
  any of the 5 repos (`*.lora` search: empty). All modelless delta sources
  (freeze/thaw, random LoRA, consolidation weight_delta) produce UNTRAINED
  deltas, which are NOT concentrated.
- [-] **T2** (deferred) When a freeze/thaw delta source exists in riir-neuron-db
  (producing real ΔW = W_frozen − W_base), run the decomposition on real
  personality deltas. Check whether the on-manifold component captures
  "personality rewiring" and the off-manifold captures "drift/noise".
- [-] **T3** (deferred) When a LoRA overlay path exists in riir-ai (producing
  real BA deltas), run the decomposition. Check whether on-manifold = capability
  and off-manifold = task-specific.
- [-] **T4** (deferred) If T1–T3 show distinctness + independent utility, open
  a plan for the unified `decompose_delta` API: takes ΔW, returns
  `(SpectralRewireResult, SparseTaskVector)` — the two-component decomposition
  as a single primitive. This is the Super-GOAT promotion candidate.

---

## Decision Matrix

| Scenario | Outcome | Action |
|---|---|---|
| Real deltas are NOT concentrated (on_manifold_fraction < 0.5) | Fusion B is moot — if on-manifold projection captures little energy, decomposition captures little | Close this issue; SAR stays a niche cold-tier tool |
| Real deltas ARE concentrated, but on/off components are statistically identical | Decomposition adds no value (no semantic distinction) | Close this issue, keep SAR as standalone |
| Real deltas concentrated, components distinct, but only one is useful | Partial win — use the useful component only | Downgrade to single-component usage, close issue |
| Real deltas concentrated, both components distinct + independently useful | **Super-GOAT** — unified decomposition API | Open implementation plan (T4) |

**G1b baseline (2026-07-10):** random deltas are NOT concentrated (0.12–0.18).
The decision hinges entirely on whether REAL deltas behave differently — which
is an empirical question that cannot be answered without a real delta source.

---

## Cross-references

- **Research 406** (`katgpt-rs/.research/406_Spectral_Rewiring_Weight_Delta_Purification.md`) — SAR, §2.3 Fusion B.
- **Research 231** (`katgpt-rs/.research/231_*`) — SOPTV, the off-principal finding.
- **Plan 423** (`katgpt-rs/.plans/423_spectral_rewire_primitive.md`) — the spectral_rewire primitive (dependency).
- **Plan 264** (`katgpt-rs/.plans/264_sparse_off_principal_task_vector_modelless.md`) — SOPTV, off_principal + SparseTaskVector (already shipped).
- **riir-train Issue 374** (`riir-train/.issues/374_spectral_concentration_trained_lora_delta.md`) — the training task to produce real LoRA deltas for the definitive concentration test.

---

## TL;DR

Fusion B decomposes a weight delta into on-principal (SAR) + off-principal
(SOPTV) components. The decomposition is trivial (SAR already returns the
residual). The question is whether the two components are semantically distinct
and independently useful at NPC scale. Blocked on Plan 423 landing + a real
delta source. If validation passes, this is a Super-GOAT (unified two-component
delta decomposition). If it fails, close the issue.
