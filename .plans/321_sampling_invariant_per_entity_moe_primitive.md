# Plan 321: Sampling-Invariant Per-Entity MoE Composition — Open Primitive

**Date:** 2026-06-25
**Research:** [302_FAME_Sampling_Invariant_Per_Entity_MoE](../.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md)
**Source paper:** [arxiv 2510.00621](https://arxiv.org/abs/2510.00621) — FAME: Adaptive Functional Attention with Expert Routing for Function-on-Function Regression (Gao/Chen/Zhang, NeurIPS 2025)
**Target:** `crates/katgpt-core/src/committed_field_blend.rs` (new module) + Cargo feature `committed_field_blend`
**Status:** Active — Phase 1 <not started>
**Tier:** Super-GOAT (open primitive half; private guide at `riir-ai/.research/158_*.md`)

---

## Goal

Ship `CommittedFieldBlend<N, D>` — a per-entity **frozen** convex blend of N operator fields over D-dim state, with sigmoid-computed weights derived once from a trajectory summary and committed via BLAKE3. The blend governs the entity's dynamics for its entire lifetime (until a major personality event triggers re-commitment). Because both the weights and the fields are frozen, the entity's trajectory is **sampling-invariant** in the FAME/Young-integral sense: observation gaps, network desync, and snapshot thaw all preserve the committed personality.

**GOAT gate:** G1 (mechanics + sigmoid correctness), G2 (sampling invariance under observation gaps — the defining property), G3 (no regression on `PersonalityWeightedComposition` hot path), G4 (zero-alloc commit + apply), G5 (BLAKE3 commitment bit-reproducibility).

---

## §3.5 Modelless Unblock — PASSED

All three paths pass (see Research 302 §3.5):
- **Path 1 (freeze/thaw):** archetype fields are frozen snapshot shards; blend weights frozen via `MerkleFrozenEnvelope`.
- **Path 2 (raw/lora hot-swap):** blended LoRA `L_π = Σ_k π_k · L_k` is a deterministic linear combination — modelless.
- **Path 3 (latent correction):** blend weights via sigmoid projection onto K direction vectors — modelless.

**No riir-train dependency for the runtime primitive.** The K archetype fields themselves are pre-trained offline once (upstream, library artifact) — that is the freeze/thaw substrate, not a per-entity training dependency.

---

## Architecture

### The primitive (generic math, no game semantics)

```rust
/// A host-supplied source of one operator field f_k(z) -> dz.
///
/// The host (game, robot, recommender) implements this per archetype.
/// The field is FROZEN — `evolve` must be a pure function of (z, dz_scratch).
pub trait ArchetypeFieldSource<const D: usize> {
    /// Apply the field at state `z`, writing the dynamics update into `dz`.
    /// Zero-allocation: caller provides scratch + dz buffers.
    fn evolve(&self, z: &[f32], dz_scratch: &mut [f32]) -> &mut [f32];

    /// BLAKE3 hash of the frozen field definition (for commitment).
    fn commitment(&self) -> [u8; 32];

    /// Optional Lipschitz bound L_k (for the safety-bound composition).
    /// Default: f32::IN (unknown — primitive does not assume boundedness).
    fn lipschitz_bound(&self) -> f32 { f32::INFINITY }
}

/// A per-entity committed archetype blend.
///
/// Computes blend weights π ONCE from a trajectory summary via sigmoid projection,
/// then FREEZES them for the entity's lifetime. The blended field
/// `f_π(z) = Σ_k sigmoid(π_k / τ) · f_k(z)` governs the entity's dynamics.
/// Because π and the fields are both frozen, the entity's trajectory is
/// sampling-invariant (FAME/Young-integral property).
pub struct CommittedFieldBlend<const N: usize, const D: usize> {
    /// Committed blend weights (pre-sigmoid logits). Signed; clamped to [−π_max, +π_max].
    /// Computed ONCE from trajectory summary; never mutated after commit.
    pub pi: [f32; N],
    /// Personality-sharpness temperature (sigmoid denominator).
    pub tau: f32,
    /// Clamp bound on pi (prevents extreme saturation).
    pub pi_max: f32,
    /// BLAKE3 of (pi, archetype_commitments, version). Set at commit time.
    pub blake3: [u8; 32],
    /// Version counter (incremented on re-commit).
    pub version: u64,
}

impl<const N: usize, const D: usize> CommittedFieldBlend<N, D> {
    /// Compute blend weights ONCE from a trajectory summary, then commit.
    ///
    /// `summary` is the host-supplied trajectory summary (e.g. KARC delay-embedding
    /// of the entity's HLA history, or a simpler ConvPool-style summary).
    /// `direction_vectors` are K pre-computed direction vectors (one per archetype),
    /// used for the sigmoid projection: `pi_k = dot(summary, dir_k)`.
    ///
    /// After this call, `pi` is frozen — call `recommit` only on major events.
    pub fn commit(
        &mut self,
        summary: &[f32],
        direction_vectors: &[[f32; D]; N],
        version: u64,
    ) -> [u8; 32];

    /// Apply the blended field at state `z`, writing the dynamics update into `dz`.
    ///
    /// `f_π(z) = Σ_k sigmoid(pi_k / tau) · f_k(z)`
    /// Zero-allocation: caller provides scratch + dz buffers.
    pub fn apply_blended(
        &self,
        fields: &[&dyn ArchetypeFieldSource<D>; N],
        z: &[f32],
        dz_scratch: &mut [f32],
        dz_out: &mut [f32],
    ) -> &mut [f32];

    /// Verify the commitment hash matches (anti-tamper check at thaw).
    pub fn verify_commitment(
        &self,
        fields: &[&dyn ArchetypeFieldSource<D>; N],
    ) -> bool;

    /// Recompute the BLAKE3 from current state (for atomic swap verification).
    fn recompute_blake3(
        &self,
        field_commitments: &[[u8; 32]; N],
    ) -> [u8; 32];
}
```

### Composition with existing primitives

- **Reuses `PersonalityWeightedComposition::compose_into`** for the inner sigmoid-blend loop (DRY — same kernel, different outer semantics).
- **Reuses `MicroRecurrentKernelSnapshot`** BLAKE3 pattern (Research 242) for the commitment hash.
- **Host supplies** the trajectory summary (KARC delay-embedding, ConvPool, or simpler EMA) and the K direction vectors.

### Sampling invariance contract

The defining property (FAME Proposition 3): if two observation grids encode the same underlying trajectory, the committed blend produces identical dynamics. **This holds because:**
1. `pi` is computed once from the summary, then frozen.
2. The fields `f_k` are frozen snapshots.
3. Therefore `f_π(z)` is a pure function of `z` — observation density does not enter the dynamics.

G2 gate tests this directly: simulate an entity with dense observations vs sparse observations (same underlying trajectory), verify the committed blend produces identical state evolution.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `crates/katgpt-core/src/committed_field_blend.rs` with `ArchetypeFieldSource<D>` trait + `CommittedFieldBlend<N, D>` struct.
- [ ] **T1.2** Implement `commit()` — sigmoid projection of summary onto K direction vectors, clamp to `pi_max`, BLAKE3 hash of `(pi, field_commitments, version)`.
- [ ] **T1.3** Implement `apply_blended()` — reuse `PersonalityWeightedComposition::compose_into` inner loop; outer loop over N fields calling `ArchetypeFieldSource::evolve`.
- [ ] **T1.4** Implement `verify_commitment()` + `recompute_blake3()` for thaw-time anti-tamper.
- [ ] **T1.5** Add feature gate `committed_field_blend = []` to `crates/katgpt-core/Cargo.toml` + re-export in `lib.rs`.
- [ ] **T1.6** Unit tests: `commit_produces_stable_pi`, `apply_blended_zero_when_all_pi_negative`, `apply_blended_selects_single_field_when_pi_extreme`, `verify_commitment_detects_tamper`, `sigmoid_stable_for_extreme_inputs`.

### Location

`crates/katgpt-core/src/committed_field_blend.rs`

---

## Phase 2 — Mechanics + Sampling Invariance GOAT Gate

### Tasks

- [ ] **T2.1 (G1)** `g1_mechanics`: random-init fields, random summaries, verify finite output, no NaN/Inf, sigmoid in [0,1], blend output in ℝ^D.
- [ ] **T2.2 (G2 — the defining property)** `g2_sampling_invariance`: 
  - Construct K=3 archetype fields (sin/cos/linear — synthetic but deterministic).
  - Generate a "true" trajectory of 1000 steps.
  - Sample it densely (every step) and sparsely (every 10 steps).
  - Compute `pi` from each summary, verify `pi_dense ≈ pi_sparse` to within 1e-3.
  - Apply the blended field from identical initial state, verify trajectories diverge by < 1e-3 over 100 steps.
- [ ] **T2.3 (G3)** `g3_no_regression`: verify `PersonalityWeightedComposition::compose_into` still passes its existing GOAT gate (Plan 297 bench) when `committed_field_blend` feature is enabled.
- [ ] **T2.4 (G4)** `g4_zero_alloc`: TrackingAllocator audit — 0 allocations in `apply_blended` after warmup.
- [ ] **T2.5 (G5)** `g5_blake3_reproducible`: two `commit()` calls with identical inputs produce identical BLAKE3; tampering any byte of `pi` flips the hash.

### Location

`crates/katgpt-core/src/committed_field_blend.rs` (inline test module)
`crates/katgpt-core/benches/committed_field_blend_bench.rs` (G4 alloc audit)

---

## Phase 3 — Lipschitz Safety Bound (commitment guarantee)

### Tasks

- [ ] **T3.1** Implement `lipschitz_bound()` on `CommittedFieldBlend` — returns `max_k { sigmoid(pi_k/tau) · L_k }` per FAME Lemma 1. This is the **deterministic safety bound** of the committed personality — a closed-form quantity that can be LatCal-committed.
- [ ] **T3.2** Test: blend of bounded fields produces bounded `lipschitz_bound`; tampering `pi` to extreme values saturates the bound at `max_k L_k`.
- [ ] **T3.3** Document the LatCal commitment story: the K-weight vector `pi` + the lipschitz bound cross the sync boundary as K+1 raw scalars; the archetype field definitions stay library-side (referenced by hash).

---

## Phase 4 — Examples + Documentation

### Tasks

- [ ] **T4.1** `examples/committed_blend_01_three_archetypes.rs` — K=3 synthetic archetype fields (aggressive/cautious/social analogues), 100 entities, each commits a blend from its trajectory summary, verify sampling invariance under fog-of-war gaps.
- [ ] **T4.2** `examples/committed_blend_02_recommit_on_event.rs` — demonstrate the re-commit trigger (major personality event → `commit()` called again → new BLAKE3, new version).
- [ ] **T4.3** Update `katgpt-rs/README.md` Feature Showcase with `committed_field_blend` entry.
- [ ] **T4.4** Update `katgpt-rs/.docs/01_overview.md` Feature Flags table.

---

## Files to Create/Modify

```
katgpt-rs/
├── crates/katgpt-core/
│   ├── Cargo.toml                          # Add `committed_field_blend = []` feature
│   ├── src/
│   │   ├── lib.rs                          # Re-export under feature gate
│   │   └── committed_field_blend.rs        # NEW: CommittedFieldBlend + ArchetypeFieldSource
│   └── benches/
│       └── committed_field_blend_bench.rs  # NEW: G4 alloc audit
├── examples/
│   ├── committed_blend_01_three_archetypes.rs  # NEW
│   └── committed_blend_02_recommit_on_event.rs # NEW
├── Cargo.toml                              # Add root feature alias
├── README.md                               # Feature Showcase entry
└── .docs/01_overview.md                    # Feature Flags table entry
```

---

## GOAT Gate — Promotion Criteria

Promote `committed_field_blend` to default-on ONLY if:
- G1 mechanics PASS (finite, bounded, sigmoid correct)
- G2 sampling invariance PASS (the defining property — dense vs sparse observation produce identical dynamics)
- G3 no regression on PersonalityWeightedComposition PASS
- G4 zero-alloc PASS
- G5 BLAKE3 reproducible + tamper-detecting PASS

**If any gate FAILS:** keep opt-in, document the failure honestly in `.benchmarks/`, do NOT promote.

---

## Cross-references

- **Research:** [302_FAME_Sampling_Invariant_Per_Entity_MoE](../.research/302_FAME_Sampling_Invariant_Per_Entity_MoE.md)
- **Closest shipped cousin (per-layer, drifting):** Plan 297 (`PersonalityWeightedComposition`)
- **Forecast partner (Bi-NCDE backward pass):** Plan 308 (`KarcForecaster`)
- **DEC sampling-invariance substrate:** Plan 314 (`line_integral`), Research 296
- **Freeze substrate:** `riir-neuron-db/src/shard.rs` (NeuronShard + future ArchetypeBlendShard subtype)
- **Commitment bridge:** `riir-chain/src/encoding/latcal.rs` (LatCal fixed-point commitment of K-weight vector)
- **Private Super-GOAT guide:** `riir-ai/.research/158_per_npc_committed_personality_blend_guide.md`
- **Runtime integration plan (deferred):** `riir-ai/.plans/336_committed_personality_runtime_integration.md`

---

## TL;DR

Ship `CommittedFieldBlend<N, D>` — a per-entity frozen convex blend of N operator fields with sigmoid-computed weights derived once from a trajectory summary and BLAKE3-committed. The defining property is **sampling invariance** (FAME Proposition 3): because both the weights and the fields are frozen, the entity's dynamics depend only on its state, not on observation density. Reuses `PersonalityWeightedComposition::compose_into` for the inner blend loop. K=3 default. GOAT gate G1–G5; the make-or-break gate is G2 (sampling invariance under observation gaps). Open primitive in katgpt-rs; private selling-point guide in riir-ai/.research/158.
