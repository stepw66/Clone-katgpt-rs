# Plan 309: Latent Field Steering — Open Primitive

**Date:** 2026-06-23
**Research:** [katgpt-rs/.research/290_latent_field_steering_open_primitive.md](../.research/290_latent_field_steering_open_primitive.md)
**Source:** Synthesized from CAA + Anthropic Functional Emotions + Gemini "wave interference" reframing
**Target:** `katgpt-rs/crates/katgpt-core/src/latent_steering.rs` (new module) + Cargo feature `latent_field_steering`
**Status:** Active — Phase 1 (skeleton, pending implementation)

---

## Goal

Ship the minimal concrete prototype of Latent Field Steering: a zero-allocation,
SIMD-accelerated primitive for injecting a frozen direction vector into a mutable
latent state slice, with optional localized support (radius, zone, graph
neighborhood). Prove or kill the Super-GOAT candidate (Research 290) via 5 GOAT
gates measuring steering strength, behavior rank preservation, localization,
crowd-scale performance, and zero-allocation steady state.

**Proves the idea:** G1 ≥30% affect shift · G2 cos(action_rank) ≥0.95 ·
G3 zero leakage at d > b+ε · G4 5000-NPC crowd <1ms · G5 0 allocs after warmup.

**Kills the idea:** G2 <0.95 (steering corrupts decisions) **OR** G3 >0.01
leakage (uncontrolled propagation) **OR** G4 >1ms (too slow for 20Hz tick).

---

## Phase 0 — Design (COMPLETE)

- [x] T0.1 Research note created ([Research 290](../.research/290_latent_field_steering_open_primitive.md))
- [x] T0.2 Private guide created ([riir-ai/.research/153](../../../riir-ai/.research/153_latent_field_steering_game_runtime_guide.md))
- [x] T0.3 Plan created (this file)
- [x] T0.4 Fusion grep complete: zero codebase hits for residual-stream steering on hot path; closest cousins are CNA (neuron-level), EmotionDirections (read-only), FPCG (explicit non-mutation)

---

## Phase 1 — Unblocking Skeleton (CORE)

**Target:** minimal compilable primitive behind feature flag. No perf optimization, no GOAT gate. Ships the trait + scalar impl + smoke tests.

### Tasks

- [ ] T1.1 Create `katgpt-rs/crates/katgpt-core/src/latent_steering.rs`:

  ```rust
  //! Latent Field Steering — top-down direction-vector injection into latent state.
  //! See katgpt-rs/.research/290_*.md and Plan 309.
  //!
  //! Modelless: direction vectors are frozen BLAKE3-committed artifacts loaded
  //! at init. No gradients. The steering is an ADDITIVE OVERLAY on mutable
  //! per-tick state — it does NOT mutate the frozen personality shard.

  /// Unit-norm direction in latent space. Reuse existing `DirectionVector`
  /// shape from `EmotionDirections` so the same artifact format works for
  /// read-side (project) and write-side (steer).
  #[derive(Debug, Clone)]
  pub struct LatentSteeringVector {
      /// Unit-norm direction, d ≤ 64 (HLA d=8).
      pub direction: Vec<f32>, // or [f32; D] generic; see T1.2 design note
      /// Strength α ∈ [0, 1]. Sigmoid-bounded.
      pub alpha: f32,
      /// BLAKE3 of (serialized direction || alpha_le_bytes), for commitment check.
      pub commitment: [u8; 32],
  }

  impl LatentSteeringVector {
      /// Verify the direction is unit-norm within tol and the commitment matches.
      pub fn verify(&self, tol: f32) -> bool {
          let norm: f32 = self.direction.iter().map(|x| x * x).sum::<f32>().sqrt();
          if (norm - 1.0).abs() > tol { return false; }
          // TODO: recompute blake3(direction || alpha_le) and compare
          true
      }
      pub fn dim(&self) -> usize { self.direction.len() }
      pub fn as_slice(&self) -> &[f32] { &self.direction }
  }

  /// Support descriptor for a localized steering field.
  #[derive(Debug, Clone, Copy)]
  pub enum FieldSupport {
      /// Global — applies to all entities.
      Global,
      /// Radius-banded — applies within `bandwidth` of `center` (Euclidean).
      /// Kernel: sigmoid((bandwidth - distance) * steepness).
      Radius { center: [f32; 2], bandwidth: f32, steepness: f32 },
      /// Zone-keyed — applies to entities whose zone hash matches.
      Zone { zone_hash: u64 },
  }

  /// A steering vector + support descriptor.
  #[derive(Debug, Clone)]
  pub struct LatentField {
      pub steering: LatentSteeringVector,
      pub support: FieldSupport,
  }

  /// Apply steering to a single latent state slice. Zero-alloc.
  /// `state` is d-dimensional (e.g., HLA 8-dim).
  #[inline]
  pub fn apply_latent_steering(state: &mut [f32], field: &LatentField) {
      debug_assert_eq!(state.len(), field.steering.dim());
      let alpha = field.steering.alpha;
      let dir = field.steering.as_slice();
      for (s, d) in state.iter_mut().zip(dir.iter()) {
          *s += alpha * d;
      }
  }

  /// Kernel weight for an entity given support. Returns 0.0 outside support.
  #[inline]
  pub fn kernel_weight(
      support: &FieldSupport,
      entity_pos: Option<[f32; 2]>,
      entity_zone: Option<u64>,
  ) -> f32 {
      match support {
          FieldSupport::Global => 1.0,
          FieldSupport::Radius { center, bandwidth, steepness } => {
              let pos = match entity_pos { Some(p) => p, None => return 0.0 };
              let dx = pos[0] - center[0];
              let dy = pos[1] - center[1];
              let dist = (dx * dx + dy * dy).sqrt();
              sigmoid((bandwidth - dist) * steepness)
          }
          FieldSupport::Zone { zone_hash } => match entity_zone {
              Some(z) if z == *zone_hash => 1.0,
              _ => 0.0,
          },
      }
  }

  #[inline]
  fn sigmoid(x: f32) -> f32 {
      1.0 / (1.0 + (-x).exp())
  }

  /// Apply a field to a crowd of latent states. Zero-alloc given borrowed slices.
  /// `states` is flattened `[e0d0, e0d1, ..., eNd(D-1)]` (N*D).
  pub fn apply_field_to_crowd(
      states: &mut [f32],
      entity_dim: usize,
      positions: &[Option<[f32; 2]>],
      zones: &[Option<u64>],
      field: &LatentField,
  ) {
      debug_assert_eq!(states.len(), positions.len() * entity_dim);
      debug_assert_eq!(positions.len(), zones.len());
      // Phase 2: rayon par_chunks_mut when N > threshold (per AGENTS.md ~5µs rule)
      for (i, entity_state) in states.chunks_mut(entity_dim).enumerate() {
          let w = kernel_weight(&field.support, positions[i], zones[i]);
          if w <= 0.0 { continue; }
          let alpha = field.steering.alpha * w;
          let dir = field.steering.as_slice();
          for (s, d) in entity_state.iter_mut().zip(dir.iter()) {
              *s += alpha * d;
          }
      }
  }

  #[cfg(test)]
  mod tests {
      use super::*;
      #[test] fn smoke_global_field_shifts_state() { /* T1.4 */ }
      #[test] fn smoke_radius_field_localizes() { /* T1.4 */ }
  }
  ```

- [ ] T1.2 Design decision: `direction: Vec<f32>` vs generic `[f32; D]`. Lean
      toward `Vec<f32>` for the open primitive (dynamically sized, matches
      `EmotionDirections` storage). Game-side hot path can wrap in a typed
      `HLAField([f32; 8])` alias in riir-ai. Document in module doc.

- [ ] T1.3 Add feature gates. In `katgpt-rs/crates/katgpt-core/Cargo.toml`:
  ```toml
  latent_field_steering = []  # Latent Field Steering — top-down direction-vector injection (Plan 309, Research 290). Opt-in until G1-G5 GOAT gate passes.
  ```
  In `katgpt-rs/Cargo.toml`:
  ```toml
  latent_field_steering = ["katgpt-core/latent_field_steering"]
  ```

- [ ] T1.4 Wire module into `katgpt-core/src/lib.rs`:
  ```rust
  #[cfg(feature = "latent_field_steering")]
  pub mod latent_steering;
  ```

- [ ] T1.5 Smoke tests pass: construct a `LatentField { Global }`, apply to an
      8-dim state, assert the state changed in the direction of `v` by exactly
      `α`. Construct a `Radius` field at (0,0) b=10, apply to a 100-NPC crowd
      half inside / half outside, assert the outside half is unchanged.

- [ ] T1.6 `cargo check -p katgpt-core --features latent_field_steering` clean.

---

## Phase 2 — GOAT Gate (PROVES OR KILLS)

Each gate is a standalone file. All must pass to promote from opt-in.

- [ ] T2.1 **G1 — Steering strength (≥30% affect shift).** File:
      `tests/latent_steering_g1_strength.rs`. Construct a "high anxiety" vector
      aligned with the HLA `fear` axis (one-hot at the fear index, normalized).
      Apply to a baseline 8-dim state with α=0.5. Measure the fear-axis
      projection before and after. **Gate:** post/pre ≥1.30 (≥30% shift).

- [ ] T2.2 **G2 — Behavior rank preservation (mean cos ≥0.95, worst ≥0.90).**
      File: `tests/latent_steering_g2_rank_preservation.rs`. **THE headline
      gate.** Generate 100 random 8-dim latent states. For each, compute an
      action ranking by dotting with a fixed 8×5 action-weight matrix (5
      candidate actions, reuse `latent_functor` action scoring or a stub).
      Apply steering with a random unit direction at α=0.3. Recompute action
      ranking. Measure cosine similarity of pre/post ranking vectors.
      **Gate:** mean cos ≥0.95, min cos ≥0.90. **If this fails, abandon the
      primitive — steering corrupts decisions.**

- [ ] T2.3 **G3 — Localization (zero leakage at d > b+ε).** File:
      `tests/latent_steering_g3_localization.rs`. Radius field at center (0,0),
      bandwidth=10, steepness=2.0. 50 NPCs inside at d=5, 50 NPCs outside at
      d=15. Apply. Measure per-NPC shift magnitude. **Gate:**
      `mean_outside_shift / mean_inside_shift < 0.01`.

- [ ] T2.4 **G4 — Crowd-scale perf (5000 NPCs <1ms).** File:
      `benches/latent_steering_g4_crowd.rs` (harness=false, std::time::Instant,
      no criterion dep — per existing bench convention). Global field applied
      to 5000 8-dim latent states. **Gate:** median wall-clock <1ms in release
      build. Run 1000 iterations, report p50/p95/p99.

- [ ] T2.5 **G5 — Zero-alloc steady state.** File:
      `tests/latent_steering_g5_zero_alloc.rs`. Debug-only via
      `katgpt_rs::alloc::TrackingAllocator`. Warmup 10 iterations, then count
      allocations on the next 1000 crowd-applies. **Gate:** 0 allocations after
      warmup.

---

## Phase 3 — SIMD Acceleration (if G1–G5 pass)

- [ ] T3.1 Replace the scalar loop in `apply_latent_steering` with manual SIMD
      (8× f32 for d=8 HLA via `std::arch::x86_64::_mm256_add_ps` and
      `_mm256_mul_ps`, with fallback scalar path for non-AVX2 targets).
- [ ] T3.2 Benchmark SIMD vs scalar on d=8 and d=16. **Gate:** ≥2× speedup at
      d=8, ≥1.5× at d=16. Re-run G4 with SIMD path; gate still <1ms.

---

## Phase 4 — Promotion Decision

- [ ] T4.1 If G1–G5 all pass → promote to opt-in default in katgpt-rs feature
      showcase; document in README "Feature Showcase" section.
- [ ] T4.2 If G2 fails → demote to research-only. Document the failure mode in
      Research 290 §5. Do not ship to any hot path.
- [ ] T4.3 If G1/G3/G4/G5 fail (but G2 passes) → fix and re-run; if persistent
      after 2 attempts, demote to Gain (plan-only, no default promotion).

---

## Phase 5 — Game Integration (DEFERRED to riir-ai)

Blocked on Phase 2 G1–G2 pass. See
[riir-ai/.research/153](../../../riir-ai/.research/153_latent_field_steering_game_runtime_guide.md)
for the integration guide.

- [ ] T5.1 HLA post-evolve wiring in `riir-engine/src/hla/`: after
      `evolve_hla(...)`, call `apply_latent_steering(state, field)` for each
      active field on the entity. Field registry per zone.
- [ ] T5.2 CWM soft-rule → field mapping: induced rules with `soft: true`
      emit a `LatentField` instead of a hard constraint.
- [ ] T5.3 Faction "battle stance" frozen field: stored as
      `MerkleFrozenEnvelope<LatentSteeringVector>`, atomic Arc swap on stance
      change.
- [ ] T5.4 `GrudgeMemory` (Plan 317) integration: grudge emits a long-lived
      `LatentField` applied when the target is within visible radius.

---

## Cross-Refs

- [katgpt-rs/.research/290_latent_field_steering_open_primitive.md](../.research/290_latent_field_steering_open_primitive.md) — research note
- [riir-ai/.research/153_latent_field_steering_game_runtime_guide.md](../../../riir-ai/.research/153_latent_field_steering_game_runtime_guide.md) — private guide
- [katgpt-rs/.plans/162_emotion_vector_inference_control.md](162_emotion_vector_inference_control.md) — read-only counterpart
- [katgpt-rs/.plans/087_cna_contrastive_neuron_attribution.md](087_cna_contrastive_neuron_attribution.md) — neuron-level mutation counterpart
- [katgpt-rs/.plans/286_functional_attention_spectral_transport.md](286_functional_attention_spectral_transport.md) — F2 fusion target (cross-domain steering)
- [katgpt-rs/.plans/297_personality_weighted_composition.md](297_personality_weighted_composition.md) — F3 cousin (composition vs overlay)
