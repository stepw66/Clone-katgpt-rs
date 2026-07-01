# Plan 357: Motor-Gated DEC Propagation Primitive

**Date:** 2026-07-01
**Research:** [katgpt-rs/.research/359_Isomorphic_Neural_Field_World_Model_Motor_Gated_DEC_Propagation.md](../.research/359_Isomorphic_Neural_Field_World_Model_Motor_Gated_DEC_Propagation.md)
**Source paper:** [arXiv:2602.18690](https://arxiv.org/abs/2602.18690) — Nunley, *Neural Fields as World Models* (CogSci 2026)
**Private guide:** [riir-ai/.research/168_Motor_Gated_Isomorphic_World_Model_Game_Runtime_Guide.md](../../riir-ai/.research/168_Motor_Gated_Isomorphic_World_Model_Game_Runtime_Guide.md)
**Target:** `katgpt-rs/crates/katgpt-dec/src/motor_gated.rs` (new module) + Cargo feature `motor_gated_field` (opt-in)
**Status:** Active — Phase 0 (skeleton pending)

---

## Goal

Ship the **missing glue primitive** that unifies the DEC substrate (Plan 251 `hodge_laplacian`) with latent steering (Plan 309 `apply_latent_steering_weighted`) into a single Amari-style motor-gated neural-field evolution step. The wrapper implements `h_{t+1} = h_t + dt·(-h_t + K*ReLU(h_t) + motor·h_t)` where `K*ReLU(h_t)` is `relu_gate → hodge_laplacian` and `motor·h_t` is elementwise per-channel gain. This is the open half of the Super-GOAT declared in Research 359 — the adoption hook. The closed half (game-runtime selling point: per-NPC offline rehearsal through a frozen spatial field) lives in riir-ai Research 168.

**The GOAT gate (Phase 2)** proves five properties: no-teleporting (G1), motor-gate locality (G2), belief-mass conservation (G3), zero-alloc steady state (G4), and sub-100µs latency on a 64×64 grid (G5). If all five pass, `motor_gated_field` is **ready for downstream consumption** (stays opt-in by design — it's a primitive, not a default-on capability).

**Modelless (katgpt-rs mandate):** every step is closed-form algebra over the shipped DEC operators. No training, no backprop. The Amari `K` is the analytic `hodge_laplacian`; the motor gate is elementwise scalar multiply; the ReLU is a per-element gate. The `InducedCwmKernel`-as-frozen-world-model composition (Experiment 2) and the body-schema emergence tuning (Experiment 3) are riir-ai concerns, gated on this primitive landing first.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-dec/src/motor_gated.rs` with module doc referencing Research 359 + Plan 357 + the paper.
- [ ] **T1.2** Add Cargo feature `motor_gated_field = []` to `katgpt-dec/Cargo.toml` (opt-in, no extra deps — composes existing `dec_operators` surface).
- [ ] **T1.3** Implement `evolve_motor_gated_field(cx, h, motor_vec, motor_dim, dt, relu_slope, scratch_lap, scratch_relu)`:
  - Step 1: `relu_gate_into(h, relu_slope, scratch_relu)` — per-element `max(0, x)` (or leaky `relu_slope·x` for negative) into scratch.
  - Step 2: `hodge_laplacian_into(cx, scratch_relu, scratch_lap, ...)` — the lateral propagation.
  - Step 3: Elementwise update `h[i] += dt · (-h[i] + scratch_lap[i])` for all cells/channels.
  - Step 4: Motor gate — for cells, for channels `0..motor_dim`: `h[cell, ch] *= (1.0 + dt · motor_vec[ch])` (the `m_i · h̃` gain).
  - All four steps zero-alloc (caller-owned scratch).
- [ ] **T1.4** Re-export `evolve_motor_gated_field` from `katgpt-dec/src/lib.rs` under `#[cfg(feature = "motor_gated_field")]`.
- [ ] **T1.5** Re-export from `katgpt-core/src/lib.rs` as `katgpt_core::dec::evolve_motor_gated_field` under the same feature (mirror the DEC re-export shim).
- [ ] **T1.6** Write 3 smoke tests:
  - `motor_free_ballistic_propagates` — 32×32 grid, single bump at center, no motor (`motor_dim=0`), 10 ticks; bump spreads locally (max displacement ≤ 2 cells).
  - `motor_gate_shifts_only_gated_channels` — 4-channel field, motor on channels 0..2 only; channels 2..4 unchanged after one tick.
  - `zero_alloc_steady_state` — `TrackingAllocator` audit on 100 ticks; 0 allocations after warmup.

---

## Phase 2 — GOAT Gate (G1–G5)

### Tasks

- [ ] **T2.1 G1 — No-teleporting.** Propagate a ballistic bump on a 32×32 grid (mirroring paper Experiment 1); measure max frame-to-frame centroid displacement across 50 ticks. **Gate:** ≤ kernel radius (no jumps > stencil width). Compare against a naive dense-matrix propagation baseline (should show teleporting).
- [ ] **T2.2 G2 — Motor-gate locality.** Apply motor gate to channels 0..M on a 16-channel field; verify only those channels shift, others conserve. **Gate:** channel-isolation ratio > 100× (gated channel L1 shift / ungated channel L1 shift).
- [ ] **T2.3 G3 — Conservation.** `belief_mass_divergence(cx, &h_propagated) < τ` for `τ` derived from the cell complex's harmonic component (`harmonic_projector` norm). **Gate:** divergence < 5% of field L1 norm across 100 ticks.
- [ ] **T2.4 G4 — Zero-alloc steady state.** `TrackingAllocator` debug audit on the hot path (1000 ticks, 64×64 grid, 16 channels). **Gate:** 0 allocations after warmup.
- [ ] **T2.5 G5 — Latency.** Criterion bench: 64×64 grid, 16 channels, single `evolve_motor_gated_field` call. **Gate:** < 100µs (vs the paper's GPU conv at ~ms scale; we're CPU SIMD).
- [ ] **T2.6** Write `.benchmarks/357_motor_gated_field_goat.md` with the G1–G5 results table + promotion decision.

**Promotion rule:** all 5 PASS → keep `motor_gated_field` opt-in but mark ready for downstream consumption (riir-ai Research 168 Phase 2). Any FAIL → stay opt-in, file `.issues/NNN_*` follow-up, do NOT promote.

---

## Phase 3 — SIMD Acceleration (optional, if G5 is tight)

### Tasks

- [ ] **T3.1** If G5 latency is within 2× of target (50–100µs), add explicit SIMD chunking to the motor-gate step (mirrors `hodge_laplacian_into`'s 4-wide chunk pattern in `katgpt-dec/src/operators.rs`).
- [ ] **T3.2** Re-run G5; document speedup.

---

## Phase 4 — Cross-References + Examples

### Tasks

- [ ] **T4.1** Add `katgpt-rs/examples/motor_gated_field_01_ballistic.rs` — single bump propagation on a 32×32 grid, ASCII-art visualization of the field every 10 ticks.
- [ ] **T4.2** Add `katgpt-rs/examples/motor_gated_field_02_motor_gate.rs` — 4-channel field with motor on 2 channels; show gated channels shift, others conserve.
- [ ] **T4.3** Update `katgpt-rs/crates/katgpt-dec/src/lib.rs` module doc to mention `motor_gated_field` in the "What's here" list.

---

## Out of Scope (do NOT bundle)

- **HLA-cell-complex wiring** — riir-ai Research 168 concern.
- **Fourier-physics → motor-gated-field bridge** — riir-ai Research 168 concern.
- **`InducedCwmKernel` integration** (the frozen-world-model composition) — riir-ai Research 168 concern. This primitive is the *building block*; the composition lives downstream.
- **`sleep_time` offline-rehearsal pipeline** (Experiment 2) — riir-ai Research 168 concern.
- **Body-schema emergence tuning** (Experiment 3) — riir-ai Research 168 concern.
- **End-to-end backprop training of the lateral kernel** — riir-train (non-blocking follow-up; our modelless path uses the analytic `hodge_laplacian`).
- **SE(2) equivariant variant** — riir-ai Research 166 is the rotation-equivariant cousin; different equivariance class, separate primitive.

---

## References

- **Paper:** [arXiv:2602.18690](https://arxiv.org/abs/2602.18690) — Nunley, *Neural Fields as World Models*.
- **Research:** [359_Isomorphic_Neural_Field_World_Model_Motor_Gated_DEC_Propagation.md](../.research/359_Isomorphic_Neural_Field_World_Model_Motor_Gated_DEC_Propagation.md)
- **Private guide:** [riir-ai/.research/168_Motor_Gated_Isomorphic_World_Model_Game_Runtime_Guide.md](../../riir-ai/.research/168_Motor_Gated_Isomorphic_World_Model_Game_Runtime_Guide.md)
- **DEC substrate:** [Plan 251](251_dec_operators_cell_complex.md), [Research 219](../.research/219_Topological_Neural_Operators_DEC_Inference.md)
- **Stokes wrappers:** [Plan 314](314_stokes_calculus_wrappers.md), [Research 296](../.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md) — `belief_mass_divergence` is the G3 validator.
- **Latent steering:** [Plan 309](309_latent_field_steering_primitive.md) — the motor-gate algebra.
- **InducedCwmKernel:** [Plan 296](296_induced_cwm_kernel_primitive.md) — the frozen-world-model substrate (downstream consumer).
- **sleep_time:** [Plan 341](../.plans/341_npc_sleep_time_anticipation_runtime.md) (riir-ai) — the offline consolidation cycle (downstream consumer).
