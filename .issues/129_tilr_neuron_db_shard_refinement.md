# Issue 129 — TILR Consumer Wiring: riir-neuron-db Freeze/Thaw Shard Refinement

> **Spawned from:** Plan 425 (TILR), T4.3 consumer wiring follow-up
> **Date:** 2026-07-10
> **Type:** feature (consumer integration)
> **Severity:** MEDIUM — concrete consumer value, but no blocking trigger
> **Status:** IMPLEMENTED (Plan 317, opt-in `tilr_consolidation` feature) — GOAT G1-G4 PASS, promotion deferred pending real-data gain validation

## Context

Plan 425 shipped the TILR primitive as DEFAULT-ON in `katgpt-core`. This issue
tracks wiring TILR into riir-neuron-db's freeze/thaw shard refinement path:
when a `NeuronShard` is consolidated and committed to the Cold tier, use TILR
to refine the shard's `style_weights` along the invariant subspace discovered
from the consolidation delta, with a no-harm guarantee for shards whose style
doesn't align with the calibrated direction.

## Why TILR fits this use case

`NeuronShard::style_weights` is a fixed-size `[f32; 64]` Pod. TILR's
`tilr_refine_into` operates on flat `&[f32]` slices, making it a natural fit.
The no-harm contract ensures that if a shard's consolidation delta doesn't
align with the invariant subspace (e.g., the shard hasn't changed in a
meaningful direction), the style_weights are left bit-identically unchanged —
no spurious corruption of committed shards.

The `can_freeze` gate (Plan 002 Phase 5) already validates consolidation
quality; TILR adds a complementary refinement step that can improve shard
quality without risk of degradation.

## Proposed integration

1. During the Raven/δ-Mem consolidation pipeline (`consolidation.rs`), collect
   the consolidation delta (before/after style_weights difference).
2. At sleep-cycle boundaries, accumulate deltas and call
   `discover_invariant_subspace` to build the basis.
3. Apply `tilr_refine_into` to the committed shard's style_weights.

**Scale note:** `style_weights[64]` → `d=64`. Plan 425's GOAT gate measured
shard-scale latency at 123.0 ns (d=64, r=12), well under the 200 ns target.

## Tasks

- [x] **T1** Identify the consolidation commit path in `riir-neuron-db/src/consolidation/mod.rs`.
      ✅ Located: `ConsolidationPipeline::consolidate` at `consolidation/mod.rs:1011`,
      `shard.apply_delta(&delta, 0.3)` at L1024.
- [x] **T2** Collect consolidation deltas across a sleep cycle. Build the
      contrastive difference set.
      ✅ `wake_events[].embedding: [f32; 64]` already available; fed to
      `discover_invariant_subspace` to build the basis.
- [x] **T3** Wire `tilr_refine_into` into the commit path behind a feature flag.
      ✅ `consolidate_tilr` method (Plan 317), gated on `tilr_consolidation`.
- [x] **T4** Benchmark: verify zero-harm on non-aligned shards, refinement on
      aligned shards.
      ✅ G1-G4 GOAT gate PASS (6 tests). No-harm verified on fallback paths
      (bit-identical BLAKE3). Refinement verified on truncated/full-rank
      subspaces.
- [-] **T5** If the gain is real and modelless → promote to default-on.
      DEFERRED: GOAT gate passes, gain is modelless, but "proof gain" on
      real consolidation workloads needed before promotion. riir-ai G8
      follow-up.

## Cross-references

- `katgpt-rs/.plans/425_tilr_invariant_subspace_refinement.md` — COMPLETE, DEFAULT-ON
- `katgpt-rs/.research/408_*.md` — TILR research note (GOAT verdict)
- `katgpt-rs/.docs/05_adaptation/tilr_subspace_family.md` — family overview
- `riir-neuron-db/AGENTS.md` — `can_freeze` gate audit lesson (Plan 002 Phase 5)
