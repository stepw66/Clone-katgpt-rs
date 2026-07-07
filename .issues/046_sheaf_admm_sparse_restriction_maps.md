# Issue 046 — Sheaf-ADMM CSR sparse restriction maps for K>1000

**Source:** Extracted from Plan 407 Phase 3 T3.2 (post-promotion scaling optimization).
**Primitive:** `sheaf_admm` (katgpt-dec, **DEFAULT-ON** since 2026-07-07, G1–G6 ALL PASS).
**Parent research:** [`.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md`](../.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md)

## Why this is an issue, not a plan task

Per `AGENTS.md`: *optimization/refactor tasks go to `.issues/`, not plans*. This is a server-scale memory-layout optimization of a shipped primitive.

## The optimization

`SheafMaps` currently materializes all edges as dense `Vec<[MatrixDimDExDV; 2]>`. At K>1000 vertices on sparse zone-residency graphs (each NPC adjacent to a handful of neighbors, not all), the dense edge list wastes memory and the z-update matvec scans zeros.

A CSR-like sparse representation (`SheafMapsSparse`) would store only the actual edges + their restriction-map rows, making the z-update `O(|E| · d_e)` instead of `O(K² · d_e)`.

- **Target regime:** K>1000 vertices on graphs with average degree << K (server-scale zones).
- **Current floor:** dense maps — correct, but wasteful on sparse graphs.
- **Promotion rule:** sparse variant is opt-in until a consumer (riir-ai Plan 394 Crowd MCGS) actually exercises K>1000 AND the dense layout shows measurable memory/latency pain. File this only when a real consumer needs it.

## Acceptance

- [ ] Implement `SheafMapsSparse` (CSR edge list + per-edge restriction-map rows).
- [ ] Bench: dense-vs-sparse z-update at K=2000, avg degree 8, d_v=8, d_e=5. Record memory + latency.
- [ ] If sparse wins on memory AND latency at K>1000 → ship as opt-in alternative. If a consumer doesn't materialize → leave dormant.

## Notes

- Coordinate with riir-ai Plan 394 Phase 3 (Crowd MCGS integration). Do NOT implement speculatively — wait for the consumer to demonstrate the need.
- The sparse path must remain modelless (deterministic edge construction from caller-supplied adjacency).
