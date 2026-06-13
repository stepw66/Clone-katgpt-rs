# Plan 261: Dynamic DEC Topology for Destructible Terrain

**Date:** 2026-06
**Status:** 🔵 ACTIVE
**Research:** [230 Arena × Latent Space](../.research/230_Worms_Armageddon_Latent_Space_Artillery_Game.md)
**Depends On:** Plan 251 (DEC Operators), Plan 242 (Fourier Flow Fields)

---

## Goal

Extend existing DEC operators (`katgpt-rs-core/src/dec/`) to support **dynamic topology updates** — terrain cells that can be destroyed in real-time, invalidating and recomputing DEC operators on the fly. This is the modelless primitive layer that the riir-armageddon game will consume.

## Why Separate Plan

DEC dynamic topology is a **generic modelless primitive** — it's useful for any game with destructible terrain, not just one specific game. It belongs in katgpt-rs (MIT/engine), not riir-armageddon (private/fuel).

---

## Tasks

### Phase 0: Dynamic CellComplex
- [ ] Add `CellComplex::remove_cell()` — removes a cell from the complex and updates incidence matrices
- [ ] Add `CellComplex::remove_face()` — removes a face (terrain chunk) and updates edge-vertex incidence
- [ ] Add `CellComplex::topology_version()` — monotonically increasing version counter, incremented on any mutation
- [ ] Add `CellComplex::is_dirty_since(version: u64) -> bool` — cheap dirty check for caching
- [ ] Ensure `d₀`, `d₁`, `d₂`, `δₖ`, `Δₖ` operators correctly recompute after topology change

### Phase 1: Incremental DEC Updates
- [ ] Add `DecCache` struct — caches Hodge decomposition results keyed by `topology_version()`
- [ ] Implement incremental Hodge recomputation — only recompute affected rows/columns after local topology change
- [ ] Add dirty-region tracking — mark which regions of the cell complex changed, only recompute those
- [ ] Benchmark: full recomputation vs incremental for 1-cell, 10-cell, 100-cell destruction events

### Phase 2: Terrain-Specific Cochains
- [ ] Add `terrain_cochains` module with arena-relevant cochain definitions:
  - `SafetyCochain` (C₀) — scalar per vertex: how safe is this position?
  - `ThreatCochain` (C₁) — vector per edge: threat direction/magnitude
  - `OccupancyCochain` (C₂) — scalar per face: how many entities in this area?
  - `DestructionCochain` (C₀) — scalar per vertex: how destroyed is this terrain?
- [ ] Add bridge functions: `SafetyCochain::from_projectile_threat()` — raw trajectory → safety score via sigmoid

### Phase 3: GOAT Gate Validation
- [ ] Create `examples/dec_terrain_bench.rs` — benchmark DEC terrain update vs naive grid scan
- [ ] Measure: time to update navigation after N terrain destructions
- [ ] Measure: quality of Hodge-decomposed routes vs A* on modified terrain
- [ ] If DEC wins → promote `dec_terrain_ai` to default feature
- [ ] If DEC loses → demote, document why, create issue for optimization

### Phase 4: Integration with Existing Flow Fields
- [ ] Add `DecFlowField::recompute_if_dirty(&mut self, complex: &CellComplex, cache: &DecCache)` — only recompute if topology changed
- [ ] Wire `FlowFieldCache` to use `topology_version()` for dirty threshold
- [ ] Ensure `flow_steering()` works correctly on post-destruction terrain

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Incremental vs full recomputation | Both, benchmark | Full is simpler, incremental is faster for small changes |
| Cache key | topology_version (u64) | O(1) dirty check, no hash needed |
| Terrain cochains | Separate module, not in core DEC | Arena-specific (fuel), but built on engine primitives |
| SIMD threshold | Same as existing DEC backend | < 1K scalar, 1K-10K SIMD, > 10K GPU |

---

## Performance Targets

| Metric | Target | Backend |
|--------|--------|---------|
| Remove 1 cell + recompute d₀ | < 10μs | CPU |
| Remove 100 cells + full Hodge | < 500μs | SIMD |
| `is_dirty_since()` check | < 1ns | CPU |
| Terrain cochain projection | < 50μs | CPU |

TL;DR: Dynamic DEC topology — cells can be destroyed, operators recomputed incrementally, cached by version. Modelless primitive for destructible terrain games.
