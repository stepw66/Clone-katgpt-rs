# Plan 319 — G8e: AOI Pairwise Complementarity Latency Gate

**Date:** 2026-06-25
**Primitive:** `katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs`
**Bench:** `cargo bench -p katgpt-core --features geometric_product --bench bench_319_g8e_aoi_latency -- --nocapture`
**Hardware:** macOS (Apple Silicon, aarch64), macOS 26.5.1

---

## TL;DR

**G8e: PASS.** AOI-scored pairwise complementarity at crowd scale (1000 NPCs ×
20 partners × D=64) fits the 5 ms 60Hz budget with **1.50× headroom** (3.34 ms
mean tick). Zero allocations per tick in the steady state. This validates the
perf budget for Research 299's Super-GOAT Q3 product-selling-point gate — the
complementarity bridge can evaluate every NPC's AOI partner set every tick in
real time.

---

## The Gate

Simulate a worst-case-density crowd:
- **1000 NPCs** (social hub zone density)
- **20 AOI partner candidates** per NPC (generous AOI)
- **D=64** (CGSP `DEFAULT_HLA_DIM`)
- Per tick: every NPC scores every partner via `geometric_product_wedge_into`
  + L1 norm + sigmoid + tau gate (the exact `clifford_bridge::
  complementarity_target` workload)
- **20,000 pairs/tick** total

**PASS criterion:** mean tick wall-clock < **5 ms** (60Hz-frame headroom slice)
AND zero allocations per tick (scratch reused).

---

## Results

```
Crowd: 1000 NPCs × 20 partners = 20000 pairs/tick, D=64
Shifts: [1, 2, 4, 8, 16, 32] (|S|=6), beta=1, tau=0.6
Budget: mean tick < 5.0 ms (100 ticks measured, 20 warmup)

[warmup: 20 ticks in 67.4 ms, 3.371 ms/tick]
── Results ──
  mean tick:       3.340 ms  (target < 5.0 ms)  ✓ PASS
  min tick:        3.147 ms
  p50 tick:        3.331 ms
  p99 tick:        3.571 ms
  max tick:        4.094 ms
  per-pair:        167.0 ns  (20000 pairs/tick)
  allocs/tick:         0   (target 0)  ✓ PASS
  complementarity hit rate: 100.0%  (20000/20000 pairs above tau=0.6)

════════════════════════════════════════════════════════════════
  G8e VERDICT:  perf=PASS  alloc=PASS
  → G8e PASS. AOI-scored pairwise complementarity fits the 5 ms
    60Hz budget with 1.50× headroom (3.340 ms / 5.0 ms).
  → Unblocks G8c/G8d runtime sims (perf budget is non-blocking).
════════════════════════════════════════════════════════════════
```

| Metric | Value | Target | Result |
|--------|-------|--------|--------|
| mean tick | 3.340 ms | < 5.0 ms | ✓ **PASS** (1.50× headroom) |
| p50 tick | 3.331 ms | — | stable mean (no jitter) |
| p99 tick | 3.571 ms | < 5.0 ms | ✓ **PASS** (excellent tail) |
| max tick | 4.094 ms | < 5.0 ms | ✓ **PASS** (worst case still safe) |
| per-pair | 167.0 ns | ~200 ns (G4-wedge ref) | ✓ matches G4-wedge (201ns isolated, better in-context) |
| allocs/tick | 0 | 0 | ✓ **PASS** |
| hit rate | 100.0% | — | expected in D=64 (curse of dimensionality → all random pairs orthogonal) |

---

## Analysis

### Per-pair latency (167 ns) vs G4-wedge (201 ns)

The per-pair latency (167 ns) is slightly *better* than the isolated G4-wedge
measurement (201 ns). This is because the crowd bench accesses directions
sequentially per NPC (all 20 partners of NPC N are scored before moving to
NPC N+1), so NPC N's `h_self` direction stays in L1 cache across all 20
partner evaluations. The G4-wedge bench had both u and v in L1, but measured
a single isolated call with function-call overhead amortized over fewer
iterations.

### Hit rate 100% — expected in high dimensions

In D=64, random Gaussian unit vectors are nearly orthogonal (the expected
inner product is `O(1/sqrt(D)) ≈ 0.125`). The wedge L1 norm of an orthogonal
pair is O(1) (not O(D)), so `sigmoid(1.0 × O(1)) ≈ 0.73 > tau = 0.6`. With
100% of random pairs passing the gate, **all** 20,000 pairs emit Sociability
curiosity targets. This is correct behavior in high dimensions — the bridge
correctly identifies that randomly-different NPCs are complementary.

In production, real HLA directions are NOT uniformly random — they cluster
around personality archetypes (valence/arousal/desperation/calm/fear
directions). Real hit rates will be lower because same-archetype NPC pairs
have lower wedge norms. The 100% rate here is an artifact of the synthetic
isotropic distribution, not a gate failure.

### Single-threaded is sufficient

The shipped `complementarity_targets_batch` is single-threaded (sequential
per-NPC loop, one reused `CliffordScratch`). At 3.34 ms/tick, it consumes
**20% of a 16.67 ms 60Hz tick** — well within budget. If a future game needs
to scale beyond 1000 NPCs × 20 partners, the workload is embarrassingly
parallel (rayon `par_iter` over NPC rows would recover near-linear speedup:
8× on 8 cores → 0.42 ms/tick). But the single-threaded baseline is already
PASS, so parallelization is not needed for the current spec.

---

## What This Unblocks

G8e PASS means the **perf budget is non-blocking** for the remaining
Super-GOAT gates:
- **G8c** (formation robustness sim) — can run at crowd scale without perf
  concerns
- **G8d** (faction diversity sim) — can run a 100-NPC sandbox within the
  perf budget
- **G5** (riir-neuron-db compaction quality) — independent of this gate

Super-GOAT elevation is now gated on G8c/G8d/G5 only.
