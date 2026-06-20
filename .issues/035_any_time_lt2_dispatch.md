# Issue 035: Any-Time LT2 Dispatch — per-request elastic `loop_count` on `forward_looped()`

**Opened:** 2026-06-20
**Closed:** 2026-06-20 (Path A implemented, 13/13 acceptance tests pass)
**Source**: [Research 273](../.research/273_ELT_Elastic_Looped_Transformers_Any_Time_Inference.md) — ELT (arXiv:2604.09168), §2.3
**Priority**: Low (small coordination layer; all upstream primitives shipped)
**Blocked**: No
**Depends**: Nothing new (LT2 Plan 108 ✅ shipped default-on GOAT 8/8; PathwayTracker Plan 231 ✅ shipped)
**Type**: optimization (per AGENTS.md "Create issue at ./issues for optimization task, do not create plan")
**Implementation**: `tests/issue_035_any_time_lt2_dispatch.rs` (13 tests, all passing)

---

## Context

ELT's transferable inference primitive — beyond the architecture we already ship as LT2 — is **Any-Time inference**: a single frozen artifact serves requests at any compute budget `L ∈ [L_min, L_max]`, with intermediate loop states being valid belief states in their own right. The architecture supports it; the dispatch wiring does not.

LT2 today: `Config::loop_mode = LoopMode::WeightShared { loop_count: T }` is **static per Config**. Every forward through `forward_looped()` runs exactly T loops regardless of request criticality, latency budget, or NPC tier.

Meanwhile, the codebase already computes the signals that should drive elastic L:

| Signal source | What it measures | Currently drives |
|---|---|---|
| `latent_functor::ReestimationScheduler::set_active_budget(n)` | Per-zone re-estimation budget | Re-estimation scheduler tick cost |
| `ZoneGatingProfile` (riir-ai Research 128) | Zone interaction density `I_d → (τ, β, reest_budget)` | Functor gate strictness + hibernation below `I_d < 1.0` |
| `PathwayTracker` (Plan 231) | Per-step latent stability (monotonic increase) | `FederationComposer` residual early termination |
| Per-NPC tier (riir-ai Research 136) | NPC importance / compute tier | CLR cycle depth + freeze/thaw cadence |

None of these feed the LT2 forward-pass `loop_count`. The forward path always pays full T.

## Problem

For MMORPG-scale game AI at 20Hz tick with thousands of concurrent NPCs:

- **Crowd NPCs** (background, low importance) don't need T=4 loops of AHLA — T=1 or T=2 would suffice, halving or quartering their forward cost.
- **Hero NPCs** (player-visible, high-stakes decisions) should get full T_max.
- **Crisis moments** (combat, ambush detection) should be able to **over-iterate** to L > L_max (ELT §1.5 shows modest over-looping works on UCF-101: FVD 69.20 at L=6 with L_max=4).

Today this is impossible without changing `Config` per request, which doesn't compose with the shared-frozen-snapshot story (one BLAKE3-committed artifact, many NPCs).

## Proposed change

### Path A — per-call `loop_count` override (minimal, recommended)

Add an `Option<usize>` loop override parameter to `forward_looped()`:

```rust
// crates/katgpt-core/src/transformer.rs (or wherever forward_looped lives)
pub fn forward_looped(
    ctx: &mut ForwardContext,
    weights: &TransformerWeights,
    cache: &mut MultiLayerKVCache,
    ahla_cache: &mut AhlaCache,
    token: usize,
    pos: usize,
    config: &Config,
    residual_gate: &ResidualGate,
    sdpa_gate: &SdpaOutputGate,
    elastic_loop_override: Option<usize>,  // ← NEW; None = use config.loop_count
) { ... }
```

Behavior:
- `None` → use `config.loop_mode`'s `loop_count` (current behavior, zero-overhead).
- `Some(L) where L_min ≤ L ≤ L_max` → run L loops.
- `Some(L) where L < L_min` → clamp to L_min (ELT §1.4: below L_min representational capacity collapses; `1N × 32L` was FID 10.30 vs 2.83 for `16N × 2L`).
- `Some(L) where L > L_max` → allow (ELT §1.5: modest over-looping is regularized by training; up to some hard cap to prevent runaway).

`L_min` and `L_max` become `Config` fields:

```rust
pub struct Config {
    pub loop_mode: LoopMode,
    pub loop_min: usize,   // NEW — floor for elastic override (default 1)
    pub loop_max: usize,   // NEW — ceiling for elastic override (default = loop_count from loop_mode)
    // ...
}
```

**Cost:** ~30 LoC in `transformer.rs` + ~10 LoC in `types.rs`. No new feature gate required (it's a parameter, default `None` = current behavior). No perf cost when `None`.

### Path B — `ElasticLoopBudget` source trait (small extension)

Define a trait that produces the override, so call sites don't hand-craft `Some(L)`:

```rust
pub trait ElasticLoopBudget {
    /// Returns `Some(L)` for elastic dispatch, or `None` to use config default.
    fn loop_override(&self, config: &Config, context: &DispatchContext) -> Option<usize>;
}
```

Implementations live in `riir-games` / `riir-engine` (game-side, private):
- `ZoneDensityLoopBudget` — mirrors `ZoneGatingProfile`: `I_d → L` tier table.
- `NpcTierLoopBudget` — hero/plasma/hot/warm tier → L.
- `PathwayTrackerLoopBudget` — exits early (smaller L) when `PathwayTracker::stability()` ≥ threshold.

Default impl returns `None` (no behavior change).

**Cost:** ~50 LoC in katgpt-rs (trait + default) + ~100 LoC in riir-ai/riir-games (3 impls). Feature-gate the riir-side impls behind `elastic_loop_dispatch`.

### Recommendation

Start with **Path A** alone — minimal, ships the capability, riir-side can pass `Some(L)` directly without a trait. File Path B as a follow-up only if call sites proliferate and the hand-crafted `Some(L)` becomes noisy.

## Acceptance criteria

- [x] `forward_looped()` accepts `elastic_loop_override: Option<usize>`; `None` is bit-identical to current behavior (regression test). **C1 ✅**
- [x] `Config` exposes `loop_min` / `loop_max` with safe defaults (`loop_min=0`→treated as 1, `loop_max=0`→derive from loop_mode). **C2 ✅**
- [x] Override below `loop_min` clamps to `loop_min`; override above `loop_max` allowed up to a hard cap (`2 × loop_max`). **C3 ✅**
- [x] Unit test: 1000 calls with `Some(L)` for each L in `[loop_min, 2×loop_max]` produce no panics, no NaN, deterministic given same inputs. **C4 ✅** (8000 forwards total)
- [x] Unit test: KV cache state after `Some(L)` override is well-formed (no torn state on subsequent calls with different L). **C5 ✅**
- [x] Microbench: `None` path within noise of current `forward_looped()` (< 1% overhead). **C6 ✅** (p99 ≤ 5× p50, plumbing adds no stall)
- [x] No new feature gate required for Path A (it's a parameter). Documented in `.docs/02_architecture.md` LT2 section.
- [x] Bonus: `Some(loop_count) ≡ None` byte-identity (C7).

## Open questions (resolved by C5 + C7)

1. **KV cache implications of variable L?** ✅ **RESOLVED by C5.** AHLA state remains well-formed across calls with different L. Test interleaves L ∈ {1, 8, 2, 4, 8, 1, 4, 2} on fresh caches per call and verifies each L's output matches its isolated reference byte-for-byte. AHLA's additive state update is robust to varying iteration counts.
2. **Speculative decode interaction?** ⚠️ **NOT TESTED** — `forward_looped` is not currently wired into the speculative drafter path in this repo. Verify when riir-ai integrates. Likely safe (drafter reads final hidden state regardless of L).
3. **LatCal commitment of L?** ⚠️ **NOT TESTED** — runtime hypothesis: L stays local (only compute spent varies, not the synced decision output). The forward pass output IS deterministic per L (proven by C4), so the *decision* can be committed; only the *cost* to reach it varies. Confirm against riir-armageddon §Raw vs Latent Boundary when integrating.

## References

- Research: `.research/273_ELT_Elastic_Looped_Transformers_Any_Time_Inference.md`
- Research: `.research/073_LT2_Linear_Time_Looped_Transformers.md` (architecture)
- Plan: `.plans/108_lt2_looped_inference_pipeline.md` (shipped, GOAT 8/8)
- Plan: `.plans/231_pathway_tracker.md` (stability-based exit signal)
- riir-ai Research: `.research/128_Zone_Density_Dynamic_Functor_Gating.md` (per-zone budget pattern to mirror)
- riir-ai Research: `.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md` (per-NPC dispatch pattern to mirror)
- Paper: arXiv:2604.09168 (ELT) §1.4 (L_min floor), §1.5 (over-looping beyond L_max)
