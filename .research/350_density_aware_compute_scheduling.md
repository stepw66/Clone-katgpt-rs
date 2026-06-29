# Research 350: Density-Aware Compute Scheduling — Continuum-Crowds Mobility × Fokker-Planck-on-Cochains

> **Source:**
> - Treuille, Cooper, Popović (2006) — *"Continuum Crowds"* — ACM Transactions on Graphics (SIGGRAPH 2006). [ubc.cs page](https://www.cs.ubc.ca/~van/papers/2006-continuum-crowds/continuum-crowds-siggraph-2006.pdf)
> - van Toll, Pettré et al. — *Density-aware navigation meshes* (line of work, multiple papers)
> - Fokker-Planck / continuity equation on cochains — internal mapping via [Plan 314](../.plans/314_stokes_calculus_wrappers.md); also Research [296](296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md) §3.
> **Date:** 2026-06-29
> **Status:** Active
> **Related Research:** 128 (zone-density cognitive gating), 219 (DEC substrate), 246 (manifold power-iteration MoE router), 296 (Stokes vocabulary crosswalk)
> **Related Plans:** 251 (DEC operators), 305 (ZoneGatingProfile cognitive tier), 314 (Stokes wrappers, G-A FAILED as ICT detector), 334 (Stokes→HLA wiring), 242 (Fourier crowd flow — already cites Treuille 2006, **different application**), 001-Phase-1 zone manifold (just shipped)
> **Cross-ref (riir-ai):** [Plan 305](../../riir-ai/.plans/305_*.md) (existing density-tiered cognitive gating), [Plan 334](../../riir-ai/.plans/334_stokes_calculus_hla_wiring.md) (DEC→HLA wiring), [Plan 242](../.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md) (Fourier crowd flow, different Treuille application)
> **Classification:** Public (katgpt-rs)

---

## TL;DR

**The distilled primitive** is a modelless classifier that turns a per-zone population count (raw, synced scalar) into a per-zone `(mobility_weight, tier_class, cache_key)` triple. The mobility weight is `sigmoid(-β·(ρ−ρ₀))` — dense cores collapse to ~0 mobility, sparse peripheries saturate to ~1. The tier class (`Sparse` / `Transitional` / `Dense`) drives compute scheduling: sparse zones get full per-tick recompute (high movement freedom → high event entropy), dense zones get LRU-cached projections (NPCs physically constrained, near-zero information change tick-to-tick). A `papaya` lock-free cache keyed on `(tier, density_bucket)` provides O(1) hit/miss with explicit invalidation rules for stampede events.

**Why it matters here:** this is the missing third layer that fuses the just-shipped zone manifold (Plan 001, crowd-scale PCA) with the existing cognitive gating (`ZoneGatingProfile`, Plan 305) and the DEC mass-conservation validator (`belief_mass_divergence`, Plan 314). The current codebase has cognitive economics per zone (how hard to *think*) but not physical compute economics per zone (how hard to *move*). The user's instinct — "outer ring first, dense core cached, bell-shape over density" — is exactly Treuille et al. (2006)'s continuum-crowds speed field, recast as a scheduling weight instead of a movement cost.

**Distilled for katgpt-rs (modelless, inference-time):**
- `zone_density_classify(population, config, out_mobility, out_tier, out_cache_key)` — pure function, deterministic, zero-alloc hot path (caller-owned output slices).
- `schedule_outer_first(population, out_order, scratch)` — stable sort of zone indices ascending by density. O(Z log Z), trivially cheap for Z < 1024.
- `ZoneDensityCache<V>` — `papaya::HashMap`-backed LRU with three invalidation rules (tier transition, density drift > δ, TTL expiry).
- All three compose as a **sibling layer** to `ZoneGatingProfile` (Plan 305), not a replacement. Cognitive gating and physical scheduling are orthogonal axes.

---

## 1. Paper Core Findings

### 1.1 Treuille, Cooper, Popović (2006) — "Continuum Crowds"

The canonical reference. The paper's central observation: **a dense crowd behaves as a continuum fluid**, where each location has (a) a density `ρ(x)` (people per unit area), (b) a velocity field `v(x)` (average direction + speed), and (c) a **density-dependent speed constraint** `‖v(x)‖ ≤ v_max · g(ρ(x))` where `g` is a monotonically decreasing function (typically `g(ρ) = 1 − ρ/ρ_max` for `ρ < ρ_max`, else 0). The crowd self-organizes because individuals choose paths that minimize a cost functional integrating speed, length, and discomfort.

**What's transferable (the distilled primitive):**

1. **Density-dependent mobility** — `g(ρ)` is a deterministic, closed-form function. No training. Treuille uses linear `1 − ρ/ρ_max`; we'll use **`sigmoid(-β·(ρ−ρ₀))`** for two reasons: (a) smoother derivative at the transition (better for stable tier classification), (b) AGENTS.md mandates sigmoid not softmax for any latent gate.
2. **Continuum vs individual** — the paper's win is computing one field per zone instead of per individual. Our analog: one `(mobility, tier, cache_key)` per zone instead of per NPC. The just-shipped cluster-and-distribute PCA (Plan 001) already partitions N NPCs into G groups; this primitive adds a per-group density classification.
3. **Path cost = ∫ discomfort dx** — the paper computes geodesic cost as line integral of a per-cell discomfort field. This maps directly to `line_integral` (Plan 314 T1.4, shipped) on a rank-1 cochain. **Fusion hook with Plan 314**.

**What's NOT transferable (training-bound, → riir-train):**

- The paper's "least-cost path" optimization is solved via fast-marching each tick. We don't need that — game NPCs have their own pathfinder. We only borrow the density→mobility curve and the crowd-as-continuum viewpoint.
- The paper's discomfort field is hand-tuned per scene. We do NOT learn discomfort; we derive mobility deterministically from raw population count.

### 1.2 van Toll, Pettré et al. — Density-aware navigation meshes

A line of work (multiple papers 2011–2018) extending navmeshes with explicit density metadata per polygon. Key relevant idea: **per-polygon density buckets** drive pathfinding cost. A sparse polygon is "cheap to traverse", a dense polygon is "expensive" (slow movement, collision risk). This is the LRU analog: dense polygons get cached pathfinding results because their internal structure rarely changes; sparse polygons get fresh pathfinding because their next-state space is large.

**Distilled:** the `(tier, density_bucket)` cache key structure mirrors van Toll's polygon density annotation. The novelty is using it for **compute scheduling**, not pathfinding cost.

### 1.3 Fokker-Planck / Continuity equation on cochains

The mathematical substrate. For a population density `ρ(x, t)` evolving under a velocity field `v(x, t)`, mass conservation reads:

```
∂ρ/∂t + ∇·(ρ·v) = 0      (continuity / Fokker-Planck with zero diffusion)
```

Discretized on a `CellComplex` (Plan 251), this becomes a per-cell update: the change in cell `c`'s density equals the negative net flux across `c`'s boundary. The flux is the rank-1 cochain `ρ·v`, and the boundary operator is `exterior_derivative` (or equivalently `codifferential` for the divergence form). Plan 314 already ships `belief_mass_divergence()` as the L1-norm validator of this identity on belief cochains.

**Two uses in this primitive:**

1. **Mass-conservation self-check** — at each tick, `belief_mass_divergence(ρ·v_flow)` should be ≈ 0. A spike means NPCs are appearing/vanishing (sync bug, anti-cheat violation, or stampede). This is the **stampede detector** that drives LRU invalidation: if mass divergence exceeds τ, all dense-tier caches are invalidated immediately.
2. **Mobility as inverse density** — Treuille's `g(ρ)` is the closed-form solution to "how fast can an individual move given local density". Reading `ρ` as a 0-form on the zone cochain and `mobility` as its pointwise inverse-sigmoid gives the same physics, modellessly.

---

## 2. Distillation

### 2.1 Transferable primitive (modelless)

```text
input:  population[0..Z]  (raw, synced, BLAKE3-committed in ZoneExpertBundle)
config: rho0=5.0, beta=0.5, tier_high=0.7, tier_low=0.3,
        cache_invalidation_delta=2.0, ttl_ticks_dense=64

per zone z:
  m[z]    = fast_sigmoid(-beta * (population[z] - rho0))        # latent, NOT synced
  tier[z] = match m[z] {
              mh if mh > tier_high => Sparse,                    # full recompute
              ml if ml > tier_low  => Transitional,              # short-TTL cache
              _                    => Dense,                     # long-TTL cache
            }
  bucket[z] = (population[z] * 0.5).floor() as u64              # buckets of size 2.0
  cache_key[z] = (tier[z] as u64) << 32 | bucket[z]             # composite, instant
```

Three properties:
- **Deterministic**: same input + same config → bit-identical output (no RNG, IEEE-754 fixed op order).
- **Zero-alloc hot path**: caller owns all output slices; the function is single-pass.
- **Latent outputs**: `mobility`, `tier`, `cache_key` are NOT synced — they're derived locally per zone from the raw synced `population`. They never cross the chain boundary (per AGENTS.md sync rule).

### 2.2 Why modelless (per skill §3.5)

The user's proposal could superficially look "trainable" — one might imagine learning the `β, ρ₀` parameters from crowd data. The §3.5 modelless-unblock check says: don't.

- **Path 1 (freeze/thaw)?** — N/A. No weights involved.
- **Path 2 (deterministic LoRA)?** — N/A. No matrix multiplication.
- **Path 3 (latent projection)?** — **YES, and that's all that's needed.** `sigmoid(-β·(ρ−ρ₀))` is already a latent projection (raw scalar → latent gate). The parameters `(β, ρ₀)` are **deterministically derivable** from Plan 305's existing tier thresholds (transitional=1.0, dense=10.0 → midpoint ρ₀=5.0, slope β≈0.5 puts the 0.1→0.9 transition across one tier step). No gradient descent needed; closed-form.

**No riir-train deferral.** This is pure modelless inference.

### 2.3 Fusion (the Super-GOAT attempt)

This primitive alone is GOAT-tier (a perf optimization with a quality side-effect). The fusion angle is what could push it toward Super-GOAT:

| Fusion ingredient | What it ships | What it adds to this primitive |
|---|---|---|
| **Plan 001 (zone manifold)** | Crowd-scale PCA, cluster-and-distribute mood axes per group | Each cluster gets a density classification → per-cluster scheduling, not just per-zone |
| **Plan 305 (ZoneGatingProfile)** | Cognitive-tiered `(tau, beta, reest_budget)` per zone | Orthogonal axis: cognitive economics. Composes as sibling, not conflict. |
| **Plan 314 (belief_mass_divergence)** | L1 mass-conservation validator on belief cochains | Stampede detector — invalidates dense-tier cache when mass-divergence spikes |
| **Plan 242 (Fourier crowd flow)** | FFT-smoothed shared flow field per zone (already cites Treuille 2006) | Provides the velocity field `v(x)` for the Fokker-Planck check |

**The fusion primitive**: a 60s sim where (a) each NPC's per-tick compute budget is gated by its zone's `(mobility, tier, cache_key)`, (b) the cluster-and-distribute PCA runs per cluster with clusters scheduled outer-first, (c) the LRU cache is invalidated by mass-divergence spikes, (d) the resulting event stream has measurably higher Shannon entropy than mean-aggregation baseline. This is what Plan 351 benchmarks.

**Super-GOAT-conditional emergence**: if the combined primitive produces **lane formation, queue emergence, or stampede triggers** that no mean-aggregation baseline can reproduce, that's a new capability class (no competitor does density-aware emergent crowd behavior at MMORPG scale with modelless compute). That would touch ≥2 pillars (Fourier Spatial + Reasoning Pack) and become a moat.

### 2.4 Latent vs raw boundary (per AGENTS.md)

| Quantity | Domain | Tier | Synced? | Notes |
|---|---|---|---|---|
| `population[z]` (NPC count per zone) | Physical | Raw | YES | Countable, deterministic, BLAKE3-committed in ZoneExpertBundle |
| `mobility[z]` (sigmoid weight) | Semantic | Latent | NO | Locally derived from raw. Equivalent to the 5 synced affect scalars' status — if needed for chain commit, project to raw via clamp. |
| `tier[z]` (Sparse/Transitional/Dense) | Semantic | Latent | NO | Locally derived. If synced, sync the underlying `population` and re-derive deterministically. |
| `cache_key[z]` (composite u64) | Semantic | Latent | NO | Locally derived. Cache itself never crosses sync boundary. |
| `belief_mass_divergence` (Fokker-Planck check) | Semantic | Latent | NO | Local validator. Spikes trigger cache invalidation only. |
| `schedule_order` (outer-first sort) | Semantic | Latent | NO | Local scheduler decision. Each NPC's actual position is still raw and synced. |

**Bridge pattern (raw → latent)**: `fast_sigmoid(-β·(ρ−ρ₀))` — dot-product-style projection onto a single direction (the "high-mobility" direction), bounded with sigmoid. Satisfies AGENTS.md's bridge rule: zero-allocation, gateable by feature flag, no sync dependency.

**Bridge pattern (latent → raw)**: not needed in this primitive — the latent outputs never need to be committed as raw. If a future chain application needs to commit "this zone is Dense", sync the underlying `population[z]` scalar and re-derive the tier deterministically on the receiver. Never reconstruct density from the latent tier.

---

## 3. Novelty gate (per skill §1.5)

### Q1: No prior art?

**Partial.** Required grep across all 5 repos + `.docs/`:

- `mobility|outer_first|density_cache|lru_cache|density_bucket` in `**/*.rs` → **0 hits** (the user's specific primitives are not shipped under any vocabulary).
- `mobility_weight|outer.first|density.cache|treuille|continuum.crowd` in `**/*.md` → **1 hit**: [Plan 242](../.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md) cites Treuille et al. 2006 for **Fourier-smoothed potential fields for LEO crowd flow** — a different application (FFT flow smoothing, not compute scheduling). The Treuille mapping is acknowledged; the user's specific contribution (mobility-sigmoid → tier → LRU + outer-first scheduler) is genuinely novel in this corpus.
- `ZoneGatingProfile` (Plan 305 / Research 128) exists as **density-tiered cognitive gating** with `min_density` thresholds and `reest_budget=0` hibernation. This is the closest cousin — but it gates *cognitive* compute (functor coherence quality), not *physical* compute (movement scheduling). Different axis, composes as sibling.
- `belief_mass_divergence` (Plan 314) ships as mass-conservation validator. Different purpose, but reused here as **stampede detector** for cache invalidation. Not prior art for the scheduling primitive itself.

**Verdict for Q1**: novel application, acknowledged prior paper mapping (Treuille already cited for different use case). **Not 100% YES.**

### Q2: New class of behavior?

**Maybe.** Density-aware *cognitive* gating exists (Plan 305). Density-aware *physical* compute scheduling does not. If the combined primitive (this × Plan 001 × Plan 314 stampede detector) produces emergent crowd behavior (lanes, queues, stampedes from local density gradients) that mean-aggregation cannot reproduce, then yes. Without that empirical result, this is "better numbers on the same capability" (GOAT, not Super-GOAT).

**Verdict for Q2**: TBD empirically via G5 gate. Not committing YES.

### Q3: Product selling point?

**Maybe.** Candidate sentence: "Our NPCs schedule compute by physical density — dense cores are LRU-cached (near-zero per-tick cost), sparse peripheries get full compute (where movement actually happens). Result: 10× crowd scale at fixed tick budget, with emergent lane/queue/stampede behavior free."

Cannot finish the sentence with confidence until G5b (compute saved) and G5c (stampede invalidation correctness) run.

**Verdict for Q3**: TBD empirically. Not committing YES.

### Q4: Force multiplier?

**Yes.** Touches Plan 001 (manifold), Plan 305 (cognitive gating), Plan 314 (DEC mass), Plan 242 (Fourier flow). Four-plan fusion.

### Novelty gate result

**Cannot commit all 4 YES → verdict is GOAT, not Super-GOAT.** Per skill §1.5, no private riir-ai guide is created in this session. If G5a+G5b+G5c all pass AND emergent crowd behavior is observed in the 60s sim, a follow-up Super-GOAT-promotion note + riir-ai guide can be filed at that point.

---

## 4. Verdict

| Tier | Criteria | Routing | This primitive |
|---|---|---|---|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | open primitive + private guide + plans | ❌ Not committed (Q2/Q3 TBD empirically) |
| **GOAT** | Provable gain over existing approach; not a new class | Plan + implement + feature flag + benchmark | ✅ **THIS** |
| **Gain** | Incremental improvement | Plan only, behind feature flag | — |
| **Pass** | Not relevant / training-only | One-line note | — |

**One-line reasoning:** Treuille's density→mobility curve is recast as a modelless compute scheduler with three concrete primitives (`zone_density_classify`, `schedule_outer_first`, `ZoneDensityCache`) that compose orthogonally with the existing cognitive gating (Plan 305) and the just-shipped crowd PCA (Plan 001). Provable perf gain (dense-tier cache hit rate) + measurable quality gain (event entropy on sparse zones). Promotion to default gated on G5a+G5b+G5c all passing.

**Why not Super-GOAT (honestly):** the user's instinct is sound and the literature is real, but the Super-GOAT claim would require observing *emergent* crowd behavior (lanes, stampedes) that mean-aggregation provably cannot reproduce. That's an empirical result we don't have yet. Filing as GOAT now, with an explicit Super-GOAT-promotion path if G5 reveals emergence. This is the honest call per skill §1.5 — premature Super-GOAT claims are the #1 failure mode this protocol prevents.

---

## 5. Implementation hooks (forward-references into Plan 351)

- **Primitive ships in**: `katgpt-rs/crates/katgpt-core/src/zone_density.rs` (~500 LOC + tests, behind `zone_density_routing` feature).
- **Reuses**: `katgpt_core::simd::fast_sigmoid` (already shipped, pervasively used).
- **Cache backend**: `papaya::HashMap` (per AGENTS.md "Use papaya as possible for lock-free `Arc<RwLock<HashMap>>`").
- **Hashing**: NOT used for cache keys — composite `(tier << 32 | density_bucket)` is instant and decodable. (blake3 would be overkill; AGENTS.md's "use blake3" rule applies to commitment, not bucket keys.)
- **riir-ai wiring**: filed as riir-ai plan after Phase 1–3 lands. Sibling module to `zone_gating.rs` (NOT replacement).

## TL;DR

Treuille et al. (2006)'s continuum-crowds density→speed field, recast as a modelless compute scheduler. Three primitives: `zone_density_classify` (per-zone `(mobility, tier, cache_key)`), `schedule_outer_first` (stable ascending-density sort), `ZoneDensityCache` (papaya-backed LRU with mass-divergence-driven stampede invalidation). **GOAT verdict** — provable perf gain (dense-tier cache) + measurable quality gain (sparse-zone entropy), gated on Plan 351's G5a+G5b+G5c. **Super-GOAT conditional** on observing emergent crowd behavior in the 60s sim; not committed in this note per skill §1.5 (no premature Super-GOAT claims). Composes orthogonally with Plan 305's cognitive gating (sibling layer, not replacement) and Plan 001's cluster-and-distribute PCA (per-cluster scheduling).
