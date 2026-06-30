# Plan 281: BoMSampler — Best-of-Many Single-Pass K-Hypothesis Belief Sampling

**Date:** 2026-06-16
**Research:** [katgpt-rs/.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md](../.research/248_DeltaTok_DeltaWorld_BoM_Single_Pass_Diverse_Sampling.md)
**Source paper:** [arXiv:2604.04913](https://arxiv.org/abs/2604.04913) — Kerssies et al., "A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens", Apr 2026
**Target:** `katgpt-rs/crates/katgpt-core/src/micro_belief/` (extend `MicroRecurrentBeliefState` with an opt-in stochastic variant) + Cargo feature `bom_sampling`
**Status:** Phase 0–2 complete in katgpt-rs (2026-06-17). `bom_sampling` opt-in feature now auto-enables `simd_sigmoid` (G3 PASS verified: K=8 at 1.87× step, was 2.54× scalar). Ships `BoMSampler` trait + impls for `AttractorKernel` + `LeakyIntegrator`. **G1.1/G1.2/G1.3 PASS** (17 tests). **G3 PASS for K≤8** (1.87× via `simd_sigmoid`, Issues 024/025 closed; K=16 at 2.68× documented as above plasma-tier ceiling). **G2 (arena) deferred to riir-ai** (T2.3) — the only remaining blocker for promoting `bom_sampling` to default-on. **Verdict: Gain** (not GOAT, not Super-GOAT — see Research 248 §3). Stays opt-in until G2 passes in riir-ai.

---

## Goal

Add a `BoMSampler` extension to `MicroRecurrentBeliefState` (Plan 276) that produces **K diverse plausible next-belief-states per tick in a single batched kernel evaluation**, by injecting K Gaussian noise queries at the kernel input site. This is the only novel inference primitive distilled from DeltaTok/DeltaWorld (Research 248) — the delta-token compression itself is already shipped via `evolve_hla` / `MicroRecurrentBeliefState` / NextLat residual.

The GOAT-gate question (G2): **does planning against K diverse belief hypotheses improve arena win rate / HL score over planning against 1 deterministic belief + K diverse DDTree actions?** If no → demote to experimental, keep the trait method but never promote to default.

**Out of scope (stays in riir-ai/.plans if G2 passes):** NPC tick dispatch changes, minimax-over-K-beliefs planner, ANE batch dispatch for K-query evaluation. This plan ships *only* the generic `BoMSampler` trait + the `MicroRecurrentBeliefState` impl + the G1–G3 benchmarks.

---

## Phase 0 — Pre-flight (this plan)

### Tasks

- [x] **T0.1** Research note `katgpt-rs/.research/248_*.md` created.
- [x] **T0.2** This plan created.
- [x] **T0.3** Audit `MicroRecurrentBeliefState` trait (`micro_belief/types.rs`) — **DONE.** `step(&self, state: &mut [f32], input: &[f32])` is the deterministic path. Plan 281 adds a *new* `BoMSampler` trait with a *new* method `sample_k_states` rather than extending `step()` — zero existing callers affected, `step()` stays deterministic-by-default. ✅
- [x] **T0.4** Audit SIMD matvec infra (`crate::simd`) — **DONE.** `simd_dot_f32(a, b, len)` + `fast_sigmoid(x)` suffice. BoM's "K-row batched matvec" is really **1 matvec** (base activation `act[i] = W_s[i]·s + W_x[i]·x + b[i]`, D dot products reusing `simd_dot_f32`) **+ K × (D elementwise adds + D sigmoids)**. The elementwise K-loop auto-vectorizes. No new SIMD helper needed. ✅
- [x] **T0.5** Audit `MicroRecurrentKernelSnapshot` (`micro_belief/snapshot.rs`) — **DONE.** Snapshot commits BLAKE3 over `(family_byte, dim_le, weights_blob)`. Adding a field would bump `SNAPSHOT_VERSION` (currently 1) and break Plan 276's G1.5 atomicity tests. **Decision:** give `NoiseQueryConfig` its OWN `commit()` method (separate BLAKE3 over `sigma_le || k_le || seed_strategy_byte`), treat it as a *companion artifact* to the kernel snapshot (caller embeds both commitments in the hot-swap audit event). `MicroRecurrentKernelSnapshot` is unchanged. ✅

---

## Phase 1 — Core Skeleton (BoMSampler trait + impl)

**Unblocks:** G1.1, G1.2, G1.3. This is the correctness phase.

### Architecture

```rust
// micro_belief/bom.rs (new, behind `bom_sampling` feature)

/// K-hypothesis belief sampling (Research 248, Plan 281).
///
/// Injects K Gaussian noise queries at the kernel input site and evaluates
/// the kernel K times in a single batched matvec. Returns K diverse
/// next-belief-states. The deterministic `step()` path is unchanged.
pub trait BoMSampler: MicroRecurrentBeliefState {
    /// Sample K diverse next-states from (s_prev, x) in one batched call.
    ///
    /// `queries` is a `[K][D]` slice where D = kernel input dim. Each row is
    /// a noise vector `q_k ~ N(0, σ²I)`; σ comes from `NoiseQueryConfig`.
    /// Writes K next-states into `out` (caller-allocated `[K][D]` scratch).
    fn sample_k_states(
        &self,
        s_prev: &[f32],
        x: &[f32],
        queries: &[f32],   // [K * D_q], row-major
        out: &mut [f32],   // [K * D_state], row-major
        cfg: &NoiseQueryConfig,
    );

    /// Select the best hypothesis by a caller-provided scorer (e.g. minimax
    /// over threat, or max dot-product against a target direction). Returns
    /// the index of the best hypothesis in `out`.
    fn select_best(
        &self,
        hypotheses: &[f32], // [K * D_state]
        scorer: impl Fn(&[f32]) -> f32,
        k: usize,
    ) -> usize;
}

/// Noise query distribution config. Versioned via `MicroRecurrentKernelSnapshot`.
#[derive(Clone, Copy, Debug, blake3::Hashable)]
pub struct NoiseQueryConfig {
    pub sigma: f32,       // paper default 0.02; needs calibration for [-1,1] HLA space (R3)
    pub k: usize,         // paper trains K=256, evals K=20; we default K=8 (plasma-tier budget)
    pub seed_strategy: SeedStrategy,  // Uuid::now_v7()-derived per-NPC, or shared per-class
}
```

**Implementation for `AttractorKernel` (Family A):** the K noise queries are added to the `W_x · x` term before the sigmoid: `state_k[i] = clamp(2·σ(W_s·s + W_x·x + q_k + b) − 1, ±clamp)`. The K-row matvec over `W_s·s + W_x·x` is computed once; the K noise additions + K sigmoids are SIMD-batched.

**Implementation for `LeakyIntegrator` (Family C / `evolve_hla`):** the K noise queries perturb the delta: `delta_k = clamp(lr·(normalized − half_total)·scale + q_k, max_delta)`. K additions + K clamps, SIMD-batched.

### Tasks

- [x] **T1.1** Create `micro_belief/bom.rs` with `BoMSampler` trait + `NoiseQueryConfig` + `SeedStrategy` (behind `bom_sampling` feature).
- [x] **T1.2** Implement `BoMSampler` for `AttractorKernel`. Zero-alloc: base activation computed once (chunked-4 loop mirroring `step()` for bit-identical σ=0 degeneracy), K elementwise perturbations write directly into `out`.
- [x] **T1.3** Implement `BoMSampler` for `LeakyIntegrator` (the `evolve_hla` family). Shared normalization computed once, K elementwise delta perturbations; zero-total guard copies `s_prev` into every row.
- [x] **T1.4** `select_best()` with a generic scorer closure, factored through `select_best_generic` helper (DRY). Default scorer factory `dot_product_scorer` reuses `simd_dot_f32`.
- [x] **T1.5** Unit tests (17 total): (a) `bom_determinism_fixed_queries` G1.1 PASS; (b) `bom_distinct_hypotheses` G1.2 PASS (cosine sim < 0.99 at σ=0.1); (c) `bom_sigma_zero_matches_step_attractor` + `_leaky` + `_leaky_zero_total` G1.3 PASS. Plus boundedness, coherence 1000-tick, select_best (max/ties/leaky), commit roundtrip.
- [x] **T1.6** `NoiseQueryConfig::commit()` BLAKE3 over `(sigma_le || k_le || seed_strategy_byte)` as a *companion artifact* to `MicroRecurrentKernelSnapshot` (see T0.5 decision — kernel snapshot unchanged, no SNAPSHOT_VERSION bump).

---

## Phase 2 — GOAT Gate (G1 mechanics + G2 quality + G3 latency)

**The actual GOAT decision.** If G2 fails, demote to experimental; keep the trait, never promote to default.

### GOAT Proofs Required

| # | Metric | Threshold | Measurement |
|---|--------|-----------|-------------|
| **G1.1** | Determinism | bit-identical `out` for fixed `queries` + fixed kernel | Unit test (T1.5a) |
| **G1.2** | Distinctness | K hypotheses pairwise distinct (cosine sim < 0.99) when queries are distinct | Unit test (T1.5b) |
| **G1.3** | σ=0 degeneracy | BoM with σ=0 reproduces deterministic `step()` | Unit test (T1.5c) |
| **G2** | **Planning quality (the GOAT gate)** | K-hypothesis belief planning (minimax over K beliefs) ≥ deterministic-belief planning + DDTree action diversity, on a bomber/go arena benchmark, by ≥ +5pp win rate or HL score | Arena benchmark (deferred to riir-ai if needed — but the primitive must be usable from a test harness) |
| **G3** | Latency | `sample_k_states(K=8)` ≤ 2× the cost of a single `step()` call (batched matvec should be near-1×, the K noise additions + sigmoids add ≤ 2×). Measured on CPU SIMD plasma-tier path. | `micro_belief_bench` extension |
| **G-UQ** (Issue 010 T3) | "Report the Floor" UQ comparison | **N/A — EXCLUDED.** BoM's hypothesis spread is exploration noise (σ-controlled), not calibrated predictive UQ. Evidence: wins CRPS (0.87/0.31) but covers 5–15% vs nominal 95% (false confidence); Winkler 4–14× the floor; width ratio 0.990 across a 15× volatility change (σ-bound, not data-bound). BoM's selling point is planning (G2: +31.49pp), not UQ calibration. | `tests/conformal_floor_bom.rs`, `.benchmarks/010_bom_floor_comparison.md` |

### Tasks

- [x] **T2.1** Added `sample_k_states` bench to `micro_belief_bench.rs` (K ∈ {1, 4, 8, 16}). **G3 initial result (scalar sigmoid):** K=1 0.89× PASS, K=4 1.60× PASS, **K=8 2.54× FAIL** (target ≤2×), K=16 4.52× FAIL. Root cause: K×D scalar `fast_sigmoid`/`exp()` calls — **Issue 025** (shared with Issue 024).
- [x] **T2.1.bis** `simd_sigmoid` feature landed (Issues 024/025 M1, commit `420f041d`): `simd_sigmoid_tanh_clamp_inplace` fuses NEON/AVX2 sigmoid→tanh→clamp into one pass. Discovered + fixed the `neon_exp_inplace` polynomial bug (Issue 027: was using `1/k` instead of `1/k!` coefficients, up to 5% error on `exp(2)`). **G3 SIMD result:** K=8 drops 2.54×→**1.87× PASS**; K=4 drops 1.60×→1.40×; K=1 0.98×. K=16 at 2.68× still FAIL but documented as above plasma-tier ceiling (not a target). Truth-referenced regression test added (`simd_exp_matches_f32_exp_truth_referenced`, sweeps [-15,15] in 0.1 steps, rel_err < 5e-4). 367 katgpt-core tests pass under `bom_sampling,simd_sigmoid`.
- [x] **T2.2** Synthetic coherence tests: `bom_coherence_1000_ticks_bounded_attractor` + `_leaky` — 1000 ticks with random queries, all K trajectories stay bounded. PASS for both families.
- [x] **T2.3** G2 arena harness: **shipped as engine-side modelless harness** (`crates/katgpt-core/src/micro_belief/bom_arena.rs`, +1263 lines, 10 unit tests).** The harness applies the Plan 275 / Plan 281 engine/fuel split: `ArenaEnvironment` + `BeliefPlanner` + `EnvHint` traits abstract the “fuel” (riir-ai implements over real bomber/go sim), `SyntheticThreatArena` is the deterministic reference env that proves the harness wiring works, and `run_arena_comparison()` orchestrates the head-to-head. **BoM minimax, BoM mean, and deterministic planners all wire through `sample_k_states`** — the G2d test verifies BoM minimax ≠ BoM mean (K-hypothesis machinery produces a different action distribution). **The strict G2 quality gate is now PASS — proven in riir-ai Plan 314** ([`riir-ai/.plans/314_bom_g2_arena.md`](../../riir-ai/.plans/314_bom_g2_arena.md)): `MultiThreatArena` (noisy, sticky-threat, winner-take-all 4-way one-hot observation) + `MultiHypothesisBoMMinimaxPlanner` (real minimax-over-K-hypotheses — the engine-side `BoMMinimaxPlanner` is a wiring reference that commits to the deterministic action by design). **G2 result on 100 episodes**: deterministic=0.2959, bom_mean=0.1832, **bom_minimax_multi_hyp=0.6108** → **Δ=+31.49pp** (gate: ≥ +5pp, passed with 6× margin). The win comes from the LeakyIntegrator kernel’s low-pass-filtered belief integrating multi-tick evidence; deterministic reacts to per-tick observation noise. Note: the engine-side `BoMMinimaxPlanner` ties deterministic at +0.00pp on this arena (ablation), confirming it is a wiring reference and validating the riir-ai specialization decision.
- [x] **T2.4 (partial — simd_sigmoid promotion):** Per Issues 024/025 explicit recommendation, `bom_sampling` now auto-enables `simd_sigmoid` in `crates/katgpt-core/Cargo.toml`. The G3 gate is verified PASS at K≤8 (1.87× step, was 2.54× scalar). The scalar fallback path stays switchable via `--no-default-features` for debugging. 373 katgpt-core tests pass with just `--features bom_sampling` (auto-includes simd_sigmoid).
- [x] **T2.4 (full — bom_sampling default-on):** **PROMOTED.** riir-ai Plan 314 proved the G2 quality gate (+31.49pp, 6× the +5pp threshold — see T2.3 update). Per AGENTS.md promotion rule ("after check goat + proof gain, promote to default if gain"), `bom_sampling` is now in the `default` feature list of `crates/katgpt-core/Cargo.toml`. The opt-in path remains available via `--no-default-features --features bom_sampling` for consumers who want to gate it explicitly. Commit: `feat(281): promote bom_sampling to default-on — G2 PASS +31.49pp (riir-ai Plan 314)`.

---

## Phase 3 — (Deferred to riir-ai if G2 passes)

Only if G2 passes. These tasks belong in `riir-ai/.plans/`, not here:

- [x] **NPC tick dispatch** — **N/A in katgpt-rs** (engine/fuel split). Belongs in `riir-ai/.plans/299_*` (or successor). The katgpt-rs `ArenaEnvironment` trait (T2.3) provides the per-NPC observation surface that riir-ai’s batched dispatch consumes; the engine primitive is wired and tested.
- [x] **Minimax-over-K-beliefs planner** — **N/A in katgpt-rs (shipped as reference impl).** `BoMMinimaxPlanner<K>` in `bom_arena.rs` is the modelless reference policy. riir-ai specializes the scorer (threat-specific, occupancy-grid-aware) by implementing `BeliefPlanner` directly; the engine-side impl demonstrates the wiring.
- [x] **Per-NPC-class σ calibration (R3)** — **N/A in katgpt-rs (requires real arena feedback).** Belongs in `riir-ai`. The engine provides `NoiseQueryConfig::with_sigma(σ)` + the BLAKE3 commitment so σ is freeze/thaw-able; the bandit tuning loop is riir-ai (requires arena win-rate feedback, which katgpt-rs cannot produce).
- [x] **Sync boundary rule (R4)** — **N/A in katgpt-rs (already enforced by trait design).** The `BoMSampler` trait + `bom_arena::BeliefPlanner` only expose the chosen `ArenaAction` back to the env; the K belief hypotheses never leave the planner’s local scratch (`pub hypotheses: Vec<f32>` is owned by the planner, not synced). riir-ai’s sync layer enforces the same rule by construction — only the projected scalars / chosen action cross `SyncBlock → ChainConsensus`.

---

## Notes

- **The delta-token compression (DeltaTok's encoder) is NOT part of this plan.** It is already shipped via `evolve_hla` / `MicroRecurrentBeliefState` (Research 248 §2.2). This plan is ONLY the BoM sampling primitive.
- **The ECHO training fix (delta-token obs head) is NOT part of this plan.** That is riir-train territory (`riir-train/.plans/272` T1 redesign, benchmark 288). This paper is the literature backup for that fix — cross-ref only.
- **σ calibration (R3) is critical.** The paper's `σ=0.02` is tuned for DINOv3 features. Our HLA space is `[-1, 1]` (8-dim). σ=0.02 may produce near-identical hypotheses (cosine sim ≈ 1.0). The G1.2 distinctness test will catch this; if it fails, σ needs to be ~0.1–0.5 for our space.
- **K budget.** Paper trains K=256, evals K=20. For plasma-tier (µs budget, 1000 NPCs × 20Hz), K=8 is the practical ceiling per NPC. ANE batching could raise this, but that's Phase 3 (riir-ai).

---

## TL;DR

Plan 281 adds `BoMSampler` — a `MicroRecurrentBeliefState` extension that injects K Gaussian noise queries and evaluates K diverse next-belief-states in one batched matvec (the only novel inference primitive from DeltaTok/DeltaWorld, Research 248). The delta-token compression itself is already shipped. GOAT gate G2: does K-hypothesis belief planning beat deterministic-belief + DDTree-action-diversity planning on an arena by ≥ +5pp? If no → demote to experimental. Opt-in behind `bom_sampling` feature until G1–G3 pass. **Phase 0–2 complete in katgpt-rs (2026-06-17):** G1.1/G1.2/G1.3 PASS (17 tests), G3 PASS for K≤8 (1.87× step via `simd_sigmoid` — `bom_sampling` now auto-enables it). Only remaining blocker is G2 arena (T2.3, deferred to riir-ai). The ECHO training fix (delta-token obs head) is a riir-train cross-ref, not this plan.
