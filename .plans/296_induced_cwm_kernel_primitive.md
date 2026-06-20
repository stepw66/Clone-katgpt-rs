# Plan 296: Induced CWM Kernel Primitive — Open Traits + ISMCTS + Tournament + Commitment

**Date:** 2026-06-20
**Research:** [katgpt-rs/.research/275_Code_World_Model_Induced_Forward_Model.md](../.research/275_Code_World_Model_Induced_Forward_Model.md)
**Source paper:** [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542) — Lehrach et al., Code World Models for General Game Playing (DeepMind, Oct 2025)
**Target:** `katgpt-rs/crates/katgpt-core/src/induced_cwm/` (new module, open) + re-export through `katgpt-rs/src/lib.rs`
**Cargo features:** `induced_cwm` (katgpt-core, **opt-in**); `induced_cwm_ismcts` (depends on `induced_cwm` + `game_state`)
**Status:** Active — Phase 1 (unblocking skeleton)

---

## Goal

Ship the **generic, IP-free half** of the CWM Super-GOAT (Research 275):
- A marker trait `InducedCwmKernel: GameState` for forward-model impls that are *verifiable, committable, hot-swappable*.
- A generic belief-sampler trait `BeliefInferenceFn<S>` with a deterministic posterior-support test harness.
- An Information-Set MCTS `ismcts_search_with_inference<S, B>` that consumes both an induced CWM and a belief fn.
- A `ValueFnTournament<S, V>` arena-play selector over `StateHeuristic` candidates.
- A `CwmCommitment` BLAKE3 commitment over canonicalized induced-kernel bytes.
- A `TransitionUnitTest<S>` generator that turns observed trajectories into pass/fail unit tests.

**What stays OUT of katgpt-rs:** LLM synthesis, prompting, refinement tree, NPC integration, game-specific code, chain bridging. Those are private → `riir-ai/.plans/326_cwm_npc_runtime_integration.md`.

**GOAT gate (per AGENTS.md):** the open primitive must pass G1–G4 from Research 275 §7 before promoting any default-on wiring. v1 ships **opt-in** (`induced_cwm` cargo feature off by default). Demote-on-fail: if G1 < 0.95 on Bomber PoC → keep opt-in, file `.issues/` follow-up.

**Hard constraints (per Research 275 §4 + AGENTS.md):**
- LLM never in hot path. The trait surface is pure Rust — whatever induces the impl is the integrator's problem.
- Raw→raw deterministic transition (paper uses JSON dicts; we use `GameState::advance()` contract verbatim).
- Belief fn outputs are latent (local) — never cross sync boundary as embeddings, only via scalar projections.
- BLAKE3 commitment must be byte-stable across re-runs (deterministic canonicalization).
- Use `Uuid::now_v7()` for snapshot IDs (AGENTS.md). Use `blake3` for hashes (AGENTS.md). Use `papaya` if any lock-free map is needed (AGENTS.md).

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/induced_cwm/mod.rs` with module-level docs that mirror Research 275 §2.1: this is the open half of the CWM primitive; the LLM-induction pipeline lives in riir-ai. Re-export from `katgpt-core/src/lib.rs` gated by `induced_cwm` feature.
- [ ] **T1.2** Add `induced_cwm = []` and `induced_cwm_ismcts = ["induced_cwm", "game_state"]` to `katgpt-core/Cargo.toml` `[features]`. Add `blake3` (already a dep) — no new external crates.
- [ ] **T1.3** Define `pub trait InducedCwmKernel: GameState` in `induced_cwm/kernel.rs`:
  ```rust
  /// Marker trait for forward-model impls that are:
  /// (a) verifiable — pass auto-generated transition unit tests,
  /// (b) committable — BLAKE3 over canonicalized representation,
  /// (c) snapshot-able — atomic Arc swap via the existing freeze/thaw machinery.
  ///
  /// The trait surface is empty (marker) — implementors only need to provide
  /// `fn canonical_bytes(&self) -> Vec<u8>` for commitment, plus the existing
  /// `GameState` impl. The LLM that induced the impl is the integrator's concern.
  pub trait InducedCwmKernel: GameState {
      /// Canonical byte representation for BLAKE3 commitment.
      /// MUST be deterministic: same logical kernel → same bytes across runs.
      /// Recommended: serialize state schema + action enum + canonicalized rule
      /// source/bytecode. Integrators decide the canonical form.
      fn canonical_bytes(&self) -> Vec<u8>;

      /// BLAKE3 hash over `canonical_bytes`. Convenience wrapper.
      fn commitment(&self) -> [u8; 32] {
          blake3::hash(&self.canonical_bytes()).into()
      }
  }
  ```
- [ ] **T1.4** Define `pub struct CwmCommitment { pub blake3: [u8; 32], pub snapshot_id: Uuid, pub created_at_tick: u64 }` in `induced_cwm/commitment.rs`. Use `Uuid::now_v7()` (AGENTS.md). Add `Eq`/`Hash`/`Debug`/`Clone`. Document the rule: two kernels with identical `canonical_bytes` MUST produce identical `blake3`; `snapshot_id` distinguishes induction events.
- [ ] **T1.5** Define `pub trait BeliefInferenceFn<S: GameState>` in `induced_cwm/belief.rs`:
  ```rust
  /// Stochastic belief-state sampler for imperfect-information games.
  ///
  /// Paper §4.2: `sample(obs_history, action_history) -> Vec<HiddenStateSample>`
  /// where each sample is a plausible hidden state from the agent's posterior
  /// `p_M(s_t | o_{1:t}, a_{1:t})`.
  ///
  /// Posterior-support guarantee (paper §4.2, hidden-history variant): if the
  /// offline unit tests all pass, then each emitted sample is a valid CWM
  /// state AND reproduces the observed observation sequence. The guarantee is
  /// about SUPPORT, not DISTRIBUTION — that is sufficient for ISMCTS.
  pub trait BeliefInferenceFn<S: GameState> {
      /// Hidden-state sample — opaque to this trait; integrators define the type.
      type Sample;

      /// Draw `n` samples from the belief at the current observation horizon.
      /// Implementations SHOULD be deterministic given a seed (for unit tests).
      fn sample(&self, obs_history: &[S::Action], action_history: &[S::Action],
                player_id: u8, n: usize, seed: u64) -> Vec<Self::Sample>;
  }
  ```
- [ ] **T1.6** Define `pub struct TransitionUnitTest<S: GameState> { pub pre: S, pub action: S::Action, pub player_id: u8, pub expected_post: S, pub expected_legal_actions: Option<Vec<S::Action>> }` and `pub fn verify_transition<K: InducedCwmKernel>(kernel: &K, test: &TransitionUnitTest<K::Action>) -> Result<(), TransitionTestFailure<K::Action>>` in `induced_cwm/unit_test.rs`. Failure enum carries the diff (expected vs actual post-state and legal-action set). Mirrors paper §4.1's auto-generated transition unit tests.
- [ ] **T1.7** Add a `pub fn make_transition_tests_from_trajectory<S, I>(trajectory: I) -> Vec<TransitionUnitTest<S>>` helper that walks an observed `(state, action, next_state)` sequence and emits one `TransitionUnitTest` per step.
- [ ] **T1.8** Add `#[cfg(test)]` unit tests:
  - canonical_bytes determinism (same kernel → same blake3)
  - transition test passes for a correct kernel, fails for a wrong one (use a tiny in-test mock `GameState` impl)
  - belief fn sampler returns the requested count
- [ ] **T1.9** Re-export everything from `induced_cwm/mod.rs` and from `katgpt-core/src/lib.rs` under `#[cfg(feature = "induced_cwm")]`.

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/
├── mod.rs            # module docs + re-exports
├── kernel.rs         # InducedCwmKernel trait
├── commitment.rs     # CwmCommitment struct
├── belief.rs         # BeliefInferenceFn trait + Sample
├── unit_test.rs      # TransitionUnitTest + verify_transition + trajectory helper
└── tests/
    └── mod.rs        # in-test mock GameState impl + canonical-bytes + verify tests

katgpt-rs/crates/katgpt-core/Cargo.toml        # +induced_cwm, +induced_cwm_ismcts features
katgpt-rs/crates/katgpt-core/src/lib.rs        # gated re-export
```

---

## Phase 2 — ISMCTS Search (extends `mcts_search` to partial observability)

### Tasks

- [ ] **T2.1** In `induced_cwm/ismcts.rs` (gated by `induced_cwm_ismcts`), add `pub fn ismcts_search_with_inference<S, B>(root_obs_history: &[S::Action], root_action_history: &[S::Action], player_id: u8, kernel_sample: &S, belief: &B, heuristic: Option<&dyn StateHeuristic<S>>, budget: usize, rng_seed: u64) -> S::Action where S: GameState, B: BeliefInferenceFn<S>`:
  - At each iteration: (a) draw one hidden-state sample `s̃` from `belief`, (b) treat `s̃` as the MCTS root, (c) run standard UCB1 tree search using `kernel.advance()` (the CWM) as forward model, (d) statistics aggregate at the information-set level (per paper §B ISMCTS — Cowling et al. 2012).
  - Default to no heuristic (pure rollouts). When heuristic is provided, use it for leaf initialization (paper §4.3).
- [ ] **T2.2** Define `InformationSet<S: GameState> { pub key: u64, pub edges: HashMap<S::Action, NodeStats> }` where `key` is a hash of the observation-action history that the player cannot distinguish. Use `papaya` HashMap (AGENTS.md) if concurrent access is needed in v2; v1 uses `std::collections::HashMap` behind `&mut`.
- [ ] **T2.3** Add `NodeStats { visits: u32, total_value: f32 }` mirroring `mcts_search`'s existing node struct (DRY — reuse if exported).
- [ ] **T2.4** Unit test: a 2-card Leduc-poker-style mock where the inference fn deterministically enumerates the (small) hidden-state support; assert that ISMCTS picks a non-folding action when the agent's hand is strong.
- [ ] **T2.5** Add `examples/induced_cwm_01_mock_iig.rs` showing: a mock induced CWM (3 hidden states, 4 actions), a mock belief fn returning 1–3 samples, ISMCTS picks an action, prints the chosen action and information-set statistics.

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/
├── ismcts.rs         # ismcts_search_with_inference + InformationSet + NodeStats
└── tests/
    └── ismcts.rs     # mock Leduc-style test

katgpt-rs/examples/
└── induced_cwm_01_mock_iig.rs   # smoke example
```

---

## Phase 3 — Value Function Tournament

### Tasks

- [ ] **T3.1** In `induced_cwm/tournament.rs`, add `pub struct ValueFnTournament<S: GameState, V: StateHeuristic<S>> { candidates: Vec<V>, games_per_match: usize, rng_seed: u64 }` with `pub fn run<K: InducedCwmKernel>(&self, kernel: &K, baseline: &dyn Fn(&S, u8) -> S::Action) -> TournamentWinner<V>`:
  - Each candidate plays `games_per_match` games as player 0 vs `baseline`, then `games_per_match` as player 1, vs each other candidate (round-robin).
  - Winner = highest win-rate-vs-baseline, tie-break by head-to-head.
  - Returns `TournamentWinner { winner_idx, stats: Vec<PlayerStats> }`.
- [ ] **T3.2** Add `PlayerStats { wins: u32, losses: u32, draws: u32, avg_reward: f32 }` with `Display` impl.
- [ ] **T3.3** Reuse existing `mcts_search` for the policy (no heuristic = pure rollouts) so the tournament measures "heuristic-vs-no-heuristic" effect cleanly.
- [ ] **T3.4** Unit test: 3 mock heuristics (one near-perfect, one mediocre, one random). Assert tournament picks the near-perfect one.
- [ ] **T3.5** Example `induced_cwm_02_value_tournament.rs`: mock CWM + 3 mock heuristics → tournament prints ranking.

### Files

```
katgpt-rs/crates/katgpt-core/src/induced_cwm/tournament.rs
katgpt-rs/examples/induced_cwm_02_value_tournament.rs
```

---

## Phase 4 — Commitment Roundtrip + Hot-Swap Hook

### Tasks

- [ ] **T4.1** In `induced_cwm/hot_swap.rs`, add:
  ```rust
  /// Hot-swap slot for an induced CWM kernel, using the same atomic Arc
  /// pattern as LoRAWeightVersion. Readers never see a torn snapshot.
  pub struct InducedCwmSlot<K: InducedCwmKernel + Send + Sync> {
      inner: std::sync::Arc<std::sync::RwLock<Option<(K, CwmCommitment)>>>,
  }
  ```
  Methods: `pub fn induce(&self, kernel: K) -> CwmCommitment`, `pub fn current(&self) -> Option<(K, CwmCommitment)>` (clone out), `pub fn current_blake3(&self) -> Option<[u8; 32]>`.
- [ ] **T4.2** Document that this is the SAME pattern as `LoRAHotSwap` / `LoRAWeightVersion` (cite Plan 092). No new concurrency primitive — reuse existing.
- [ ] **T4.3** Unit test: induce kernel A → read → induce kernel B → read returns B with new snapshot_id → BLAKE3 differs.
- [ ] **T4.4** Roundtrip test: serialize `CwmCommitment` → deserialize → assert fields preserved. Use `serde` (already in deps).

---

## Phase 5 — Documentation + GOAT Gate Proof

### Tasks

- [ ] **T5.1** Update `katgpt-rs/.docs/01_overview.md` Module Structure table to list `induced_cwm/` under `katgpt-core` with the `⎗` symbol (new) and reference Plan 296.
- [ ] **T5.2** Update `katgpt-rs/.docs/21_opt_in_features.md` with a new section "Induced CWM (Plan 296)" listing the two features and pointing at the example.
- [ ] **T5.3** Update `katgpt-rs/README.md` Feature Showcase with a short "🧩 Induced CWM — LLM-Induced Forward Models (Plan 296, arxiv 2510.04542)" subsection linking to the research note.
- [ ] **T5.4** Create `katgpt-rs/.benchmarks/296_induced_cwm_primitive_goat.md` with the G1–G4 proof structure (per Research 275 §7). Run all four gates on the mock Leduc-style test fixture; record results. Promote/demote per outcome.
- [ ] **T5.5** Commit with prefix `feat(induced_cwm):` per AGENTS.md. Rebase non-interactively onto `develop` (AGENTS.md — `GIT_EDITOR=true git rebase --no-edit` if needed; default to fast-forward when safe).

---

## GOAT gate (G1–G4)

Per Research 275 §7. The open primitive provides the harness; riir-ai Plan 326 runs the same gates on game-IP fixtures. Bench file: `katgpt-rs/.benchmarks/296_induced_cwm_primitive_goat.md`.

| Gate | Target | Pass condition |
|------|--------|----------------|
| **G1 — Verifiability** | Mock CWM with 100% known-correct transitions | `verify_transition` returns Ok on all 100; returns Err with correct diff on injected mutation |
| **G2 — Play strength** | Mock 2-card Leduc-style IIG | ISMCTS picks non-fold action ≥ 70% when posterior P(strong hand) ≥ 0.6 |
| **G3 — Latency** | `apply_action` on mock CWM | ≤ 10µs/call (plasma-tier budget) |
| **G4 — Commitment integrity** | Canonicalization determinism | Same logical kernel → identical BLAKE3 across 10 re-runs |

Promotion rule: all 4 PASS → keep `induced_cwm` opt-in but mark ready for downstream consumption (riir-ai Plan 326). Any FAIL → stay opt-in, file `.issues/NNN_*` follow-up, do NOT promote.

---

## Out of scope

- **LLM synthesis pipeline.** Private → riir-ai Plan 326. The open primitive accepts any `InducedCwmKernel` impl regardless of how it was produced.
- **Game-specific CWM impls.** Bomber/Go/Monopoly CWMs (if ever induced) are private IP.
- **NPC integration, fog-of-war interaction, HLA projection.** Private → riir-ai Plan 326.
- **LatCal bridging / chain commitment.** Private → riir-ai Plan 326 (uses `CwmCommitment` from this plan as input).
- **PPO-on-CWM (paper Appendix D).** Out of scope — modelless-first; if ever wanted, that's riir-train.

---

## References

- Research note: [`.research/275_Code_World_Model_Induced_Forward_Model.md`](../.research/275_Code_World_Model_Induced_Forward_Model.md)
- Source paper: [arxiv 2510.04542](https://arxiv.org/pdf/2510.04542)
- Direct ancestor: Plan 056 (GameState forward model + generic MCTS) — `.plans/056_game_state_forward_model.md`
- Forward-model distillation: Research 027 (STRATEGA)
- Belief-inference cousin (latent side): `katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs` (`evolve_hla`)
- riir-ai Super-GOAT guide: `riir-ai/.research/145_CWM_Runtime_Induced_Game_Rules_Guide.md`
- riir-ai runtime plan: `riir-ai/.plans/326_cwm_npc_runtime_integration.md`
