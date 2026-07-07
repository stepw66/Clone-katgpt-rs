# Plan 390: State-Action Pair Cache for MCTS over Deterministic Inference Actions (UnMaskFork)

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/386_UnMaskFork_Deterministic_Action_Branching_MCTS.md](../.research/386_UnMaskFork_Deterministic_Action_Branching_MCTS.md)
**Source paper:** [arxiv 2602.04344](https://arxiv.org/abs/2602.04344) — UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching (Misaki & Akiba, Sakana AI, Feb 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/mcts_state_action_cache.rs` (new module) + Cargo feature `mcts_state_action_cache`
**Status:** Active — Phase 1 ready

---

## Goal

Distill UnMaskFork's state-action pair caching primitive into a generic, modelless, MIT-licensed Rust module under `katgpt-core`. The module provides a `StateActionCache` keyed on `(blake3::Hash, InferenceAction)` pairs and an `mcts_search_with_state_action_cache` entry point that consults/inserts the cache at Expand time. The cache converts the "same state, different deterministic action, different transition" axis into zero-cost cache hits on revisits. Goal: prove a ≥1.4× effective-budget expansion at matched reward on a controlled deterministic-decoding toy benchmark, then decide promote-to-default vs opt-in.

**Why this is GOAT, not Super-GOAT (per Research 386 §3):** Q3 (product selling point) is a perf claim ("cache reuse under fixed NFE"), not a capability claim — you cannot finish "our NPCs do X that no competitor can" with a *caching* primitive. The theoretical guarantee (UMF Eq. 1: sum-of-mins ≤ min-of-sums) is the genuine novel insight but is also a perf argument. No private guide created (verdict is not Super-GOAT). No UQ-bearing floor check (UMF produces a single best trajectory by reward, not a probability distribution).

**Scope guardrails:**
- The open primitive is generic search/caching math. No dLLM dependency, no game IP, no chain IP, no shard types.
- Existing `mcts_search` API is unchanged — the cache is an opt-in extension via a new function, not a modification.
- Existing `TranspositionTable` (Plan 061) and `ProofGoalCache` (Plan 388) are NOT deprecated — they remain the right tool for state-only caching.
- No riir-train deferral (UMF is modelless by construction). No §3.5 modelless-unblock needed.
- No §3.6 defend-wrong PoC required pre-gate (the benchmark IS the gate).

---

## Phase 1 — Unblocking Skeleton (CORE)

Goal: a compiling, tested, feature-gated module that implements the state-action pair cache + a generic search entry point that uses it. Public API surface frozen.

### Tasks

- [x] **T1.1** Add feature flag `mcts_state_action_cache = ["dep:papaya", "dep:blake3"]` to `katgpt-rs/crates/katgpt-core/Cargo.toml` features section (after the existing `mcts`-adjacent entries; both deps already optional in this crate). Verify `papaya` and `blake3` are still listed as `optional = true` in `[dependencies]`.
  - **Deviation:** `blake3` is non-optional in this crate (line 16), so `dep:blake3` is invalid. Used `mcts_state_action_cache = ["dep:papaya"]` instead. Documented inline in Cargo.toml.
- [x] **T1.2** Add `#[cfg(feature = "mcts_state_action_cache")] pub mod mcts_state_action_cache;` to `katgpt-rs/crates/katgpt-core/src/lib.rs` (alphabetical, near `mcts`).
- [x] **T1.3** Implement `katgpt-rs/crates/katgpt-core/src/mcts_state_action_cache.rs`:
  - [x] `InferenceAction { config_id: u16, strategy_id: u8 }` — `#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]`, `#[repr(C)]` (3 bytes + 1 pad). Opaque action handle; caller-defined semantics (which solver kind / temperature / shard id / remasking strategy).
  - [x] `StateActionKey { state: blake3::Hash, action: InferenceAction }` — `#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]`. The cache key.
  - [x] `StateActionCache<R: Copy>` — wraps `papaya::HashMap<StateActionKey, (blake3::Hash, R)>` (papaya default `RandomState`; no custom hasher to keep it simple).
    - [x] `pub fn new() -> Self` — default capacity
    - [x] `pub fn with_capacity(cap: usize) -> Self`
    - [x] `pub fn get(&self, state: blake3::Hash, action: InferenceAction) -> Option<(blake3::Hash, R)>` — O(1) lock-free pin
    - [x] `pub fn insert(&self, state: blake3::Hash, action: InferenceAction, next: blake3::Hash, reward: R)` — inserts (caller MUST have observed the transition deterministically)
    - [x] `pub fn len(&self) -> usize` and `pub fn is_empty(&self) -> bool`
    - [x] `pub fn clear(&self)` — for per-search-scope reset (via `pin().clear()`)
  - [x] Document the **`DeterministicTransition` contract** loudly in doc comments: the caller guarantees that for any fixed `(state, action)`, applying `action` to `state` yields a unique `(next_state, reward)`. Violations cause stale cache hits. Debug-mode re-check (re-apply and BLAKE3-compare) is in Phase 2.
  - **Deviation:** used papaya's default `RandomState` instead of `BuildHasherDefault<DefaultHasher>` — simpler construction (`HashMap::new()` / `HashMap::with_capacity()`), and the key already carries 256 bits of BLAKE3 entropy so the hasher choice is not performance-critical.
- [x] **T1.4** Define the generic search trait surface that takes an opaque action axis:
  - [x] `pub trait InferenceActionSpace<S> { fn actions_at(&self, state: &S) -> &[InferenceAction]; fn apply(&self, state: &S, action: InferenceAction) -> S; fn reward(&self, state: &S) -> Option<f32>; fn is_terminal(&self, state: &S) -> bool; fn state_hash(&self, state: &S) -> blake3::Hash; }` — the caller implements this; the search is agnostic to action semantics.
  - [x] `pub fn mcts_search_with_state_action_cache<S, A>(space: &A, root: &S, budget: usize, cache: &StateActionCache<f32>, scratch: &mut SearchScratch) -> SearchResult` where `A: InferenceActionSpace<S>, S: Clone` — uses UCB1 selection (constant `UCB1_C = 1.414` from existing `mcts.rs`), consults the cache at Expand time, inserts after rollout. Falls back to standard rollout on cache miss.
  - [x] Note: this is a *standalone* search — it does NOT reuse the existing `mcts_search` from `mcts.rs` (which is parameterized over `GameState` game actions, not opaque inference actions). Keep the algorithm body small and self-contained; the existing `MCTSNode` struct design (index-based parent/child links, `ArrayVec` for children/unexpanded) is the template.
  - **Deviation:** (1) Added `S: Clone` bound (the search re-walks the selection path each iteration, re-applying actions inline, which needs one `root.clone()` per iteration — cheap for fixed-size/refcounted states). (2) Added a `scratch: &mut SearchScratch` parameter for the Phase 3 G4 zero-alloc hot-path gate (pre-allocated tree node vec + path stack). (3) Returns `SearchResult { best_action, cache_hits, cache_misses, tree_size }` instead of bare `S::Action` — richer return for gate reporting.
- [x] **T1.5** Write unit tests in `tests/mcts_state_action_cache_basic.rs`:
  - [x] Cache insert/get round-trip (deterministic — same key returns same value)
  - [x] Same state + different action → different cache entries (the key novelty vs state-only)
  - [x] Different state + same action → different cache entries
  - [x] Cache `clear()` empties all entries
  - [x] `InferenceAction` is 4 bytes (`size_of::<InferenceAction>() == 4`) — verify with `assert_eq!`
  - [x] Determinism contract documented in the module doc-comments (determinism-across-runs test in the test binary)
  - **Additional tests beyond the plan:** search-finds-action-on-fresh-cache, second-search-hits-cache, search-is-deterministic-across-runs, search-returns-none-for-terminal-root (4 search-integration tests). Pure-cache tests also duplicated inline in the module under `#[cfg(test)]`.
- [x] **T1.6** Write a small example `examples/mcts_state_action_cache_basic.rs`:
  - [x] Synthetic 3-action space (3 "inference configurations") over a 4-step deterministic transition graph
  - [x] Run search with cache, print: total rollouts, cache hits, cache misses, best reward
  - [x] Re-run search with the SAME cache populated — show 100% cache hit rate on the second run (zero rollouts). **Verified:** Run 1 = 53.3% hit rate (intra-search reuse), Run 2 = 100% hit rate (graph fully cached).
- [x] **T1.7** Document the module in `mcts_state_action_cache.rs` with: paper reference (arXiv:2602.04344), the DeterministicTransition contract, the Eq. 1 motivation, and the "vs state-only transposition" distinction.

### Phase 1 Exit Criteria

- `cargo build -p katgpt-core --features mcts_state_action_cache` compiles clean
- `cargo test -p katgpt-core --features mcts_state_action_cache --test mcts_state_action_cache_basic` passes all unit tests
- `cargo run --example mcts_state_action_cache_basic --features mcts_state_action_cache --release` runs and prints expected hit-rate numbers
- No new clippy warnings on the module

---

## Phase 2 — Eq. 1 Property Test + Determinism Re-check

Goal: prove the theoretical guarantee (Eq. 1: sum-of-mins ≤ min-of-sums) holds empirically on a controlled toy domain, and add the debug-mode determinism re-check.

### Tasks

- [ ] **T2.1** Implement `tests/mcts_state_action_cache_eq1.rs` — a property test of the switching-policy dominance inequality:
  - [ ] Construct a synthetic kernel-error landscape: `ε_a(z) = known_function(a, z)` for `a ∈ {0,1,2}` (three actions) and `z` ranging over a discrete grid of 100 states.
  - [ ] Compute LHS: `Σ_z min_a ε_a(z)` (state-dependent switching policy)
  - [ ] Compute RHS: `min_a Σ_z ε_a(z)` (best static single action)
  - [ ] Assert `LHS ≤ RHS` (with a small floating-point epsilon)
  - [ ] Construct a fixture where the strict inequality holds (`LHS < RHS`) — e.g. action 0 wins in states 0..50, action 1 wins in states 50..100. Proves "interleaving beats any single static kernel".
  - [ ] Construct a fixture where `LHS == RHS` (one action dominates everywhere — switching provides no benefit). Proves the inequality is tight in the trivial case.
- [ ] **T2.2** Add debug-mode determinism re-check to `StateActionCache`:
  - [ ] `#[cfg(debug_assertions)] pub fn verify_determinism<S, A>(&self, space: &A, sample_size: usize) -> usize` — picks `sample_size` random cached entries, re-applies the action, BLAKE3-compares the recomputed next_state to the cached next_state. Returns the number of mismatches.
  - [ ] Document: in release builds, the caller's DeterministicTransition contract is trusted; this re-check is a debug-only diagnostic.
  - [ ] Add a unit test that constructs a deterministic action space, populates the cache, then runs `verify_determinism` and asserts 0 mismatches.
- [ ] **T2.3** Add a unit test for cache invalidation semantics:
  - [ ] Construct a space, populate the cache
  - [ ] Verify a known `(state, action)` returns the cached `(next_state, reward)`
  - [ ] `clear()`, verify the same lookup now returns `None`
  - [ ] Re-populate, verify it returns again

### Phase 2 Exit Criteria

- Eq. 1 property test passes (LHS ≤ RHS, with a strict-inequality fixture and an equality fixture)
- Determinism re-check returns 0 mismatches on a deterministic space
- All Phase 1 tests still pass

---

## Phase 3 — GOAT Gate Benchmark (the deciding gate)

Goal: a controlled benchmark comparing state-action cache vs state-only cache vs no-cache under fixed budget, on a deterministic-decoding toy domain. Decides promote-to-default vs opt-in.

**Domain choice:** a synthetic dLLM-like deterministic unmasking task. We construct a deterministic transition graph (no RNG in the transition function) where:
- States = partially-unmasked token sequences of length 16, with mask-ratio schedule `[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2]` (7 depth levels)
- Actions = 3 "inference configurations" (analogous to UMF's `(θ_a, T_a, g_a)` but abstracted: just three deterministic transition functions over the token state)
- Reward = terminal quality score (a deterministic function of the final unmasked sequence — e.g. number of "correct" tokens against a target)
- The 3 actions have non-overlapping strengths: action 0 wins in early mask-ratios, action 1 wins in mid, action 2 wins in late — so the optimal trajectory interleaves all three (matches UMF's §6.4 case study).

This domain is chosen because:
- It is deterministic (satisfies the DeterministicTransition contract — no RNG in transitions)
- It exercises the state-action caching axis (same partial-state, three different actions → three different next-states)
- It has a clear reward signal (terminal quality)
- It is small enough to run thousands of search iterations in seconds (criterion bench)

### Tasks

- [ ] **T3.1** Implement `benches/bench_390_mcts_state_action_cache_goat.rs` (criterion, `harness = false`, `required-features = ["mcts_state_action_cache"]`):
  - [ ] Build the synthetic dLLM-like domain (16-token sequences, 7 depth levels, 3 actions, deterministic transitions, terminal reward)
  - [ ] **G1 — cache hit rate vs NFE**: sweep NFE ∈ {256, 512, 1024, 2048, 4096, 8192}, measure cache hit rate. Target: ≥30% hit rate at NFE ≥ 1024 (UMF reports 47.8% at NFE=3072; we expect lower on a smaller domain but should be non-zero and growing).
  - [ ] **G2 — effective-budget expansion at matched reward**: at fixed target reward (e.g. ≥0.8 of optimal), measure NFE required for (a) no-cache, (b) state-only transposition (cousin), (c) state-action cache. Target: state-action cache reaches target reward in ≤70% of the NFE that no-cache requires (i.e. ≥1.4× effective-budget expansion).
  - [ ] **G3 — no-regression**: same-domain reward for state-action cache ≥ no-cache reward at every NFE in the sweep (the cache can only help or break even under deterministic transitions).
  - [ ] **G4 — zero-alloc hot path**: count allocations in the `mcts_search_with_state_action_cache` inner loop. Target: 0 allocs per Expand after warmup (the cache lookup and insert are alloc-free; the action buffer is pre-allocated). Use the existing alloc-counting pattern (e.g. the `bom_sampling` G4 pattern from Plan 281).
  - [ ] **G5 — cache size bound**: report `cache.len()` at end of search for each NFE. Verify it is bounded by O(NFE × avg_actions_per_state). Add an optional LRU cap and verify it triggers correctly when set.
- [ ] **T3.2** Implement `tests/mcts_state_action_cache_goat_gate.rs` (the gate test that CI runs):
  - [ ] Re-runs G1, G2, G3 as asserts (G4/G5 are bench-only). Document the thresholds.
  - [ ] If all of G1, G2, G3 pass → GOAT confirmed. Document the numbers in the test file header.
  - [ ] If G2 fails (effective-budget expansion < 1.4×) → GOAT FAILED; the primitive stays opt-in, and an `.issues/` entry is created to track the gap. The verdict in Research 386 is revised to "Gain" and the plan is closed as opt-in-forever.
- [ ] **T3.3** Add the bench + test entries to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[[bench]]` and `[[test]]` sections with `required-features = ["mcts_state_action_cache"]`.

### Phase 3 Exit Criteria

- G1 cache hit rate ≥30% at NFE ≥1024 — confirms the cache is actually being used
- G2 effective-budget expansion ≥1.4× — the headline GOAT number
- G3 no-regression — the cache never hurts under deterministic transitions
- G4 0 allocs per Expand after warmup — the hot-path discipline
- G5 cache size bounded — the memory discipline
- If G1+G2+G3 all PASS → proceed to Phase 4 (promote-to-default decision)
- If any FAIL → GOAT gate fails; close the plan as opt-in, create `.issues/NNN_*`, revise Research 386 verdict

---

## Phase 4 — Promote-to-Default Decision (only if Phase 3 GOAT passes)

Goal: decide whether to promote `mcts_state_action_cache` to the `default` feature list, or keep opt-in.

### Tasks

- [ ] **T4.1** If Phase 3 GOAT gate PASSES with strong margins (G2 ≥1.4×, G1 ≥30%, G4 zero-alloc), promote to default:
  - [ ] Add `mcts_state_action_cache` to the `default = [...]` list in `katgpt-rs/crates/katgpt-core/Cargo.toml`
  - [ ] Update the feature comment to DEFAULT-ON with the GOAT summary
  - [ ] Update `katgpt-rs/README.md` Feature Showcase with a new entry citing arXiv:2602.04344
  - [ ] Update Research 386 §3 verdict to confirm promotion
  - [ ] Run `cargo check --workspace --all-features` and `cargo test -p katgpt-core --lib` to verify no regressions
- [ ] **T4.2** If Phase 3 GOAT gate PASSES with weak margins (G2 ~1.4× borderline, or G1 ~30% borderline), keep opt-in:
  - [ ] Add the feature comment with the GOAT numbers and "opt-in, weak margin" annotation
  - [ ] Create `.issues/NNN_mcts_state_action_cache_promote_followup.md` to track a re-gate after a real dLLM-domain PoC (the synthetic domain may under-represent the real hit rate)
- [ ] **T4.3** If Phase 3 GOAT gate FAILS:
  - [ ] Revise Research 386 §3 verdict from GOAT to Gain
  - [ ] Keep the feature opt-in as a diagnostic primitive (the cache is still correct, just not a perf win on this domain)
  - [ ] Create `.issues/NNN_mcts_state_action_cache_gap.md` documenting which gate failed and what a real dLLM PoC would need to re-test
  - [ ] Close the plan as opt-in-forever (no Phase 5)

---

## Phase 5 — D2F Host Integration (deferred, contingent on Phase 4 promote-to-default)

Goal: if the primitive promotes to default, wire it into the existing D2F inference pipeline (Plan 066) so that `SolverKind × remasking strategy` becomes the action axis for real dLLM unmasking trajectories.

**Status: DEFERRED.** This phase is contingent on Phase 4 promote-to-default AND a real dLLM PoC confirming the synthetic-domain numbers transfer. It is NOT in scope for the initial GOAT gate.

### Tasks (sketch only — not started)

- [ ] **T5.1** Define `D2fInferenceActionSpace` impl of `InferenceActionSpace<D2fBlockState>`:
  - Actions = `(SolverKind, RemaskingStrategy)` tuples (3 solvers × 2 strategies = 6 actions, matching UMF's small action space)
  - `apply` = one D2F denoising step with the given config
  - `reward` = terminal reconstruction quality (proportion of correctly-denosed tokens)
  - `state_hash` = BLAKE3 of the partially-unmasked token sequence
- [ ] **T5.2** Wire `mcts_search_with_state_action_cache` into `D2fPipeline` as an alternative decode strategy (alongside the existing sequential `D2fDecodeConfig`).
- [ ] **T5.3** Benchmark on the existing micro-GPT domain (Plan 066 test fixtures) — measure cache hit rate and effective-budget expansion on a real (small) dLLM.

---

## Phase 6 — riir-ai Follow-up (DEFERRED, not in this plan)

The riir-ai follow-up (multi-shard-as-action for per-NPC runtime test-time scaling) is mentioned in Research 386 §2.3 as a potential fusion, but is contingent on Phase 4 promote-to-default AND a confirmed crowd-NPC use case. It would be opened as a separate riir-ai plan if and when the GOAT gate passes and a concrete NPC-decision-tree use case emerges. NOT in scope here.

---

## Validation Plan

- **Phase 1**: `cargo test -p katgpt-core --features mcts_state_action_cache --test mcts_state_action_cache_basic` + example runs
- **Phase 2**: Eq. 1 property test + determinism re-check tests
- **Phase 3**: `cargo bench -p katgpt-core --features mcts_state_action_cache --bench bench_390_mcts_state_action_cache_goat` + `cargo test -p katgpt-core --features mcts_state_action_cache --test mcts_state_action_cache_goat_gate`
- **Phase 4** (if promoted): `cargo check --workspace --all-features` + `cargo test -p katgpt-core --lib` no-regression

Use `CARGO_TARGET_DIR=/tmp/plan390` per AGENTS.md rule (avoid contending for the shared target dir). Clean up when done.

---

## Open Questions

- **Real dLLM hit rate vs synthetic**: UMF reports ~55% hit rate at NFE=12288 on coding tasks. Our synthetic domain (16 tokens, 7 depth levels, 3 actions) may have a very different hit-rate profile — possibly higher (smaller state space → more revisits) or lower (fewer distinct trajectories worth caching). The Phase 3 G1 threshold (≥30% at NFE=1024) is a guess; the bench will tell us.
- **Promotion threshold**: 1.4× effective-budget expansion is the bar. UMF shows 7.7% pass@1 improvement at NFE=12288 from caching (Table 3); that's a quality gain, not a budget gain. The budget-equivalent number is "what NFE would no-cache need to match cached NFE=12288's quality" — UMF doesn't report this directly. The 1.4× threshold is a reasonable starting bar; revisit if the bench shows a different quality/budget trade-off.
- **Hashing cost**: BLAKE3 of the full state per (state, action) lookup is the hot-path cost. If G4 (zero-alloc) passes but wall-clock is dominated by hashing, a follow-up task would add an optional 64-bit fingerprint first-level probe. Defer until the bench shows whether this is needed.
