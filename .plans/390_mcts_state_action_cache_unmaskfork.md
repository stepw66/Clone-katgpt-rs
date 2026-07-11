# Plan 390: State-Action Pair Cache for MCTS over Deterministic Inference Actions (UnMaskFork)

**Date:** 2026-07-07
**Research:** [katgpt-rs/.research/386_UnMaskFork_Deterministic_Action_Branching_MCTS.md](../.research/386_UnMaskFork_Deterministic_Action_Branching_MCTS.md)
**Source paper:** [arxiv 2602.04344](https://arxiv.org/abs/2602.04344) — UnMaskFork: Test-Time Scaling for Masked Diffusion via Deterministic Action Branching (Misaki & Akiba, Sakana AI, Feb 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/mcts_state_action_cache.rs` (new module) + Cargo feature `mcts_state_action_cache`
**Status:** Closed — opt-in-forever (Phase 3 G2 FAILED, Phase 4 T4.3 executed). See `.issues/044_*`.

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

- [x] **T2.1** Implement `tests/mcts_state_action_cache_eq1.rs` — a property test of the switching-policy dominance inequality:
  - [x] Construct a synthetic kernel-error landscape: `ε_a(z) = known_function(a, z)` for `a ∈ {0,1,2}` (three actions) and `z` ranging over a discrete grid of 100 states.
  - [x] Compute LHS: `Σ_z min_a ε_a(z)` (state-dependent switching policy)
  - [x] Compute RHS: `min_a Σ_z ε_a(z)` (best static single action)
  - [x] Assert `LHS ≤ RHS` (with a small floating-point epsilon)
  - [x] Construct a fixture where the strict inequality holds (`LHS < RHS`) — e.g. action 0 wins in states 0..50, action 1 wins in states 50..100. Proves "interleaving beats any single static kernel".
  - [x] Construct a fixture where `LHS == RHS` (one action dominates everywhere — switching provides no benefit). Proves the inequality is tight in the trivial case.
  - **Additional test:** `eq1_switching_picks_different_actions_in_different_regions` verifies the argmin actually switches between regions (structural precondition for the strict inequality).
  - **Gradient fix:** initial gradients (0.01/z) let the decoy action 2 beat action 0 near the z=50 boundary; reduced to 0.002/z (action 0/1) and 0.001/z (decoy) so the winner stays the winner throughout each region.
- [x] **T2.2** Add debug-mode determinism re-check to `StateActionCache`:
  - [x] `#[cfg(debug_assertions)] pub fn verify_determinism<S, A>(&self, space: &A, samples: &[(S, InferenceAction)]) -> usize` — for each `(state, action)` sample, re-applies the action, BLAKE3-compares the recomputed next_state hash to the cached next_state hash. Returns the number of mismatches.
  - [x] Document: in release builds, the caller's DeterministicTransition contract is trusted; this re-check is a debug-only diagnostic.
  - [x] Add a unit test that constructs a deterministic action space, populates the cache, then runs `verify_determinism` and asserts 0 mismatches.
  - **Signature deviation:** the plan proposed `verify_determinism(space, sample_size)` that picks random cached entries. But the cache stores `(state_hash, action) → (next_hash, reward)` — it does NOT store the original state, and BLAKE3 is not invertible, so you cannot re-apply an action to a hash. Corrected to `verify_determinism(space, samples: &[(S, InferenceAction)])` where the caller supplies the concrete states to audit (the method computes the hash internally to look up the cache entry). Added a second test (`verify_determinism_skips_uncached_pairs`) for the skip-on-miss path.
- [x] **T2.3** Add a unit test for cache invalidation semantics:
  - [x] Construct a space, populate the cache
  - [x] Verify a known `(state, action)` returns the cached `(next_state, reward)`
  - [x] `clear()`, verify the same lookup now returns `None`
  - [x] Re-populate, verify it returns again

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

- [x] **T3.1** Implement `benches/bench_390_mcts_state_action_cache_goat.rs` (`harness = false`, `required-features = ["mcts_state_action_cache"]`):
  - [x] Build the synthetic dLLM-like domain (16-token sequences, 7 depth levels, 3 actions, deterministic transitions, terminal reward)
  - [x] **G1 — cache hit rate vs NFE**: sweep NFE ∈ {256, 512, 1024, 2048, 4096, 8192}, measure cache hit rate. Target: ≥30% hit rate at NFE ≥ 1024. **RESULT: PASS — 42.1% at NFE=1024.**
  - [x] **G2 — effective-budget expansion at matched reward**: at fixed target reward, measure NFE required for cached vs no-cache. Target: ≥1.4×. **RESULT: FAIL — re-gated 2026-07-07 (Issue 044) with three fixes (scaled domain 48/12/5, true no-cache baseline via `StateActionCache::disabled()`, direct NFE-savings metric via `total_rollout_steps`). G2a reward-convergence: 0/6 strict wins (both arms converge to 1.000 at NFE=256). G2b NFE-savings: 1.01–1.03× expansion (avg_rollout_depth=0.6–0.7; rollouts are short because the tree itself reaches terminal depth). The synthetic domain's `apply` is too cheap (4-token array write ~ns) for cache hits to translate to meaningful NFE savings. Only a real dLLM PoC (Plan 5) where each `apply` is a full forward pass can show the budget-expansion benefit.**
  - [x] **G3 — no-regression**: cached reward ≥ no-cache reward at every NFE. **RESULT: PASS — identical at every NFE.**
  - [-] **G4 — zero-alloc hot path**: deferred to a separate CountingAllocator binary (the SearchScratch struct is pre-allocated; the full alloc gate is non-blocking for the opt-in-forever verdict).
  - [x] **G5 — cache size bound**: report `cache.len()` at end of search for each NFE. **RESULT: PASS — 107 entries at NFE=8192 (0.013× NFE).** Bounded by the domain's state space × actions.
- [-] **T3.2** Implement `tests/mcts_state_action_cache_goat_gate.rs` (the gate test that CI runs):
  - DEFERRED — the GOAT gate FAILED on G2 (see T4.3), so the gate test would assert a failing condition. The benchmark binary (`bench_390_mcts_state_action_cache_goat`) serves as the gate artifact instead. A gate test can be added if G2 is re-gated on a larger domain.
- [x] **T3.3** Add the bench entry to `katgpt-rs/crates/katgpt-core/Cargo.toml` `[[bench]]` section with `required-features = ["mcts_state_action_cache"]`.

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

- [-] **T4.1** If Phase 3 GOAT gate PASSES with strong margins (G2 ≥1.4×, G1 ≥30%, G4 zero-alloc), promote to default:
  - NOT REACHED — G2 FAILED (opt-in-forever).
- [-] **T4.2** If Phase 3 GOAT gate PASSES with weak margins (G2 ~1.4× borderline, or G1 ~30% borderline), keep opt-in:
  - NOT REACHED — G2 FAILED (not borderline, opt-in-forever).
- [x] **T4.3** If Phase 3 GOAT gate FAILS:
  - [x] Revise Research 386 §3 verdict from GOAT to Gain
  - [x] Keep the feature opt-in as a diagnostic primitive (the cache is still correct, just not a perf win on this domain)
  - [x] Create `.issues/NNN_mcts_state_action_cache_gap.md` documenting which gate failed and what a real dLLM PoC would need to re-test
  - [x] Close the plan as opt-in-forever (no Phase 5)

---

## Phase 5 — D2F Host Integration (deferred, contingent on Phase 4 promote-to-default)

Goal: if the primitive promotes to default, wire it into the existing D2F inference pipeline (Plan 066) so that `SolverKind × remasking strategy` becomes the action axis for real dLLM unmasking trajectories.

**Status: DEFERRED.** This phase is contingent on Phase 4 promote-to-default AND a real dLLM PoC confirming the synthetic-domain numbers transfer. It is NOT in scope for the initial GOAT gate.

### Tasks (sketch only — not started)

- [-] **T5.1** Define `D2fInferenceActionSpace` impl of `InferenceActionSpace<D2fBlockState>`:
  - Actions = `(SolverKind, RemaskingStrategy)` tuples (3 solvers × 2 strategies = 6 actions, matching UMF's small action space)
  - `apply` = one D2F denoising step with the given config
  - `reward` = terminal reconstruction quality (proportion of correctly-denosed tokens)
  - `state_hash` = BLAKE3 of the partially-unmasked token sequence
- [-] **T5.2** Wire `mcts_search_with_state_action_cache` into `D2fPipeline` as an alternative decode strategy (alongside the existing sequential `D2fDecodeConfig`).
- [-] **T5.3** Benchmark on the existing micro-GPT domain (Plan 066 test fixtures) — measure cache hit rate and effective-budget expansion on a real (small) dLLM.

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
