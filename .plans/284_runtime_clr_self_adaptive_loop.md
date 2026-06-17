# Plan 284: Runtime CLR — Claim-Level Reliability + Self-Adaptive Test-Time Scaling (Open Primitive)

**Date:** 2026-06-17
**Research:** [katgpt-rs/.research/255_VibeThinker_CLR_Test_Time_Reliability.md](../.research/255_VibeThinker_CLR_Test_Time_Reliability.md)
**Private guide:** [riir-ai/.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md](../../riir-ai/.research/136_Per_NPC_Runtime_Test_Time_Scaling_Guide.md)
**Source paper:** [arxiv 2606.16140](https://arxiv.org/pdf/2606.16140) — Xu et al., "VibeThinker-3B" (Sina Weibo Inc.), 15 Jun 2026
**Target:** `katgpt-rs/src/clr/` (new module) + Cargo feature `clr` (opt-in until GOAT G1–G5 pass)
**Status:** Active — Phase 1 not started
**Depends On:** existing SIMD helpers (`simd_dot_f32`, `simd_sum_f32`, `simd_exp_inplace` from `crates/katgpt-core/src/simd.rs`), `ConstraintPruner` trait (existing, for the fallback binary verifier path)
**GOAT Criteria:** G1 (CLR-vote ≥ +3pp over best-of-N majority on synthetic suite), G2 (verifier sigmoid ECE ≤ 0.10), G3 (≤200µs/call at K=32, M=5, 8-dim direction vectors — target ≤50µs), G4 (zero heap allocation on the vote path), G5 (feature isolation — compiles with/without `clr`, zero overhead when disabled)

---

## Goal

Ship the four modelless primitives distilled from Research 255 as a generic, MIT-licensed, no-game-semantics module in katgpt-rs:

1. **`clr_vote()`** — the headline nonlinear reliability gate. Given K candidate trajectories, M decision-relevant claims per trajectory, a `ClaimExtractor`, and a `ClaimVerifier`, produce a `VoteResult` containing the winning cluster. Core math: `r_k = (mean_m v_k,m)^M` where `v_k,m = sigmoid(dot(claim_vec_k,m, direction_vec_m))` — dot-product + **sigmoid, never softmax** (per AGENTS.md).
2. **`ClaimExtractor` / `ClaimVerifier` traits** — open extension points. Concrete extractors/verifiers live in the consumer crate (riir-ai Plan 316 ships game-specific ones; katgpt-rs ships only the generic trait + a `FnClaimExtractor` adapter for tests).
3. **`brevity_tiebreak()`** — the Long2Short zero-sum tiebreak. Among clusters tied on `Σ r_k` within `ε`, pick the one whose representative trajectory has the shortest length. Pure algorithm, no quality change.
4. **`learning_potential()` + `mgpo_sampling_weight()`** — the curiosity feedback signals. `learning_potential(y, log_prob_fn)` returns `-(1/|y|) Σ log π(y_t|...)`. `mgpo_sampling_weight(p, gamma)` returns `exp(-gamma * |2p - 1|)` (peaks at p=0.5, the maximum-entropy / calibration boundary).

**GOAT gate:** G1–G5 (defined in detail in Phase 3). The headline per-NPC task-gain proof (G6–G11 from the riir-ai guide) lives in riir-ai Plan 316 — this plan ships only the open math + traits.

**Non-goals (explicitly out of scope here):**
- Per-NPC wiring, game-specific claim extractors, freeze/thaw cycle → riir-ai Plan 316.
- Direction-vector training via backprop → riir-train.
- The VibeThinker-3B post-training recipe (SFT + MGPO RL + offline self-distillation + Instruct RL) → riir-train redirect. This plan ships only the modelless primitives.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Create `src/clr/mod.rs` with module root + re-exports. Add `clr` feature to root `Cargo.toml` (opt-in, NOT in `default` or `full` until G1–G5 pass). Gate all module code behind `#[cfg(feature = "clr")]`. Update `src/lib.rs` to declare `pub mod clr;` behind the feature.
- [ ] **T1.2** Define types in `src/clr/types.rs`:
  - `pub struct ClrConfig { pub k: usize, pub m: usize, pub tau_v: f32, pub tau_reliable: f32, pub tau_curiosity: f32, pub alpha_freeze_thaw: f32, pub gamma_mgpo: f32, pub lambda_long2short: f32, pub tiebreak_eps: f32 }` — the full saCLR config. Defaults (paper): `k=32, m=5, tau_v=0.5, tau_reliable=0.5, tau_curiosity=0.7, alpha_freeze_thaw=0.01, gamma_mgpo=2.0, lambda_long2short=0.2, tiebreak_eps=1e-3`.
  - `pub struct Trajectory<T> { pub outcome: T, pub tokens_or_steps: usize, pub claims: Vec<T>, pub log_probs: Option<Vec<f32>> }` — generic over the outcome/claim type. `tokens_or_steps` is the length used by Long2Short. `claims` is filled by `ClaimExtractor::extract()`. `log_probs` is optional — present only when `learning_potential` is being computed (cheap path: don't compute if no consumer).
  - `pub struct Claim<T> { pub embedding: Vec<f32>, pub payload: T }` — `embedding` is the latent vector for dot-product + sigmoid projection onto a direction vector; `payload` is the opaque claim data for downstream consumers.
  - `pub type Verdict = f32;` — sigmoid output in `[0, 1]`. The binary threshold `v > tau_v` is applied inside `clr_vote()`, not by the verifier.
  - `pub type ReliabilityScore = f32;` — the `(mean)^M` score.
  - `pub struct Cluster<T> { pub outcome: T, pub total_reliability: ReliabilityScore, pub representative_idx: usize, pub member_indices: Vec<usize> }` — output of the vote. `representative_idx` is the trajectory chosen to represent the cluster (by Long2Short after tiebreak).
  - `pub struct VoteResult<T> { pub winner: Cluster<T>, pub all_clusters: Vec<Cluster<T>>, pub per_trajectory_reliability: Vec<ReliabilityScore>, pub per_trajectory_verdicts: Vec<[Verdict; M_DYNAMIC]> }` — caller gets the winner + full audit trail for visualization/debugging. Use `Vec<Verdict>` rather than `[Verdict; M]` to keep M dynamic at runtime (avoids const-generics complexity in v1).
- [ ] **T1.3** Define traits in `src/clr/traits.rs`:
  - `pub trait ClaimExtractor<T> { fn extract(&self, trajectory: &Trajectory<T>) -> Vec<Claim<T>>; }` — returns exactly `M` claims (caller asserts length). Domain-specific.
  - `pub trait ClaimVerifier<T> { fn verify(&self, claim: &Claim<T>, direction_idx: usize) -> Verdict; }` — returns sigmoid(dot(claim.embedding, direction_vec[direction_idx])). `direction_idx ∈ [0, M)` indexes into a direction-vector pool that the verifier owns.
  - `pub trait DirectionVectorSource { fn direction(&self, idx: usize) -> &[f32]; fn blake3(&self) -> [u8; 32]; fn version(&self) -> u64; }` — for freeze/thaw versioning. Concrete impls in consumer crates.
- [ ] **T1.4** Implement `FnClaimExtractor` reference adapter in `src/clr/extractor.rs`:
  - `pub struct FnClaimExtractor<F, T> { pub m: usize, pub f: F, _phantom: PhantomData<T> } where F: Fn(&Trajectory<T>) -> Vec<Claim<T>>`
  - Implements `ClaimExtractor<T>` by delegating to `f`. Asserts `result.len() == m`. Used in tests + as a quick adapter for callers that don't want to define a full struct.
- [ ] **T1.5** Implement `SigmoidProjectionVerifier` reference impl in `src/clr/verifier.rs`:
  - `pub struct SigmoidProjectionVerifier<'a> { pub directions: &'a DirectionVectorSource, pub direction_dim: usize }`
  - `verify(claim, direction_idx)`: `let d = directions.direction(direction_idx); let dot = simd_dot_f32(&claim.embedding, d, direction_dim); sigmoid(dot)` where `sigmoid(x) = 1.0 / (1.0 + simd_exp_inplace_one(-x))`. Reuse `simd_dot_f32` from `crates/katgpt-core/src/simd.rs`. **No softmax anywhere.**
- [ ] **T1.6** Implement `brevity_tiebreak()` in `src/clr/brevity.rs`:
  - `pub fn brevity_tiebreak<T>(candidates: &[&Cluster<T>], trajectories: &[Trajectory<T>], eps: f32) -> usize` — among candidates whose `total_reliability` is within `eps` of the max, return the index of the one whose representative trajectory has the smallest `tokens_or_steps`. Pure algorithm, zero allocation beyond the input scan.

---

## Phase 2 — Core Vote + Curiosity Signals

- [ ] **T2.1** Implement `clr_vote()` in `src/clr/vote.rs`:
  - Signature:
    ```
    pub fn clr_vote<T, E: ClaimExtractor<T>, V: ClaimVerifier<T>>(
        trajectories: &[Trajectory<T>],
        extractor: &E,
        verifier: &V,
        config: &ClrConfig,
        outcome_eq: &impl Fn(&T, &T) -> bool,
        scratch: &mut ClrScratch,
    ) -> VoteResult<T>
    ```
  - Algorithm (per Research 255 §2.3):
    1. For each `k in [0, K)`: `extractor.extract(&trajectories[k])` → `claims[k]` (asserts `claims[k].len() == M`).
    2. For each `(k, m)`: `scratch.verdicts[k*M + m] = verifier.verify(&claims[k][m], m)`.
    3. For each `k`: `let mean_v = simd_sum_f32(&scratch.verdicts[k*M..(k+1)*M]) / M as f32; scratch.reliability[k] = mean_v.powf(M as f32);` — the `(mean)^M` gate. Use `powf` (or a fixed-M integer-power unroll for `M=5`).
    4. Cluster by outcome equivalence (use `outcome_eq` callback; naive O(K²) for small K is fine — K≤32).
    5. For each cluster: sum reliabilities of members.
    6. Pick winner via `brevity_tiebreak` among clusters within `eps` of the max.
  - **Allocation discipline:** `scratch.verdicts` is `Vec<f32>::with_capacity(K*M)` allocated once by the caller; `clr_vote` writes into it via indexing, no growth. `scratch.reliability` similarly. `scratch.cluster_id` is `Vec<u8>::with_capacity(K)`. The returned `VoteResult` does allocate (`all_clusters`, `per_trajectory_*`) — but those are *output*, not hot-path; callers that don't need the audit trail can use `clr_vote_minimal()` (Phase 2 T2.3) which returns just the winner index.
- [ ] **T2.2** Implement `ClrScratch` in `src/clr/scratch.rs`:
  - `pub struct ClrScratch { pub verdicts: Vec<f32>, pub reliability: Vec<f32>, pub cluster_id: Vec<u8> }`
  - `pub fn ClrScratch::new(k: usize, m: usize) -> Self` — pre-allocates all three buffers.
  - `pub fn ClrScratch::reset(&mut self)` — `clear()` without freeing capacity; called by `clr_vote()` at entry.
  - **Zero allocation after the first `new()`.** Subsequent `clr_vote()` calls reuse the buffers.
- [ ] **T2.3** Implement `clr_vote_minimal()` in `src/clr/vote.rs`:
  - Like `clr_vote` but returns only `(winner_idx: usize, winner_reliability: f32)`. Skips the `all_clusters` / `per_trajectory_*` allocation. Used by hot-path callers (the per-NPC CLR cycle in riir-ai Plan 316).
- [ ] **T2.4** Implement `learning_potential()` in `src/clr/learning_potential.rs`:
  - `pub fn learning_potential<F: Fn(usize) -> f32>(len: usize, log_prob_at: F) -> f32` — returns `-(1.0 / len as f32) * sum_{t=0..len} log_prob_at(t)`. Higher = more surprising under the current frozen brain. The caller supplies the per-token log-prob accessor; katgpt-rs doesn't depend on any model.
  - Companion: `pub fn should_write_memory(reliability: f32, s_lp: f32, config: &ClrConfig) -> bool` — `reliability > config.tau_reliable && s_lp > config.tau_curiosity`. The gateable curiosity-feedback predicate.
- [ ] **T2.5** Implement `mgpo_sampling_weight()` in `src/clr/mgpo.rs`:
  - `pub fn mgpo_sampling_weight(p: f32, gamma: f32) -> f32` — `(-gamma * (2.0 * p - 1.0).abs()).exp()`. Peaks at `p=0.5` (calibration boundary), decays toward `p=0` (too hard) and `p=1` (saturated). Caller maintains an EMA `p` per sampling seed.
  - Companion: `pub fn allocate_budget(weights: &[f32], total_budget: usize) -> Vec<usize>` — proportional allocation, returns per-seed sample counts. Used by the next-cycle budget step.

---

## Phase 3 — SIMD Optimization (G3, G4)

- [ ] **T3.1** Verify the inner loop vectorizes. The `verifier.verify()` call for fixed `direction_dim=8` should auto-vectorize via the existing `simd_dot_f32` helper. Add `#[inline(always)]` to `SigmoidProjectionVerifier::verify` and `FnClaimExtractor::extract` to encourage inlining across the `clr_vote` boundary.
- [ ] **T3.2** Add a fixed-M unrolled path for `M=5` (paper default). `powf(5.0)` is general but slow; an unrolled `v*v*v*v*v` for integer `M=5` is faster. Gate behind `if config.m == 5 { ... } else { ... }`.
- [ ] **T3.3** Add `#[cfg(test)]` allocation counter. Use `std::alloc::System` with a global allocator hook in `tests/bench_284_clr_goat.rs` to assert 0 allocations after `ClrScratch::new()` warmup.
- [ ] **T3.4** Profile with `cargo bench` (criterion). Establish baseline numbers at K=8, K=16, K=32, each at M=5, direction_dim=8. Record in `.benchmarks/284_clr_goat.md`.

---

## Phase 4 — GOAT Gate Benchmark (G1, G2, G3, G4, G5)

### Tasks

- [ ] **T4.1** G1 test `g1_clr_beats_best_of_n_majority` in `tests/bench_284_clr_goat.rs`:
  - Synthetic suite: 50 trajectory-groups, each with 5 clusters of 10 trajectories. In each cluster, exactly 1 trajectory has a ground-truth-flawed claim (its `embedding[m_flaw]` is set to a vector orthogonal to `direction_vec[m_flaw]`, forcing `v < 0.5` for that claim).
  - Run `clr_vote` (K=50, M=5) vs best-of-N majority (pick cluster with most members).
  - Assert CLR picks the flawless cluster ≥3pp more often than majority. Run over 100 random seeds, report mean + stddev.
- [ ] **T4.2** G2 test `g2_calibration_ece`:
  - Ground-truth binary verdicts (constructed so `v_k,m` is calibrated: random `embedding` projections, true verdict is `Bernoulli(sigmoid(dot))`).
  - Compute Expected Calibration Error of `SigmoidProjectionVerifier::verify` outputs.
  - Assert ECE ≤ 0.10 over 10K samples.
- [ ] **T4.3** G3 test `g3_hot_path_under_200us`:
  - `cargo bench` criterion group. K=32, M=5, direction_dim=8. Time per `clr_vote_minimal()` call.
  - Assert mean ≤ 200µs. Stretch target ≤ 50µs.
- [ ] **T4.4** G4 test `g4_zero_allocation`:
  - Custom global allocator that counts `alloc`/`dealloc` calls.
  - Warm up `ClrScratch::new(32, 5)` once.
  - Call `clr_vote_minimal()` 1000 times. Assert 0 net allocations after the first warmup.
- [ ] **T4.5** G5 test `g5_feature_isolation`:
  - `cargo build --no-default-features --features clr` compiles cleanly.
  - `cargo build --no-default-features` (no `clr`) compiles cleanly and `clr` symbols are absent from the binary (`nm` check or trait-resolution failure when attempting to use `clr_vote`).
  - Zero overhead when disabled — assert no `clr` code paths reachable from the default-features build.
- [ ] **T4.6** Create `.benchmarks/284_clr_goat.md` placeholder with sections for G1–G5 results. Fill in after Phase 4 runs.

---

## Phase 5 — Documentation + Examples + Promotion

- [ ] **T5.1** Add `examples/clr_minimal.rs` — synthetic reliability suite, run `clr_vote`, print winner + reliability scores for each trajectory. <150 lines.
- [ ] **T5.2** Add `examples/clr_brevity_tiebreak.rs` — two clusters tied on `Σ r_k`, show `brevity_tiebreak` picking the shorter representative.
- [ ] **T5.3** Add `examples/clr_learning_potential.rs` — given a trajectory + a fake log-prob accessor, compute `S_LP` and demo `should_write_memory`.
- [ ] **T5.4** Add `clr` row to the feature table in `katgpt-rs/.docs/01_overview.md` (opt-in until G1–G5 pass, then promote).
- [ ] **T5.5** Add CLR section to `katgpt-rs/README.md` Feature Showcase (after the most recent Super-GOAT — find the right insertion point by grepping for the latest entry). Cross-ref to Research 255 and the riir-ai guide.
- [ ] **T5.6** Promotion decision:
  - If G1–G5 all pass → move `clr` from opt-in to default-on in root `Cargo.toml`. Update README to note "GOAT-proved".
  - If G1 fails (CLR doesn't beat majority on synthetic) → keep opt-in, note in `.benchmarks/284_clr_goat.md` that the mechanism is correct but the synthetic suite may be too easy (revisit with harder suite).
  - If G3 fails (>200µs) → keep opt-in, profile to find the bottleneck (likely the `powf` or the outcome_eq callback). Demote hot-path callers to `clr_vote_minimal` only.
  - If G4 fails (allocates) → critical bug in scratch discipline; fix before any promotion.

---

## Dependencies

| Dependency | Source | Status |
|-----------|--------|--------|
| `simd_dot_f32`, `simd_sum_f32`, `simd_exp_inplace` | `crates/katgpt-core/src/simd.rs` | ✅ Shipped |
| `ConstraintPruner` trait | existing | ✅ Shipped (for fallback binary verifier path, not used by default) |
| `fastrand` | existing dep | ✅ Available (for synthetic tests) |
| `criterion` | dev-dep | ✅ Available (for `cargo bench`) |
| riir-ai Plan 316 | runtime consumer | ⏳ Blocked on this plan's Phase 1 |

---

## File Changes Summary

| File | Action | Phase |
|------|--------|-------|
| `src/clr/mod.rs` | NEW | 1 |
| `src/clr/types.rs` | NEW | 1 |
| `src/clr/traits.rs` | NEW | 1 |
| `src/clr/extractor.rs` | NEW | 1 |
| `src/clr/verifier.rs` | NEW | 1 |
| `src/clr/brevity.rs` | NEW | 1 |
| `src/clr/vote.rs` | NEW | 2 |
| `src/clr/scratch.rs` | NEW | 2 |
| `src/clr/learning_potential.rs` | NEW | 2 |
| `src/clr/mgpo.rs` | NEW | 2 |
| `src/lib.rs` | EXTEND (add `pub mod clr;` behind feature) | 1 |
| `Cargo.toml` | EXTEND (add `clr = []` feature) | 1 |
| `tests/bench_284_clr_goat.rs` | NEW | 4 |
| `.benchmarks/284_clr_goat.md` | NEW | 4 |
| `examples/clr_minimal.rs` | NEW | 5 |
| `examples/clr_brevity_tiebreak.rs` | NEW | 5 |
| `examples/clr_learning_potential.rs` | NEW | 5 |
| `.docs/01_overview.md` | EXTEND (feature table) | 5 |
| `README.md` | EXTEND (Feature Showcase section) | 5 |

---

## Risks

1. **`powf(M)` performance.** General `powf` is ~10× slower than an unrolled integer power. Mitigation: T3.2 unrolled path for `M=5`. If still slow, expose `M` as a const-generic and specialize at compile time.
2. **Outcome equality callback is caller-defined.** For game NPCs (Plan 316) this is "destination tile + action type"; for LLMs it's "answer-equivalence hash". The naive O(K²) clustering is fine for K≤32 but doesn't scale. If a caller needs K=128+, expose a `hash_outcome` alternative that pre-hashes into a `HashMap<u64, Vec<usize>>`. Defer until a real caller needs it.
3. **Calibration of `SigmoidProjectionVerifier`.** If the consumer's `direction_vec` is poorly scaled, the sigmoid outputs cluster at 0 or 1 (saturated). Mitigation: G2 catches this; consumer is responsible for normalizing direction vectors at freeze/thaw. Document in the trait doc-comment.
4. **Feature-isolation surprise.** If any *default-on* feature pulls in `clr` transitively, G5 fails. Mitigation: T1.1 declares `clr = []` with no deps; verify with `cargo tree --features default`.
5. **The `verifier.verify()` indirection may inhibit auto-vectorization.** The dot-product is inside a trait method called in a tight loop. Mitigation: `#[inline(always)]` on the impl (T3.1); if still not vectorizing, expose a `verify_batch` method that takes `&[Claim<T>; M]` and returns `[Verdict; M]`, allowing the compiler to see the full inner loop.

---

## TL;DR

Open primitive for Research 255's Super-GOAT. Ships `clr_vote()` (the `(mean_m v_k,m)^M` nonlinear reliability gate, zero-alloc, SIMD), `ClaimExtractor` / `ClaimVerifier` traits (open extension points — concrete impls in consumer crate), `brevity_tiebreak()` (Long2Short zero-sum tiebreak), `learning_potential()` (the `-(1/|y|) Σ log π(y_t)` curiosity score), and `mgpo_sampling_weight()` (`exp(-γ|2p-1|` calibration-boundary weighting). No game semantics, no per-NPC wiring (that's riir-ai Plan 316), no training (that's riir-train redirect). GOAT gate: G1 (CLR beats best-of-N majority by ≥3pp on synthetic suite), G2 (verifier ECE ≤ 0.10), G3 (≤200µs/call K=32 M=5, target ≤50µs), G4 (zero heap alloc on vote path), G5 (feature isolation). Feature `clr`, opt-in until G1–G5 pass; promotes to default-on if all pass.
