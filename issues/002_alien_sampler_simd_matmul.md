# Issue 002: Alien Sampler SIMD + GEMM Perf Optimization

> **Type:** Optimization (perf-only — addresses Plan 311 G3 gate, NOT G1/G2)
> **Status:** Open — needs gain proof before any feature-default consideration
> **Owner:** unassigned
> **Created:** 2026-06-23
> **Cross-repo:** lands in katgpt-rs only (open primitive). No riir-ai change.
> **Origin:** Plan 311 GOAT gate — G3 fail (38.86× slower than baseline).
> **References:** [Plan 311](../.plans/311_alien_sampler_primitive.md) · [GOAT bench](../.benchmarks/311_alien_sampler_goat.md) · [Issue 001](./001_hla_windowed_eigenbasis_recovery.md) (same perf-gate pattern)

---

## TL;DR

Plan 311's `alien_sampler` primitive shipped opt-in because the GOAT gate failed 3/4 (only G4 passed). **G3 is the perf gate**: Arm C (AlienSampler) is **38.86×** slower per cycle than Arm B (scalar local-redundancy), against a target of ≤5×. This issue tracks the SIMD + matmul pass that would close G3.

**Important scope discipline (per AGENTS.md honesty rule):** G3 is *perf only*. It does NOT unblock promotion. G1 (borderline 0.5010 vs 0.50) and G2 (0.67 vs 0.90) are coherence-surface problems, not math-speed problems — the β sweep in the GOAT bench proves no β satisfies both on the current single-peak scenario. Closing G3 makes the primitive fast enough to *consider* for warm paths; it does not make it default-on material. A separate plan (multi-peak coherence scorer, TBD number) is required to address G1+G2.

---

## Current state (what's slow)

Two bottlenecks identified in `.benchmarks/311_alien_sampler_goat.md`:

### Bottleneck 1 — per-candidate cosine, nested loops

Per cycle on the GOAT scenario:
```
100 NPCs × 32 pool × 200 bank × 16 dim = ~10.24M FMAs
```
Done as 100 × 32 separate `availability_embedded_with_scratch` calls, each walking 200 bank rows × 16 dims in a scalar loop. Cache-unfriendly: `community_bank: Vec<Vec<f32>>` is AoS (each bank row is its own heap allocation), so each candidate sweep strides across 200 disjoint allocations.

File: `katgpt-rs/src/alien_sampler/median_top_m.rs` — `MedianTopMAvailability` struct (line ~55).

### Bottleneck 2 — bank rebuild on every `MedianTopMAvailability::new`

Construction clones the entire bank + recomputes all L2 norms. Mitigated in the bench to rebuild only every 10 cycles, but each rebuild is still `O(bank_size × dim)` — and `bank_norms: Vec<f32>` is recomputed from scratch rather than incrementally updated as the bank grows.

---

## Proposed changes

### C1 — Bank as SoA flat slice, not `Vec<Vec<f32>>`

Replace `community_bank: Vec<Vec<f32>>` with a flat `Vec<f32>` of shape `(bank_size, dim)` row-major, plus a `bank_norms: Vec<f32>` slice. Public API on `MedianTopMAvailability` stays the same; the storage layout is an internal refactor.

Rationale: one contiguous allocation → cache-resident during the per-candidate sweep, and a prerequisite for both auto-vectorization and explicit SIMD.

### C2 — GEMM the whole cosine matrix per (NPC, cycle)

Currently: 32 separate GEMVs (`candidate[1×16] × bank[200×16]^T → [200]`).
Proposed: one GEMM per NPC per cycle (`pool[32×16] × bank[200×16]^T → [32×200]`), then `select_nth_unstable` per row to get top-m, then median.

Two implementation tiers:

- **Tier A (portable, no new dep):** hand-rolled blocked loop with `std::simd` (`f32x8`) or autovec-friendly chunked loop. Targets ~4-8× on the inner FMAs alone. Fits katgpt-rs's "no heavy numeric dep" convention (matches `newton_schulz.rs`, `off_principal.rs` pattern).
- **Tier B (katgpt-core LatCalMatrix, IF it lands):** if a fixed-size POD matrix type ships in `katgpt-core` (the AGENTS.md references `LatCalMatrix` / `LatCalWalletExt` but no such struct currently exists in the repo — verify before depending), route the cosine GEMM through it. Zero-alloc, fixed-layout, BLAKE3-committable. This is the "lattice calculus" angle — same matmul, but with deterministic-fixed-layout semantics that would also future-proof any sync-boundary commitment of derived scalars.

Decision rule: ship Tier A first. Promote to Tier B only if/when LatCalMatrix exists AND G3 still needs more headroom.

### C3 — Incremental bank norm updates

Instead of recomputing `bank_norms` from scratch on every `new()` (or every 10-cycle rebuild), expose:
```rust
impl MedianTopMAvailability {
    pub fn push_bank_item(&mut self, items: &[&[f32]]);
    pub fn invalidate_norms(&mut self);  // optional, for full rebuilds
}
```
Each push is `O(items × dim)`, not `O(bank × dim)`. For the GOAT scenario's "append as NPCs emit" pattern, this removes the periodic rebuild cliff entirely.

---

## GOAT / Gain proof — REQUIRED

Per AGENTS.md: "Dont defer benchmark task." This issue is **not done** until the G3 re-measurement is committed.

### Bench setup

Reuse `benches/alien_sampler_goat.rs` unchanged. The three arms (A/B/C) must produce statistically identical diversity/quality numbers (within seed noise) to before — this is a perf-only change, the math must be bit-equivalent on the ranking output. Re-run on the same machine (Apple Silicon dev laptop) with the same seed/cycle counts (2 seeds × 1000 cycles × 100 NPCs) so numbers are directly comparable.

### Gates

| Gate | Metric | Threshold | Notes |
|------|--------|-----------|-------|
| **G3** perf | Arm C / Arm B per-cycle wall time | ≤ **5.0×** | Original Plan 311 target. Currently 38.86×. |
| **G3★** optimal | Same as G3 | ≤ **3.0×** | Stretch — would make C viable for plasma-tier paths if G1+G2 ever pass. |
| **R1** correctness | Bit-identical ranking output vs pre-optimization on the same seeds | 0 diffs at `ScoredCandidate` level (idx + score round-trip) | Mandatory — perf change must not move scores. |
| **R2** microbench | `rank` 10k candidates (Phase 2 T2.2 bench) | ≤ **5.0 ms** | Currently 5.49ms (was already 10% over). SIMD should close this too. |
| **R3** microbench | `median_top_m` bank=10k | ≤ **500 µs** | Currently 35µs (already 14× under). Sanity — must not regress. |

### Outcome matrix

- **G3 + R1 + R2 all pass** → **Gain**. Update `.benchmarks/311_alien_sampler_goat.md` with the new G3 number. Module stays opt-in (G1+G2 still block default). Note in Plan 311 Phase 4 that T4.1 is DONE.
- **G3 fails but R1 passes** → **Partial**. Profile, document the residual bottleneck, file as a follow-up sub-issue. Keep the layout refactor + incremental norms if they help even without SIMD closure.
- **R1 fails (ranking changed)** → **Revert**. Perf is unacceptable if it moves math. Investigate numeric-order instability in the SIMD reduction.

**Promotion to default still requires G1+G2** — those are out of scope for this issue and need the multi-peak coherence plan (TBD).

---

## Where to implement

| Layer | File | Notes |
|---|---|---|
| Storage refactor | `katgpt-rs/src/alien_sampler/median_top_m.rs` | C1: `Vec<Vec<f32>>` → flat `Vec<f32>` SoA |
| SIMD kernel | `katgpt-rs/src/alien_sampler/median_top_m.rs` (or new `simd.rs` submodule if >200 LOC) | C2 Tier A: `f32x8` cosine GEMM |
| Incremental norms | `katgpt-rs/src/alien_sampler/median_top_m.rs` | C3: `push_bank_items` + `invalidate_norms` |
| Feature flag | none (no new feature — `alien_sampler` already exists) | Stays opt-in |
| Bench | `katgpt-rs/benches/alien_sampler_goat.rs` (unchanged) + `katgpt-rs/benches/alien_sampler_bench.rs` (Phase 2 microbench, unchanged) | Re-run, do not modify scenarios |
| Bench report | `katgpt-rs/.benchmarks/311_alien_sampler_goat.md` | Update G3 row + add "Post-SIMD" section |

---

## Latent vs raw boundary

Per AGENTS.md sync-boundary rule:

- **Stays latent / local:** cosine matrix `S[32×200]`, bank embeddings, per-candidate top-m slices. All `f32`, all scratch, never escaped.
- **No change to G4.** The public `rank()` / `rank_into()` / `rank_precomputed()` API already returns `ScoredCandidate` (Copy POD). The SoA refactor must not leak `Vec<f32>` into any public signature.
- **If Tier B (LatCalMatrix) is adopted later:** the matrix type itself is latent-only by design (fixed-layout deterministic arithmetic, not a sync primitive). Do NOT route cosine outputs through `LatCalFixed` i64 commitment — those are for raw scalars crossing the sync boundary, not for f32 latent math.

---

## Risks

1. **SIMD reduction non-determinism.** f32 SIMD reductions can reorder FMAs, producing different rounding vs the scalar loop. **Mitigation:** gate behind a benchmark that requires R1 (bit-identical ranking). If different FMA order moves scores, accept it only if the *ranking order* is unchanged and update the test tolerance explicitly with a comment explaining why.
2. **`Vec<Vec<f32>>` → flat layout breaks the public constructor.** **Mitigation:** keep `MedianTopMAvailability::new` accepting `Vec<Vec<f32>>` (or `&[Vec<f32>]`) and flatten internally. Add a new `from_flat_bank(bank: Vec<f32>, dim: usize, m: usize)` constructor for hot-path callers.
3. **`select_nth_unstable` per row may dominate after GEMM is fast.** With GEMM at ~1µs/NPC, 32 partial sorts of 200 elements may become the new bottleneck. **Mitigation:** if profiled as such, switch to a fixed-size min-heap of size `m` per row (already noted in Plan 311 risk register).
4. **Tier B (LatCalMatrix) depends on a struct that doesn't exist yet.** **Mitigation:** Tier A is shippable without it; do not block on Tier B.

---

## Acceptance

- [x] C1: bank storage refactored to flat SoA, public API preserved (`new` signature unchanged; `bank()` now returns `&[f32]` flat — no external callers used the old `&[Vec<f32>]` form).
- [~] C2 Tier A: `dot_4acc` (4-accumulator `mul_add`) ships but is **NOT used in the default hot path**. Benchmarks showed it slower than sequential `+=` without `target-cpu=native` (no autovec, added register pressure → 48.81× vs 40.22× sequential vs 38.86× pre-issue). `dot_seq` is the shipped default; `dot_4acc` is kept for future `target-cpu=native` builds.
- [x] C3: incremental bank norm update path ships (`from_flat_bank`, `push_bank_items`, `invalidate_norms`).
- [x] R1: existing test suite passes (63 tests). Run-to-run determinism preserved. Cross-version bit-identical scores NOT guaranteed (mul_add form kept available but not default; sequential default preserves original semantics).
- [ ] **G3 re-measured: 40.22× (target ≤ 5.0×).** ❌ NOT CLOSED. Root cause: GOAT bench bank grows to ~100K items (100 NPCs × 1000 cycles), making per-cycle cost O(bank_size). No constant-factor primitive optimization can overcome linear bank growth. Microbench (fixed bank=100) confirms the primitive is 8% faster post-Issue-002 — the G3 failure is a scenario design issue, not a primitive speed issue.
- [ ] R2 microbench: `rank` 10k = 5.10ms (target ≤ 5.0ms). ❌ Improved 7% from 5.49ms but still 2% over.
- [x] `.benchmarks/311_alien_sampler_goat.md` updated with post-Issue-002 G3 number + root cause analysis.
- [ ] Plan 311 Phase 4 T4.1 + T4.3: NOT marked DONE (G3 not closed). Stays as "MOVED to Issue 002".
- [x] Commit on `develop` with `perf:` prefix per AGENTS.md.

### Outcome verdict (honest)

**Partial — infrastructure lands, G3 not closed.** Issue 002 shipped useful infrastructure (SoA storage, C3 incremental API, microbench 8% faster) but did NOT achieve its primary goal (G3 ≤ 5×). The root cause is structural: the GOAT bench's unbounded bank growth makes G3 unreachable by primitive optimization alone. Per AGENTS.md honesty rule, this is documented rather than hidden.

**What would actually close G3 (filed as follow-up, not in this issue):**
1. **Bank bounding** — cap the zone bank at a max size with FIFO/LRU eviction. Algorithmic scenario change.
2. **Adopt C3 in the bench** — use `push_bank_items` instead of `new(clone)`. Avoids rebuild clone but doesn't fix O(bank_size) cosine cost.
3. **`target-cpu=native` builds** — would enable `dot_4acc` autovec (4× inner-loop speedup). Not portable for CI.

**Explicitly NOT in scope:**
- Multi-peak coherence scorer (G1+G2 fix) — separate plan.
- Promotion of `alien_sampler` to default — blocked on G1+G2 (and now G3).
- Tier B LatCalMatrix wiring — separate issue, blocked on LatCalMatrix existing.
- riir-ai consumer (`cgsp_runtime/alien_bridge.rs`) — separate plan in riir-ai.
- Bank-bounding the GOAT scenario — separate issue (scenario change, not primitive change).

---

## TL;DR of the TL;DR

Plan 311's `alien_sampler` failed G3 (38.86× slower than scalar baseline, target ≤5×). This issue attempted three perf changes: (1) flatten the bank from `Vec<Vec<f32>>` to SoA for cache/SIMD, (2) 4-accumulator `mul_add` dot for autovec, (3) incremental bank norm updates. **Result: infrastructure landed (C1+C3 done, microbench 8% faster), but G3 NOT closed (40.22× post-issue vs 38.86× pre-issue).** Root cause: the GOAT bench's zone bank grows to ~100K items, making per-cycle cost O(bank_size) — no constant-factor primitive optimization can overcome linear bank growth. The C2 `dot_4acc` (4-accumulator FMA) was slower than sequential without `target-cpu=native` and is kept available but not default. **The real G3 fix is bank-bounding (scenario change), filed as a follow-up. Module stays opt-in — G1+G2 still block promotion independently of G3.**
