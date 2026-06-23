# Plan 311 — Alien Sampler GOAT Gate Benchmark

**Date:** 2026-06-23
**Plan:** [katgpt-rs/.plans/311_alien_sampler_primitive.md](../.plans/311_alien_sampler_primitive.md)
**Bench:** `benches/alien_sampler_goat.rs` (`cargo bench --bench alien_sampler_goat --features alien_sampler`)
**Machine:** macOS dev laptop (Apple Silicon). Numbers are wall-clock medians over 2 seeds × 1000 cycles × 100 NPCs.

---

## GOAT Gate — 1/4 PASS → DEMOTE (opt-in, not default)

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** motif collapse | Arm C / Arm B ≤ 0.50 | **0.5010** (β=0.7: Arm C=0.50, Arm B=0.9978) | ❌ BORDERLINE (within 0.2% of threshold) |
| **G2** quality preservation | Arm C / Arm A ≥ 0.90 | **0.6722** (β=0.7: Arm C=0.6553, Arm A=0.9747) | ❌ FAIL |
| **G3** perf | C/B ≤ 5.0× | **38.86×** (Arm C=2890µs, Arm B=74µs) | ❌ FAIL |
| **G4** latent boundary | no Vec<f32> escapes rank() | type-system enforced (ScoredCandidate is Copy POD) | ✅ PASS |

**Decision: DEMOTE.** The module ships as opt-in (`alien_sampler` feature, default-OFF) for paper reproduction and future research. Not promoted to default.

---

## β sweep (G2 recovery attempt per plan decision tree)

The plan says: "G1 PASS but G2 FAIL → Investigate β sweep (try β=0.5, 0.6); if no β satisfies both, demote."

| β | G1 ratio (≤0.50) | G2 ratio (≥0.90) | Concentration | Quality | Verdict |
|---|------------------|------------------|---------------|---------|---------|
| 0.7 | 0.2505 ✅ | 0.3361 ❌ | 0.4999 | 0.6553 | G2 fail |
| 0.5 | 0.2419 ✅ | 0.3812 ❌ | 0.4828 | 0.7431 | G2 fail |
| 0.3 | 0.5011 ❌ | 0.4999 ❌ | 1.0000 | 0.9746 | G1 fail (collapse) |
| 0.2 | 0.5011 ❌ | 0.5000 ❌ | 1.0000 | 0.9747 | G1 fail (collapse) |

**Phase transition at β≈0.4.** Below β=0.4, the availability signal is too weak to overcome the coherence gradient → full motif collapse (concentration=1.0). Above β=0.4, the availability signal dominates → excessive quality sacrifice (quality drops to 0.65-0.74). **No β satisfies both gates simultaneously** on this scenario.

---

## What the result means

### The alien sampler DOES work (mechanism validated)
- At β=0.7, concentration drops from **0.9978 → 0.4999** — a **2× reduction** in motif collapse. The dual-encoder community-availability signal is genuinely more effective than scalar local-redundancy (which stays at 0.9978, essentially fully collapsed).
- The paper's analog was 95.7%→34.3% (2.8× reduction). We see 99.8%→50.0% (2.0× reduction). Same mechanism, slightly weaker effect.
- G4 passes trivially (type-system guarantee).

### Why it fails the gate (scenario limitation, not primitive limitation)
- The synthetic coherence surface has a **single dominant peak** (archetype 0 = the global coherence direction). Alternative archetypes have materially lower coherence by construction.
- This creates a **sharp phase transition**: either the availability signal is too weak (β<0.4 → collapse) or too strong (β>0.4 → excessive quality loss).
- The paper's real-world coherence surface (research-paper quality scores) is presumably **flatter and multi-modal** — multiple "good" research topics with comparable coherence. On such a surface, a moderate β would spread scientists across high-quality alternatives without the quality cliff.
- **Transfer to synthetic NPC populations is unvalidated** — exactly as the plan's risk register predicted ("the paper's evidence is on real research corpora, not synthetic NPC populations — transfer to our domain is unvalidated").

### Perf (G3) is also a fail, but fixable
- Arm C is **38.86× slower** than Arm B. Two causes:
  1. **Bank rebuild cost**: `MedianTopMAvailability::new` clones the bank + recomputes norms. Mitigated by rebuilding only every 10 cycles, but the clone is still O(bank_size × dim) per rebuild.
  2. **Per-candidate cosine cost**: 100 NPCs × 32 pool × 200 bank × 16 dim = 10.2M FMAs per cycle. SIMD (Phase 4) would 4-8× this.
- Even with both fixes, Arm C does fundamentally more work than Arm B (which is just 100 × 32 coherence dots). The G3 target of "≤5×" may be unreachable without SIMD + bank-incremental updates.

---

## What ships

- **`alien_sampler` feature (opt-in, default-OFF).** The open primitive is complete and tested (50 unit tests pass). It's useful for:
  - Paper reproduction (the math is correct).
  - Future research on flatter coherence surfaces where the β tradeoff is more favorable.
  - Consumers that want the dual-encoder signal and can tune β for their domain.
- **NOT promoted to default.** Per the plan's decision tree and the AGENTS.md honesty rule.
- **Phase 4 SIMD deferred.** No point optimizing a primitive that fails the GOAT gate. If a future scenario passes G1+G2, Phase 4 SIMD would address G3.
- **Phase 5 promotion skipped.**

---

## What stays for future work

1. **Better synthetic scenario.** A flatter, genuinely multi-modal coherence surface (N comparable peaks, not one dominant peak) might allow a β that passes both G1 and G2. Filed as a follow-up, not blocking.
2. **Bank-incremental updates.** `MedianTopMAvailability` currently rebuilds on bank change. An incremental norm-update path would cut Arm C's per-cycle cost substantially.
3. **SIMD cosine.** Phase 4 T4.1 — straightforward 4× or 8× speedup on the inner loop.
4. **Real-domain validation.** The paper's evidence is on research corpora. The riir-ai consumer (Plan 312+, out of scope here) would validate on real NPC behavior emissions.

---

## Module-level microbench (Phase 2 T2.2)

For reference, the Phase 2 microbench (separate from the GOAT gate):

| Kernel | Target | Pre-Issue-002 | Post-Issue-002 | Verdict |
|--------|--------|---------------|----------------|---------|
| `rank` 1k candidates (batch path) | ≤ 500µs | 498µs | **456µs** | ✅ PASS (~8% faster) |
| `rank` 10k candidates (batch path) | ≤ 5ms | 5.49ms | **5.10ms** | ❌ FAIL (~7% faster, still 2% over) |
| `median_top_m` bank=100 | ≤ 5µs | 0.42µs | **0.42µs** | ✅ PASS (unchanged) |
| `median_top_m` bank=10k | ≤ 500µs | 35µs | **34µs** | ✅ PASS (unchanged) |

The `median_top_m` kernel is excellent (12-14× under target). The `rank` kernel improved 7-8% from the SoA flat-bank refactor (Issue 002 C1) — better cache locality on the per-candidate bank sweep. The batch path (`availability_batch` + `rank_precomputed`) is the recommended hot-path entry point.

---

## Post-Issue-002 G3 re-measurement (2026-06-23)

Issue 002 (SoA flat bank + SIMD-friendly dot + incremental norms) landed C1/C2/C3 as infrastructure but **did not close G3** on the GOAT bench:

| Metric | Pre-Issue-002 | Post-Issue-002 | Verdict |
|--------|---------------|----------------|---------|
| G3 C/B ratio | 38.86× | **40.22×** | ❌ FAIL (target ≤5×) |
| Arm C per-cycle | 2890µs | 3015µs | ~neutral (within noise) |
| `rank` 1k batch (fixed bank=100) | 498µs | **456µs** | ✅ 8% faster |

**Root cause of G3 non-closure:** the GOAT bench's zone bank grows unboundedly (100 NPCs × 1 selection/cycle × 1000 cycles = up to 100K bank items). Per-cycle cosine work is O(bank_size), which no constant-factor primitive optimization (SoA, SIMD, inv-norms) can overcome. The microbench (fixed bank=100) confirms the primitive itself is 8% faster post-Issue-002; the GOAT G3 failure is a **scenario design issue** (unbounded bank growth), not a primitive speed issue.

**What Issue 002 shipped (useful infrastructure despite G3 non-closure):**
- **C1 SoA flat bank** — cleaner storage, cache-friendly, 8% faster on fixed-size banks.
- **C2 `dot_4acc`** — available but NOT used in the default hot path. Benchmarks showed the 4-accumulator `mul_add` form is slower than sequential `+=` without `target-cpu=native` (no autovec, added register pressure). `dot_seq` (sequential, R1-preserving) is the shipped default; `dot_4acc` is kept for future `target-cpu=native` builds.
- **C3 incremental API** (`from_flat_bank`, `push_bank_items`, `invalidate_norms`) — lets future consumers avoid the clone+rebuild cliff. The GOAT bench is unchanged per Issue 002 spec, so it still uses `new(zone_bank.clone(), M)` every 10 cycles.

**What would actually close G3 (out of scope for Issue 002):**
- **Bank bounding**: cap the zone bank at a max size (e.g. 1000 items) with FIFO/LRU eviction. This is an algorithmic change to the scenario, not a primitive change. Filed as a follow-up.
- **Adopt C3 in the bench**: change the bench to use `push_bank_items` instead of `new(clone)`. Avoids the rebuild clone but doesn't fix the O(bank_size) cosine cost.
- **`target-cpu=native` build**: would enable `dot_4acc` autovec. Not a portable fix (CI builds are generic-target).

---

## TL;DR

**Plan 311 Alien Sampler GOAT gate: 1/4 PASS → DEMOTE to opt-in.** G1 is borderline (0.5010, within 0.2% of the 0.50 threshold), G2 fails (0.67 vs 0.90 target — β sweep shows a sharp phase transition with no β satisfying both gates), G3 fails (38.86× slower than baseline), G4 passes (type-system). The dual-encoder mechanism IS validated — concentration drops 2× vs the scalar-redundancy baseline — but the synthetic scenario's single-peak coherence surface creates an unfavorable quality/diversity tradeoff that no β can resolve. **The module ships as opt-in for paper reproduction and future research; not promoted to default.** Phase 4 SIMD deferred. This matches the plan's most-likely failure mode ("the paper's evidence is on real research corpora, not synthetic NPC populations — transfer to our domain is unvalidated").
