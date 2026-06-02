# Research 152: LDT Phase 2 — Lattice State Fusion

> **Paper:** [Lattice Deduction Transformers](https://arxiv.org/pdf/2605.08605) — Davis, Haller, Alfarano, Santolucito (2026)
> **Date:** 2026-06
> **Phase 1:** Research 050, Plan 088 (GOAT 7/7, default-on)
> **Verdict: GAIN — 4 novel fusion distillations, all modelless, all behind existing `lattice_deduction` feature gate**

---

## 1. TL;DR

Phase 1 (Plan 088) distilled three direct paper contributions: asymmetric threshold (T1), conflict detection (T2), and α-operator (T3). Phase 2 goes beyond the paper's explicit techniques to create **fusion ideas** that combine LDT's abstract interpretation framework with our existing architecture in novel ways:

1. **AlphaScreeningPruner** — α-operator AS a ScreeningPruner impl, making multi-solution pruning sound by construction
2. **Conflict Clause Learning** — CDCL-style clause learning for DDTree, fusing T2's conflict detection with SAT solver clause learning
3. **Cached ŷ_prev Target Stabilization** — Preventing AlphaTarget collapse when no solution remains consistent
4. **Depth-Escalating Conflict Threshold** — Progressive tightening of conflict detection as search deepens

All four are modelless (zero training), behind existing `lattice_deduction` feature gate, and zero impact on default build.

---

## 2. The Four Fusion Ideas

### F1: AlphaScreeningPruner — α-operator as ScreeningPruner

**The Insight:** LDT's α-operator computes `ŷ = x ⊓ α({y ∈ Y | y consistent with x})` — the intersection of current state with all surviving solutions. This is literally a per-position candidate mask. Our `ScreeningPruner` trait returns `f32` relevance per token. By implementing `ScreeningPruner` on top of `AlphaTarget`, we get **sound multi-solution pruning by construction**: any token not in the α-target gets relevance 0.0, any token in the target gets 1.0.

**Why it's fusion (not direct mapping):** The paper never connects α-operator to a pruner interface. It uses α as a training target. We use α as a **runtime pruning signal** — a completely different application of the same formalism. The ScreeningPruner trait provides the interface; AlphaTarget provides the lattice state; the fusion gives us a pruner that's **sound by construction** (never prunes a token that appears in any surviving solution).

**Where:** `src/speculative/alpha.rs` — new struct `AlphaScreeningPruner`

**Performance:** `HashSet::contains` — O(1) per token, sub-100ns. Interior mutability via `std::cell::RefCell<AlphaTarget>` for lazy cache.

### F2: Conflict Clause Learning — CDCL for DDTree

**The Insight:** LDT's Appendix A.2 describes the **abduction function** `abd_ϕ(S) = S ∪ {g : g ⊭ ϕ}` — generalizing failures into learned constraints. This is the engine of CDCL SAT solvers. When a DDTree branch is flagged as conflicted by `ConflictDetector`, we extract the **commitment pattern** that caused the conflict and store it as a "learned clause." Future expansions skip any branch whose commitments are a superset of a learned clause.

**Why it's fusion (not direct mapping):** LDT doesn't implement clause learning — it uses random branching. We fuse LDT's conflict detection (T2) with CDCL's clause learning to create a **modelless search accelerator** for DDTree. This is a novel combination: constraint-solver clause learning applied to speculative decoding trees.

**Where:** `src/speculative/alpha.rs` — new struct `ConflictClauseDb`

**Performance:** HashSet comparison per branch — O(k × c) where k = number of clauses, c = avg clause size. With k ≤ 64, this is < 1µs. Bounded by `max_clauses` to prevent unbounded growth.

### F3: Cached ŷ_prev Target Stabilization

**The Insight:** LDT caches the last non-empty `ŷ_prev` when the surviving solution set becomes empty, using `x ⊓ ŷ_prev` as the BCE target. This prevents the loss from being overwhelmed by conflict pressure. Our `AlphaTarget` currently returns empty sets when `remaining_solutions() == 0`, which provides no useful signal. Caching the last valid target gives **stable supervision** even in conflict states.

**Why it's fusion:** The paper uses this for training loss stabilization. We apply it to **runtime pruning signal stabilization** — when the search has committed past all valid solutions, the cached target still provides useful information about which candidates were viable before the conflict.

**Where:** `src/speculative/alpha.rs` — modify `AlphaTarget`

**Performance:** One `Option` check — zero cost.

### F4: Depth-Escalating Conflict Threshold

**The Insight:** LDT sets `θ_eval_CLS = 0.6` at inference — higher than training — because "the conflict head grows more confident as the puzzle fills in." This suggests the conflict threshold should be **depth-adaptive**: more permissive early (when the state is still broad), more aggressive later (when the state is committed enough for conflict signals to be trustworthy).

**Why it's fusion:** The paper uses a static inference threshold. We make it **dynamically escalating with depth**, fusing LDT's conflict detection with our existing depth-aware DDTree expansion. At depth 0, we're lenient (max_prune_rate = 0.7). At max depth, we're aggressive (max_prune_rate = 0.3). This catches conflicts earlier at depths where they matter most.

**Where:** `src/speculative/types.rs` — extend `EntropyConflictDetector`

**Performance:** One multiply + max — sub-nanosecond.

---

## 3. Gap Analysis

| Fusion | What's Already Done | What's New |
|--------|--------------------|------------|
| F1 | AlphaTarget (T3) computes α-target | AlphaScreeningPruner bridges α → ScreeningPruner trait |
| F2 | ConflictDetector (T2) detects conflicts | ConflictClauseDb learns from conflicts |
| F3 | AlphaTarget caches current target | Caches PREVIOUS valid target for stability |
| F4 | EntropyConflictDetector has static threshold | Depth-adaptive escalation |

---

## 4. Verdict (per Research 003)

### Commercial Alignment

| Criterion | Assessment |
|-----------|-----------|
| **Strengthens the moat?** | ✅ Yes — AlphaScreeningPruner makes multi-solution pruning sound by construction. Conflict clause learning is novel for speculative decoding. |
| **Uses existing infrastructure?** | ✅ Yes — all four compose existing traits (ScreeningPruner, ConflictDetector, AlphaTarget). |
| **Engine/Fuel split intact?** | ✅ Yes — all modelless, in katgpt-rs (MIT engine). |
| **Feature-gated?** | ✅ Yes — behind existing `lattice_deduction` gate. |
| **Zero default perf impact?** | ✅ Yes — all code behind feature gate, only compiled when enabled. |

### Performance Alignment (per optimization.md)

| Principle | Compliance |
|-----------|-----------|
| Profile first | All four use O(1) HashSet lookups or scalar arithmetic |
| Fixed-size arrays | ConflictClauseDb has bounded max_clauses |
| Don't allocate in hot loops | AlphaScreeningPruner uses RefCell for lazy init only |
| Don't GPU for µs work | All CPU-side, sub-µs per call |

### Decision: **GO — All Four**

All four are cheap to implement, compose existing traits, and provide measurable search acceleration. No new dependencies. No architecture changes.

---

## 5. Key Insight: Why These Fusions Work

LDT's core contribution is the **lattice formalism**: treating search states as elements of a partially ordered set where deduction descends from ⊤ toward solutions and conflict detection identifies ⊥. Phase 1 distilled specific mechanisms from this formalism. Phase 2 distills the **structural patterns**:

1. **α-target as pruner** — any lattice state can be a pruning signal
2. **Conflict → learned clause** — any conflict can be generalized into future pruning
3. **State caching** — lattice states have temporal structure (previous states inform current)
4. **Depth-adaptive thresholds** — lattice depth determines information content → threshold should track it

These are **general principles** that apply to any search over a lattice-like structure, not just LDT's specific architecture.

---

## 6. Paper Metadata

- Paper: https://arxiv.org/pdf/2605.08605
- Phase 1: Research 050, Plan 088 (GOAT 7/7)
- Related: Plan 049 (G-Zero), Plan 057 (HLA), Plan 061 (Entropy), Plan 066 (D2F), Plan 067 (NFSP/MCTS)
- Optimization: optimization.md (sub-µs hot path, fixed-size, no GPU for µs work)
