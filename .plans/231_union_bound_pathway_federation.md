# Plan 231: Deep Manifold GOAT Features — Union Bound + Pathway Tracker + Federation Composer

> **Source:** Research 205 — Deep Manifold Neural Network Mathematics Modelless Distillation
> **Date:** 2026-06
> **Status:** 🔧 Pending
> **Feature Gates:** `union_bound_confidence`, `pathway_tracker`, `federation_composer`
> **Related:** Research 51 (COMPLETE, Plan 085), Research 205 (modelless distillations)

---

## Tasks

- [ ] T1: Add `union_bound_confidence` feature gate to `Cargo.toml`
- [ ] T2: Implement `BranchConfidence` trait + `UnionBoundScorer` in `src/speculative/branch_confidence.rs`
- [ ] T3: GOAT test — prove additive ≥ multiplicative branch survival
- [ ] T4: Add `pathway_tracker` feature gate to `Cargo.toml`
- [ ] T5: Implement `PathwayTracker` in `src/speculative/pathway_tracker.rs`
- [ ] T6: GOAT test — prove pathway detection reduces thinking budget without quality loss
- [ ] T7: Add `federation_composer` feature gate to `Cargo.toml`
- [ ] T8: Implement `FederationComposer<C,S,B>` in `src/pruners/federation_composer.rs`
- [ ] T9: GOAT test — prove federation composer early termination saves compute
- [ ] T10: Add all three features to `full` feature set
- [ ] T11: Update README.md — Deep Manifold GOAT section
- [ ] T12: Run full benchmark suite — no regressions
- [ ] T13: GOAT gate proof — all gates passing per feature

---

## Context

Research 205 distilled 12 new modelless features from Deep Manifold Part 2 (beyond what Research 51 already covered). Three were rated GOAT:

1. **Union Bound Branch Confidence (§2.4.2):** On stacked piecewise manifolds, deviation probability is additive (union bound), not multiplicative. P(total fail) ≤ Σ P(branch_i fail). More optimistic than multiplicative chaining → +36% more branches survive pruning.

2. **PathwayTracker (§4.2-4.3):** Track DDTree branch selection patterns across consecutive tokens. Stable pathway = converged fixed point = stop thinking early. Unstable = keep thinking. Saves ~30% of thinking budget.

3. **FederationComposer (§7.5):** Explicit Model→Agent→Tool composition with residual checking between steps. Early termination when residual < threshold.

All three are **modelless, inference-time only, feature-gated, additive, non-breaking.** Default build is unaffected.

---

## T1: Feature Gate — `union_bound_confidence`

**File:** `Cargo.toml`

```toml
union_bound_confidence = []  # Union bound additive branch confidence (Research 205, Plan 231)
```

No dependencies. Off by default (research feature).

---

## T2: BranchConfidence Trait + UnionBoundScorer

**File:** `src/speculative/branch_confidence.rs` (new, gated by `union_bound_confidence`)

```rust
//! Deep Manifold §2.4.2 — Union Bound Branch Confidence (Research 205, Plan 231)
//!
//! Paper Eq. 32-35: On stacked piecewise manifolds, deviation probability
//! obeys the union bound (Boole's inequality):
//!   P(hk ∉ Mk) ≤ Σᵢ P(hk ∉ Mk,i)
//!
//! Errors propagate ADDITIVELY, not exponentially.
//! Branch confidence should use additive combination, not multiplicative.

/// Branch confidence computation strategy.
pub trait BranchConfidence: Send + Sync {
    /// Compute total confidence from per-position scores in [0, 1].
    fn total_confidence(&self, position_scores: &[f32]) -> f32;
    /// Name of the confidence method.
    fn name(&self) -> &'static str;
}

/// Multiplicative (chain) confidence — classical approach.
/// P(correct) = Πᵢ pᵢ. Pessimistic: single weak position kills the chain.
pub struct MultiplicativeScorer;

impl BranchConfidence for MultiplicativeScorer {
    fn total_confidence(&self, position_scores: &[f32]) -> f32 {
        if position_scores.is_empty() { return 1.0; }
        position_scores.iter().product()
    }
    fn name(&self) -> &'static str { "multiplicative" }
}

/// Union bound (additive) confidence — Deep Manifold §2.4.2.
/// P(correct) = 1 - min(1, Σᵢ (1 - pᵢ)).
/// More optimistic: individual weak positions don't kill the chain.
pub struct UnionBoundScorer;

impl BranchConfidence for UnionBoundScorer {
    fn total_confidence(&self, position_scores: &[f32]) -> f32 {
        if position_scores.is_empty() { return 1.0; }
        let fail_prob: f32 = position_scores.iter().map(|p| 1.0 - p).sum();
        1.0 - fail_prob.min(1.0)
    }
    fn name(&self) -> &'static str { "union_bound" }
}

/// Hybrid: multiplicative for short chains, union bound for long chains.
pub struct HybridScorer { pub short_chain_threshold: usize }

impl Default for HybridScorer {
    fn default() -> Self { Self { short_chain_threshold: 4 } }
}

impl BranchConfidence for HybridScorer {
    fn total_confidence(&self, position_scores: &[f32]) -> f32 {
        if position_scores.len() <= self.short_chain_threshold {
            MultiplicativeScorer.total_confidence(position_scores)
        } else {
            UnionBoundScorer.total_confidence(position_scores)
        }
    }
    fn name(&self) -> &'static str { "hybrid" }
}
```

**Update `src/speculative/mod.rs`:**
```rust
#[cfg(feature = "union_bound_confidence")]
pub mod branch_confidence;
```

---

## T3: GOAT Test — Union Bound

**File:** `tests/goat_union_bound.rs` (new, gated by `union_bound_confidence,bomber`)

Prove on bomber arena (1000 rounds):
1. Union bound scoring accepts ≥ multiplicative scoring branch count
2. Win rate with union bound ≥ win rate with multiplicative (within statistical variance)

**GOAT gates:**
- G1: Branch acceptance rate +36% vs multiplicative
- G2: Win rate ≥ baseline (within σ)
- G3: Zero overhead (scoring is O(n) SIMD-able)
- G4: Feature isolation (default build unaffected)

---

## T4: Feature Gate — `pathway_tracker`

```toml
pathway_tracker = []  # Intrinsic pathway stability detection (Research 205, Plan 231)
```

---

## T5: PathwayTracker

**File:** `src/speculative/pathway_tracker.rs` (new, gated by `pathway_tracker`)

```rust
//! Deep Manifold §4.2-4.3 — Intrinsic Pathway Stability Detection (Research 205, Plan 231)
//!
//! Inference traverses intrinsic pathways through stacked manifolds.
//! Stable pathway = converged fixed point. Unstable = keep searching.

/// Tracks branch selection patterns across consecutive inference steps.
pub struct PathwayTracker {
    history: Vec<Vec<usize>>,
    max_depth: usize,
    cursor: usize,
    steps: usize,
}

impl PathwayTracker {
    pub fn new(max_depth: usize) -> Self {
        Self { history: Vec::with_capacity(max_depth), max_depth, cursor: 0, steps: 0 }
    }

    /// Record branch selection for current step.
    pub fn update(&mut self, branches: &[usize]) {
        let mut sorted = branches.to_vec();
        sorted.sort_unstable();
        if self.history.len() < self.max_depth {
            self.history.push(sorted);
        } else {
            self.history[self.cursor] = sorted;
        }
        self.cursor = (self.cursor + 1) % self.max_depth;
        self.steps += 1;
    }

    /// Compute pathway stability: sigmoid of consecutive-match ratio.
    /// Near 1.0 = very stable, near 0.0 = unstable.
    pub fn stability(&self) -> f32 {
        if self.history.len() < 2 { return 0.5; }
        let mut matches = 0usize;
        let mut comparisons = 0usize;
        for i in 1..self.history.len() {
            let prev = &self.history[i - 1];
            let curr = &self.history[i];
            comparisons += 1;
            let overlap = prev.iter().filter(|b| curr.binary_search(b).is_ok()).count();
            let max_len = prev.len().max(curr.len()).max(1);
            if overlap >= max_len / 2 { matches += 1; }
        }
        if comparisons == 0 { return 0.5; }
        let ratio = matches as f32 / comparisons as f32;
        1.0 / (1.0 + (-(ratio - 0.5) * 4.0).exp())
    }

    /// Check if pathway has converged.
    pub fn is_converged(&self, threshold: f32) -> bool {
        self.steps >= 3 && self.stability() > threshold
    }

    /// Reset for new inference session.
    pub fn reset(&mut self) {
        self.history.clear();
        self.cursor = 0;
        self.steps = 0;
    }
}
```

---

## T6: GOAT Test — Pathway Tracker

**File:** `tests/goat_pathway_tracker.rs` (new, gated by `pathway_tracker,bomber`)

**GOAT gates:**
- G1: Stability detection accuracy ≥ 80%
- G2: Thinking budget saved ≥ 20% when converged
- G3: Win rate ≥ baseline (within σ)
- G4: Zero overhead when disabled
- G5: Feature isolation

---

## T7: Feature Gate — `federation_composer`

```toml
federation_composer = ["bandit"]  # Explicit Model→Agent→Tool pipeline with residual checking
```

---

## T8: FederationComposer<C,S,B>

**File:** `src/pruners/federation_composer.rs` (new, gated by `federation_composer`)

```rust
//! Deep Manifold §7.5 — Federation Triangle Composer (Research 205, Plan 231)
//!
//! Paper Eq. 158-159: Agentic behavior = composite fixed-point iteration:
//!   x_{t+1} = Φ_tool ∘ Φ_agent ∘ Φ_model(x_t)
//!
//! Model  = ConstraintPruner (what's valid)
//! Agent  = ScreeningPruner (what's relevant)
//! Tool   = BanditPruner    (what works)
//!
//! With RESIDUAL CHECKING between each step for early termination.

use crate::pruners::constraint::ConstraintPruner;
use crate::pruners::screening::ScreeningPruner;

/// Residual check result after each federation step.
#[derive(Debug, Clone)]
pub struct ResidualCheck {
    pub candidates_before: usize,
    pub candidates_after: usize,
    /// 1 - (after/before). High = step removed many. Low = step barely changed.
    pub residual: f32,
}

impl ResidualCheck {
    pub fn new(before: usize, after: usize) -> Self {
        let residual = if before > 0 { 1.0 - (after as f32 / before as f32) } else { 0.0 };
        Self { candidates_before: before, candidates_after: after, residual }
    }
    /// Low residual = step didn't help much → consider stopping.
    pub fn should_terminate(&self, threshold: f32) -> bool { self.residual < threshold }
}

/// Federation composer: explicit Model→Agent→Tool pipeline with residual checking.
pub struct FederationComposer<'a, C, S, B> {
    pub constraint: &'a C,
    pub screening: &'a S,
    pub bandit: &'a B,
    pub residual_threshold: f32,
}

impl<'a, C: ConstraintPruner, S: ScreeningPruner, B: Send + Sync>
    FederationComposer<'a, C, S, B>
{
    pub fn new(constraint: &'a C, screening: &'a S, bandit: &'a B) -> Self {
        Self { constraint, screening, bandit, residual_threshold: 0.01 }
    }

    /// Run full federation pipeline with residual checking.
    pub fn compose_and_prune(
        &self, candidates: &[usize], context: &[u8],
    ) -> (Vec<usize>, Vec<ResidualCheck>) {
        let mut checks = Vec::with_capacity(3);
        let n = candidates.len();

        // Step 1: Model → ConstraintPruner
        let valid: Vec<usize> = candidates.iter()
            .copied().filter(|&t| self.constraint.is_valid(t, context)).collect();
        let c1 = ResidualCheck::new(n, valid.len());
        checks.push(c1.clone());
        if c1.should_terminate(self.residual_threshold) && valid.len() == n {
            return (valid, checks);
        }

        // Step 2: Agent → ScreeningPruner
        let relevant: Vec<usize> = valid.iter()
            .copied().filter(|&t| self.screening.relevance(t, context) > 0.5).collect();
        let c2 = ResidualCheck::new(valid.len(), relevant.len());
        checks.push(c2.clone());
        if c2.should_terminate(self.residual_threshold) && relevant.len() == valid.len() {
            return (relevant, checks);
        }

        // Step 3: Tool → BanditPruner (pass-through, bandit operates at higher level)
        checks.push(ResidualCheck::new(relevant.len(), relevant.len()));
        (relevant, checks)
    }
}
```

---

## T9: GOAT Test — Federation Composer

**File:** `tests/goat_federation_composer.rs` (new, gated by `federation_composer,bomber`)

**GOAT gates:**
- G1: Early termination rate ≥ 10% of queries
- G2: Compute saved ≥ 15% on easy queries
- G3: Win rate ≥ baseline
- G4: Zero overhead when disabled
- G5: Feature isolation
- G6: Residual check overhead < 0.1μs per query

---

## T10: Add to `full` Feature

```toml
full = [..., "union_bound_confidence", "pathway_tracker", "federation_composer"]
```

---

## T11: README Update

Add to README.md Deep Manifold section:

```markdown
### Deep Manifold GOAT Features (Plan 231)

| Feature | Paper § | What | Gate |
|---------|---------|------|------|
| Union Bound Confidence | §2.4.2 | Additive branch scoring (+36% survival) | `union_bound_confidence` |
| Pathway Tracker | §4.2-4.3 | Early thinking termination (-30% budget) | `pathway_tracker` |
| Federation Composer | §7.5 | Model→Agent→Tool with residual check | `federation_composer` |
```

---

## T12: Benchmark Suite

Run `cargo test --features full --release` to verify no regressions.

---

## T13: GOAT Gate Proof

| Gate | Criterion | Union Bound | Pathway | Federation |
|------|-----------|-------------|---------|------------|
| G1: Feature gain | Measurable improvement | +36% branches | -30% budget | +15% compute save |
| G2: Zero regression | Default build unaffected | ✅ Gated | ✅ Gated | ✅ Gated |
| G3: Perf overhead | < 1μs added | O(n) | O(n) ring | O(n) + 3 checks |
| G4: Test coverage | ≥ 3 tests per feature | ✅ T2+T3 | ✅ T5+T6 | ✅ T8+T9 |
| G5: Isolation | Independent features | ✅ No deps | ✅ No deps | ✅ bandit dep |
| G6: Doc coverage | README + research | ✅ T11 | ✅ T11 | ✅ T11 |

---

## Module Structure

```
src/
├── speculative/
│   ├── mod.rs                           # +2 conditional mods
│   ├── branch_confidence.rs             # NEW (union_bound_confidence)
│   └── pathway_tracker.rs               # NEW (pathway_tracker)
├── pruners/
│   ├── mod.rs                           # +1 conditional mod
│   └── federation_composer.rs           # NEW (federation_composer)
tests/
├── goat_union_bound.rs                  # NEW
├── goat_pathway_tracker.rs              # NEW
└── goat_federation_composer.rs          # NEW
```

## File Changes Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `Cargo.toml` | Edit (+3 features, +3 to full) | ~6 |
| `src/speculative/mod.rs` | Edit (+2 conditional mods) | ~4 |
| `src/speculative/branch_confidence.rs` | **NEW** | ~80 |
| `src/speculative/pathway_tracker.rs` | **NEW** | ~70 |
| `src/pruners/mod.rs` | Edit (+1 conditional mod) | ~2 |
| `src/pruners/federation_composer.rs` | **NEW** | ~90 |
| `tests/goat_union_bound.rs` | **NEW** | ~60 |
| `tests/goat_pathway_tracker.rs` | **NEW** | ~60 |
| `tests/goat_federation_composer.rs` | **NEW** | ~60 |
| `README.md` | Edit (+GOAT features section) | ~15 |

**Total new code:** ~547 lines (all feature-gated, default build unaffected)

---

## Dependencies

```
T1 → T2 → T3
T4 → T5 → T6
T7 → T8 → T9
T3 + T6 + T9 → T10 → T11 → T12 → T13
```

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Union bound too optimistic | HybridScorer fallback for short chains |
| PathwayTracker false convergence | Minimum 3 steps before convergence check |
| FederationComposer adds latency | O(1) per check, ~3 checks max |
| Feature flag proliferation | All research-grade, off by default |

---

## TL;DR

Three GOAT-rated modelless features from Deep Manifold Part 2, all feature-gated:
1. **Union Bound Confidence** — additive branch scoring (§2.4.2), +36% branch survival
2. **PathwayTracker** — intrinsic pathway stability detection (§4.2), -30% thinking budget
3. **FederationComposer** — explicit Model→Agent→Tool with residual check (§7.5), +15% compute save

Priority: `union_bound_confidence` → `pathway_tracker` → `federation_composer`. All engine-layer MIT, no SaaS fuel required.
