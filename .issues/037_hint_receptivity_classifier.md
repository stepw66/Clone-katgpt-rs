# Issue 037 — HintReceptivity Classifier / `Solver::hint_receptivity()` Trait Method

**Filed:** 2026-07-03
**Priority:** P3 (future-proofing — not load-bearing for any shipped solver today)
**Origin:** Research 372 (arXiv 2607.02491 — G-RRM, Bertram et al. §2.3 / §3)
**Related:** `.research/372_G_RRM_Guiding_Symbolic_Solvers_Pass.md` §2.3

## Problem

G-RRM's empirical contribution (§3) is that hint-injection efficacy is
**solver-architecture-dependent**:

| Solver | Regime | Hint effect |
|---|---|---|
| Backtracking | search-dominated | large speedup (up to 33.3×) |
| Glucose 4.1 (CDCL) | search-dominated | large speedup |
| CaDiCaL 3.0.0 (CDCL) | **overhead-dominated** | **no benefit, 0.896× mean slowdown** |

CaDiCaL strictly honors faulty hints (no early-abort on contradiction) and has
a fixed startup overhead that dominates the search savings — so injecting the
`Ŷ` branching-order prior is a net negative for that solver class.

Our `DifficultyFilter::admit`
(`crates/katgpt-core/src/cgsp/traits.rs:229`) only drops candidates at the
solve-rate extremes (`solve_rate = 0.0` or `1.0`). It has **no concept of**
"this downstream `Solver` is overhead-dominated — skip hint injection
entirely." The `Solver` trait (`cgsp/traits.rs:92`) exposes only `attempt()`
— there is no introspection of solver architecture.

## Root Cause

The CGSP loop assumes all `Solver` impls are hint-receptive. This holds today
because every shipped solver is search-dominated:

- Custom backtracking (`katgpt-pruners`)
- DDTree speculation (`src/speculative/alpha.rs` — CDCL-*inspired* unit
  propagation, but not a real CDCL solver)

No real CDCL SAT backend (cadical / glucose bindings) is shipped — `grep` for
`cadical|glucose|sat_backend|cdcl` returns only comment-level mentions in
`alpha.rs` (L265, L307). So the assumption is currently correct, but it is
**invisible and unenforced** — a future SAT-backend integration would silently
regress by routing hints into an overhead-dominated solver with no gate.

## Proposed Fix (deferred — not blocking)

Add a solver-side hint-receptivity classification:

```rust
/// How much a `Solver` benefits from injected branching-order hints.
///
/// Distilled from G-RRM §3 (arXiv 2607.02491): overhead-dominated solvers
/// (cadical3) see a net slowdown from hints; search-dominated solvers
/// (backtracking, glucose4) see large speedups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HintPolicy {
    /// Hints only reorder the search — never injected as hard constraints.
    /// Correct for our custom backtracking + DDTree speculation.
    #[default]
    OrderOnly,
    /// Hints seed phase initialization (warm-start), then fall back to
    /// order-only. Future hook for warm-start solvers.
    PhaseInit,
    /// Skip hint injection entirely — this solver is overhead-dominated
    /// (e.g. a future cadical/glucose binding with fixed startup cost).
    Skip,
}

pub trait Solver {
    // ... existing attempt() ...

    /// Whether this solver benefits from injected hints. Default
    /// `OrderOnly` matches all shipped solvers (search-dominated).
    #[inline]
    fn hint_receptivity(&self) -> HintPolicy {
        HintPolicy::OrderOnly
    }
}
```

The CGSP dispatch loop would then consult `solver.hint_receptivity()` before
routing the `Ŷ` prior into the attempt, short-circuiting to `Skip` when the
solver declines hints.

### Why a default method, not a separate trait

A default method keeps the change **non-breaking** — every existing `Solver`
impl inherits `OrderOnly` (the hint-receptive default matching today's
behavior). A separate `HintReceptive` trait would require touching every impl
site for no behavioral gain.

## Tasks

- [ ] Add `HintPolicy` enum to `crates/katgpt-core/src/cgsp/traits.rs` (or a
      `types.rs` per the AGENTS.md decoupling rule)
- [ ] Add `fn hint_receptivity(&self) -> HintPolicy` default method to `Solver`
- [ ] Wire the dispatch loop in the CGSP runner to consult `hint_receptivity()`
      and short-circuit hint injection on `Skip`
- [ ] Add a unit test: a `Skip`-policy stub solver receives no hint routing
- [ ] Benchmark: confirm no regression on shipped solvers (default `OrderOnly`
      path is a no-op compared to today)

## Deferral Rationale

P3 because:
1. **No overhead-dominated solver ships today.** The gate is a no-op for
   every existing `Solver` impl — all return the default `OrderOnly`.
2. **No measurable Gain exists yet.** G-RRM's speedup numbers are on real
   CDCL SAT backends we don't have. A GOAT gate cannot be run without a
   target solver to measure against.
3. **The hook is only relevant if/when a SAT backend (cadical/glucose
   bindings) is added to the roadmap.** Until then this is pure
   future-proofing documentation.

Reopen as P2 the day a real CDCL SAT backend is wired in — at that point the
0.896× slowdown finding becomes directly measurable and the GOAT gate becomes
runnable.

## TL;DR

G-RRM §3 shows hint injection helps search-dominated solvers (backtracking,
glucose4) but *hurts* overhead-dominated ones (cadical3, 0.896× slowdown). Our
`DifficultyFilter`/`Solver` traits can't express this distinction. Fix: add a
`hint_receptivity() -> HintPolicy { OrderOnly | PhaseInit | Skip }` default
method on `Solver`. Deferred P3 — no overhead-dominated solver ships today, so
the gate is a no-op until a real SAT backend lands.
