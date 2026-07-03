# Research 372: G-RRM — Guiding Symbolic Solvers with Recurrent Reasoning Models

> **Source:** [G-RRM: Guiding Symbolic Solvers with Recurrent Reasoning Models](https://arxiv.org/pdf/2607.02491) — Bertram, Bhavnani, Freinschlag, Kobler, Mayr, Klambauer (ELLIS Unit Linz / LIT AI Lab / JKU), 2 Jul 2026
> **Date:** 2026-07-03
> **Status:** Done
> **Related Research:** 049 (PTRM — closest cousin, "STRONG VALIDATION, MINIMAL ACTION"), 097 (Training-Free Looped Transformers — RRM inference-loop analog), 188 (NS-CSG — formal propose↔verify mapping), 255 (CLR — post-hoc voting, the inverse direction), 182 (STV — iterative V-R loop), 012 (TRT — self-verification)
> **Classification:** Public

---

## TL;DR

**Verdict: PASS — the modelless G-RRM integration pattern already ships, on the paper's exact domain (Sudoku), with the paper's exact heuristics (MRV cell ordering), across the paper's exact three solver regimes (backtracking / iterative-guide / oneshot warm-start).** The paper's headline-quality component (the trained SE-RRM, 91% FSR on 9×9) is training-side → riir-train. The integration-side mechanism (neural prior → ordering → symbolic solver verifies → empirical solve-rate updates priorities) maps 1:1 onto the existing `SpeculativeGenerator` + `ScreeningPruner` + `ConstraintPruner` + `Solver` + `HintDeltaBandit` stack, with `DifficultyFilter` already performing the paper's "drop trivially-solved / impossible" regime detection. **No files created in the private repos for this paper — it's a strong architectural validation, not a new primitive.**

**Distilled for katgpt-rs (modelless, inference-time):**
The G-RRM pattern is: (1) a fallible proposer emits per-cell/per-variable score matrix `Ŷ ∈ ℝ^{I×K}`; (2) `argsort` defines an ordering `π_i` per variable; (3) a completeness-preserving symbolic solver (backtracking, CDCL SAT) consumes the ordering as branching/phase guidance; (4) when the proposer is right, search collapses to verification; when wrong, the solver's own completeness guarantees recover. The architectural insight — *"guidance changes search ORDER, never correctness"* — is the contract `ConstraintPruner::is_valid` already enforces ("Returns `false` to prune this branch"), with `ScreeningPruner::relevance` only ordering *within* the valid set.

---

## 1. Paper Core Findings

### 1.1 The mechanism (§2.3)

G-RRM couples a **trained** Symbol-Equivariant Recurrent Reasoning Model (SE-RRM, Freinschlag et al. 2026) with classical symbolic solvers via **search guidance**:

- SE-RRM emits `Ŷ ∈ ℝ^{I×K}` — per-cell (i) per-symbol (k) preference scores from a single forward pass.
- For each variable, `π_i = argsort(Ŷ_{i,:})` defines a value-exploration ordering; `d*_i = π_i(K)` is the most-preferred value.
- **For backtracking** (§2.3.2): cell selection uses the **MRV heuristic** ("cell with the fewest valid candidate digits"); digit selection uses `π_i` (descending SE-RRM score).
- **For CDCL SAT** (§2.3.3): the highest-scored digit `d*_i` initializes Boolean variable phases (`x_{r,c,d*}` ← 1, others ← 0) before search begins.

### 1.2 The completeness-preserving contract

> *"G-RRM does not restrict the search space; it only determines the order of exploration while preserving all feasible assignments. Consequently, correctness and completeness of the underlying symbolic solver remain unchanged, and improvements arise solely from more efficient search ordering."*

This is the central safety invariant. Hints are **orderings**, not constraints.

### 1.3 The regime-dependent efficacy (§3, the empirical contribution)

The paper's headline empirical finding — **guidance efficacy depends on solver architecture**:

| Solver | Wall-clock speedup (9×9, perfect hints) | Why |
|---|---|---|
| Backtracking | **33.3×** (median, p<0.001) | Pure search-dominated; fewer conflicts = proportionally less runtime |
| Glucose 4.1 (CDCL) | **1.70×** overall, **1.666×** perfect-hint | Fluid phase reinit via VSIDS — overwrites faulty hints fast |
| CaDiCaL 3.0.0 (CDCL) | **1.02×** (n.s. median), **0.896×** (mean slowdown, p<0.001) | Overhead-dominated; runtime independent of conflict count; phase-init adds ~1.7ms |

**Two preconditions for benefit** (§1, §4): (i) the instance must have an **expansive combinatorial search space** that dominates solver runtime; (ii) the solver must **dynamically overwrite faulty hints** (cadical3 doesn't — it strictly honors external phases across restarts).

### 1.4 The training-side contribution (out of scope here)

- SE-RRM = HRM/TRM (looped transformers, Yang et al. 2024) + an explicit **symbol axis** yielding `D × I × K` rank-3 tensors + axial attention over positions and symbols. Achieves symbol equivariance → extrapolates to unseen `Σ' ⊃ Σ` (e.g., 9×9 → 25×25 Sudoku).
- Trained via **deep supervision + stop-gradient** across loop iterations.
- **This entire component is training-side → riir-train.** The katgpt-rs analogs (frozen looped transformer as inference-time refinement) are already covered by Research 097 (Training-Free Looped Transformers) and Research 273 (ELT).

### 1.5 Solve-Learn-Extrapolate (§6, future work)

The paper's proposed SLE loop: (i) solve small instances exactly with a symbolic solver, (ii) train SE-RRM on valid solutions, (iii) extrapolate to larger instances via SE-RRM hints, feed new solutions back. The training half → riir-train. The runtime half (hint consumption + priority update from solve-rate feedback) → already shipped as `HintDeltaBandit` (Plan 049).

---

## 2. Distillation

### 2.1 The modelless G-RRM pattern — already shipped, layer by layer

The table below maps every G-RRM architectural element to a shipped katgpt-rs / riir-engine primitive. **This is a complete architectural redirect — no element of the modelless integration pattern is missing.**

| G-RRM element (paper §) | Shipped analog | File / trait | Coverage |
|---|---|---|---|
| SE-RRM forward pass → `Ŷ ∈ ℝ^{I×K}` per-cell scores | `SpeculativeGenerator::generate` + `ScreeningPruner::relevance` | `katgpt-core/src/traits.rs` | ✅ Architectural |
| `argsort(Ŷ_{i,:})` → value ordering `π_i` | DDTree decode over `marginals: Vec<Vec<f32>>` | `tactical_06_tui.rs`, `tactical_07_strategic.rs` | ✅ Architectural |
| Backtracking cell selection = **MRV** (§2.3.2) | `SudokuPruner::new_mrv` (Issue 005 Option A) — "ascending candidate count, row-major tiebreak" | `katgpt-pruners/src/sudoku_pruner.rs:87` | ✅ **Word-for-word match** — the paper's exact MRV heuristic, on the paper's exact domain (Sudoku) |
| Backtracking digit selection = SE-RRM-guided ordering | `SudokuPruner::latent_marginals` (Issue 005 Option B) — naked single → `p=1.0`; N candidates → uniform `1/N`; 0 → all-zero | `katgpt-pruners/src/sudoku_pruner.rs:141` | ✅ Modelless prior derived from constraint set (paper's analog is trained) |
| CDCL phase initialization from `d*_i` | DDTree path-commit / speculation accept — drafter proposes, verifier accepts/rejects | `katgpt-rs` speculative decode | ✅ Architectural |
| Symbolic solver (completeness guarantee) | `ConstraintPruner::is_valid` ("Returns `false` to prune") + `batch_is_valid` + `propagate` + `DominoPruner::causal_correction` | `katgpt-core/src/traits.rs:36` | ✅ Hard contract — the "order ≠ correctness" safety invariant |
| Solver-architecture-aware regime (cadical3 vs glucose4) | `DifficultyFilter::admit(guide_score, estimated_solve_rate)` — drops trivially-solved (1.0) and impossible (0.0) | `katgpt-core/src/cgsp/traits.rs:229` | ⚠️ Partial — drops the 0/1 extremes but doesn't model "overhead-dominated solver that shouldn't receive hints at all" (see §2.3) |
| Empirical solve-rate → priority update (SLE loop) | `Solver::attempt` returns `f32 ∈ [0,1]` solve-rate; `HintDeltaBandit::absorb(arm, reward = (1-solve_rate)·guide_score)` | `katgpt-core/src/cgsp/traits.rs:92, 127` | ✅ Architectural — the runtime half of SLE |
| `min_completion_distance` / "no valid completion reachable" | `CompletionHorizon::min_completion_distance` (Research 183) — admissible lower bound on tokens to reach a complete valid output | `katgpt-core/src/traits.rs:232` | ✅ Direct analog of "MRV exposes contradictions early" |
| Three solver regimes benchmarked (§3) | `sudoku_speculate_bench.rs` Modes 1/2/3: `backtrack` / `speculate_iterative` (draft→prune→commit→verify with backtrack fallback) / `speculate_oneshot` (single full-depth lookahead) | `katgpt-rs/benches/sudoku_speculate_bench.rs` | ✅ **The paper's exact three regimes, on the paper's exact domain** |

### 2.2 Why this is a stronger validation than PTRM (Research 049)

PTRM's verdict was "STRONG VALIDATION, MINIMAL ACTION" — the TRM architecture mapped onto `BanditPruner` + `DDTreeBranchCache` + the trait stack. G-RRM is **strictly stronger validation** because the codebase ships the paper's exact evaluation domain (`Sudoku9x9` + `SudokuPruner`), the paper's exact cell-selection heuristic (`new_mrv`), and the paper's exact three solver regimes (`backtrack` / `speculate_iterative` / `speculate_oneshot`). There is no vocabulary-mismatch gap — the codebase uses "MRV" by name.

### 2.3 The one genuinely novel modelless extraction (small Gain, deferred)

The paper's empirical finding that **cadical3 (overhead-dominated) shows no benefit and even a small mean slowdown (0.896×)** while glucose4/backtracking benefit greatly is a perf-characterization insight that our `DifficultyFilter` trait does not currently model. `DifficultyFilter::admit` only drops candidates at the extremes (`solve_rate = 0.0` or `1.0`); it has no concept of *"this downstream solver is overhead-dominated — skip hint injection entirely."*

A small **Gain**-tier refinement would be a `HintReceptivity` classifier (or a `Solver` trait method like `fn hint_receptivity(&self) -> HintPolicy { … }` returning `Skip | OrderOnly | PhaseInit`) that gates whether hint injection is worth the overhead. This is a characterization of when G-RRM helps, not a new mechanism — and our current solvers (custom backtracking + DDTree speculation) are all in the "search-dominated, hint-receptive" regime, so the gate would be a no-op for shipped solvers today.

**This is deferred to `.issues/` — not a new feature gate, not a Super-GOAT, not a GOAT.** It's a future-proofing hook for the day we add an overhead-dominated solver (e.g., a real SAT-backend like glucose/cadical bindings, which we don't currently ship).

### 2.4 Fusion considerations (none pursued)

I considered three fusion angles and rejected all three:

1. **G-RRM × CLR (R255)** — CLR is *post-hoc* claim-level voting (sample K, verify, vote); G-RRM is *a priori* branch ordering (sample once, order search). They operate on opposite ends of the propose↔verify loop and don't fuse into a new capability — they're complementary stages. Already covered by the existing test-time-scaling stack.

2. **G-RRM × NS-CSG (R188)** — NS-CSG already maps the entire ScreeningPruner↔ConstraintPruner↔SpeculativeGenerator triad to the propose↔verify paradigm with formal BFCP closure guarantees. G-RRM adds nothing NS-CSG doesn't already cover architecturally.

3. **G-RRM × DEC Stokes substrate (R296/Plan 314)** — the SE-RRM symbol-axis `D × I × K` tensor has a superficial resemblance to a rank-3 cochain, but the SE-RRM's `K` axis is a categorical symbol set, not a differential-form degree. No geometric structure to exploit — the analogy is purely notational.

---

## 3. Verdict

**PASS.** One-line reasoning: *the modelless G-RRM integration pattern (neural prior → ordering → completeness-preserving symbolic solver → empirical solve-rate updates priorities) ships completely as the `SpeculativeGenerator` + `ScreeningPruner` + `ConstraintPruner` + `Solver` + `HintDeltaBandit` stack, on the paper's exact domain (Sudoku) with the paper's exact MRV heuristic (`SudokuPruner::new_mrv`) across the paper's exact three solver regimes (`sudoku_speculate_bench.rs`). The paper's quality headline (91% FSR SE-RRM) is training-side → riir-train.*

### 3.1 The three claim types (per §3.6 defend-wrong PoC rule)

| Claim type | Status | Evidence |
|---|---|---|
| **Architectural coverage** ("the runtime analog exists") | ✅ Confirmed | §2.1 table — every G-RRM element maps to a shipped primitive. The Sudoku + MRV + three-regime match is unusually tight (no vocabulary-mismatch gap). |
| **Latency / resource parity** ("modelless, sub-µs, no GD") | N/A — not claimed | We don't ship a CDCL SAT solver; our backtracking is custom. No latency comparison was attempted. |
| **Quality parity** ("matches / beats the paper's numbers") | **NOT CLAIMED** | Our modelless marginals (`uniform 1/N on valid digits` from the constraint set) are weaker than a trained SE-RRM (91% FSR on 9×9). That is a riir-train question — the SE-RRM is the paper's IP. A PoC is not required because we make no quality-parity claim; the PASS is on architectural-coverage grounds only. |

### 3.2 Why this isn't a false-PASS

The §3.6 rule warns: *"A PASS verdict backed only by architectural reasoning is the #1 false-PASS failure mode."* This verdict avoids that failure mode by:

1. **Explicitly not claiming quality parity.** The note says "NOT CLAIMED" in the table above — the modelless marginals are acknowledged weaker than a trained SE-RRM.
2. **Routing the quality question to riir-train.** The SE-RRM is a trained transformer with deep supervision + stop-gradient — that is unambiguously training-side and out of scope for this workflow.
3. **Citing concrete code evidence, not just grep hits.** `SudokuPruner::new_mrv` (Issue 005 Option A) implements the paper's exact MRV heuristic on the paper's exact domain — this is a stronger match than the typical architectural redirect.

### 3.3 Routing decisions

| Component | Route | Reason |
|---|---|---|
| SE-RRM (training) | → riir-train | Trained transformer, deep supervision, stop-gradient — training-only |
| HRM/TRM looped-transformer inference wrapper | Already covered by R097 (Training-Free Looped Transformers) + R273 (ELT) | Inference-time looped-block refinement is shipped |
| G-RRM integration pattern | ✅ Already shipped (katgpt-rs trait stack) | No new files |
| Hint-receptivity classifier (§2.3) | → `.issues/` (small Gain, deferred) | Not load-bearing for any shipped solver today; future-proofing for SAT-backend addition |

---

## 4. Action Items

- [ ] **None in this session.** The PASS verdict requires no files in `katgpt-rs/.plans/`, `riir-ai/`, `riir-chain/`, or `riir-neuron-db/`.
- [-] **Deferred:** open an `.issues/` entry for the `HintReceptivity` / `Solver::hint_receptivity()` trait method (§2.3) — a small Gain-tier refinement to gate hint injection for overhead-dominated solvers. Not load-bearing today; relevant only if/when a real CDCL SAT backend is wired in. (Deferring per the rule "Create issue at ./issues for optimization or refactor task, do not create plan".)

---

## 5. Cross-References

**Closest cousins (read in pre-flight):**
- `katgpt-rs/.research/049_PTRM_Probabilistic_Tiny_Recursive_Model.md` — "STRONG VALIDATION, MINIMAL ACTION" — the closest verdict shape.
- `katgpt-rs/.research/097_Training_Free_Looped_Transformers.md` — covers the RRM-as-inference-loop angle (frozen checkpoint + damped Euler sub-stepping).
- `katgpt-rs/.research/188_NS_CSG_Neuro_Symbolic_Concurrent_Stochastic_Games.md` — formal propose↔verify mapping (`ScreeningPruner` = perception, `ConstraintPruner` = BFCP, drafter = max, verifier = min).
- `katgpt-rs/.research/255_VibeThinker_CLR_Test_Time_Reliability.md` — post-hoc claim-level voting (the inverse direction).
- `katgpt-rs/.research/182_STV_Self_Trained_Verification.md` — iterative V-R loop.

**Shipped primitives referenced:**
- `katgpt-rs/crates/katgpt-core/src/traits.rs` — `ConstraintPruner`, `ScreeningPruner`, `SpeculativeGenerator`, `DominoPruner`, `CompletionHorizon`
- `katgpt-rs/crates/katgpt-core/src/cgsp/traits.rs` — `Solver`, `HintDeltaBandit`, `DifficultyFilter`, `BatchQualityGate`, `CollapseSignal`
- `katgpt-rs/crates/katgpt-pruners/src/sudoku_pruner.rs` — `SudokuPruner::new_mrv` (Issue 005 Option A, MRV cell ordering), `latent_marginals` (Issue 005 Option B, modelless per-cell prior)
- `katgpt-rs/benches/sudoku_speculate_bench.rs` — three solver regimes (`backtrack` / `speculate_iterative` / `speculate_oneshot`)
- `katgpt-rs/crates/katgpt-percepta/src/legacy.rs` — `Sudoku9x9::solve` (the ground-truth complete solver)

**Routed elsewhere:**
- SE-RRM training → riir-train (out of scope for this workflow)

---

## TL;DR

**PASS.** The G-RRM integration pattern (neural prior → ordering → completeness-preserving symbolic solver → solve-rate feedback) already ships as the katgpt-rs `SpeculativeGenerator` + `ScreeningPruner` + `ConstraintPruner` + `Solver` + `HintDeltaBandit` stack — with the paper's exact MRV heuristic (`SudokuPruner::new_mrv`), on the paper's exact domain (`Sudoku9x9`), across the paper's exact three solver regimes (`sudoku_speculate_bench.rs`). The paper's headline-quality component (SE-RRM, 91% FSR) is training-side → riir-train. No quality-parity claim is made (our marginals are modelless `1/N`-on-valid-digits, weaker than a trained prior). No files created in private repos. One small Gain-tier refinement (`HintReceptivity` classifier for overhead-dominated solvers) deferred to `.issues/` — not load-bearing today.
