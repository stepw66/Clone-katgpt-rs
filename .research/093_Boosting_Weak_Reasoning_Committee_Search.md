# Research 93: Agentic Systems as Boosting — Committee Search Protocol

> Source: [Agentic Systems as Boosting Weak Reasoning Models](https://arxiv.org/pdf/2605.14163) by Varun Sunkaraneni (TAMU), Pierfrancesco Beneventano, Riccardo Neumarker, Tomaso Poggio, Tomer Galanti (MIT/TAMU), arXiv 2605.14163, May 2026
> Local: N/A (paper-only distillation)
> Date: 2026-05 (paper), distilled 2026-05
> **Verdict: STRONG VALIDATION — Our DDTree + BtRank + ScreeningPruner + ConstraintPruner stack IS the paper's committee protocol Π_{k,m,r}. The paper formalizes what we already built. Actionable items: oracle-gap recovery metric, budget sizing rules, position-swap debiasing. Feature-gate new diagnostics under `committee_boost`.**

---

## TL;DR

The paper proves that verifier-backed committee search can boost weak reasoning models to match much stronger ones. The key insight is a **four-way separation**: proposal coverage (can a good move appear?), local identifiability (can the system recognize it?), progress (can local choices compose into trajectories?), and diversity (do more calls escape different failure modes?).

The protocol Π_{k,m,r} samples k proposals, applies m critic votes per proposal, then r comparator votes per surviving pair — exactly our DDTree (k branches) + ScreeningPruner (critic) + BtRank (comparator). The paper proves:
1. **Coverage ≠ Identifiability** — sampling more candidates cannot create critics (Proposition 1)
2. **Bridge theorem** — coverage + identifiability together give reliable amplification (Theorem 1)
3. **Error decomposition** — err ≤ L × (ε_prop + k²e^{-βm-2rσ²}) (Theorem 2)
4. **Blind-spot ceiling** — oracle best-of-k → 1-B as k→∞, B = blind-spot mass (Lemma 2)

Empirically: GPT-5.4 nano + committee orchestration (k=8) reaches 76.4% on SWE-bench Verified, matching Gemini 3 Pro and Claude Opus 4.5 Thinking (standalone 69.8%). Oracle best-of-8 = 79.0%, showing most remaining failures are proposal-coverage (blind spots), not selection.

---

## Core Architecture

### The Committee Protocol Π_{k,m,r}

At each non-terminal state s:
1. **Propose (k)**: Sample k candidate actions from proposer harness
2. **Critique (m)**: Apply m independent critic calls per candidate; discard rejected ones
3. **Compare (r)**: Among survivors, r comparator votes per unordered pair; Copeland tournament selects winner
4. **Advance**: Apply winner, repeat until terminal state

```text
s_t ──[propose k]──> {a₁,...,aₖ} ──[critique m]──> survivors ──[compare r]──> winner ──> s_{t+1}
```

### Four Amplification Quantities

| Quantity | Role | Formal Definition | Our Equivalent |
|----------|------|-------------------|----------------|
| **Coverage** | Can a good move appear? | α₀ = P(LLM outputs sound action) ≥ α₀ > 0 | DDTree branch probability |
| **Identifiability** | Can the system recognize good moves? | Critic edge β, comparator edge σ | ScreeningPruner β, BtRank σ |
| **Progress** | Do local choices compose? | Rank d_x(s) decreases per step | TreeNode depth / rollout steps |
| **Diversity** | Do more calls escape different failures? | Blind-spot floor B = P(q_s(Z)=0) | BanditPruner exploration |

### Key Theoretical Results

**Proposition 1 (Coverage ≠ Identifiability):**
For any M ≥ 2, there exists a task where proposer coverage is strong (α₀ = 1-1/M) but NO black-box procedure can construct a useful critic or comparator. Local identifiability requires an external signal (execution, proof checking, tests, constraint solving).

**Theorem 1 (Bridge Theorem):**
Under coverage (Assumption 1) + identifiability (Assumption 2), with k ≥ |P_N| × ⌈ln(1/δ)/α₀⌉ proposer calls:
- α_committee ≥ 1 - δ_prop (high probability of good proposal appearing)
- Critic/comparator edges preserved: β ≥ β₀, σ ≥ σ₀

**Theorem 2 (Error Decomposition):**
```
ε_loc(s) ≤ ε_prop(k;s) + k² × e^{-βm - 2rσ²}
```
- First term: no good proposal appeared
- Second term: bad proposal survived criticism AND won comparison

**Lemma 2 (Blind-Spot Floor):**
```
ε_prop(k;s) = B_s + R_k(s)
```
- B_s = P(q_s(Z)=0) — irreducible blind-spot mass
- R_k(s) → 0 as k → ∞ — finite-sampling residual
- More proposals reduce R_k but CANNOT reduce B_s

**Global Error Bound:**
```
err_x(k,m,r) ≤ L_x × (B + R_k + k² × e^{-βm - 2rσ²})
```

### Budget Sizing Rules

For target failure probability δ over depth-L trajectory:
```text
k ≥ |P_N| × ⌈ln(2L/δ) / α₀⌉          (proposer width)
m ≥ ⌈(1/2β₀) × ln(2k²L/δ)⌉            (critic depth per candidate)
r ≥ ⌈(1/4σ₀²) × ln(2k²L/δ)⌉           (comparator votes per pair)
```

Total role calls: O(L × (k + mk + rk²))

### Oracle-Gap Recovery Metric

The paper introduces a key diagnostic:
```
Rec(k,m,r) = (p_system - p₁) / (p_oracle(k) - p₁)
```
- p₁ = single-shot solve rate (Pass@1)
- p_oracle(k) = best-of-k with perfect selector (hidden-test oracle)
- p_system = actual deployed harness solve rate
- Rec measures **how much of the latent capability the selector recovers**

### Position-Swap Debiasing

Each unordered pair (pᵢ, pⱼ) is compared in BOTH orders:
1. pᵢ as "Patch A", pⱼ as "Patch B"
2. pⱼ as "Patch A", pᵢ as "Patch B"

A pairwise win is counted ONLY if both orders agree. Disagreements → treated as tie. This eliminates lead-position bias.

---

## Mapping to Our Architecture

### Direct Component Mapping

| Paper Concept | Our Component | Location | Type |
|---------------|---------------|----------|------|
| Proposer (k samples) | DDTree branch expansion | `speculative/dd_tree.rs` | Model-based |
| Critic (m votes) | ScreeningPruner + ConstraintPruner | `speculative/types.rs` | Modelless→model-based |
| Comparator (r votes) | BtRank pairwise tournament | `pruners/bt_rank.rs` | Model-based |
| Copeland tournament | `BtScores::rank()` | `pruners/bt_rank.rs` | Modelless |
| Verifier R_x | `ConstraintPruner::is_valid()` | `speculative/types.rs` | Modelless |
| Valid state system | TreeNode state space | `speculative/types.rs` | Structural |
| Progress rank d_x | TreeNode depth / rollout position | `speculative/types.rs` | Structural |
| Portfolio P_N | BanditPruner strategies | `pruners/bandit.rs` | Modelless |
| Blind-spot floor B | Unseen states / unsolvable positions | (diagnostic) | Measurement |
| SR²AM configurator | Budget allocator (k,m,r sizing) | `pruners/configurator_bandit.rs` | Modelless |

### Our Stack IS the Committee Protocol

```rust
// Our existing code already implements Π_{k,m,r}:
//
// DDTree expansion → k proposals (Assumption 1: coverage)
// ScreeningPruner  → m critic votes (Assumption 2: identifiability β)
// BtRank           → r comparator votes per pair (Assumption 2: identifiability σ)
// ConstraintPruner → verifier R_x (Assumption 3: one-sided local verifier)
// BanditPruner     → portfolio diversity (reduces blind-spot floor B)
```

The paper proves our architecture is **theoretically sound** — the three-layer separation (propose/critique/compare) is not just a design choice but a **necessary decomposition** (Proposition 1 shows you can't derive critics from proposers alone).

### Role-wise Spectrum (Modelless ↔ Model-Based)

| Layer | Modelless (Zero Inference) | Model-Based (Forward Pass) |
|-------|---------------------------|---------------------------|
| **Proposer** | TemplateProposer, BanditPruner Q-values | DDTree + drafter LoRA |
| **Critic** | ConstraintPruner (syntax/rules), NoScreeningPruner | ScreeningPruner::relevance() |
| **Comparator** | BtRank (pairwise logit comparison) | Full LLM-as-judge comparator |
| **Verifier** | ConstraintPruner::is_valid() (deterministic) | Execution/testing/proof checking |

This is the same modelless↔model-based spectrum identified in Research 037 (REAP) and Research 021 (G-Zero).

---

## What We Already Have (Paper Validates Our Design)

### 1. DDTree k-branch expansion = Paper's k proposals ✅
Our DDTree already samples k branches at each depth level. The paper's coverage condition (Assumption 1) is exactly what DDTree provides: with enough branches, at least one contains a progressing-sound action.

### 2. BtRank pairwise tournament = Paper's Copeland comparator ✅
Our `bt_fit()` implements the same Bradley-Terry model P(i≻j) = σ(sᵢ-sⱼ) that the paper uses for pairwise comparison. Our `BtScores::rank()` implements Copeland-style tournament aggregation.

### 3. ScreeningPruner = Paper's critic ✅
Our `ScreeningPruner::relevance()` provides the binary filtering signal. The paper shows this is essential — you can't skip critics and rely on comparators alone (Table 1: dropping zero-support patches improves 75.8% → 76.4%).

### 4. ConstraintPruner = Paper's verifier R_x ✅
Our `ConstraintPruner::is_valid()` is the one-sided local verifier (Assumption 3): sound actions always pass, unsound actions rejected with probability ≥ 1-ν.

### 5. BanditPruner = Paper's diversity mechanism ✅
BanditPruner explores different strategies, reducing the blind-spot floor B by ensuring the portfolio P_N covers different latent subpopulations.

### 6. SR²AM Configurator = Paper's budget allocator ✅
Our configurator bandit (Plan 112) already adapts k (tree budget), m (screening depth), r (comparison rounds) based on context — exactly the paper's sizing rules.

---

## What IS Worth Exploring (Gap Analysis)

### Priority 1: Oracle-Gap Recovery Metric (High Value, ~100 LOC)

The paper's key diagnostic we DON'T have:

```rust
/// Oracle-gap recovery: how much latent capability the selector recovers.
///
/// Rec = (p_system - p1) / (p_oracle - p1)
///
/// - Rec ≈ 1.0: selector recovers nearly all latent capability
/// - Rec ≈ 0.5: selector misses half the recoverable gains
/// - Rec near 0: selector is barely better than single-shot
#[derive(Debug, Clone)]
pub struct OracleGapRecovery {
    /// Single-shot solve rate (Pass@1).
    pub p1: f64,
    /// Oracle best-of-k solve rate (perfect selector).
    pub p_oracle: f64,
    /// Deployed system solve rate.
    pub p_system: f64,
}

impl OracleGapRecovery {
    /// Recovery fraction: 0.0 to 1.0 (or NaN if no gap).
    pub fn recovery(&self) -> f64 {
        let gap = self.p_oracle - self.p1;
        if gap.abs() < f64::EPSILON { return f64::NAN; }
        (self.p_system - self.p1) / gap
    }
}
```

This tells us whether failures are **selection failures** (low Rec → improve critic/comparator) or **coverage failures** (high Rec, low p_oracle → improve proposer or add diversity).

### Priority 2: Position-Swap Debiasing for BtRank (Medium Value, ~30 LOC)

The paper shows that swapping A/B order in pairwise comparisons and requiring agreement eliminates position bias. Our `bt_pair_random` already randomizes pair selection but doesn't do position-swap debiasing:

```rust
/// Compare pair (i,j) in both orders; count win only if both agree.
pub fn debiased_compare<F>(i: usize, j: usize, compare: &F) -> BtOutcome
where F: Fn(usize, usize) -> BtOutcome,
{
    let fwd = compare(i, j); // i as "A", j as "B"
    let rev = compare(j, i); // j as "A", i as "B"
    // Map rev back to original indices
    let rev_mapped = match rev {
        BtOutcome::Win => BtOutcome::Lose,  // j won → i lost
        BtOutcome::Lose => BtOutcome::Win,  // j lost → i won
        BtOutcome::Tie => BtOutcome::Tie,
    };
    match (fwd, rev_mapped) {
        (BtOutcome::Win, BtOutcome::Win) => BtOutcome::Win,
        (BtOutcome::Lose, BtOutcome::Lose) => BtOutcome::Lose,
        _ => BtOutcome::Tie, // disagreement → tie
    }
}
```

### Priority 3: Budget Sizing from Theory (Low Value, ~50 LOC)

The paper gives explicit sizing rules. We could add a function that computes optimal (k, m, r) given target δ, depth L, and estimated (α₀, β₀, σ₀):

```rust
/// Compute committee budget from theoretical sizing rules.
pub fn committee_budget(
    depth: usize,       // L_x: trajectory depth
    delta: f64,         // target failure probability
    alpha: f64,         // proposer coverage α₀
    beta: f64,          // critic edge β₀
    sigma: f64,         // comparator edge σ₀
    portfolio_size: usize, // |P_N|
) -> CommitteeBudget {
    let k = (portfolio_size as f64 * (2.0 * depth as f64 / delta).ln() / alpha).ceil() as usize;
    let m = ((1.0 / (2.0 * beta)) * (2.0 * k.pow(2) as f64 * depth as f64 / delta).ln()).ceil() as usize;
    let r = ((1.0 / (4.0 * sigma * sigma)) * (2.0 * k.pow(2) as f64 * depth as f64 / delta).ln()).ceil() as usize;
    CommitteeBudget { k: k.max(1), m: m.max(1), r: r.max(1) }
}
```

### Priority 4: Blind-Spot Floor Diagnostic (Low Value, ~80 LOC)

Measure B by comparing oracle best-of-k across increasing k values. If p_oracle saturates below 1.0, the residual is the blind-spot floor:

```rust
/// Estimate blind-spot floor from oracle best-of-k curve.
///
/// If p_oracle(k) saturates at some value < 1.0, the gap is the blind-spot floor B.
/// B = 1 - lim_{k→∞} p_oracle(k)
pub fn estimate_blind_spot_floor(oracle_rates: &[(usize, f64)]) -> f64 {
    // Fit asymptote: B ≈ 1 - max(oracle_rates)
    1.0 - oracle_rates.iter().map(|&(_, r)| r).fold(f64::NEG_INFINITY, f64::max)
}
```

---

## What We DON'T Need

| Paper Component | Why Not Needed |
|-----------------|----------------|
| SWE-bench specific harness | We have game domains (Bomber, Go, Monopoly) as verifiers |
| LLM-as-judge prompts | Our BtRank uses model logit comparisons, not prompt-based judging |
| Proposal caching | Our DDTree naturally generates and stores k branches |
| Hidden test oracle | Our validators (ConstraintPruner) ARE the local verifier |
| Presentation order concerns | Our BtRank operates on latent scores, not text order |

---

## Application to Model-Based vs Modelless Paths

### Modelless Path (katgpt-rs, Primary)

The committee protocol is **naturally modelless** at the pruner layer:
- ConstraintPruner: deterministic validity (zero inference)
- ScreeningPruner: heuristic relevance scoring (zero inference)
- BanditPruner: Q-value exploration (O(1) per step)
- BtRank: pairwise logit comparison (uses existing forward pass outputs)

The paper's error decomposition gives us a **principled way to allocate budget**:
- If Rec ≈ 0.8+: good selector, focus on proposer diversity (add templates, strategies)
- If Rec ≈ 0.5: selector needs work (more critic votes, better relevance scoring)
- If Rec ≈ 0.2: selector is barely working (check critic edge β, comparator edge σ)

### Model-Based Path (riir-ai, Opt-In)

The paper validates that model-based critics/comparators provide stronger edges (β₀, σ₀), which means:
- Lower m and r needed for same reliability
- Higher Rec possible with same budget
- But cost per call is higher (forward pass vs heuristic)

This aligns with our existing G-Zero Phase 1 (modelless) → Phase 2 (model-based) design.

---

## Key Takeaways

1. **Our architecture is theoretically grounded.** The paper proves that the propose/critique/compare decomposition is not just convenient but **necessary** — coverage alone cannot create identifiability.

2. **Blind-spot floor is the real ceiling.** More proposals (larger k) help only up to the blind-spot floor B. Further gains require **diversifying the proposal portfolio** (different prompts, strategies, models), not more of the same.

3. **Oracle-gap recovery is the key diagnostic.** We should measure Rec = (p_system - p1) / (p_oracle - p1) to determine whether to invest in selection (improve critic/comparator) or generation (improve proposer diversity).

4. **Budget sizing is tractable.** Given (α₀, β₀, σ₀, L, δ), the paper gives explicit formulas for k, m, r. Our SR²AM configurator can implement these directly.

5. **Position-swap debiasing is cheap and effective.** Requiring both presentation orders to agree for a pairwise win costs 2× comparisons but eliminates position bias. Worth adding to BtRank.

6. **The paper's SWE-bench result (67% → 76.4% with k=8) validates our approach.** Our DDTree k-branch + BtRank pipeline should show similar gains on our game benchmarks.

---

## Verdict

**STRONG VALIDATION, MINIMAL NEW CODE.**

The paper formalizes exactly what our DDTree + BtRank + ScreeningPruner stack already implements. The main actionable items are:

1. **Oracle-gap recovery metric** (~100 LOC) — the key diagnostic we're missing
2. **Position-swap debiasing** (~30 LOC) — cheap accuracy improvement for BtRank
3. **Budget sizing function** (~50 LOC) — principled k,m,r from theory
4. **Blind-spot floor estimation** (~80 LOC) — diagnostic for coverage ceiling

Total: ~260 LOC, all behind `committee_boost` feature gate.

The conceptual alignment is near-perfect:
- Paper's Π_{k,m,r} = our DDTree + ScreeningPruner + BtRank
- Paper's coverage/identifiability = our modelless/model-based spectrum
- Paper's blind-spot floor = our BanditPruner diversity motivation
- Paper's oracle-gap recovery = the benchmark metric we should have had

---

## References

- Paper: https://arxiv.org/pdf/2605.14163
- Related research:
  - `021_G-Zero_Self-Play_Open-Ended_Generation.md` (Hint-δ, TemplateProposer)
  - `037_REAP_Model-Based_Modelless_Duality.md` (modelless↔model-based spectrum)
  - `058_GRAM_Generative_Recursive_Reasoning.md` (width scaling, DDTree validation)
  - `040_OpenDeepThink_Bradley_Terry_Pairwise_Ranking.md` (BtRank design)
  - `076_SR2AM_Self_Regulated_Simulative_Reasoning.md` (budget configurator)
- Key code:
  - `src/pruners/bt_rank.rs` — BtRank pairwise tournament (comparator)
  - `src/speculative/dd_tree.rs` — DDTree branch expansion (proposer)
  - `src/speculative/types.rs` — ScreeningPruner, ConstraintPruner (critic/verifier)
  - `src/pruners/bandit.rs` — BanditPruner (diversity)
  - `src/pruners/configurator_bandit.rs` — SR²AM (budget allocator)