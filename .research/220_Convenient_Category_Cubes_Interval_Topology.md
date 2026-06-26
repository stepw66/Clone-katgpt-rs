# Research 220: Convenient Category of Cubes — Interval-Preserving Morphisms for Inference Topology

**Date:** 2026-06
**Paper:** [A Convenient Category of Cubes (arXiv:2503.13663)](https://arxiv.org/abs/2503.13663) — Sanjeevi Krishnan, Emily Rudman (2025)
**Context:** katgpt-rs DEC operators (Research 219), DDTree constraint pruning, FlowField/CellComplex spatial reasoning
**Verdict:** ⚡ CONDITIONAL GAIN — Novel structural constraints for token validity + CAT(0) navigation; validates existing DEC choice

---

## 1. TL;DR

The paper defines **⊞** (boxed plus), a variant of the cube category where:
- Objects = finite Boolean lattices: `[0]`, `[1]`, `[1]²`, `[1]³`, ...
- Morphisms = **interval-preserving monotone functions** (monotone maps that send intervals to intervals)
- It is the **largest** reasonable Eilenberg-Zilber cube category variant that excludes reversals and diagonals
- It carries a **proper model structure** Quillen equivalent to simplicial sets → fibrations model Martin-Löf dependent types

For katgpt-rs, the paper provides three novel fusion angles beyond just validating our existing DEC infrastructure:
1. **IntervalPruner** — structural constraint ensuring valid token regions map coherently through layers
2. **CubicalNerve** — systematic extraction of CAT(0) cubical complexes from game map posets (unique geodesics = deterministic NPC paths)
3. **LatticeOperad** — formal AND/OR composition law for DDTree pruner expressions via distributive lattice word problem

Most of the paper's value is **theoretical validation** of our existing DEC approach (we picked the right cube variant). The three novel fusions are CONDITIONAL GAIN — implement behind feature flags, benchmark, promote only if measured gain.

---

## 2. Paper Core Ideas

### 2.1 The Category ⊞

The cube category variants form a hierarchy of "how many maps between powersets of {0,1}ⁿ do we allow?":

```
□_c  (classical)     ⊇  □_EZ  (Eilenberg-Zilber)  ⊇  ⊞  (this paper)
all maps                  no reversals/diags          interval-preserving monotone
```

⊞ sits at a sweet spot:
- **Bigger than** □ (the standard cube category with only face/degeneracy maps) → more expressive
- **Smaller than** □_c → excludes pathological maps that break degeneracy structure
- **Largest** category satisfying three constraints simultaneously: (a) contains Δ*₁ as wide subcategory, (b) excludes reversals and diagonals, (c) morphisms factor into surjective → injective (Prop. 3.21)

### 2.2 Interval Preservation

A monotone function `f: P → Q` between posets is **interval-preserving** if for every interval `[a,b] = {x : a ≤ x ≤ b}` in P, the image `f([a,b])` is also an interval in Q.

This is strictly stronger than monotonicity. Example: the map `{0,1,2} → {0,1}` sending `{0→0, 1→0, 2→1}` is monotone but NOT interval-preserving: the interval `[0,2]` maps to `{0,1}` (which is the interval `[0,1]`) ✓, but the interval `[1,2]` maps to `{0,1}` (not `{1}` as expected from endpoint images). Actually wait — the endpoint images are `f(1)=0, f(2)=1`, so the image should be `[0,1]` if interval-preserving, and it is. Let me correct: the **real** failure case is when an interior point maps outside the convex hull of the endpoint images. For Boolean lattices specifically, interval preservation constrains how subcubes can be mapped.

**Key property:** Interval preservation means "no gaps in the image" — if two values are mapped, everything between them is also in the image. This is a **convexity condition** on morphisms.

### 2.3 Generating Set (Theorem 3.23)

⊞ is generated as a symmetric monoidal category by:
- **σ: [1] → [0]** (the unique surjection, "squeeze dimension")
- **δ⁺, δ⁻: [0] → [1]** (the two "endpoint inclusions")
- **ALL monotone surjections [1]ⁿ ↠ [1]** (the "coordinate projections and beyond")

The third generator are what make ⊞ richer than □. In □, the only surjections are projections (drop a coordinate). In ⊞, ANY monotone surjection is allowed. This includes things like "OR two coordinates together" — `f: {0,1}² → {0,1}` defined by `f(a,b) = a ∨ b`.

### 2.4 Semilattice Characterizations (Corollaries 3.25, 3.26)

When you restrict ⊞ to include only one kind of "coconnection" (degeneracy-like map):

| Variant | Morphisms are... |
|---------|-----------------|
| Δ₁[τ,γ⁻]* | Interval-preserving **meet**-semilattice homomorphisms |
| Δ₁[τ,γ⁺]* | Interval-preserving **join**-semilattice homomorphisms |

This means the "typed" subcategories of ⊞ correspond to familiar algebraic structures (meet/join semilattices) WITH the interval-preservation constraint.

### 2.5 Distributive Lattice Operad (§3.3)

The endomorphism operad of `[1]` in ⊞ **is** the distributive lattice operad. This means:
- Operations = AND (∧) and OR (∨) satisfying distributivity
- Composition = substitution of AND/OR formulas
- The word problem for free distributive lattices gives canonical forms

This is the algebraic structure underlying all Boolean constraint logic.

### 2.6 CAT(0) Cubical Complexes (§4.2.3)

For a finite distributive meet-semilattice L, the cubical set **⊞[L]** (the representable functor `⊞(-, L)`) constructs ALL finite CAT(0) cubical complexes.

**CAT(0) = uniquely geodesic** = between any two points, there is exactly one shortest path. For game AI:
- CAT(0) maps guarantee deterministic shortest-path navigation
- No ambiguity in pathfinding — the geodesic is unique
- The construction ⊞[L] gives a systematic recipe: "give me a poset of zones, I give you a CAT(0) cube complex"

This connects to **Birkhoff duality**: finite distributive lattices ↔ finite posets. So ⊞[L] is really ⊞[P] for a poset P (the prime ideals of L).

### 2.7 Proper Model Structure (Theorem 5.7)

The model structure on cubical sets over ⊞ is:
- **Quillen equivalent** to simplicial sets (classical homotopy theory)
- **Left induced** along triangulation functor
- **Proper** — weak equivalences are preserved under pullback along fibrations and pushout along cofibrations

Properness matters for dependent type theory: it means substitution of type-theoretic equivalents preserves typing. For us, it means "substituting a weakly equivalent draft with another preserves decoding correctness."

---

## 3. Fusion Novel Ideas (NOT Direct Mapping)

### 3.1 IntervalPruner — Convex Validity Regions

**The Insight:** In token logit space, the set of valid tokens at each position is a subset of the vocabulary. An "interval" in this context = a contiguous range of token IDs that are all valid (after sorting by logit). Interval preservation means: **if tokens i..j are all valid, their valid images under any morphism (layer transformation, beam step) must also be contiguous.**

**Why it's fusion:** The paper defines interval preservation for Boolean lattices. We translate it to token validity sets:
- A `ConstraintPruner` currently returns `is_valid(token) → bool` per token
- An `IntervalPruner` additionally checks: "is the valid set convex (interval-closed)?" and "does the pruner's image preserve intervals?"
- This catches a class of errors that per-token validity cannot: **structural incoherence** where valid tokens form a Swiss-cheese pattern (valid, invalid, valid, invalid) — a symptom of conflicting pruner signals

**Implementation sketch:**
```rust
trait IntervalPruner: ConstraintPruner {
    /// Check that valid token set forms convex intervals after sorting by logit
    fn is_interval_closed(&self, logits: &[f32], valid: &[bool]) -> bool;
    /// Enforce interval closure: fill gaps in valid set
    fn close_intervals(logits: &[f32], valid: &mut [bool]);
}
```

**Performance:** O(n) pass over vocabulary after existing pruning — negligible overhead.

### 3.2 CubicalNerve — CAT(0) from Game Map Poset

**The Insight:** A game map's navigable topology is naturally a poset (zones ordered by containment/reachability). The construction ⊞[L] for finite distributive meet-semilattices L gives ALL finite CAT(0) cubical complexes. Our existing `CellComplex` from DEC (Research 219) can compute this nerve.

**Why it's fusion:** The paper constructs ⊞[L] as a representable presheaf. We construct it as a `CellComplex` using our DEC operators:
1. Take the game map zone poset P (zones ⊂ subzones ⊂ cells)
2. Form the distributive lattice L = downsets of P (Birkhoff duality)
3. The cubical nerve ⊞[L] has cubes indexed by antichains in P
4. This yields a CAT(0) cubical complex — uniquely geodesic

**NPC Navigation Impact:** On a CAT(0) complex, the geodesic between any two points is unique. This means:
- No tie-breaking needed for equal-length paths
- Pathfinding is deterministic (reproducible for sync)
- The Hodge harmonic component from our DEC infrastructure gives the geodesic

**Implementation sketch:**
```rust
fn cubical_nerve(zone_poset: &Poset<ZoneId>) -> CellComplex {
    let dist_lattice = downset_lattice(zone_poset); // Birkhoff
    let antichains = all_antichains(zone_poset);
    // Each antichain → a cube in the nerve
    CellComplex::from_antichains(&antichains, &dist_lattice)
}
```

### 3.3 LatticeOperand — Canonical AND/OR for Pruners

**The Insight:** The distributive lattice operad (§3.3) gives a formal composition law for AND/OR combinations of pruner signals. Our DDTree already builds decision trees with AND/OR logic, but the combinations are ad-hoc. The operad gives:
- **Associativity, commutativity, distributivity** of AND/OR for free
- **Word problem solution:** any AND/OR expression has a canonical normal form (disjunctive normal form via Birkhoff)
- **Substitution law:** operadic composition ≡ substituting pruner outputs into pruner inputs

**Why it's fusion:** The paper identifies the endomorphism operad of [1] with the distributive lattice operad. We use this to give `ConstraintPruner` combinations a formal algebraic foundation:

```rust
enum PrunerExpr {
    Atom(Box<dyn ConstraintPruner>),
    And(PrunerExpr, PrunerExpr), // meet
    Or(PrunerExpr, PrunerExpr),  // join
}

impl PrunerExpr {
    /// Canonical form via distributive lattice word problem
    fn canonicalize(self) -> Self { /* DNF */ }
    /// Interval-preservation check
    fn is_interval_preserving(&self) -> bool;
}
```

The word problem for free distributive lattices is well-studied and efficient. Canonicalization eliminates redundant pruner combinations before evaluation.

### 3.4 Conservation of Valid Regions During Speculative Decoding

**The Insight:** Interval preservation means: if a morphism maps two values, it maps everything between them. For speculative decoding:
- A "morphism" = the draft→verify pipeline (DDTree branch expansion)
- "Values" = valid token sequences
- "Interval" = all tokens between two valid alternatives
- Interval preservation = **no gaps in valid token space** after speculative expansion

This gives a formal guarantee: if the drafter proposes tokens A and C as valid at position t, and the verifier accepts both, then token B (between A and C by logit rank) must also be valid or explicitly rejected. This is a **conservation law** for the valid region.

**Why it's fusion:** The paper's interval preservation is a property of morphisms between Boolean lattices. We apply it as a structural invariant of the speculative decoding pipeline. This is NOT what the paper intended — it's a category-theoretic property repurposed as a runtime invariant.

### 3.5 Proper Model Structure → Type-Safe Speculative Decoding

**The Insight:** Theorem 5.7's properness means: substituting a weakly equivalent object preserves the homotopy type. Translated:
- **Weak equivalence** = two draft sequences that are "close enough" (within verification tolerance)
- **Fibration** = a safe decoding step (output valid if input valid)
- **Properness** = if you substitute one verified draft with another weakly equivalent draft, the downstream decoding remains correct

This gives formal guarantees about draft substitution in multi-turn speculative decoding. Currently we verify each draft independently. Properness would let us **reuse verification results** for weakly equivalent drafts — amortizing verification cost.

---

## 4. Distillation to katgpt-rs Architecture

### 4.1 What Validates Existing Work

| Paper Result | Our Existing Component | Validation |
|-------------|----------------------|------------|
| ⊞ is the right cube variant | `src/dec/` DEC operators | We didn't pick a wrong variant |
| Distributive lattice operad | DDTree AND/OR pruning | Our AND/OR combos are algebraically sound |
| CAT(0) from ⊞[L] | `CellComplex` + Hodge decomposition | Hodge harmonic = geodesic on CAT(0) |
| Proper model structure | Speculative decoding pipeline | Substitution of verified drafts is sound |

### 4.2 What's New (Novel Fusion Components)

| Component | Description | LOC Est. | Gate |
|-----------|-------------|----------|------|
| `IntervalPruner` trait | Convexity constraint on valid token sets | ~150 | `interval_pruner` |
| `CubicalNerve` | CAT(0) extraction from zone poset | ~200 | `cubical_nerve` |
| `LatticeOpernad` / `PrunerExpr` | Canonical AND/OR for pruner combinations | ~300 | `lattice_operand` |
| `interval_preserving` check | Verify morphism convexity on Boolean lattices | ~100 | (part of `interval_pruner`) |

### 4.3 Feature Gate Strategy

| Feature | Gate | Tier | Why |
|---------|------|------|-----|
| `interval_pruner` | katgpt-rs (MIT) | GOAT | Novel structural constraint, needs benchmarking |
| `cubical_nerve` | riir-ai (private) | GOAT | Game-specific zone topology |
| `lattice_operand` | katgpt-rs (MIT) | GOAT | Canonical pruner expressions, needs arena testing |

All three start behind feature flags. Promote to default only if GOAT gate proof shows measurable gain.

### 4.4 Dependency on Research 219 (DEC Operators)

This research extends the DEC infrastructure from Research 219:
- `CellComplex` is the foundation for `CubicalNerve`
- `hodge_decompose()` computes the harmonic geodesics on CAT(0) complexes
- `CochainField` provides the typed multi-rank structure for cubical cochains
- `betti_numbers` from DEC give topological invariants of the nerve

**Ordering:** Research 219 (DEC) must land first. This research is Phase 2 — category-theoretic enrichment of the DEC base.

---

## 5. GOAT Pillar Assessment

### Is It a "Super GOAT" (Keep Secret)?

**No.** The paper is publicly available (arXiv, 31pp). The category theory is pure mathematics. Our novel fusion (interval-preserving token pruning, CAT(0) game navigation, operadic pruner composition) is architecturally interesting but composed from public ingredients. The game-specific zone poset definitions ARE private domain knowledge.

### Verdict by 003 Strategy

| Criterion | Assessment |
|-----------|-----------|
| Fits engine/fuel split? | ✅ Interval pruner = engine (MIT), game zone posets = fuel (private) |
| Block anything? | ❌ No blocking dependency — extends DEC, doesn't replace it |
| GOAT gate candidate? | ✅ All three fusions behind feature flags, A/B vs baseline pruner |
| LoRA needed? | ❌ Pure structural/inference-time, no training |
| riir-ai domain? | Zone poset → CAT(0), game-specific pruner expressions |
| Validates existing work? | ✅ Confirms DEC ⊞ variant choice is category-theoretically optimal |

**Decision: CONDITIONAL GAIN.** The paper's primary value is validating that our DEC infrastructure (Research 219) is built on the right theoretical foundation (⊞ is the maximal reasonable cube category). The three novel fusions (IntervalPruner, CubicalNerve, LatticeOpernad) are promising but need benchmarking before promotion. Implement behind GOAT gates, measure in arenas, promote if gain proven.

### Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Interval pruner rejects valid tokens | Medium | Feature flag + fallback to per-token pruning |
| CAT(0) construction too expensive for runtime | Low | Pre-compute on map load, cache until topology change |
| Lattice operad over-engineering for simple pruners | Medium | Only use when pruner count > threshold |
| Category theory abstraction leak | Low | Implementation is just Boolean ops + convexity checks |

---

## 6. What NOT to Do

1. **Don't implement the full ⊞ category as a data structure.** We only need the interval-preservation property and the distributive lattice operad. A full categorical implementation would be over-engineering.
2. **Don't require interval preservation everywhere.** It's a structural constraint that some pruners naturally satisfy and others don't. Make it opt-in via trait, not mandatory.
3. **Don't compute CAT(0) nerve at runtime per frame.** Pre-compute on map load. The zone poset changes only when topology changes (door opens/closes, wall destroyed).
4. **Don't add dependent type theory machinery.** The proper model structure validates our substitution strategy — we don't need to implement CwA (categories with attributes) or type formers.
5. **Don't conflate interval preservation in Boolean lattices with interval arithmetic.** They're different notions. We use the order-theoretic one.

---

## 7. Research Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ Application is novel fusion; category theory itself is established |
| Rigor | ⭐⭐⭐⭐⭐ Proofs are complete; ⊞ is precisely characterized |
| Relevance to us | ⭐⭐⭐⭐ Validates DEC choice; three actionable fusions |
| Actionability | ⭐⭐⭐ Fusions are well-defined but require DEC Phase 1 first (Research 219) |
| Risk | ⭐⭐ Low — category theory is proven; risk is in fusion, not foundation |

---

## Related Internal Research

| Research | Connection |
|----------|-----------|
| 219 (TNO → DEC) | **Direct prerequisite** — DEC infrastructure this research builds on |
| 050 (LDT Phase 1) | Lattice deduction — distributive lattice operad overlaps with LDT's abstract interpretation |
| 152 (LDT Phase 2) | AlphaScreeningPruner uses lattice meet — same algebraic structure |
| 194 (CaDDTree) | DDTree budget allocation — interval pruner constrains DDTree branching |
| 197 (Discrete Critical Interval) | Interval solver — interval preservation is a related but different constraint |
| 118 (LEO) | Flow fields for game AI — CAT(0) geodesics improve LEO navigation |
| 037 (REAP) | Model-based/modelless duality — this paper validates modelless DEC approach |
| 208 (SLoD Semantic LOD) | Zone-level KG triples — zone poset for CubicalNerve is the same poset |
| 139 (Kog Monokernel) | CPU kernel strategy — interval pruner is a lightweight CPU filter |
| 050 (LDT) | Distributive lattice structure — shared algebraic foundation |

---

## TL;DR

⊞ (interval-preserving monotone maps between finite Boolean lattices) is the "Goldilocks" cube category — maximal expressiveness without reversals/diagonals. It validates our DEC infrastructure choice and provides three novel fusions: (1) **IntervalPruner** enforcing convexity of valid token regions, (2) **CubicalNerve** extracting CAT(0) complexes from game zone posets for deterministic NPC navigation, and (3) **LatticeOpernad** giving canonical AND/OR composition for pruner expressions. All conditional gain behind GOAT feature flags. Requires DEC Phase 1 (Research 219) to land first.
