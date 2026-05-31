# Plan 164: PhraseBoost — Context Trie Phrase Boosting for DDTree

> **Research:** 147 (Parakeet Context Trie Phrase Boosting)
> **Source:** [Frikallo/parakeet.cpp](https://github.com/Frikallo/parakeet.cpp) — `phrase_boost.hpp`
> **Feature Gate:** `phrase_boost` (default-OFF until GOAT proves gain)
> **Priority:** Medium — composable enhancement, no model training required
> **Date:** 2026-05-31

---

## Summary

Implement a `PhraseBoostPruner` that wraps any `ScreeningPruner` and adds domain-specific token biasing via a Context Trie. Zero training cost — phrases are provided at call site. Modeled after parakeet.cpp's phrase boosting, adapted to our DDTree + ScreeningPruner pipeline.

---

## Tasks

### T1: PhraseTrie — Compact Token-Level Trie
- [ ] Create `src/pruners/phrase_trie.rs` behind `#[cfg(feature = "phrase_boost")]`
- [ ] `PhraseTrieNode` with `children: Vec<Option<usize>>` (vocab-indexed, O(1) child lookup)
- [ ] `PhraseTrie::insert(token_ids: &[usize])` — insert single phrase
- [ ] `PhraseTrie::build(phrases: &[&str], encode_fn)` — bulk build from strings
- [ ] `PhraseTrie::get_boosted_tokens(active: &FixedBitSet) -> Vec<usize>` — union of children
- [ ] `PhraseTrie::advance(active: &mut FixedBitSet, token_id: usize)` — advance active states
- [ ] Unit tests: insert + lookup + advance roundtrip

### T2: PhraseBoostPruner — ScreeningPruner Wrapper
- [ ] Create `src/pruners/phrase_boost.rs` behind `#[cfg(feature = "phrase_boost")]`
- [ ] `PhraseBoostPruner<P: ScreeningPruner>` wrapping any inner pruner
- [ ] `relevance()` delegates to inner, adds normalized boost for boosted tokens
- [ ] Boost normalization: `boost_score / (1.0 + boost_score)` to stay in [0, 1+]
- [ ] Active state tracking: `HashMap<u128, FixedBitSet>` keyed by DDTree parent_path
- [ ] Pre-allocate FixedBitSet per path on first access, reuse with `clear()`
- [ ] Default `boost_score = 0.833` (= 5.0 / 6.0, matching parakeet's 5.0 in [0,1] scale)
- [ ] Register in `src/pruners/mod.rs` behind feature gate

### T3: GOAT Proof — Bomber Arena
- [ ] Add `phrase_boost` to test feature flags
- [ ] Create benchmark: Bomber arena 1000 rounds, release build
- [ ] A/B: `NoScreeningPruner` vs `PhraseBoostPruner<NoScreeningPruner>`
- [ ] Boost phrases: Bomber action tokens (bomb, wall, open, block, walk, idle)
- [ ] Metric: DDTree acceptance rate, win rate
- [ ] Pass criteria: acceptance rate improves ≥5%
- [ ] Target: `.benchmarks/021_phrase_boost_goat.md`

### T4: GOAT Proof — RIIR SynPruner
- [ ] Benchmark: SynPruner vs `PhraseBoostPruner<SynPruner>` on Rust token validation
- [ ] Boost phrases: Rust keywords + stdlib identifiers (~128 tokens)
- [ ] Metric: valid-node rate in DDTree
- [ ] Pass criteria: valid-node rate improves ≥3%

### T5: Performance Proof — Overhead Measurement
- [ ] Profile per-step overhead: phrase_trie advance + boost computation
- [ ] Must be <1μs per DDTree step
- [ ] If >1μs: optimize (consider flat bitvec instead of HashMap for active states)
- [ ] Document in benchmark file

### T6: Default-ON Decision (Post-GOAT)
- [ ] If T3 or T4 shows gain AND T5 shows no perf hurt → move to `default = ["phrase_boost"]`
- [ ] Update `Cargo.toml` default features
- [ ] Update README.md with phrase boosting section
- [ ] If no gain → keep feature-gated, document as "opt-in for domain-heavy workloads"

---

## Optimization Alignment

Per `optimization.md`:
- Pre-compute trie once at load (✅ `build()`)
- O(1) child lookup via Vec<Option<usize>> (✅ not HashMap)
- Pre-allocate FixedBitSet per path (✅ not HashSet)
- Zero alloc on hot path (✅ clear() + reuse)
- Keep inner relevance call branch-free (✅ `bool as f32` multiply)

---

## Feature Gate

```toml
[features]
phrase_boost = []  # Context trie phrase boosting for DDTree (Research 147, Plan 164)
```

**Default: OFF** until T3/T4/T5 prove gain.

After GOAT proof:
```toml
default = ["phrase_boost"]  # If T6 passes
```
