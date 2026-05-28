# Issue 101: NarrowingPruner::is_valid Uses O(n) Linear Scan Per Call

## Severity: Medium
## Files: `katgpt-rs/crates/katgpt-core/src/questbench.rs` (L608-623)

## Description
`is_valid()` is called in a hot inner loop (for every candidate token at every depth). It uses `Vec::contains()` which is O(n) linear scan on `valid_at_depth` and `narrowing[last]`. Called up to 256 times per `score_relevance_into` call, and repeatedly in `count_valid_extensions_with`.

Per optimization.md: "Don't: Linear scan for hot-path queries. Use O(1) precomputed index."

## Fix
Replace `Vec<usize>` with `Vec<bool>` bitmap indexed by token value for O(1) lookup:
```rust
struct NarrowingPruner {
    _vocab_size: usize,
    valid_at_depth: Vec<bool>,  // O(1) lookup by token index
    narrowing: Vec<Vec<bool>>,  // O(1) lookup per depth
}
```

## Impact
Medium — called in tight CSP-solver loops. Each O(n) scan is replaced with O(1) index lookup.
