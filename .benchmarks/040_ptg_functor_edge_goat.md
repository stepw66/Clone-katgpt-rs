# Issue 040 — PTG × latent_functor edge GOAT gate benchmark

**Date:** 2026-07-04
**Primitive:** `FunctorPtg` composite + `FunctorEdgeParams` + `apply_functor_edge_into` + `functor_edge_gate`
**Feature flag:** `ptg_functor_edges` (implies `closure_instrument`)
**Status:** ✅ **DEFAULT-ON PROMOTED** (T7, 2026-07-04) — all GOAT gates pass, gain is modelless.
**Issue:** `.issues/040_ptg_latent_functor_edge_composition.md`
**Bench source:** `crates/katgpt-core/benches/bench_040_ptg_functor_edge.rs`

## TL;DR

All six GOAT gates pass on a pure-modelless primitive (cosine coherence +
sigmoid gate + in-place SAXPY). The composite design (`FunctorPtg` wraps an
unchanged `PrimitiveTransitionGraph`) makes wire-format safety a structural
invariant — no round-trip test could fail because no `PtgEdge` field was
added.

| Gate | Result | Detail |
|---|---|---|
| G1 correctness | ✅ PASS | 6 spec-match checks + 17 unit tests |
| G2 perf | ✅ PASS | **28.5 ns/call** at D=64 (target < 200 ns) — 7× margin |
| G2-alloc | ✅ PASS | 0 hot allocations (CountingAllocator, 1000-call loop) |
| G3 no-regression | ✅ PASS | default / `--no-default-features` / `--all-features` all clean; 1046 lib tests pass |
| G4 struct size | ✅ PASS | `size_of::<FunctorEdgeParams>() = 44` bytes (≤64 target) |
| G5/G6 modelless | ✅ PASS | Closed-form cosine + sigmoid + SAXPY, no training |

## The design pivot (T1 wire-format finding)

The issue's original proposal (`Option A`) was to extend `PtgEdge` with
`functor: Option<FunctorEdgeParams>` annotated with
`#[serde(skip_serializing_if, default)]`. **Empirical testing proved this is
broken**: postcard is positional, so `#[serde(default)]` cannot kick in on
EOF — old bytes serialized before the field was added fail to deserialize
with "Hit end of buffer". Plain `Option<T>` works for round-trip but changes
the wire format (+1 byte None discriminant per edge).

The shipped design (`FunctorPtg` composite) sidesteps the entire problem:

```rust
pub struct FunctorPtg {
    pub ptg: PrimitiveTransitionGraph,                 // byte-identical to bare PTG
    pub edge_functors: Vec<Option<FunctorEdgeParams>>, // parallel array, indexed by edge
}
```

The inner `ptg` is byte-identical to a bare PTG (verified by
`bare_ptg_bytes_identical_to_inner_ptg_bytes` unit test). Commitment is
preserved 100% (PTG bytes unchanged → BLAKE3 root unchanged). The functor
layer commits separately if/when a caller needs it.

## G2 perf breakdown

D=64 apply path on a single edge:

| Step | Op | Estimated cost |
|---|---|---|
| 1 | `simd_dot_f32(state, direction, 64)` — cosine numerator | ~20 ns |
| 2 | `simd_dot_f32(state, state, 64)` + `.sqrt()` — ‖s‖ | ~20 ns |
| 3 | `simd_dot_f32(direction, direction, 64)` + `.sqrt()` — ‖d‖ | ~20 ns |
| 4 | 1 division + 1 sigmoid | ~5 ns |
| 5 | in-place SAXPY `out[i] = state[i] + gate * direction[i]` for 64 elems | ~20 ns |
| **Total measured** | | **28.5 ns** (LLVM folds + SIMD) |

The measured 28.5 ns is well under the 200 ns target. The apply path uses
the existing `simd::simd_dot_f32` helper (Plan 290-era SIMD wrapper), so it
inherits the platform auto-vectorization already tuned for the rest of the
closure substrate.

## G2-alloc verification

The bench installs a `CountingAllocator` global allocator that increments
an `AtomicUsize` on every `alloc`. The hot-path test runs 1000 calls to
`apply_functor_edge_into` and asserts `alloc_delta == 0`. All `Vec` usage
(state, direction, out) is allocated outside the timed region.

## Wire-format safety (T3)

Two unit tests cover this:

1. `bare_ptg_bytes_identical_to_inner_ptg_bytes` — serializes a bare PTG
   and the inner `.ptg` field of a `FunctorPtg` constructed from an
   equivalent PTG. Asserts `bare_bytes == inner_bytes`.
2. `functor_ptg_serializes_and_round_trips` — serializes a `FunctorPtg`
   (with one functor set), deserializes it, and asserts edge count + the
   set functor match.

Because `FunctorPtg` is a composite (not a field on `PtgEdge`), wire-format
break is **structurally impossible** — the inner PTG serializes via the
unchanged `PrimitiveTransitionGraph` Serde impl.

## Reproduction

```bash
# Run the GOAT bench (default features now include ptg_functor_edges)
cargo bench -p katgpt-core --bench bench_040_ptg_functor_edge -- --nocapture

# Run the 17 unit tests
cargo test -p katgpt-core --lib closure::functor_edge

# Confirm feature matrix
cargo check -p katgpt-core --lib                          # default (now includes ptg_functor_edges)
cargo check -p katgpt-core --no-default-features --lib    # zero-dep baseline
cargo check -p katgpt-core --all-features --lib           # combo regression check
```

## Cross-references

- **Issue:** `.issues/040_ptg_latent_functor_edge_composition.md`
- **Implementation:** `crates/katgpt-core/src/closure/functor_edge.rs` (~520 LOC incl. 17 tests)
- **Bench:** `crates/katgpt-core/benches/bench_040_ptg_functor_edge.rs`
- **T1 audit:** `.issues/040_ptg_latent_functor_edge_composition.md` §"T1 Wire-Format Finding"
- **Sibling primitive:** Issue 039 (whole-architecture commitment) — once this ships,
  a `FunctorEdgeParams.direction_set` reference can be included in the architecture root.
- **riir-ai counterpart:** `riir-engine/src/latent_functor/arithmetic.rs` (Plan 273) —
  the full HLA-aware functor (rank-k, tropical, KARC consumers). katgpt-rs ships only
  the edge-apply numerics (cosine + sigmoid + SAXPY); the caller pre-resolves the
  direction vector from the `direction_set` BLAKE3 ref + `direction_index`.
