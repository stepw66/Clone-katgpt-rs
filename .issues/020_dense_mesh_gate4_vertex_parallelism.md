# Issue 020: DenseMesh Gate 4 — Vertex Parallelism for width-4 ≤ 2.5× bound

**Source**: Plan 266 Phase 7 gate 4 measurement — `tests/dense_mesh_goat_gates.rs::test_dense_mesh_gate4_hard_bound_width4_measured`
**Priority**: Medium (blocks true GOAT promotion of `dense_mesh`; gate is currently `#[ignore]` and documents the gap)
**Blocked**: No
**Depends**: Nothing (rayon already in tree; transformer.rs is local)

## Problem

The paper's ≤ 2.5× latency bound at width 4 assumes **vertex parameter sharing + parallel execution** — the 4 hidden nodes in a layer share one LLM and execute in parallel (batched GPU forward or rayon on CPU).

katgpt-rs's current `LayerwiseTopology::forward` runs all hidden nodes **sequentially**. As a result, the measured ratio at `[1, 4, 1]` topology is:

```
baseline (1×fwd)     │    0.20μs   │  1.00x
mesh[1,4,1] (5×fwd)  │    1.87μs   │  9.27x   ← measured, paper bound 2.5x
```

This is the expected sequential cost (5 forwards × ~1 vanilla + aggregation overhead). The bound is **unreachable** without parallel execution.

## Reproduction

```bash
# Gate 4 measurement (ignored by default — measurement, not pass/fail)
cargo test --release --features dense_mesh --test dense_mesh_goat_gates \
  test_dense_mesh_gate4_hard_bound_width4_measured -- --nocapture --include-ignored
```

See `.benchmarks/266_densemesh_goat.md` for full numbers.

## Proposed fix (two paths, both likely needed)

### Path A — Rayon across hidden nodes (smaller change)

Modify `LayerwiseTopology::forward` to use `rayon::scope` when the hidden layer width ≥ `gpu_width_threshold` (default 4). Each hidden node borrows `&TransformerWeights` shared, with its own `ForwardContext` + `MultiLayerKVCache` per thread.

Expected speedup at width 4: ~2.5× (4 parallel threads → ~1.5× wall-clock after overhead). Ratio drops from 9.27× → ~3.7×. Still over 2.5×.

**Cost:** ~50 LoC in `src/dense_mesh/topology.rs`. Thread-safety analysis on `DenseNode` (currently `&self` — good, no mutation needed).

### Path B — Batched forward in transformer.rs (larger change)

Add `forward_batched(ctx, weights, cache, tokens: &[usize], pos, config) -> Vec<&mut [f32]>` that processes N tokens at once, amortising KV cache writes and matmul setup.

Expected speedup at width 4: ~1.2× on top of rayon (better memory locality). Combined with Path A, ratio drops to ~3× → 2.5×.

**Cost:** ~200 LoC in `src/transformer.rs` (new entry point + re-organisation of the per-token loop). Risk of regressing existing forward paths.

### Recommendation

Start with **Path A** (rayon) — small, isolated, measurable. If ratio still > 2.5× after Path A, file a follow-up for Path B.

## Acceptance criteria

- [ ] Gate 4 test un-ignored (remove `#[ignore]`)
- [ ] Measured ratio at `[1, 4, 1]` topology ≤ 2.5× vanilla forward
- [ ] No regression in `prof_dense_mesh` aggregation/forward scaling tests
- [ ] No data race in `MultiLayerKVCache` (currently single-threaded — needs per-thread instances under rayon)

## References

- Research: `.research/234_DenseMesh_Latent_Node_Network.md` (gate 4)
- Plan: `.plans/266_densemesh_latent_node_network.md` Phase 7
- Benchmark: `.benchmarks/266_densemesh_goat.md`
- Paper: arXiv:2505.12741 §3.3 (vertex parameter sharing) + §3.1.3 (cost model)
