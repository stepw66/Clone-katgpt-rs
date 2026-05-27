# 🟡 Perf: Data structure improvements and redundant computation

## Summary
Medium-priority optimizations: flat arrays instead of `Vec<Vec<T>>`, missing capacity hints, unnecessary clones, and redundant computation in nested loops.

---

## Flat Arrays Instead of `Vec<Vec<T>>`

`Vec<Vec<T>>` causes pointer-chasing (n separate heap allocations) and poor cache locality. A single flat `Vec<T>` with `i * width + j` indexing is one allocation and cache-friendly.

| File | Struct / Function | Current | Fix |
|------|-------------------|---------|-----|
| `cache_prune/sat.rs` L20 | `SummedAreaTable` | `&mut [Vec<f32>]` | Flat `Vec<f32>` with `i*n+j` indexing |
| `stiff_anomaly/subspace.rs` L17 | `StiffSoftDecomposition` | `Vec<Vec<f32>>` eigenvectors | Flat `Vec<f32>` with `k*d` layout |
| `turboquant/rotation.rs` L20 | `generate_rotation_matrix` | `Vec<Vec<f32>>` | Flat `Vec<f32>` (low priority — one-time init) |

---

## Missing `Vec::with_capacity()`

| File | Function | Fix |
|------|----------|-----|
| `transformer.rs` L1387 | `cluster_map_round_robin` inner Vecs | `Vec::with_capacity(cluster_size)` per inner Vec |
| `transformer.rs` L3359 | `PagedKVCache` page tables | `Vec::with_capacity(block_size / PAGE_SIZE + 1)` |
| `cache_prune/rolling_hash.rs` L220 | `find_matches` | `Vec::with_capacity(segments.len())` |
| `gdn2/types.rs` L110 | `Gdn2LayerState::new` heads Vec | `Vec::with_capacity(config.n_kv_head)` |

---

## Unnecessary Clones

| File | Issue | Fix |
|------|-------|-----|
| `katgpt-core/types.rs` L1122 | `with_overrides()` deep-clones entire `Config` including `Vec<String>` per routing decision | Use `Arc<Vec<String>>` for `lora_targets`, or accept `&mut Config` |
| `peira.rs` L729 | `predictor()` clones k×k matrix (`self.n.clone()`) every call | Pre-allocate scratch buffer, write in-place |
| `peira.rs` L850, L907 | `invert_spd` / `matmul` allocate 5 Vecs per predictor call | Pre-allocate scratch buffers in `PeiraCovariance` |
| `traits.rs` L629 | `DualLeoMixer::combine()` returns `Vec<f32>` — allocates per training step | Add `combine_into()` that writes to `&mut [f32]` |
| `dirichlet.rs` L58 | `functor_adjacency()` clones input `pairs.to_vec()` unnecessarily | Return `&[(usize, usize)]` or take ownership |
| `katgpt-core/types.rs` L1701 | `LoraAdapter` field ordering: `alpha: f32` splits `usize` fields → 4 bytes padding | Reorder: `a, b, rank, in_dim, out_dim, alpha` |
| `gdn2/types.rs` L83 | `Gdn2LayerState`: 1-byte `gate_config` between `Vec`s → 7 bytes padding | Move `gate_config` after all `Vec` fields |

---

## Redundant Computation in Nested Loops

### 1. `traits.rs` L684 — Goal norms recomputed per observation
**Issue**: In `update_goals_seen`, for each (obs, goal) pair, `norm_goal` is recomputed — same value for all observations of the same goal.
**Fix**: Pre-compute `norm_goals: Vec<f32>` once outside the obs loop:
```rust
let norm_goals: Vec<f32> = all_goals.iter()
    .map(|g| g.iter().map(|x| x * x).sum::<f32>().sqrt())
    .collect();
```

### 2. `transformer.rs` L2620, L3090, L3786 — Double RMSNorm
**Issue**: `forward_prefill`, `forward_paged`, and `forward_raven` all call `rmsnorm` twice sequentially on `ctx.x`:
```rust
crate::types::rmsnorm(&mut ctx.x);
ctx.xr[..n].copy_from_slice(&ctx.x[..n]);
crate::types::rmsnorm(&mut ctx.x);
```
The first norm output is saved as `xr`, then a second identical norm is applied. `forward_base` does NOT have double-norm.
**Fix**: Verify if this is intentional (double-norm architecture). If the second norm is redundant, removing it saves O(n_embd) per position. **This may be a correctness bug, not just perf.**

### 3. `spectralquant/spectral.rs` L67-95 — O(n⁴) Jacobi pivot scan
**Issue**: "Find largest off-diagonal" scan is O(n²) per sweep × 50 sweeps. For dim=128: 819K iterations just for pivot finding.
**Fix**: Use cyclic Jacobi (fixed-order iteration, skip max-finding) or maintain a priority queue. Low priority — one-time calibration cost.

---

## Effort Estimate
- Flat arrays: ~2-3 hours (careful index refactoring + testing)
- Capacity hints: ~30 minutes
- Clone removal: ~1-2 hours
- Redundant computation: ~1 hour
- Double RMSNorm investigation: ~30 minutes (verify correctness first)
