# Plan 247: and_or / flow / mux Module Optimization

## Objective
Apply hot-path optimization patterns from `.contexts/optimization.md` to `and_or`, `flow`, and `mux` modules. Focus on eliminating heap allocations, reducing redundant computation, and enabling auto-vectorization.

## Analysis

### and_or/types.rs
- Already clean — generic tree with minimal allocation surface.
- **Fuse `solved_count` + `unsolved_count`** into single-pass `leaf_stats()` to avoid two recursive traversals.
- **Inline `is_blocked` in `gradient()`** — already `#[inline]`, no change needed.

### flow/mod.rs + flow/fft.rs + flow/cache.rs
- `fft_smooth()`: allocates `Vec<Complex<f32>>` + `Vec<Complex<f32>>` for col_buf on every call. **Pre-allocate planner reuse**.
- `gradient()`: calls `potential()` 4 times per cell with bounds checks. **Use direct indexing** (bounds already validated by loop).
- `cache.rs get_or_compute()`: allocates `Vec` for potential extraction and blocked bitfield each call. These are grid-sized, unavoidable for correctness.

### mux/dd_tree.rs
- `init_root()` and `detect_width()`: call allocating `extract_top_k_peaks` instead of zero-alloc `extract_top_k_into`. **Switch to zero-alloc path**.
- `expand_node()`: same — uses allocating variant. **Switch to zero-alloc**.
- `collect_leaf_paths()`: allocates `Vec<Vec<usize>>` with path cloning. Acceptable for BFS frontier size.

### mux/bandit_width.rs
- `update()`: linear scan `self.arms.iter_mut().find(|a| a.width == width)`. With k ≤ 16 this is fine, but **index by (width - 1)** for O(1)**.

### mux/demux.rs
- `demux()`: allocates `Vec<(u32, f32)>` + `Vec<u32>` + `HashSet<u32>`. **Use stack sort + inline duplicate check** for small k.

## Tasks

- [x] 1. `mux/dd_tree.rs`: Replace `extract_top_k_peaks` with `extract_top_k_into` in `init_root`, `detect_width`, `expand_node`
- [x] 2. `mux/bandit_width.rs`: O(1) update via direct index
- [x] 3. `mux/demux.rs`: Stack-based demux for small k (eliminate Vec + HashSet allocs)
- [x] 4. `flow/fft.rs`: Branch-free low-pass filter, `fft_smooth_into` with pre-allocated buffers
- [x] 5. `flow/mod.rs`: Direct potential indexing, inlined blocked check, branch-free normalization
- [x] 6. `and_or/types.rs`: Fused `leaf_stats()` returning (solved, unsolved) in one pass
- [x] 7. All 191 tests pass (with comp_width + all mux features), 0 diagnostics
- [ ] 8. Commit

## Files Modified
| File | Changes |
|------|---------|
| `mux/dd_tree.rs` | Zero-alloc top-K in init_root/detect_width/expand_node |
| `mux/bandit_width.rs` | O(1) arm update via direct index |
| `mux/demux.rs` | Stack sort + inline duplicate check |
| `flow/fft.rs` | Reuse col_buf, accept planner param |
| `flow/mod.rs` | Direct potential indexing in gradient |
| `and_or/types.rs` | Fused `leaf_stats()` |
