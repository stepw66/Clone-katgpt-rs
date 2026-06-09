# Plan 233: Core Optimization Audit — katgpt-core

## Summary
Apply optimization.md guidelines to `crates/katgpt-core/src/`. Focus on allocation elimination, O(n)→O(1) conversions, and zero-alloc hot paths.

## Tasks
- [x] P0: `mux/top_k.rs` — Replace full copy + sort with `select_nth_unstable_by` O(n) partial sort
- [x] P0: `linoss.rs` — Add `imex_step_inplace` to eliminate 2 Vec allocs per timestep in `draft` loop
- [x] P0: `linoss.rs` — Add `draft_into` with pre-allocated scratch buffers for zero-alloc drafting
- [x] P0: `sense/bandit.rs` — Eliminate intermediate Vec allocation in `average_reward`
- [x] P1: `sense/hotswap.rs` — Replace `Vec` + O(n) scan with fixed-size array indexed by `SenseKind as usize`
- [x] P1: `sense/hotswap.rs` — Replace `Mutex<bool>` with `AtomicBool`
- [x] P1: `sense/brain.rs` — Add `project_all_into` for zero-alloc batch projection
- [x] P1: `sense/batch.rs` — Use `project_all_into` with flat output buffer
- [x] P2: `shard_embedding.rs` — Use `simd_dot_f32` for projection dot products
- [x] P2: `types.rs` — Reorder `SenseModule` fields: `u64`/`[u64]` first, then `f32`, then `u8` fields
- [x] P3: `mux/span_pruner.rs` — Use `extract_top_k_peaks` returning `SmallVec<[f32; 8]>` via updated top_k

## GOAT Gate
- All changes are behind existing APIs (non-breaking additions)
- `extract_top_k_peaks` signature changes to return `[f32; N]` via new `extract_top_k_peaks_arr` for fixed-k callers
- Benchmark before/after with `cargo test --features mux_bfs -- --nocapture` for BFS path
