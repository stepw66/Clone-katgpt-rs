---
name: rust-optimize
description: >-
  Optimize Rust code until nothing left to improve. Loops automatically.
  Use when the user says "optimize" or "/optimize".
disable-model-invocation: true
---

# Rust Optimization

Optimize `*.rs` files in the target scope using the checklist below.
This is a LOOP. You keep optimizing until you cannot find anything to optimize.

**Full reference**: `.contexts/optimization.md` — consult for ambiguous cases. Contains concrete thresholds (5μs/10μs/1000-count Rayon floors, 50μs GPU launch overhead), BAD/GOOD anti-pattern code blocks, profiling methodology + template, WASM FFI batch template, Rayon-vs-std::iter-vs-tokio comparison table, and 12 "Don't" sections with examples.

## Termination

After each turn, end your response with exactly one of:

- `Continue optimizing remaining files.` — you made code changes this turn
- `No optimizations this pass.` — you read files but changed nothing

If you only summarized what you read without changing code, say "no optimizations this pass".

## Checklist

1. **Complexity**: O(n) scan → O(1) lookup, merge HashMap lookups, merge loops
2. **Allocations**: `String` → `&'static str`, pre-allocate, eliminate `.to_string()`
3. **Layout**: field reorder (u64→u32→u8), `#[repr(u8)]` enums, remove `#[repr(C)]`
4. **Arithmetic**: `f32` counters → `u32`/`u64`
5. **Iterators**: fix double `chunks_exact()`, use index arithmetic
6. **Concurrency**: `Arc<RwLock<HashMap>>` → `papaya`, `Mutex<u32>` → `AtomicU32`
7. **SIMD**: chunked loops, branch-free inner loops
8. **Caching**: pre-compute lookup tables, compute once not N×M
9. **Errors**: `unwrap()` → `?`, `let _ =` → `.log_err()`

## Rules

- Do NOT create plans or issues.
- Do NOT create new files unless necessary.
- Commit when done (`perf:` or `refactor:` prefix).
- Use sigmoid, not softmax.
