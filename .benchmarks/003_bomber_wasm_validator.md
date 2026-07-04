# Benchmark 003: Bomber WASM Validator — Native vs WASM Performance

**Plan**: 034 (Bomber WASM Validator)
**Test**: `cargo test --test bomber_wasm_bench --features bomber-wasm --release -- --nocapture`

## Revision history

| Date | Runtime | Notes |
|---|---|---|
| 2025-05-12 | wasmtime (JIT) | Original Plan 034 gate — kept below as the historical record. |
| 2026-07-04 | wasmi (interpreter) | **Current.** Plan 167 migrated runtime to wasmi for dependency/reproducibility; numbers re-verified on a clean release build (post-Issue-016 fix, ~1.2s total runtime — not thermal-throttled). See [Issue 017](../../riir-ai/.issues/017_wasm_validator_perf_regression_wasmi.md). |

## Setup

- **WASM Module**: `bomber_validator.wasm` (39.3 KB, built with `--release`, wasmi/simd enabled)
- **Grid**: 13×13 Bomberman arena (seed=42)
- **Iterations**: 10,000 per micro-benchmark (100 warmup)
- **Game sim**: 200 ticks × 4 players × 6 actions = 4,800 checks per game
- **Hardware**: macOS (Apple Silicon, release build)

## Current results — wasmi (2026-07-04, post-Issue-016 fix)

### Per-call overhead

| Metric | Native Rust | WASM (wasmi) | Overhead | Target | Status |
|--------|-------------|--------------|----------|--------|--------|
| `is_safe_action` (Up, no bombs) | 2 ns | 3.00 µs | 1359× | < 10µs | ✅ |
| `is_safe_action` (Down, no bombs) | 2 ns | 3.03 µs | 1426× | < 10µs | ✅ |
| `is_safe_action` (Left, no bombs) | 2 ns | 3.07 µs | 1364× | < 10µs | ✅ |
| `is_safe_action` (Right, no bombs) | 2 ns | 3.11 µs | 1310× | < 10µs | ✅ |
| `is_safe_action` (Bomb, no bombs) | 76 ns | 4.87 µs | 64× | < 10µs | ✅ |
| `is_safe_action` (Wait, no bombs) | 2 ns | 3.13 µs | 1567× | < 10µs | ✅ |
| `is_safe_action` (Up, 3 bombs) | 5 ns | 3.44 µs | 651× | < 10µs | ✅ |
| `action_relevance` (Up, 3 bombs) | — | 4.19 µs | — | < 20µs | ✅ |
| `action_relevance` (Bomb, 3 bombs) | — | 3.42 µs | — | < 20µs | ✅ |

### Full-game simulation (200 ticks × 4 players × 6 actions = 4,800 checks)

| Metric | Native | WASM individual | WASM batch | Target |
|--------|--------|-----------------|------------|--------|
| Per game | 0.17 ms | 17.83 ms | **3.73 ms** | < 50ms |
| Per check (avg) | 30 ns | 3,714 ns | **778 ns** | — |
| Batch speedup | — | — | **4.8×** | — |

### Per-tick micro (1 tick, 4 players × 6 actions = 24 calls)

| Path | Latency | Note |
|---|---|---|
| Individual (24 × `is_safe_action`) | 87.72 µs | cold path |
| Individual (24 × `action_relevance`) | 93.77 µs | cold path |
| Batch (1 × `batch_validate`) | **17.89 µs** | hot path |
| Batch (1 × `batch_relevance`) | **22.63 µs** | hot path |
| Batch speedup (`is_safe_action`) | 4.9× | |
| Batch speedup (`action_relevance`) | 4.1× | |

### Infrastructure

| Metric | wasmtime (2025-05) | wasmi (2026-07) | Delta |
|--------|--------------------|-----------------|-------|
| WASM instantiation (one-time) | 4.10 ms | **157.29 µs** | 26× faster (wasmi interpreter compile ≪ Cranelift JIT) |
| Serialization (no bombs, 13×13, vec) | 0.15 µs | 0.19 µs | within noise |
| Serialization (3 bombs, 13×13, vec) | 0.19 µs | 0.14 µs | within noise |
| Zero-copy `ZeroCopyStateBuffer::serialize` | — | 0.04 µs | new (Plan 037) |
| Zero-copy speedup vs vec | — | **4.2×** (185 ns → 44 ns) | new |
| WASM binary size | 33.0 KB | 39.3 KB | +6.3 KB (wasmi/simd opcodes) |

## Analysis

### Per-call regression (wasmtime → wasmi): 6–13×

Plan 167's migration from wasmtime (Cranelift JIT) to wasmi (pure-Rust interpreter) regressed individual `is_safe_action` from ~0.5 µs to ~3.0 µs. The native movement check is trivial (one bounds check + one array lookup, ~2 ns), so the WASM calling overhead dominates. The fixed ~3 µs floor is wasmi's interpreter dispatch + memory copy + fuel accounting, not algorithmic cost. This is an acceptable cost for the dependency/reproducibility win (no native JIT dependency, deterministic across hosts) **provided production uses the batch API**.

### Batch API is load-bearing

The individual path (3.7 µs/check) leaves only 2.7× headroom against the 10 µs budget — too thin for noisy production hosts. The **batch API** (778 ns/check, 11× headroom) is now the production path; the individual call exists as a cold-path convenience. Plan 037's batch API was originally framed as an optimization; under wasmi it is a load-bearing correctness-of-perf dependency.

### Bomb action: smallest regression

Bomb placement runs the same BFS-based `has_escape_route` (169-cell BFS) on both native and WASM. The algorithmic work dominates the interpreter overhead, so WASM is "only" 64× slower (vs 1300× for trivial moves). This is the realistic worst case — bomb placement is the complex safety check.

### Full game: 7.4× slower individually, 1.5× slower batched

A full 200-tick, 4-player game takes 17.8 ms individually (2.8× headroom) or **3.73 ms batched (13× headroom)**. The 50 ms target is met by both paths, but only batch has comfortable margin for noisy hosts.

### Instantiation: wasmi is 26× faster

wasmi's interpreter compile (157 µs) is dramatically faster than wasmtime's Cranelift JIT (4.1 ms). This matters for the hot-swap-between-rounds pattern (Pillar 5): a tournament round can hot-swap a new validator in 0.16 ms vs 4.1 ms. **This is an unexpected wasmi win** that partially offsets the per-call regression.

### G1 correctness — RESOLVED (Issue 016)

`test_ab_correctness_many_states` now reports **0 critical mismatches** across 35,000 comparisons (100 grids × 5 bomb configs × 4 players × ~7 actions). The 1,996 bomb-action differences are all "WASM stricter than native" (native allows, WASM rejects — the safe direction), and are expected per Plan 034's design (the WASM validator adds a conservative BFS safety check). Issue 016's three bugs (bomb-token stride, missing Detonate action, wasmi/simd feature flag) were fixed in commits `ad37ea77` (katgpt-rs) and `f2db5c53` (riir-ai).

## Conclusion (current — wasmi)

| Target | Result (batch) | Result (individual) | Margin |
|--------|----------------|----------------------|--------|
| `is_safe_action` < 10µs | **778 ns/check** | 3.0–4.9 µs/call | batch: 11× headroom; individual: 2–3× headroom |
| `relevance` < 20µs | **943 ns/check** (22.6µs/24) | 3.4–4.2 µs/call | both pass comfortably |
| Full game < 50ms | **3.73 ms** | 17.83 ms | batch: 13× headroom; individual: 2.8× headroom |
| G1 correctness | **0 critical mismatches** / 35,000 | — | ✅ PASS |

**Production guidance:** use the batch API (`batch_validate` / `batch_relevance`). The individual call path is a cold-path convenience and leaves insufficient headroom for noisy hosts under wasmi.

## Historical record — wasmtime (2025-05-12, Plan 034 original gate)

> Kept for reference. These numbers are **superseded** by the wasmi section above; they reflect the pre-Plan-167 wasmtime runtime and are no longer reproducible from `develop`.

### Per-call overhead (wasmtime)

| Metric | Native Rust | WASM (wasmtime) | Overhead | Target | Status |
|--------|-------------|------------------|----------|--------|--------|
| `is_safe_action` (Up, no bombs) | 2 ns | 502 ns | 251× | < 10µs | ✅ |
| `is_safe_action` (Down, no bombs) | 2 ns | 503 ns | 251× | < 10µs | ✅ |
| `is_safe_action` (Left, no bombs) | 2 ns | 505 ns | 224× | < 10µs | ✅ |
| `is_safe_action` (Right, no bombs) | 2 ns | 509 ns | 226× | < 10µs | ✅ |
| `is_safe_action` (Bomb, no bombs) | 492 ns | 543 ns | 1.1× | < 10µs | ✅ |
| `is_safe_action` (Wait, no bombs) | 2 ns | 446 ns | 255× | < 10µs | ✅ |
| `is_safe_action` (Up, 3 bombs) | 6 ns | 470 ns | 77× | < 10µs | ✅ |
| `action_relevance` (Up, 3 bombs) | — | 550 ns | — | < 20µs | ✅ |
| `action_relevance` (Bomb, 3 bombs) | — | 370 ns | — | < 20µs | ✅ |

### Full-game simulation (wasmtime)

| Metric | Native | WASM | Overhead |
|--------|--------|------|----------|
| Per game (200 ticks × 4 players × 6 actions) | 0.68 ms | 2.41 ms | 3.6× |
| Per check (avg across all actions) | 141 ns | 502 ns | 3.6× |

### Infrastructure (wasmtime)

| Metric | Value |
|--------|-------|
| WASM instantiation (one-time) | 4.10 ms |
| Serialization (no bombs, 13×13) | 0.15 µs |
| Serialization (3 bombs, 13×13) | 0.19 µs |
| WASM binary size | 33.0 KB |

## Repro

```bash
# 1. Build the WASM validator (riir-ai, release for accurate perf)
cd riir-ai
cargo build --example bomber_validator --target wasm32-unknown-unknown --release -p riir-validator-sdk

# 2. Run the perf bench (katgpt-rs)
cd ../katgpt-rs
cargo test --test bomber_wasm_bench --features bomber-wasm --release -- --nocapture

# 3. Run the A/B correctness gate (G1)
cargo test --test bomber_wasm_ab --features bomber-wasm --release -- --nocapture test_ab_correctness_many_states
```
