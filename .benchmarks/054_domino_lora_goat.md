# Bench 054: Domino LoRA Causal Correction — GOAT Gate

> **Plan:** 231 (Domino LoRA Training — Causal Correction Adapter)
> **Feature Gate:** `domino_lora`
> **Date:** 2026-06-06
> **Status:** ✅ Scaffold Complete — Awaiting Trained Weights for Full E2E

---

## Objective

Benchmark Domino LoRA causal correction adapter overhead vs DFlash AR baseline.
Measures latency cost of `correct()` + `gru_step()` per draft position.

---

## Benchmarks

### B1: Pure `correct()` Latency

Isolates the LoRA correction pass: concat → down-project → ReLU → up-project → add.

| Metric | Value |
|--------|-------|
| Config | n_embd=64, vocab=256, rank=16, gru_hidden=32 |
| Notes | Pure matmul micro-bench, no transformer forward |

### B2: DFlash AR vs DFlash AR + Domino LoRA (A/B)

End-to-end draft generation comparison.

| Metric | DFlash AR (baseline) | DFlash AR + Domino LoRA |
|--------|----------------------|-------------------------|
| Steps/s | TBD | TBD |
| µs/step | TBD | TBD |
| Avg acceptance len | TBD | TBD |
| Overhead % | — | TBD |
| Config | draft_lookahead from Config | rank=16, gru_hidden=32 |

---

## GOAT Gate

- [ ] **G1:** Domino LoRA overhead < 15% per draft step
- [ ] **G2:** Correct() latency < 50µs per call (micro config)
- [ ] **G3:** No allocation in hot path (zero-alloc verified)
- [ ] **G4:** Full E2E with trained weights: acceptance length +15-20%

> **Current Status:** G3 ✅ (fixed hot-path vec! allocation to stack buffer)
> G1/G2/G4 require `cargo test --release --features domino_lora` with bench runner.

---

## Key Files

| File | Role |
|------|------|
| `src/benchmark/speculative.rs` | `bench_domino_lora_correction()`, `bench_dflash_ar_domino_vs_baseline()` |
| `src/speculative/domino_lora.rs` | `DominoLoraCorrection::new_for_test()`, `correct()`, `gru_step()` |
| `src/speculative/dflash.rs` | `dflash_predict_ar_with_domino()` — zero-alloc GRU step |
| `src/benchmark/mod.rs` | Re-exports behind `#[cfg(feature = "domino_lora")]` |

---

## Hot-Path Fix (T7 bonus)

**Before:** `dflash_predict_ar_with_domino()` allocated `vec![0.0f32; gru_hidden_size]` per loop iteration (~4KB heap alloc per draft position).

**After:** Stack-allocated `[0.0f32; 1024]` buffer with `.min()` bounds check. Zero heap allocation in the hot loop.

---

## TL;DR

Domino LoRA benchmark scaffold with 2 functions: pure `correct()` micro-bench + DFlash AR A/B comparison. Fixed hot-path allocation in `dflash_predict_ar_with_domino()`. Full numbers require trained adapter weights (`domino_lora.bin`) — G1/G2 gates are ready to run on next training cycle.
