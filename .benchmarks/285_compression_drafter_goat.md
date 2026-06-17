# Benchmark 285: CompressionDrafter Quest Grammar — GOAT **FAILED**

**Date:** 2026-06-17
**Plan:** [285_compression_drafter_quest_grammar.md](../.plans/285_compression_drafter_quest_grammar.md)
**Research:** [256_GzipLM_Compression_Drafter.md](../.research/256_GzipLM_Compression_Drafter.md) (revised GOAT → demoted)
**Feature gates:** `compression_drafter` (katgpt-core), `quest_compression_draft` (riir-games)
**Status:** ❌ **GOAT FAILED** — CompressionQuestDrafter stays opt-in; `TernaryDraftModel` remains default-on.

---

## TL;DR

**The CompressionDrafter loses to TernaryDraftModel on both quality and latency for the quest grammar use case.** The open primitive (`compression_drafter` in katgpt-core) is sound and stays as an opt-in module; the quest grammar wiring is preserved as opt-in but **does NOT promote to default-on**. Honest negative result — the corpus-as-format insight is correct, but candidate-set-scoring is the wrong algorithm (we need beam search, see §Why this failed).

---

## GOAT Gate Results

| Gate | Pass criterion | Actual | Verdict |
|------|----------------|--------|---------|
| **G1 Diversity** | compression ≥ 3× unique outputs vs ternary | **0.12× (1 unique vs 8)** | ❌ **FAIL** |
| **G2 Latency** | compression ≤ 2× ternary p99 | **407× (50.875µs vs 125ns)** | ❌ **FAIL** |
| **G3 Composition** | freeze/thaw roundtrip preserves generation | ✓ matches; tamper detected | ✅ **PASS** |
| **G4 Zero-alloc** | scratch buffer doesn't grow over 1000 calls | ✓ stable | ✅ **PASS** |

**Run command:**
```bash
cd riir-ai
cargo test --features quest_compression_draft --test bench_285_compression_drafter --release -- --nocapture
```

---

## Why this failed (root cause analysis)

### G1 root cause: corpus dominates, ctx is invisible

The compression-drafter scores `compress(corpus + ctx + candidate) - compress(corpus + ctx)`. With a 2KB corpus and ctx of `"quest context N"` (~15 bytes), **ctx is 0.7% of the input**. The compressor's hash table is dominated by corpus patterns; ctx barely shifts the match search. Result: same winner across all 100 contexts (compression_unique = 1).

This is the **opposite failure** of nathan.rs's `tail=80` trick. nathan.rs uses a SHORT corpus (variable, primed by user) and a SHORTER ctx (80 bytes tail of recent output) — ctx and corpus are comparable in size. We have a LONG static corpus and a TINY dynamic ctx.

### G2 root cause: lz4_flex on 2KB is ~50µs, ternary is ~125ns

LZ4 is fast for a general-purpose compressor (~500MB/s) but it's still 400× slower than a ternary matvec over 64-dim weights. The Hot-tier latency assumption in Plan 285 was wrong: lz4 is **Warm-tier** (ms), not Hot (sub-ms), and definitely not Plasma (µs). Ternary wins by 3 orders of magnitude.

### G3 + G4 passed because they don't measure quality

G3 (BLAKE3 roundtrip) and G4 (zero-alloc scratch) test the infrastructure, not the algorithm. Both work — the corpus-as-format serialization is sound. The problem is the *scoring algorithm*, not the storage format.

---

## What this means

### The honest downgrade

The original verdict (R256 Super-GOAT → revised GOAT → now **demoted below GOAT**) was overclaimed at every step. The user's pushback to actually run the gate caught it. The correct verdict for the quest-grammar use case is **Pass (negative result)** — compression-as-scorer doesn't beat template selection here.

### What stays

1. **`compression_drafter` module in katgpt-core stays** — it's a sound primitive (6/6 unit tests pass), useful for any consumer that wants corpus-as-scorer. Generic, no game IP. Default-off.
2. **`CompressionQuestDrafter` and `CorpusSnapshot` stay opt-in** — they're correct code, just not the right tool for quest grammar. Future work might use them elsewhere.
3. **`TernaryDraftModel` remains default-on** for quest grammar — it won this gate fair and square.
4. **riir-ai/.research/137 (per-NPC plasma exploration)** stays as a research note. The G1 latency fit (sub-µs custom LZ77) is the only path that would beat ternary, and it's still unvalidated.

### What we learned

1. **The corpus-as-format insight is correct** — CorpusSnapshot's BLAKE3 roundtrip works perfectly. The format IS committable, tamper-evident, zero-parser. That part of the user's intuition was right.
2. **The candidate-scoring algorithm is wrong for this use case.** Compression-as-scorer needs **beam search** (nathan.rs's actual algorithm — extend candidates byte-by-byte with horizon), not "score a fixed candidate set". The fixed candidate set collapses to the corpus-mode winner regardless of ctx.
3. **lz4 is Warm-tier (ms), not Hot-tier (sub-ms).** The plasma tier (µs) would need a custom SIMD LZ77 over a tiny alphabet — still unvalidated.

### What would need to change to make this work

- **For diversity**: implement actual beam search (nathan.rs's algorithm). Don't score a fixed candidate set — extend byte-by-byte, beam_width=32, horizon=24. The output space becomes the corpus manifold, not the candidate enumeration.
- **For latency**: a custom plasma-tier LZ77 (sub-µs SIMD over tiny alphabet) — the `riir-ai/.research/137` exploration. lz4 is too slow.

Both are **non-trivial** and out of scope for this plan. They'd be a new plan if pursued.

---

## Action items

- [x] **T4.3 executed**: G1 failed, tried candidate-set-scoring, confirmed it doesn't produce useful signal for quest grammar.
- [x] **Demotion**: `compression_drafter` stays opt-in in katgpt-core. `quest_compression_draft` stays opt-in in riir-games. Neither promotes to default.
- [x] **README update**: NOT promoted — quest grammar README continues to describe TernaryDraftModel as the default drafter.
- [x] **Negative result documented** (this file).
- [ ] **Follow-up issue**: should we create `.issues/029_compression_beam_search.md` to track the beam-search variant? (User decision.)

---

## Files touched

| File | Change |
|------|--------|
| `katgpt-rs/crates/katgpt-core/Cargo.toml` | Added `lz4_flex` optional dep + `compression_drafter` feature (stays opt-in) |
| `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs` | NEW: `CompressionDrafter` trait + `Lz4FlexDrafter` impl + 6 unit tests (all pass) |
| `katgpt-rs/crates/katgpt-core/src/lib.rs` | Added `pub mod compression_drafter;` behind feature gate |
| `katgpt-rs/Cargo.toml` | Added `compression_drafter` passthrough feature + example registration |
| `katgpt-rs/src/lib.rs` | Added `pub use katgpt_core::compression_drafter;` re-export |
| `katgpt-rs/examples/compression_drafter_01_basic.rs` | NEW: corpus-as-model demo |
| `katgpt-rs/.research/256_GzipLM_Compression_Drafter.md` | Status revised: Super-GOAT → GOAT → (now) **demoted** with negative-result note |
| `katgpt-rs/.plans/285_compression_drafter_quest_grammar.md` | NEW: plan with G1-G4 gate |
| `katgpt-rs/.benchmarks/285_compression_drafter_goat.md` | NEW: this file (negative result) |
| `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md` | Status revised: exploration, not committed Super-GOAT |
| `riir-ai/crates/riir-games/Cargo.toml` | Added `quest_compression_draft` opt-in feature |
| `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs` | NEW: `CompressionQuestDrafter` + 6 unit tests (all pass) |
| `riir-ai/crates/riir-games/src/quest_grammar/freeze_thaw.rs` | Added `CorpusSnapshot` + `Blake3Mismatch` / `Truncated` errors + 4 unit tests (all pass) |
| `riir-ai/crates/riir-games/src/quest_grammar/mod.rs` | Registered `compression_draft` module behind feature gate |
| `riir-ai/crates/riir-games/tests/bench_285_compression_drafter.rs` | NEW: GOAT bench (G1 FAIL, G2 FAIL, G3 PASS, G4 PASS) |

---

## TL;DR

Plan 285 GOAT **FAILED** on G1 (diversity 0.12× vs 3× target) and G2 (latency 407× vs 2× target). G3 (freeze/thaw composition) and G4 (zero-alloc) passed cleanly. **The open primitive and quest wiring stay as opt-in modules** — the code is correct and the CorpusSnapshot format works, but compression-as-scorer is the wrong algorithm for fixed-candidate-set quest generation. The right algorithm (nathan.rs's actual beam search) is a follow-up plan. TernaryDraftModel remains default-on. Honest negative result — the user's pushback to actually run the gate caught an overclaim that would have shipped as fake-GOAT.
