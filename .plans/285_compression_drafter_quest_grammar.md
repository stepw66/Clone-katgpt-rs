# Plan 285: Compression-Drafter — Quest Grammar Corpus-as-Format

**Date:** 2026-06-17
**Research:** [katgpt-rs/.research/256_GzipLM_Compression_Drafter.md](../.research/256_GzipLM_Compression_Drafter.md) (revised to GOAT)
**Private companion:** `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md` (exploration, not committed)
**Source paper:** [nathan.rs/gzip-lm](https://nathan.rs/posts/gzip-lm/) — beam-search text generation by compression
**Target:** `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs` (new module, open) + `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs` (game wiring, private)
**Cargo features:** `compression_drafter` (katgpt-core, opt-in), `quest_compression_draft` (riir-games, opt-in, depends on `quest_grammar` + `compression_drafter`)
**Status:** COMPLETE — GOAT FAILED (2 runs). Demoted. `TernaryDraftModel` remains default-on for Hot-tier quest grammar. `compression_drafter` open primitive + `quest_compression_draft` private wiring stay opt-in, unused, ready for any future Warm-tier consumer.

---

## Final Outcome (2026-06-17)

| Run | Algorithm | Scorer | G1 Diversity | G2 Latency | G3 Composition | G4 Zero-alloc |
|-----|-----------|--------|--------------|------------|----------------|---------------|
| Phase 3 | Fixed-candidate scoring | lz4_flex | 0.12× (1 unique) ❌ | 407× ❌ | ✅ PASS | ✅ PASS |
| Phase 7 | Beam search (nathan.rs algorithm) | MatchLengthScorer (inverted index) | 1.50× (12 unique) ❌ | 1077× ❌ | ✅ PASS | ✅ PASS |

Target: G1 ≥ 3× (24 unique), G2 ≤ 2× (~600ns).

**Honest verdict:** Compression-based generation loses to `TernaryDraftModel` template selection for Hot-tier quest grammar — fundamentally, not just implementationally. Beam search structurally needs ~1440 scorer calls/generation (~313µs); template selection needs 1 matvec + 1 hash (~290ns). Open primitive is correct and tested (15/15 + 6 + 4), just not the right tool for this tier. See `.issues/029_compression_beam_search_followups.md` for two unexplored paths (per-NPC corpus, Warm-tier repositioning) — neither pursued without concrete consumer.

---

## Goal

Ship a **Hot-tier modelless CompressionDrafter** that scores quest-continuation candidates by compressed length over the registered quest corpus. Replace `TernaryDraftModel::generate()`'s 8-hardcoded-template selection with corpus-as-scorer generation. **The corpus IS the wired format** — no parser, no struct deserialization; quest packs serialize as a single byte buffer + BLAKE3 commitment, loaded by feeding the bytes straight into the compressor window.

**GOAT gate:** on the existing QuestGrammarPipeline benchmark, compression-drafter generation must (a) produce higher-vocabulary-diversity quest text than template selection (≥3× unique outputs on a 100-quest corpus) at (b) ≤2× the latency of template selection, and (c) compose cleanly with the existing freeze/thaw pipeline (BLAKE3-commit the corpus). If gate fails → demote, keep `TernaryDraftModel`.

---

## Phase 1 — Unblocking Skeleton (CORE — open primitive in katgpt-core)

### Tasks

- [x] **T1.1** Add `lz4_flex = "3"` to `katgpt-rs/crates/katgpt-core/Cargo.toml` under optional dep `lz4_flex` (feature-gated by `compression_drafter`). Pure Rust, no unsafe, BSD-2 license — compatible with MIT.
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs` with:
  - `pub trait CompressionDrafter { fn score(&mut self, ctx: &[u8], candidate: &[u8]) -> i32; fn score_batch(&mut self, ctx: &[u8], candidates: &[&[u8]]) -> Vec<i32>; fn corpus(&self) -> &[u8]; }`
  - `pub struct Lz4FlexDrafter { corpus: Vec<u8>, scratch: Vec<u8> }` — wraps `lz4_flex::compress_prepend_size` for batched scoring.
  - `score(ctx, candidate)` returns `compressed_len(ctx) - compressed_len(ctx + candidate)` (higher = more compressible = more likely). Negative values mean candidate added entropy.
  - Zero-allocation hot path: reuse `scratch` buffer across calls.
  - Corpus-append `pub fn append(&mut self, bytes: &[u8])` for online learning.
- [x] **T1.3** Add `compression_drafter = ["dep:lz4_flex"]` feature to `katgpt-core/Cargo.toml`. Export the module from `lib.rs` behind `#[cfg(feature = "compression_drafter")]`.
- [x] **T1.4** Unit tests in `compression_drafter.rs`:
  - `score_test_repeated_pattern_dominates()` — corpus `"guard needs sword guard needs potion"`, score `"guard needs"` > score `"king finds"`.
  - `score_test_unseen_byte_penalized()` — corpus without `b'z'`, score of any candidate containing `b'z'` < score of corpus-only-byte candidate.
  - `append_test_grows_corpus()` — append, then re-score (must change).
  - `batch_score_consistent_with_single()` — sum of single-call scores equals batch-call scores within ±1 byte (lz4_flex header overhead).
  - `zero_alloc_no_regression_under_load()` — 10k calls in a tight loop, `scratch` capacity doesn't grow.
- [x] **T1.5** Add a tiny example `katgpt-rs/examples/compression_drafter_01_basic.rs` showing: build corpus from 8 hardcoded S-V-O triples, score 4 candidates, pick argmax, append the winner. Mirror of `phrase_boost` example style.

## Phase 2 — Quest Grammar Wiring (PRIVATE — riir-games)

### Tasks

- [x] **T2.1** Add `compression_drafter` to `katgpt-rs` re-export from `katgpt-rs/src/lib.rs` (passthrough, like `micro_belief`). Add `quest_compression_draft = ["quest_grammar", "katgpt-rs/compression_drafter"]` to `riir-ai/crates/riir-games/Cargo.toml`.
- [x] **T2.2** Create `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs` with:
  - `pub struct CompressionQuestDrafter { inner: katgpt_core::compression_drafter::Lz4FlexDrafter, }`
  - `pub fn from_registered_quests(quests: &[String]) -> Self` — join with `\n`, feed as initial corpus.
  - `pub fn generate(&self, ctx: &str, candidates: &[&str]) -> Option<&str>` — score each candidate as `ctx + cand`, return argmax. Returns `None` if `candidates` empty.
  - Drop-in API match with `TernaryDraftModel::generate` where possible (different signature — `candidates` parameter, since we score rather than template-select).
- [x] **T2.3** Wire  *(partial — drafter exists but NOT wired into QuestGrammarPipeline; feature stays opt-in, no default behavior change)* into `QuestGrammarPipeline`: add a `drafter_mode: QuestDrafterMode` enum `{ Ternary, Compression, Hybrid }` to `QuestGrammarConfig`. When `Compression`, route `generate_quest` through `CompressionQuestDrafter`. When `Hybrid`, use ternary for fast path + compression for diverse fallback.
- [x] **T2.4** Snapshot integration: implement `CorpusSnapshot { bytes: Vec<u8>, blake3: [u8; 32] }` in `quest_grammar/freeze_thaw.rs`. Add `fn snapshot_corpus(&self) -> CorpusSnapshot` and `fn restore_corpus(snapshot: &CorpusSnapshot) -> Result<Self>`. **The corpus IS the wired format** — no separate struct serialization; quest packs serialize as `[QuestPackHeader][CorpusSnapshot]` directly.
- [x] **T2.5** Update *(skipped — GOAT FAILED, no demo update needed; basic example exists in katgpt-rs)* `quest_grammar_demo.rs` example to show the corpus-as-format flow: register 8 quests → snapshot corpus to bytes → reload from bytes → generate. BLAKE3 roundtrip assert.

## Phase 3 — Benchmark (GOAT gate proof)

### Tasks

- [x] **T3.1** Create `katgpt-rs/.benchmarks/285_compression_drafter_goat.md` with this plan's G1–G3 protocol.
- [x] **T3.2** G1 — Diversity gate: on a 100-quest synthetic corpus (variations of S-V-O triples), generate 100 quests via (a) `TernaryDraftModel::generate` (current) vs (b) `CompressionQuestDrafter::generate`. Measure unique output count. **Pass:** compression-drafter produces ≥3× more unique outputs (template selection is capped at 8 hardcoded templates).
- [x] **T3.3** G2 — Latency gate: bench `generate()` per-call latency on a 30KB corpus. **Pass:** compression-drafter p99 < 2× ternary p99 (ternary is ~µs, lz4 on 30KB is ~µs-to-tens-of-µs).
- [x] **T3.4** G3 — Composition gate: freeze/thaw roundtrip preserves generation determinism. Same corpus → same outputs. BLAKE3 verifies integrity.
- [x] **T3.5** Document results in `.benchmarks/285_compression_drafter_goat.md`. Mark each gate PASS/FAIL with the actual number.

## Phase 4 — Promote or Demote (post-bench decision)

### Tasks

- [x] **T4.1** *(NOT EXECUTED — G1+G2 both failed; promote branch not taken. TernaryDraftModel stays default-on.)* Original: If G1+G2+G3 all pass: promote `quest_compression_draft` to default-on.
- [x] **T4.2** *(executed — G2 failed, both features stay opt-in)* Original: If G2 fails (latency too high): keep open primitive opt-in.
- [x] **T4.3** *(executed — tried beam search variant from Phase 5, G1 still failed at 1.50×, demoted honestly)* Original: If G1 fails, try multi-byte candidate variant.
- [x] **T4.4** *(executed — committed with `feat:` prefix and "GOAT FAILED" tag; commits 13cffebb, 3c52905d in katgpt-rs and 8c562602, 36794f3e in riir-ai)*

## Initial Result — G1+G2 FAILED (2026-06-17)

The first bench run (Phase 3) produced:
- **G1 Diversity: FAIL 0.12×** — corpus dominates scoring, ctx is invisible. Same winner regardless of context.
- **G2 Latency: FAIL 407×** — lz4_flex on 2KB is ~50µs/call; ternary matvec is 125ns. lz4 is Warm-tier, not Hot-tier.
- G3 Composition: PASS. G4 Zero-alloc: PASS.

Root cause analysis:
1. **Wrong algorithm.** I implemented candidate-set-scoring. nathan.rs/gzip-lm actually uses **beam search** — extend byte-by-byte with horizon, the growing tail becomes the new scoring context. Fixed candidate sets collapse to the corpus-mode winner; growing tails produce diversity.
2. **Wrong backend.** lz4_flex is a general-purpose compressor. We don't need compressed *length* — we need a **match-length proxy** (longest suffix of ctx+candidate appearing in corpus). That's a much simpler computation amenable to inverted-index acceleration.

→ Add Phase 5 (beam search), Phase 6 (fast scorer), Phase 7 (re-bench).

## Phase 5 — Beam Search Algorithm (the real nathan.rs algorithm)

### Tasks

- [x] **T5.1** Add `beam_search()` function to `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs`:
  - Signature: `pub fn beam_search<S: MatchScorer>(scorer: &S, seed_ctx: &[u8], alphabet: &[u8], horizon: usize, beam_width: usize, tail_len: usize) -> Vec<u8>`
  - Algorithm: nathan.rs/gzip-lm's actual beam search (see research §1.1). At each of `horizon` steps:
    1. For each beam × each alphabet byte, score `tail(seed_ctx + beam) + [byte]`.
    2. Keep top `beam_width` beams by cumulative score.
    3. The growing beam IS the tail — it becomes scoring context for the next step. This is the diversity source nathan.rs exploits.
  - Return the highest-scoring beam's bytes.
  - Anti-repeat: cap visible ctx at `tail_len` bytes (nathan.rs's `tail=80` trick — without it the compressor matches its own older output and loops).
- [x] **T5.2** Define `pub trait MatchScorer { fn score(&self, ctx: &[u8], candidate: &[u8]) -> i32; }` — a narrower trait than `CompressionDrafter`. Any scorer (lz4, match-length, future SIMD) implements it. Beam search is scorer-agnostic.
- [x] **T5.3** Implement `Lz4MatchScorer` wrapping the existing `Lz4FlexDrafter` as a `MatchScorer`. Used for correctness validation of beam search (slow but correct).
- [x] **T5.4** Unit tests for beam search:
  - `beam_search_produces_nonempty_output()`
  - `beam_search_extends_corpus_patterns()` — corpus `"guard needs sword\n"`, seed_ctx `"guard"`, output starts with `"guard needs"` or similar.
  - `beam_search_tail_prevents_loop()` — without tail_len cap, output repeats; with cap, diverse.
  - `beam_search_horizon_controls_length()` — output length ≤ horizon.

## Phase 6 — Fast Match-Length Scorer (latency fix)

### Tasks

- [x] **T6.1** Add `MatchLengthScorer` to `compression_drafter.rs`:
  - Inverted index: `byte_positions: Vec<Vec<u32>>` (256 buckets, positions in corpus).
  - `pub fn new(corpus: &[u8]) -> Self` — builds inverted index once. O(corpus_len).
  - `pub fn rebuild(&mut self, corpus: &[u8])` — for online-learning corpus updates.
  - `fn suffix_match_len(&self, ctx: &[u8], candidate: &[u8]) -> usize` — longest suffix of `ctx + candidate` appearing in corpus. Uses inverted index to skip non-matching positions. O(matches × avg_match_len).
  - Implements `MatchScorer`: `score(ctx, candidate) = suffix_match_len(ctx, candidate) as i32`.
- [x] **T6.2** Unit tests:
  - `match_length_finds_short_patterns()`
  - `match_length_finds_long_patterns()`
  - `match_length_zero_for_unseen()`
  - `inverted_index_accelerates_lookup()` — benchmark vs naive substring search.
- [x] **T6.3** Bench *(MatchLengthScorer hits 217ns/call — sub-µs, Hot-tier for a single call. Beam search multiplication is the bottleneck, not the scorer.)* `MatchLengthScorer::score()` on 2KB corpus: target < 1µs p99. If it passes, G2 becomes achievable.

## Phase 7 — Re-bench with Beam Search + Fast Scorer

### Tasks

- [x] **T7.1** Update `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs`:
  - Add `pub fn generate_beam(&mut self, seed_ctx: &str, alphabet: &[u8], horizon: usize, beam_width: usize, tail_len: usize) -> String` — uses `beam_search` with `MatchLengthScorer`.
  - Keep existing `generate()` (candidate-set) for backward compat.
  - The alphabet for quest grammar: the ASCII bytes appearing in the registered corpus (use `corpus_alphabet()` helper).
- [x] **T7.2** Update `bench_285_compression_drafter.rs`:
  - **G1 (beam)**: generate 100 quests via `generate_beam()` with varying seed contexts. Target ≥3× unique outputs vs ternary. **The growing tail is the diversity mechanism.**
  - **G2 (beam + MatchLengthScorer)**: bench `generate_beam()` latency. Target ≤2× ternary p99.
  - Keep G3 (freeze/thaw) and G4 (zero-alloc).
- [x] **T7.3** Run bench. Document PASS/FAIL per gate in `.benchmarks/285_compression_drafter_goat.md`.
- [x] **T7.4** Decision *(executed — G1+G2 both FAIL → demote, keep open primitive, create `.issues/029` for follow-ups)*:
  - All gates PASS → promote `quest_compression_draft` to default-on, demote `TernaryDraftModel`.
  - G1 PASS, G2 FAIL → keep lz4 backend for correctness, ship `MatchLengthScorer` as the fast path. Opt-in until SIMD optimization.
  - G1 FAIL → beam search doesn't help. Demote honestly, create issue for byte-level vs token-level beam search experiment.

---

## Why this is the right cut

1. **Honest verdict.** GOAT, not Super-GOAT. The user's pushback was correct: don't commit Super-GOAT before G1–G4 validation. The quest grammar angle is the clean win — template selection already works, this is a quality/scalability improvement, not a new feature.
2. **New wired format (user's insight).** The corpus IS the storage format. No QuestPack struct deserialization — the bytes go straight into the compressor window. BLAKE3 over the corpus bytes is the commitment. This is structurally identical to how we already commit weight snapshots, just on a different representation.
3. **Swap-in point exists.** `TernaryDraftModel::generate()` is the current template selector. `CompressionQuestDrafter::generate()` is the drop-in replacement. No invasive changes to `QuestGrammarPipeline`.
4. **Perf budget honest.** Hot tier (sub-ms), not plasma (µs). `lz4_flex` on 30KB corpus is ~tens of µs per call — fits the existing `hot_budget_ms` in `QuestGrammarConfig`. No speculative SIMD LZ77 kernel.
5. **4-repo discipline.** Open generic primitive in katgpt-core, private quest wiring in riir-ai, no chain IP, no training anywhere.

---

## TL;DR

GOAT plan: ship a Hot-tier CompressionDrafter that replaces `TernaryDraftModel`'s template selection with corpus-as-scorer generation. The corpus IS the wired format — bytes + BLAKE3, no parser. G1 diversity (3× unique outputs vs 8 templates), G2 latency (≤2× ternary), G3 freeze/thaw roundtrip. If all pass → default-on in riir-games. If G2 fails → opt-in. If G1 fails → demote + document negative result. Per-NPC plasma angle stays in `riir-ai/.research/137` as exploration, not committed.
