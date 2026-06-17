# Plan 285: Compression-Drafter — Quest Grammar Corpus-as-Format

**Date:** 2026-06-17
**Research:** [katgpt-rs/.research/256_GzipLM_Compression_Drafter.md](../.research/256_GzipLM_Compression_Drafter.md) (revised to GOAT)
**Private companion:** `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md` (exploration, not committed)
**Source paper:** [nathan.rs/gzip-lm](https://nathan.rs/posts/gzip-lm/) — beam-search text generation by compression
**Target:** `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs` (new module, open) + `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs` (game wiring, private)
**Cargo features:** `compression_drafter` (katgpt-core, opt-in), `quest_compression_draft` (riir-games, opt-in, depends on `quest_grammar` + `compression_drafter`)
**Status:** Active — Phase 1

---

## Goal

Ship a **Hot-tier modelless CompressionDrafter** that scores quest-continuation candidates by compressed length over the registered quest corpus. Replace `TernaryDraftModel::generate()`'s 8-hardcoded-template selection with corpus-as-scorer generation. **The corpus IS the wired format** — no parser, no struct deserialization; quest packs serialize as a single byte buffer + BLAKE3 commitment, loaded by feeding the bytes straight into the compressor window.

**GOAT gate:** on the existing QuestGrammarPipeline benchmark, compression-drafter generation must (a) produce higher-vocabulary-diversity quest text than template selection (≥3× unique outputs on a 100-quest corpus) at (b) ≤2× the latency of template selection, and (c) compose cleanly with the existing freeze/thaw pipeline (BLAKE3-commit the corpus). If gate fails → demote, keep `TernaryDraftModel`.

---

## Phase 1 — Unblocking Skeleton (CORE — open primitive in katgpt-core)

### Tasks

- [ ] **T1.1** Add `lz4_flex = "3"` to `katgpt-rs/crates/katgpt-core/Cargo.toml` under optional dep `lz4_flex` (feature-gated by `compression_drafter`). Pure Rust, no unsafe, BSD-2 license — compatible with MIT.
- [ ] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/compression_drafter.rs` with:
  - `pub trait CompressionDrafter { fn score(&mut self, ctx: &[u8], candidate: &[u8]) -> i32; fn score_batch(&mut self, ctx: &[u8], candidates: &[&[u8]]) -> Vec<i32>; fn corpus(&self) -> &[u8]; }`
  - `pub struct Lz4FlexDrafter { corpus: Vec<u8>, scratch: Vec<u8> }` — wraps `lz4_flex::compress_prepend_size` for batched scoring.
  - `score(ctx, candidate)` returns `compressed_len(ctx) - compressed_len(ctx + candidate)` (higher = more compressible = more likely). Negative values mean candidate added entropy.
  - Zero-allocation hot path: reuse `scratch` buffer across calls.
  - Corpus-append `pub fn append(&mut self, bytes: &[u8])` for online learning.
- [ ] **T1.3** Add `compression_drafter = ["dep:lz4_flex"]` feature to `katgpt-core/Cargo.toml`. Export the module from `lib.rs` behind `#[cfg(feature = "compression_drafter")]`.
- [ ] **T1.4** Unit tests in `compression_drafter.rs`:
  - `score_test_repeated_pattern_dominates()` — corpus `"guard needs sword guard needs potion"`, score `"guard needs"` > score `"king finds"`.
  - `score_test_unseen_byte_penalized()` — corpus without `b'z'`, score of any candidate containing `b'z'` < score of corpus-only-byte candidate.
  - `append_test_grows_corpus()` — append, then re-score (must change).
  - `batch_score_consistent_with_single()` — sum of single-call scores equals batch-call scores within ±1 byte (lz4_flex header overhead).
  - `zero_alloc_no_regression_under_load()` — 10k calls in a tight loop, `scratch` capacity doesn't grow.
- [ ] **T1.5** Add a tiny example `katgpt-rs/examples/compression_drafter_01_basic.rs` showing: build corpus from 8 hardcoded S-V-O triples, score 4 candidates, pick argmax, append the winner. Mirror of `phrase_boost` example style.

## Phase 2 — Quest Grammar Wiring (PRIVATE — riir-games)

### Tasks

- [ ] **T2.1** Add `compression_drafter` to `katgpt-rs` re-export from `katgpt-rs/src/lib.rs` (passthrough, like `micro_belief`). Add `quest_compression_draft = ["quest_grammar", "katgpt-rs/compression_drafter"]` to `riir-ai/crates/riir-games/Cargo.toml`.
- [ ] **T2.2** Create `riir-ai/crates/riir-games/src/quest_grammar/compression_draft.rs` with:
  - `pub struct CompressionQuestDrafter { inner: katgpt_core::compression_drafter::Lz4FlexDrafter, }`
  - `pub fn from_registered_quests(quests: &[String]) -> Self` — join with `\n`, feed as initial corpus.
  - `pub fn generate(&self, ctx: &str, candidates: &[&str]) -> Option<&str>` — score each candidate as `ctx + cand`, return argmax. Returns `None` if `candidates` empty.
  - Drop-in API match with `TernaryDraftModel::generate` where possible (different signature — `candidates` parameter, since we score rather than template-select).
- [ ] **T2.3** Wire into `QuestGrammarPipeline`: add a `drafter_mode: QuestDrafterMode` enum `{ Ternary, Compression, Hybrid }` to `QuestGrammarConfig`. When `Compression`, route `generate_quest` through `CompressionQuestDrafter`. When `Hybrid`, use ternary for fast path + compression for diverse fallback.
- [ ] **T2.4** Snapshot integration: implement `CorpusSnapshot { bytes: Vec<u8>, blake3: [u8; 32] }` in `quest_grammar/freeze_thaw.rs`. Add `fn snapshot_corpus(&self) -> CorpusSnapshot` and `fn restore_corpus(snapshot: &CorpusSnapshot) -> Result<Self>`. **The corpus IS the wired format** — no separate struct serialization; quest packs serialize as `[QuestPackHeader][CorpusSnapshot]` directly.
- [ ] **T2.5** Update `quest_grammar_demo.rs` example to show the corpus-as-format flow: register 8 quests → snapshot corpus to bytes → reload from bytes → generate. BLAKE3 roundtrip assert.

## Phase 3 — Benchmark (GOAT gate proof)

### Tasks

- [ ] **T3.1** Create `katgpt-rs/.benchmarks/285_compression_drafter_goat.md` with this plan's G1–G3 protocol.
- [ ] **T3.2** G1 — Diversity gate: on a 100-quest synthetic corpus (variations of S-V-O triples), generate 100 quests via (a) `TernaryDraftModel::generate` (current) vs (b) `CompressionQuestDrafter::generate`. Measure unique output count. **Pass:** compression-drafter produces ≥3× more unique outputs (template selection is capped at 8 hardcoded templates).
- [ ] **T3.3** G2 — Latency gate: bench `generate()` per-call latency on a 30KB corpus. **Pass:** compression-drafter p99 < 2× ternary p99 (ternary is ~µs, lz4 on 30KB is ~µs-to-tens-of-µs).
- [ ] **T3.4** G3 — Composition gate: freeze/thaw roundtrip preserves generation determinism. Same corpus → same outputs. BLAKE3 verifies integrity.
- [ ] **T3.5** Document results in `.benchmarks/285_compression_drafter_goat.md`. Mark each gate PASS/FAIL with the actual number.

## Phase 4 — Promote or Demote (post-bench decision)

### Tasks

- [ ] **T4.1** If G1+G2+G3 all pass: promote `quest_compression_draft` to default-on in `riir-games` default features. Update README quest grammar section. Demote `TernaryDraftModel` to legacy fallback (keep for `plasma_path` consumers, but document as default-OFF path).
- [ ] **T4.2** If G2 fails (latency too high): keep `compression_drafter` open primitive in katgpt-core, but leave `quest_compression_draft` opt-in. Document the latency tradeoff in README.
- [ ] **T4.3** If G1 fails (no diversity gain over template selection): the corpus isn't producing useful signal. Try the multi-byte candidate variant (horizon=4 from nathan.rs/gzip-lm). If still fails → demote `compression_drafter` to opt-in in katgpt-core, document negative result.
- [ ] **T4.4** Commit with `feat:` prefix on success, `fix:` prefix if a follow-up fix was needed. Stay on `develop`. Tag the benchmark file in the commit message.

## Out of Scope (deferred to follow-up issues)

- **Per-NPC plasma-tier LZ77** (G1 latency fit at µs budget) — that's the Super-GOAT exploration in `riir-ai/.research/137`. Out of scope for this GOAT plan.
- **Fusion with PhraseBoost / IrreducibilityGate** — Phase 5+ once GOAT ships.
- **HLA moment corpus composition** — Phase 6+ once corpus-as-format is proven.
- **AbsorbCompress corpus eviction** — when corpus exceeds 30KB window, evict low-info bytes. Phase 7+.

---

## Why this is the right cut

1. **Honest verdict.** GOAT, not Super-GOAT. The user's pushback was correct: don't commit Super-GOAT before G1–G4 validation. The quest grammar angle is the clean win — template selection already works, this is a quality/scalability improvement, not a new feature.
2. **New wired format (user's insight).** The corpus IS the storage format. No QuestPack struct deserialization — the bytes go straight into the compressor window. BLAKE3 over the corpus bytes is the commitment. This is structurally identical to how we already commit weight snapshots, just on a different representation.
3. **Swap-in point exists.** `TernaryDraftModel::generate()` is the current template selector. `CompressionQuestDrafter::generate()` is the drop-in replacement. No invasive changes to `QuestGrammarPipeline`.
4. **Perf budget honest.** Hot tier (sub-ms), not plasma (µs). `lz4_flex` on 30KB corpus is ~tens of µs per call — fits the existing `hot_budget_ms` in `QuestGrammarConfig`. No speculative SIMD LZ77 kernel.
5. **3-repo discipline.** Open generic primitive in katgpt-core, private quest wiring in riir-games, no training anywhere.

---

## TL;DR

GOAT plan: ship a Hot-tier CompressionDrafter that replaces `TernaryDraftModel`'s template selection with corpus-as-scorer generation. The corpus IS the wired format — bytes + BLAKE3, no parser. G1 diversity (3× unique outputs vs 8 templates), G2 latency (≤2× ternary), G3 freeze/thaw roundtrip. If all pass → default-on in riir-games. If G2 fails → opt-in. If G1 fails → demote + document negative result. Per-NPC plasma angle stays in `riir-ai/.research/137` as exploration, not committed.
