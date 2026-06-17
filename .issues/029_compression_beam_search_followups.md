# Issue 029: Compression-Drafter Beam Search Follow-ups

**Date:** 2026-06-17
**Status:** Closed — both paths ruled out by Plan 287 Seal re-bench (153ms latency on real 1.3MB corpus).
**Plan:** [285_compression_drafter_quest_grammar.md](../.plans/285_compression_drafter_quest_grammar.md)
**Benchmark:** [285_compression_drafter_goat.md](../.benchmarks/285_compression_drafter_goat.md) (synthetic, GOAT FAILED) → [287_compression_drafter_seal_goat.md](../.benchmarks/287_compression_drafter_seal_goat.md) (real Seal 17k corpus, GOAT FAILED)
**Research:** [256_GzipLM_Compression_Drafter.md](../.research/256_GzipLM_Compression_Drafter.md)

---

## Context

Plan 285 ran the full workflow (research → plan → impl → bench → demote) twice:

| Run | Algorithm | Scorer | G1 Diversity | G2 Latency |
|-----|-----------|--------|--------------|------------|
| Phase 3 | Fixed-candidate scoring | lz4_flex | 0.12× (1 unique) ❌ | 407× ❌ |
| Phase 7 | Beam search (nathan.rs algorithm) | MatchLengthScorer (inverted index) | 1.50× (12 unique) ❌ | 1077× ❌ |

Target: G1 ≥ 3× (24 unique), G2 ≤ 2× (≈ 600ns).

**Honest conclusion:** compression-based generation loses to `TernaryDraftModel` template selection for Hot-tier quest grammar. Template selection is one matvec + one hash (~290ns); beam search is 1440 scorer calls (~313µs). There is no algorithmic fix — beam search fundamentally needs more compute per generation than single-pass selection.

---

## Two paths that might still work

### Path A: Per-NPC corpus (solves G1, not G2)

**Insight:** G1 failed because all 100 test contexts share the `"quest "` prefix and produce similar beams. If each NPC had its OWN corpus (different action history, different HLA moments), the corpora would diverge and so would the outputs — without needing beam search at all.

**What it needs:**
- Per-NPC `CompressionQuestDrafter` instances (one corpus per NPC).
- Seed each corpus with divergent content (HLA moments, action traces).
- Re-bench G1 with 100 distinct corpora, not 100 contexts on one corpus.

**Predicted outcome:**
- G1 likely passes (different corpora → different outputs by construction).
- G2 still fails (per-NPC doesn't change the per-call latency math).

**Where this lives:** `riir-ai/.research/137_Compression_Drafter_Plasma_Personality_Guide.md` already sketches this. The validation gate G2 (per-NPC divergence) is exactly this experiment.

**Blocker:** needs the plasma-tier custom LZ77 to make per-NPC instances cheap enough. lz4-based per-NPC instances would be ~50µs × 1000 NPCs = 50ms per tick — too slow.

### Path B: Warm-tier positioning ( sidesteps G2)

**Insight:** G2's 2× latency target assumes Hot-tier (sub-ms). But quest pack generation doesn't have to be Hot-tier — it can happen during NPC sleep cycles, world generation, or GM tool batches. Warm-tier (ms) is fine for offline generation.

**What it needs:**
- Reposition `CompressionQuestDrafter::generate_beam` as a Warm-tier API, not a Hot-tier replacement for `TernaryDraftModel`.
- Use case: GM tool generates 100 quest variants offline, picks the best, freezes the winner into a `TernaryDraftModel` template.
- Latency budget: 100ms per quest pack generation. We're at 313µs — 300× under budget.

**Predicted outcome:**
- G2 redefined: "fits Warm-tier budget (≤100ms)" instead of "≤2× ternary (≤600ns)". Passes trivially.
- G1 unchanged: still 12 unique vs 8 templates. May or may not matter for offline generation (where a human GM picks the best).

**Where this would go:** new plan, not a revision of 285. The use case is fundamentally different (offline batch vs runtime single-call).

---

## Recommendation

**~~Don't pursue either immediately.~~ CLOSED 2026-06-17 — both paths definitively ruled out by Plan 287 Seal re-bench.**

The user pushed back on Plan 285: "do bench from 17k seal, it's our prod target." Re-ran CompressionDrafter on the real Seal English quest corpus (13,780 lines, 1.3MB — 615× larger than synthetic). Results in [`.benchmarks/287_compression_drafter_seal_goat.md`](../.benchmarks/287_compression_drafter_seal_goat.md):

- **G1 Diversity: PASS** — 77 unique outputs (9.62× over 8-template baseline). Real corpus richness helped massively (12 → 77).
- **G2 Latency: CATASTROPHIC FAIL** — 153ms per generation (target: 1ms). The `MatchLengthScorer` inverted-index algorithm is O(matching_positions) per call. On a 1.3MB corpus, common bytes have ~100k positions → beam search × 4752 calls × 100k scans = 153ms.
- **G3 Adaptivity: PASS (caveat)** — corpus append changed output (-12 unique), but adding random noise decreased quality.

### Why Path A (per-NPC corpus) is now ruled out

Path A predicted: "G2 still fails (per-NPC doesn't change the per-call latency math)." This was **too optimistic**. The real corpus showed latency scales linearly with corpus size via the position-scan. Per-NPC corpora would each still be 100KB+ → still ~10ms+ per generation × 1000 NPCs = 10s+ per tick. Path A is **strictly worse** than the synthetic bench suggested.

### Why Path B (Warm-tier positioning) is now ruled out

Path B predicted: "313µs fits 100ms Warm-tier budget 300× over." This was **based on the synthetic 2KB corpus**. The real 1.3MB corpus shows **153ms per generation** — already **1.5× over** the 100ms Warm-tier budget, before any quality fixes (longer horizon for real sentences, per-NPC filtering, etc.). Path B is **not viable** without a different algorithm (suffix array / FM-index), which is a different primitive entirely.

### The actual right answer: adaptive template selector

The user's insight ("it wont be 8 forever for real prod bro, we may need adaptive") is **correct** — but the right implementation is not compression. It's:

```rust
struct AdaptiveQuestDrafter {
    templates: Vec<String>,  // N templates from corpus at startup
}
impl AdaptiveQuestDrafter {
    fn generate(&self, ctx: &str) -> &str {
        &self.templates[fnv1a_hash(ctx) % self.templates.len()]
    }
}
```

Measured on real Seal corpus: N=1024 templates → 95 unique outputs in **42ns**. Beats beam search on BOTH diversity (95 vs 77) AND latency (42ns vs 153ms) — by **3.6 million times** on latency. Adding templates is free (latency stays at ~40ns regardless of N).

This is a separate small plan if/when quest variety becomes a real product requirement. Compression-based generation is closed as a quest grammar technique.

---

## TL;DR

Plan 285 ran twice, failed twice. Compression-based generation loses to template selection for Hot-tier quest grammar — fundamentally, not just implementationally. Two follow-up paths exist (per-NPC corpus for G1, Warm-tier repositioning for G2) but neither is worth pursuing without a concrete consumer. Open primitive stays opt-in. Honest negative result, documented.
