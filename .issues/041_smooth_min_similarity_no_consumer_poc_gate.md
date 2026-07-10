# Issue 041: Smooth-Min Soft Similarity — No Consumer Ready, PoC-Gated

> **Spawned from:** Research 385 (SoftMatcha 2 smooth-min soft pattern matching — Gain)
> **Confidence:** LOW — utility is trivial to write, but every candidate consumer either lacks a per-token embedding path (ItemEmbedIndex), is a stub (AnyRAG gateway), or doesn't exist (soft Engram fallback). Shipping the primitive now = YAGNI.
> **Date:** 2026-07-07
> **Status:** OPEN

---

## TL;DR

Research 385 distilled the **smooth-min similarity** + Zipfian-norm edit penalty from SoftMatcha 2 (ICML 2026) as a modelless latent-space utility for fuzzy multi-token retrieval. The function is ~20 lines. But **no consumer is ready to call it**:

| Consumer | Why blocked |
|---|---|
| `ItemEmbedIndex` (`riir-neuron-db/src/item_index.rs`) | Uses a **single 8-dim vector per item** (schema-centroid, Plan 362). Smooth-min aggregates *per-position* cosines — it needs per-token embeddings, which ItemEmbedIndex doesn't have. Adding a per-token path is a bigger architectural change than the smooth-min function itself. |
| `AnyRAG gateway` (`riir-neuron-db/src/gateway.rs`) | `request_ruling` is a **stub** — returns empty `ExternalRuling`, no real retrieval backend wired. Smooth-min has nothing to score against. |
| `Engram soft fallback` (`katgpt-rs/crates/katgpt-core/src/engram/`) | Doesn't exist. Engram is exact-hash only (Plan 299). A "soft Engram" layer that falls back to cosine on hash-miss is itself an unimplemented feature. |

**Recommendation: do not impl the primitive now.** Re-open when any consumer lands its prerequisite. Track the PoC gate below.

---

## The Primitive (for reference — do not write until a consumer exists)

```rust
/// katgpt-rs/crates/katgpt-core/src/similarity.rs (hypothetical)
///
/// Smooth-minimum similarity for variable-length soft pattern matching.
/// `cosines` = per-position cosine similarities (each in [-1, 1]).
/// `beta` = sharpness (paper β=1e4; β→∞ = plain min, β≈1 = plain sum).
/// Returns similarity in [0, 1].
fn smooth_min_similarity(cosines: &[f32], beta: f32) -> f32 {
    let log_beta = beta.ln();
    let sum = cosines.iter()
        .map(|&c| ((1.0 - c) * log_beta).exp() - 1.0)
        .sum::<f32>() + 1.0;
    1.0 - sum.ln() / log_beta
}

/// Zipfian-norm insertion/deletion penalty.
/// `norm_sq` = squared norm of the edited token's embedding (post-Zipfian whitening).
/// `gamma` = penalty scale (paper: γ = m·γ').
fn edit_penalty(norm_sq: f32, gamma: f32) -> f32 {
    (-norm_sq / gamma).exp()
}
```

Feature flag: `smooth_min_similarity` (opt-in). Would live in `katgpt-core` as a generic utility — no game/chain/shard semantics.

---

## The PoC Gate (mandatory before any impl)

Per AGENTS.md §3.6 (defend-wrong PoC) and the GOAT gate rule, before promoting smooth-min from "trivial utility" to "shipped primitive with a feature flag", a PoC must show:

1. **Quality gain:** smooth-min recall@5 > plain-cosine recall@5 on multi-token retrieval queries with ≥2 token mismatches (e.g., "enchanted silver sword" vs catalog entries).
2. **Latency budget:** < 100 ns overhead per query on top of the existing cosine path (smooth-min is O(m) for m ≤ 10 query tokens; should be sub-µs).
3. **β sensitivity:** show the β=10⁴ operating point from the paper holds on OUR retrieval domain (game item names, quest text, dialog cues), not just FineWeb-Edu.

**Where the PoC would live:** `riir-ai/crates/riir-poc/benches/smooth_min_vs_cosine.goat.rs` — the defend-wrong R&D crate. Three competitors: plain cosine (baseline), smooth-min (candidate), frozen/no-retrieval (floor).

**PoC blocker:** the PoC needs a multi-token retrieval task with per-token embeddings. ItemEmbedIndex doesn't have per-token embeddings yet. So the PoC itself is blocked on the consumer prerequisite.

---

## Consumer Prerequisites (the real blockers)

### Path A — ItemEmbedIndex per-token path (riir-neuron-db)

ItemEmbedIndex (Plan 362, default-on) currently stores one 8-dim schema-centroid embedding per item. To use smooth-min, it needs a **per-token embedding path**: decompose "enchanted silver sword" into 3 token embeddings, compute per-position cosine against catalog item token-embeddings, aggregate via smooth-min.

**Effort:** medium. Requires defining a token-level embedding for item names (could reuse schema-centroid per token type, or add a small embedding table). The smooth-min call is then a 3-line addition to `ItemEmbedIndex::query`.

**Open question:** does Seal's item naming have enough multi-token structure to benefit? Many items are single-token ("Sword", "Potion"). The gain may be marginal for short item names.

### Path B — AnyRAG real retrieval backend (riir-neuron-db)

`gateway.rs::request_ruling` is a stub. When AnyRAG gets a real backend (HTTP to an external retrieval service, or an in-process corpus index), smooth-min would score retrieved patterns against the conflict context.

**Effort:** large. The real backend itself is the work; smooth-min is a small scoring function on top.

**Timeline:** AnyRAG backend is not currently planned in any active `.plans/` file (grep confirms). This path is blocked indefinitely.

### Path C — Soft Engram fallback (katgpt-rs)

Engram (Plan 299) is exact-hash only. A "soft Engram" would add a cosine-fallback tier when the exact hash misses, scored by smooth-min over the Engram table's stored patterns.

**Effort:** medium. Requires Engram to expose its stored patterns for cosine scan (currently it exposes hash → hidden-state-slot only). The `ZipfianCacheHierarchy` (Plan 299 Phase 6) could host the fallback tier.

**Risk:** changes Engram's character from O(1) hash lookup to O(N) cosine scan on miss. Could violate the 48 ns/retrieval G1 gate if the fallback fires often.

---

## Decision Matrix

| Path | Consumer ready? | PoC possible now? | Impl smooth-min now? |
|---|---|---|---|
| A — ItemEmbedIndex | No (no per-token path) | No | **No** |
| B — AnyRAG | No (stub) | No | **No** |
| C — Soft Engram | No (no fallback tier) | No | **No** |

**All three paths: do not impl.** Re-open this issue when any consumer lands its prerequisite.

---

## Tasks (tracking only — no impl)

- [-] **T1** (deferred) When ItemEmbedIndex grows a per-token embedding path (Path A), re-open this issue and write the PoC at `riir-ai/crates/riir-poc/benches/smooth_min_vs_cosine.goat.rs`.
- [-] **T2** (deferred) When AnyRAG gets a real retrieval backend (Path B), re-open and wire smooth-min as the scoring function.
- [-] **T3** (deferred) When Engram adds a soft-fallback tier (Path C), re-open and evaluate whether smooth-min or plain cosine is the right fallback scorer (PoC required — smooth-min changes Engram's latency character).
- [-] **T4** (won't-do unless a consumer lands) Write `smooth_min_similarity` + `edit_penalty` in `katgpt-core/src/similarity.rs` behind feature flag `smooth_min_similarity`. Blocked on T1/T2/T3.

---

## Cross-references

- **Research 385** (`katgpt-rs/.research/385_SoftMatcha2_Smooth_Min_Soft_Pattern_Match.md`) — the Gain-tier verdict.
- **Research 012** (`riir-neuron-db/.research/012_egg_shell_pruner_funcattn_item_retrieval_fusion.md`) — ItemEmbedIndex Super-GOAT strategy guide.
- **Plan 362** (`riir-neuron-db/.plans/362_*`) — ItemEmbedIndex implementation (default-on).
- **Plan 299** (`katgpt-rs/.plans/299_Engram_Hash_Addressed_Pattern_Memory.md`) — Engram, ZipfianCacheHierarchy.
- **.benchmarks/287** — CompressionDrafter GOAT FAILED; confirms suffix-array is not our path.

---

## TL;DR

**Don't impl.** The smooth-min function is trivial (~20 lines) but every candidate consumer is blocked: ItemEmbedIndex lacks per-token embeddings, AnyRAG is a stub, soft Engram doesn't exist. Shipping the primitive now lands zero callers — YAGNI. Re-open when a consumer prerequisite lands; the PoC gate (smooth-min vs plain cosine on multi-token retrieval) is mandatory before any feature flag.
