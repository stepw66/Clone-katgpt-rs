# Plan 397: Hierarchical Global Attention (HGA) — Chunk→Group→Token Routing with RoPE-Aware Summaries

**Date:** 2026-07-05
**Research:** [katgpt-rs/.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md](../.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md)
**Source paper:** [arxiv 2606.30709](https://arxiv.org/abs/2606.30709) — Hierarchical Global Attention (Frank, Fedosov, Grinenko, BMW Group, Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/hga/` (new module) + `katgpt-rs/crates/katgpt-core/src/tiered_kv/` (new module) + Cargo feature `hga`
**Status:** Active — Phase 0 <state: plan written, no code yet>

---

## Goal

Ship three refinements of the shipped sparse-attention routing slot behind an opt-in `hga` feature flag, then run a head-to-head GOAT gate against the default-on DashAttention primitive on a synthetic NIAH harness at 32K–64K context. The MSA R225 GOAT-FAILED precedent looms — the gate is non-trivial. If HGA wins (G2), promote to default-on and demote the loser; if it loses, document as a negative result; if inconclusive, keep as an alternative routing-summary construction.

**Three deliverables (all modelless, all inference-time):**

1. **`GroupSummaryCache<C, GS, D>`** — sub-chunk group middle routing tier between DashAttention's chunk-level entmax routing and token-level attention. No shipped primitive has a sub-chunk middle tier.
2. **`MixedRopeSummarizer`** — per-frequency-pair RoPE-aware chunk/group summary construction (high-freq rotate-then-average, low-freq average-then-rotate-at-mid). Alternative to RTPurbo's pre-RoPE low-dim projection and DashAttention's learned summary query.
3. **`TieredKvStore` trait + `InMemoryTieredKvStore`** — Hot (always-resident summaries + sink + local) / Warm (LRU shard cache) / Cold (host-RAM-backed full token K/V) tiered store with `route_and_fetch(query, sink_local, budget) -> WorkingSet`. Reference implementation; production mmap-backed variant is a riir-ai/riir-neuron-db follow-up.

**Fusion wins (the part that justifies GOAT not Gain):**
- F1 (load-bearing): HGA group tier + mixed-RoPE on DashAttention backbone → fewer fetched tokens at iso-quality.
- F2: HGA mixed-RoPE vs RTPurbo low-dim projection head-to-head on retrieval heads.
- F3: PFlash `block_select` driving the new tiered route-and-fetch substrate.
- F4 (longer-horizon): HOLA warm + AM cold + HGA route → complete tiered long-context KV stack (depends on Plan 378 HOLA shipping first).

**Constraints checklist (per AGENTS.md):**
- [x] Modelless first — pure inference-time, no LLM training
- [x] Latent-to-latent preferred — chunk/group summaries are multi-resolution centroids in key latent space; route in latent, fetch real tokens at boundary
- [x] Freeze/thaw over fine-tuning — no weight mutation, route-and-fetch only
- [x] Self-learn/adaptive CoT — N/A for this primitive
- [x] 5-repo discipline — generic open primitive in katgpt-rs; private latent-reframe follow-ups noted (riir-neuron-db dendritic branch tiered retrieval)
- [x] SOLID, DRY — reuses DashAttention's `entmax_1p5`, `entmax_gqa_aggregate`; trait-based `TieredKvStore`
- [x] Tests/examples before/after — Phase 2 GOAT gate G1–G5
- [x] CPU/GPU/ANE auto-route — `InMemoryTieredKvStore` reference; GPU mmap-backed variant deferred to riir-ai
- [x] Plasma→Hot→Warm→Cold→Freeze tiering — `TieredKvStore` is the explicit instantiation of constraint 8 for sparse-attention K/V
- [x] Raw scalars at sync boundary — N/A (inference primitive, no sync)

---

## Phase 1 — Skeleton (CORE, unblocks Phase 2 GOAT gate)

### Tasks

- [ ] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/hga/` module skeleton under feature flag `hga` (gated in `lib.rs`):
  - `mod.rs` — module index + re-exports
  - `group_summary.rs` — `GroupSummaryCache<C, GS, D>` + `score_groups`
  - `summary.rs` — `MixedRopeSummarizer` + threshold derivation `θ_threshold = 2π / C`
  - `forward.rs` — HGA forward path (compose DashAttention entmax + group routing + route-and-fetch + exact softmax)
  - `tests.rs` — unit tests
- [ ] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/tiered_kv/` module skeleton (NOT feature-gated; this is a generic primitive):
  - `mod.rs` — `TieredKvStore` trait, `SinkLocalSet`, `RouteBudget`, `WorkingSet<D>`
  - `in_memory.rs` — `InMemoryTieredKvStore` reference impl (hot Vec + warm LRU + cold Vec)
  - `tests.rs` — unit tests
- [ ] **T1.3** Add `hga` feature to `katgpt-rs/crates/katgpt-core/Cargo.toml` (default `[]`, opt-in).
- [ ] **T1.4** Implement `MixedRopeSummarizer::summarize(keys, positions, rope_freqs) -> [f32; D]`:
  - For each RoPE frequency pair `(x_i, y_i)`:
    - Compute `θ_i` from `rope_freqs` (passed in, not hardcoded).
    - If `θ_i · C ≥ 2π` (high-frequency): rotate each key at its position, then mean over chunk.
    - Else (low-frequency): mean raw keys, then rotate at chunk-mid position.
  - **Critical:** threshold derived from `rope_freqs` and chunk size `C`, must work for Gemma 2 (`rope_theta = 10000`) AND Qwen3 (`rope_theta = 1000000`).
  - Unit test: `mixed_rope_summary_matches_mean_at_zero_rotation` (when all positions are 0, both branches degenerate to mean).
  - Unit test: `mixed_rope_summary_threshold_derivation` (verify the `θ_i · C ≈ 2π` crossover lands at the expected frequency-pair index for known `rope_theta` values).
- [ ] **T1.5** Implement `GroupSummaryCache::append_chunk(chunk_keys, positions, rope_freqs)`:
  - Compute `C/GS` group summaries per chunk using `MixedRopeSummarizer`.
  - Append to fixed-layout `[n_chunks, C/GS, n_kv_head, D]` buffer (use `Vec::with_capacity` once, write in-place).
- [ ] **T1.6** Implement `GroupSummaryCache::score_groups(query, selected_chunks) -> Vec<(chunk_idx, group_idx, score)>`:
  - Dot-product scoring of query against group summaries within selected chunks only (no full scan).
  - Pre-allocated scratch buffer; reuse across calls.
- [ ] **T1.7** Implement `InMemoryTieredKvStore`:
  - Hot: `Vec<(chunk_summary, sink_keys, sink_values, local_keys, local_values)>` — always resident.
  - Warm: `LruCache<ChunkIdx, (keys, values)>` — bounded LRU shard cache.
  - Cold: `Vec<(ChunkIdx, keys, values, group_summaries)>` — full token K/V in process RAM.
  - `append_chunk(keys, values, positions)` → store in cold tier; compute and store summary in hot tier.
  - `route_and_fetch(query, sink_local, budget) -> WorkingSet` → use `GroupSummaryCache::score_groups` + `DashAttention::entmax_1p5` for chunk selection, then group selection within chunks, then fetch real K/V for selected groups from cold tier (or warm tier cache hit).
- [ ] **T1.8** Implement `forward_hga(query, store, sink_local, route_budget, entmax_alpha, k_c, k_g) -> [f32; D]`:
  - Stage 1: chunk-level entmax routing (reuses `dash_attn::entmax_1p5` from Plan 106).
  - Stage 2: group-level top-K routing within selected chunks (uses `GroupSummaryCache::score_groups`).
  - Stage 3: `route_and_fetch` from tiered store + exact softmax over fetched token K/V (standard SDPA, never summary K/V).
- [ ] **T1.9** Wire `forward_hga` into the existing sparse-attention dispatch path (mirror DashAttention's wiring). Add `AttnMode::Hga` config variant.
- [ ] **T1.10** `cargo check -p katgpt-core --features hga` compiles clean.
- [ ] **T1.11** `cargo check -p katgpt-core --all-features` compiles clean (combo-regression — the `merkle_root` / `can_freeze` lesson class applied).
- [ ] **T1.12** Unit test: `forward_hga_full_coverage_equals_causal_sdpa` — with `route_budget = infinity`, `k_c = all chunks`, `k_g = all groups`, output bit-identical to causal SDPA within f32 noise (`< 1e-5`).

---

## Phase 2 — GOAT gate (the promote/demote decision)

### Tasks

- [ ] **T2.1** Build a synthetic NIAH harness mirroring DashAttention's GOAT 9/9 harness:
  - Synthetic context: 32K tokens, single needle at controlled depth (25%, 50%, 75%).
  - RoPE: configurable `rope_theta ∈ {10000, 50000, 1000000}`.
  - Model: micro-GPT (the existing 40M SmallLM harness used by DashAttention/AM/MSA GOAT gates).
  - Compare: dense SDPA vs DashAttention (default-on baseline) vs HGA at matched fetched-token budget.
- [ ] **T2.2 (G1)** Correctness — full-coverage HGA = causal SDPA within `< 1e-5` abs diff. Mirrors paper Table 7 (`< 10⁻⁶` abs diff).
- [ ] **T2.3 (G2 — LOAD-BEARING)** Per-quality-vs-sparsity head-to-head:
  - At `rope_theta = 1000000` (Qwen3-style) and 32K context:
    - Sparsity sweep: HGA at {3.13%, 6.25%, 12.5%, 25%} vs DashAttention at same sparsities.
    - Metric: per-token loss gap vs dense (`ΔLoss = Loss_sparse − Loss_dense`).
    - **Pass criterion:** HGA `ΔLoss` ≤ DashAttention `ΔLoss` at all four sparsity levels, OR HGA matches DashAttention `ΔLoss` at half the fetched tokens (≥2× sparsity compounding from the group tier).
  - At `rope_theta = 10000` (Gemma 2-style): same sweep (the mixed-RoPE threshold risk).
- [ ] **T2.4 (G3)** No-regression — `cargo check --all-features` zero warnings; existing DashAttention tests still pass.
- [ ] **T2.5 (G4)** Alloc-free hot path — `forward_hga` route + fetch + attend in pre-allocated scratch buffers (verify via `cargo bench` with allocator hooks or heap profiling on a 1K-iteration loop).
- [ ] **T2.6 (G5)** Latency — group-routing pass latency ≤ 1.5× DashAttention chunk-routing pass latency at the same context length (criterion bench on 32K context, single query).
- [ ] **T2.7 (G2 risk — MSA precedent)** If G2 fails (HGA does NOT beat DashAttention at iso-quality on our harness), document as a negative result in `katgpt-rs/.benchmarks/397_hga_goat.md` mirroring the MSA R225 GOAT-FAILED format. Keep `hga` opt-in. Do NOT promote.
- [ ] **T2.8** Write benchmark report to `katgpt-rs/.benchmarks/397_hga_goat.md`.

---

## Phase 3 — Promote/demote decision

### Tasks

- [ ] **T3.1** If G2 PASS with clear margin:
  - Promote `hga` to default-on in `katgpt-rs/crates/katgpt-core/Cargo.toml`.
  - Run head-to-head vs DashAttention on the same NIAH harness at 32K and 64K.
  - Demote the loser to opt-in (per the §1.6 promote/demote ledger rule). Update `katgpt-rs/README.md` to reflect the winner.
- [ ] **T3.2** If G2 PASS within noise (inconclusive):
  - Keep `hga` opt-in as an alternative routing-summary construction (mixed-RoPE vs DashAttention learned summary).
  - Document the trade-off in the README sparse-attention slot table.
- [ ] **T3.3** If G2 FAIL:
  - Keep `hga` opt-in. Document as negative result.
  - Investigate which sub-component failed (group tier overhead, mixed-RoPE threshold mismatch, route-and-fetch cost). May split into separate opt-in features (e.g., `mixed_rope_summary` survives without the group tier).

---

## Phase 4 — Documentation (after Phase 3 verdict)

### Tasks

- [ ] **T4.1** Update `katgpt-rs/README.md` sparse-attention slot table with HGA row (default-on or opt-in per Phase 3).
- [ ] **T4.2** Update `katgpt-rs/.docs/02_architecture.md` with HGA module entry.
- [ ] **T4.3** Update `katgpt-rs/.docs/01_overview.md` sparse-attention section.
- [ ] **T4.4** Add cross-reference from DashAttention (R071) note to HGA (R379) note and vice versa.
- [ ] **T4.5** Commit with message `feat(hga): hierarchical global attention chunk-group-token routing with RoPE-aware summaries`.

---

## Phase 5 — Private follow-ups (NOT in this plan; tracked for riir-* repos)

These are noted in the research note §2.5 as latent reframings, but tracked separately:

- [ ] **P5.1 (riir-neuron-db)** Apply `TieredKvStore` primitive to NeuronShard dendritic branch retrieval — shard = cold tier, branch = warm tier, weight = hot tier. The closest structural analog (see Research 379 §2.5(e)). Tracked as a separate riir-neuron-db plan if scoped.
- [ ] **P5.2 (riir-ai)** Production mmap-backed `TieredKvStore` variant for GPU inference — mirrors `ZoneGeometryCache` (Plan 335) Arc<Mmap> + lock-free papaya + LRU pattern. Tracked as a separate riir-ai plan if scoped.
- [ ] **P5.3 (orthogonal)** Position-modulo RoPE wrapping `p ← p mod 65536` (paper §5.5 side finding) — separate issue/plan if long-context Gemma 2 inference shows unexplained quality degradation past training length. NOT part of HGA primitive.

---

## Open risks (recap from Research 379 §5)

1. **MSA precedent (R225 GOAT FAILED).** Blockwise sparse with per-GQA-group + max-pool failed on our harness. HGA shares the class. G2 is non-trivial.
2. **Mixed-RoPE threshold is paper-vague.** `θ_threshold = 2π/C` is our derivation; may not match paper's intended crossover. Mitigation: G2 sweeps `rope_theta ∈ {10000, 50000, 1000000}`.
3. **Group tier adds a scoring pass.** Two-level routing is strictly more compute than DashAttention's one-level. Mitigation: G5 latency gate + feature-gate to long-context-only.
4. **Tiered K/V store is systems-level.** In-memory reference provides no VRAM benefit. Production mmap variant deferred to riir-ai.
5. **Contested slot.** Adding HGA without a clean winner risks feature-flag proliferation. Mitigation: G2 head-to-head must produce a single winner; loser demotes.

---

## TL;DR

Plan 397 ships HGA (chunk→group→token routing with mixed-RoPE summaries and tiered route-and-fetch K/V store) behind opt-in `hga` feature flag. **Phase 1** = skeleton (5 new files, 12 tasks). **Phase 2** = GOAT gate G1–G5 on synthetic NIAH harness at 32K–64K with `rope_theta ∈ {10000, 50000, 1000000}`; **G2 is load-bearing** — HGA must beat DashAttention at iso-quality (MSA R225 GOAT-FAILED precedent). **Phase 3** = promote-to-default + demote-loser if G2 passes; opt-in + negative-result doc if G2 fails. **Phase 4** = docs. **Phase 5** = private follow-ups noted (riir-neuron-db dendritic branch tiered retrieval, riir-ai mmap-backed store, orthogonal position-modulo RoPE wrapping). All modelless, all inference-time. No quality-parity claim with paper — full-model parity is a riir-train job for the fine-tuning variant, explicitly out of scope.
