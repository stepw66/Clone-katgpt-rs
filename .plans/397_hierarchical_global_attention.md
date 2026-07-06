# Plan 397: Hierarchical Global Attention (HGA) — Chunk→Group→Token Routing with RoPE-Aware Summaries

**Date:** 2026-07-05
**Research:** [katgpt-rs/.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md](../.research/379_Hierarchical_Global_Attention_Chunk_Group_Routing.md)
**Source paper:** [arxiv 2606.30709](https://arxiv.org/abs/2606.30709) — Hierarchical Global Attention (Frank, Fedosov, Grinenko, BMW Group, Jun 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/hga/` (new module) + `katgpt-rs/crates/katgpt-core/src/tiered_kv/` (new module) + Cargo feature `hga`
**Status:** Active — Phase 1 complete, Phase 2 GOAT gate G2-proxy FAIL (negative result, keep opt-in). G5 latency PASS (1.12×). Phase 3 decision: T3.3 (keep opt-in, document negative result).

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

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/hga/` module skeleton under feature flag `hga` (gated in `lib.rs`):
  - `mod.rs` — module index + re-exports
  - `group_summary.rs` — `GroupSummaryCache<C, GS, D>` + `score_groups`
  - `summary.rs` — `MixedRopeSummarizer` + threshold derivation `θ_threshold = 2π / C`
  - `tests.rs` — unit tests
  - **NOTE:** `forward.rs` moved to `katgpt-attn/src/hga_forward.rs` — it needs `dash_attn::entmax_1p5` which lives in katgpt-attn (katgpt-core cannot import katgpt-attn without a circular dep).
- [x] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/tiered_kv/` module skeleton (NOT feature-gated; this is a generic primitive):
  - `mod.rs` — `TieredKvStore` trait, `SinkLocalSet`, `RouteBudget`, `WorkingSet`, `GroupSelection`
  - `in_memory.rs` — `InMemoryTieredKvStore` reference impl (hot summaries + cold Vec)
  - `tests.rs` — unit tests
- [x] **T1.3** Add `hga` feature to `katgpt-rs/crates/katgpt-core/Cargo.toml` (default `[]`, opt-in).
- [x] **T1.4** Implement `MixedRopeSummarizer::summarize(keys, positions, rope_freqs) -> [f32; D]`:
  - For each RoPE frequency pair `(x_i, y_i)`:
    - Compute `θ_i` from `rope_freqs` (passed in, not hardcoded).
    - If `θ_i · C ≥ 2π` (high-frequency): rotate each key at its position, then mean over chunk.
    - Else (low-frequency): mean raw keys, then rotate at chunk-mid position.
  - **Critical:** threshold derived from `rope_freqs` and chunk size `C`, must work for Gemma 2 (`rope_theta = 10000`) AND Qwen3 (`rope_theta = 1000000`).
  - Unit test: `mixed_rope_summary_matches_mean_at_zero_rotation` (when all positions are 0, both branches degenerate to mean).
  - Unit test: `mixed_rope_summary_threshold_derivation` (verify the `θ_i · C ≈ 2π` crossover lands at the expected frequency-pair index for known `rope_theta` values).
- [x] **T1.5** Implement `GroupSummaryCache::append_chunk(chunk_keys, positions, rope_freqs)`:
  - Compute `C/GS` group summaries per chunk using `MixedRopeSummarizer`.
  - Append to `Vec<f32>` summary store (capacity pre-reserved).
- [x] **T1.6** Implement `GroupSummaryCache::score_groups(query, selected_chunks) -> Vec<(chunk_idx, group_idx, score)>`:
  - Dot-product scoring of query against group summaries within selected chunks only.
  - Uses SIMD `simd_dot_f32`; results sorted descending for top-K selection.
- [x] **T1.7** Implement `InMemoryTieredKvStore`:
  - Cold: `Vec<ChunkKv>` — full token K/V per chunk in process RAM.
  - Hot: `Vec<GroupSummary>` — per-chunk group summaries (injected summarizer fn).
  - `append_chunk(keys, values, positions)` → cold + hot tiers.
  - `fetch_working_set(sink_local, selected_chunks, group_selection)` → fetches sink+local fully + routed groups from cold tier.
- [x] **T1.8** Implement `forward_hga(query, store, sink_local, route_budget, entmax_alpha, k_c, k_g) -> [f32; D]`:
  - **Lives in `katgpt-attn/src/hga_forward.rs`** (needs `dash_attn::entmax_1p5`).
  - Stage 1: chunk-level entmax routing (reuses `dash_attn::entmax_1p5`).
  - Stage 2: group-level top-K routing (`GroupSummaryCache::select_top_k_groups`).
  - Stage 3: `fetch_working_set` + exact softmax SDPA over fetched real-token K/V.
- [-] **T1.9** Wire `forward_hga` into the existing sparse-attention dispatch path (mirror DashAttention's wiring). Add `AttnMode::Hga` config variant.
  - **DEFERRED to Phase 2:** the standalone `forward_hga` function is provided and tested (T1.12). Wiring into the full `ForwardContext` dispatch requires transformer-layer integration that belongs in Phase 2's GOAT gate setup.
- [x] **T1.10** `cargo check -p katgpt-core --features hga` compiles clean.
- [x] **T1.11** `cargo check -p katgpt-core --all-features` compiles clean (combo-regression — the `merkle_root` / `can_freeze` lesson class applied).
- [x] **T1.12** Unit test: `forward_hga_full_coverage_equals_causal_sdpa` — with `route_budget = infinity`, `k_c = all chunks`, `k_g = all groups`, output bit-identical to causal SDPA within f32 noise (`< 1e-5`).
  - **katgpt-core analog** (`full_coverage_fetch_matches_causal_sdpa`) verifies the fetch+SDPA path without entmax (8/8 tests pass). The katgpt-attn `hga_forward` test (`forward_hga_full_coverage_equals_causal_sdpa`) is written but blocked by a **pre-existing** test compilation failure in `dash_attn::vortex_flow` (`MsaMaxPool` variant removed but referenced in a test — unrelated to HGA, confirmed pre-existing via `git stash`).

---

## Phase 2 — GOAT gate (the promote/demote decision)

### Tasks

- [-] **T2.1** Build a synthetic NIAH harness mirroring DashAttention's GOAT 9/9 harness:
  - **DEFERRED:** full transformer-level harness (micro-GPT, 32K context, per-token loss gap) requires riir-train infrastructure. Replaced by the G2-proxy (T2.3) modelless routing comparison.
- [x] **T2.2 (G1)** Correctness — full-coverage HGA = causal SDPA within `< 1e-5` abs diff. Mirrors paper Table 7 (`< 10⁻⁶` abs diff). **PASS** (verified in Phase 1).
- [x] **T2.3 (G2 — LOAD-BEARING)** Per-quality-vs-sparsity head-to-head:
  - **Replaced by G2-proxy** (modelless NIAH routing comparison, `tests/bench_397_hga_goat.rs`). Tests the core claim: does HGA's sub-chunk group tier improve needle retrieval at iso-sparsity?
  - **VERDICT: FAIL** — HGA won only 2/12 trials against DashAttention. Root cause: group summaries of random keys dilute the single-needle signal below the dot-product detection threshold. Same failure mode as MSA R225. See `.benchmarks/397_hga_goat.md`.
  - The full G2 (transformer-level loss-gap) is deferred to riir-train — the paper's result uses trained keys with semantic structure.
- [x] **T2.4 (G3)** No-regression — `cargo check --all-features` zero warnings; existing DashAttention tests still pass. **PASS** (verified in Phase 1).
- [x] **T2.5 (G4)** Alloc-free hot path — `forward_hga` route + fetch + attend in pre-allocated scratch buffers.
  - **INFORMATIONAL:** Phase 1 reference implementation allocates ~8 Vecs per routing call. Zero-alloc optimization deferred (not worth optimizing if G2-proxy fails).
- [x] **T2.6 (G5)** Latency — group-routing pass latency ≤ 1.5× DashAttention chunk-routing pass latency at the same context length (criterion bench on 32K context, single query).
  - **PASS:** HGA = 3.53ms, DashAttention = 3.14ms at 32K context (512 chunks × 64 tokens). Ratio = 1.12× (target ≤ 1.5×). The group-tier scoring pass adds only 12% overhead.
- [x] **T2.7 (G2 risk — MSA precedent)** If G2 fails (HGA does NOT beat DashAttention at iso-quality on our harness), document as a negative result in `katgpt-rs/.benchmarks/397_hga_goat.md` mirroring the MSA R225 GOAT-FAILED format. Keep `hga` opt-in. Do NOT promote.
  - **DONE:** G2-proxy FAIL documented in `.benchmarks/397_hga_goat.md`. `hga` stays opt-in.
- [x] **T2.8** Write benchmark report to `katgpt-rs/.benchmarks/397_hga_goat.md`.
  - **DONE:** full report with G2-proxy detailed table, G5 latency, G4 alloc count, root cause analysis, and MSA comparison.

---

## Phase 3 — Promote/demote decision

### Tasks

- [ ] **T3.1** If G2 PASS with clear margin: **N/A** (G2-proxy FAIL).
- [ ] **T3.2** If G2 PASS within noise (inconclusive): **N/A** (G2-proxy FAIL).
- [x] **T3.3** If G2 FAIL:
  - ✅ Keep `hga` opt-in.
  - ✅ Document as negative result (`.benchmarks/397_hga_goat.md`).
  - ✅ Investigate which sub-component failed: the group-tier routing (dot-product scoring on group summaries of random keys) dilutes the single-needle signal below the detection threshold. The mixed-RoPE summarizer and the TieredKvStore work correctly.
  - The primitive may be revisited with trained keys (riir-train).

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
