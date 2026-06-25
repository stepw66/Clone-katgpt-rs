# Plan 328: Tucker/HOSVD Consumer Applications — Chain Collusion & Game Economy Anomaly Detection

**Date:** 2026-06-25
**Primitive:** [Plan 326](326_tucker_hosvd_factorization.md) — `katgpt-core/linalg::tucker` (DEFAULT-ON, G1–G4 PASS)
**Research:** `.research/307_FNO_Practical_Perspective_Spectral_Primitives_Survey.md` (§3 candidate #3)
**Status:** Active — consumer applications phase
**Motivation:** Plan 326 shipped a generic N-mode HOSVD primitive with zero consumers. Plan 328 finds the obvious-shine consumers — the ones where 3-mode tensor factorization is *self-evidently* the right tool, not a stretch.

---

## Goal

Give Plan 326's Tucker primitive **two obvious, high-value consumers** — one in `riir-chain`, one in `seal-online-remaster` — where the 3-mode tensor factorization model is the textbook solution and the existing rule-based heuristics are the obvious upgrade target.

Both consumers share the same pattern: **factor-model anomaly detection**. Build a 3-mode tensor of observed behavior, factorize it via HOSVD, and flag entities whose residual against the low-rank reconstruction is anomalous (too high = outlier; too low = too-compressible = scripted/colluding). This is exactly how real-world market surveillance and behavioral fraud detection work — it is not a contrived fit.

### Why these two are "obvious shine" (not a stretch)

| Consumer | Tensor shape | Why Tucker is self-evidently right |
|----------|-------------|------------------------------------|
| **Chain: curator collusion** | `V[curator, round, tier]` | Consensus voting is *inherently* a 3-mode tensor. Colluding validators are the textbook "low-rank voting bloc" signal. The existing `detect_collusion` does brittle exact-match grouping; Tucker catches *soft* blocs (coordinate-mostly-but-not-always) that evade exact match. |
| **Game: RMT economy anomaly** | `P[item, window, zone]` | Market price surveillance is *the* canonical PCA/HOSVD application (portfolio factor models). Cross-zone factor = arbitrage/RMT-transfer signal. The MMO already has `rmt_alert_count` + `flagged` shops; Tucker is the factor-model upgrade to the threshold rules. |

Both are **modelless** (closed-form HOSVD per batch), **cold-path** (analytics/monitoring, not the hot NPC-tick loop — so Tucker's 40–220µs compaction latency is irrelevant), and the **N ≤ 16 cap is a non-issue** (chunk by tier / item-category / cohort, each ≤ 16).

---

## Consumer 1 — Chain: Curator Collusion Detection (riir-chain)

### Current state (the upgrade target)

`riir-chain/src/consensus/curator_slashing.rs:143` — `detect_collusion`:

```rust
pub fn detect_collusion(&self, votes: &[(u64, [[u8; 32]; 5])]) -> Vec<Vec<u64>>
```

**Current heuristic:** single-pass O(N) HashMap grouping by *exact* 160-byte tier-root match. Two curators are "colluding" only if all 5 tier roots are byte-identical.

**Weakness:** sophisticated collusion (a voting bloc that coordinates but occasionally diverges to evade exact-match detection) sails through. Real attackers don't vote byte-identically forever — they vote together *most of the time*, with strategic noise. Exact match catches lazy cartels; it misses adversarial ones.

### The Tucker upgrade

Build a voting tensor across a rolling window of `R` consensus rounds:

```
V[curator, round, tier] ∈ {0,1}  (or normalized agreement score)
```

- **Mode 0 (curator):** one row per curator
- **Mode 1 (round):** the last R rounds (R ≤ 16, fits the SVD cap)
- **Mode 2 (tier):** the 5 data tiers

`V[c, r, t] = 1` if curator `c` voted the majority root on tier `t` in round `r`, else `0` (or a continuous agreement score: fraction of co-voters sharing that root).

**HOSVD factorization yields:**
- **Factor `curator` (mode-0 loadings):** curators who vote together cluster on a shared low-rank direction. A genuine voting **bloc** = a tight cluster in the curator-factor subspace, *regardless of whether they ever voted byte-identically*.
- **Factor `round` (mode-1):** temporal regime (which rounds had coordinated activity).
- **Factor `tier` (mode-2):** which tiers the bloc coordinates on.

**The detection signal:** cluster the curator-factor rows (e.g., via cosine similarity on the retained mode-0 singular vectors). Clusters of size ≥ `collusion_threshold` = blocs. This catches *soft* collusion (correlated voting, not identical voting) that exact-match misses. The existing `detect_collusion` becomes the fast pre-filter (cheap, catches the lazy cases); Tucker runs as the slow analytics pass on the rolling window to catch the adversarial cases.

### Tasks

- [ ] **T1.1** Add `VotingTensor` builder in `riir-chain/src/consensus/` — rolls up `Vec<CuratorVote>` over a rolling window of `R` rounds into a flat `&[f32]` of shape `(N_curators, R, 5)`. R capped at 16 (SVD_MAX_RANK).
- [ ] **T1.2** Add `detect_collusion_tucker(votes, window, ranks) -> Vec<CollusionBloc>` that:
  - Builds the tensor via T1.1
  - Calls `katgpt_core::linalg::tucker_decompose_into` with ranks `(r_c, R, 5)` where `r_c` is small (2–4 — a real bloc is low-rank)
  - Clusters curator-factor rows by cosine similarity ≥ threshold → blocs
  - Returns `CollusionBloc { members: Vec<u64>, cohesion: f32, rounds_active: Vec<u64> }`
- [ ] **T1.3** Wire `detect_collusion_tucker` alongside `detect_collusion` — exact-match stays the fast path; Tucker runs on the analytics cadence (e.g., every `R` rounds) and feeds `curator_slashing`.
- [ ] **T1.4** Tests: synthetic bloc injection (K curators vote together with ε noise across R rounds) → Tucker detects the bloc; exact-match misses it when ε > 0.

### GOAT gate (Chain)

- [ ] **G1 (soft-collusion detection):** Inject a 5-curator bloc that agrees 85% of the time (15% strategic divergence) across 16 rounds. Tucker `detect_collusion_tucker` flags the bloc; the existing exact-match `detect_collusion` does NOT (they never vote byte-identically). **PASS** = Tucker recall > exact-match recall at ε = 0.15.
- [ ] **G2 (no false positives on honest voting):** Inject N=16 honest curators voting independently. Tucker flags 0 blocs (no cluster exceeds the cohesion threshold). **PASS** = 0 false-positive blocs.
- [ ] **G3 (modelless):** Pure closed-form HOSVD + cosine clustering, no training. **PASS** = no `riir-train` dependency.
- [ ] **G4 (perf):** `detect_collusion_tucker` on (16 curators, 16 rounds, 5 tiers) ≤ 1ms (cold analytics path, not hot consensus). **PASS** = mean ≤ 1ms.

---

## Consumer 2 — Game: RMT Economy Anomaly Detection (seal-online-remaster)

### Current state (the upgrade target)

The MMO already has rule-based economy anomaly detection:
- `seal-gm-tools/src/state.rs` — `EconomyDashboard { rmt_alert_count, recent_flows, ... }`, `ShopEntry { price, volume_24h, flagged }`, `ItemStat { anomaly_flag }`, `Guild { anomaly_flag }`
- `seal-gm-tools/src/tabs/shops.rs` — `anomaly_section()` UI with suspend/flag/investigate actions on `flagged != 0` shops
- `seal-gm-tools/src/rerun_stream.rs` — `log_economy_graph(flows)` already visualizes gold flows

**Current heuristic:** `flagged` is a hand-tuned threshold rule (e.g., price > N× median). Brittle: misses novel RMT patterns, generates false positives on legitimate event-driven price spikes.

### The Tucker upgrade

Build a price/volume tensor over a rolling window:

```
P[item, window, zone] = log-median price of item i in time-window w at zone/server z
V[item, window, zone] = trade volume
```

- **Mode 0 (item):** ≤ 16 items per factor group (chunk by item category — weapons, armor, mats, etc., each ≤ 16)
- **Mode 1 (window):** last W time windows (W ≤ 16)
- **Mode 2 (zone):** ≤ 16 zones/servers

**HOSVD factorization yields:**
- **Factor `item`:** which items co-move (the "market sectors")
- **Factor `window`:** temporal regimes (patch drops, events, drift)
- **Factor `zone`:** cross-server price structure (servers that track each other vs divergent)

**The detection signal:** for each `(item, window, zone)` entry, compute the residual `|observed − reconstructed|`. High residual = anomalous — the price/volume doesn't fit the factor model, suggesting RMT transfer, exploit, or manipulation. This is literally how exchange market surveillance (FINRA, Nasdaq SMARTS) flags unusual trades.

The `flagged` field on `ShopEntry`/`ItemStat` gets populated from the Tucker residual score instead of the threshold rule. The factor matrices feed new GM-dashboard views ("market regime", "cross-server arbitrage map").

### Tasks

- [ ] **T2.1** `EconomyTensor` builder in `seal-gm-tools/src/analytics/` (or `seal-container-service`) — rolls up `ShopEntry`/`GoldFlow`/`ItemStat` history into `P[item, window, zone]`. **T0 prerequisite:** confirm the persistence layer retains enough per-item-per-zone-per-window history (if not, T0 adds a retention table).
- [ ] **T2.2** `detect_rmt_tucker(tensor, ranks) -> Vec<RmtAnomaly>` that:
  - Calls `katgpt_core::linalg::tucker_decompose_into` with ranks `(r_i, r_w, r_z)` (low-rank — the normal market is low-dimensional)
  - Reconstructs via `tucker_reconstruct_into`
  - Residual-scores each entry → top-K residuals = anomalies
  - Returns `RmtAnomaly { item, window, zone, residual, observed, expected }`
- [ ] **T2.3** Wire into `EconomyDashboard` — populate `ShopEntry.flagged` / `ItemStat.anomaly_flag` / `rmt_alert_count` from Tucker residuals instead of (or alongside) the threshold rule.
- [ ] **T2.4** GM dashboard: add "Market Factor" view (the item-factor loadings as a heatmap) and "Cross-Server Divergence" view (the zone-factor), alongside the existing anomaly section.
- [ ] **T2.5** Tests: synthetic RMT injection (K items with manipulated prices in 1 zone) → Tucker flags them; threshold rule misses them when the manipulation stays under the static threshold.

### GOAT gate (Game)

- [ ] **G1 (RMT detection beats threshold):** Inject 3 items whose prices are pumped 20% in zone A only (RMT transfer pattern) across 8 windows. Tucker residual flags all 3; a static 2×-median threshold rule misses them (20% is under 2×). **PASS** = Tucker recall > threshold recall.
- [ ] **G2 (no false positives on event spikes):** Inject a legitimate event-driven 50% price spike across ALL zones simultaneously (patch drop). Tucker does NOT flag it (it's low-rank — explained by the window factor). Threshold rule false-positives. **PASS** = Tucker FP = 0, threshold FP > 0.
- [ ] **G3 (modelless):** Pure closed-form HOSVD, no training. **PASS** = no `riir-train` dependency.
- [ ] **G4 (perf):** `detect_rmt_tucker` on (16 items, 16 windows, 16 zones) ≤ 5ms (cold GM-analytics path). **PASS** = mean ≤ 5ms.

---

## Phase 3 — Promotion + Plan 326 closure

- [ ] **T3.1** If Consumer 1 (chain) G1–G4 pass → the Tucker primitive has its first validated consumer. Update Plan 326's "Consumer / adoption status" section to cite Plan 328.
- [ ] **T3.2** If Consumer 2 (game) G1–G4 pass → second validated consumer. Same update.
- [ ] **T3.3** Once ≥1 consumer is validated, evaluate whether the `riir-neuron-db` `compact_tucker` integration (Plan 326 Phase 2) should be deprecated — the primitive is consumed directly from `katgpt-core` by these consumers; the riir-neuron-db wrapper adds no value for them.
- [ ] **T3.4** Benchmark record: `.benchmarks/328_tucker_consumer_applications.md` with both consumers' G1–G4 results.

---

## Why the N ≤ 16 cap is a non-issue here

Plan 326's `MAX_SHARDS_PER_TUCKER = SVD_MAX_RANK = 16` was a blocker for the riir-neuron-db "archive a 200-NPC zone" framing. For these consumers it's irrelevant:

- **Chain:** ≤ 16 curators per quorum is typical; chunk rounds into windows of ≤ 16.
- **Game:** chunk items by category (weapons, armor, mats — each ≤ 16) and zones into groups of ≤ 16. The factor model is per-chunk; cross-chunk aggregation is the GM dashboard's job.

The cap only hurt the "one giant batch" framing. The factor-model-anomaly framing naturally works in small correlated batches — exactly the regime Tucker is sized for.

---

## Risks

- **T2.1 data retention:** the MMO may not persist per-item-per-zone-per-window price history in a tensor-friendly shape. T0 must confirm; if absent, add a retention table (cheap — it's analytics data, not hot game state).
- **Chain quorum size:** if production quorums exceed 16 curators, T1.1 must chunk curators into sub-tensors (e.g., by stake-weighted clustering pre-pass). Detectable in T1.1; not a blocker.
- **Collusion ≠ correlation:** honest validators may legitimately co-vote (same client implementation, same mempool view). G2 (no false positives on honest voting) guards against this; the cohesion threshold may need tuning. The factor-model residual gives a *score*, not a boolean — the GM/chain-operator sets the threshold.

---

## References

- **Primitive:** [Plan 326](326_tucker_hosvd_factorization.md) — `katgpt-core/linalg::tucker` (DEFAULT-ON)
- **Chain integration point:** `riir-chain/src/consensus/curator_slashing.rs` (`detect_collusion`), `riir-chain/src/consensus/curator.rs` (`CuratorVote`, `CuratorConsensus`)
- **Game integration point:** `seal-online-remaster/crates/seal-gm-tools/src/state.rs` (`EconomyDashboard`, `ShopEntry`), `seal-online-remaster/crates/seal-gm-tools/src/tabs/shops.rs` (`anomaly_section`)
- **Benchmark record:** `.benchmarks/328_tucker_consumer_applications.md` (created in T3.4)
