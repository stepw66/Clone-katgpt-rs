---
name: research
description: Research workflow for distilling ML/AI papers into modelless inference primitives, freeze/thaw runtime patterns, and latent-space operations across the katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train repo quintet. Use when reading arxiv papers, deciding which repo a paper belongs in, creating .research/ notes or .plans/ files, implementing modelless inference primitives, or routing training-vs-inference insights. Enforces the 5-repo commercial strategy (public engine / private runtime / private chain / private neuron-db / private training), modelless-first constraint, latent-to-latent preference, and freeze/thaw-over-fine-tuning rule.
---

# Research Workflow — Modelless Inference, Freeze/Thaw, Latent-to-Latent

Training-method research lives in `riir-train`. This repo (`katgpt-rs`), `riir-ai` (freeze/thaw runtime + self-learn/adaptive NPCs + game systems), `riir-chain` (neuro-symbolic chain transport, LatCal, chain economics), and `riir-neuron-db` (neuron weight shards, BLAKE3/Merkle commitment, freeze/thaw envelope, consolidation, AnyRAG gateway, vibe KG triples) ship **runtime + latent-space operations**. No LoRA training, no adapter fine-tuning, no optimizer research here. If a paper's value is its training loop → `riir-train/.research`. If its value is a latent-space insight, a routing trick, a freeze/thaw pattern, a chain-commitment bridge, a neuron-shard primitive, or a modelless inference primitive → distill here.

## When to use this skill

Activate when the user (or you) are doing any of:

- Reading / fetching / summarizing an ML, AI, or systems paper (arxiv, PDF, blog).
- Deciding which of the 5 repos a paper or idea belongs in (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / riir-train).
- Creating a new `.research/NNN_*.md` note or `.plans/NNN_*.md` plan.
- Implementing a modelless inference primitive (pruner, bandit, router, speculative decode, KV-cache op, sparse attention, quantization-aware inference).
- Designing freeze/thaw snapshot cycles, adapter hot-swap, or runtime adapter routing.
- Designing latent-to-latent operations (dot-product projection, sigmoid gating, manifold geometry, spectral methods on activations).
- Designing MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick, fog-of-war, zone attention, emergent social/economic behavior).

Do NOT activate for: pure refactor tasks, bug fixes with no research angle, or ordinary feature work that doesn't touch the research/plans folders.

## Repos (siblings under the same parent)

- `katgpt-rs/` — public MIT engine. Generic modelless inference primitives. **No game IP, no chain IP, no neuron-shard IP.**
- `riir-ai/` — private game product. Freeze/thaw runtime, self-learn, game systems. **Chain was spun off to `riir-chain/` (see `.plans/001_chain_spinoff.md` in that repo); neuron_db was spun off to `riir-neuron-db/` (see `.plans/001_extraction_from_riir_chain.md` in that repo).**
- `riir-chain/` — private neuro-symbolic chain transport. LatCal (Lattice Calculus), `riir-chaind` daemon, chain economics, Solana-parity features, asset lifecycle / forensic fingerprinting, `catchup/` (Turso/libSQL persistence, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root`. **The sync-boundary bridge repo. Re-exports `riir-neuron-db` under its `neuron_db` feature, but the canonical shard source is `riir-neuron-db/`.**
- `riir-neuron-db/` — private leaf crate (extracted from `riir-chain/src/neuron_db/`). `NeuronShard` (`#[repr(C)]` Pod, zero-copy mmap), `ShardIndex` (lock-free `papaya::HashMap`), generic `MerkleTree`/`MerkleProof` (moved here from `riir-chain::catchup::merkle`), MAPE-K self-healing loop, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates + arch agent, `MerkleFrozenEnvelope` (freeze/thaw integrity), spectral initialization, `ShardCompactor`, dendritic LoRA branch view. **No chain dependency — usable standalone.**
- `riir-train/` — private training vault. Adapter training, optimizers, loss functions. Out of scope for this workflow — just note "→ riir-train" and stop.

(`riir-armageddon/` is a sixth sibling — arena/game-product domain types only, not a distillation target. Read its README for the raw-vs-latent boundary; do not distill into it.)

**Routing rule of thumb (chain vs neuron-db):** if the mechanism is about *how a shard is structured, committed, frozen, consolidated, retrieved, or projected* → `riir-neuron-db`. If it is about *how a shard is committed to a chain block, transported across quorum, or bridged to LatCal fixed-point* → `riir-chain`. The `LatCalWalletExt` trait (typed wallet accessors on `NeuronShard` using `LatCalMatrix`) stays in `riir-chain` because it is the bridge.

Always reference files with project-relative paths (e.g. `katgpt-rs/.research/238_*.md`, `riir-ai/.plans/NNN_*.md`, `riir-chain/.plans/NNN_*.md`, `riir-neuron-db/.plans/001_*.md`). The agent can `read_file` these directly.

## Read first (grounding) — MANDATORY pre-flight

**Hard rule:** before any distillation, verdict, or file creation, you MUST do **both** of these in this session:

1. **`read_file` all five READMEs** — these define repo purpose, current state, and the raw-vs-latent sync boundary the research must respect. Skipping this is the #1 cause of research notes that ignore the actual codebase architecture.
2. **`list_directory` all four `.research/` folders** — these hold the existing distillation corpus you must not duplicate. (Create `riir-chain/.research/` and `riir-neuron-db/.research/` if they do not yet exist and you are about to drop a note there.)

**Mandatory (before any verdict):**
- `katgpt-rs/README.md` (`read_file`) — public engine purpose, architecture, current feature set.
- `riir-ai/README.md` (`read_file`) — private runtime context (freeze/thaw, self-learn, game systems). Chain is no longer here; neuron_db is no longer here.
- `riir-chain/README.md` (`read_file`) — private neuro-symbolic chain transport, LatCal, chain economics, feature-flag umbrellas (`chain`, `chain_economics`, `chain_solana_parity`, `chain_catchup`, `chain_asset_*`, `shard_compactor`, `lora_posterior`). **Required reading for any LatCal / commitment / sync-bridge research. (Neuron-shard data structures and freeze/consolidation now live in `riir-neuron-db/` — chain re-exports via `neuron_db` feature.)**
- `riir-neuron-db/README.md` (`read_file`) — private neuron-shard leaf crate. `NeuronShard` Pod layout, BLAKE3/Merkle commitment, feature gates (`spectral_shard`, `shard_compactor`, `merkle_freeze`, `dendritic_lora`, `state_compression`). **Required reading for any shard / freeze-envelope / consolidation / AnyRAG / vibe-KG / Merkle-tree research.**
- `riir-armageddon/README.md` (`read_file`) — arena/game-product domain types, raw-vs-latent boundary, sync semantics, anti-cheat rules. The research MUST respect this boundary.
- `katgpt-rs/.research/` (`list_directory`) — public modelless research corpus (do not duplicate).
- `riir-ai/.research/` (`list_directory`) — private runtime/game research corpus (do not duplicate). Historical chain notes (pre-spin-off) still live here — grep them too.
- `riir-chain/.research/` (`list_directory` — create the folder on first use) — private chain research corpus. New chain-flavored notes land here, not in `riir-ai/.research/`.
- `riir-neuron-db/.research/` (`list_directory` — create the folder on first use) — private neuron-shard research corpus. New shard / freeze / consolidation / AnyRAG / vibe-KG notes land here, not in `riir-chain/.research/`.
- `riir-ai/crates/riir-engine/src/` (`list_directory`) — **runtime module tree = codebase vocabulary at the highest level.** Module names (`latent_functor/`, `cgsp_runtime/`, `micro_belief/`, `adapters/`, ...) are how the codebase describes its own mechanisms. Skipping this caused the Research DiPOD miss: `latent_functor/reestimation.rs` ships the exact "drift-triggered self-healing swap" pattern under the name "coherence-driven re-estimation scheduler" — invisible to a paper-vocabulary grep.
- `riir-ai/crates/riir-games/src/` (`list_directory`) — game systems module tree (same rationale).
- `riir-chain/src/` (`list_directory`) — chain module tree: `encoding/` (LatCal), `consensus/`, `economics/`, `asset_lifecycle/`, `forensic/`, `programs/`, `validator/`, `wallet/`, `batch/`, `catchup/`, `deploy/`, `shell/`. **(The old `neuron_db/` sub-tree is gone from here — it moved to `riir-neuron-db/src/`. The chain-side `LatCalWalletExt`, `DataTier`, `DATA_TIERS`, `build_tier_root`, `build_block_root` STAY in `riir-chain/src/catchup/merkle.rs`.)**
- `riir-neuron-db/src/` (`list_directory`) — **shard module tree = neuron-db vocabulary.** Files: `shard.rs` (NeuronShard Pod layout, dendritic branch view), `index.rs` (ShardIndex lock-free papaya), `merkle.rs` (generic MerkleTree/Proof — moved here from chain), `freeze.rs` (`MerkleFrozenEnvelope`), `mape_k.rs` (self-healing loop), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG escalation), `vibe.rs` (KG triple templates + arch agent), `spectral_flatness.rs` (lottery-ticket init), `shard_compactor.rs` (cold-tier compaction), `reconstruction_metrics.rs`. Skipping this caused pre-spin-off research to overclaim "novel shard mechanism" when `shard.rs` already shipped it.

If you have NOT `read_file` all five READMEs AND `list_directory` all four `.research/` folders AND the four runtime/chain/neuron-db crate src trees, STOP and do so now. Do not create any file until all thirteen are done.

Then read for additional context (as relevant to the topic):
- `katgpt-rs/src/` + `katgpt-rs/crates/katgpt-core/src/` — existing modelless primitives (ConstraintPruners, bandits, DDTree, speculative decode).
- `riir-ai/crates/` — runtime IP: `riir-engine`, `riir-games`, `riir-ffi`, `riir-data`, `riir-examples`. **(`riir-chain` and `riir-chaind` moved to the `riir-chain/` repo; `neuron_db` moved to the `riir-neuron-db/` repo — do not look for any of them here.)**
- `riir-chain/crates/` — chain daemon crate: `riir-chaind`. (LatCal, encoding, etc. live under `riir-chain/src/`; shard types come from `riir-neuron-db`.)
- `katgpt-rs/.plans/` + `riir-ai/.plans/` + `riir-chain/.plans/` + `riir-neuron-db/.plans/` — existing plans. **Do NOT list these in pre-flight.** Grep them during fusion search (§Workflow step 1), not as grounding — they describe what we *plan to build*, not what the repos *are*.

## Primary focus (distill HERE in katgpt-rs / riir-ai)

**Fusion-first mindset:** The highest-value Super-GOATs in this codebase come from **fusing 2–3 papers/primitives into a novel combination**, not from direct-mapping a single paper. Always grep `.research/` + `.plans/` for the 2–3 closest cousins before verdict, and ask: "what does paper × note A × note B produce that none of them alone can?" Examples that shipped: Gemini Fourier × LatCal (research 212 → plan 242); EGA × SpectralQuant (research 100 × 039); collapse-aware × bandit × sigmoid-margin (plans 212 × 157 × 061). See §Workflow step 1 for the full fusion protocol.

- **Latent-to-latent operations** — anything that stays in embedding/latent space: dot-product projections, cosine similarity retrieval, sigmoid-gated routing, manifold geometry, spectral methods on activations. Prefer operating on latents over decoding to tokens then re-encoding. **Fusion hook:** combine with freeze/thaw to version latent-direction vectors; combine with self-learn to update direction vectors from runtime curiosity signal.
- **Freeze/thaw patterns** — versioned weight snapshots, atomic hot-swap, lock-free read paths, BLAKE3/commitment-checked adapter reload, per-entity personality divergence via snapshot versioning. **Fusion hook:** combine with runtime adapter routing to dispatch by latent-state similarity; combine with self-learn to snapshot emergent NPC personalities.
- **Runtime adapter routing** — selecting between frozen adapters by state/objective/context (Dynamic Pair, Polytope, dMoE — all inference-time, zero training). **Fusion hook:** combine with freeze/thaw to make the adapter pool itself versioned and BLAKE3-committed; combine with bandits to learn routing policy online.
- **Self-learn / adaptive CoT** — runtime curiosity, entropy-driven exploration, collapse detection/recovery, latent prediction SSL, trajectory folding. No LLM training, no backprop through weights — runtime self-improvement via latent-space updates is welcome. **Fusion hook:** combine with MMORPG-scale game AI to give thousands of NPCs independent curiosity/entropy signals; combine with freeze/thaw to checkpoint learned latent directions.
- **Modelless inference primitives** — ConstraintPruners, bandits, DDTree, speculative decode, sparse attention, quantization-aware inference.
- **MMORPG-scale game AI** — thousands of concurrent NPCs each with independent latent state, real-time latency budgets (20Hz tick, plasma/hot tier), spatial partitioning + fog-of-war, emergent social/economic behavior (factions, trade routes, reputation), zone-level attention routing, crowd-scale curiosity/exploration signals. Latent ops must batch across many entities; raw sync must stay bit-identical for deterministic replay/anti-cheat.

### Super-GOAT factory modules — grep FIRST, explicitly

The highest-value latent-space Super-GOATs cluster in seven module trees. When grepping for fusion cousins and prior art, `list_directory` these explicitly — do NOT rely on keyword grep alone (vocabulary mismatch is the #3 cause of false verdicts):

| Module | What ships | Super-GOAT angle |
|---|---|---|
| `katgpt-rs/crates/katgpt-core/src/sense/` | HLA belief-state kernels, `evolve_hla`, `SenseModule::project`, ternary bit-plane projection | Per-NPC recurrent latent state — the runtime substrate for any "hidden state" / "belief" / "activation" paper |
| `riir-ai/crates/riir-engine/src/latent_functor/` | `zone_gating.rs`, `reestimation.rs`, `arithmetic.rs`, `cross_game.rs`, `k_selector.rs`, `quality_gate.rs` | **Game-theory in latent space** — functors as vector ops, coherence-driven re-estimation, zone-gated activation. Maps any "stage" / "application" / "bypass" / "collapse" paper |
| `riir-ai/crates/riir-engine/src/hla/` | `kernel.rs`, `forward.rs`, `types.rs` — per-NPC 8-dim latent state (valence/arousal/desperation/calm/fear + 3) | The emotional/cognitive latent state — maps any "subspace" / "width" / "channel" paper to per-NPC affect |
| `riir-ai/crates/riir-engine/src/cgsp_runtime/` | Curiosity-guided self-play, latent prediction SSL, MCTS collapse bridge | Runtime curiosity/exploration — maps any "self-learn" / "entropy-driven" / "collapse recovery" paper |
| `riir-neuron-db/src/` | `shard.rs` (NeuronShard Pod, `style_weights[64]`, dendritic branch), `freeze.rs` (`MerkleFrozenEnvelope`), `consolidation.rs` (Raven/δ-Mem), `gateway.rs` (AnyRAG escalation), `vibe.rs` (KG triple arch agent), `merkle.rs` (generic MerkleTree/Proof), `mape_k.rs` (self-healing loop), `spectral_flatness.rs` (lottery-ticket init), `shard_compactor.rs` | **Frozen latent-state storage + integrity + retrieval** — the persistence substrate for any "snapshot" / "integrity envelope" / "memory consolidation" / "external knowledge escalation" / "KG triple emission" paper. Maps any "memory" / "replay buffer" / "experience replay" / "spectral init" / "Merkle commitment" paper. Lives in the standalone `riir-neuron-db/` repo, not `riir-chain/src/neuron_db/`. |
| `riir-chain/src/encoding/latcal*.rs` + `latcal_fixed.rs` | Lattice Calculus: 2×2 matrix arithmetic obfuscation, fixed-point bridge, spectral fixed-point, batch determinant validation, DeFi programs | **The sync-boundary bridge** — deterministic, committed, raw-numeric. Maps any "fixed-point" / "deterministic commitment" / "raw↔latent bridge" / "arithmetic obfuscation" paper. LatCal is how latent ops become chain-committed raw values. **Lives in the standalone `riir-chain/` repo, not `riir-ai`.** |
| `katgpt-rs/crates/katgpt-core/src/dec/` | `operators.rs` (d=`exterior_derivative`, δ=`codifferential`, Δ=`hodge_laplacian`), `hodge.rs` (`hodge_decompose` exact/coexact/harmonic, `betti_numbers`, `harmonic_projector`), `flow.rs` (`DecFlowField` exact/coexact/harmonic channels), `terrain_cochains.rs` (Safety/Threat/Occupancy/Destruction typed cochains) — **shipped Plan 251, Research 219** | **The Generalized Stokes' Theorem substrate** — `d∘d=0` enforced by construction (tests verify `curl(grad)=0`, `div(curl)=0`). Maps any "divergence" / "boundary flux" / "line integral" / "curl" / "Hodge decomposition" / "Fokker-Planck" / "mass conservation" / "manifold geometry" / "exterior calculus" / "Stokes theorem" paper. The math ships; thin wrapper primitives (`boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) are Plan 314 (Research 296). **Curse-of-dimensionality caveat: boundary-vs-volume wins only for d ≤ 3 (game maps, HLA regions, KG embeddings) — NOT high-dim shards.** |

**Adapter routing, KV compression, and speculative decode are GOAT-tier framings. Latent-to-latent operations on HLA/functor/neuron-shard/LatCal state are Super-GOAT-tier framings. Attempt the Super-GOAT framing first.** Defaulting to adapter routing when a latent-space reframing is stronger is a documented failure mode (see R269 in §1.5).

## Redirect to riir-train (do NOT distill here)

**MANDATORY pre-check:** before redirecting ANY mechanism to riir-train, exhaust the modelless unblock paths in §3.5 below. A mechanism that *looks* training-only may be modelless-validable via freeze/thaw, raw/lora hot-swap, or latent-space correction. Only redirect if §3.5's decision protocol returns "genuine riir-train dependency".

If a paper is training-only (after §3.5 check) → note "→ riir-train" in one line and stop. Do not create files in this session for it.

**By topic:**
- LoRA / OFT / SPEFT / IA3 / QLoRA / ManifoldE / BAKE / GPart / MSA / Dendritic and all adapter-**training** methods.
- Training optimizers (Muon, Adam variants, symmetry-compatible optimizers).
- Training loss functions, curricula, distillation recipes.
- Quantization-aware **training** (quantization-aware **inference** stays here).
- DPO / GRPO / SFT / RL **training** pipelines (runtime GRPO self-play stays in `riir-ai` — it updates latent state, not weights).
- Anything that requires backpropagation through base weights.

**By user-request phrasing (these mean "→ riir-train"):**
- "Train a LoRA adapter to do X"
- "Fine-tune with method Y"
- "Optimizer Z improves convergence"
- "Distillation recipe from teacher to student"
- "Quantization-aware training" (but "quantization-aware inference" stays here)
- "DPO/GRPO/SFT/RL training pipeline" (but runtime GRPO self-play stays in riir-ai)

## Distillation targets (5-repo strategy)

Per `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (public routing-rules subset; full moat analysis at `riir-ai/.research/003_*` internal, revised 2026-06-27 to add the tier model + cognitive/reasoning moat):

| Repo | Role | What lands here |
|------|------|-----------------|
| `katgpt-rs` (public, MIT) | Engine — modelless inference framework | Generic primitives: ConstraintPruner traits, bandits, DDTree, speculative decode, sparse attention kernels. **No game IP, no chain IP, no neuron-shard IP.** |
| `riir-ai` (private) | Game product — freeze/thaw runtime, self-learn, game systems | Runtime IP: `LoRAWeightVersion`, `LoRAHotSwap`, `dispatch_lora_merge`, `TrainingProvider` trait, routing, game systems. **No chain code, no neuron-shard code — chain lives in `riir-chain/`, shards live in `riir-neuron-db/`.** |
| `riir-chain` (private) | Neuro-symbolic chain transport — LatCal, chaind | Chain IP: LatCal encoding/bridges, split-key ledger, chain economics, Solana-parity features, asset lifecycle / forensic, `riir-chaind` daemon, validator SDK bridges, `catchup/` (Turso/libSQL, quorum), `DataTier` / `DATA_TIERS` / `build_tier_root` / `build_block_root`. **Re-exports `riir-neuron-db` via `neuron_db` feature, but the shard source of truth is `riir-neuron-db/`.** |
| `riir-neuron-db` (private) | Neuron-shard leaf crate — shards, freeze, consolidation, retrieval | Shard IP: `NeuronShard` Pod layout + `style_weights[64]` + dendritic branch, `ShardIndex` lock-free papaya, generic `MerkleTree`/`MerkleProof`, `MerkleFrozenEnvelope`, MAPE-K self-healing, Raven/δ-Mem consolidation, AnyRAG escalation gateway, vibe KG triple templates + arch agent, spectral lottery-ticket init, `ShardCompactor`. **No chain dependency — usable standalone.** |
| `riir-train` (private) | Training research vault | **Only if the paper's value is its training method.** Out of scope for this workflow — just note "→ riir-train" and move on. |

Distill into:
- **Modelless** → `katgpt-rs/.research/` + `katgpt-rs/.plans/` + `katgpt-rs/src/` (or `katgpt-rs/crates/katgpt-core/`)
- **Runtime/game** → `riir-ai/.research/` + `riir-ai/.plans/` + `riir-ai/crates/`
- **Chain / LatCal / sync-bridge / commitment / quorum / catchup** → `riir-chain/.research/` (create if missing) + `riir-chain/.plans/` + `riir-chain/src/` (or `riir-chain/crates/`)
- **Neuron shards / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction** → `riir-neuron-db/.research/` (create if missing) + `riir-neuron-db/.plans/` + `riir-neuron-db/src/`
- **Training-only** → note the redirect, do not create files in this session

## Workflow

### 0. Read & classify the paper

Fetch via `https://r.jina.ai/https://arxiv.org/pdf/{ID}` (per AGENTS.md). Ask: *is the value in the training loop, or in a latent-space / inference / routing insight?* If training-only → note "→ riir-train", stop.

### 1. Distill fundamentally — fuse, don't just direct-map

Don't direct-map the paper to our code. Find the transferable primitive: the geometric, spectral, or information-theoretic insight that works without the paper's training setup. **Then look for fusion opportunities**: cross-pollinate this paper's insight with existing `.research/` notes, `.plans/`, and shipped primitives to synthesize a *novel* combination. The highest-value Super-GOATs in freeze/thaw runtime and self-learn/adaptive CoT almost always come from **fusing** 2–3 papers, not from a single-paper direct mapping.

**Fusion examples that shipped:**
- Gemini Fourier × LatCal → `katgpt-rs/.research/212_Gemini_Fourier_LatCal_Fusion_Verdict.md` → `katgpt-rs/.plans/242_Fourier_Smoothed_Potential_Fields_LEO.md`
- EGA spectral salience × SpectralQuant eigenbasis KV compression → `katgpt-rs/.research/100_*.md` + `039_*.md`
- Collapse-aware × bandit coverage × sigmoid margin → `katgpt-rs/.plans/212_*`, `157_*`, `061_*`
- G-Zero self-play × Hint-δ bandit × absorb-compress → `katgpt-rs/.plans/049_*` (modelless self-play distillation, 1.16M cycles/sec)

**Fusion protocol:**
1. **MANDATORY — grep ALL FIVE repos in this session, BOTH layers (notes AND code). Do NOT stop after the first repo or the first layer.** Run keyword / paper-title / author / primitive-name grep across:
   - `katgpt-rs/.research/` + `katgpt-rs/.plans/` (intent — what we planned)
   - `riir-ai/.research/` + `riir-ai/.plans/` (intent — runtime/game, plus historical chain notes pre-spin-off)
   - `riir-chain/.research/` + `riir-chain/.plans/` (intent — current chain research; `.research/` may need creating on first use)
   - `riir-neuron-db/.research/` + `riir-neuron-db/.plans/` (intent — current shard research; `.research/` may need creating on first use. **Historical shard notes (pre-spin-off) still live in `riir-chain/.research/` or `riir-ai/.research/` — grep them too.**)
   - `katgpt-rs/src/` + `katgpt-rs/crates/` (shipped primitives — what actually exists)
   - `riir-ai/crates/` (shipped runtime — no longer contains `riir-chain`/`riir-chaind`/`neuron_db`)
   - `riir-chain/src/` + `riir-chain/crates/` (shipped chain — LatCal, encoding, economics, forensic, catchup, etc. **No more `neuron_db/` here — it moved to `riir-neuron-db/src/`.**)
   - `riir-neuron-db/src/` (shipped shards — `shard.rs`, `freeze.rs`, `consolidation.rs`, `gateway.rs`, `vibe.rs`, `merkle.rs`, `mape_k.rs`, `spectral_flatness.rs`, `shard_compactor.rs`)
   - `riir-armageddon/crates/` (shipped game/arena domain types, raw-vs-latent boundary)
   - **Super-GOAT factory modules** (from §Primary focus) — `list_directory` these explicitly even if the paper looks pure-training: `katgpt-rs/crates/katgpt-core/src/sense/`, `riir-ai/crates/riir-engine/src/latent_functor/`, `riir-ai/crates/riir-engine/src/hla/`, `riir-ai/crates/riir-engine/src/cgsp_runtime/`, `riir-neuron-db/src/` (shards/freeze/consolidation/AnyRAG/vibe/merkle), `riir-chain/src/encoding/latcal*.rs`, `katgpt-rs/crates/katgpt-core/src/dec/` (Stokes/exterior-derivative/Hodge — maps any divergence/boundary/line-integral/Fokker-Planck/manifold-geometry paper)

   (riir-train is deliberately excluded — training methods are out of scope for this workflow.)

   Two layers, five repos. The closest cousin is frequently in the OTHER repo (e.g., a `katgpt-rs` modelless primitive fused with a `riir-chain` LatCal commitment bridge — see Gemini Fourier × LatCal; or a `riir-neuron-db` freeze envelope fused with a `riir-ai` runtime adapter hot-swap) OR in the CODE not the notes. **Notes describe intent; code describes what shipped.** A mechanism can ship without a research note — e.g., HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; a notes-only grep misses it and produces a false Super-GOAT claim (this exact failure happened on Research 242 — verdict had to be revised Super-GOAT → GOAT). If you only grep `katgpt-rs/.research/`, you will miss both axes and produce a duplicate, weaker note, or an overclaimed verdict.

2. **MANDATORY — vocabulary translation before grepping.** Papers and our codebase use different words for the same mechanism. Before any grep, list the paper's 3–5 key mechanism terms, then for EACH, brainstorm ≥2 codebase-equivalent terms by asking: "if we shipped this, what would we call it?" Then grep BOTH sets.

   **Standing latent-state vocabulary (ALWAYS include, even for non-latent papers — most architecture/training papers have a latent-space reframing):**
   - "residual stream" / "hidden state" / "activation" → "HLA state", "belief state", "latent subspace", "sense projection"
   - "layer" / "depth" / "stage" → "decision stage", "functor application", "cgsp cycle", "consolidation tick"
   - "width" / "dimension" / "capacity" → "latent subspace", "active projection channel", "sense channel"
   - "carry-forward" / "bypass" / "skip" → "leaky integrator", "dormant subspace", "decay gate", "persistence"
   - "collapse" / "degeneration" / "valley" → "coherence decay", "re-estimation trigger", "staleness"
   - "bottleneck" / "narrowing" → "subspace projection", "channel selection", "zone gating"
   - "fixed-point" / "deterministic" / "committed" → "LatCal", "lattice calculus", "BLAKE3 commitment", "raw scalar bridge"
   - "divergence" / "flux" / "∇·F" / "density change" → "codifferential", "δ", "DEC divergence", "belief_mass_divergence"
   - "curl" / "vorticity" / "∇×F" / "circulation" → "d₁", "DEC curl", "exterior_derivative rank 1→2"
   - "boundary" / "∂M" / "frontier" / "perimeter" → "exterior_derivative", "d", "coboundary operator", "boundary_flux_mass"
   - "line integral" / "trajectory energy" / "path cost" / "geodesic cost" → "line_integral", "edge field sum", "rank-1 cochain path sum"
   - "Stokes theorem" / "divergence theorem" / "Green's theorem" / "Gauss" / "generalized Stokes" → "DEC identity d∘d=0", "curl(grad)=0", "div(curl)=0", "hodge_decompose"
   - "Hodge decomposition" / "exact/coexact/harmonic" / "Helmholtz" / "conservative/solenoidal" → "hodge_decompose", "DecFlowField", "exact_flow/coexact_flow/harmonic_flow"
   - "Fokker-Planck" / "continuity equation" / "mass conservation" / "probability flow" → "belief_mass_divergence", "codifferential on belief cochain"
   - "cell complex" / "mesh" / "simplicial" / "cubical" → "CellComplex", "CochainField", "grid_2d"

   Example (DiPOD paper → riir-ai code):
   - "double drift" / "ELBO drift" → "coherence decay", "staleness", "divergence"
   - "self-distillation" → "re-estimation", "re-derive", "recommit"
   - "tight bound" / "adequate estimator" → "coherence > tau", "parallelism quality", "confidence gate"
   - "policy-preserving" → "atomic Arc swap", "readers keep old snapshot"
   - "drop-in regularizer" → "feature flag", "warm-tier scheduler tick"

   Grep ONLY paper vocabulary → misses `latent_functor/reestimation.rs` (which ships DiPOD's exact pattern under the name "coherence-driven re-estimation scheduler"). Grep BOTH sets → hits it on the first pass. **This is the #2 cause of false Super-GOAT claims (after notes-only grep) — and arguably worse, because the mechanism DOES have a research note (Research 123 + Plan 303) that also uses codebase vocabulary, so even a notes grep misses it.**

   Example (Stokes/divergence-theorem paper → katgpt-rs DEC code):
   - "divergence" / "flux" / "density tracking" → "codifferential", "δ", "DEC divergence"
   - "boundary integral" / "CDF via boundary" / "surface flux" → "exterior_derivative", "d", "coboundary", "boundary_flux_mass"
   - "line integral" / "path energy" → "line_integral", "rank-1 cochain sum"
   - "Stokes theorem" / "∫_M dω = ∫_∂M ω" → "DEC identity d∘d=0", "curl_of_gradient_is_zero"
   - "Hodge decomposition" / "exact/coexact/harmonic" → "hodge_decompose", "DecFlowField"
   - "Fokker-Planck" / "continuity equation" → "belief_mass_divergence", "codifferential"

   Grep ONLY paper vocabulary → ZERO hits across all repos (a corpus grep for `stokes|divergence theorem|boundary integral|fokker-planck` returns nothing). Grep BOTH sets → hits `dec/operators.rs` (`codifferential`, `exterior_derivative`), `dec/hodge.rs` (`hodge_decompose`), `dec/flow.rs` (`DecFlowField`). **This is the #5 canonical failure mode (Research 296, 2026-06-24): the Generalized Stokes' theorem machinery ships as DEC operators (Plan 251), but no note framed it in Stokes-theorem vocabulary, so a paper-vocabulary grep missed BOTH notes AND code.** The standing DEC vocabulary above is the prophylactic.

3. **MANDATORY — latent-space reframing before verdict.** Before any verdict, re-cast the paper's core mechanism as a latent-to-latent operation on the codebase's latent-state kernels (the seven Super-GOAT factory modules above). Ask explicitly: "How does this mechanism look when operating on (a) HLA's per-NPC latent state, (b) `latent_functor/` operations, (c) `cgsp_runtime/` curiosity signals, (d) LatCal fixed-point commitment (in `riir-chain/src/encoding/`), (e) `NeuronShard` style_weights / dendritic branch / `MerkleFrozenEnvelope` / Raven consolidation / AnyRAG escalation (in `riir-neuron-db/src/`), (f) DEC Stokes-calculus operators (`katgpt-rs/crates/katgpt-core/src/dec/` — `exterior_derivative` d, `codifferential` δ, `hodge_decompose`, `DecFlowField` exact/coexact/harmonic)?" If your fusion idea only touches adapter routing / KV compression / speculative decode without a latent-state reframing, you are likely in GOAT territory and have probably missed the Super-GOAT angle. The latent reframing is mandatory even for papers that look pure-training/architecture — most have a latent subspace / stage-gating / persistence / memory-consolidation / manifold-geometry angle that lands in HLA/functor/neuron-shard/DEC.

4. **Zero grep hits ≠ novelty.** If your paper-vocabulary grep AND your codebase-vocabulary grep BOTH return zero hits, that is evidence of one of three things, in order of likelihood: (a) you are still using the wrong vocabulary — try a third semantic angle (e.g., grep for the *output behavior* like "swap when X" instead of the *mechanism name* like "tightness monitor"); (b) the mechanism is genuinely not shipped; (c) the mechanism is novel. Do NOT jump to (c). Default to (a): re-grep with at least one more semantic alternative before claiming "no prior art".
5. After finding the transferable primitive of *this* paper, list the 2–3 closest existing notes/plans **across all five repos** and ask: "what novel combination of this paper + note A + note B produces a capability none of them has alone?" Write that combination into the research note's §Distillation as a **Fusion** subsection, even if you don't plan it yet.
6. Verdict by the commercial strategy doc (`003_*.md`): **Super-GOAT** > GOAT > Gain > Pass (see §Verdict tiers below). **A fusion that produces a new capability class is a strong Super-GOAT candidate — check the novelty gate (§1.5).**
7. Create research `.md` at the right repo (see table above).

**File naming:** `{NNN}_{Short_Title_with_Underscores}.md` where NNN is the next free number (zero-padded to 3 digits, e.g. `239_`, `240_`). Check the folder first — numbers may be non-contiguous; pick the next free slot.

**Research note format** (see `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` for a canonical example):

```markdown
# Research NNN: <Title>

> **Source:** <paper title + arxiv link + authors + date>
> **Date:** YYYY-MM-DD
> **Status:** Active | Done | Shelved
> **Related Research:** NNN (short note), ...
> **Related Plans:** NNN (short note), ...
> **Cross-ref (riir-ai / riir-chain / riir-neuron-db):** Research NNN, Plan NNN   ← only if cross-repo (game runtime → riir-ai; chain/LatCal → riir-chain; shards/freeze/consolidation/AnyRAG/vibe → riir-neuron-db)
> **Classification:** Public | Private   ← katgpt-rs notes are always Public

---

## TL;DR

<2-4 sentences: the distilled primitive, why it matters here, what it unblocks>

**Distilled for katgpt-rs (modelless, inference-time):**
<the transferable insight, stripped of training setup>

---

## 1. Paper Core Findings
...
## 2. Distillation
...
## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism (no prior art) + new capability class + product selling point + force multiplier (≥2 pillars). Creates a moat. | Open primitive → katgpt-rs. **Architectural guide → riir-ai/.research/ (game runtime) OR riir-chain/.research/ (chain/LatCal) OR riir-neuron-db/.research/ (shards/freeze/consolidation/AnyRAG/vibe/Merkle)**. Plans → appropriate repo(s) as needed. |
| **GOAT** | Provable gain (latency/quality/security) over existing approach, but not a new class of capability. Promotes to default if it wins. | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement, useful but not headline-worthy. | Plan only, behind feature flag. |
| **Pass** | Not relevant to modelless/latent/freeze-thaw/runtime, OR training-only (→ riir-train note, stop). | One-line note. No files created in this session. |

**One-line reasoning required for each verdict.** For Super-GOAT: state the selling point explicitly.
```

### 1.5. Novelty gate — is this Super-GOAT?

Before planning, score novelty. Ask all four:

1. **No prior art?** Grep `.research/` + `.plans/` across all repos AND grep the shipped code (`katgpt-rs/src/`, `katgpt-rs/crates/`, `riir-ai/crates/`, `riir-chain/src/`, `riir-chain/crates/`, `riir-neuron-db/src/`, `riir-armageddon/crates/`) for the primitive name and mechanism keywords. **You MUST grep BOTH paper vocabulary AND codebase-vocabulary alternatives (see §Workflow fusion protocol step 2 — vocabulary translation).** **Notes describe intent; code describes what shipped.** A mechanism can ship under either of two failure modes:
   - **No notes framing at all** — canonical example: HLA's `evolve_hla` (`katgpt-rs/crates/katgpt-core/src/sense/reconstruction.rs`) is a per-NPC recurrent belief-state kernel with no `.research/` note framing it as such; missing it caused the Research 242 Super-GOAT overclaim.
   - **Notes framing uses different vocabulary than the paper** — canonical example: DiPOD's "interleave self-distillation when ELBO drifts" is shipped as `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` "coherence-driven re-estimation scheduler when coherence < tau_reest". Research 123 + Plan 303 DO frame the mechanism, but using codebase vocabulary, so a paper-vocabulary grep misses it on BOTH notes AND code layers. This is strictly worse than the `evolve_hla` failure: even a diligent notes grep fails. **Vocabulary translation (fusion protocol step 2) is the only defense.**
   If the code already covers the mechanism → not novel, Gain at best. **This three-layer check (notes + code + vocabulary translation) is mandatory — notes-only is the #1 cause of false Super-GOAT claims; paper-vocabulary-only is the #2 cause; skipping the seven Super-GOAT factory modules is the #3 cause.**
2. **New class of behavior?** Not better numbers, but something no incumbent can do (a new capability, not an optimization).
3. **Product selling point?** Can you finish the sentence: "Our NPCs/systems do X that no competitor can"? If you can't → Gain.
4. **Force multiplier?** Connects to ≥2 existing pillars/systems (check connection map in `.research/`). Solo novelty without integration = GOAT, not Super-GOAT.

**Documented failure (R269, 2026-06-19):** `> <former` (variable-width transformers) initially verdict'd Pass, then revised to "fusion TBD" with an **adapter-routing** framing after user pushback on on-the-fly LoRA. User's second pushback ("aim for latent-to-latent, HLA, functor") revealed the stronger framing the skill should have found first: variable width = **stage-gated HLA subspace activation** (combat→survival subspace, dialog→social subspace); carry-forward = **dormant subspace persistence** via HLA leaky integrator; ×-shape = **LatCal-style projection profile** across decision stages. The adapter-routing framing was GOAT-tier at best; the latent-functor-LatCal framing is Super-GOAT-tier. **Lesson: default to the latent-space reframing (fusion protocol step 3) before committing any verdict. Adapter routing is a fallback framing, not the primary one.**

**Documented failure (R296, 2026-06-24):** Stokes-calculus / divergence-theorem / Fokker-Planck research nearly produced a false Super-GOAT ("boundary-only mass via Stokes is novel!"). Corpus grep for `stokes|divergence theorem|boundary integral|fokker-planck` returned ZERO hits — but the Generalized Stokes' theorem machinery ALREADY SHIPS as the DEC module (`katgpt-rs/crates/katgpt-core/src/dec/`, Plan 251, Research 219): `exterior_derivative` (d), `codifferential` (δ = discrete divergence), `hodge_decompose` (exact/coexact/harmonic = Helmholtz), `DecFlowField`. Tests verify `curl(grad)=0` and `div(curl)=0` by construction. Only three thin wrapper primitives (`boundary_flux_mass`, `belief_mass_divergence`, `line_integral`) were missing — Plan 314. **Lesson: a paper-vocabulary grep for `stokes`/`divergence theorem`/`fokker-planck` misses BOTH notes AND code because the math ships under operator names (`exterior_derivative`, `codifferential`, `hodge_decompose`). This is the #5 canonical failure. Vocabulary translation (fusion protocol step 2) with the standing DEC vocabulary is the only defense — DEC is now the 7th Super-GOAT factory module.**

**If YES to all 4 → verdict = Super-GOAT.** Mandatory outputs:
1. **Open primitive** → `katgpt-rs` (generic math, no game semantics).
2. **Architectural GUIDE** → the private selling-point doc. **Pick the repo by where the selling point lives**: `riir-ai/.research/NNN_*.md` for game-runtime / HLA / functor / self-learn selling points; `riir-chain/.research/NNN_*.md` for chain / LatCal / commitment / quorum / catchup / sync-bridge selling points (create folder on first use); `riir-neuron-db/.research/NNN_*.md` for shard / freeze envelope / consolidation / AnyRAG / vibe KG / Merkle tree / spectral init / shard compaction selling points (create folder on first use). If the selling point spans multiple repos (e.g., latent ops that cross the chain sync boundary via a shard commitment), create the primary guide in the repo that owns the boundary being crossed, and cross-reference from the others. The guide MUST include:
   - TL;DR with commercial value (the selling point in one sentence)
   - Distilled primitive (how the mechanism works modellessly)
   - Connection map (which existing systems it multiplies)
   - Latent vs raw boundary (what crosses sync, what stays local)
   - What stays private vs open
   - Validation protocol (how to prove it's Super-GOAT, not just hype)
   - Implementation priority table (P0–P3)
3. **Plan(s)** → `katgpt-rs/.plans/` (open) and/or `riir-ai/.plans/` (private runtime) and/or `riir-chain/.plans/` (private chain) and/or `riir-neuron-db/.plans/` (private shards).

**If NO to any → proceed to GOAT/Gain verdict.** Plan only, no guide.

> **Rule:** Super-GOAT ideas are the private IP moat. The open primitive is the adoption hook; the riir-ai/riir-chain/riir-neuron-db guide is the selling point. Never ship the guide publicly. Never skip the guide for a Super-GOAT — that's losing the knowledge.
>
> **No "candidate" escape hatch.** If you write "all 4 YES", "passes the novelty gate", or "Super-GOAT candidate" anywhere in a note (main verdict OR a fusion subsection), the mandatory outputs above apply **in this same session** — open primitive in katgpt-rs, **private guide (riir-ai OR riir-chain OR riir-neuron-db, by selling-point domain) created now**, plans as needed. The guide *contains* the validation protocol (G1–Gn gate), so you create it **before** running the gate, not after. Deferring the guide "until validation passes" inverts the order and silently drops the moat doc — this is the #1 way selling points leak into the public repo.
>
> If you are NOT confident enough to commit all 4 YES right now, **do not write "Super-GOAT candidate"**. Write "fusion idea — novelty TBD, needs Q1–Q4 check before verdict" and create an issue in `.issues/` to track the follow-up. "Candidate" is not a deferred-commitment escape hatch — it either triggers the guide now, or it gets downgraded to an issue.

### 2. If gain (or GOAT), plan it

Add plan `.md` to `katgpt-rs/.plans/` (modelless), `riir-ai/.plans/` (runtime/game), and/or `riir-chain/.plans/` (chain / LatCal / neuron_db). Use `## Phase N` sections with `- [ ]` per task (mark `- [x]` when done). **Never** plan into `riir-train` from this workflow.

> Super-GOAT plans should be created AFTER the riir-ai guide. The guide is the strategy; the plan is the execution.

**Plan format** (see `katgpt-rs/.plans/271_attention_matching_compaction.md` for a canonical example):

```markdown
# Plan NNN: <Title>

**Date:** YYYY-MM-DD
**Research:** [katgpt-rs/.research/NNN_*.md](../.research/NNN_*.md)
**Source paper:** [arxiv ID.NNN](https://arxiv.org/abs/ID) — <short cite>
**Target:** `katgpt-rs/src/<module>/` (new module) + Cargo feature `<feature_name>`
**Status:** Active — Phase N <state>

---

## Goal

<one paragraph: what ships, what it enables, GOAT gate>

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** <concrete, verifiable task>
- [ ] **T1.2** ...
```

**GOAT gate rule** (AGENTS.md): every plan that introduces a new technique must have a feature flag and a benchmark proving the gain before promoting to default. Demote the loser if the new technique wins.

### 3. Implement to unblock

If a plan is blocked by a missing primitive, implement the minimal version. After GOAT check + proof of gain: promote to default if it wins, demote the loser.

### 3.5. Modelless unblock protocol — MANDATORY before any riir-train deferral

**Hard rule:** before deferring ANY GOAT gate, plan task, or mechanism to riir-train ("this needs training"), you MUST exhaust all modelless correction paths first. A gate that *appears* to need training may be passable modellessly via freeze/thaw, raw/lora hot-swap, or latent-space correction. Deferring to riir-train without checking is a documented failure mode (see AC-Prefix G1, Plan 313, 2026-06-24).

**The three modelless unblock paths (check ALL before deferring):**

1. **Freeze/thaw snapshot correction** (`riir-neuron-db/src/freeze.rs`, `MerkleFrozenEnvelope`) — can a frozen snapshot state, thawed at inference, fix the issue? If the failure is a systematic bias from a runtime construction (e.g., doubled signal, position mismatch, attention pattern asymmetry), a corrected snapshot + thaw may eliminate it without gradient descent.
2. **Raw/lora reader-writer hot-swap** (`LoraPair { reader, writer }`, Plan 025; `LoRAHotSwap`, `dispatch_lora_merge` in riir-ai) — can a **deterministically constructed** (not trained) reader or writer adapter fix the issue? Applying a pre-constructed LoRA overlay is modelless (weight addition, no backprop). The question is: can the correction be derived in closed form (e.g., scale-by-0.5, zero-out-specific-positions, identity-minus-projection) rather than learned via gradient descent?
3. **Latent-space correction** (dot-product projection + sigmoid gate, per constraint #2) — can the bias be corrected by projecting the latent state onto a correction direction and gating the output? This is the modelless analog of a trained adapter: instead of learning the correction, derive it analytically from the failure mode.

**Decision protocol:**

```
Gate/mechanism appears to need training
  → Does the failure have a SYSTEMATIC, characterizable cause (e.g., "signal doubled", "position offset", "attention asymmetry")?
    NO → genuine riir-train dependency. Note "→ riir-train", stop.
    YES → Can freeze/thaw (path 1) fix it? Check the freeze envelope API.
      NO → Can a deterministically constructed reader/writer LoRA (path 2) fix it? Check the LoraPair API.
        NO → Can a latent-space projection/gate (path 3) fix it?
          NO → genuine riir-train dependency. Note "→ riir-train", stop.
          YES → implement the latent correction modellessly. Gate is MODELLESS-VALIDABLE.
        YES → construct the LoRA correction modellessly. Gate is MODELLESS-VALIDABLE.
      YES → freeze the corrected state. Gate is MODELLESS-VALIDABLE.
    
  → MODELLESS-VALIDABLE gates must be implemented and checked BEFORE any riir-train deferral.
  → If all three paths fail, THEN note "→ riir-train" with explicit documentation of WHY each path failed.
```

**Documentation requirement:** every "→ riir-train" deferral MUST include:
- Which of the three modelless paths were checked.
- Why each failed (concrete reason, not "doesn't apply").
- What specifically requires gradient descent that no deterministic construction can provide.

**Canonical failure — AC-Prefix G1 (Plan 313, 2026-06-24):** G1 ("AC-GPT conditional logprob matches iterative-MLM to 1e-4") failed at 7.5e-4 on untrained micro-GPT. The agent deferred to riir-train WITHOUT checking whether the doubled-signal bias (each `xc` token appears as copy in r0 AND in-place in r1) could be corrected modellessly via a deterministically constructed reader-LoRA that down-weights in-place `xc` contributions. The bias IS systematic and characterizable — exactly the case where path 2 (raw/lora hot-swap) might work. The deferral was premature. **Lesson: systematic, characterizable biases are modelless-correctable candidates, NOT automatic riir-train dependencies. Check path 2 (deterministic LoRA construction) first.**

### 4. Search if curious

Keyword search arxiv:

```
https://r.jina.ai/https://arxiv.org/search/advanced?advanced=&terms-0-operator=AND&terms-0-term={KEYWORD}&terms-0-field=abstract&classification-computer_science=y&classification-mathematics=y&classification-physics_archives=all&classification-statistics=y&classification-include_cross_list=include&date-filter_by=all_dates&date-year=&date-from_date=&date-to_date=&date-date_type=submitted_date&abstracts=show&size=50&order=-announced_date_first
```

Good keywords: `latent space routing`, `adapter hot-swap`, `inference-time composition`, `spectral pruning`, `sigmoid gating`, `snapshot consistency`, `lock-free weight swap`.

## Constraints (non-negotiable)

1. **Modelless first** — inference-time only. No LLM training, no backprop through base weights. Closest to "training" allowed: freeze/thaw snapshot cycles, raw/lora reader-writer hot-swap with **deterministically constructed** adapters (not trained), and latent-space direction-vector updates at runtime. **Before deferring any gate/mechanism to riir-train, exhaust §3.5 modelless unblock paths (freeze/thaw, raw/lora, latent correction).**
2. **Latent-to-latent preferred** — operate in embedding/latent space as long as possible. Decode to tokens or project to raw scalars only at the boundary. Use dot-product + **sigmoid** (never softmax) for projections onto learned direction vectors. Semantic domain (emotion, mood, curiosity, style) → latent. Physical domain (position, HP, wallet balance) → raw, deterministic, synced.
3. **Freeze/thaw over fine-tuning** — the only weight mutation allowed at runtime is swapping a frozen snapshot (atomic, versioned, BLAKE3-checked) or applying a deterministically constructed LoRA overlay (raw/lora hot-swap, no gradient descent). Never mutate weights in-place during inference. If a paper needs gradient updates (after exhausting §3.5 modelless paths), redirect to riir-train.
4. **Self-learn / adaptive CoT welcome** — runtime curiosity, latent prediction, trajectory folding, collapse detection. These update latent state / direction vectors / routing tables, NOT base weights.
5. **5-repo discipline** — katgpt-rs (public engine) → riir-ai (private runtime/game) → riir-chain (private chain) → riir-neuron-db (private neuron-shard leaf) → riir-train (private training). Keep the commercial strategy intact. Training know-how never leaks to katgpt-rs; chain IP stays in `riir-chain/`, not `riir-ai/`; neuron-shard IP stays in `riir-neuron-db/`, not `riir-chain/` (chain only re-exports via the `neuron_db` feature).
6. **SOLID, DRY** — per `katgpt-rs/.contexts/optimization.md`. Zero-allocation hot paths. Pre-computed lookup tables. Fixed-size arrays for bounded domains.
7. **Tests/examples** — before/after showing the gain (latency, quality, or security). For latent ops: show the projection preserves ranking. For freeze/thaw: show readers never see torn snapshots.
8. **CPU/GPU/ANE auto-route** — threshold-adaptive dispatch. Plasma (µs, CPU/SIMD) → Hot (sub-ms, GPU) → Warm/Cold (ms+, GPU/ANE). Latent ops that fit in L1 cache stay on SIMD; manifold ops that need batched matmul go to GPU.
9. **Plasma → Hot → Warm → Cold → Freeze tiering** — aim for perf on game side (plasma/hot latency budget) AND security on chain side (cold/freeze commitment, BLAKE3-hashed, tamper-evident). Latent state that crosses the sync boundary MUST be raw scalars (valence/arousal/desperation/calm/fear), never the full embedding vector.

## Latent vs raw space rules (critical for game AI)

Reinforce these when designing game systems or chain state:

- **Physical domain** (position, velocity, HP, wallet balance): MUST remain raw exact values. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic domain** (emotion, mood, curiosity, style, habit): SHOULD operate in latent space via dot-product + sigmoid onto learned direction vectors.
- **Social domain** (encounters, relationships, factions): SHOULD produce KG triples from proximity in latent/embedding space, not from raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum commit → Cold tier, it MUST be raw and deterministic. If data is consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-allocation, gateable by feature flag, and not introduce sync dependency.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat validation.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone-level KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent behavior, not a bug.

## Cross-references (read on demand)

- `katgpt-rs/.contexts/optimization.md` — perf rules (zero-alloc, SIMD, rayon, caching)
- `katgpt-rs/.contexts/ibraheemdev-papaya-v0.2.3-examples.md` — papaya lock-free hashmap usage
- `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md` — public routing-rules subset (5-repo roles, What/How rule, Super-GOAT routing). **Full strategy doc** (moat analysis, "why hard to replicate", cognitive/reasoning moat, tier model detail) is internal at `riir-ai/.research/003_Commercial_Open_Source_Strategy_Verdict.md` (moved 2026-06-27 — too commercially sensitive for a public MIT repo).
- `katgpt-rs/.research/004_LoRA_Architecture_Verdict.md` — LoRA / validator terminology
- `katgpt-rs/.research/005_Artifact_Definition.md` — artifact terminology
- `katgpt-rs/.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md` — canonical research-note example
- `katgpt-rs/.plans/271_attention_matching_compaction.md` — canonical plan example
- `riir-chain/AGENTS.md` — repo-local context for the chain spin-off (workspace layout, `merkle_root` lesson, drift resolution, `develop` branch policy)
- `riir-chain/.plans/001_chain_spinoff.md` — chain crate migration record (riir-ai → riir-chain)
- `riir-chain/.plans/002_chaind_spinoff.md` — chaind crate migration record
- `riir-neuron-db/AGENTS.md` — repo-local context for the neuron-db spin-off (what this crate owns vs what stays in `riir-chain`, feature → chain feature mapping, `merkle_root` lesson still applies, `develop` branch policy)
- `riir-neuron-db/.plans/001_extraction_from_riir_chain.md` — shard crate migration record (riir-chain → riir-neuron-db)
- `katgpt-rs/.research/219_Topological_Neural_Operators_DEC_Inference.md` — DEC operators distillation (the parent note that shipped the Stokes substrate)
- `katgpt-rs/.research/271_MIT_6S184_Diffusion_Flow_Textbook_Vocabulary_Crosswalk.md` — diffusion/flow vocabulary crosswalk (also flags Fokker-Planck as a known gap, closed by Research 296)
- `katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md` — Stokes/Divergence/Hodge vocabulary crosswalk + GOAT verdict for the three wrapper primitives
- `katgpt-rs/.plans/251_dec_operators_cell_complex.md` — DEC operators (COMPLETE — ships `d`, `δ`, `Δ`, `hodge_decompose`, `DecFlowField`)
- `katgpt-rs/.plans/314_stokes_calculus_wrappers.md` — Stokes-calculus wrapper primitives (`belief_mass_divergence`, `boundary_flux_mass`, `line_integral`)

## TL;DR

This skill packages the katgpt-rs research workflow: **MANDATORY pre-flight: `read_file` all five READMEs (`katgpt-rs/README.md`, `riir-ai/README.md`, `riir-chain/README.md`, `riir-neuron-db/README.md`, `riir-armageddon/README.md`) AND `list_directory` all four `.research/` folders (`katgpt-rs/.research/`, `riir-ai/.research/`, `riir-chain/.research/`, `riir-neuron-db/.research/` — create the last two on first use) AND the four runtime/chain/neuron-db crate src trees (`riir-ai/crates/riir-engine/src/`, `riir-ai/crates/riir-games/src/`, `riir-chain/src/`, `riir-neuron-db/src/`) before any verdict** → read paper → classify (training? → riir-train, stop) → **distill + fuse** (find the transferable primitive, then **vocabulary-translate** paper terms to ≥2 codebase-equivalent terms each INCLUDING the standing latent-state vocabulary + the standing DEC/Stokes vocabulary, then grep BOTH layers — `.research/`+`.plans/` for intent AND `src/`+`crates/` for shipped code — across all five repos, using BOTH paper vocabulary AND codebase vocabulary, AND `list_directory` the **seven Super-GOAT factory modules** explicitly: `katgpt-rs/crates/katgpt-core/src/sense/` (HLA), `riir-ai/crates/riir-engine/src/latent_functor/`, `riir-ai/crates/riir-engine/src/hla/`, `riir-ai/crates/riir-engine/src/cgsp_runtime/`, `riir-neuron-db/src/` (shards/freeze/consolidation/AnyRAG/vibe/Merkle — **now in the standalone `riir-neuron-db/` repo, not `riir-chain/src/neuron_db/`**), `riir-chain/src/encoding/latcal*.rs` (LatCal — **in the standalone `riir-chain/` repo, not `riir-ai/crates/riir-chain/`**), `katgpt-rs/crates/katgpt-core/src/dec/` (Stokes/exterior-derivative/Hodge — maps any divergence/boundary/line-integral/Fokker-Planck/manifold-geometry paper) — for the 2–3 closest cousins to synthesize a novel combination) → **MANDATORY latent-space reframing before verdict** (re-cast the mechanism as a latent-to-latent op on HLA/functor/cgsp/neuron-shard/LatCal/DEC state; adapter routing / KV compression / speculative decode are GOAT-tier fallback framings, NOT the primary) → **novelty gate** (Super-GOAT? → open primitive + private riir-ai or riir-chain or riir-neuron-db guide depending on whether the selling point is game runtime, chain transport, or neuron-shard storage; else GOAT/Gain → plan only). **Zero grep hits ≠ novelty — try one more semantic angle before claiming "no prior art"** → implement behind feature flag → benchmark → promote GOAT or demote loser. Hard constraints: modelless-first, latent-to-latent with sigmoid (never softmax), freeze/thaw over fine-tuning, 5-repo commercial discipline (public engine / private runtime / private chain / private neuron-db / private training), raw scalars at the sync boundary, **fusion-first mindset** (the best Super-GOATs come from fusing papers across all five repos, not direct-mapping one). **Super-GOAT = private moat; never skip the riir-ai/riir-chain/riir-neuron-db guide. Never grep only katgpt-rs — riir-ai + riir-chain + riir-neuron-db are three-fifths of the corpus. Never grep only notes — code is half the prior art. Never grep only paper vocabulary — codebase vocabulary is the other half (DiPOD's "self-distillation when ELBO drifts" ships as `latent_functor/reestimation.rs` "coherence-driven re-estimation scheduler when coherence < tau_reest"; missed by paper-vocabulary grep on BOTH notes AND code, even though Research 123 + Plan 303 frame the mechanism under codebase vocabulary). Never default to adapter routing when a latent-functor/HLA/neuron-shard/LatCal reframing is available — that is the R269 failure mode. Five canonical failures: `evolve_hla` (no notes framing at all, Research 242 Super-GOAT overclaim), `latent_functor/reestimation.rs` (notes framing under different vocabulary, DiPOD false Super-GOAT claim), `> <former`/R269 (defaulted to adapter routing instead of stage-gated HLA subspace activation + LatCal projection profile), AC-Prefix G1/Plan 313 (premature riir-train deferral without checking freeze/thaw + raw/lora modelless correction — §3.5 protocol now mandatory before any "→ riir-train"), and DEC/Stokes/Research 296 (Generalized Stokes' theorem machinery ships as DEC operators `exterior_derivative`/`codifferential`/`hodge_decompose` but no note framed it in Stokes-theorem vocabulary, so `stokes|divergence theorem|fokker-planck` grep returned ZERO hits — DEC is now the 7th Super-GOAT factory module).**
