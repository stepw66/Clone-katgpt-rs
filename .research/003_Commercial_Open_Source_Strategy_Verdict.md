# Commercial Strategy — Engine & Platform Split

**Date:** 2026-06
**Status:** Active
**Purpose:** Guide for AI agents to reason about what is public vs internal when creating research, plans, and docs.

---

## The Boundary

Three repos. The split is absolute.

| Repo | License | Role |
|------|---------|------|
| `katgpt-rs` | MIT (public) | Engine — generic inference framework. Adoption funnel. |
| `riir-ai` | Private (internal) | **Game product** — freeze/thaw runtime, self-learn/adaptive NPCs, neuro-symbolic chain, game systems. The ship-focus repo. |
| `riir-train` | Private (internal) | **Training research** — 90+ LoRA/adapter training methods, training data, trained weights. The training know-how vault. |

**Rule: anything `riir-*` is internal. No exceptions. No per-crate deliberation.**

### Why three repos (not two)

riir-ai accumulated 90+ training-method plans with most being stability proofs, not quality proofs (audit 2026-06-14: MPNS failed GOAT, SPEFT/OFT had identical losses to LoRA because the Trainer ignored `tuning_method`). Training research was creating noise that blocked ship focus on the actual product: runtime NPC intelligence (freeze/thaw + self-learn + chain). Training know-how is still a moat — it just lives in `riir-train` now so riir-ai can ship the game.

riir-ai exposes a generic `TrainingProvider` trait; riir-train implements it. Same pattern as Issue 003 (chain spinoff) — trait bridge, zero dynamic dispatch.

---

## Why katgpt-rs Is Public (MIT)

`katgpt-rs` is the **generic inference framework** — DDTree, ConstraintPruner trait, bandit, pruners infra, speculative decode. These are the hooks that make game devs depend on the engine.

**Why public:**
- It's the adoption funnel. Developers build on the engine, then need the platform.
- The engine alone produces no useful game AI output — it's a runtime, not the intelligence.
- MIT attracts contributors and creates dependency without exposing know-how.
- No legal friction for enterprise adoption.

**"Ferrari, no gas":** `katgpt-rs` is the open Ferrari. Without the private platform, it runs but produces nothing competitive. The gas is inside `riir-ai`.

---

## What riir-ai Can Do (Capabilities — Not How)

`riir-ai` is the platform. Below is **what it can do**, not how it's built. The how stays internal.

### Freeze/Thaw Runtime (the product value)

- **Versioned adapter snapshots** — lock-free `ArcSwap` + `AtomicU64` swap of (A, B) matrices; readers never block, writers atomically bump version after both matrices are stored (`Release`/`Acquire` ordering guarantees a reader observing the new version sees both new matrices)
- **Runtime hot-swap** — file-watcher on `lora.bin` (BLAKE3-hashed) reloads NPC personality adapters at runtime, zero downtime
- **Dynamic adapter routing** based on game state and objective (100% win rate vs 0% static routing — the one GOAT-proved training-adjacent win; this is routing, not training)
- **Inference-time sparse attention** reducing KV cache 6× without quality loss
- **Per-NPC personality versioning** — each NPC can hold a different adapter snapshot version, so two NPCs of the same type can diverge behaviorally over time
- **Trained-adapter inference path** — `dispatch_lora_merge` fuses base + α·BA in one kernel; no separate adapter forward pass

### Self-Learn / Adaptive NPCs (the selling point)

- **Self-play**: learns game strategy without external oracle (2.4T ops/sec)
- **LEO all-goals**: NPCs learn every objective at once, no curriculum hand-holding
- **GRPO open-ended**: collapse-aware policy gradient for emergent NPC behavior
- **DeGRPO**: detect and recover from mode collapse mid-session
- **Trajectory folding**: 78% reduction in redundant self-play moves
- **Curiosity pulse**: entropy-driven information gathering drives NPC exploration without a reward oracle

This is what ships in the game. The capability that turns scripted NPCs into living ones — and it runs at runtime, not during offline training.

### Generic Training Interface (the seam)

riir-ai exposes a `TrainingProvider` trait. riir-train implements it with 90+ adapter-training methods (OFT, SPEFT, IA3, QLoRA, ManifoldE, BAKE, GPart, SSD-LoRA, MSA, Dendritic, and the rest). riir-ai never names a specific method — it consumes whatever the trait produces.

### Neuro-Symbolic Chain

- **AI state and wallet state co-located** in the same zero-copy data structure
- **Latent-encoded balances** — not plaintext integers, tamper-resistant by inspection
- **Split-key transactions** — neither party holds the combined key in transit
- **Self-healing chain** — detects anomalies and repairs automatically
- **Five-tier memory** with graceful degradation — engine never fails to boot
- **Cross-chain bridge** to Solana
- **Full DeFi economy**: gas, rent, slashing
- **9 GOAT proofs**: roundtrip fidelity, key security, pipeline throughput, tamper rejection

### Arena Proofs (Outcomes, Not Methods)

- Bomber: adaptive AI +475 vs baseline
- Go: 100% win vs Random 35%
- FFT Tactics: 99% win rate (game theory optimal)
- Frame-sampling: 939K decisions/sec
- Dynamic Pair Routing: 100% win rate vs 0% static routing (runtime adapter selection — the one training-adjacent GOAT that shipped)
- Adaptive reasoning: +177% quality on hard queries at ≤50% cost
- Browser NPC inference: 11.856 µs/call for a 6K-param brain (WASM SIMD 5.47-7.27× over scalar)

### Trained Weight Assets (in riir-train)

- LoRA adapters trained across Bomber, Go, FFT, Civ
- Cross-game universal concept neurons
- Per-zone weight snapshots
- Episode DB (game strategy history, edge cases)
- 6.9GB training data, 815MB models, 368MB output artifacts

**These live in `riir-train` (internal). riir-ai consumes them via the runtime freeze/thaw path — it never ships them in-game as raw files; it ships the snapshotted, versioned adapter the runtime hot-swaps.**

---

## Why riir-ai Is Hard to Replicate

| Pillar | Capability | Why Hard to Copy |
|--------|-----------|-----------------|
| **Freeze/thaw runtime** | Lock-free `ArcSwap` adapter snapshots, BLAKE3-hashed hot-swap, per-NPC personality versioning, fused `dispatch_lora_merge` inference | The runtime is small but every detail is tuned: `Release`/`Acquire` ordering, zero-copy `read_arc`, atomic version bump after both matrices stored. A re-implementation will race or stall. |
| **Self-learn / adaptive NPCs** | LEO all-goals, GRPO open-ended, DeGRPO collapse-aware, self-play, curiosity pulse — all runtime, no offline training round-trip | Turning scripted NPCs into living ones at runtime is the product. The collapse-detection and curiosity signals are tuned against real game sessions. |
| **Chain design** | Co-located AI+wallet, latent-encoded balances, split-key, self-healing, five-tier memory | Novel neuro-symbolic economic design. No incumbent co-locates AI weights with wallet state in zero-copy fixed-size structures. |
| **Training know-how (riir-train)** | 90+ adapter-training methods, consumer-GPU training, zero-forgetting retention, trained weight assets | Algorithms are published (arXiv). The implementations + configs + GPU kernels took years of GOAT-gated validation (and honest audit showed many need re-validation — see riir-ai issue 004). Still a moat, but the moat now lives in riir-train, not riir-ai. |
| **Network effects** | Live chain with real economic activity | A forked chain has no players, validators, or economy. Can't be copied. |

---

## Decision Rules for AI (When Creating Research / Plans / Docs)

Use these rules to decide what is safe for public `katgpt-rs/.research/` vs what must stay in `riir-ai` internal docs.

### Ask: Is this the WHAT or the HOW? And which repo?

| If it's about... | Goes in | Because |
|------------------|---------|---------|
| Inference engine mechanics (DDTree, ConstraintPruner trait, bandit theory, speculative decode) | `katgpt-rs/.research/` (public) | Generic framework — adoption value, no moat risk |
| An arXiv paper survey (what algorithm exists, why it's interesting) | `katgpt-rs/.research/` (public) | Literature review — tells WHAT exists, not HOW we use it |
| A capability description ("riir-ai hot-swaps NPC personalities at runtime") | `katgpt-rs/.research/` (public, if needed for context) | Outcome — doesn't reveal the method |
| **Training-method research, plans, benchmarks** (which LoRA variant, which optimizer, training loss curves) | `riir-train` internal | Training know-how vault — separate repo so riir-ai ships clean |
| **Trained weights, training data, training artifacts** (`.bin`, `data/`, `models/`) | `riir-train` internal (never shipped) | Data assets — GPU-hours to produce |
| Which specific training method produced a given adapter | `riir-train` internal | Naming the technique hands competitors the implementation direction |
| Exact hyperparameters, configs, or fusion recipes | `riir-train` internal | That's the fuel — the HOW that achieves the result |
| GPU kernel source for a specific training method | `riir-train` internal | Kernel implementations are the implementation detail that took years |
| **Freeze/thaw runtime internals** (snapshot ordering, hot-swap watcher, fused merge kernel) | `riir-ai` internal | Runtime IP — this is the ship-focus product |
| **Self-learn / adaptive internals** (LEO mixer α, GRPO collapse detector, curiosity pulse) | `riir-ai` internal | The selling point — keep private |
| Chain internals (encoding projections, key derivation, data layout, healing loop) | `riir-ai` internal | The implementation IS the IP |
| Game domain configs (character classes, zone behavior, economy rules, quest grammar) | `riir-ai` internal | Game design IP |
| Our GOAT proof configs, benchmark numbers beyond what's already public | `riir-ai` (runtime/game/chain) or `riir-train` (training) | Match the repo to the proof's subject |

### Rule of Thumb

**What = public. How = private. Training how = riir-train. Runtime how = riir-ai.**

- "NPCs hot-swap personalities at runtime via versioned snapshots" → public (capability)
- "The snapshot uses `ArcSwap` with `Release`/`Acquire` ordering and atomic version bump" → `riir-ai` private (runtime how)
- "We train adapters with [specific method] at [specific config]" → `riir-train` private (training how)
- "Balances are encoded as latent vectors" → public (concept)
- "The projection uses [specific learned values]" → `riir-train` private (implementation)

### When Unsure

Default to `riir-ai` internal. It is always safe to keep something private. It is never safe to un-leak something public.

---

## Related

| Doc | Connection |
|-----|-----------|
| 119 — Worms Armageddon Latent Space Game | Game product concept. Internal (`riir-ai/.research/119`). Moved out of public repo. |
