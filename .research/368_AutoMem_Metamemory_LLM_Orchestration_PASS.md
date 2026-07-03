# Research 368: AutoMem — Automated Learning of Memory as a Cognitive Skill (PASS)

> **Source:** [AutoMem: Automated Learning of Memory as a Cognitive Skill](https://arxiv.org/pdf/2607.01224) — Wu, Zhu, Zhang, Wang, Yeung-Levy (Stanford), 2026-07
> **Date:** 2026-07-03
> **Status:** Done
> **Related Research:** 133 (FluxMem — canonical NO-GAIN precedent), 060 (MeMo), 024 (δ-Mem); riir-ai 169 (AgentMemBench PASS — the closest cousin), 147 (Engram NPC Guide), 007 (Four-Tier Memory)
> **Classification:** Public

---

## TL;DR

**Verdict: ⚠️ PASS — LLM-orchestration paper, no modelless inference primitive to distill.** AutoMem promotes file-system operations to first-class memory actions and automates two outer loops: (1) meta-LLM reviews full episode traces (10⁴–10⁵ steps) and rewrites agent code/prompts/schema; (2) meta-LLM curates training data and trains a LoRA memory specialist. Both the inner-loop agent (2 LLM calls/step for LOG/PLAN routines) and the outer-loop revision generation require LLM calls — same NO-GAIN failure mode as FluxMem (R133) and AgentMemBench (riir-ai R169). Every architecture class it instantiates already ships in the quintet under modelless vocabulary. The Loop #2 training axis is a clean → riir-train redirect. The one characterizable bias (consult-before-write discipline, write/search ratio drops 54–72%) is modelless-unblockable via a deterministic gate but too thin for a feature flag — it is a design-validation signal for existing decisions (consolidation pipeline, AnyRAG escalation policy), same as R169's F5–F9. **No plan, no feature gate, no private guide. Research only.**

**Distillation insight (modelless, captured for future grep):** the paper's strongest conceptual contribution — that structural scaffold revision (Loop #1, no weight changes) delivers 80–90% of the 2–4× gain, with the LoRA training (Loop #2) adding only 9–18% — empirically validates the freeze/thaw-over-fine-tuning mandate. This is a validation signal, not a new primitive.

---

## 1. Paper Core Findings (verified by full read)

### 1.1 The metamemory framing

Memory management is a separable, trainable skill ("metamemory" — Flavell 1979, Nelson 1990). The agent decides what to encode, when to retrieve, and how to organize, via file-system operations (read/write/search/append/create) promoted to first-class actions alongside task actions. Both axes — the supporting structure (prompts, schema, action vocabulary) and the model's proficiency — resist manual optimization because memory mistakes hide for thousands of steps before surfacing.

### 1.2 The two outer loops

| Loop | What it does | Cost | Gain share |
|------|-------------|------|-----------|
| **#1 Scaffold optimization** | Meta-LLM (Claude Opus 4.6) reads complete episode traces + agent code, diagnoses memory failure patterns, rewrites code/prompts/file schema. Revision gated on progression improvement on fixed seeds. | Offline, ~2–5 iterations, each reviews a full 10⁴–10⁵ step trace | **~80–90% of total gain** (Crafter 25→47%, MiniHack 7.5→27.5%, NetHack 0.42→1.57%) |
| **#2 Memory proficiency training** | Meta-LLM (Claude Opus 4.7) curates supervised training data from the agent's own good memory decisions, picks LoRA configs, trains a dedicated memory specialist. Two-model deployment: LoRA specialist handles LOG + memory-consult in PLAN; frozen base commits world actions. | Offline, LoRA finetune (rank 128–256) | **~9–18% marginal** on top of optimized scaffold |

### 1.3 The behavioral metrics (Figure 4 — the modelless signal)

Scaffold optimization produces measurable behavioral shifts:
- Unproductive action rate (stuck + oscillation): **−32 to −65%**
- Repeat WRITE rate: **−68 to −83%**
- Empty SEARCH rate (searches returning nothing): **−13 to −50%**
- Input tokens/step: **−3 to −30%** (leaner memory compresses context)

### 1.4 The consult-before-write discipline (Table 2)

Training the memory specialist internalizes a "consult-before-write" pattern — the LOG-phase write/search ratio drops 54–72% across all three environments. The trained specialist searches existing files before appending new content.

### 1.5 Concrete scaffold revisions (Appendix B)

The meta-LLM's revisions are domain-specific:
- **NetHack v1:** add `<|UPSERT_MAP|>` coordinate-keyed dedup operation (replaces append-only map that accumulated duplicates); auto-trim action log; pre-fill strategy.txt.
- **NetHack v2:** auto-maintain `current_status.txt` + `inventory.txt` from observation; descend-staircase directive.
- **Crafter v1–v5:** pre-load crafting tree + survival rules; achievement checklist; inventory-change logging; craft-feasibility verification; sleep-next-to-monster block.

---

## 2. Distillation — Why NO-GAIN

### 2.1 Three-layer NO-GAIN failure (mirrors R133 FluxMem, riir-ai R169 AgentMemBench)

| Failure mode (R133 §"Why No Gain") | AutoMem? | Detail |
|----|----|----|
| LLM-agent level, not model level | ✅ | Inner-loop agent makes 2 LLM calls/step (LOG + PLAN). Outer loops call Claude Opus 4.6/4.7. Our per-NPC tick budget is 50µs at 20Hz — one LLM call is 10–100ms. |
| LLM-call heavy (violates optimization.md) | ✅ | Inner loop: ~2 calls/step = 40 calls/sec/NPC. At 1000 NPCs: 40,000 LLM calls/sec. Orders of magnitude over budget. |
| Not LoRA-independent | ✅ | Loop #2 is LoRA finetune (rank 128–256). → riir-train. |
| Conceptual overlap without perf gain | ✅ | Every concept maps to a shipped modelless equivalent (§2.2 below). |
| Paper's own limitations | ✅ | Episodic memory only (no cross-episode persistence); single-environment scaffolds; game environments only. |

### 2.2 Architecture-class mapping (every class already ships, modelless)

| AutoMem concept | Shipped equivalent (modelless) | Notes |
|-----------------|-------------------------------|-------|
| File-system memory substrate | `NeuronShard` Pod + `ShardIndex` lock-free papaya (`riir-neuron-db/src/shard.rs`, `index.rs`) | Fixed-layout `#[repr(C)]` Pod, key-indexed slots, zero-copy mmap. The "file-system" is the shard; the "files" are the `style_weights[64]` + dendritic branch slots. |
| Memory operations as first-class actions | Per-NPC entity cognition tick (sense → project → act), `riir-engine/src/entity_cognition/` | Shard read/write/retrieve are first-class per-tick ops already. |
| `<|UPSERT_MAP|>` coordinate-keyed dedup | `ShardIndex` papaya upsert (key → shard overwrite) | The lock-free upsert IS coordinate-keyed dedup. Already shipped. |
| Auto-synced inventory/status files | HLA raw→latent scalar bridge (valence/arousal/desperation/calm/fear auto-derived from observation) | The 5 synced scalars ARE the auto-synced status. Full embedding stays local. |
| Pre-populated strategy reference | `ZoneGeometryPod` + `LatCalEggshell` + frozen direction vectors | The frozen zone scaffold IS the pre-populated strategy reference. |
| Two-phase LOG/PLAN decomposition | Per-NPC tick: sense phase (HLA projection) → act phase (functor + action commit) | The entity cognition stack already decomposes into observe-then-act. |
| Meta-LLM scaffold revision (Loop #1) | Raven/δ-Mem consolidation + MAPE-K self-healing (`riir-neuron-db/src/consolidation.rs`, `mape_k.rs`) | Offline review of traces → produce new frozen shard. **Quality parity unproven** — see §2.3. |
| Memory specialist LoRA (Loop #2) | → riir-train | Training axis. §3.5 modelless-unblock check below. |
| Consult-before-write discipline | AnyRAG escalation policy (`riir-neuron-db/src/gateway.rs`) — escalate only when local retrieval fails | Partially covered: AnyRAG escalates after local miss; consolidation pipeline only writes wake events (observed events). **KG triple emission does NOT enforce consult-before-write** — but it's a rare offline op with minimal quality impact. |

### 2.3 The quality-parity gap (why "already ships" is not enough — §3.6)

AutoMem's Loop #1 produces domain-specific semantic revisions (e.g., "pre-load the Crafter crafting tree", "add a `<|UPSERT_MAP|>` operation for NetHack coordinate dedup"). These require understanding the game domain — a meta-LLM (Claude Opus 4.6) generates them from trace analysis.

Our Raven/δ-Mem consolidation is a **deterministic** algorithm (averages wake events, applies weight deltas). It cannot generate "add a UPSERT_MAP operation for this game's coordinate system" — that requires semantic code generation.

So while the *architectural analog* exists (offline review → revised frozen artifact), **quality parity is unproven**. A deterministic consolidation algorithm does not produce the same kind of domain-specific structural revisions a meta-LLM produces. Per §3.6, this means:
- **Architectural coverage:** ✅ (consolidation ships)
- **Latency/resource parity:** ✅ (modelless, sub-µs)
- **Quality parity:** ❓ **unproven** — would require a PoC to claim

This note does NOT claim quality parity. It claims only that the *architecture class* is covered. The quality gap (deterministic consolidation vs meta-LLM revision generation) is a genuine limitation, not a false-PASS.

### 2.4 §3.5 Modelless-unblock check on Loop #2 (LoRA memory specialist)

The LoRA training produces a "consult-before-write" discipline (write/search ratio drops 54–72%). This is a **systematic, characterizable bias** — exactly the case §3.5 targets.

| Path | Check | Result |
|------|-------|--------|
| 1. Freeze/thaw snapshot correction | Can a frozen snapshot state fix the consult-before-write gap? | **No** — there's no systematic weight bias to correct; the discipline is a behavioral pattern, not a weight configuration. |
| 2. Raw/lora reader-writer hot-swap (deterministic) | Can a deterministically-constructed reader/writer LoRA enforce consult-before-write? | **Partially** — a deterministic gate that defers WRITE if no recent SEARCH has occurred captures the characterizable subset. But the broader "memory proficiency" gain (search precision, write quality) is not characterizable. |
| 3. Latent-space correction | Can a sigmoid gate on recent read activity modulate write probability? | **Yes for the consult-before-write subset** — but this is a thin gate (one EMA + one sigmoid), not a full memory specialist. |

**Verdict:** the consult-before-write discipline is modelless-unblockable via path 3 (a deterministic gate). But this captures only the characterizable subset of Loop #2's gain. The broader memory proficiency gain (9–18% on top of scaffold) requires empirical curation of good traces = learning. **Loop #2 → riir-train** for the full gain.

---

## 3. Verdict

**⚠️ PASS — LLM-orchestration paper. No modelless inference primitive to distill.**

### Reasoning

1. **Inner-loop agent requires 2 LLM calls/step** (LOG + PLAN routines) — fundamentally incompatible with the 20Hz NPC tick budget (50µs/tick). Same failure mode as FluxMem (R133) and AgentMemBench (riir-ai R169).

2. **Outer-loop scaffold optimization (Loop #1)** is offline meta-LLM work. While the offline nature is compatible with our sleep-time consolidation tier, the revision *generation* requires semantic understanding (domain-specific code/schema rewrites) that a deterministic consolidation algorithm cannot produce. The architecture class ships (Raven/δ-Mem), but quality parity is unproven (§2.3).

3. **Memory specialist training (Loop #2)** is LoRA finetune → **riir-train**. The §3.5 modelless-unblock check (§2.4) shows the consult-before-write subset is modelless-unblockable via a deterministic gate, but the broader proficiency gain requires empirical curation.

4. **Every architecture class already ships** under modelless vocabulary (§2.2). The paper adds no new class of capability.

5. **The one characterizable modelless distillation** — the consult-before-write gate — is too thin for a feature flag. It's a single EMA + sigmoid gate that partially overlaps with the existing AnyRAG escalation policy (escalate-after-local-miss) and the consolidation pipeline (write-only-observed-events). The marginal quality impact on our runtime is minimal because the main write paths already enforce consult-before-write implicitly.

### What IS captured (design-validation signals, not primitives)

The paper's empirical findings validate existing design decisions:

| Paper finding | Validates | Where |
|---------------|-----------|-------|
| Scaffold revision (no weight changes) gives 80–90% of gain | Freeze/thaw-over-fine-tuning mandate (global AGENTS.md) | The modelless-first thesis |
| Localized maintenance > global reorganization | Raven region-targeted consolidation, MAPE-K region healing | `riir-neuron-db/src/consolidation.rs`, `mape_k.rs` (also validated by riir-ai R169 F5) |
| Raw evidence > compressed summaries | Raw-vs-latent sync rule; `NeuronShard` ships raw Pod | Global AGENTS.md latent-vs-raw rules |
| Conservative consolidation cadence | Sleep-cycle consolidation cadence; `FreezeGateReport` two-sided contract | `riir-neuron-db/src/consolidation.rs` (also validated by riir-ai R169 F9) |
| Memory mistakes have delayed consequences (step 50 → step 800) | Trajectory-level offline consolidation (Raven reviews wake-event sequences, not per-tick metrics) | `riir-neuron-db/src/consolidation.rs` |

### → riir-train redirect (Loop #2 only)

AutoMem's Loop #2 (memory specialist LoRA training) is a training method. If riir-train ever needs a "train a dedicated memory-management adapter from the agent's own good traces" recipe, this paper is the reference. The §3.5 check (§2.4) documents why the modelless paths capture only the characterizable subset.

---

## Cross-references

- **Canonical NO-GAIN precedents:** `katgpt-rs/.research/133_FluxMem_Connectivity_Evolving_Memory.md`, `riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md` — same LLM-orchestration-vs-modelless-inference failure class. Consult before re-evaluating any agent-memory paper whose mechanism lives at the orchestration layer.
- **Architecture validation targets:** `riir-ai/.research/007_Four_Tier_Memory_Architecture.md`, `riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md`.
- **Shipped consolidation:** `riir-neuron-db/src/consolidation.rs` (Raven/δ-Mem), `riir-neuron-db/src/mape_k.rs` (self-healing loop), `riir-neuron-db/src/gateway.rs` (AnyRAG escalation).

## Re-evaluation guard

This note exists to prevent a future agent from re-running the full mandatory pre-flight + 5-repo fusion search on the same paper. If you arrived here from a grep, the verdict is PASS; do not re-distill unless the paper has a new version with a novel mechanism (e.g., a modelless scaffold-revision algorithm that replaces the meta-LLM).
