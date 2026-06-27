# Research 317: Google Pixel Frozen Multi-Token Prediction — PASS (Already Shipped)

> **Source:** [Accelerating Gemini Nano models on Pixel with frozen Multi-Token Prediction](https://research.google/blog/accelerating-gemini-nano-models-on-pixel-with-frozen-multi-token-prediction/) — Galgani, Homburger, Consul, Markwell, Kumar (Google Research), 2026; with Gemini team (Schuster, Ji, Korotkov, Jawahar — DeepMind)
> **Date:** 2026-06-27
> **Status:** Done — **PASS** (already shipped in Plans 055 + 117; not a new class of capability for this codebase)
> **Related Research:** 026 (Gemma 4 MTP — same architecture, distilled 2026-Q1), 078 (MTP Cluster Top-K + LoRA drafter)
> **Related Plans:** 055 (Gemma 4 MTP drafter — shared KV cross-attn), 117 (MTP LoRA drafter), 025 (reader/writer LoRA shared KV), 166 (FlashAR Path V D2F), 185 (`kv_share`), 089 (D2F tri-mode), 310 T1 (soft-reject leniency)
> **Classification:** Public

---

## TL;DR

**Verdict: PASS.** The Google blog is the production Pixel deployment of the Gemma 4 MTP architecture we already distilled (Research 026, 078) and shipped (Plans 055 + 117). Every headline mechanism — **frozen backbone + trained MTP head**, **zero-copy KV cross-attention**, **drafter prefill elimination**, **bit-for-bit identical output via rollback** — has shipped prior art in `katgpt-rs`. The blog's two future directions are also covered: **verification leniency** ships as Plan 310 T1 (`soft_reject_with_relax` + `RelaxationStrategy`), and **parallel decoding without auxiliary heads** is adjacent to Plan 166 (FlashAR Path V D2F) and Plan 089 (D2F tri-mode).

**No new file created in private repos. No plan created. No code changes.** This note exists only to prevent a future agent from re-distilling the same paper into a duplicate, weaker research note — the canonical failure mode §1.5 Q1 warns about.

---

## 1. Paper Core Findings (one-paragraph summary)

Google retrofits Multi-Token Prediction onto a **frozen** Gemini Nano v3 backbone by attaching a dense transformer MTP head to the final layers, training only the head's parameters. The MTP head **cross-attends zero-copy** to the main model's frozen KV cache (no separate KV, no drafter prefill), saving ~130MB/instance on Pixel 9/10. Production workloads (Notification Summaries, Proofread) gain ~2 extra accepted tokens/inference pass, with bit-for-bit identical output (incorrect drafts discarded during verification). Future directions: parallel decoding without auxiliary heads, branching speculation, and **verification leniency** (relaxing exact-match for edge use cases).

---

## 2. Distillation — already shipped (this is the novelty-gate negative result)

### Vocabulary translation (paper → codebase)

| Blog term | Codebase equivalent | Shipped in |
|-----------|---------------------|------------|
| "frozen backbone" | frozen `TransformerWeights` (target) + LoRA-trained drafter | Plan 117 |
| "MTP head" / "drafter" | `DrafterLoraWeights` (6 LoRA adapters: q/k/v/o/mlp1/mlp2) | `src/speculative/drafter_lora.rs` |
| "zero-copy KV cross-attention" | cross-attention to target's prefill-ed KV cache, gated by `mtp_shared_kv_prompt_threshold` | Plan 055 T12-T14 |
| "eliminates drafter prefill latency" | same — drafter reads target's pre-computed KV instead of re-prefilling | Plan 055 T13 |
| "~130MB / instance savings" | Q-K=V projection sharing + cross-KV reuse | Plan 185 (`kv_share`), Plan 025 |
| "bit-for-bit identical output" | LeviathanVerifier + DDTree verification rollback | `src/speculative/verifier.rs` |
| "trained head loaded at runtime" | `drafter_lora_path: Option<PathBuf>` in `InferenceOverrides` | Plan 117 T6 |
| "BLAKE3-committed drafter artifact" | `DRAFTER_LORA_MAGIC` + blake3 checksum in `save_drafter_lora` / `load_drafter_lora` | Plan 117 T7 |
| "out-of-the-box speedup, no per-task fine-tune" | hot-load via `with_drafter_lora()` / `set_drafter_lora()` | Plan 117 T4-T5 |

### Three-layer prior-art check (§1.5 Q1 protocol)

- **Notes layer:** Research 026 (Gemma 4 MTP — exact same architecture, distilled 2026-Q1) + Research 078 (Cluster Top-K + LoRA drafter) — both grep-hit on `MTP`, `cross-attend`, `shared KV`, `drafter LoRA`.
- **Code layer:** `src/speculative/drafter_lora.rs` ships `DrafterLoraWeights`, `forward_drafter_with_lora`, `train_drafter_lora`, `save/load_drafter_lora` with BLAKE3. `Config.mtp_shared_kv_prompt_threshold` ships the cross-KV gate. `belief_drafter.rs` already references "frozen-pretrained drafter" semantics (Plan 306 depth-invariance audit).
- **Vocabulary translation layer:** paper-vocabulary grep (`frozen backbone`, `zero-copy KV`, `cross-attend`) hits on all three layers via the table above. No miss.

**Verdict on Q1 (no prior art?): NO.** The mechanism is comprehensively shipped. This is not a candidate for Super-GOAT, GOAT, or Gain.

### Future directions — also covered

| Blog future direction | Our prior art |
|-----------------------|---------------|
| "verification leniency: relaxing the strict exact token match" | **Plan 310 T1** — `soft_reject_with_relax` + `RelaxationStrategy` trait + `SoftRejectConfig { tau_low, tau_high }` (`src/pruners/soft_reject.rs`). Ships a 3-way verdict (Accept / SoftReject-band / Reject) with pluggable relaxer. Benchmarks: `bench_310_sigmoid_graded_reject_goat.rs`, `bench_310_t31_false_reject_rate_goat.rs`. |
| "parallel decoding and paradigms without auxiliary heads" | Plan 166 FlashAR Consensus Path V (D2F block draft, no aux head), Plan 089 D2fDrafterVerifier (tri-mode), Research 034/055 (D2F / Nemotron tri-mode diffusion-AR). |
| "branching possibilities in parallel" | DDTree (`src/speculative/dd_tree.rs`) — already ships multi-branch speculation with verification. |

---

## 3. Verdict

**PASS.**

**One-line reasoning:** every headline mechanism (frozen backbone + trained MTP head, zero-copy KV cross-attention, drafter prefill elimination, bit-for-bit rollback, hot-loadable drafter) and both named future directions (verification leniency, head-free parallel decoding) already ship in `katgpt-rs` via Plans 055 + 117 + 025 + 185 + 166 + 089 + 310. The blog is the production Pixel deployment of the architecture we distilled in Research 026/078. No new class of capability, no force multiplier, no selling point we don't already have.

### What did NOT transfer / is NOT new

- **Mobile-only framing:** the blog's "130MB / instance savings" and "Pixel 9 energy budget" constraints are not our problem (we are not shipping on Pixel). Our analog constraint is the plasma/hot latency budget at 20Hz × 1000 NPCs, which the existing MTP plumbing already serves.
- **Frozen-backbone-as-novelty:** in our modelless-first mandate, *everything* applied at inference is frozen by definition (constraint #1, #3 in `katgpt-rs/AGENTS.md`). The blog frames freezing as a retrofit optimization; we treat it as the only allowed runtime weight mutation. Same mechanism, different framing — no delta.

### One genuine delta worth noting (not actionable, no plan)

The blog's claim that **MTP drafters beat standalone fine-tuned drafters on instruction-following tasks** (50%+ speedup, 55% acceptance improvement on smart replies) is an empirical measurement we have not reproduced at production scale. Our Plan 117 GOAT gate (T9) measured +12% acceptance at micro scale (baseline=0.140 → trained=0.157). The Google number is the ceiling, not the floor, for well-trained heads on real workloads. This is informational — there is no code change to make, just a calibration data point that our micro-benchmark is conservative.

---

## Cross-references

- Research 026 — Gemma 4 MTP distillation (the architectural parent)
- Research 078 — MTP Cluster Top-K + LoRA drafter (the implementation parent)
- Plan 055 — shared KV cross-attention implementation (T12-T14)
- Plan 117 — `DrafterLoraWeights` + `forward_drafter_with_lora` + BLAKE3 artifact
- Plan 025 — reader/writer LoRA shared KV (the zero-copy pattern)
- Plan 185 — `kv_share` Q-K=V projection halving
- Plan 166 — FlashAR Path V (D2F, the head-free parallel decoding adjacent)
- Plan 310 T1 — `soft_reject_with_relax` + `RelaxationStrategy` (the verification leniency primitive)
- `katgpt-rs/.research/003_Commercial_Open_Source_Strategy_Verdict.md` — verdict tiers

## TL;DR

PASS. The Google blog "Accelerating Gemini Nano on Pixel with Frozen MTP" is the production Pixel deployment of the architecture we already distilled (Research 026/078) and shipped (Plans 055 + 117 + 025 + 185). Frozen backbone + trained MTP head = `DrafterLoraWeights`; zero-copy KV cross-attention = `mtp_shared_kv_prompt_threshold`; bit-for-bit rollback = LeviathanVerifier; verification leniency (blog's future direction) = Plan 310 T1 `soft_reject_with_relax`. No new capability, no plan, no code change. This note exists solely to prevent a future agent from re-distilling the same paper into a duplicate, weaker research note (the §1.5 Q1 prior-art check protocol's documented failure mode).
