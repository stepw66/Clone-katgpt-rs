# Issue 001 — Deferred crate-promotion candidates

Status: **open** (tracking, not blocked)
Created: 2026-07-01
Related proposal: `proposals/001_quant_crate_promotion.md`

## Context

Promotion analysis of `katgpt-rs/src/` surfaced four candidates beyond the
quant family. The quant family is being promoted first (Proposal 001) because
it has the cleanest boundary. The three below are deferred — each has a
boundary or coupling issue that needs untangling before promotion is
worth the churn.

This issue tracks the deferred work so it isn't lost.

---

## Candidate A — `mux_latent/` (12 files)

**Status:** deferred — fuzzy MUX dependency boundary.

**What it is:** Inference-time context compression via vocabulary
superposition (distilled from LCLM, arXiv:2606.09659). Pipeline: input
tokens → MUX superposition encoder → latent slots → domain_latent
mid-layer injection. Modelless (uses existing MUX infrastructure).

```
mux_latent/
├── buffer.rs          # LatentContextBuffer, EvictionPolicy
├── config.rs          # MuxLatentConfig, CompressionRatio
├── context.rs         # CompressedContext, LatentSegment
├── encoder.rs         # MuxLatentEncoder
├── expand.rs          # segment expansion
├── inject.rs          # LatentPrefillAdapter, MixedPrefillSequence
├── mod.rs
├── prefill.rs         # forward_prefill_with_compression
├── spectral_lod.rs    # SpectralLOD
├── octree_bridge.rs   # gated mux_latent_wire
├── patcher.rs         # gated mux_latent_wire
└── wire.rs            # gated mux_latent_wire
```

**Why deferred:** `mux_latent` depends on existing MUX infrastructure that
already lives in `katgpt-core/src/mux/`. Promoting `mux_latent` alone would
create a circular or awkward dep (new crate → katgpt-core::mux, while
katgpt-core may want to re-export the new crate). The MUX substrate needs to
be split out *first* (or `mux_latent` needs to be folded into the existing
mux module rather than its own crate).

**Unblock criteria:**
- [ ] Audit `katgpt-core/src/mux/` vs `src/mux_latent/` — is the split
      "mux primitive" (core) vs "mux application" (mux_latent) clean?
- [ ] Decide: (a) promote mux primitive + mux_latent together into a
      `katgpt-mux` crate, or (b) fold mux_latent into katgpt-core::mux,
      or (c) keep as-is.
- [ ] If (a): write Proposal 003 with the full MUX closure.

---

## Candidate B — `proof_cert/` (7 files)

**Status:** deferred — cross-cuts chain/WASM runtime.

**What it is:** Proof certificate chain — verification/integrity substrate.
Emits and validates certificates for runtime artifacts.

```
proof_cert/
├── certificate.rs
├── chain.rs
├── macros.rs
├── mod.rs
├── serde_impls.rs
├── wasm_certificates.rs
├── wasm_proof_witness.rs
```

**Why deferred:** `proof_cert` cross-cuts the chain runtime (riir-chain has
its own proof concerns per its AGENTS.md) and the WASM runtime
(`wasm_certificates.rs`, `wasm_proof_witness.rs`). Promoting it into a
`katgpt-proof-cert` crate risks duplicating or conflicting with riir-chain's
proof envelope (`riir-neuron-db` owns `freeze.rs` / `FreezeGateReport`;
riir-chain owns `catchup/merkle.rs`). The boundary across the 5-repo
quintet needs design, not just a local lift.

**Unblock criteria:**
- [ ] Map the proof surface across the quintet: what does katgpt-rs's
      `proof_cert` prove that riir-chain's merkle proofs and
      riir-neuron-db's `FreezeGateReport` don't?
- [ ] Decide: is this a katgpt-rs-local crate, or does it belong in a
      different repo (riir-chain)?
- [ ] If katgpt-rs-local: confirm the WASM coupling can be feature-gated
      so the crate compiles without a WASM runtime.
- [ ] Write Proposal 004 with the cross-repo boundary decision.

---

## Candidate C — `dash_attn/` (16 files) — NOT deferred, separate proposal

**Status:** strong candidate, deserves own proposal (002).

Listed here only for cross-reference. `dash_attn` is the biggest single
module in `src/` (adaptive sparse hierarchical attention via α-entmax
routing, Plan 106/196 lineage, vortex_flow/msa_*/cache_prune feature
surface). It is *not* deferred — it's the natural Proposal 002 follow-up
to the quant promotion. Distinct from both `katgpt-core`'s base attention
primitives and `katgpt-attn-match` (KV compaction, Plan 271).

- [ ] Write Proposal 002 — `katgpt-dash-attn` crate promotion.

---

## Non-candidate — `src/sleep/` vs `katgpt-sleep` crate (clarification)

**Not a promotion candidate — they are different things. Do not merge.**

| | `src/sleep/` | `crates/katgpt-sleep/` |
|---|---|---|
| Paper | Plan 154 (GDN2 fast-weight consolidation) | arXiv:2504.13171 (Lin et al., Sleep-Time Query Anticipator) |
| Concern | offline recursive memory consolidation *at eviction* | offline query *anticipation*, wake-time consume() |
| Mechanism | N recurrent passes into GDN2 fast weights → evict KV | per-direction sleep-time compute → AnticipatedQuerySet (c' artifact) |
| Feature gate | `sleep_consolidation` (deps `lt2_looped`, `gdn2_attention`) | `sleep_time_anticipation` (forwards to `dep:katgpt-sleep`) |

They share the word "sleep" but are unrelated substrates. `src/sleep/` is
*not* stale source for the `katgpt-sleep` crate. No action.

---

## Priority order when revisiting

1. **Proposal 002 — `dash_attn`** (strongest standalone lift, clean boundary).
2. **Candidate A — `mux_latent`** (needs MUX substrate split decision first).
3. **Candidate B — `proof_cert`** (needs cross-repo quintet boundary design).

## TL;DR

`mux_latent` and `proof_cert` are real promotion candidates but each has a
coupling issue that makes the lift not worth it right now. `dash_attn` is
the next clean win (Proposal 002). `src/sleep/` is NOT a duplicate of the
`katgpt-sleep` crate — different papers, leave alone.
