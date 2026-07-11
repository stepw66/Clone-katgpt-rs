# Issue 001 — Deferred crate-promotion candidates

Status: **PARTIALLY SUPERSEDED** by Proposal 003 (`003_src_consolidation_master.md`).
Verified accurate 2026-07-04 by grep against Proposal 003's destination map +
phase list:
  - **Candidate C — `dash_attn/`**: ✅ genuinely tracked → Proposal 002 (done)
    and Proposal 003 Phase 2 (`katgpt-attn` crate).
  - **Candidate A — `mux_latent/`**: ❌ NOT in Proposal 003. Phase 6
    (`katgpt-speculative` absorption) lists `distill/{ilc,trd}`, `spechop`,
    `rt_turbo`, `precision_aware_draft`, `sparse_compose`,
    `spec_reconciliation` — **mux_latent is absent**. The only `mux*` entry
    anywhere in Proposal 003 is `mux_demux.rs` (Phase 10, `katgpt-core`),
    which is the MUX *primitive*, not the mux_latent *application*.
    The 3 unblock-criteria checkboxes below remain genuinely OPEN.
  - **Candidate B — `proof_cert/`**: ❌ NOT in Proposal 003. Zero mentions
    of `proof_cert` or `proof/` anywhere in the proposal (verified by grep).
    Phase 12 ("final sweep") does not reference it. The 4 unblock-criteria
    checkboxes below remain genuinely OPEN.
The earlier "all candidates scheduled in Proposal 003" claim was aspirational
and inaccurate. This issue remains the live tracker for Candidates A and B.
Created: 2026-07-01
Status corrected: 2026-07-04
Related proposal: `.proposals/003_src_consolidation_master.md` (covers Candidate C only)

## Context

Promotion analysis of `katgpt-rs/src/` surfaced four candidates beyond the
quant family. The quant family is being promoted first (Proposal 001) because
it has the cleanest boundary. The three below are deferred — each has a
boundary or coupling issue that needs untangling before promotion is
worth the churn.

This issue tracks the deferred work so it isn't lost.

---

## Candidate A — `mux_latent/` (12 files)

**Status:** deferred — fuzzy MUX dependency boundary. **(DEFERRAL PREMISE DISPROVED 2026-07-04 — see T1 audit result below.)**

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

**Why deferred (ORIGINAL rationale — now disproved at T1):** `mux_latent`
depends on existing MUX infrastructure that already lives in
`katgpt-core/src/mux/`. Promoting `mux_latent` alone would create a
circular or awkward dep (new crate → katgpt-core::mux, while katgpt-core
may want to re-export the new crate). The MUX substrate needs to be split
out *first* (or `mux_latent` needs to be folded into the existing mux
module rather than its own crate).

**⚠ AUDIT RESULT (T1, 2026-07-04): the deferral premise is FALSE.** Grep
of `src/mux_latent/` shows:
  - **Zero** `use katgpt_core::...` imports (the entire subsystem is
    std-only internally; cross-refs are all `crate::mux_latent::*`).
  - **Zero** references to `katgpt_core::mux` or `core::mux`.
  - Only one external consumer: `src/lib.rs:320` (`pub mod mux_latent;`
    under `#[cfg(feature = "mux_latent_context")]`).

  The name collision is misleading: `katgpt-core/src/mux/` holds
  speculative-decoding multi-token drafting primitives (`dd_tree`, `demux`,
  `span_pruner`, `top_k`, `bfs`, `bandit_width`, `freeze_thaw`) — i.e.
  "multiplexed draft tokens". `src/mux_latent/` is LCLM-style context
  compression via vocabulary superposition into latent slots. They are
  unrelated subsystems that happen to share the "mux" prefix; there is no
  primitive/application relationship and no dependency edge in either
  direction.

  **Consequence:** the circular-dep concern that drove the deferral does
  not exist. mux_latent is fully self-contained and could be lifted to its
  own crate without any katgpt-core entanglement. T2's option (b) "fold
  into katgpt-core::mux" is also wrong — they don't belong together.

**Unblock criteria:**
- [x] **T1 — Audit `katgpt-core/src/mux/` vs `src/mux_latent/` — is the
      split "mux primitive" (core) vs "mux application" (mux_latent) clean?**
      **ANSWER: there is no split — they are unrelated. mux_latent has
      zero katgpt-core deps. Deferral premise disproved.** (Audit
      performed 2026-07-04: grep `use katgpt_core` in `src/mux_latent/`
      returns zero hits; grep `katgpt_core::mux` returns zero hits; only
      consumer is `src/lib.rs:320`.)
- [ ] **T2 — Decide promotion target. Original options reframed by T1
      finding:**
      - ~~(a) promote mux primitive + mux_latent together into `katgpt-mux`~~ —
        **INVALID**: they are unrelated; bundling them would create a false
        semantic grouping.
      - ~~(b) fold mux_latent into `katgpt-core::mux`~~ — **INVALID**:
        they are different concerns; `katgpt-core::mux` is spec-decode
        multi-token drafting, mux_latent is LCLM context compression.
      - **(c) keep as-is in root `src/mux_latent/`** — viable but leaves
        ~104 KB of LCLM code in the root crate.
      - **(d) NEW — promote mux_latent to its own `katgpt-mux-latent`
        crate** (or fold into an existing attention/context crate if one
        emerges from Proposal 003 Phase 2 `katgpt-attn`). Cleanest option
        given T1: zero cross-crate deps to resolve.
      - **(e) NEW — fold into `katgpt-sleep` or a future `katgpt-context`
        crate** if the LCLM context-compression concern groups better with
        sleep-time anticipation than with attention. Decision needs a
        semantic call, not a dep-graph call (deps are trivial either way).
- [-] **T3 — If (a): write Proposal 003 with the full MUX closure.**
      **DEFERRED — moot.** T1 disproved the premise that made (a) an
      option. Whatever T2 decides, it won't be "write Proposal 003 with
      MUX closure" — Proposal 003 doesn't currently cover mux_latent
      (grep-verified, see status header) and shouldn't be retrofitted to.
      Replaced by: "if T2 picks (d) or (e), write the matching proposal."

---

## Candidate B — `proof_cert/` (7 files)

**Status:** deferred — cross-cuts chain/WASM runtime. **(DEFERRAL PREMISE LARGELY DISPROVED 2026-07-04 — see T1 audit result below.)**

**What it is:** Proof certificate chain — verification/integrity substrate.
Emits and validates certificates for runtime artifacts. Origin: Plan 145
("Hierarchical GOAT Proof Certificates") — standalone, serializable proof
certificates with dependency chains, topological verification, and blake3
checksum integrity.

```
proof_cert/
├── certificate.rs
├── chain.rs
├── macros.rs
├── mod.rs
├── serde_impls.rs
├── wasm_certificates.rs
└── wasm_proof_witness.rs   # gated: feature wasm_proof_witness
```

**Why deferred (ORIGINAL rationale — now largely disproved at T1):**
`proof_cert` cross-cuts the chain runtime (riir-chain has its own proof
concerns per its AGENTS.md) and the WASM runtime (`wasm_certificates.rs`,
`wasm_proof_witness.rs`). Promoting it into a `katgpt-proof-cert` crate
risks duplicating or conflicting with riir-chain's proof envelope
(`riir-neuron-db` owns `freeze.rs` / `FreezeGateReport`; riir-chain owns
`catchup/merkle.rs`). The boundary across the 5-repo quintet needs design,
not just a local lift.

**⚠ AUDIT RESULT (T1, 2026-07-04): the deferral premise is largely FALSE.**
Grep of `src/proof_cert/` shows:
  - **Zero** `use crate::...` imports and **zero** `use katgpt_core::...`
    imports — the subsystem is fully self-contained (only `super::` refs).
  - **Zero** runtime deps on a WASM engine. Despite the misleading names
    (`wasm_certificates.rs`, `wasm_proof_witness.rs`), grep for
    `wasmi|wasmtime|wasm_bindgen` returns **zero hits**. The "wasm" in the
    names refers to *certificates that describe wasm-validator outcomes*
    (e.g. `lora_wasm_delta: i32` as a metric value, `challenger: "wasm"`
    as a tag) — the module produces certificates *about* wasm validation,
    it does not *execute* wasm. The entire module compiles std-only.
  - Only one external consumer: `src/lib.rs:184` (`pub mod proof_cert;`
    under `#[cfg(feature = "proof_cert")]`). No internal katgpt-rs consumers
    at all.

  **Consequence:** the "cross-cuts WASM runtime" concern is wrong — there
  is no wasm runtime dep. The "cross-cuts chain runtime" concern is
  *semantic*, not technical: proof_cert (Plan 145 GOAT proof certificates)
  and riir-neuron-db's freeze/Merkle (shard integrity envelopes) serve
  different proof domains but might overlap conceptually. That's a
  design/semantic question, not a dep-graph blocker — proof_cert can be
  lifted cleanly on dep-graph grounds alone.

  **Cross-repo quintet proof surface (audit, 2026-07-04):**
    - `katgpt-rs/crates/katgpt-core/src/merkle.rs` + `content_store/merkle.rs`
      — content-addressed blob storage (local).
    - `katgpt-rs/crates/katgpt-core/src/mux/freeze_thaw.rs` — spec-decode
      freeze/thaw (unrelated to proof_cert).
    - `riir-neuron-db/src/freeze.rs` — `FreezeGateReport`, freeze/thaw
      integrity envelope (shard-level, committed).
    - `riir-neuron-db/src/merkle.rs` — generic BLAKE3 binary Merkle tree
      (shard-level proofs).
    - `riir-chain/src/catchup/merkle.rs` (per riir-chain AGENTS.md) — chain
      block commitment.

  None of these implement "GOAT gate proof certificates with dependency
  chains + topological verification" (proof_cert's actual concern). The
  overlap is at the word "proof" / "certificate", not at the algorithm.

**Unblock criteria:**
- [x] **T1 — Map the proof surface across the quintet: what does
      katgpt-rs's `proof_cert` prove that riir-chain's merkle proofs and
      riir-neuron-db's `FreezeGateReport` don't?**
      **ANSWER: proof_cert (Plan 145) implements hierarchical GOAT proof
      certificates — dependency chains + topological verification of GOAT
      gate outcomes (ProofProperty/ProofResult/ProofEvidence). The
      quintet's other proof surfaces are shard-integrity (freeze/Merkle)
      or chain-commitment (catchup/Merkle) — they do NOT implement GOAT
      gate dependency chains. The surfaces are disjoint at the algorithm
      level; the "overlap" was lexical (the word "proof"), not technical.**
      (Audit 2026-07-04: grep `use katgpt_core` and `use crate::` in
      `src/proof_cert/` returns zero; grep `wasmi|wasmtime|wasm_bindgen`
      returns zero; quintet proof-surface map above.)
- [ ] **T2 — Decide: is this a katgpt-rs-local crate, or does it belong
      in a different repo (riir-chain)?**
      Reframed by T1: dep-graph says lift is trivial in either repo.
      Semantic call: GOAT proof certificates are *engine-side* (the GOAT
      gate runs in katgpt-rs/riir-ai runtime), so proof_cert belongs with
      the engine, NOT with the chain. The chain commits *results*; it
      doesn't *produce* GOAT gate evidence. **Recommendation: katgpt-rs-local
      `katgpt-proof-cert` crate** (or fold into katgpt-core alongside
      merkle.rs). riir-chain would *consume* proof certificates if a
      future bridge commits them on-chain, but it doesn't own the algorithm.
- [x] **T3 — If katgpt-rs-local: confirm the WASM coupling can be
      feature-gated so the crate compiles without a WASM runtime.**
      **ANSWER: yes — it already is.** `wasm_proof_witness` is gated by
      `#[cfg(feature = "wasm_proof_witness")]` in mod.rs:6,11. The other
      "wasm" file (`wasm_certificates.rs`) has no wasm-runtime dep at all
      (T1 finding) — it compiles std-only. So the entire crate compiles
      with zero wasm deps under the default feature set; only the
      opt-in `wasm_proof_witness` feature adds the witness generator.
- [-] **T4 — Write Proposal 004 with the cross-repo boundary decision.**
      **DEFERRED — pending T2 decision.** T1 + T3 unblock the dep-graph
      side; T2 is a semantic call that the user should make (the audit's
      recommendation is katgpt-rs-local, but the call is not the agent's
      to make unilaterally — it affects the public crate surface). Note:
      Proposal 004 number is already taken (`004_adaptive_causal_calibration.md`);
      the actual number for this proposal would be 005+.

---

## Candidate C — `dash_attn/` (16 files) — NOT deferred, separate proposal

**Status:** strong candidate, deserves own proposal (002).

Listed here only for cross-reference. `dash_attn` is the biggest single
module in `src/` (adaptive sparse hierarchical attention via α-entmax
routing, Plan 106/196 lineage, vortex_flow/msa_*/cache_prune feature
surface). It is *not* deferred — it's the natural Proposal 002 follow-up
to the quant promotion. Distinct from both `katgpt-core`'s base attention
primitives and `katgpt-attn-match` (KV compaction, Plan 271).

- [x] Write Proposal 002 — `katgpt-dash-attn` crate promotion. See
      `.proposals/002_dash_attn_crate_promotion.md`.
      **Key nuance captured:** unlike the quant family, `dash_attn` is NOT
      a clean leaf — `forward.rs` + `tests.rs` are hard-coupled to
      `crate::transformer::ForwardContext` (which lives in root, not in
      the types-only `katgpt-transformer` crate). Proposal 002 splits the
      module: 13 primitive/routing files promote to the crate; 2
      transformer-integration files stay in root. Mirrors the
      `katgpt-attn-match` (Plan 271 / Issue 359) precedent.

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
