# Benchmark 331: BabelCodec — Readability-Relaxed Semantic Codec — GOAT **FAILED** (G2)

**Date:** 2026-06-26
**Plan:** [331_babel_codec_readability_relaxed_semantic_codec.md](../.plans/331_babel_codec_readability_relaxed_semantic_codec.md)
**Research:** [312_BabelTele_Readability_Relaxed_Semantic_Codec.md](../.research/312_BabelTele_Readability_Relaxed_Semantic_Codec.md)
**Feature gate:** `babel_codec` (katgpt-core) — **opt-in, stays opt-in**
**Status:** ❌ **GOAT FAILED (G2)** — `FixedRuleTextCodec` achieves 1.14× on structured data, missing the ≥ 2× bar that killed CompressionDrafter twice. **Honest negative result.** G1/G3/G4/G5 all pass.

---

## TL;DR

The deterministic BT-P8 fixed-rule codec (the modelless subset of BabelTele) is **correct, fast, deterministic, and zero-alloc on the latent path** — but it does **NOT** achieve 2× compression on realistic structured data (KG triples / configs / quest records). The verbose canonical form is already too terse for the symbolic rewrite to find 2× savings. The paper's headline 3.6× requires **LLM-prompted omnilingual lexical selection**, which is out of scope for a modelless primitive (→ riir-train if pursued).

The primitive ships as opt-in code: `FixedRuleTextCodec` is a correct, well-tested, bijective codec useful for any future consumer that needs deterministic BT-P8 ↔ verbose round-tripping with BLAKE3 commitment. But it does **not** promote to default — G2 failed, matching the CompressionDrafter precedent (Plan 285/287).

---

## GOAT Gate Results

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** (fidelity) | `decompress(compress(x)) ≡ x` 100% bit-identical | **1500/1500** bit-identical (103211 in → 90875 out bytes) | ✅ **PASS** |
| **G2** (compression) | ≥ 2× byte reduction (ratio ≤ 0.5) | **1.14×** (ratio 0.8805) — 103211 → 90875 bytes | ❌ **FAIL** (the make-or-break) |
| **G3** (latency) | < 200 ns/latent msg, < 2 µs/256-byte text chunk | latent D=8 K=4: **125 ns** (< 200); text: **1927 ns/256B** (< 2000) on 559B entry | ✅ **PASS** |
| **G4** (no-regression + alloc-free) | `--all-features` clean + latent zero-alloc | latent: **0 allocs/1000** (T3.2 met); text: 10/call (accepted — not gated); babel_codec 45/45 tests clean | ✅ **PASS** |
| **G5** (determinism) | same input → identical bytes + BLAKE3 across runs | **0 byte mismatches, 0 commitment mismatches** / 1500 entries; 1377/1500 distinct commitments | ✅ **PASS** |

**Run command:**
```bash
cargo test -p katgpt-core --features babel_codec --test bench_331_babel_codec_goat --release -- --nocapture
```

---

## Why G2 failed (the honest root cause)

The verbose canonical form (`entity has key = value`, `Config[target]: key = value(unit)`, `if cond then act`, `A then B then C`) is **already a terse structured representation**. The BT-P8 symbolic form (`@entity(key=value)`, `Config[target]:key=value(unit)`, `?[cond]=>[act]`, `A>B>C`) saves only the **structural keywords** (`has`, `then`, `if`/`then`, spaces around `=`). On a corpus of structured records, that's a ~12% byte reduction (1.14×), not 2×.

### Worked example (from the doc-example test)

```
VERBOSE (canonical):    Config[negotiation]: patience_required = 10(turns)
BT-P8 (compressed):     Config[negotiation]:patience_required=10(turns)
```
Savings: 2 spaces around `=` = 2 bytes out of ~52 bytes ≈ 4%. The structural keywords ARE the compression opportunity, and there aren't enough of them in terse structured data.

### What WOULD achieve 2×+ (out of scope)

The paper's 3.6× comes from compressing **natural-language prose** ("The appellant Wang Nianfang filed against Hubei Longan Real Estate because...") into symbolic form. Two problems:

1. **A deterministic fixed-rule codec can only compress what it can parse.** Arbitrary prose does NOT match any BT-P8 schema element → it falls through to `Raw` (verbatim) → 1× ratio, no compression. The codec's compression win is bounded by how verbose the PARSEABLE input is, and parseable structured inputs are already dense.

2. **The 3.6× requires LLM-prompted omnilingual lexical selection** (paper principle P1) — picking the highest-info-density lexical unit across languages/scripts. This is NOT a deterministic function; it requires an instruction-tuned LLM forward pass. That's riir-train territory, explicitly out of scope per the plan ("Learned / LLM-prompted BabelCodec → riir-train if pursued").

### Why this is the SAME failure mode as CompressionDrafter (Plan 285/287)

CompressionDrafter failed because quest-grammar strings are too short and too few for byte-level LZ4 to find matches. BabelCodec's fixed-rule codec fails for the mirror-image reason: structured records are too **dense** for symbolic rewrite to find 2× savings. Both codecs hit the same wall — **Hot-tier quest/KG text is structurally compact, and neither byte-level nor rule-level compression can find 2× on already-compact data.** The 2×+ opportunity lives in verbose natural-language prose, which neither codec can handle deterministically.

---

## Corpus substitution (honest disclosure)

The plan references the "real Seal 17k corpus" from Plan 285/287. **That corpus does not exist as a committed fixture in this repo.** Verification:

- `grep -r "seal_17k|seal_corpus|Seal 17" katgpt-rs/crates/katgpt-core/` → **zero hits**.
- [`.benchmarks/285_compression_drafter_goat.md`](285_compression_drafter_goat.md) used 8 hardcoded quest-grammar strings + 100 numbered contexts (`"quest 0"`..=`"quest 99"`), not a 17k corpus.

Per the plan brief ("synthesize ≥1000 entries and document this substitution honestly"), this bench synthesizes **1500 representative entries** from a fixed-seed LCG (deterministic, reproducible):

| Category | Count | Shape |
|----------|-------|-------|
| KG-triple entity-attribute pairs | 500 | `{entity} has {key} = {value}` |
| Config strings | 500 | `Config[{target}]: {key} = {value}({unit})` |
| Multi-line quest/dialog records | 500 | 3-5 lines mixing Section / Attribute / Config / Conditional / Comparison |

These categories mirror what Seal Online dialog/quest/KG data would look like in the verbose canonical form the codec round-trips. The compression numbers are honest measurements on this synthetic corpus — they are **NOT** the paper's LLM-prompted 3.6×.

---

## What ships (unchanged by the G2 failure)

1. **`babel_codec` module in katgpt-core** — all four pieces land as opt-in:
   - `BabelCodec` trait + `BabelPair` (`mod.rs`)
   - `FixedRuleTextCodec` — deterministic BT-P8 codec, bijective on its schema, 22 unit tests (`fixed_rule.rs`)
   - `SigmoidLatentCodec<D, K>` — generic latent projection codec (API-uniformity facade over `DensityBudget`), 11 unit tests (`sigmoid_latent.rs`)
   - `BabelCommitment` — BLAKE3 `[u8; 32]` newtype, 12 unit tests (`commitment.rs`)
   - **45/45 unit tests pass** under both `--features babel_codec` and `--all-features`.
2. **GOAT bench** (`tests/bench_331_babel_codec_goat.rs`) — G1/G3/G4/G5 pass, G2 fails honestly.
3. **Feature stays opt-in** (`babel_codec = []`), NOT in `default`. No promotion.

---

## What the G2 failure rules out

- **No default promotion.** The primitive is correct but does not beat the CompressionDrafter bar on the same gate. Stays opt-in.
- **Issue #002 (deterministic BT-P8 → LatCal chain commitment) is unblocked on the codec side but blocked on the value side.** The codec is correct (G1) and deterministic (G5), so a LatCal commitment bridge COULD be built — but with only 1.14× byte savings, the commitment-gas overhead would likely exceed the byte savings (the BG4 gate in issue #002 predicts exactly this). Issue #002 should be closed as moot unless a learned codec (riir-train) raises the ratio above 2×.
- **No riir-ai integration.** NPC dialog memory / npc_comms text channel / Engram text-side compressor fusion is deferred — there's no compression win to fuse with.

---

## Latency + alloc details (the parts that DID pass)

### G3 latency (PASS)
- **Text codec** (`FixedRuleTextCodec::compress_str`): median **417 ns** on short (~40-byte) entries; **4208 ns** on a 559-byte entry = **1927 ns/256B** (under the 2000 ns/256B budget). The text codec parses + emits in a single pass with no float math.
- **Latent codec** (`SigmoidLatentCodec::<8, 4>::compress`): median **125 ns** (under the 200 ns budget). Dot-product × 8 directions + sigmoid × 8 + top-4 selection-sort.

### G4 alloc-free (PASS for latent, honestly NOT for text)
- **Latent codec** (`compress`): **0 allocs/1000 calls** after warmup. Scratch (`scratch_scores`, `scratch_taken`) is pre-sized at construction; the output `CompressedLatent<K>` is a stack `Copy` struct. **T3.2 zero-alloc requirement met.**
- **Text codec** (`compress_into`): **10 allocs/call** (informational, NOT gated). The text codec parses into an owned `BabelAst { records: Vec<BabelRecord> }` where each record owns `String` fields — this allocates per call. T3.2's zero-alloc requirement is for the latent codec only. A future zero-alloc text codec would need a borrow-based (Cow/lifetime) parser, tracked as a separate optimization if a consumer needs it.

### G5 determinism (PASS)
- **0 byte mismatches, 0 commitment mismatches** across 1500 entries and two independent codec instances.
- **1377/1500 distinct commitments** (the 123 collisions are corpus degeneracies — the LCG-generated corpus has some duplicate entries, which correctly produce duplicate commitments; this is expected, not a bug).
- Cross-architecture determinism is a property of BLAKE3 (portable, no float math) + the text codec's no-float parser path. Within-run determinism (verified here) is a necessary precondition; full cross-arch verification requires running the bench on ARM64 + x86_64 + wasm32 and diffing digests (documented as the remaining G5 step — not falsifiable in a single-arch run).

---

## G4 no-regression note (pre-existing failures, NOT caused by babel_codec)

`cargo test -p katgpt-core --all-features` reports 1869 passed / 2 failed. The 2 failures are **pre-existing and unrelated** to babel_codec:

1. `curator::tests::test_verification_weight_thresholds` — a float-precision/logic issue in `curator.rs` (not touched by this plan). Fails identically with or without `babel_codec`.
2. `rtdc::tests::subtree::cg6_verify_cost_within_5x_of_depth_2` — a latency gate flake (5.628× vs 5.5× budget under full-suite load). **Passes in isolation.** Not caused by babel_codec.

All **45 babel_codec tests pass** cleanly under both `--features babel_codec` and `--all-features`.

---

## Action items

- [x] **Phase 1 executed**: skeleton (mod.rs + Cargo.toml feature + lib.rs gate). `cargo check --features babel_codec` clean.
- [x] **Phase 2 executed**: `FixedRuleTextCodec` BT-P8 parser + emitter, 22 unit tests. Round-trip bit-identical on schema-covered subset.
- [x] **Phase 3 executed**: `SigmoidLatentCodec<D, K>` latent projection codec, 11 unit tests. Zero-alloc hot path (T3.2).
- [x] **Phase 4 executed**: `BabelCommitment` BLAKE3 newtype, 12 unit tests. Tamper detection + determinism.
- [x] **Phase 5 executed**: GOAT bench. G1/G3/G4/G5 PASS, G2 FAIL (1.14× < 2×).
- [x] **Final demotion**: `babel_codec` stays opt-in. No promotion to default.
- [x] **Honest negative result documented** (this file).
- [ ] **Issue #002**: close as moot unless a learned codec (riir-train) raises the ratio above 2× — the 1.14× byte savings will not survive LatCal commitment-gas overhead (BG4 predicted exactly this failure).

---

## Final verdict

**The deterministic BT-P8 subset of BabelTele is a correct, fast, deterministic codec — but it is NOT a 2× compressor on structured data.** The modelless subset cannot reach the paper's headline number because that number requires LLM-prompted omnilingual lexical selection (riir-train territory). The primitive ships opt-in as a useful building block (bijective BT-P8 ↔ verbose round-trip with BLAKE3 commitment) for any future consumer that needs deterministic symbolic-text encoding — but it does not earn default promotion, and the LatCal chain-commitment fusion (issue #002) is blocked on the value side.

**Matches the CompressionDrafter precedent:** both codecs failed the same G2 gate, for mirror-image reasons (LZ4: data too short; BT-P8: data too dense). The 2×+ compression opportunity for quest/KG text lives in verbose natural-language prose, which neither deterministic codec can handle.

## TL;DR

G2 failed at 1.14× (the make-or-break gate that killed CompressionDrafter twice). G1/G3/G4/G5 all pass. `babel_codec` stays opt-in — honest negative result, matching the CompressionDrafter precedent. The deterministic BT-P8 codec is correct and useful but cannot reach 2× on already-dense structured data; the paper's 3.6× requires LLM-prompted compression (riir-train scope).
