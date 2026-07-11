# Issue 038 — HOLA Cache Perplexity + RULER Gate (G5) — needs trained GDN2 weights

**Filed:** 2026-07-06
**Priority:** P2 (modelless G1–G4 PASS; G5 is the quality-parity gate for default-on promotion)
**Source paper:** [A Hippocampus for Linear Attention](https://arxiv.org/abs/2607.02303) — Cui 2026, HOLA
**Plan:** [`.plans/395_hippocampal_exact_kv_cache.md`](../.plans/395_hippocampal_exact_kv_cache.md)
**Research:** [`.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md`](../.research/378_HOLA_Hippocampal_Exact_KV_for_Linear_Attention.md)
**Status:** Open — riir-train job, not blocking katgpt-rs modelless promotion

---

## Problem

The HOLA hippocampal exact KV cache (Plan 395) ships as a modelless opt-in
primitive with G1–G4 GOAT gates all PASS:

- **G1** — eviction correctness: 8/8 needles retained, distractors evicted, order-independent.
- **G2** — latency: observe 28.7 ns (W=64); read 2.87 µs (W=64, D=256, fast path).
- **G3** — no-regression: byte-identical GDN2 state with/without cache observer.
- **G4** — retrieval: HOLA softmax recovers 8/8 needles (cosine ≈ 1.0); recency baseline 0/8.

What G1–G4 do NOT prove: that adding the HOLA cache to a **trained** GDN2 model
**improves perplexity and long-context recall** on real text. The paper reports
−16% Wikitext perplexity at 340M and robust RULER S-NIAH-1 at 16× training
length. Reproducing that requires training a matched GDN2 model with and without
the cache — a riir-train job.

## G5 gate (the quality-parity gate)

Train a matched GDN2 model at **46M** (smallest paper scale, App. A) on
**FineWeb-Edu 0.5B tokens** with:
- (A) bare GDN2 (no cache).
- (B) GDN2 + HOLA cache (w=64, γ=ones, softmax read).

Report:
- **Wikitext perplexity** — target: ≥ −10% PPL vs (A).
- **RULER S-NIAH-1 @ 4k context** (1× training length) — target: ≥ 0.7.
- **RULER S-NIAH-1 @ 8k context** (2× training length) — stretch target.

## Why this is deferred (not blocking)

Per Research 378 §3 MOAT gate and the katgpt-rs modelless-first mandate: the
cache *mechanism* (surprise-evicted bounded KV + decoupled RMSNorm-γ read) is
modelless at inference. The γ vector is a model parameter (like any RMSNorm γ),
not a runtime-learned value. G1–G4 prove the mechanism works modellessly on a
controlled synthetic. G5 proves it improves a real model — that requires
training, which is riir-train's domain.

## Modelless γ unblock status (§3.5)

Both deterministic γ variants PASS G4:
- **γ = ones** (identity RMSNorm): 8/8 needles, cosine ≈ 1.0.
- **Per-key norm rescale** (`γ_i = √d / max(‖k_i‖, ε)`): 8/8 needles, cosine ≈ 1.0.

No γ-tuning deferral needed — both modelless variants work. Trained γ may still
improve G5 perplexity, but the modelless baseline is strong.

## Cross-references

- Research 378 §3 MOAT gate (G5 deferred per §3.6 — no quality-parity claim without training).
- Plan 105 (GDN2 — the backbone, default-on).
- Plan 271 (AM — KV-compression slot competitor).
- Plan 287 (Sink-Aware — KV-compression slot competitor).

## Promotion decision (after G5)

If G5 PASSES: HOLA is a candidate for default-on in the KV-compression slot,
weighed against AM (Plan 271) and Sink-Aware (Plan 287). Demote the loser when
the slot is contested.

If G5 FAILS: keep HOLA opt-in. The mechanism is still GOAT (G1–G4 pass); G5
failure would indicate the synthetic toy doesn't translate to real text, which
is a finding about the gate, not the mechanism.
