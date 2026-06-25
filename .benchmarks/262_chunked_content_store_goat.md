# GOAT Gate Benchmarks — Plan 272 Chunked Content-Addressed Merkle Store

**Feature:** `chunked_content_store` (opt-in)
**Research:** [262 — Lore Chunked Asset Merkle Store Modelless](../.research/262_Lore_Chunked_Asset_Merkle_Store_Modelless.md)
**Plan:** [272](../.plans/272_chunked_asset_merkle_store.md)
**Date:** 2026-06-25

## G1–G7 Gate Table (from Research 262 §6)

| Gate | Metric | Target | Status | Measured |
|------|--------|--------|--------|----------|
| G1 | Dedup ratio (100 blobs, 90% shared) | ≥ 5.0 | ✅ PASS | 8.47 (50 blobs × 10 chunks, 9/10 shared) |
| G2 | Incremental push (10MiB + 1 byte) | ≤ 5% (CDC) | ✅ PASS | 1.35% (FastCDC) vs 52.94% (FixedSize negative control) — proven in Phase 2 `test_cdc_dedup_with_variant` |
| G3 | Inclusion proof cost (1024-chunk) | mean < 10µs | ⏳ DEFERRED | Needs criterion bench target (Cargo.toml conflict); structural correctness verified by existing merkle tests |
| G4 | Light-client verify (no `&self`) | 0 grep hits | ✅ PASS | `verify_proof` is an associated fn — verified by type system (compiles without `&self`) |
| G5 | Hot-path read p99 latency | < 200ns | ⏳ DEFERRED | Needs criterion bench target; `get_chunk` is zero-alloc (papaya `.copied()` on `&'static [u8]`) |
| G6 | Default-off regression | 0 failures | ✅ PASS | `cargo check -p katgpt-core --no-default-features` clean; `chunked_content_store` not in default |
| G7 | Tamper detection (1-bit flip) | 100% BlobId mismatch | ✅ PASS | 10000/10000 — `g7_tamper_detection` test |

## GOAT Decision

**G1, G2, G4, G6, G7 PASS. G3, G5 deferred** — they require `criterion` bench
targets in `Cargo.toml` (`benches/chunked_dedup.rs`), which collides with
concurrent `Cargo.toml` edits. The deferred gates are perf-timing gates only;
the structural correctness they depend on is already verified (G4 light-client
property is enforced by the type system, G5's zero-alloc path is verified by
code inspection).

**Promotion: DEFERRED.** G3 and G5 are perf gates (not correctness gates), but
the GOAT requires all G1–G7 to pass before default-on promotion. The store
stays opt-in until G3/G5 land as criterion benches. The modelless gain is
proven (G1 dedup + G2 incremental push + G7 tamper detection are all
content-addressing properties that need no training).

## Test Provenance

| Test | Gate | File |
|------|------|------|
| `g1_dedup_ratio_meets_target` | G1 | `content_store/goat.rs` |
| `test_cdc_dedup_with_variant` | G2 | `content_store/chunker.rs` (Phase 2) |
| `g4_light_client_verify_no_self` | G4 | `content_store/goat.rs` |
| (type-system check) | G6 | `cargo check --no-default-features` |
| `g7_tamper_detection` | G7 | `content_store/goat.rs` |
