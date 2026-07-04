# Plan 045 Bomber Tech Isolation A/B — LoRA Artifacts

Canonical LoRA artifacts for the **Bomber Tech Isolation A/B tournament**
([Plan 045](../../riir-ai/.plans/045_bomber_tech_isolation_ab.md)). These are
the "Secret A" weights consumed by `bomber_tech_ab_demo` and the dendritic-LoRA
training example.

> **Artifact files are local-only.** The `.bin` files live in
> `katgpt-rs/output/lora_ab/` which is gitignored (`.gitignore` line 4:
> `output/`). They are environment-specific training outputs. This document
> captures their expected shape so future runs don't silently grab the wrong
> artifact (see
> [Issue 018](../../riir-ai/.issues/018_ab_tournament_synergy_regression.md)).

## Files

| File | Size | Magic | n_adapters | rank | alpha | in_dim | out_dim | LossWeighting |
|------|-----:|-------|-----------:|-----:|------:|-------:|--------:|---------------|
| `output/lora_ab/game_lora_flat_baseline.bin` | 9316 B | `LORA` | 6 | 4 | 8.0 | 32 | 32 | `Uniform` |
| `output/lora_ab/game_lora_action_only.bin`   | 9316 B | `LORA` | 6 | 4 | 8.0 | 32 | 32 | `ActionOnly` |

Binary structure (verified 2026-07-04 via `xxd`, offsets from
`katgpt-types/src/lora.rs`):

```
offset 0..4    magic           = b"LORA"
offset 4..8    version         = 1 (u32 LE)
offset 8..40   blake3(32)      = payload checksum (full BLAKE3-256)
offset 40..    payload         = n_adapters(u32) | rank(u32) | alpha(f32) | per-adapter...
```

Per-adapter in payload: `in_dim(u32) | out_dim(u32) | a_f32s(rank×in_dim) | b_f32s(out_dim×rank)`

The two files share `a`-array sentinels but differ in `b`-array values — only
the `b` (output projection) is trained; `a` is the frozen random init.

## LossWeighting semantics

From `riir-train/crates/riir-train-engine/src/cpu_lora_train.rs:143-153`:

- **`Uniform`** (`game_lora_flat_baseline.bin`) — every position contributes
  equally to loss and gradient (mean CE). The historical baseline.
- **`ActionOnly`** (`game_lora_action_only.bin`) — only the action position
  (last input token, `seq_len - 1`) contributes; earlier board positions are
  masked to zero gradient.

The `flat_baseline` name is a synonym for `Uniform` — it is the "flat"
(uniform-weighted) baseline that `ActionOnly` and `ActionUpweighted(w)` are
compared against.

## Loader compatibility

Both files load cleanly via:

- `LoraAdapter::load(path)` → `Vec<LoraAdapter>` of length 6 (full multi-adapter).
- `LoraAdapter::load_first(path)` → first adapter only.

**Note:** `LoraPlayer::new_with_lora`, `LoraWasmPlayer::new_with_lora`, and
`HLPlayer::new_with_secrets` (in `katgpt-rs/src/pruners/bomber/players.rs`)
call `load_first` — only adapter 0 is wired into inference. Layers 1–5 are
silently dropped on these players. This is the documented single-forward-pass
heuristic limitation; multi-adapter full wiring requires the L2+ inference
path.

## Relation to other LoRA artifacts

These are **NOT** the same as the various `lora_final.bin` files scattered
across `riir-ai/output/`. Per
`riir-train/.docs/09_training_data_pipeline.md:376`
(`cp output/lora_final.bin raw/game_lora.bin`), the convention is:

```
training run produces → output/lora_final.bin
                   cp → raw/game_lora.bin (consumed by inference)
```

Each training run produces its own `lora_final.bin`. The two files in
`output/lora_ab/` are the **specific** artifacts used by Plan 045's 1000-round
tournament that produced the documented synergy results (+31, +271, +490).
Using a different `lora_final.bin` (e.g. `riir-ai/output/lora_final.bin`, which
is rank=2, 16-dim not rank=4, 32-dim) silently regresses the A/B result — see
[Issue 018](../../riir-ai/.issues/018_ab_tournament_synergy_regression.md).

## Consumption

| Consumer | Path | File |
|----------|------|------|
| `bomber_tech_ab_demo` (Plan 045 T6) | `--lora output/lora_ab/game_lora_flat_baseline.bin` | flat_baseline |
| `train_bomber_dendritic_cpu` (`--dense-path`) | `output/lora_ab/game_lora_flat_baseline.bin` | flat_baseline |
| Comparison / ablation vs `ActionOnly` | manual | action_only |

The `flat_baseline` variant is the **default** for downstream consumers;
`action_only` is retained for the `Uniform` vs `ActionOnly` ablation.

## Regeneration

```bash
cd riir-train

# flat_baseline (Uniform loss)
cargo run --release -p riir-train-gpu --example train_bomber_cpu -- \
    --output ../../katgpt-rs/output/lora_ab/game_lora_flat_baseline.bin \
    --loss-mode uniform

# action_only (ActionOnly loss)
cargo run --release -p riir-train-gpu --example train_bomber_cpu -- \
    --output ../../katgpt-rs/output/lora_ab/game_lora_action_only.bin \
    --loss-mode action-only
```

Training is **out of scope for `katgpt-rs`** (modelless-first mandate). These
artifacts are checked in as frozen inputs; if they drift, file an issue in
`riir-train` and re-run the trainer above.

## Verification

```bash
# Structural check — magic, n_adapters, rank, alpha, dims
xxd output/lora_ab/game_lora_flat_baseline.bin | head -3
# Expected first 4 bytes: 4c4f 5241 (b"LORA")
# Expected bytes 40-47:   0600 0000 0400 0000 (n=6, rank=4)

# Loader check (in katgpt-rs)
cargo test --features bomber-wasm --lib lora
```

## See also

- [Plan 045 — Bomber Tech Isolation A/B](../../riir-ai/.plans/045_bomber_tech_isolation_ab.md)
- [Issue 018 — A/B Tournament Synergy Regression](../../riir-ai/.issues/018_ab_tournament_synergy_regression.md) — the artifact-confusion lesson
- [Issue 016 — WASM Validator Safety Mismatch](../../riir-ai/.issues/016_wasm_validator_safety_mismatch.md) — paired WASM artifact
- `riir-train/crates/riir-train-engine/src/cpu_lora_train.rs` — `LossWeighting` enum
- `riir-train/crates/riir-train-gpu/examples/train_bomber_cpu.rs` — regeneration command
