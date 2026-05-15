# MTP Threshold Guide (Plan 055)

## When Each Feature Activates

### Target Activations (`mtp_activation_threshold`)

**Purpose**: Feed the target model's final hidden state into the drafter, giving it richer context than just the previous token.

**Activation condition**: `target_config.n_embd >= mtp_activation_threshold`

**Threshold values**:
| Config | n_embd | Threshold | Active? |
|--------|--------|-----------|---------|
| micro | 16 | MAX | ❌ |
| game | 32 | MAX | ❌ |
| draft | 4 | MAX | ❌ |
| small_target | 64 | 64 | ✅ |
| gqa_draft | 64 | 64 | ✅ |
| bpe | 32 | 32 | ✅ |
| bpe_draft | 16 | 16 | ✅ |

**Fallback**: When `mtp_activation_proj` weights are not loaded (always the case currently), uses truncate/pad — copies `min(draft_n_embd, target_n_embd)` elements from target's hidden state.

**Expected gain**: Higher draft acceptance rate on complex prompts (Python→Rust translation, etc.)

### Shared KV Cache (`mtp_shared_kv_prompt_threshold`)

**Purpose**: Preload the drafter's KV cache with the target's pre-computed keys/values for past positions.

**Activation condition**: `pos > mtp_shared_kv_prompt_threshold` AND `target_kv_dim == draft_kv_dim`

**Threshold values**:
| Config | Threshold | Active? |
|--------|-----------|---------|
| micro | MAX | ❌ |
| game | MAX | ❌ |
| small_target | 128 | ✅ (pos > 128) |
| bpe | 64 | ✅ (pos > 64) |

**Constraint**: Only works when target and draft have matching `kv_dim` (n_kv_head × head_dim). When dimensions differ (e.g., bpe kv_dim=32 vs bpe_draft kv_dim=16), silently skips preload.

### Clustered LM Head (`mtp_cluster_vocab_threshold`)

**Purpose**: Two-stage vocab lookup: predict cluster → compute exact logits only for tokens in that cluster.

**Activation condition**: `vocab_size >= mtp_cluster_vocab_threshold` AND `mtp_cluster_classifier` AND `mtp_cluster_map` weights are loaded.

**Threshold values**:
| Config | vocab | Threshold | Active? |
|--------|-------|-----------|---------|
| micro | 27 | MAX | ❌ |
| game | 10 | MAX | ❌ |
| small_target | 4096 | MAX | ❌ |
| bpe | 4096 | 4096 | ✅ (when weights present) |

**Current status**: Cluster weights are never loaded (always `None`), so the standard full-vocab LM head is always used. To activate, load trained cluster weights into `TransformerWeights::mtp_cluster_classifier` and `mtp_cluster_map`.

**Cluster assignment**: Round-robin by token ID (baseline). K-means from embedding similarity planned for riir-burner (Plan 056).

## Overriding at Inference Time

All MTP thresholds can be overridden via `InferenceOverrides`:

```rust
let overrides = InferenceOverrides {
    mtp_activation_threshold: Some(64),
    mtp_shared_kv_prompt_threshold: Some(128),
    ..Default::default()
};
let config = Config::bpe().with_overrides(&overrides);
```

## Validation

`Config::validate()` enforces:
- `mtp_cluster_size > 0` — cluster size must be positive when clustered LM head is in use

Other thresholds use `usize::MAX` as "disabled" sentinel, which is valid and needs no special enforcement.

## Architecture Diagram

```
Standard DFlash (current):
  token → [drafter forward] → draft tokens → [target verify] → accepted tokens

MTP-Enhanced DFlash (this plan):
  token + target_hidden_state
    → [projection: truncate/pad OR learned matmul]
    → [drafter forward with projected context]
    → draft tokens
    → [target verify]
    → accepted tokens
```

## Composability with Existing Features

| Feature | Relationship |
|---------|-------------|
| DFlash (`speculative/dflash.rs`) | **Orthogonal** — MTP feeds richer context INTO the drafter. DFlash's tree verification still runs on the output. |
| LeviathanVerifier (`speculative/verifier.rs`) | **Modified** — target→draft activation transfer happens here (target already exposes `hidden_state`) |
| PagedKVCache (`transformer.rs`) | **Extended** — read-only cross-attention view for drafter |
| Sparse MLP threshold (`Config.sparse_threshold`) | **Same pattern** — threshold-gated feature activation |
| TurboQuant | **Independent** — compresses precision, MTP improves draft quality |
| PFlash | **Independent** — compresses sequence, MTP improves draft quality |

## References

- [Gemma 4 architecture](https://blog.google/technology/ai/gemma-technical-report/) — Multi-Token Prediction design
- Plan 055 — `microgpt-rs/.plans/055_gemma_mtp_drafter.md`
- Plan 056 — riir-burner cluster weight training