# Plan 225: RAT+ Recurrence Bridge — Modelless Dilated Inference

**Status**: ✅ Complete (All Tasks Done)
**Research**: `.research/201_RAT_Plus_Train_Dense_Infer_Sparse.md`
**Feature Gate**: `rat_plus_bridge` (default-off → GOAT gate → default-on if proved)
**Dependencies**: `gdn2_attention`, `dash_attn`, `vortex_flow`

---

## Goal

Wire existing GDN2 recurrent state as a "bridge" for dilated sparse attention during decode. No retraining — pure inference-time adaptation. Target: 8-64× attention FLOPs reduction with <2% quality degradation.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              RAT+ Bridge Decode Pipeline                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Prefill (unchanged):                                    │
│    Full dense attention → full KV cache + GDN2 state     │
│                                                          │
│  Decode (with bridge):                                   │
│    ┌─────────────────┐    ┌─────────────────────────┐    │
│    │ GDN2 State      │───▶│ Bridge Projection       │    │
│    │ (per-head S)    │    │ (sigmoid-gated readout) │    │
│    └─────────────────┘    └─────────┬───────────────┘    │
│                                    │                     │
│    ┌─────────────────┐             │                     │
│    │ Dilated KV      │             │                     │
│    │ (every D-th     │◀────────────┤                     │
│    │  token)         │  merge via  │                     │
│    └────────┬────────┘  gating     │                     │
│             │                       │                     │
│             ▼                       ▼                     │
│    ┌──────────────────────────────────────────┐          │
│    │ Fused Bridge Attention                   │          │
│    │ output = α·attn(dilated_kv) +            │          │
│    │         (1-α)·bridge(gdn2_state)         │          │
│    │ where α = sigmoid(gate)                  │          │
│    └──────────────────────────────────────────┘          │
│                                                          │
│  Dilation D selected by:                                 │
│    TriggerGate (QPS-based) + River Valley (per-layer)    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Tasks

### Phase 1: Core Infrastructure

- [x] **T1.1** Add `DilationConfig` enum to `katgpt-core/src/types.rs`
  ```rust
  #[repr(u8)]
  #[derive(Clone, Copy, Debug, PartialEq)]
  pub enum DilationConfig {
      D1 = 1, D2 = 2, D4 = 4, D8 = 8, D16 = 16, D32 = 32, D64 = 64,
  }
  ```

- [x] **T1.2** Add `rat_plus_bridge` feature flag to `katgpt-core/Cargo.toml` and `Cargo.toml`

- [x] **T1.3** Create `src/rat_bridge/mod.rs` with module structure
  - `bridge.rs` — GDN2 state → bridge projection (RatBridgeState + sigmoid gate)
  - `dilated_kv.rs` — strided KV cache access (DilatedKvAccessor + dilated_indices + dilated_len)
  - `fuse.rs` — fused bridge attention (α-blend via bridge_attention())

### Phase 2: Dilated KV Access

- [x] **T2.1** Implement `DilatedKvAccessor` in `dilated_kv.rs`
  - `fn stride_access(kv_cache: &[T], d: DilationConfig) -> &[T]`
  - Zero-copy slice view into existing KV cache with stride D
  - No allocation — just offset arithmetic
  - Added `dilated_decode_step()` — full decode with D-strided KV + sigmoid-gated bridge readout

- [x] **T2.2** Implement `dilated_decode_step()` in `fuse.rs`
  - Replace full KV scan with D-strided KV access
  - Add GDN2 bridge readout as complementary signal
  - Sigmoid gate: `α = sigmoid(⟨q, gdn2_readout⟩)`
  - Added `rat_decode_step()` — high-level decode combining `RatBridgeState` + dilated KV + bridge attention

### Phase 3: Bridge Projection

- [x] **T3.1** Implement `RatBridgeState` in `bridge.rs`
  - Wraps `Gdn2LayerState` with bridge projection
  - Bridge projection: reuses GDN2's `gdn2_readout()` output
  - No new parameters — just reuses existing recurrent state
  - Added `set_dilation_from_qps()` — QPS-based adaptive dilation
  - Added `update_projection()` — copies GDN2 readout into bridge state

- [x] **T3.2** Implement `bridge_attention()` in `fuse.rs`
  - `y = α · softmax(Q·K_dilated^T)·V_dilated + (1-α) · S·q`
  - Where S is GDN2 state, q is query, K_dilated/V_dilated are strided
  - α computed per-head via sigmoid

### Phase 4: Adaptive Dilation Routing

- [x] **T4.1** Add dilation selection to `TriggerGate`
  - Low QPS → D=1 (dense), High QPS → D=16 or D=64
  - Standalone `select_dilation()` function in `dilation_router.rs`

- [x] **T4.2** Add River Valley per-layer dilation in `DilationRouter`
  - High RV (peaked) layers → tolerate D=16
  - Low RV (flat) layers → stay at D=1
  - `DilationRouter` struct with per-layer RV scores + QPS thresholds

- [x] **T4.3** Register `dilation_router` module in `rat_bridge/mod.rs`
  - Feature-gated behind `rat_plus_bridge`
  - Re-exported via `pub use dilation_router::*`

### Phase 5: VortexFlow Integration

- [x] **T5.1** Implement `DilationBridgeRouter` as VortexFlow trait
  - `compute_centroids`: store dilated KV centroids from D-strided keys
  - `score_blocks`: GDN2 bridge + dilated centroids for block scoring (sigmoid gate)
  - Demonstrates RAT+ insight: recurrence improves block scoring
  - File: `src/rat_bridge/vortex.rs` (4 tests)

- [x] **T5.2** Register `vortex` module in `rat_bridge/mod.rs`
  - Feature-gated behind `rat_plus_bridge`
  - Re-exported via `pub use vortex::*`

### Phase 6: Benchmarks & GOAT Proof

- [x] **T6.1** Create `tests/goat_225_rat_bridge.rs` (11 tests)
  - Test: bridge attention output dims match dense attention
  - Test: sigmoid gate valid [0,1] range (high positive → >0.99, high negative → <0.01)
  - Test: sigmoid gate at zero → exactly 0.5
  - Test: dilation D=1 matches dense (all 16 positions)
  - Test: dilation D=4 accesses every 4th position
  - Test: dilation D=16 accesses every 16th position
  - Test: dilated_len matches indices across all D × lengths
  - Test: empty KV → bridge readout only
  - Test: DilationBridgeRouter centroid count (D1=4, D4=1)
  - Test: DilationBridgeRouter score ordering (aligned > orthogonal)
  - Test: rat_decode_step valid output across all 7 dilation configs

- [x] **T6.2** Create `tests/bench_225_rat_bridge.rs` (3 benchmarks)
  - bench_decode_latency_per_dilation: D=1 1.66ms, D=4 426µs, D=16 103µs, D=64 28.7µs
  - bench_bridge_projection_overhead: 778ns per gate call
  - bench_kv_cache_memory_per_dilation: D=1 100%, D=4 25%, D=16 6.2%, D=64 1.6%
  - Confirmed: decode latency ∝ 1/D, KV cache ∝ 1/D

- [x] **T6.3** Before/after comparison test
  - D=1 (dense) vs D=16 (bridge): both produce valid dim-matched finite output
  - Outputs differ (different KV positions used), α=0.832 for both
  - Quality validation requires real model evaluation

- [x] **T6.4** GOAT gate decision
  - D=16: 16× FLOPs reduction meets ≥8× threshold ✓
  - D=64: 64× FLOPs reduction meets ≥40× threshold ✓
  - Decision: keep `rat_plus_bridge` as opt-in until real quality benchmarks pass

---

## Feature Gate Strategy

```toml
# katgpt-core/Cargo.toml
rat_plus_bridge = ["gdn2_attention", "dash_attn"]

# Main Cargo.toml
rat_plus_bridge = ["katgpt-core/rat_plus_bridge"]
# Default: OFF until GOAT proof passes
```

**Promotion criteria** (per constraints: "if GOAT+gain proof, must be on by default"):
- ✅ D=16: <2% quality loss, >8× decode FLOPs reduction → DEFAULT-ON
- ✅ D=64: <5% quality loss, >40× decode FLOPs reduction → DEFAULT-ON
- ❌ If perf hurt or >5% quality loss → keep OFF, investigate

---

## Constraints Checklist

- [x] **Modelless first** — inference-time only, no LLM training
- [x] **SOLID, DRY** — reuses GDN2 state, VortexFlow trait, TriggerGate
- [x] **Tests/examples** — Phase 6 dedicated to before/after benchmarks
- [x] **CPU/GPU auto-route** — TriggerGate handles QPS-based routing
- [x] **No perf hurt guarantee** — zero cost when disabled, measured when enabled
- [x] **Sigmoid not softmax** — all gating uses sigmoid

---

## Related Plans

- Plan 105: GDN2 (provides recurrent state)
- Plan 106: DashAttention (provides sparse attention)
- Plan 196: VortexFlow (provides router trait)
- Plan 202: RV-gated compute routing (per-layer routing)
- Plan 204: Selectivity Router (adaptive CoT)
- Plan 212: Collapse-Aware Adaptive Thinking (thinking budget)
