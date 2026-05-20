# Plan 082b: HRM-Text Technique Distillation

> **Status:** 📋 Proposed
> **Branch:** `develop/feature/082b_hrm_distill`
> **Depends on:** Plan 082 (RowNormM, complete), Plan 066 (D2F, complete)
> **Research:** `.research/48_HRM_Text_Hierarchical_Recurrent_Pretraining.md`
> **Source:** https://github.com/sapientinc/HRM-Text
> **Goal:** Distill 4 proven techniques from HRM-Text into our training stack: Adam-atan2 optimizer, multipack LPT batching, backprop warmup scheduling, and learned initial states.

## Summary

HRM-Text achieves 1B-scale pretraining with ~$1000. While the hierarchical recurrent architecture itself is outside our scope (we do LoRA fine-tuning, not pretraining), several supporting techniques distill cleanly:

1. **Adam-atan2** — Replace `momentum / (v + eps)` with `atan2(momentum, v)` in our WGSL Adam kernel
2. **Multipack LPT batching** — Smart sequence packing for LoRA training batches
3. **Backprop warmup** — Ramp compute depth during early training
4. **Learned initial states** — Non-zero initialization for recurrent carry

## Tasks

- [ ] **T1: Adam-atan2 WGSL kernel** — Replace Adam update in training shaders
  - Modify `crates/riir-gpu/src/kernels/adam_optimizer.wgsl` (or equivalent)
  - Change: `let update = momentum / (v_sqrt + eps)` → `let update = atan2(momentum, v_sqrt)`
  - Remove epsilon from optimizer config (atan2 handles near-zero)
  - Add EMA buffer (optional, for evaluation weight smoothing)
  - Location: `crates/riir-gpu/src/kernels/`

- [ ] **T2: Multipack LPT batch sampler (Rust)** — Smart sequence packing for training
  - Implement `MultipackSampler` struct in Rust
  - LPT (Longest Processing Time) scheduling via min-heap
  - Binary search to find optimal pack size per batch
  - Target: Go position training, game replay batches
  - ~99.5% token-slot utilization (vs ~70% with naive padding)
  - Location: `microgpt-rs/src/training/sampler.rs` or `riir-ai/crates/riir-gpu/src/sampler.rs`

- [ ] **T3: Backprop warmup scheduler** — Ramp compute depth during training
  - Generic warmup formula: `bp_steps = min + ramp_frac * (max - min)`
  - Apply to: HLA scan steps, MCTS rollout depth, training iterations
  - Config: `bp_min_steps`, `bp_max_steps`, `bp_warmup_ratio`
  - Location: `microgpt-rs/src/training/scheduler.rs`

- [ ] **T4: Learned initial states** — Non-zero init for recurrent carry
  - Replace zero-init with truncated normal init for:
    - HLA carry state (SK, CQV, mQ, G, h in Plan 057)
    - Raven RSM slot initialization (Plan 020)
    - zL_init equivalent for any recurrent module
  - Use `trunc_normal_init_(std=1.0)` as in HRM-Text
  - Location: relevant init functions in both projects

- [ ] **T5: Integration tests** — Verify each technique
  - T1: Adam-atan2 produces bounded updates (all < π/2 in magnitude)
  - T2: Multipack sampler achieves >95% utilization on test sequences
  - T3: Warmup schedule ramps correctly from min to max
  - T4: Learned init produces different (non-zero) states across runs
  - Location: `crates/riir-gpu/tests/` or `tests/`

- [ ] **T6: Benchmark** — Measure training impact
  - T1: Adam-atan2 vs Adam training loss on same data (Go positions)
  - T2: Multipack vs naive padding throughput (tokens/sec)
  - Location: `crates/riir-gpu/tests/bench_*.rs`

## Architecture

```text
┌──────────────────────────────────────────────────────┐
│                Training Pipeline                      │
│                                                      │
│  ┌─────────────┐   ┌───────────────┐                │
│  │ Multipack   │──▶│ Batch         │                │
│  │ LPT Sampler │   │ Construction  │                │
│  │ (T2)        │   └───────┬───────┘                │
│  └─────────────┘           │                         │
│                            ▼                         │
│  ┌─────────────────────────────────────────────┐    │
│  │            Training Loop                     │    │
│  │                                              │    │
│  │  bp_steps = warmup(step, total) (T3)         │    │
│  │  for i in 0..bp_steps:                       │    │
│  │    loss = forward(batch)                     │    │
│  │    Adam-atan2.step() (T1)                    │    │
│  │                                              │    │
│  │  Carry: learned_init() (T4)                  │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Adam-atan2 over standard Adam** | Bounded updates prevent LoRA weight explosion. No epsilon tuning. Proven in 1B-scale pretraining. |
| **LPT over first-fit packing** | ~99.5% vs ~70% utilization. O(n log n log k) complexity. Balances quadratic attention cost. |
| **Warmup as generic scheduler** | Same formula applies to any variable-depth compute. Reusable across MCTS, HLA, game training. |
| **Truncated normal over zero init** | HRM-Text uses std=1.0 truncated normal for zL_init. Gives richer starting state for recurrent modules. |
| **No EMA in initial version** | EMA is useful but adds complexity. Ship atan2 first, add EMA later if needed. |

## Non-Goals

- HRM architecture implementation — outside our scope (we do LoRA, not pretraining)
- FlashAttention 3 kernels — CUDA/Hopper specific, we use wgpu/Metal
- FSDP2 distributed training — single GPU only
- PrefixLM two-pass kernel — we already have bidirectional attention
- Full pretraining framework — not applicable to our use case

## Dependencies

- T1 → T5, T6 (tests/benchmarks need kernel)
- T2 → T5 (sampler needs utilization test)
- T3 → T5 (warmup needs correctness test)
- T4 → T5 (init needs verification test)

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Adam-atan2 doesn't help for LoRA | HRM-Text validates on large scale. For LoRA, bounded updates still prevent explosion. If no gain, revert is trivial (1 line). |
| LPT overhead for small batches | For Go positions (361 moves), batches are small. LPT overhead may exceed padding savings. Measure first. |
| Warmup schedule too aggressive | Configurable min/max/ratio. Default to conservative values from HRM-Text. |
| Learned init causes instability | std=1.0 is what HRM uses. For low-rank LoRA, may need smaller std. Configurable. |

## Success Criteria

- [ ] Adam-atan2 WGSL kernel passes unit tests (bounded updates)
- [ ] Multipack sampler achieves >90% utilization on Go position batches
- [ ] Warmup schedule produces correct ramp from min to max steps
- [ ] Learned init produces non-zero, stable initial states
- [ ] All existing tests still pass
- [ ] Training benchmark: Adam-atan2 ≤ 5% overhead vs Adam

## References

- Source: https://github.com/sapientinc/HRM-Text
- Research: `.research/48_HRM_Text_Hierarchical_Recurrent_Pretraining.md`
- HRM-Text optimizer: `.raw/HRM-Text/models/adam_atan2.py`
- HRM-Text sampler: `.raw/HRM-Text/multipack_sampler.py`
- HRM-Text HRM model: `.raw/HRM-Text/models/baselines/hrm_nocarry_bp_warmup.py`
- Related: Plan 082 (RowNormM), Plan 066 (D2F), Plan 057 (HLA)