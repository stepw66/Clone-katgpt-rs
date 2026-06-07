# Plan 214: CoExplain Bidirectional Alignment — Modelless Pruner Evolution

**Date:** 2026-06
**Status:** Plan
**Depends On:** Research 189 (Editable XAI CoExplain Bidirectional Alignment)
**Feature Flags:** `coexplain_pruner`, `ted_lite`, `coexplain_riir`

---

## Goal

Implement CoExplain's Read/Write/Enhance cycle for the modelless inference pipeline, enabling self-refining pruners and the Curator marketplace without lora.bin.

## Tasks

### Phase 1: TED-Lite Divergence Metric (Enabling Infrastructure)
- [ ] Add `PrunerDivergence` struct to `katgpt-rs/src/speculative/types.rs`
  - `threshold_divergence: f32` — Σ |τ_current - τ_original| / N
  - `topology_divergence: f32` — Hamming distance on branch existence vectors
  - `lambda_t: f32` — developer-configurable divergence clamp (default: 0.1)
- [ ] Implement `PrunerDivergence::compute()` — O(k) per pruner
- [ ] Add `PrunerDivergence::clamp_adjustment()` — reject bandit updates that exceed lambda_t
- [ ] Add diagnostic log: emit divergence metrics per N tokens (behind `tracing`)

### Phase 2: Self-Refining Pruner (Extends BanditPruner)
- [ ] Add `PrunerAccuracy` tracker to `katgpt-rs/src/freq_bandit.rs` or new module
  - Track TP, TN, FP, FN per pruner slot
  - Compute precision, recall, F1 per slot
- [ ] Implement threshold mode: adjust SynPruner rejection threshold based on FP/FN ratio
  - `threshold_adjustment = sigmoid(α * (FP_rate - FN_rate))` where α is learning rate
  - Clamp via TED-Lite lambda_t
- [ ] Implement topology mode: prune low-value DDTree branches (acceptance rate < ε)
  - Expand high-value branches (acceptance rate > 1-ε)
  - Clamp via TED-Lite lambda_t
- [ ] Feature gate: `coexplain_pruner`

### Phase 3: CoEditable ConstraintPruner (Bidirectional)
- [ ] Add `EditableConstraintPruner` trait extending `ConstraintPruner`
  - `fn edit_threshold(&mut self, slot: usize, new_threshold: f32) -> Result<(), DivergenceError>`
  - `fn edit_topology(&mut self, branch: &[usize], action: TopologyAction) -> Result<(), DivergenceError>`
  - `fn snapshot(&self) -> PrunerSnapshot` — captures golden reference for TED-Lite
  - `fn divergence(&self) -> &PrunerDivergence` — current divergence from snapshot
- [ ] Implement `SynPruner` changes to support `EditableConstraintPruner`
  - Dynamic threshold adjustment (Tier 0 bracket balancer thresholds)
  - Topology changes (add/remove validation rules)
- [ ] Add rule editor backend: accept JSON rule format → compile to ConstraintPruner config
  - Input: `{"rules": [{"attribute": "bracket_depth", "threshold": 0, "action": "reject"}]}`
  - Output: Updated SynPruner config
- [ ] Feature gate: `coexplain_pruner`

### Phase 4: Neuro-Symbolic RIIR Feedback Loop (Curator Marketplace)
- [ ] Add rule extraction from successful RIIR translations
  - After successful translation, extract DDTree paths that led to compilable output
  - Store as "translation rules" in Episode DB
- [ ] Add Curator rule ingestion endpoint (depends on riir-ai Curator API)
  - Accept Curator decision tree rules via MCP/Web UI
  - Compile to WASM ConstraintPruner via riir-validator-sdk
  - Hot-swap into inference pipeline
- [ ] Add bandit refinement loop for Curator rules
  - Track translation success rate per Curator rule
  - Adjust thresholds via Phase 2 self-refining mechanism
  - Report accuracy back to Curator (Read mode)
- [ ] Add before/after comparison tests
  - Without CoExplain: baseline pruner accuracy on Python→Rust corpus
  - With CoExplain: improved accuracy after bandit refinement
  - With Curator rules: further improvement from domain knowledge injection
- [ ] Feature gate: `coexplain_riir` (implies `coexplain_pruner`)

### Phase 5: CPU/GPU Auto-Route Integration
- [ ] Add CoExplain workload to inference router (`katgpt-rs/src/inference_router.rs`)
  - Bandit updates → CPU (lightweight, O(1) per token)
  - Rule compilation → async worker (WASM compile is CPU-bound but infrequent)
  - TED-Lite computation → CPU (O(k), negligible)

## Tests/Examples

- [ ] `tests/coexplain_ted_lite.rs` — divergence metric correctness, clamping behavior
- [ ] `tests/coexplain_self_refining.rs` — pruner accuracy improves over N iterations
- [ ] `examples/coexplain_demo.rs` — before/after pruner accuracy with bandit refinement
- [ ] Integration test: Curator rule → WASM → DDTree → valid Rust output

## Expected Results

| Metric | Before | After (Self-Refining) | After (Curator Rules) |
|--------|--------|----------------------|----------------------|
| Pruner accuracy | ~90% (SynPruner) | ~95% (bandit-refined) | ~98% (domain knowledge) |
| Invalid tokens accepted | 10% | 5% | 2% |
| Compilation success rate | ~60% | ~70% | ~85% |

## Feature Gate Summary

```
[features]
ted_lite = []                           # Phase 1: Divergence metric
coexplain_pruner = ["ted_lite"]         # Phase 2+3: Self-refining + editable
coexplain_riir = ["coexplain_pruner"]   # Phase 4: Full RIIR feedback loop
```

Default: `ted_lite` ON after GOAT proof (no perf hurt).
