# CausalHeadImportance — Causal Head-Importance Calibration & Scale-Normalized Fusion

**Plan:** [358](../.plans/358_causal_head_importance_calibration.md)
**Research:** [362 — HydraHead Causal Head Importance & Hybrid Attention](../.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md)
**Source paper:** [Tan et al. 2026 — HydraHead (arXiv:2606.20097)](https://arxiv.org/abs/2606.20097), Alibaba, Jun 2026
**Benchmark:** [358_causal_head_importance_goat.md](../.benchmarks/358_causal_head_importance_goat.md)

---

## TL;DR

Two modelless primitives distilled from HydraHead. Both zero-training, zero-backprop, allocation-free hot paths.

| Primitive | Purpose | Feature | Cost |
|---|---|---|---|
| `direct_effect_importance` / `indirect_effect_importance` | Activation/path-patching IE score for a head — is it *necessary* for a capability? | `causal_head_importance` (opt-in) | `#[inline]` f32 arithmetic, 0 allocs |
| `SpanLogitDiffReadout` | Exponentially-decayed span-level logit-difference readout (Eq 9) — the capability-expression scalar | `causal_head_importance` (opt-in) | `&[(f32,f32)]` → f32, 0 allocs |
| `partition_by_causal_score` | Rank heads by IE + partition into critical/convertible (mirrors RTPurbo `HeadCalibration`) | `causal_head_importance` (opt-in) | offline, sub-µs at n=144 |
| `ScaleNormalizedFusion` | Fuse two heterogeneous attention branches via per-head RMSNorm + learnable γ (Eq 13–14) | `causal_head_importance` (opt-in) | caller-scratch, 0 allocs |

**Calibration slot competition:** `CausalNecessity` competes with RTPurbo's `AttentionMass` (Plan 126) for the head-calibration slot. Causal is strictly stronger on bystander-heavy workloads but ~10–100× more expensive to *produce* scores. `AttentionMass` stays the default; `CausalNecessity` is opt-in for the long-context-extreme regime.

---

## Feature Flags

```toml
[dependencies]
katgpt-rs = { version = "...", features = ["causal_head_importance"] }
# For the RTPurbo wiring (CalibrationMode::CausalNecessity + calibrate_from_causal_scores):
katgpt-rs = { version = "...", features = ["causal_head_importance", "rt_turbo"] }
```

- **`causal_head_importance`** (opt-in, katgpt-core): enables the scorer module + `ScaleNormalizedFusion`. The root crate forwards it via passthrough.
- **`rt_turbo`** (opt-in, root): additionally enables `calibrate_from_causal_scores` + the `CalibrationMode::CausalNecessity` enum variant wired into `RtTurboConfig`.

---

## The bystander pathology (why causal > attention-mass)

A **correlated bystander** head attends strongly to the needle (high attention-mass score!) but its output projects to zero in the readout direction (orthogonal — overridden downstream). Attention-mass ranks it high; causal IE = 0 (patching it doesn't move the readout).

G2 evidence (synthetic harness, 16 heads, 4 load-bearing vs 4 bystanders who attend 0.92 > load-bearing 0.78):

| bystanders | causal Jaccard | attention-mass Jaccard |
|---|---|---|
| 0 | 1.000 | 1.000 (agree — no bystanders) |
| 4 (25%) | 1.000 | 0.000 (bystanders displace all load-bearing) |
| 8 (50%) | 1.000 | 0.000 |

Causal is **invariant** to bystander fraction; attention-mass collapses once bystanders exist. The attention-mass baseline is the **real** `calibrate_from_scores` (RTPurbo, Plan 126), not a reimplementation.

---

## API Reference

### Scoring (hot path, allocation-free)

```rust
use katgpt_core::causal_head_importance::{direct_effect_importance, indirect_effect_importance};

// IE = (m_clean − m_patched) / (m_clean − m_corrupt) ∈ [0, 1]
// Caller supplies m_patched from their own patched forward pass.
let ie = direct_effect_importance(m_clean, m_corrupt, m_patched);
let ie_send = indirect_effect_importance(m_clean, m_corrupt, m_path_patched);
```

### Readout (Eq 9)

```rust
use katgpt_core::causal_head_importance::SpanLogitDiffReadout;
let readout = SpanLogitDiffReadout::default(); // λ = 0.9
let m = readout.readout(&[(z_correct_0, z_counterfactual_0), /* ... */]);
```

### Partition (offline, mirrors RTPurbo)

```rust
use katgpt_core::causal_head_importance::partition_by_causal_score;
let (critical, convertible) = partition_by_causal_score(&ie_scores, 0.25, None, false);
// min_one_per_layer (paper "Constrained Global Screening" §5.6) when layer_ids provided.
```

### RTPurbo wiring (root crate, needs `rt_turbo`)

```rust
use katgpt_rs::rt_turbo::calibrate_from_causal_scores;
use katgpt_rs::types::{CalibrationMode, RtTurboConfig};
let mut cfg = RtTurboConfig::default();
cfg.calibration_mode = CalibrationMode::CausalNecessity;
let calibration = calibrate_from_causal_scores(&ie_scores, &cfg);
```

### Scale-normalized branch fusion (Eq 13–14)

```rust
use katgpt_core::causal_head_importance::ScaleNormalizedFusion;
let fusion = ScaleNormalizedFusion::new(n_heads, 1e-5);
fusion.fuse_into(&[&fa_head, &gdn_head], head_dim, &mut out);
```

---

## Caller responsibility (what this primitive does NOT do)

The patched forward pass (selective head-output substitution + downstream-attention freezing) that produces `m_patched` is the **caller's** responsibility — it requires a full transformer forward and lives in riir-engine / riir-games. This module is the *scorer*; the patched forward pass is supplied by the caller via a closure. This keeps katgpt-core leaf-clean (no transformer dep) and matches the FaithfulnessProbe pattern (probe is generic, consumer supplies the behavior metric).

---

## What's out of scope (→ riir-train / riir-engine / riir-ai / riir-neuron-db)

- **→ riir-train:** head-wise FA/LA mixing architecture; three-stage transfer pipeline; branch-specific architecture refinements (FA NoPE+scale+gate, GDN RoPE+MHA, query decomposition); 15B-token scaling run.
- **→ riir-engine:** the patched-forward-pass implementation (selective head-output substitution + downstream-attention freezing) needed to actually compute `m_patched` on a real transformer.
- **→ riir-ai:** HLA direction-vector causal importance (Research 362 §2.5(a)) — applies the open `CausalHeadImportance` primitive to HLA's 8-dim affect space.
- **→ riir-neuron-db:** NeuronShard dendritic-branch causal importance (Research 362 §2.5(e)) — applies the primitive to `dendritic_lora` branch views for selective branch freeze/thaw.

---

## GOAT gate (G1/G2/G3/G4 all PASS)

See [`.benchmarks/358_causal_head_importance_goat.md`](../.benchmarks/358_causal_head_importance_goat.md) for full numbers. Summary: G1 IE discrimination + knockout faithfulness PASS; G2 causal strictly dominates attention-mass on bystander workload (Jaccard 1.0 vs 0.0); G3 partition ≤ 2× attention-mass (faster at n≥64); G4 hot-path scoring fns allocation-free. **Decision: opt-in, `AttentionMass` stays default.**

## TL;DR

Causal head-importance scoring is the strictly-stronger alternative to RTPurbo's
attention-mass calibration: it filters correlated bystanders that attention-mass
wrongly promotes (G2 Jaccard 1.0 vs 0.0). Ships opt-in behind
`causal_head_importance`; `CalibrationMode::AttentionMass` stays the default
because causal score production is ~10–100× more expensive (patched forwards) and
real-world bystander prevalence is unknown. Use `CausalNecessity` for the
long-context-extreme regime.
