# Plan 358 GOAT Gate — CausalHeadImportance + ScaleNormalizedFusion

**Date:** 2026-07-02
**Plan:** [katgpt-rs/.plans/358_causal_head_importance_calibration.md](../.plans/358_causal_head_importance_calibration.md)
**Research:** [katgpt-rs/.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md](../.research/362_HydraHead_Causal_Head_Importance_Hybrid_Attention.md)
**Source paper:** [arXiv:2606.20097](https://arxiv.org/abs/2606.20097) — Tan et al., HydraHead, Alibaba, Jun 2026
**Feature:** `causal_head_importance` (opt-in, katgpt-core) + root passthrough

---

## Verdict: GATE PASS — opt-in, `AttentionMass` stays default

`CausalNecessity` is **strictly stronger** than `AttentionMass` on the bystander
discrimination workload (G2: Jaccard 1.0 vs 0.0), at acceptable partition latency
(G3: causal ≤ 2× attention-mass, actually faster at n≥64), with allocation-free
hot paths (G4). **However, it stays opt-in** — `CalibrationMode::AttentionMass`
remains the default because (1) causal calibration is ~10–100× more expensive
to *produce* (n_heads × n_samples patched forward passes vs 1 forward pass),
and (2) the bystander prevalence in production game-AI models is unknown
(synthetic-only G1/G2 validation per Risk #3/#4). When there are no bystanders,
both methods agree (G2 at 0 bystanders: both Jaccard 1.0).

**Use `CausalNecessity` for the long-context-extreme regime where correlated
bystander heads matter. Use `AttentionMass` for the common case (cheaper,
agrees with causal when no bystanders are present).**

---

## Gate table

| Gate | Target | Result | Verdict |
|------|--------|--------|---------|
| **G1** (correctness — IE discrimination) | load-bearing IE > threshold; bystander IE < threshold; ranking Spearman ρ = 1.0; knockout faithfulness | load-bearing IE=0.25 > 0.01; bystander IE=0; partition Jaccard=1.0; IE-ordered knockout ratio 0.0 < 0.2; random mean ratio 0.875 > 0.8 (2000 trials) | **PASS ✅** |
| **G2** (bystander discrimination) | causal top-K = load-bearing set (Jaccard 1.0); attention-mass top-K includes bystanders (Jaccard < 1.0) | causal Jaccard 1.000 at all bystander fractions {0,4,8}; attention-mass Jaccard 1.000→0.000→0.000 | **PASS ✅** |
| **G3** (calibration latency) | causal partition ≤ 2× attention-mass | n=16: **1.11×**; n=64: **0.51×** (causal faster); n=144: **0.77×** (causal faster) | **PASS ✅** |
| **G4** (zero-alloc hot path) | scoring fns allocation-free | `direct/indirect_effect_importance`/`per_capability_score` pure f32 arithmetic; `SpanLogitDiffReadout::readout` `&[(f32,f32)]`→f32; `ScaleNormalizedFusion::fuse_into` caller scratch; `partition_by_causal_score` (offline) n=144: 1527 ns | **PASS ✅** |

---

## G2 bystander discrimination detail

Synthetic harness: 16 heads, K=4 load-bearing (attend 0.78, project into
readout), M correlated bystanders (attend **0.92** — MORE than load-bearing,
modeling the real pathology where bystanders win on pure attention-mass — but
project to zero, so causal IE = 0), rest local (attend 0.08). Both partitions
use the same ratio (4/16=0.25 → K=4) via `fair_config()`.

| bystanders | causal Jaccard | attention-mass Jaccard | verdict |
|---|---|---|---|
| 0 | 1.000 | 1.000 | tie/agree (no bystanders → both correct) |
| 4 (25%) | 1.000 | 0.000 | causal wins (bystanders displace all load-bearing) |
| 8 (50%) | 1.000 | 0.000 | causal wins |

**Key insight:** causal partition is **invariant** to bystander fraction (always
recovers the exact load-bearing set). Attention-mass collapses to 0.0 the moment
bystanders exist, because bystanders attend more strongly (0.92 > 0.78) and
displace the actual load-bearing heads entirely in the top-K. This is the
"correlated bystander" pathology the causal score is designed to fix.

The attention-mass baseline is the **real** `calibrate_from_scores` (RTPurbo,
Plan 126), not a reimplementation — apples-to-apples.

---

## G3 calibration latency detail

The benchmark compares only the **partition step** (after scores are computed),
not the patched forward passes that produce IE scores (those are ~10–100× more
expensive for causal and are a separate, amortized offline cost).

| n_heads | causal ns/call | attention-mass ns/call | x_ratio | gate (≤ 2×) |
|---|---|---|---|---|
| 16 | 327.6 | 293.9 | 1.11× | PASS |
| 64 | 591.3 | 1169.1 | 0.51× | PASS (causal faster) |
| 144 | 1616.0 | 2099.9 | 0.77× | PASS (causal faster) |

Causal partition is leaner (sort indices + `split_off`) than attention-mass
(builds full `HeadClassification` structs + more allocation per head). The causal
*partition* is not the bottleneck — the patched forward passes that *produce*
the IE scores are.

---

## Promote/demote decision (Plan 358 T4.4)

**Decision: `AttentionMass` stays default; `CausalNecessity` opt-in.**

| Factor | Evidence | Direction |
|---|---|---|
| Quality on bystander workload | G2: causal strictly dominates (Jaccard 1.0 vs 0.0) | → promote causal |
| Latency of partition step | G3: causal ≤ 2× (often faster) | → promote causal |
| Cost to *produce* scores | causal needs n_heads × n_samples patched forwards; attention-mass needs 1 forward | → keep attention-mass default |
| Real-model bystander prevalence | Unknown — synthetic-only validation (Risk #3/#4) | → keep attention-mass default |
| Agreement when no bystanders | G2 at 0 bystanders: both Jaccard 1.0 | → attention-mass sufficient for common case |

The cost asymmetry (patched forwards vs 1 forward) and unknown real-world
bystander prevalence tip the decision toward keeping the cheaper
`AttentionMass` as the default. `CausalNecessity` is the strictly-stronger tool
for the regime where bystanders matter (long-context-extreme). This matches the
plan §Goal second bullet: "keep `AttentionMass` default, leave `CausalNecessity`
opt-in for the long-context-extreme regime where bystander heads matter."

No feature is demoted — both ship opt-in at the primitive level
(`causal_head_importance` feature) and the root forwards `causal_head_importance`
passthrough. `CalibrationMode::AttentionMass` is the `#[default]` variant.

---

## Files changed

| File | Change |
|---|---|
| `crates/katgpt-core/src/causal_head_importance/{mod,readout,patching,scorer,fusion}.rs` | New module — 5 files, 27 unit tests |
| `crates/katgpt-core/Cargo.toml` | `causal_head_importance = []` feature + `causal_head_importance_g1` test registration |
| `crates/katgpt-core/src/lib.rs` | Module registration + re-exports (incl. `CalibrationMode`) |
| `crates/katgpt-types/src/enums.rs` | `CalibrationMode` enum + `RtTurboConfig.calibration_mode` field |
| `crates/katgpt-types/src/lib.rs` | `CalibrationMode` re-export |
| `crates/katgpt-core/tests/causal_head_importance_g1.rs` | G1 correctness gate (5 tests) |
| `tests/causal_head_importance_g2.rs` | G2 bystander discrimination gate (3 tests, root tests/) |
| `benches/causal_head_importance_g3.rs` | G3 latency bench (root benches/) |
| `src/rt_turbo/calibration.rs` | `calibrate_from_causal_scores` sibling fn |
| `src/rt_turbo/mod.rs` | Re-export `calibrate_from_causal_scores` |
| `src/rt_turbo/forward.rs` | Fix struct literal (add `..Default::default()`) |
| `tests/test_126_rt_turbo_goat.rs` | Fix struct literal |
| `examples/rt_turbo_01_calibration.rs` | Step 6 causal-mode demo |
| `examples/rt_turbo_02_decode_bench.rs` | Fix struct literal |
| `Cargo.toml` (root) | `causal_head_importance` passthrough + bench/test registration |

---

## Validation

- `cargo test -p katgpt-core --features causal_head_importance --lib` → **693 pass** (666 + 27 new)
- `cargo test -p katgpt-core --test causal_head_importance_g1` → **5 pass** (G1)
- `cargo test --test causal_head_importance_g2` → **3 pass** (G2)
- `cargo test --features rt_turbo rt_turbo::` → **84 pass** (no regression from CalibrationMode field)
- `cargo test --features rt_turbo --test test_126_rt_turbo_goat` → **6 pass** (no regression)
- `cargo bench --bench causal_head_importance_g3` → G3 PASS at all head counts
- Default + `--features causal_head_importance` builds clean
- `CARGO_TARGET_DIR` isolated per AGENTS.md

---

## TL;DR

CausalHeadImportance ships behind opt-in `causal_head_importance`. G1/G2/G3/G4
all PASS. Causal scoring strictly dominates attention-mass on bystander-heavy
workloads (G2 Jaccard 1.0 vs 0.0) and is faster at the partition step (G3).
**Verdict: opt-in, `AttentionMass` stays default** — causal calibration is
~10–100× more expensive to produce (patched forwards) and real-world bystander
prevalence is unknown. Use `CausalNecessity` for the long-context-extreme regime.
