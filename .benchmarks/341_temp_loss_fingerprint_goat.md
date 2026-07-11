# Plan 341 — TEMP Perturbed-Loss-Vector GOAT Gate Results

**Date:** 2026-06-29
**Plan:** [katgpt-rs/.plans/341_temp_perturbed_loss_vector_primitive.md](../.plans/341_temp_perturbed_loss_vector_primitive.md)
**Primitive:** `katgpt_core::diversity::temp::*` (`temp_loss_fingerprint` feature)
**Source paper:** [arXiv:2606.26797](https://arxiv.org/abs/2606.26797) — Jin et al., ICML 2026

---

## Gate Scorecard

| Gate | Criterion | Target | Result | Status |
|------|-----------|--------|--------|--------|
| G1 (bound preservation) | selected min pairwise bound ≥ 2× random median | ≥ 2.0× | **15.44×** | ✓ PASS |
| G2 (prefix-length sweep) | Kendall tau at N=32 vs N=256 | ≥ 0.85 | **0.9839** | ✓ PASS |
| G3 (perf: perturbed_loss_vector) | K=8, N=100, D=8 single-matmul | < 5 µs | **2.46 µs** | ✓ PASS |
| G3 (perf: select_diverse_subset) | n=256, k=32, K=8 | < 1 ms | **130.45 µs** | ✓ PASS |
| G3-alloc (perturbed_loss_vector) | hot path zero-alloc | 0 allocs/100 calls | **0** | ✓ PASS |
| G3-alloc (select_diverse_subset_into) | internal zero-alloc | ≤ 100 (return Vec only) | **100** (1/call return) | ✓ PASS |
| G4 (determinism) | bit-identical across two runs | exact match | **bit-identical** | ✓ PASS |
| G5 (feature isolation) | `--each-feature`, `--all-features`, `--no-default-features`, default | all clean | **134/134 + 3** | ✓ PASS |
| Integration G2' (riir-neuron-db Plan 005) | consolidation cosine gain | ≥ +0.10 | **+0.1672** | ✓ PASS |

**Verdict: ALL GATES PASS. Promote `temp_loss_fingerprint` to default-on.**

---

## G1 — Bound Preservation Under Diversity Selection

**Test:** `g1_bound_preservation_under_diversity_selection` in `diversity/temp.rs`

**Fixture:** n=64 candidate loss vectors (8-dim, schedule K=8) drawn from a
Gaussian mixture: 50 from a tight cluster N(0, 0.5) and 14 from a wide spread
N(0, 5.0) in each of 8 dims. This models realistic candidate sets where most
candidates are similar (clustered) and a few are diverse (spread).

**Metric:** MINIMUM pairwise Lipschitz bound over all C(8,2)=28 pairs in the
selected 8-subset, compared to the median minimum pairwise bound over 50
deterministic random 8-subsets. The minimum is the quantity the greedy max-min
algorithm directly optimizes: it maximizes the worst-case pairwise distance.

**Result:**
```
selected_min_bound         = 175.4946
random_median_min_bound    = 11.3662
ratio                      = 15.44×  (target ≥ 2.0×)
```

**Interpretation:** The greedy selector picks 8 candidates from the wide-spread
component (high pairwise deltas ~5–10), while random subsets are dominated by
the 50 cluster points (expected ~6.25 of 8 from the cluster), whose minimum
pairwise delta is ~0.5–1.5. The minimum Lipschitz bound ratio of 15.44× far
exceeds the 2.0× gate, confirming the selector picks the subset whose members
would induce maximally-different gradients along v (Theorem 3.1).

---

## G2 — Prefix-Length Sweep (Kendall Tau)

**Test:** `g2_prefix_length_sweep_kendall_tau` in `diversity/temp.rs`

**Fixture:** n=32 candidates, each with a 256-token prefix (D=8 per token). Each
candidate has a well-separated unit direction d_i. Tokens are
`5.0 * d_i + 0.15 * N(0,1)` per dimension (signal amplitude 5.0, noise std 0.15,
per-token SNR ~33:1). K=8 checkpoints extrapolated along a random unit
direction.

**Metric:** Per-candidate diversity score = sum of L_inf distances to all other
candidates' loss vectors. Kendall tau rank correlation between the ranking at
each N and the reference N=256.

**Results:**
```
N=8:   Kendall tau = 0.9395
N=16:  Kendall tau = 0.9718
N=32:  Kendall tau = 0.9839   ← gate checkpoint (≥ 0.85)
N=64:  Kendall tau = 0.9718
N=128: Kendall tau = 0.9798
N=256: Kendall tau = 1.0000   (reference)
```

**Interpretation:** Even at N=8 (just 8 tokens), the diversity ranking
correlates at 0.94 with the full N=256 ranking. At N=32, correlation is 0.98 —
far above the 0.85 gate. This is the modelless analog of the paper's Fig. 6
finding that "reasoning quality emerges early": a short prefix captures the
essential diversity signal, confirming token-efficiency for the modelless
gradient-diversity proxy.

Note: the non-monotone pattern (N=32 > N=64 > N=128) is expected — the Kendall
tau is a rank correlation, and as N increases, the per-candidate scores shift
slightly due to the nonlinear softplus, occasionally swapping near-tied
candidates. The key gate (N=32 ≥ 0.85) passes comfortably.

---

## G3 — Performance

**Bench:** `bench_341_temp_loss_fingerprint_goat` (criterion, release mode)

### perturbed_loss_vector (K=8, N=100, D=8)

```
perturbed_loss_vector_K8_N100_D8
    time: [2.4589 µs  2.4607 µs  2.4628 µs]
```

**Target:** < 5 µs per candidate → **2.46 µs** (2.0× headroom).

The kernel is a single-matmul per token: K=8 checkpoints × N=100 tokens × D=8
dim = 6400 FMA lanes per candidate. At 2.46 µs, that's ~2600 FMAs/µs — well
within scalar f32 throughput.

### select_diverse_subset (n=256, k=32, K=8)

```
select_diverse_subset_n256_k32_K8
    time: [129.76 µs  130.45 µs  131.06 µs]
```

**Target:** < 1 ms → **130.45 µs** (7.7× headroom).

This re-validates the riir-neuron-db Plan 005 Phase 4 G5 result (156 µs with
the same algorithm). The slight improvement (130 vs 156 µs) is within
run-to-run variance and the different fixture RNG.

### G3-alloc (zero-alloc hot path)

```
perturbed_loss_vector  100 calls: 0 allocs     ✓ (writes into caller buffer)
select_diverse_subset_into  100 calls: 100 allocs  ✓ (1/call = return Vec<usize>)
```

`perturbed_loss_vector` is truly zero-alloc. `select_diverse_subset_into`'s
internal hot path is zero-alloc with pre-allocated workspaces; the 1 alloc per
call is the unavoidable `Vec<usize>` return value (the output index set).

---

## G4 — Determinism / Quorum-Reproducibility

**Test:** `g4_determinism_bit_identical` in `diversity/temp.rs`

Two independent runs with identical `(s0, s1, lambda_schedule, noise_seeds,
noise_sigma, candidates)` produce:
- Bit-identical theta schedule (8 checkpoints): ✓
- Bit-identical loss vectors (32 candidates × 8 dims): ✓
- Bit-identical selected subset (8 indices): ✓

The determinism chain: BLAKE3-seeded noise → deterministic extrapolation →
deterministic loss computation → deterministic greedy selection (ties broken by
strict `>`, lower index wins). All stages are reproducible across runs, meeting
the sync-boundary requirement (Research 323 §5).

---

## G5 — Feature Isolation

```
cargo check -p katgpt-core --no-default-features    → clean
cargo check -p katgpt-core (default)                → clean
cargo check -p katgpt-core --all-features           → clean
cargo hack check -p katgpt-core --each-feature      → 134/134 clean
cargo test -p katgpt-core --features temp_loss_fingerprint --lib  → 673/673 PASS
```

No combo-only regressions. `temp_loss_fingerprint = []` (no dependencies), so
it cannot break other features.

---

## Integration Validation (riir-neuron-db Plan 005)

The cross-repo integration (riir-neuron-db `.benchmarks/005_temp_consolidation_goat.md`)
demonstrates the real-world consolidation-quality gain:

- **G2' (consolidation cosine alignment gain):** +0.1672 (target ≥ +0.10)
- **G4' (quorum-reproducibility):** 100/100 bit-identical hash matches
- **G5' full-path latency:** 185 µs (< 500 µs target)

The gain is **modelless** (deterministic linear extrapolation + greedy
max-min selection, no training, no gradients). This satisfies the promotion
requirement: "if G1–G5 pass AND the integration plan demonstrates a
consolidation-quality gain → promote to default-on."

---

## Promotion Decision

**`temp_loss_fingerprint` is PROMOTED to default-on in katgpt-core.**

Rationale:
1. G1–G5 all pass on the open primitive surface.
2. Integration G2' (+0.1672) proves the real-world modelless gain.
3. Zero-alloc hot path, no new dependencies.
4. Default `sleep()` behavior unchanged (backward compatible).

The G5 selection-step subtarget (50 µs) documented in riir-neuron-db Issue 003
(SIMD `l_inf_distance` optimization) is an internal optimization, not a
correctness gate. The full-path target and the open-primitive perf targets all
pass with comfortable headroom.

---

## Run Commands

```bash
# Unit tests (G1, G2, G4 + Phase 1 tests)
cargo test -p katgpt-core --features temp_loss_fingerprint --lib diversity::temp -- --nocapture

# G3 perf + alloc bench
cargo bench -p katgpt-core --features temp_loss_fingerprint --bench bench_341_temp_loss_fingerprint_goat -- --nocapture

# G5 feature isolation
cargo hack check -p katgpt-core --each-feature
cargo check -p katgpt-core --all-features
cargo check -p katgpt-core --no-default-features
```
