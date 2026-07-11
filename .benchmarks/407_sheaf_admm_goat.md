# Plan 407 — Sheaf-ADMM GOAT Gate (G1–G6) Results

**Date:** 2026-07-07
**Plan:** [katgpt-rs/.plans/407](../.plans/407_sheaf_admm_coordination_primitive.md)
**Research:** [katgpt-rs/.research/384](../.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md) §7
**Status:** ✅ **ALL 6 GATES PASS — promoted to DEFAULT-ON**

---

## Gate-by-gate results

| Gate | Criterion | Target | Measured | Verdict |
|------|-----------|--------|----------|---------|
| **G1** | DEC identity consensus reached after K=100 ADMM iterations on a 32×32 grid with identity maps. Cross-check: converged z ∈ ker(L_F). | `‖F x‖_∞ < 1e-5`; `‖Lz‖/‖z‖ < 1e-4` | `‖F x‖_∞ = 3.26e-8`; `‖Lz‖/‖z‖ = 1.16e-7`; hodge non-harm/harm = 0.0 | ✅ PASS |
| **G2** | Dual conservation: `u^{k+1} − u^k == x^{k+1} − z^{k+1}` bit-exactly after one step (with `u^0 = 0`, IEEE-754 `0 + δ == δ` is exact). | bit-exact | All 48 elements bit-identical (`to_bits()` match) | ✅ PASS |
| **G3** | Heterogeneous compression: `‖F x‖ ≤ ‖x‖` for selector maps (orthonormal rows). | `‖F x‖/‖x‖ ≤ 1.0` | max observed `‖F x‖/‖x‖ = 0.898` | ✅ PASS |
| **G4** | Latency: one `sheaf_admm_step` call, K=100 vertices (10×10 grid), d_v=8, d_e=5, T=5. | `< 5 µs` | `1.808 µs` (release, grid-stencil identity fast path) | ✅ PASS |
| **G5** | Zero-alloc: 0 allocations in steady state (100 calls after warmup). | `0 allocs` | `0 allocs` | ✅ PASS |
| **G6** | Determinism: same input → bit-identical output across 100 runs, debug AND release. | bit-exact | All 100 outputs bit-identical (debug + release) | ✅ PASS |

---

## G1 parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Grid | 32×32 (1024 vertices, 1984 edges) | Plan 407 spec |
| d_v | 4 | Plan 407 spec |
| d_e | 4 (identity, d_e = d_v) | Homogeneous consensus |
| K (ADMM iterations) | 100 | Plan 407 spec |
| T (diffusion steps) | 100 | Tuned: T=50 gave 3.25e-7; T=100 gives 3.26e-8 |
| rho | 1.0 | Plan 407 spec |
| eta | 0.2 | Stability: `eta·λ_max ≈ 0.2·8 = 1.6 < 2` |
| diag_q | 0.0 | f_i ≡ 0: ADMM reduces to pure sheaf diffusion |
| q | 0.0 | f_i ≡ 0 |
| Initial z | Random (splitmix64, seed `0xC0FF_EEBA_BE56_7812`) | Non-zero harmonic component drives consensus |

### G1 convergence analysis

With `diag_q = 0, q = 0` (local objective `f_i ≡ 0`) and random initial `z`, the
ADMM reduces to pure sheaf diffusion. The x-update is `x = z − u`, the z-update
warm-starts `z = x + u = z` (the z-trajectory decouples from u), and z is
diffused `T` steps per iteration. The harmonic component of z is preserved
(it's in `ker(L_F)`), while non-harmonic modes decay at `ρ_j = (1 − η·λ_j)` per
diffusion step.

The primal x at step K for eigenmode j:
```
x^K[j] = z⁰[j] · ρ_j^{(K−1)T} · (2·ρ_j^T − 1)
```

For the slowest mode (`λ₁ ≈ 0.019`, `ρ₁ = 0.9962`):
- `ρ₁^100 = 0.684`, so `2·0.684 − 1 = 0.368`
- `ρ₁^{99·100} ≈ e^{-37.6} ≈ 4.7e-17`
- `‖x^K[1]‖ ≈ ‖z⁰[1]‖ · 4.7e-17 · 0.368 ≈ 1.7e-17 · ‖z⁰[1]‖`

Actual measured `‖F x‖_∞ = 3.26e-8` — higher than the theoretical bound because
f32 rounding in 10000 cumulative diffusion steps introduces noise at the ~1e-8
level. Well below the 1e-5 target.

---

## G2 bit-exactness rationale

The ADMM u-update is `u^{k+1} = u^k + (x^{k+1} − z^{k+1})`. With `u^0 = 0`:
- `u^{k+1} = 0 + (x − z) = x − z` (IEEE-754: adding 0 is exact)
- `u_diff = u^{k+1} − u^k = (x − z) − 0 = x − z` (IEEE-754: subtracting 0 is exact)
- `xz_diff = x − z`

Both sides compute the same f32 subtraction `x − z`. Bit-identical by
construction.

**Note:** With non-zero `u^0`, the invariant `(a + b) − a == b` does NOT hold
in general f32 (catastrophic cancellation). The test uses `u^0 = 0` to
guarantee bit-exactness.

---

## G3 design note: orthonormal vs unit-norm rows

The original Plan 407 / Research 384 spec called for "random unit-norm rows."
This is **mathematically incorrect** for `d_e > 1`: a matrix with unit-norm
rows is NOT a contraction in general (its spectral norm can exceed 1).

Example: `F = [[1,1],[1,1]]/√2` has unit-norm rows, but `F^T F = [[1,1],[1,1]]`
has eigenvalue 2, so `‖F x‖² ≤ 2‖x‖²`.

**Correct claim:** restriction maps with **orthonormal** rows are contractions.
`F^T F` is then a projection (eigenvalues 0 or 1), so `‖F x‖² ≤ ‖x‖²`.

Selector maps produce orthonormal rows (standard basis vectors). We test
selector maps, which is the correct case and matches the modelless mandate
(identity / selector are the only constructors we ship).

---

## G4 identity fast path (Plan 407 T2.4)

The Phase 1 general explicit-maps matvec (`sheaf_laplacian_via_maps`) wastes
~`d_v` scalar multiplies per row (most against zero entries in the identity
block). For K=100, d_v=8, d_e=5, T=5, this gave **12.8 µs** — well above the
5 µs target.

The Phase 2 identity fast path (`sheaf_laplacian_identity_grid_into`) computes
the graph Laplacian on the first `d_e` dims directly, using a 5-point stencil
with stride `d_v`. Key optimizations:
1. **Grid stencil** (not edge-list): writes each output element exactly once
   (no scattered read-modify-write → no store-forwarding stalls).
2. **Skip `fill(0.0)`**: the grid stencil writes every element; dims `d_e..d_v`
   stay at 0 from `AdmmScratch::new` initialization (they're never modified).
3. **Only compute first `d_e` dims**: dims `d_e..d_v` have zero disagreement
   for identity maps, so they're skipped entirely.

Result: **1.808 µs** — a **7.1× speedup** over the general path, well within
the 5 µs target.

---

## Promotion decision

**All 6 gates pass → `sheaf_admm` promoted to DEFAULT-ON in `katgpt-dec`.**

Edit in `crates/katgpt-dec/Cargo.toml`:
```toml
default = ["heat_kernel_trajectory", "sheaf_admm"]
sheaf_admm = []  # DEFAULT-ON (Plan 407 Phase 2 GOAT G1–G6 ALL PASS)
```

The runtime fusion (riir-ai Research 314 / Plan 394) has its own gates.

---

## Validation commands and results

```bash
export CARGO_TARGET_DIR=/tmp/sheaf_admm_phase2

# 1. Correctness gates (G1, G2, G3, G6)
cargo test -p katgpt-dec --features sheaf_admm --no-default-features \
  --test sheaf_admm_goat -- --nocapture
# → 4 passed; 0 failed

# 2. Perf gates (G4, G5)
cargo bench -p katgpt-dec --features sheaf_admm --no-default-features \
  --bench bench_407_sheaf_admm_goat -- --nocapture
# → G4: 1.808 µs < 5.0 µs PASS; G5: 0 allocs PASS

# 3. All-features combo check (merkle_root lesson)
cargo check -p katgpt-dec --all-features
# → Finished

# 4. Default check (no-regression)
cargo check -p katgpt-dec
# → Finished

# 5. Full test suite with sheaf_admm on
cargo test -p katgpt-dec --all-features
# → all passed

# 6. Default test suite (sheaf_admm is now default-on)
cargo test -p katgpt-dec
# → all passed

# 7. G6 release-mode determinism
cargo test -p katgpt-dec --features sheaf_admm --no-default-features \
  --test sheaf_admm_goat --release g6_determinism_bit_exact_across_runs -- --nocapture
# → 1 passed; 0 failed
```

---

## Files changed

| File | Change |
|------|--------|
| `crates/katgpt-dec/src/sheaf_admm.rs` | Added identity fast path: `sheaf_laplacian_identity_grid_into` + grid-stencil dispatch in `sheaf_laplacian_via_maps` |
| `crates/katgpt-dec/Cargo.toml` | Added `[[bench]]` entry; promoted `sheaf_admm` to `default` |
| `crates/katgpt-dec/tests/sheaf_admm_goat.rs` | NEW: G1, G2, G3, G6 correctness gates |
| `crates/katgpt-dec/benches/bench_407_sheaf_admm_goat.rs` | NEW: G4, G5 perf gates |
| `.plans/407_sheaf_admm_coordination_primitive.md` | Marked T2.1–T2.7 done; filled GOAT table |

---

## Cross-references

- [Research 384](../.research/384_Sheaf_ADMM_Multi_Agent_Coordination.md) §7 — GOAT gate criteria
- [Plan 407](../.plans/407_sheaf_admm_coordination_primitive.md) — Phase 2 task breakdown
- Plan 251 — DEC operators (`graph_laplacian` identity reduction substrate)
- Plan 357 — Motor-gated field GOAT (bench file template)
